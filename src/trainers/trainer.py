"""Shadow Posterior Matching training."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel

from src.config import TrainingConfig
from src.data.grouped_prompt_dataset import create_grouped_dataloader
from src.data.grouped_rollout import weighted_reward_advantages
from src.parallel import release_fsdp_caches, synchronized_reward_scoring
from src.rewards.multi_scorer import MultiScorer
from src.rewards.source_rewards import SourceRewardWindow
from src.rewards.text_reuse import reuse_reward_text
from src.trainers.models import ModelTrainingBase
from src.trainers.objective import average_gradients, clean_prediction_loss, flow_matching_loss, matching_force
from src.utils.scalars import scalar_metrics


def _unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _time_embedding(model):
    return _unwrap(model).time_text_embed


def _add_source_bias(module, args, output):
    indices = module._source_indices
    if indices is None:
        return output
    if indices.shape != (len(output),):
        raise ValueError('Source conditioning requires one stop index per image')
    return output + module.source_bias[indices].to(output)


def install_source_conditioning(model, *, dtype=torch.float32):
    """Add a trainable offset per stop and use zero offset for endpoint queries."""
    embedding = _time_embedding(model)
    if hasattr(embedding, 'source_bias'):
        raise ValueError('SPM source conditioning is already installed')
    projection = embedding.timestep_embedder.linear_2
    embedding.register_parameter('source_bias', torch.nn.Parameter(
        torch.zeros(4, projection.out_features, device=projection.weight.device,
                    dtype=projection.weight.dtype if dtype is None else dtype)))
    embedding._source_indices = None
    embedding.register_forward_hook(_add_source_bias)
    return embedding.source_bias


@contextmanager
def source_condition(model, indices):
    embedding = _time_embedding(model)
    indices = torch.as_tensor(indices, device=embedding.source_bias.device)
    if indices.dtype != torch.long or indices.ndim != 1 or (indices < 0).any() or (indices >= 4).any():
        raise ValueError('SPM source indices must be a vector of integers in [0, 4)')
    previous = embedding._source_indices
    embedding._source_indices = indices
    try:
        yield
    finally:
        embedding._source_indices = previous


class Trainer(ModelTrainingBase):
    def __init__(self, config: TrainingConfig):
        config.sampling.__post_init__()
        config.objective.__post_init__()
        if config.distributed.strategy not in ('ddp', 'fsdp'):
            raise ValueError('distributed.strategy must be ddp or fsdp')
        accumulation = config.train.gradient_accumulation_steps
        if type(accumulation) is not int or accumulation < 1:
            raise ValueError('gradient_accumulation_steps must be a positive integer')
        if not (config.model.generator_lora.load_path and config.model.fake_score_lora.load_path):
            raise ValueError('Set model.generator_lora.load_path and model.fake_score_lora.load_path')
        super().__init__(config)
        self._reward_scorer: MultiScorer | None = None
        self._source_rewards = SourceRewardWindow.from_config(config.objective)
        self._batch_prompts = []
        self._batch_metadata = []
        self._pending_batches = []
        self._accum_index = 0

    def _pre_wrap_models(self):
        super()._pre_wrap_models()
        # FSDP needs a common dtype inside its flat parameter.
        dtype = None if self.config.distributed.strategy == 'fsdp' else torch.float32
        install_source_conditioning(self.fake_score_net, dtype=dtype)

    def _create_dataloader(self):
        return create_grouped_dataloader(self.config.prompt_path, self.config.train.batch_size,
            self.config.objective.num_image_per_prompt, self.config.train.seed,
            stratify_sources=self.config.objective.source_stratified_sampling)

    def _set_batch_context(self, batch):
        if isinstance(batch, dict):
            prompts, metadata = batch.get('prompts'), batch.get('metadata')
        else:
            prompts = batch
            metadata = [{'prompt': p} for p in prompts] if isinstance(prompts, list) else None
        if (not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts)
                or not isinstance(metadata, list) or len(metadata) != len(prompts)
                or not all(isinstance(m, dict) for m in metadata)):
            raise ValueError('batch requires nonempty prompts and aligned metadata dictionaries')
        self._batch_prompts, self._batch_metadata = prompts, metadata
        return prompts

    @torch.no_grad()
    def _score_endpoints(self, endpoint, names):
        """Return one row of raw scores per requested reward."""
        if len(self._batch_prompts) != len(endpoint) or len(self._batch_metadata) != len(endpoint):
            raise ValueError('reward batch must align with prompts and metadata')
        size = self.config.objective.reward_batch_size
        if self._reward_scorer is None:
            weights = {name: self.config.objective.reward_fn[name] for name in names}
            self._reward_scorer = MultiScorer(endpoint.device, weights).freeze()
        scorer = self._reward_scorer.to(endpoint.device)
        chunks = []
        with reuse_reward_text():
            for start in range(0, len(endpoint), size):
                stop = min(start + size, len(endpoint))
                images = self._decode_latents_to_tensor(endpoint[start:stop].detach())
                prompts = self._batch_prompts[start:stop]
                metadata = self._batch_metadata[start:stop]
                details = scorer.score_details(images, prompts, metadata, names=names)
                scores = [
                    torch.as_tensor(details[name], device=endpoint.device, dtype=torch.float32).detach()
                    for name in names
                ]
                if any(value.shape != (stop - start,) for value in scores):
                    raise ValueError('reward must return one raw score per endpoint')
                chunks.append(torch.stack(scores))
        result = torch.cat(chunks, dim=1)
        if not torch.isfinite(result).all():
            raise ValueError('reward must return finite scores')
        return result

    def _rollout_layout(self):
        return {'batches': self.config.train.gradient_accumulation_steps,
                'batch_size': self.config.train.batch_size,
                'group_size': self.config.objective.num_image_per_prompt}

    def train_step(self, batch):
        """Consume a rollout window and update the generator and fake score."""
        count = self.config.train.gradient_accumulation_steps
        if self._student_microstep == 0:
            self._pending_batches.clear()
        prompts = self._set_batch_context(batch)
        self._pending_batches.append((list(prompts), list(self._batch_metadata)))
        if self._student_microstep + 1 < count:
            return {}

        records, rollouts = [], []
        try:
            for index, (prompts, _) in enumerate(self._pending_batches):
                self._accum_index = index
                with self.timer.measure('text_encoding'):
                    prompt, pooled = self._encode_prompts(prompts)
                with self.timer.measure('fake_score_update'):
                    endpoint, steps = self._generate_latents(
                        prompt, pooled, model=_unwrap(self.generator),
                    )
                    fake = self._update_fake_score(prompt, pooled, endpoint, steps)
                rollouts.append((prompt.detach(), None if pooled is None else pooled.detach(), endpoint, steps))
                records.append({'fake_score': fake})

            self._batch_prompts = [p for prompts, _ in self._pending_batches for p in prompts]
            self._batch_metadata = [m for _, metadata in self._pending_batches for m in metadata]
            with self.timer.measure('reward_scoring'):
                credit, credit_metrics = self._group_credit(torch.cat([item[2] for item in rollouts]))

            offset = 0
            for index, (prompt, pooled, endpoint, steps) in enumerate(rollouts):
                self._accum_index = index
                uncond, uncond_pooled = self._get_uncond_embeds(len(endpoint))
                points = self._label_points(endpoint)
                rewards = credit[:, offset:offset + len(endpoint)]
                offset += len(endpoint)
                with self.timer.measure('generator_update'):
                    records[index]['generator'] = self._update_generator(
                        prompt, pooled, uncond, uncond_pooled, endpoint, steps, points, rewards,
                    )
            cfg = self.config.train
            step = self.global_step + 1
            if self.generator_ema is not None and step % cfg.ema_every == 0:
                self.ema_update(self.generator, self.generator_ema, cfg.ema_decay, trainable_only=True)

            collected = {}
            for record in records:
                for section, metrics in record.items():
                    for key, value in metrics.items():
                        collected.setdefault(section, {}).setdefault(key, []).append(value)
            result = {section: {key: values[-1] if key.endswith('grad_norm') or key == 'lr'
                                else sum(values) / len(values) for key, values in metrics.items()}
                      for section, metrics in collected.items()}
            result['generator'].update(credit_metrics)
            return result
        finally:
            self._pending_batches.clear()
            self._accum_index = 0

    def _zero_accumulated_grad(self, role):
        if self._accum_index == 0:
            self.optimizers[role].zero_grad(set_to_none=True)

    def _deployed_stops(self, clean):
        """Return the stop indices, sigmas, and model timesteps used during rollout."""
        self.x0_scheduler.set_timesteps(sigmas=self._get_denoising_sigmas())
        sigmas, timesteps = self.x0_scheduler.sigmas, self.x0_scheduler.timesteps
        return [(index, float(sigmas[index]),
                 timestep.expand(len(clean)).to(device=clean.device, dtype=torch.float32))
                for index, timestep in enumerate(timesteps)]

    def _step(self, role, model):
        raw = _unwrap(model)
        count = self.config.train.gradient_accumulation_steps
        if self._accum_index + 1 < count:
            return 0.

        if not isinstance(raw, FSDP):
            average_gradients(raw)
        solver = getattr(self.config.solver, role)
        norm = self._clip_grad_norm(raw, solver.max_grad_norm if solver.max_grad_norm > 0 else float('inf'))
        self.optimizers[role].step()
        self.schedulers[role].step()
        return norm

    def _source_bias(self):
        return _time_embedding(self.fake_score_net).source_bias

    def _full_source_bias(self, *, writeback=False):
        model = self.fake_score_net
        return (FSDP.summon_full_params(model, recurse=False, writeback=writeback)
                if isinstance(model, FSDP) else nullcontext())

    def _predict_source(self, y, prompt, time, pooled, source):
        model = _unwrap(self.fake_score_net)
        with source_condition(model, source):
            return self._predict_noise(model, y, prompt, time, pooled, guidance_scale=1.)

    def _gamma(self, source_index):
        return self.config.objective.gamma if source_index < 3 else 0.

    def _contract(self):
        return {
            'objective': asdict(self.config.objective),
            'reward_order': [name for name, weight in self.config.objective.reward_fn.items() if weight],
            'initial_models': asdict(self.config.model),
            'data': {'prompt_path': self.config.prompt_path, 'seed': self.config.train.seed},
            'sampling': asdict(self.config.sampling),
        }

    def _get_extra_checkpoint_state(self):
        # LoRA weight files contain adapter parameters only.
        with self._full_source_bias():
            bias = self._source_bias().detach().cpu().clone()
        return {'objective_contract': self._contract(),
                'rollout_layout': self._rollout_layout(),
                'source_bias': bias}

    def _restore_extra_checkpoint_state(self, extra):
        if (extra.get('objective_contract') != self._contract()
                or extra.get('rollout_layout') != self._rollout_layout()):
            raise ValueError('SPM resume objective or rollout layout differs')
        bias = extra.get('source_bias')
        with torch.no_grad(), self._full_source_bias(writeback=True):
            if not torch.is_tensor(bias) or bias.shape != self._source_bias().shape or not torch.isfinite(bias).all():
                raise ValueError('SPM resume requires all four finite source bias vectors')
            self._source_bias().copy_(bias)

    @torch.no_grad()
    def _group_credit(self, endpoint):
        c = self.config.objective
        num_stops = len(self.config.sampling.denoising_step_list)
        if self._source_rewards is not None:
            scores, codes = synchronized_reward_scoring(
                lambda: self._source_rewards.score(
                    self._decode_latents_to_tensor, endpoint, self._batch_prompts, self._batch_metadata),
                context='SPM reward scoring')
            credit, metrics = self._source_rewards.credit(scores, codes, self._batch_prompts)
            return credit.expand(num_stops, -1), metrics
        names = tuple(name for name, weight in c.reward_fn.items() if weight)
        weights = torch.tensor([float(c.reward_fn[name]) for name in names], device=endpoint.device)
        n, groups = len(endpoint), self._batch_prompts

        scores = synchronized_reward_scoring(
            lambda: self._score_endpoints(endpoint, names), context='SPM reward scoring')
        rank = 0
        if dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
            gathered_scores = [torch.empty_like(scores) for _ in range(world)]
            dist.all_gather(gathered_scores, scores.contiguous())
            gathered_groups = [None]*world
            dist.all_gather_object(gathered_groups, groups)
            scores = torch.cat(gathered_scores, dim=1)
            groups = [key for shard in gathered_groups for key in shard]
        rewards = (weights[:, None]*scores).sum(0)
        values = scores.double().cpu().numpy()
        a = weighted_reward_advantages(groups, dict(zip(names, values)),
            {name: c.reward_fn[name] for name in names},
            global_std=c.per_prompt_global_std)
        credit = torch.as_tensor(a, device=scores.device, dtype=torch.float32)
        credit = credit.clamp(-c.adv_clip_max, c.adv_clip_max)
        metrics = scalar_metrics({'reward_mean': rewards.mean(),
            **{f'reward/{name}': scores[i].mean() for i, name in enumerate(names)}})
        credit = credit[rank*n:(rank+1)*n].expand(num_stops, -1)
        return credit, metrics

    def _label_points(self, clean):
        """Generate fresh noise and training inputs at each deployed stop."""
        points = []
        for index, sigma, timestep in self._deployed_stops(clean):
            y, u = self._noise_clean_latents(clean.detach().float(), sigma)
            points.append({'index': torch.full((len(clean),), index, device=clean.device, dtype=torch.long),
                           'source_index': index, 'sigma': sigma, 't': timestep, 'y': y, 'u': u})
        return points

    @staticmethod
    def _selected_stop_schedule(clean, stops, selected):
        sigmas = clean.new_tensor([s for _, s, _ in stops], dtype=torch.float32)[selected]
        times = torch.stack([t[0] for _, _, t in stops])[selected]
        return sigmas.reshape((-1,) + (1,) * (clean.ndim - 1)), times

    @torch.no_grad()
    def _shadow_predictions(self, endpoint, prompt, pooled, stops):
        probabilities = endpoint.new_full((len(stops),), 1 / len(stops), dtype=torch.float32)
        source = torch.multinomial(probabilities, len(endpoint), replacement=True)
        sigma, times = self._selected_stop_schedule(endpoint, stops, source)
        y, _ = self._noise_clean_latents(endpoint.detach().float(), sigma)
        y = y.to(endpoint.dtype)
        with self._autocast():
            prediction = self._predict_noise(_unwrap(self.generator), y, prompt, times, pooled, guidance_scale=1.)
        return y.float() - sigma * prediction.float(), source

    def _update_fake_score(self, prompt_embeds, pooled_embeds, endpoint, steps):
        fake, accum = _unwrap(self.fake_score_net), self.config.train.gradient_accumulation_steps
        clean = endpoint.detach().float()
        stops = self._deployed_stops(clean)
        with torch.no_grad():
            query_t, query_sigma = self._query_score_schedule(clean)
            selected = torch.randint(len(stops), (len(clean),), device=clean.device)
            grid_sigma, grid_t = self._selected_stop_schedule(clean, stops, selected)

        self._zero_accumulated_grad('fake_score')
        metrics, total = {}, 0.
        for name, sigma, t in (('grid', grid_sigma, grid_t), ('nu', query_sigma, query_t)):
            with torch.no_grad():
                y, u = self._noise_clean_latents(clean, sigma)
            with self._autocast():
                prediction = self._predict_noise(fake, y.to(endpoint.dtype), prompt_embeds, t, pooled_embeds,
                                                 guidance_scale=1.)
            loss = flow_matching_loss(prediction, u)
            (0.5 * loss / accum).backward()
            release_fsdp_caches(fake, no_shard_only=True)
            metrics[f'loss_fake_{name}'] = loss.detach()
            total += 0.5 * loss.detach().double()

        output, source = self._shadow_predictions(endpoint, prompt_embeds, pooled_embeds, stops)
        with torch.no_grad():
            t, sigma = self._query_score_schedule(output)
            y, u = self._noise_clean_latents(output, sigma)
        with self._autocast():
            prediction = self._predict_source(y.to(endpoint.dtype), prompt_embeds, t, pooled_embeds, source)
        loss = flow_matching_loss(prediction, u)
        (loss / accum).backward()
        release_fsdp_caches(fake, no_shard_only=True)
        metrics['loss_fake_query'] = loss.detach()
        total += loss.detach().double()
        norm = self._step('fake_score', fake)
        return scalar_metrics({'loss_fake': total, 'grad_norm': norm, 'gen_steps': steps,
                'lr': self.schedulers['fake_score'].get_last_lr()[0], **metrics})

    def _teacher_velocity(self, latents, prompt, timestep, pooled, uncond, uncond_pooled):
        return self._predict_noise(
            self.teacher, latents, prompt, timestep, pooled,
            guidance_scale=self.config.sampling.teacher_guidance_scale,
            uncond_text_embeddings=uncond, uncond_pooled_prompt_embeds=uncond_pooled,
        ).float()

    def _update_generator(self, prompt, pooled, uncond_embeds, uncond_pooled, endpoint, steps, points, credit):
        c = self.config.objective
        student, fake = _unwrap(self.generator), _unwrap(self.fake_score_net)
        share = 1 / len(points)
        total_loss = 0.
        self._zero_accumulated_grad('generator')
        for point, reward in zip(points, credit):
            y, u, s = point['y'].to(endpoint.dtype), point['u'].float(), point['sigma']
            source = point['index']
            with torch.no_grad(), self._autocast():
                f0_input = self._predict_noise(fake, y, prompt, point['t'], pooled, guidance_scale=1.).float()
            with self._autocast():
                velocity = self._predict_noise(student, y, prompt, point['t'], pooled, guidance_scale=1.)
            clean = y.float() - s * velocity.float()
            with torch.no_grad():
                t, tau = self._query_score_schedule(clean.detach())
                query, query_target = self._noise_clean_latents(clean.detach(), tau)
                query = query.to(endpoint.dtype)
                gamma = self._gamma(point['source_index'])
                with self._autocast():
                    f1_output = self._predict_source(query, prompt, t, pooled, source).float()
                    f0_output = self._predict_noise(fake, query, prompt, t, pooled, guidance_scale=1.).float() \
                        if gamma else None
                    teacher = self._teacher_velocity(query, prompt, t, pooled, uncond_embeds, uncond_pooled)
                force, *_ = matching_force(f0_input, f1_output, teacher, u, reward,
                    alpha=c.alpha, student_sigma=s, query_sigma=tau, query_target=query_target,
                    gamma=gamma, f0_output=f0_output)
            loss = clean_prediction_loss(clean, force, s)
            (share * loss / self.config.train.gradient_accumulation_steps).backward()
            release_fsdp_caches(student, no_shard_only=True)
            total_loss += share * loss.detach().double()
        norm = self._step('generator', self.generator)
        return scalar_metrics({'loss_velocity': total_loss, 'model_grad_norm': norm,
            'gen_steps': steps, 'lr': self.schedulers['generator'].get_last_lr()[0]})
