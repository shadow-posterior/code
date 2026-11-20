"""Route task rewards by prompt source and normalize only applicable scores."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from src.rewards.multi_scorer import MultiScorer


class SourceRewardScorer:
    """Score each sample with the rewards applicable to its source.

    Inactive rewards are NaN, distinguishing them from a real zero task score.
    ``_source_index`` is an internal, fixed-order code for cross-rank gathering.
    """

    def __init__(self, scorer: MultiScorer, source_weights: dict):
        self.scorer = scorer
        self.source_weights = {}
        for source, weights in sorted(source_weights.items()):
            if not weights or any(not math.isfinite(float(w)) or float(w) < 0 for w in weights.values()):
                raise ValueError(f'Invalid reward weights for source {source!r}')
            active = {k: float(w) for k, w in weights.items() if w > 0}
            if not active or not set(active).issubset(scorer.score_dict):
                raise ValueError(f'Unknown or empty rewards for source {source!r}: {active}')
            self.source_weights[source] = active
        if not self.source_weights:
            raise ValueError('source_weights must be nonempty')
        self._codes = {source: code for code, source in enumerate(self.source_weights)}
        self._names = sorted({k for weights in self.source_weights.values() for k in weights})

        self._weights = np.array([[weights.get(k, 0.) for k in self._names]
                                  for weights in self.source_weights.values()], dtype=np.float64)

    def to(self, device):
        self.scorer.to(device)
        return self

    def __call__(self, images, prompts, metadata=None, only_strict=True):
        if not isinstance(metadata, list) or len(metadata) != len(prompts):
            raise ValueError('Source rewards require aligned dataset_source metadata')
        sources = [m.get('dataset_source') for m in metadata]
        if any(s not in self.source_weights for s in sources):
            raise ValueError('Missing or unknown dataset_source in prompt metadata')
        n = len(prompts)
        codes = np.array([self._codes[s] for s in sources], dtype=np.int64)
        result = {k: np.full(n, np.nan) for k in self._names}

        calls = {}
        for column, name in enumerate(self._names):
            rows = np.flatnonzero(self._weights[codes, column] > 0)
            if rows.size:
                calls.setdefault(rows.tobytes(), (rows, []))[1].append(name)
        for rows, names in calls.values():
            if rows.size == n:
                batch = images, prompts, metadata
            else:
                selected = (images[torch.as_tensor(rows, device=images.device)] if isinstance(images, torch.Tensor)
                            else images[rows] if isinstance(images, np.ndarray) else [images[i] for i in rows])
                batch = selected, [prompts[i] for i in rows], [metadata[i] for i in rows]
            scores = self.scorer.score_details(*batch, names=names, only_strict=only_strict)
            for name in names:
                values = _host_scores(scores[name])
                if values.shape != (rows.size,) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid or nonfinite applicable reward {name!r}')
                result[name][rows] = values

        result['_source_index'] = codes
        return result


def _host_scores(values):
    """Reward values as a CPU float64 array (tensors, scalar-tensor lists or floats)."""
    if isinstance(values, torch.Tensor):
        return values.detach().to(device='cpu', dtype=torch.float64).numpy()
    if isinstance(values, (list, tuple)) and any(isinstance(v, torch.Tensor) for v in values):
        device = next(v.device for v in values if isinstance(v, torch.Tensor))
        return torch.stack([torch.as_tensor(v, device=device, dtype=torch.float64).detach()
                            for v in values]).cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def source_advantages(rewards, prompts, source_weights, num_timesteps,
                      *, global_std=False, per_prompt=True):
    """Normalize and combine the rewards for valid source groups.

    Each reward uses its prompt group's mean and sample std (Bessel correction,
    matching TRL nanstd). Missing task rewards contribute nothing to the sum.
    The sum is normalized once over the entire gathered batch, as in TRL.
    Source codes only identify applicability and distinct prompt groups; they
    never introduce a source-specific final normalization or a reward-count mean.
    """
    if global_std or not per_prompt:
        raise ValueError('TRL normalize_then_sum requires per-prompt group statistics')
    codes = np.asarray(rewards['_source_index'])
    if codes.shape != (len(prompts),) or not np.isfinite(codes).all():
        raise ValueError('Invalid source indices')
    if not np.isin(codes, np.arange(len(source_weights))).all():
        raise ValueError('Unknown source indices')
    group_ids = np.asarray(rewards['_group_index'])
    if group_ids.shape != (len(prompts),) or not np.issubdtype(group_ids.dtype, np.integer):
        raise ValueError('Generation group indices must be aligned integers')
    result = np.zeros(len(prompts), dtype=np.float64)
    for code, (source, weights) in enumerate(sorted(source_weights.items())):
        mask = codes == code
        if not mask.any():
            continue
        combined = np.zeros(int(mask.sum()), dtype=np.float64)
        local_groups = group_ids[mask]
        for name, weight in weights.items():
            if weight == 0:
                continue
            values = np.asarray(rewards[name])[mask]
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError(f'Invalid/nonfinite applicable reward {source}/{name}')
            normalized = np.zeros_like(values, dtype=np.float64)
            for group_id in np.unique(local_groups):
                group = local_groups == group_id
                std = values[group].std(ddof=1) if group.sum() > 1 else 0.
                normalized[group] = (values[group] - values[group].mean()) / (std + 1e-4)
            combined += weight * normalized
        result[mask] = combined
    batch_std = result.std(ddof=1) if result.size > 1 else 0.
    result = (result - result.mean()) / (batch_std + 1e-4)
    return np.repeat(result[:, None], num_timesteps, axis=1)


def validate_source_weights(reward_fn, source_weights, stratified, prefix):
    """Validate source-specific weights for the selected training rewards."""
    if type(stratified) is not bool:
        raise ValueError(f'{prefix}.source_stratified_sampling must be a boolean')
    if not isinstance(source_weights, dict):
        raise ValueError(f'{prefix}.reward_source_weights must be a mapping')
    if stratified and not source_weights:
        raise ValueError(f'{prefix}.source_stratified_sampling requires reward_source_weights')
    trained = {name for name, weight in reward_fn.items() if weight}
    for source, weights in source_weights.items():
        if (not isinstance(source, str) or not source or not isinstance(weights, dict) or not weights
                or any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
                       for w in weights.values())
                or not any(weights.values())):
            raise ValueError(f'{prefix}.reward_source_weights[{source!r}] needs finite nonnegative weights')
        unknown = {name for name, weight in weights.items() if weight} - trained
        if unknown:
            raise ValueError(f'{prefix}.reward_source_weights[{source!r}] uses rewards missing '
                             f'from {prefix}.reward_fn: {sorted(unknown)}')


def source_credit(scores, source_index, group_keys, source_weights):
    """1-D ``source_advantages`` for a rollout window grouped by prompt text.

    ``scores`` maps reward names to per-sample values (NaN where inactive), as
    returned by ``SourceRewardScorer``. Equal keys share one statistics group,
    matching the single-source windows that pool identical prompts.
    """
    keys = list(group_keys)
    ids = {key: i for i, key in enumerate(dict.fromkeys(keys))}
    rewards = {name: np.asarray(values, dtype=np.float64) for name, values in scores.items()}
    rewards['_source_index'] = np.asarray(source_index, dtype=np.float64)
    rewards['_group_index'] = np.asarray([ids[key] for key in keys], dtype=np.int64)
    return source_advantages(rewards, keys, source_weights, 1)[:, 0]


class SourceRewardWindow:
    """Mixed-source reward credit across a rollout window.

    One object owns the whole path: score each sample with only its source's
    rewards, gather the rollout window across ranks, apply ``source_credit``
    (grouped by prompt text) and report applicable-only reward metrics.
    """

    def __init__(self, reward_fn, source_weights, *, batch_size, adv_clip_max):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('reward batch_size must be a positive integer')
        self.reward_fn, self.source_weights = dict(reward_fn), copy.deepcopy(source_weights)
        self.batch_size, self.adv_clip_max = batch_size, float(adv_clip_max)
        self.names = sorted({name for weights in source_weights.values()
                             for name, weight in weights.items() if weight})

        self.weight_table = torch.tensor([[float(weights.get(name, 0.)) for name in self.names]
                                          for _, weights in sorted(source_weights.items())])
        self._scorer = None

    @classmethod
    def from_config(cls, section):
        """Build the reward window when source-specific weights are configured."""
        if not section.reward_source_weights:
            return None
        return cls(section.reward_fn, section.reward_source_weights, batch_size=section.reward_batch_size,
                   adv_clip_max=section.adv_clip_max)

    def scorer(self, device):
        if self._scorer is None:
            weights = {name: self.reward_fn[name] for name in self.names}
            scorer = MultiScorer(device, weights).freeze()
            self._scorer = SourceRewardScorer(scorer, self.source_weights)
        return self._scorer.to(device)

    @torch.no_grad()
    def score(self, decode, latents, prompts, metadata):
        """Scores [sample, reward] (NaN where inactive) and source codes [sample].

        ``decode`` maps a latent chunk to images, so only one chunk is decoded at a time.
        """
        if len(prompts) != len(latents) or len(metadata) != len(latents):
            raise ValueError('reward batch must align with prompts and metadata')
        from src.rewards.text_reuse import reuse_reward_text
        scorer, values, codes = self.scorer(latents.device), [], []
        with reuse_reward_text():
            for start in range(0, len(latents), self.batch_size):
                stop = min(start + self.batch_size, len(latents))

                scores = scorer(decode(latents[start:stop].detach()), prompts[start:stop],
                                   metadata=metadata[start:stop])
                values.append(torch.stack([torch.as_tensor(scores[name], device=latents.device,
                                                           dtype=torch.float32) for name in self.names], dim=1))
                codes.append(torch.as_tensor(scores['_source_index'], device=latents.device).long())
        return torch.cat(values), torch.cat(codes)

    @torch.no_grad()
    def credit(self, scores, codes, groups, rows=None):
        """Clipped credit for this rank's samples and window metrics.

        By default ``scores``/``codes``/``groups`` are this rank's samples, in
        sample order, and are gathered here. A caller that already gathered the
        whole window passes it with ``rows``, the slice of it on this rank.
        Statistics always use the whole window.
        """
        import torch.distributed as dist
        n = len(scores)
        if scores.shape != (n, len(self.names)) or codes.shape != (n,) or len(groups) != n:
            raise ValueError('scores, source codes and groups must align')
        groups = list(groups)
        if rows is None:
            rows = slice(0, n)
            if dist.is_initialized():
                rank, world = dist.get_rank(), dist.get_world_size()

                packed = torch.cat([scores, codes[:, None].to(scores.dtype)], 1).contiguous()
                gathered = [torch.empty_like(packed) for _ in range(world)]
                dist.all_gather(gathered, packed)
                gathered_groups = [None] * world
                dist.all_gather_object(gathered_groups, groups)
                packed = torch.cat(gathered)
                scores, codes = packed[:, :-1], packed[:, -1].long()
                groups = [key for shard in gathered_groups for key in shard]
                rows = slice(rank * n, (rank + 1) * n)
        values = {name: scores[:, i].double().cpu().numpy() for i, name in enumerate(self.names)}
        source_codes = codes.cpu().numpy()
        credit = torch.as_tensor(source_credit(values, source_codes, groups, self.source_weights),
                                 dtype=torch.float32, device=scores.device)
        credit = credit.clamp(-self.adv_clip_max, self.adv_clip_max)[rows]

        raw = (self.weight_table.to(scores.device)[codes] * torch.nan_to_num(scores)).sum(1)
        metrics = {'reward_mean': float(raw.mean()),
                   **source_reward_metrics(values, source_codes, self.source_weights)}
        return credit, metrics


def source_reward_metrics(scores, source_index, source_weights):
    """Applicable-only reward means: overall per reward and per source/reward."""
    codes = np.asarray(source_index)
    metrics = {}
    for name, values in scores.items():
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        if finite.any():
            metrics[f'reward/{name}'] = float(values[finite].mean())
    for code, (source, weights) in enumerate(sorted(source_weights.items())):
        mask = codes == code
        metrics[f'source/{source}/num_samples'] = float(mask.sum())
        for name, weight in weights.items():
            if weight and mask.any():
                metrics[f'reward/{source}/{name}'] = float(np.asarray(scores[name])[mask].mean())
    return metrics
