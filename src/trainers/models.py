"""Model setup and rollout operations shared by SPM's student and score networks."""

from __future__ import annotations

from src.paths import local_model_path

import copy
import logging
import os

import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_sd3 import JointTransformerBlock
from diffusers.training_utils import compute_density_for_timestep_sampling

from src.config import TrainingConfig
from src.scheduler import RolloutScheduler
from src.backbone import (
    compute_denoising_sigmas,
    load_model_components,
    encode_prompts,
    extract_into_tensor,
    predict_velocity,
)
from src.parallel import (
    fsdp_wrap_model,
    get_transformer_wrap_policy,
)
from src.trainers.base import BaseTrainer
from src.utils.checkpoint import (
    _is_lora_checkpoint,
    is_peft_lora_dir,
)
from src.utils.fast_init import fast_init
from src.utils.lora import setup_lora

logger = logging.getLogger(__name__)


def _replace_patch_embed_proj_with_linear(transformer: nn.Module) -> None:
    """Replace PatchEmbed.proj Conv2d with an equivalent nn.Linear.

    FSDP use_orig_params=True has a known shape-writeback bug when a model
    contains Conv2d layers whose weights have 4-D shape (out_ch, in_ch, kH, kW).
    During unsharding, FSDP tries to write back a flat 1-D tensor into the
    original parameter, but the stored shape is 4-D, causing:

        RuntimeError: Cannot writeback when the parameter shape changes.
        Expects torch.Size([98304]) but got torch.Size([1536, 16, 2, 2])

    SD3-medium's PatchEmbed.proj is a Conv2d(16, 1536, kernel_size=2, stride=2)
    that acts as a pure linear projection (no spatial mixing because
    kernel_size == stride == patch_size). We can safely replace it with a
    weight-equivalent nn.Linear and patch the forward method of PatchEmbed to
    reshape the input appropriately.

    The replacement is done in-place before FSDP wrapping so no ignored_modules
    the projection should be converted.
    """
    if not hasattr(transformer, "pos_embed"):
        return
    patch_embed = transformer.pos_embed
    if not hasattr(patch_embed, "proj"):
        return
    conv = patch_embed.proj
    if not isinstance(conv, nn.Conv2d):
        return

    out_ch, in_ch, kH, kW = conv.weight.shape
    in_features = in_ch * kH * kW

    linear = nn.Linear(
        in_features, out_ch,
        bias=(conv.bias is not None),
        dtype=conv.weight.dtype,
    )
    linear.weight.data.copy_(conv.weight.data.view(out_ch, -1))
    if conv.bias is not None:
        linear.bias.data.copy_(conv.bias.data)

    linear.weight.requires_grad_(conv.weight.requires_grad)
    if conv.bias is not None:
        linear.bias.requires_grad_(conv.bias.requires_grad)

    linear._patch_size = kH

    patch_embed.proj = linear

    original_forward = patch_embed.__class__.forward

    def _patched_forward(self, latent):
        if isinstance(self.proj, nn.Conv2d):
            return original_forward(self, latent)

        p = self.proj._patch_size
        B, C, H, W = latent.shape
        Hp, Wp = H // p, W // p

        patches = latent.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, C, Hp, Wp, p * p)
        patches = patches.permute(0, 2, 3, 1, 4).contiguous()
        patches = patches.view(B, Hp * Wp, C * p * p)

        out = self.proj(patches)

        out = out.view(B, Hp, Wp, -1).permute(0, 3, 1, 2)

        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = Hp, Wp
        if self.flatten:
            out = out.flatten(2).transpose(1, 2)
        if self.layer_norm:
            out = self.norm(out)
        if self.pos_embed is None:
            return out.to(out.dtype)
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            from diffusers.models.embeddings import get_2d_sincos_pos_embed
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                    device=out.device,
                    output_type="pt",
                )
                pos_embed = pos_embed.float().unsqueeze(0)
            else:
                pos_embed = self.pos_embed
        return (out + pos_embed).to(out.dtype)

    import types
    patch_embed.forward = types.MethodType(_patched_forward, patch_embed)

    logger.debug(
        f"Replaced PatchEmbed.proj Conv2d({in_ch},{out_ch},{kH},{kW}) "
        f"with Linear({in_features},{out_ch}) for FSDP compatibility"
    )


def load_model_init_from_path(
    model: nn.Module,
    init_path: str,
    role_name: str,
) -> None:
    """Load role-specific base-weight initialization checkpoint into a model."""
    init_path = init_path.strip()
    if not init_path:
        return
    init_path = local_model_path(init_path)

    if is_peft_lora_dir(init_path):
        suffix = f"{role_name}_lora.load_path"
        raise ValueError(
            f"{role_name}_init_path expects base-weight checkpoints, but got peft LoRA dir: {init_path}. "
            f"Please use {suffix} for LoRA adapter initialization."
        )

    if os.path.isfile(init_path):
        state_dict = torch.load(init_path, map_location="cpu", weights_only=False)
        is_lora = _is_lora_checkpoint(state_dict)
        if is_lora:
            suffix = f"{role_name}_lora.load_path"
            raise ValueError(
                f"{role_name}_init_path expects base-weight checkpoints, but got LoRA-only state_dict: {init_path}. "
                f"Please use {suffix} for LoRA adapter initialization."
            )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(
            f"  {role_name}: loaded init checkpoint {init_path} "
            f"(keys={len(state_dict)}, missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return

    model_dir = init_path
    if not os.path.isfile(os.path.join(model_dir, 'config.json')):
        model_dir = os.path.join(init_path, 'transformer')
    if not os.path.isfile(os.path.join(model_dir, 'config.json')):
        raise FileNotFoundError(f'{role_name}_init_path has no transformer config under {init_path}')
    with fast_init(torch.device('cpu')):
        loaded_model = type(model).from_pretrained(
            model_dir, torch_dtype=next(model.parameters()).dtype, local_files_only=True,
        )

    state_dict = loaded_model.state_dict()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info('Loaded %s transformer from %s (keys=%s, missing=%s, unexpected=%s)',
                role_name, model_dir, len(state_dict), len(missing), len(unexpected))


class ModelTrainingBase(BaseTrainer):
    """Initialize the generator, fake score network and frozen teacher."""

    def __init__(self, config: TrainingConfig):
        if type(config.train.ema_every) is not int or config.train.ema_every < 1:
            raise ValueError('train.ema_every must be a positive integer')
        super().__init__(config)

        self.generator: nn.Module | None = None
        self.generator_ema: nn.Module | None = None
        self.fake_score_net: nn.Module | None = None
        self.teacher: nn.Module | None = None
        self.text_encoders: list[nn.Module] = []
        self.tokenizers: list = []
        self.scheduler: FlowMatchEulerDiscreteScheduler | None = None
        self.x0_scheduler = None

        self.sigmas: torch.Tensor | None = None

        self._uncond_embeds: tuple | None = None

        self._denoising_sigmas_cache: dict[int, list[float]] = {}

    def setup_models(self) -> None:
        """Load the student, score network and teacher.

        Creates:
        - generator: copy of pretrained transformer (trainable)
        - fake_score_net: copy of pretrained transformer (trainable)
        - teacher: original pretrained transformer (frozen)
        - text_encoders: frozen, shared across all three
        """
        model_cfg = self.config.model
        dist_cfg = self.config.distributed

        logger.info("=" * 60)
        logger.info("Setting up training models")
        logger.info("=" * 60)

        components = load_model_components(model_cfg)
        base_transformer = components["transformer"]
        self.text_encoders = components["text_encoders"]
        self.tokenizers = components["tokenizers"]
        self.scheduler = components["scheduler"]
        if self.scheduler.config.num_train_timesteps != self.config.sampling.num_train_timesteps:
            raise ValueError('sampling.num_train_timesteps must match the pretrained scheduler')

        if model_cfg.gradient_checkpointing:
            base_transformer.enable_gradient_checkpointing()

        self.generator = copy.deepcopy(base_transformer)
        self.fake_score_net = copy.deepcopy(base_transformer)

        self.teacher = base_transformer

        load_model_init_from_path(
            self.generator,
            model_cfg.generator_init_path,
            "generator",
        )
        load_model_init_from_path(
            self.fake_score_net,
            model_cfg.fake_score_init_path,
            "fake_score",
        )
        load_model_init_from_path(
            self.teacher,
            model_cfg.teacher_init_path,
            "teacher",
        )

        setup_lora(self.generator, model_cfg.generator_lora)
        setup_lora(self.fake_score_net, model_cfg.fake_score_lora)

        self.teacher.requires_grad_(False)
        self.teacher.eval()
        logger.info("  Teacher: frozen (no grad)")

        if self.config.train.use_ema:
            self.generator_ema = copy.deepcopy(self.generator)
            self.generator_ema.requires_grad_(False)
            self.generator_ema.eval()
            logger.info(
                f"  Generator EMA: created (decay={self.config.train.ema_decay}, "
                f"every={self.config.train.ema_every})"
            )
        else:
            self.generator_ema = None

        for enc in self.text_encoders:
            enc.requires_grad_(False)
            enc.eval()
        logger.info("  Text encoders: frozen")

        self.vae = components["vae"]
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(torch.device("cuda"))
        logger.info(f"  VAE: frozen, stored on GPU ({self.vae.__class__.__name__})")

        vae_scale_factor = 2 ** (len(components["vae"].config.block_out_channels) - 1)
        self.latent_channels = base_transformer.config.in_channels
        latent_divisor = vae_scale_factor
        if model_cfg.image_resolution % latent_divisor != 0:
            raise ValueError(
                f"image_resolution={model_cfg.image_resolution} must be divisible by {latent_divisor}"
            )
        self.latent_size = model_cfg.image_resolution // latent_divisor
        logger.info(
            f"  Latent shape: [{self.latent_channels}, {self.latent_size}, {self.latent_size}] "
            f"(image_res={model_cfg.image_resolution}, latent_divisor={latent_divisor})"
        )

        self.sigmas = torch.flip(self.scheduler.sigmas, dims=[0]).cuda()

        self.x0_scheduler = RolloutScheduler.from_config(self.scheduler.config)
        self._denoising_sigmas_cache.clear()
        self._denoising_sigmas = self._get_denoising_sigmas()
        self.x0_scheduler.set_timesteps(sigmas=self._denoising_sigmas)
        logger.info(
            f"  X0 scheduler: denoising_step_list={self.config.sampling.denoising_step_list}, "
            f"default_sigmas={self._denoising_sigmas}, "
            f"timesteps={self.x0_scheduler.timesteps.tolist()}"
        )

        self._fp32_master(self.generator, self.fake_score_net, self.generator_ema)

        self._pre_wrap_models()

        self._wrap_models(dist_cfg)

        self.models = {
            "generator": self.generator,
            "fake_score_net": self.fake_score_net,
        }
        if self.generator_ema is not None:
            self.models["generator_ema"] = self.generator_ema

        device = torch.device("cuda")
        for enc in self.text_encoders:
            enc.to(device)

        logger.info("Model setup complete")
        logger.info("=" * 60)

    def _pre_wrap_models(self) -> None:
        """Prepare patch embeddings for FSDP when linear projections are requested."""
        if not self.config.model.fsdp_patch_embed_linear:
            return
        for m in [self.generator, self.fake_score_net, self.teacher, self.generator_ema]:
            if m is not None:
                _replace_patch_embed_proj_with_linear(m)

    def _wrap_distributed_model(self, model, dist_cfg, *, trainable):
        """Place a model on its training device and apply distributed wrapping."""
        if dist_cfg.strategy == "fsdp":
            return fsdp_wrap_model(
                model, sharding_strategy=dist_cfg.fsdp_sharding,
                fsdp_precision=self.config.model.dtype, auto_wrap_policy=get_transformer_wrap_policy(JointTransformerBlock),
                cpu_offload=bool(not trainable and dist_cfg.fsdp_cpu_offload_frozen),
            )
        if dist_cfg.strategy == "ddp":
            from src.parallel import ddp_wrap_model
            model = model.to(torch.device("cuda"))
            if trainable:
                model = ddp_wrap_model(model, find_unused_parameters=any(
                    not p.requires_grad for p in model.parameters()))
            return model
        raise ValueError(f"Unknown distributed strategy: {dist_cfg.strategy}")

    def _wrap_models(self, dist_cfg) -> None:
        """Use the shared role wrapper for trainable and frozen models."""
        for name in ("generator", "fake_score_net", "teacher", "generator_ema"):
            model = getattr(self, name)
            if model is not None:
                setattr(self, name, self._wrap_distributed_model(
                    model, dist_cfg, trainable=name in ("generator", "fake_score_net")))

    def setup_optimizers(self) -> None:
        """Create separate optimizers for generator and fake score network.

        When LoRA is enabled, only LoRA parameters (requires_grad=True) are
        passed to the optimizer, reducing optimizer state memory.
        """
        gen_cfg = self.config.solver.generator
        fake_cfg = self.config.solver.fake_score

        gen_params = [p for p in self.generator.parameters() if p.requires_grad]
        fake_params = [p for p in self.fake_score_net.parameters() if p.requires_grad]

        self.optimizers["generator"] = torch.optim.AdamW(
            gen_params,
            lr=gen_cfg.lr,
            betas=(gen_cfg.beta1, gen_cfg.beta2),
            eps=gen_cfg.eps,
            weight_decay=gen_cfg.weight_decay,
        )
        self.optimizers["fake_score"] = torch.optim.AdamW(
            fake_params,
            lr=fake_cfg.lr,
            betas=(fake_cfg.beta1, fake_cfg.beta2),
            eps=fake_cfg.eps,
            weight_decay=fake_cfg.weight_decay,
        )

        self.schedulers["generator"] = self.create_warmup_constant_scheduler(
            self.optimizers["generator"], gen_cfg.warmup_steps,
        )
        self.schedulers["fake_score"] = self.create_warmup_constant_scheduler(
            self.optimizers["fake_score"], fake_cfg.warmup_steps,
        )

        logger.info(
            f"Optimizers created: generator (lr={gen_cfg.lr}, params={len(gen_params)}), "
            f"fake_score (lr={fake_cfg.lr}, params={len(fake_params)})"
        )

    def _get_denoising_sigmas(self, num_steps: int | None = None) -> list[float]:
        """Return denoising sigmas for generation, cached by step count."""
        max_steps = len(self.config.sampling.denoising_step_list)
        clamped_steps = max_steps if num_steps is None else max(1, min(int(num_steps), max_steps))
        if clamped_steps in self._denoising_sigmas_cache:
            return self._denoising_sigmas_cache[clamped_steps]

        sigmas = compute_denoising_sigmas(
            self.config.sampling.denoising_step_list[:clamped_steps],
            num_train_timesteps=self.scheduler.config.num_train_timesteps,
            shift=self.scheduler.config.shift,
        )

        self._denoising_sigmas_cache[clamped_steps] = sigmas
        return sigmas

    def _predict_noise(
        self,
        model: nn.Module,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_embeds: torch.Tensor,
        guidance_scale: float = 1.0,
        uncond_text_embeddings: torch.Tensor | None = None,
        uncond_pooled_prompt_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict SD3.5 velocity with optional classifier-free guidance."""
        return predict_velocity(
            model,
            noisy_latents,
            prompt_embeds,
            timesteps,
            pooled_embeds,
            guidance_scale=guidance_scale,
            uncond_text_embeddings=uncond_text_embeddings,
            uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds,
        )

    @torch.no_grad()
    def _generate_latents(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        *,
        num_steps: int | None = None,
        model: nn.Module,
    ) -> tuple[torch.Tensor, int]:
        """Run the stochastic rollout and return its clean prediction and call count."""
        batch_size = len(prompt_embeds)
        device, dtype = prompt_embeds.device, prompt_embeds.dtype
        sigmas = self._get_denoising_sigmas(num_steps)
        self.x0_scheduler.set_timesteps(sigmas=sigmas)

        x_t = torch.randn(
            batch_size, self.latent_channels, self.latent_size, self.latent_size,
            device=device, dtype=dtype,
        )
        for index, timestep in enumerate(self.x0_scheduler.timesteps):
            model_time = timestep.expand(batch_size).to(device=device, dtype=torch.float32)
            with self._autocast():
                velocity = self._predict_noise(model, x_t, prompt_embeds, model_time, pooled_embeds)
            result = self.x0_scheduler.step(velocity, timestep, x_t)
            sigma = self.x0_scheduler.sigmas[index]
            clean = (x_t.float() - sigma * velocity.float()).to(dtype)
            x_t = result.prev_sample
        return clean, len(sigmas)

    def _encode_prompts(
        self, prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of text prompts using the SD3.5 encoders.

        Returns:
            Tuple of (prompt_embeds, pooled_embeds).
        """
        device = torch.device("cuda")
        return encode_prompts(
            prompts, self.text_encoders, self.tokenizers, device,
        )

    @torch.no_grad()
    def _get_uncond_embeds(
        self, batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get unconditional (empty string) embeddings for classifier-free guidance.

        Caches the result and expands to match batch size.

        Returns:
            Tuple of (uncond_embeds, uncond_pooled_embeds).
        """
        if self._uncond_embeds is None:
            device = torch.device("cuda")
            self._uncond_embeds = encode_prompts(
                [""], self.text_encoders, self.tokenizers, device,
            )

        uncond_embeds, uncond_pooled = self._uncond_embeds
        return (
            uncond_embeds.expand(batch_size, -1, -1),
            uncond_pooled.expand(batch_size, -1),
        )

    @staticmethod
    def _clip_grad_norm(model: nn.Module, max_norm: float) -> float:
        """Clip gradient norm, FSDP-aware.

        For FSDP models, the standard torch.nn.utils.clip_grad_norm_ only
        computes the norm of the local shard, which is incorrect.
        FSDP.clip_grad_norm_ gathers the global norm across all ranks.

        Returns:
            Global gradient norm before clipping.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(model, FSDP):
            norm = model.clip_grad_norm_(max_norm)
            if not torch.isfinite(torch.as_tensor(norm)).all():
                raise FloatingPointError('Global gradient norm is not finite; optimizer step aborted.')
        else:
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm, error_if_nonfinite=True)
        return float(norm.item() if isinstance(norm, torch.Tensor) else norm)

    def _query_score_schedule(self, clean):
        """Sample queries from the shifted training sigma table."""
        c = self.config.sampling
        indices = self._sample_timesteps(len(clean), c.min_step, c.max_step,
            c.timestep_sampling, c.logit_mean, c.logit_std, clean.device)
        sigma = extract_into_tensor(self.sigmas, indices, clean.shape)
        return self._sigma_to_model_timestep(sigma).view(-1), sigma

    def _noise_clean_latents(self, clean, sigma):
        """Apply the flow-matching forward kernel and return its velocity target."""
        noise = torch.randn_like(clean)
        return (1.0 - sigma) * clean + sigma * noise, noise - clean

    def _sample_timesteps(
        self,
        batch_size: int,
        min_step: int,
        max_step: int,
        sampling: str,
        logit_mean: float,
        logit_std: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample integer timestep indices with configurable strategy.

        Args:
            batch_size: Number of timesteps to sample.
            min_step: Minimum timestep index (inclusive).
            max_step: Maximum timestep index (inclusive).
            sampling: "uniform" or "logit_normal".
            logit_mean: Logit-normal mean (only used when sampling == "logit_normal").
            logit_std: Logit-normal std (only used when sampling == "logit_normal").
            device: Target device.

        Returns:
            Integer timestep indices [batch_size] in [min_step, max_step].
        """
        step_range = max_step - min_step + 1

        if sampling == "logit_normal":
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=batch_size,
                logit_mean=logit_mean,
                logit_std=logit_std,
            )
            indices = (u * step_range).long().clamp(0, step_range - 1) + min_step
            return indices.to(device)
        else:
            return torch.randint(
                min_step, max_step + 1, (batch_size,),
                device=device, dtype=torch.long,
            )

    def _num_train_timesteps(self) -> float:
        """Return scheduler num_train_timesteps as float.

        In this codebase, the transformer-facing timestep convention is:
            timestep_model = sigma * num_train_timesteps
        where sigma is already the schedule-space sigma used by flow matching
        (including any SD3.5 shift that has already been absorbed when sigma is
        constructed from scheduler tables).
        """
        if self.scheduler is not None and hasattr(self.scheduler, "config"):
            return float(
                getattr(
                    self.scheduler.config,
                    "num_train_timesteps",
                    self.config.sampling.num_train_timesteps,
                )
            )
        return float(self.config.sampling.num_train_timesteps)

    def _sigma_to_model_timestep(
        self,
        sigma: torch.Tensor | float,
    ) -> torch.Tensor | float:
        """Convert schedule-space sigma to model timestep."""
        num_ts = self._num_train_timesteps()
        if isinstance(sigma, torch.Tensor):
            return sigma.float() * num_ts
        return float(sigma) * num_ts
