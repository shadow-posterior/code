"""Stochastic rollout steps for flow-matching models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput,
)


class RolloutScheduler(FlowMatchEulerDiscreteScheduler):
    """Predict clean latents and add fresh noise at the next level."""

    def set_timesteps(
        self, *, sigmas: Sequence[float], device: str | torch.device | None = None,
    ) -> None:
        """Install the already-shifted denoising schedule."""
        levels = torch.tensor([*sigmas, 0.], dtype=torch.float32, device=device)
        self.sigmas = levels
        self.timesteps = levels[:-1] * self.config.num_train_timesteps
        self.num_inference_steps = len(self.timesteps)
        self._step_index = None
        self._begin_index = None

    def step(
        self, model_output: torch.Tensor, timestep: float | torch.Tensor, sample: torch.Tensor,
    ) -> FlowMatchEulerDiscreteSchedulerOutput:
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        clean = sample.float() - sigma * model_output.float()
        if sigma_next != 0.:
            noise = torch.randn(sample.shape, device=sample.device, dtype=torch.float32)
            clean = (1. - sigma_next) * clean + sigma_next * noise

        self._step_index += 1
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=clean.to(model_output.dtype))
