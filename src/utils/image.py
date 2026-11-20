"""
Image decoding utilities for SPM.

Standalone VAE decode functions extracted from BaseTrainer so they can be
used by both training (evaluation) and standalone inference.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def decode_latents_to_tensor(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode latent tensors to float NCHW tensor [0, 1] using the VAE.

    Args:
        vae: AutoencoderKL instance.
        latents: Latent tensors [B, C, H, W] on GPU.

    Returns:
        Float tensor [B, C, H, W] in [0, 1] range on GPU.
    """
    device = latents.device
    if next(vae.parameters()).device != device:
        vae.to(device)

    scaling_factor = vae.config.scaling_factor
    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    scaled_latents = latents / scaling_factor + shift_factor
    images = vae.decode(scaled_latents.to(vae.dtype)).sample

    images = (images / 2 + 0.5).clamp(0, 1)
    return images
