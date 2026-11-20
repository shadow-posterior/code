"""Batch-local reuse for deterministic frozen text conditioning."""
from __future__ import annotations

import torch


def has_outer_forward_wrappers(model):
    """Unknown outer hooks/wrappers may depend on the original batch rows."""
    module_hooks = torch.nn.modules.module
    return bool(model._forward_pre_hooks or model._forward_hooks
                or module_hooks._global_forward_pre_hooks or module_hooks._global_forward_hooks
                or '_hf_hook' in model.__dict__ or 'forward' in model.__dict__)


def deduplicate_prompt_batch(prompts, text_encoder):
    """Return stable unique prompts and their inverse, or preserve unsafe batches.

    Nothing survives this call. Training-mode or trainable encoders retain every
    row so dropout and other stochastic behavior keep their original semantics.
    """
    if not prompts:
        raise ValueError('Cannot encode an empty prompt batch')
    if (text_encoder.training or has_outer_forward_wrappers(text_encoder)
            or any(p.requires_grad for p in text_encoder.parameters())):
        return prompts, None
    unique = list(dict.fromkeys(prompts))
    if len(unique) == len(prompts):
        return prompts, None
    positions = {prompt: i for i, prompt in enumerate(unique)}
    return unique, [positions[prompt] for prompt in prompts]


def restore_prompt_batch(tensor, inverse):
    """Restore the original prompt order as ordinary tensors usable by autograd."""
    if inverse is None:
        return tensor
    return tensor.index_select(0, torch.tensor(inverse, device=tensor.device, dtype=torch.long))
