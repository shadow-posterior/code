"""Save and restore LoRA training checkpoints and generator exports."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TYPE_CHECKING

import torch
from safetensors.torch import save_file as safetensors_save_file
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from torch.nn.parallel import DistributedDataParallel as DDP

if TYPE_CHECKING:
    from src.config import LoRAConfig

from src.parallel import is_main_process
from src.paths import project_path

logger = logging.getLogger(__name__)


def _gather_fsdp_state_dict(model: torch.nn.Module) -> dict:
    """Gather a full, wrapper-free state dict from an arbitrary model.

    For FSDP models, gathers shards to rank 0 with CPU offloading.
    For DDP models, returns the inner module's state dict (strips the
    ``module.`` prefix that ``DDP.state_dict()`` would otherwise add) so the
    on-disk key naming is identical to the unwrapped transformer regardless of
    the distributed strategy used at training time.
    For plain modules, returns ``state_dict()`` directly.
    """
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            return model.state_dict()
    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()


def _filter_lora_state_dict(state_dict: dict) -> dict:
    """Filter a state_dict to only keep LoRA parameters (lora_A/lora_B).

    Args:
        state_dict: Full model state_dict.

    Returns:
        Filtered state_dict containing only LoRA adapter weights.
    """
    return {k: v for k, v in state_dict.items()
            if "lora_A" in k or "lora_B" in k}


def _is_lora_checkpoint(state_dict: dict) -> bool:
    """Detect if a state_dict is a LoRA-only checkpoint.

    A LoRA checkpoint contains only lora_A/lora_B keys and is much smaller
    than a full model checkpoint. We check if ALL keys contain lora_ patterns.

    Args:
        state_dict: A loaded state_dict.

    Returns:
        True if this appears to be a LoRA-only checkpoint.
    """
    if not state_dict:
        return False
    return all("lora_A" in k or "lora_B" in k for k in state_dict.keys())


def _load_adapters(model: torch.nn.Module, state_dict: dict) -> None:
    """Load adapter weights into the wrapped model on every rank."""
    def load():
        if not _is_lora_checkpoint(state_dict):
            raise ValueError('Checkpoint must contain LoRA adapter weights')
        if isinstance(model, FSDP):
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                keys = model.load_state_dict(state_dict, strict=False)
        else:
            inner = model.module if isinstance(model, DDP) else model
            keys = inner.load_state_dict(state_dict, strict=False)
        missing = [name for name in keys.missing_keys if 'lora_A' in name or 'lora_B' in name]
        if missing or keys.unexpected_keys:
            raise RuntimeError(f'LoRA checkpoint keys differ: missing={missing}, unexpected={keys.unexpected_keys}')

    error = None
    try:
        load()
    except Exception as exc:
        if not torch.distributed.is_initialized():
            raise
        error = f'{type(exc).__name__}: {exc}'
    if torch.distributed.is_initialized():
        errors = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(errors, error)
        failures = [f'rank {rank}: {message}' for rank, message in enumerate(errors) if message]
        if failures:
            raise RuntimeError('LoRA checkpoint load failed: ' + '; '.join(failures))


def save_lora_peft_format(
    state_dict: dict,
    lora_config: LoRAConfig,
    save_dir: str,
    *,
    pretrained_path: str,
) -> None:
    """Export the generator update as a PEFT adapter for the initialized backbone."""
    os.makedirs(save_dir, exist_ok=True)

    lora_state = {
        'base_model.model.' + name.replace('.lora_A.default.', '.lora_A.').replace('.lora_B.default.', '.lora_B.'): weight
        for name, weight in _filter_lora_state_dict(state_dict).items()
    }

    safetensors_save_file(lora_state, os.path.join(save_dir, "adapter_model.safetensors"))

    adapter_config = {
        "r": lora_config.rank,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": list(lora_config.target_modules),
        "lora_dropout": 0.0,
        "bias": "none",
        "peft_type": "LORA",
        "task_type": None,
        "init_lora_weights": lora_config.init_lora_weights,
        "use_rslora": False,
        "base_model_name_or_path": pretrained_path,
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    logger.info(f"  LoRA saved in peft format: {save_dir} ({len(lora_state)} keys)")


def is_peft_lora_dir(path: str) -> bool:
    """Check if a path is a peft-format LoRA directory.

    A peft LoRA directory contains adapter_config.json and adapter_model.safetensors.
    """
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(path, "adapter_config.json"))
    )


_OPTIMIZER_MODELS = {'generator': 'generator', 'fake_score': 'fake_score_net'}


def save_checkpoint(
    output_dir: str,
    step: int,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    generator_lora: LoRAConfig,
    pretrained_path: str,
    extra_state: dict[str, Any],
) -> None:
    """Gather model and optimizer state on all ranks, then write on rank zero."""
    ckpt_dir = os.path.join(project_path(output_dir), f"checkpoint-{step}")
    model_states = {name: _gather_fsdp_state_dict(model) for name, model in models.items()}
    optimizer_states = {}
    for name, optimizer in optimizers.items():
        model = models[_OPTIMIZER_MODELS[name]]
        optimizer_states[name] = (FSDP.full_optim_state_dict(model, optimizer)
                                  if isinstance(model, FSDP) else optimizer.state_dict())

    def write_checkpoint_files():
        os.makedirs(ckpt_dir, exist_ok=True)
        generator = model_states.get('generator_ema', model_states['generator'])
        save_lora_peft_format(
            generator, generator_lora, os.path.join(ckpt_dir, 'transformer'), pretrained_path=pretrained_path,
        )

        for name, state_dict in model_states.items():
            torch.save(_filter_lora_state_dict(state_dict), os.path.join(ckpt_dir, f'{name}.pt'))
        for name, state_dict in optimizer_states.items():
            torch.save(state_dict, os.path.join(ckpt_dir, f'optimizer_{name}.pt'))

        torch.save({**extra_state, 'step': step}, os.path.join(ckpt_dir, 'meta.pt'))
        logger.info('Saved checkpoint %s', ckpt_dir)

    distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    error = [None]
    if is_main_process():
        try:
            write_checkpoint_files()
        except Exception as exc:
            if not distributed:
                raise
            error[0] = f'{type(exc).__name__}: {exc}'
    if distributed:
        torch.distributed.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(f'Checkpoint save failed on rank 0: {error[0]}')


def load_checkpoint(
    ckpt_dir: str,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
) -> dict[str, Any]:
    """Load the adapters, optimizers and metadata of a training checkpoint."""
    required = ['meta.pt', *(f'{name}.pt' for name in models),
                *(f'optimizer_{name}.pt' for name in optimizers)]
    missing = [name for name in required
               if not os.path.isfile(os.path.join(ckpt_dir, name))
               or os.path.getsize(os.path.join(ckpt_dir, name)) == 0]
    if missing:
        raise FileNotFoundError(f'Incomplete checkpoint {ckpt_dir}: missing or empty {missing}')
    meta = torch.load(os.path.join(ckpt_dir, 'meta.pt'), map_location='cpu', weights_only=False)

    for name, model in models.items():
        state_dict = torch.load(os.path.join(ckpt_dir, f'{name}.pt'), map_location='cpu', weights_only=True)
        _load_adapters(model, state_dict)

    for name, optimizer in optimizers.items():
        path = os.path.join(ckpt_dir, f'optimizer_{name}.pt')
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        model = models[_OPTIMIZER_MODELS[name]]
        if isinstance(model, FSDP):
            state_dict = FSDP.optim_state_dict_to_load(model, optimizer, state_dict)
        optimizer.load_state_dict(state_dict)

    return meta
