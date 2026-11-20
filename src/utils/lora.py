"""Initialize trainable LoRA adapters from merged starting weights."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml
from src.paths import project_path
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import LoRAConfig

logger = logging.getLogger(__name__)


def _load_initial_adapter(model: nn.Module, path: str) -> None:
    """Load an exact adapter tensor dictionary without rounding FP32 weights.

    Base weights are supplied by the pretrained model and are never changed.
    Validate the entire adapter schema before copying any checkpoint values.
    """
    resolved = Path(project_path(path))
    if resolved.is_dir():
        declared = json.loads((resolved / 'adapter_config.json').read_text()).get('base_model_name_or_path')
        base = getattr(getattr(model, 'config', None), '_name_or_path', None) or declared
        if base and Path(base).name == 'transformer':
            base = str(Path(base).parent)
        state = _load_peft_adapter(model, resolved, base_model=base, device='cpu')
    else:
        state = torch.load(resolved, map_location="cpu", weights_only=True)
    parameters = {name: parameter for name, parameter in model.named_parameters()
                  if "lora_A" in name or "lora_B" in name}
    if not parameters or not isinstance(state, dict):
        raise ValueError("LoRA checkpoint requires an injected model and an adapter tensor dictionary")
    missing, unexpected = set(parameters) - set(state), set(state) - set(parameters)
    if missing or unexpected:
        raise ValueError(f"LoRA checkpoint keys differ: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for name, parameter in parameters.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"LoRA checkpoint {name} must be a floating-point tensor")
        if value.shape != parameter.shape:
            raise ValueError(f"LoRA checkpoint {name} has shape {tuple(value.shape)}; expected {tuple(parameter.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"LoRA checkpoint {name} must contain only finite values")
    with torch.no_grad():
        for name, parameter in parameters.items():
            dtype = torch.promote_types(parameter.dtype, state[name].dtype)
            parameter.data = parameter.data.to(dtype=dtype)
            parameter.copy_(state[name])
    logger.info(f"Loaded pretrained LoRA weights from {path}")


def _adapter_parameters(model):
    result = {name: p for name, p in model.named_parameters()
              if '.lora_A.default.weight' in name or '.lora_B.default.weight' in name}
    if not result:
        raise ValueError('PEFT loading requires a default projection LoRA slot')

    all_lora = {n for n, _ in model.named_parameters() if 'lora_' in n}
    if all_lora != set(result):
        raise ValueError('PEFT loading requires exactly one default additive projection adapter')
    return result


def _validate_adapter_snapshot(model, values):
    expected = _adapter_parameters(model)
    if not isinstance(values, dict) or set(values) != set(expected):
        raise ValueError('PEFT snapshot keys differ from the reference adapter')
    for name, param in expected.items():
        value = values[name]
        if (not isinstance(value, torch.Tensor) or value.shape != param.shape
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError(f'PEFT snapshot {name} has invalid shape, dtype or finite values')


def _canonical_key(key):
    for prefix in ('base_model.model.', 'transformer.'):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key.replace('.lora_A.default.', '.lora_A.').replace('.lora_B.default.', '.lora_B.')


def _same_base(declared, expected):
    if declared == expected:
        return True
    def local(value):
        return Path(value).is_absolute() or value.startswith(('weights/', './weights/')) or Path(value).is_dir()
    if local(declared) and local(expected):
        return Path(declared).resolve() == Path(expected).resolve()
    if local(expected):
        return Path(declared).name == Path(expected).name
    if local(declared):
        return Path(declared).name == Path(expected).name
    return False


def _load_peft_adapter(model, path, *, base_model, device):
    """Load a standard PEFT effective adapter, validating all executed semantics.

    Ranks must match the reference slot. Global alpha and RS-LoRA differences
    are represented exactly by rescaling B in FP32; unsupported per-layer rank
    patterns and base-mutating initializers fail explicitly.
    """
    from peft import LoraConfig
    from peft.tuners.tuners_utils import check_target_module_exists
    from safetensors.torch import load_file

    directory = Path(path)
    if not (directory / 'adapter_config.json').is_file():
        raise ValueError(f'PEFT adapter requires a PEFT directory with adapter_config.json: {directory}')
    raw = json.loads((directory / 'adapter_config.json').read_text())
    if not isinstance(raw, dict) or raw.get('peft_type') != 'LORA':
        raise ValueError('PEFT adapter requires PEFT LORA format')
    unknown = set(raw) - set(LoraConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f'unknown PEFT LoRA settings: {sorted(unknown)}')
    incoming = LoraConfig(**raw)
    unsupported = {k: raw[k] for k in (
        'use_dora', 'lora_bias', 'modules_to_save', 'target_parameters', 'trainable_token_indices',
        'layer_replication', 'alora_invocation_tokens', 'arrow_config', 'use_qalora',
        'megatron_config', 'rank_pattern', 'alpha_pattern', 'layers_to_transform', 'layers_pattern',
        'fan_in_fan_out', 'ensure_weight_tying') if raw.get(k)}
    if incoming.bias != 'none' or incoming.lora_dropout != 0:
        unsupported['bias/dropout'] = (incoming.bias, incoming.lora_dropout)
    initializer = incoming.init_lora_weights
    if initializer not in (True, False, 'gaussian'):
        unsupported['init_lora_weights'] = initializer
    if unsupported:
        raise ValueError(f'non-equivalent PEFT LoRA format: {unsupported}')
    declared_base = raw.get('base_model_name_or_path')
    if not declared_base:
        readme = directory / 'README.md'
        if readme.is_file():
            match = re.match(r'\A---\s*\n(.*?)\n---', readme.read_text(), flags=re.S)
            if match:
                card = yaml.safe_load(match.group(1))
                declared_base = card.get('base_model') if isinstance(card, dict) else None
    if (not isinstance(declared_base, str) or not declared_base.strip()
            or not _same_base(declared_base, str(base_model))):
        raise ValueError(f'PEFT adapter base model {declared_base!r} differs from {base_model!r} or lacks provenance')
    if Path(declared_base).is_dir() and Path(base_model).is_dir():
        if Path(declared_base).resolve() != Path(base_model).resolve():
            raise ValueError('PEFT adapter and reference point at different local base models')
    current = model.peft_config['default']
    if type(incoming.r) is not int or incoming.r != current.r:
        raise ValueError(f'PEFT adapter rank {incoming.r} differs from reference rank {current.r}')
    if not isinstance(incoming.lora_alpha, (int, float)) or isinstance(incoming.lora_alpha, bool):
        raise ValueError('PEFT adapter lora_alpha must be finite and positive')
    if not math.isfinite(incoming.lora_alpha) or incoming.lora_alpha <= 0:
        raise ValueError('PEFT adapter lora_alpha must be finite and positive')

    modules = [name for name, module in model.named_modules() if hasattr(module, 'lora_A')]
    selected = {name for name in modules if check_target_module_exists(incoming, name)}
    if selected != set(modules):
        raise ValueError('PEFT adapter target_modules differ from the reference adapter')

    from peft.tuners.lora.layer import LoraLayer
    selected_all = set()
    adapted = []
    for name, module in model.named_modules():
        if not name or any(name.startswith(parent + '.') for parent in adapted):
            continue
        if isinstance(module, LoraLayer):
            adapted.append(name)
        if check_target_module_exists(incoming, name):
            selected_all.add(name)
    if selected_all != set(modules):
        raise ValueError('PEFT adapter target_modules select extra modules')
    safe = directory / 'adapter_model.safetensors'
    binary = directory / 'adapter_model.bin'
    if safe.is_file():
        weights = load_file(str(safe), device='cpu')
    elif binary.is_file():
        weights = torch.load(binary, map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError(f'PEFT adapter weights missing at {directory}')
    if not isinstance(weights, dict):
        raise ValueError('PEFT adapter weights must be a tensor mapping')
    canonical = {}
    for key, value in weights.items():
        key = _canonical_key(key)
        if key in canonical:
            raise ValueError(f'duplicate PEFT tensor keys: {key}')
        canonical[key] = value
    expected = _adapter_parameters(model)
    keys = {_canonical_key(name): name for name in expected}
    if set(canonical) != set(keys):
        raise ValueError(f'PEFT adapter tensor keys differ: missing={sorted(set(keys)-set(canonical))}, unexpected={sorted(set(canonical)-set(keys))}')
    snapshot = {}
    for portable, name in keys.items():
        value, param = canonical[portable], expected[name]
        if (not isinstance(value, torch.Tensor) or value.shape != param.shape
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError(f'PEFT adapter tensor {portable} has invalid shape/dtype/finite values')
        value = value.detach().to(device=device, dtype=torch.float32).clone()
        if '.lora_B.' in name:
            module_name = name.split('.lora_B.')[0]
            scale = model.get_submodule(module_name).scaling['default']
            incoming_scale = incoming.lora_alpha / (math.sqrt(incoming.r) if incoming.use_rslora else incoming.r)
            value.mul_(incoming_scale / scale)
        snapshot[name] = value
    _validate_adapter_snapshot(model, snapshot)
    return snapshot


def setup_lora(model: nn.Module, lora_config: LoRAConfig) -> None:
    """Merge the initial adapter into the backbone and initialize the trainable adapter."""
    from peft import LoraConfig as PeftLoraConfig
    from peft.tuners.tuners_utils import BaseTunerLayer

    model.add_adapter(PeftLoraConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.lora_alpha,
        init_lora_weights=lora_config.init_lora_weights,
        target_modules=lora_config.target_modules,
        use_rslora=False,
    ))
    _load_initial_adapter(model, lora_config.load_path)

    adapters = [module for module in model.modules() if isinstance(module, BaseTunerLayer)]
    for module in adapters:
        module.merge()

    trainable = _adapter_parameters(model)
    for name, param in model.named_parameters():
        param.requires_grad_(name in trainable)
        if name in trainable:
            if 'lora_B' in name:
                nn.init.zeros_(param)
            elif lora_config.init_lora_weights == 'gaussian':
                nn.init.normal_(param)
            else:
                nn.init.kaiming_uniform_(param, a=5 ** 0.5)

    # Zeroing the adapter before unmerging keeps its initial update in the base weights.
    for module in adapters:
        module.unmerge()
    logger.info('Prepared rank %s LoRA in %s layers', lora_config.rank, len(adapters))
