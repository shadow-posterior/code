"""Load DeQA's local model code without patching the training process's Llama.

The pinned upstream snapshot targets Transformers 4.36. Its module import calls
replace_llama_modality_adaptive(), changing global Transformers classes. Build
the DeQA decoder in a private package instead, preserving checkpoint keys.
"""
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import torch


class Llama2RotaryEmbedding(torch.nn.Module):
    """Original Llama2 RoPE tables; DeQA Mix3 uses no rope scaling."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000.):
        super().__init__()
        self.dim, self.base = dim, base

    def forward(self, values, seq_len):
        inv = 1. / (self.base ** (torch.arange(0, self.dim, 2,
                                            device=values.device).float() / self.dim))
        angles = torch.outer(torch.arange(seq_len, device=values.device).float(), inv)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(values.dtype), angles.sin().to(values.dtype)


def _apply_rotary(query, key, cos, sin, position_ids):
    def rotate_half(value):
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)
    cos, sin = cos[position_ids].unsqueeze(1), sin[position_ids].unsqueeze(1)
    return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


@lru_cache(maxsize=2)
def local_deqa_classes(directory: str):
    """Import the verified local source into an isolated Python package."""
    from transformers.models.llama import modeling_llama as llama
    from transformers import LlamaTokenizer
    from transformers.cache_utils import Cache
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    from transformers.utils import logging

    root = Path(directory)
    package = '_local_deqa_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    parent = types.ModuleType(package)
    parent.__path__ = [str(root)]
    sys.modules[package] = parent

    def load(name, namespace=None):
        filename = root / f'{name}.py'
        spec = importlib.util.spec_from_file_location(f'{package}.{name}', filename)
        module = importlib.util.module_from_spec(spec)
        if namespace:
            module.__dict__.update(namespace)
        source = filename.read_text()
        source = source.replace('from icecream import ic', '')
        source = source.replace('sys.path.insert(0, dir_path)', '')
        if name == 'visual_encoder':
            source = source.replace('find_pruneable_heads_and_indices, ', '')
        if name == 'modeling_mplug_owl2_huggingface':
            source = source.replace('LlamaConfig, LlamaModel, LlamaForCausalLM',
                                    'LlamaConfig, LlamaForCausalLM')
            source = source.replace('\nreplace_llama_modality_adaptive()\n', '\n')
            source = source.replace('AutoConfig.register("mplug_owl2", MPLUGOwl2Config)', '')
            source = source.replace('AutoModelForCausalLM.register(MPLUGOwl2Config, MPLUGOwl2LlamaForCausalLM)', '')
            source = source.replace('AutoTokenizer.from_pretrained(config._name_or_path)',
                "DeQATokenizer.from_pretrained(config._name_or_path, local_files_only=True)")
        sys.modules[spec.name] = module
        try:
            exec(compile(source, str(filename), 'exec'), module.__dict__)
        except Exception:
            sys.modules.pop(spec.name, None)
            raise
        return module

    config_module = load('configuration_mplug_owl2')
    load('modeling_attn_mask_utils')
    vision = load('visual_encoder')
    if not hasattr(vision.MplugOwlVisualAbstractorModel, 'get_head_mask'):
        def get_head_mask(model, mask, num_hidden_layers):
            if mask is None:
                return [None] * num_hidden_layers
            if mask.ndim == 1:
                mask = mask[None, None, :, None, None].expand(num_hidden_layers, -1, -1, -1, -1)
            elif mask.ndim == 2:
                mask = mask[:, None, :, None, None]
            if mask.ndim != 5:
                raise ValueError('DeQA head mask must have one, two, or five dimensions')
            return mask.to(dtype=model.dtype)
        vision.MplugOwlVisualAbstractorModel.get_head_mask = get_head_mask
    namespace = {name: getattr(llama, name) for name in dir(llama) if not name.startswith('_')}
    namespace.update(Cache=Cache, BaseModelOutputWithPast=BaseModelOutputWithPast,
                     CausalLMOutputWithPast=CausalLMOutputWithPast,
                     logger=logging.get_logger(__name__))
    decoder = load('modeling_llama2', namespace)
    decoder.LlamaRotaryEmbedding = Llama2RotaryEmbedding
    decoder.apply_rotary_pos_emb = _apply_rotary
    def init_rope(attention):
        scaling = attention.config.rope_scaling
        if scaling and scaling.get('rope_type', scaling.get('type')) != 'default':
            raise ValueError('DeQA Mix3 requires the original unscaled Llama2 RoPE')
        attention.rotary_emb = Llama2RotaryEmbedding(attention.head_dim, base=attention.rope_theta)
    decoder.LlamaAttention._init_rope = init_rope

    class DeQALlamaModel(llama.LlamaPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.padding_idx, self.vocab_size = config.pad_token_id, config.vocab_size
            self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.layers = torch.nn.ModuleList([
                decoder.LlamaDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
            self.norm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.gradient_checkpointing = False
            self._use_sdpa = self._use_flash_attention_2 = False
            self.post_init()

        forward = decoder.model_forward

    model_module = load('modeling_mplug_owl2_huggingface', {
        'LlamaModel': DeQALlamaModel, 'DeQATokenizer': LlamaTokenizer})
    return config_module.MPLUGOwl2Config, model_module.MPLUGOwl2LlamaForCausalLM


def load_local_deqa(path):
    """Load frozen FP16 weights strictly from the configured checkout."""
    path = Path(path).resolve()
    if not (path / 'model.safetensors.index.json').is_file():
        raise FileNotFoundError(f'Missing local DeQA weights: {path}. Place the DeQA model in the configured reward weights directory.')
    config_class, model_class = local_deqa_classes(str(path))
    config = config_class.from_pretrained(str(path), local_files_only=True)
    config.mlp_bias = False
    config._name_or_path = str(path)
    config.use_cache = False
    config._attn_implementation = 'eager'
    return model_class.from_pretrained(
        str(path), config=config, torch_dtype=torch.float16, attn_implementation='eager',
        local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True,
    ).eval().requires_grad_(False)
