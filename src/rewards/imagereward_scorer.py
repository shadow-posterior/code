"""
ImageReward v1.0 scorer.

"""

import importlib
import os
import torch
from PIL import Image
from src.rewards.text_reuse import cached_text
from src.paths import WEIGHTS_DIR, local_model_path


def _load_imagereward_module():
    """Load ImageReward with compatibility patch for new transformers versions."""
    try:
        from transformers import modeling_utils as hf_modeling_utils
    except Exception:
        hf_modeling_utils = None

    if hf_modeling_utils is not None:
        try:
            from transformers import pytorch_utils as hf_pytorch_utils
        except Exception:
            hf_pytorch_utils = None
        if hf_pytorch_utils is not None:
            for symbol_name in (
                "apply_chunking_to_forward",
                "find_pruneable_heads_and_indices",
                "prune_linear_layer",
            ):
                if not hasattr(hf_modeling_utils, symbol_name) and hasattr(hf_pytorch_utils, symbol_name):
                    setattr(hf_modeling_utils, symbol_name, getattr(hf_pytorch_utils, symbol_name))

    module = importlib.import_module("ImageReward")
    blip = importlib.import_module("ImageReward.models.BLIP.blip")
    pretrain = importlib.import_module("ImageReward.models.BLIP.blip_pretrain")
    blip.init_tokenizer = _local_blip_tokenizer
    pretrain.init_tokenizer = _local_blip_tokenizer
    return module


def _local_blip_tokenizer():
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        local_model_path(WEIGHTS_DIR / "bert-base-uncased"), local_files_only=True)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
    return tokenizer


class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32, batch_size=8):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("ImageReward batch_size must be positive")
        self.batch_size = batch_size
        rm_module = _load_imagereward_module()
        self.model = (
            rm_module.load(
                local_model_path(WEIGHTS_DIR / "ImageReward/ImageReward.pt"),
                med_config=local_model_path(WEIGHTS_DIR / "ImageReward/med_config.json"),
                device=str(self.device),

                download_root=str(WEIGHTS_DIR / "ImageReward"),
            )
            .eval()
            .to(dtype=dtype)
        )
        self.model.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        if device is None and args:
            first = args[0]
            if isinstance(first, (str, int, torch.device)):
                device = first
        if device is not None:
            self.device = torch.device(device)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "device"):
                self.model.device = str(self.device)
        return self

    def _supports_paired_scoring(self):
        blip = getattr(self.model, "blip", None)
        return (
            not self.model.training
            and callable(getattr(self.model, "preprocess", None))
            and callable(getattr(self.model, "mlp", None))
            and hasattr(self.model, "mean") and hasattr(self.model, "std")
            and all(callable(getattr(blip, name, None))
                    for name in ("tokenizer", "visual_encoder", "text_encoder"))
        )

    @torch.no_grad()
    def __call__(self, prompts, images):
        if len(prompts) != len(images):
            raise ValueError("Images and prompts must align")
        if not prompts:
            return torch.empty(0, device=self.device)
        if hasattr(self.model, "device"):
            self.model.device = str(self.device)
        if not self._supports_paired_scoring():
            _, rewards = self.model.inference_rank(prompts, images)
            return torch.as_tensor(rewards, device=self.device).reshape(
                len(prompts), len(prompts)).diagonal().contiguous()

        blip = self.model.blip
        def tokenize():
            text = blip.tokenizer(prompts, padding="max_length", truncation=True,
                                  max_length=35, return_tensors="pt").to(self.device)

            return torch.stack((text.input_ids, text.attention_mask))
        tokens = cached_text(self.model, (self, "tokens", tuple(prompts)), tokenize)
        scores = []
        for start in range(0, len(prompts), self.batch_size):
            end = min(start + self.batch_size, len(prompts))
            pixels = []
            for image in images[start:end]:
                if isinstance(image, Image.Image):
                    pixels.append(self.model.preprocess(image))
                elif isinstance(image, str) and os.path.isfile(image):
                    with Image.open(image) as pil_image:
                        pixels.append(self.model.preprocess(pil_image))
                else:
                    raise TypeError("ImageReward requires PIL images or image file paths")

            visual = blip.visual_encoder(torch.stack(pixels).to(self.device))
            image_atts = torch.ones(visual.shape[:-1], dtype=torch.long, device=self.device)
            text = blip.text_encoder(
                tokens[0, start:end], attention_mask=tokens[1, start:end],
                encoder_hidden_states=visual, encoder_attention_mask=image_atts,
                return_dict=True,
            )
            rewards = self.model.mlp(text.last_hidden_state[:, 0, :].float())
            rewards = (rewards - self.model.mean) / self.model.std

            scores.append(rewards.reshape(-1).to(dtype=torch.get_default_dtype()))
        return torch.cat(scores).contiguous()
