"""
PickScore v1 text-image alignment scorer.

"""

import torch
from src.paths import WEIGHTS_DIR, local_model_path
from src.rewards.clip_features import projected_features
from src.rewards.text_reuse import cached_text
from transformers import AutoProcessor, AutoModel


class PickScoreScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        processor_path = local_model_path(WEIGHTS_DIR / "CLIP-ViT-H-14-laion2B-s32B-b79K")
        model_path = local_model_path(WEIGHTS_DIR / "PickScore_v1")
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
        self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def __call__(self, prompt, images):
        if len(prompt) != len(images):
            raise ValueError("Images and prompts must align")
        if not prompt:
            return torch.empty(0, device=self.device, dtype=next(self.model.parameters()).dtype)

        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}

        image_embs = projected_features(self.model.get_image_features(**image_inputs, return_dict=True))
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        def encode_text():
            text_inputs = self.processor(text=prompt, padding=True, truncation=True,
                                         max_length=77, return_tensors="pt")
            text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
            features = projected_features(self.model.get_text_features(**text_inputs, return_dict=True))
            return features / features.norm(p=2, dim=-1, keepdim=True)
        text_embs = cached_text(self.model, (self, tuple(prompt)), encode_text)

        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs * image_embs).sum(dim=-1)

        scores = scores / 26
        return scores
