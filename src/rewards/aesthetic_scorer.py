"""
Aesthetic score predictor.

Based on https://github.com/christophschuhmann/improved-aesthetic-predictor
"""

import torch
from src.paths import local_model_path
from src.rewards.clip_features import projected_features
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from src.paths import WEIGHTS_DIR


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(local_model_path(WEIGHTS_DIR / "clip-vit-large-patch14"), local_files_only=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(local_model_path(WEIGHTS_DIR / "clip-vit-large-patch14"), local_files_only=True)
        self.mlp = MLP().to(device)
        state_dict = torch.load(WEIGHTS_DIR / "sac+logos+ava1-l14-linearMSE.pth", map_location="cpu")
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.device = device
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(self.device) for k, v in inputs.items()}
        embed = projected_features(self.clip.get_image_features(**inputs, return_dict=True))

        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)
