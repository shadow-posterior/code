"""
Human Preference Score v2 (HPSv2) scorer.

"""

import torch
import torch.nn as nn
from torchvision.transforms import Normalize, Compose, InterpolationMode
import torchvision.transforms.functional as F

from hpsv2.src.open_clip import create_model, get_tokenizer
from src.paths import WEIGHTS_DIR
from src.rewards.text_reuse import cached_text, text_reuse_enabled

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class ResizeMaxSize(nn.Module):
    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn="max", fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == "min" else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=self.fill)
        return img


class MaskAwareNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.normalize = Normalize(mean=mean, std=std)

    def forward(self, tensor):
        if tensor.shape[1] == 4:
            normalized_parts = []
            for i in range(tensor.shape[0]):
                img_slice = tensor[i]
                normalized_rgb = self.normalize(img_slice[:3])
                alpha_channel = img_slice[3:]
                normalized_parts.append(torch.cat([normalized_rgb, alpha_channel], dim=0))
            return torch.stack(normalized_parts, dim=0)
        else:
            return self.normalize(tensor)


def image_transform_tensor(
    image_size: int,
    mean: tuple = None,
    std: tuple = None,
    fill_color: int = 0,
):
    mean = mean or OPENAI_DATASET_MEAN
    std = std or OPENAI_DATASET_STD
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    normalize = MaskAwareNormalize(mean=mean, std=std)
    transforms = [
        ResizeMaxSize(image_size, fill=fill_color),
        normalize,
    ]
    return Compose(transforms)


class HPSv2Scorer(nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device
        model = create_model(
            "ViT-H-14",
            str(WEIGHTS_DIR / "open_clip_pytorch_model.bin"),
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            output_dict=True,
        )
        image_mean = getattr(model.visual, "image_mean", None)
        image_std = getattr(model.visual, "image_std", None)
        image_size = model.visual.image_size
        if isinstance(image_size, tuple):
            image_size = image_size[0]
        preprocess_val = image_transform_tensor(
            image_size,
            mean=image_mean,
            std=image_std,
        )
        self.model = model.to(device)
        self.preprocess_val = preprocess_val
        checkpoint = torch.load(WEIGHTS_DIR / "HPS_v2.1_compressed.pt", map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.processor = get_tokenizer("ViT-H-14")
        self.eval()

    @torch.no_grad()
    def __call__(self, images, prompts):
        if len(prompts) != len(images):
            raise ValueError("Images and prompts must align")
        if not prompts:
            return torch.empty(0, device=self.device, dtype=self.dtype)
        image = self.preprocess_val(images.to(self.dtype).to(device=self.device, non_blocking=True))
        if text_reuse_enabled(self.model):
            image_features = self.model.encode_image(image, normalize=True)
            def encode_text():
                text = self.processor(prompts).to(device=self.device, non_blocking=True)
                return self.model.encode_text(text, normalize=True)
            text_features = cached_text(self.model, (self, tuple(prompts)), encode_text)
        else:
            text = self.processor(prompts).to(device=self.device, non_blocking=True)
            outputs = self.model(image, text)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
        hps_score = (image_features * text_features).sum(dim=-1)
        return hps_score.contiguous()
