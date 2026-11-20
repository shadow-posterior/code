"""Local DeQA image-quality scoring."""
from io import BytesIO
import os

import torch
from PIL import Image

from src.paths import WEIGHTS_DIR
from src.rewards.image_reuse import pil_images
from src.utils.rng import preserve_rng


def load_deqa_model(path):
    from src.rewards.deqa_model import load_local_deqa
    return load_local_deqa(path)


class DeQAScorer(torch.nn.Module):
    """Lazy, frozen local inference with the same JPEG inputs and /5 scale."""

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.model = None
        self.batch_size = int(os.environ.get('DEQA_BATCH_SIZE', '1'))
        if self.batch_size < 1:
            raise ValueError('DEQA_BATCH_SIZE must be positive')

    def _ensure_model(self):
        if self.model is None:
            path = WEIGHTS_DIR / 'DeQA-Score-Mix3'

            with torch.inference_mode(False), torch.no_grad(), preserve_rng():
                self.model = load_deqa_model(path).eval().requires_grad_(False).to(self.device)

                self.model.register_buffer('weight_tensor', torch.tensor(
                    [5., 4., 3., 2., 1.], device=self.device, dtype=torch.float16),
                    persistent=False)

    def to(self, target_device):
        self.device = torch.device(target_device)
        super().to(self.device)
        return self

    @torch.inference_mode()
    def forward(self, images):
        images = pil_images(images)
        if not images:
            return []
        self._ensure_model()
        scores = []
        for start in range(0, len(images), self.batch_size):
            decoded = []
            for image in images[start:start + self.batch_size]:
                buffer = BytesIO()
                image.convert('RGB').save(buffer, format='JPEG')
                buffer.seek(0)
                decoded.append(Image.open(buffer))
            values = (self.model.score(decoded) / 5).detach().cpu()
            if values.shape != (len(decoded),) or not torch.isfinite(values).all():
                raise ValueError('DeQA must return one finite score per image')
            scores.extend(values.tolist())
        return scores
