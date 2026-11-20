"""
OCR-based text rendering accuracy scorer.

Uses PaddleOCR to recognize text in generated images and compares against
the target text extracted from the prompt (text between quotes).

"""

import os
import re
import time
import fcntl
import logging
from pathlib import Path
from contextlib import contextmanager
import torch
import numpy as np
from typing import List, Union
from PIL import Image
from Levenshtein import distance

from src.paths import WEIGHTS_DIR


class OcrScorer:
    def __init__(self, use_gpu: bool = False, init_retries: int = 3):
        """
        OCR reward calculator.
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        cache_root = WEIGHTS_DIR / 'paddleocr'
        for component in ('det', 'rec', 'cls'):
            for filename in ('inference.pdmodel', 'inference.pdiparams'):
                path = cache_root / component / filename
                if not path.is_file():
                    raise FileNotFoundError(f'Missing PaddleOCR weights: {path}')
        lock_path = os.path.join(cache_root, ".paddleocr_init.lock")
        self.ocr = None

        for attempt_idx in range(init_retries):
            try:
                self.ocr = _init_paddleocr(cache_root=cache_root, use_gpu=use_gpu)
                break
            except FileNotFoundError as e:
                if ".tmp" in str(e) and attempt_idx + 1 < init_retries:
                    with _file_lock(lock_path):
                        time.sleep(0.5)
                    time.sleep(1.5 * (attempt_idx + 1))
                    continue
                raise

        if self.ocr is None:
            raise RuntimeError("Failed to initialize PaddleOCR scorer")

    @torch.no_grad()
    def __call__(self, images: Union[List[Image.Image], List[np.ndarray]], prompts: List[str]) -> list:
        """
        Calculate OCR reward.
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: List of reward scores
        """
        targets = []
        for prompt in prompts:
            match = re.search(r'"([^"]*)"', prompt)
            if match is None or not match[1].strip():
                raise ValueError('OCR prompts must include nonempty target text in double quotes')
            targets.append(match[1].replace(" ", "").lower())
        rewards = []
        if len(images) != len(targets):
            raise ValueError('Images and prompts must align')
        for img, target in zip(images, targets):
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                result = self.ocr.ocr(img, cls=False)
                recognized_text = (
                    "".join([res[1][0] if res[1][1] > 0 else "" for res in result[0]]) if result[0] else ""
                )
                recognized_text = recognized_text.replace(" ", "").lower()
                dist = 0 if target in recognized_text else distance(recognized_text, target)
            except Exception as e:
                print(f"OCR processing failed: {str(e)}")
                dist = len(target)
            reward = 1 - min(dist, len(target)) / len(target)
            rewards.append(reward)

        return rewards


def _init_paddleocr(cache_root: Path, use_gpu: bool):
    with _preserve_root_logger_state():
        from paddleocr import PaddleOCR

        import paddleocr.paddleocr as paddleocr_module

        paddleocr_module.BASE_DIR = os.path.join(cache_root, "")
        return PaddleOCR(
            use_angle_cls=False, lang="en", use_gpu=use_gpu, show_log=False,
            ocr_version="PP-OCRv4",
            det_model_dir=os.path.join(cache_root, "det"),
            rec_model_dir=os.path.join(cache_root, "rec"),
            cls_model_dir=os.path.join(cache_root, "cls"),
        )


@contextmanager
def _preserve_root_logger_state():
    root = logging.getLogger()
    level = root.level
    disabled = root.disabled
    handler_levels = [(handler, handler.level) for handler in list(root.handlers)]
    try:
        yield
    finally:
        root.setLevel(level)
        root.disabled = disabled
        for handler, handler_level in handler_levels:
            handler.setLevel(handler_level)


@contextmanager
def _file_lock(lock_path: str):
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
