"""Exact image representations shared by rewards within one scoring call."""
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
from PIL import Image
import torch


_prepared = ContextVar('reward_prepared_images', default=None)


@contextmanager
def reuse_reward_images(images):
    current = _prepared.get()
    if current is not None and current[0] is images:
        yield
        return
    token = _prepared.set((images, {}))
    try:
        yield
    finally:
        _prepared.reset(token)


def _cached(images, key, prepare):
    context = _prepared.get()
    if context is None or context[0] is not images:
        return prepare()
    cache = context[1]
    if key not in cache:
        cache[key] = prepare()
    return cache[key]


def uint8_nchw(images):
    def prepare():
        if isinstance(images, torch.Tensor):
            return (images * 255).round().clamp(0, 255).to(torch.uint8)
        arrays = images if isinstance(images, np.ndarray) else np.stack([np.array(x) for x in images])
        return torch.tensor(arrays.transpose(0, 3, 1, 2), dtype=torch.uint8)
    return _cached(images, 'uint8_nchw', prepare)


def float_nchw(images):
    return _cached(images, 'float_nchw', lambda: uint8_nchw(images).float() / 255.0)


def numpy_nhwc(images):
    def prepare():
        if isinstance(images, torch.Tensor):
            return uint8_nchw(images).cpu().numpy().transpose(0, 2, 3, 1)
        if isinstance(images, np.ndarray):
            return images
        return np.stack([np.array(x) for x in images])
    return _cached(images, 'numpy_nhwc', prepare)


def pil_images(images):
    def prepare():
        if isinstance(images, list) and images and isinstance(images[0], Image.Image):
            return images
        return [Image.fromarray(x) for x in numpy_nhwc(images)]
    return _cached(images, 'pil', prepare)
