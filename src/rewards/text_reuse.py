"""Opt-in, one-update reuse of frozen reward text features (never image scores).

The caller keeps reward weights/tokenization fixed for this scope. Nothing is
stored across updates or checkpoints. Calls outside the scope keep their
uncached path; grad-enabled or training-mode models bypass the cache. Device
and precision are part of each entry, and nested scopes share their outer cache.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch


_text_cache = ContextVar('reward_text_cache', default=None)


@contextmanager
def reuse_reward_text():
    if _text_cache.get() is not None:
        yield
        return
    token = _text_cache.set({})
    try:
        yield
    finally:
        _text_cache.reset(token)


def text_reuse_enabled(model):
    return _text_cache.get() is not None and not torch.is_grad_enabled() and not model.training


def cached_text(model, key, compute):
    if not text_reuse_enabled(model):
        return compute()
    parameter = next(model.parameters())
    device_type = parameter.device.type
    precision = (torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type))
    identity = (model, key, parameter.device, parameter.dtype, precision)
    cache = _text_cache.get()
    if identity not in cache:
        cache[identity] = compute().detach()
    return cache[identity]
