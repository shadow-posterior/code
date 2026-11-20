"""Isolate model/scorer setup from stochastic training streams."""
from contextlib import contextmanager
import random


@contextmanager
def preserve_rng():
    """Restore Python, NumPy, torch CPU and the current CUDA device, even on error.

    Each distributed process owns its current CUDA device. This deliberately
    avoids touching other visible GPUs; callers must construct scorers on
    their assigned current device.
    """
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
