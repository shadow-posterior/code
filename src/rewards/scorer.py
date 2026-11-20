"""Construct and evaluate one reward backend at a time."""

import logging
import warnings
from contextlib import contextmanager
import numpy as np
import torch
from collections import defaultdict

from src.rewards.image_reuse import (reuse_reward_images, uint8_nchw, float_nchw,
                                       numpy_nhwc, pil_images)
from src.utils.fast_init import fast_init
from src.utils.logging import setup_logging


@contextmanager
def _preserve_root_logger_state():
    """Guard root logger against side-effects from reward backend imports/init."""
    root = logging.getLogger()
    level = root.level
    disabled = root.disabled
    propagate = root.propagate
    handlers = list(root.handlers)
    handler_levels = [(handler, handler.level) for handler in handlers]
    try:
        yield
    finally:
        root.setLevel(level)
        root.disabled = disabled
        root.propagate = propagate
        root.handlers[:] = handlers
        for handler, handler_level in handler_levels:
            handler.setLevel(handler_level)


def _pil_list_to_nchw_tensor(images) -> torch.Tensor:
    return uint8_nchw(images)


def _pil_list_to_pil(images) -> list:
    return pil_images(images)


def aesthetic_score(device):
    from src.rewards.aesthetic_scorer import AestheticScorer
    scorer = AestheticScorer(dtype=torch.float32, device=device)
    def _fn(images, prompts, metadata):
        images = _pil_list_to_nchw_tensor(images)
        scores = scorer(images)
        return scores, {}
    return _fn, scorer


def deqa_score(device):
    from src.rewards.deqa_scorer import DeQAScorer
    scorer = DeQAScorer(device=device)
    def _fn(images, prompts, metadata):
        return scorer(images), {}
    return _fn, scorer


def hpsv2_score(device):
    from src.rewards.hpsv2_scorer import HPSv2Scorer
    scorer = HPSv2Scorer(dtype=torch.float32, device=device)
    def _fn(images, prompts, metadata):
        images = float_nchw(images)
        scores = scorer(images, prompts)
        return scores, {}
    return _fn, scorer


def pickscore_score(device):
    from src.rewards.pickscore_scorer import PickScoreScorer
    scorer = PickScoreScorer(dtype=torch.float32, device=device)
    def _fn(images, prompts, metadata):
        images = _pil_list_to_pil(images)
        scores = scorer(prompts, images)
        return scores, {}
    return _fn, scorer


def imagereward_score(device):
    from src.rewards.imagereward_scorer import ImageRewardScorer
    scorer = ImageRewardScorer(dtype=torch.float32, device=device)
    def _fn(images, prompts, metadata):
        images = _pil_list_to_pil(images)
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}
    return _fn, scorer


class GenevalScorerAdapter(torch.nn.Module):
    """Device-aware adapter for GenEval function backend.

    Move the detector, CLIP and cached color classifiers together.
    """

    def __init__(self, device, batch_size: int = 64, *, prefer_cuda_for_cpu_init: bool = True):
        super().__init__()
        self.batch_size = int(batch_size)
        self._requested_device = torch.device(device)
        self._runtime_device = (self._resolve_initial_device(self._requested_device)
                                if prefer_cuda_for_cpu_init else self._requested_device)
        self._force_cpu = self._runtime_device.type == "cpu"
        self._compute_geneval = None
        self._load_backend(self._runtime_device)

    @staticmethod
    def _resolve_initial_device(device: torch.device) -> torch.device:
        if device.type == "cpu" and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return device

    def _load_backend(self, device: torch.device) -> None:
        from src.rewards.geneval_scorer import load_geneval

        self._runtime_device = device
        self._compute_geneval = load_geneval(device)

    def _parse_to_device(self, args, kwargs) -> torch.device | None:
        target = kwargs.get("device")
        if target is None and args:
            first = args[0]
            if isinstance(first, (str, int, torch.device)):
                target = first
        if target is None:
            return None
        if isinstance(target, int):
            return torch.device("cuda", target)
        return torch.device(target)

    def _ensure_backend_loaded(self) -> None:
        if self._compute_geneval is not None:
            return
        load_device = self._runtime_device
        if (
            load_device.type == "cpu"
            and torch.cuda.is_available()
            and not self._force_cpu
        ):
            load_device = torch.device("cuda", torch.cuda.current_device())
        self._load_backend(load_device)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        target_device = self._parse_to_device(args, kwargs)
        if target_device is None:
            return self

        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        self._force_cpu = target_device.type == "cpu"
        if self._compute_geneval is None:
            self._load_backend(target_device)
        elif self._runtime_device != target_device:
            self._compute_geneval.to(target_device)
            self._runtime_device = target_device
        return self

    def __call__(self, images, prompts, metadatas, only_strict):
        del prompts
        self._ensure_backend_loaded()
        images = pil_images(images)
        if metadatas is None or not isinstance(metadatas, list) or len(metadatas) == 0:
            raise ValueError(
                "geneval scorer requires non-empty metadata list aligned with images. "
                "Please evaluate geneval on dataset metadata jsonl (e.g., test_metadata.jsonl)."
            )
        if len(images) != len(metadatas):
            raise ValueError(
                f"geneval scorer image/metadata size mismatch: "
                f"num_images={len(images)}, num_metadatas={len(metadatas)}"
            )
        num_sections = max(1, int(np.ceil(len(images) / self.batch_size)))
        images_batched = [[images[i] for i in indices]
                          for indices in np.array_split(np.arange(len(images)), num_sections)]
        metadatas_batched = np.array_split(np.array(metadatas, dtype=object), num_sections)
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            data = {
                "images": image_batch,
                "metadatas": list(metadata_batched),
                "only_strict": only_strict,
            }
            scores, rewards, strict_rewards, group_rewards, group_strict_rewards = (
                self._compute_geneval(**data)
            )
            all_scores += scores
            all_rewards += rewards
            all_strict_rewards += strict_rewards
            all_group_strict_rewards.append(group_strict_rewards)
            all_group_rewards.append(group_rewards)
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)
        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)
        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict


def geneval_score(device):
    adapter = GenevalScorerAdapter(device=device, batch_size=64)
    return adapter, adapter


def ocr_score(device):
    from src.rewards.ocr_scorer import OcrScorer
    scorer = OcrScorer()
    def _fn(images, prompts, metadata):
        images = numpy_nhwc(images)
        scores = scorer(images, prompts)
        return scores, {}
    return _fn, scorer


_REWARD_FACTORIES = {
    "aesthetic": aesthetic_score,
    "deqa": deqa_score,
    "geneval": geneval_score,
    "hpsv2": hpsv2_score,
    "imagereward": imagereward_score,
    "ocr": ocr_score,
    "pickscore": pickscore_score,
}


class RewardScorer:
    """Load a single reward and move its model between devices as needed."""

    def __init__(self, name: str, device, *, allow_unavailable: bool = False):
        if name not in _REWARD_FACTORIES:
            raise KeyError(f"Unsupported reward key {name!r}. Supported keys: {list(_REWARD_FACTORIES)}")
        self.name = name
        self._score = None
        self._model = None
        self.error = None
        try:
            with fast_init(torch.device("cpu")), _preserve_root_logger_state():
                self._score, self._model = _REWARD_FACTORIES[name](device)
        except Exception as error:
            if not allow_unavailable:
                raise
            self.error = f"{type(error).__name__}: {error}"
            warnings.warn(f"Reward {name!r} unavailable: {self.error}", stacklevel=2)
        finally:
            setup_logging()

    def to(self, device):
        if isinstance(self._model, torch.nn.Module):
            self._model.to(device)
            if hasattr(self._model, "device"):
                self._model.device = device
        return self

    def freeze(self):
        if isinstance(self._model, torch.nn.Module):
            self._model.requires_grad_(False).eval()
        return self

    def __call__(self, images, prompts, metadata=None, *, only_strict=True):
        if self._score is None:
            return {}
        if metadata is None:
            metadata = {}
        with reuse_reward_images(images):
            if self.name != "geneval":
                scores, _ = self._score(images, prompts, metadata)
                return {self.name: scores}
            scores, rewards, strict, groups, strict_groups = self._score(
                images, prompts, metadata, only_strict,
            )
        return {
            "accuracy": rewards,
            "strict_accuracy": strict,
            **{f"{key}_strict_accuracy": values for key, values in strict_groups.items()},
            **{f"{key}_accuracy": values for key, values in groups.items()},
            self.name: scores,
        }
