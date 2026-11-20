"""Compose named reward backends and their weighted per-sample scores."""

from collections.abc import Iterable

import torch

from src.rewards.image_reuse import reuse_reward_images
from src.rewards.scorer import RewardScorer


class MultiScorer:
    """Share one loaded backend per reward across scoring calls."""

    def __init__(self, device, score_dict, allow_unavailable: bool = False):
        self.score_dict = {}
        self.unavailable_rewards = {}
        self._backends = {}
        for name, weight in score_dict.items():
            backend = RewardScorer(name, device, allow_unavailable=allow_unavailable)
            if backend.error is not None:
                self.unavailable_rewards[name] = backend.error
                continue
            self.score_dict[name] = weight
            self._backends[name] = backend

    @property
    def active_reward_names(self) -> list[str]:
        return list(self.score_dict)

    def to(self, device):
        for backend in self._backends.values():
            backend.to(device)
        return self

    def freeze(self):
        for backend in self._backends.values():
            backend.freeze()
        return self

    def score_details(self, images, prompts, metadata=None, *,
                      names: Iterable[str] | None = None, only_strict: bool = True):
        """Score the requested rewards using shared image preprocessing."""
        details = {}
        with reuse_reward_images(images):
            for name in self.score_dict if names is None else names:
                details.update(self._backends[name](images, prompts, metadata, only_strict=only_strict))
        return details

    def __call__(self, images, prompts, metadata=None, only_strict=True):
        details = self.score_details(images, prompts, metadata, only_strict=only_strict)
        total = None
        for name, weight in self.score_dict.items():
            scores = details[name]
            weighted = scores * weight if isinstance(scores, torch.Tensor) else [weight * score for score in scores]
            if total is None:
                total = weighted
            elif isinstance(total, torch.Tensor):
                total = total + (weighted if isinstance(weighted, torch.Tensor)
                                 else total.new_tensor(weighted))
            elif isinstance(weighted, torch.Tensor):
                total = weighted.new_tensor(total) + weighted
            else:
                total = [left + right for left, right in zip(total, weighted)]
        details["mean"] = [] if total is None else total
        return details, {}
