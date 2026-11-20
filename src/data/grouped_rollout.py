"""Grouped prompt sampling and reward advantages for SPM rollouts."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


def weighted_reward_advantages(prompts, rewards, reward_weights, *, global_std=True):
    """Standardize each reward within prompt groups, then combine the rewards."""
    prompts = np.asarray(prompts)
    groups = [prompts == prompt for prompt in np.unique(prompts)]
    combined = np.zeros(len(prompts), dtype=np.float64)
    for name, weight in reward_weights.items():
        values = np.asarray(rewards[name], dtype=np.float64)
        batch_std = values.std() if global_std else None
        for group in groups:
            group_values = values[group]
            std = batch_std if global_std else group_values.std()
            combined[group] += weight * ((group_values - group_values.mean()) / (std + 1e-4))
    return (combined - combined.mean()) / (combined.std() + 1e-4)


class DistributedKRepeatSampler(Sampler):
    """Distributed sampler ensuring each prompt appears exactly K times globally.

    For per-prompt stat tracking, we need K samples per prompt to compute
    meaningful per-prompt advantages. This sampler:
    1. Selects M = total_samples // K unique prompts
    2. Repeats each prompt K times -> M*K = total_samples indices
    3. Shuffles and shards across ranks so each rank gets batch_size indices

    Constraints (validated at init):
    - total_samples = num_replicas * batch_size
    - total_samples % K == 0  (so M is integer)
    - batch_size * num_replicas % K == 0

    This is an infinite sampler: each `__iter__` yields one batch of indices
    per rank. Call `set_epoch(epoch)` before each iteration for determinism.

    Used with DataLoader's `batch_sampler` param and `num_workers=0` to avoid
    prefetching issues with resume.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        k: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        sample_weights=None,
        strata=None,
    ):
        """
        Args:
            dataset_size: Total number of prompts in the dataset.
            batch_size: Per-GPU batch size for sampling.
            k: Number of images per prompt (K-repeat count).
            num_replicas: World size (number of GPUs).
            rank: This process's rank.
            seed: Base random seed for reproducibility.
            strata: Optional per-row labels (e.g. dataset_source). Each draw
                then takes exactly M / num_strata prompts from every label, and
                every rank gets an equal share (up to one) of each label's images.
        """
        for name, value in (('dataset_size', dataset_size), ('batch_size', batch_size),
                            ('k', k), ('num_replicas', num_replicas)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(rank) is not int or not 0 <= rank < num_replicas:
            raise ValueError('rank must be an integer in [0, num_replicas)')
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        if self.total_samples % self.k:
            raise ValueError(
                f"total_samples ({self.total_samples} = {num_replicas} * {batch_size}) "
                f"must be divisible by k ({k})"
            )
        self.m = self.total_samples // self.k
        if self.dataset_size < self.m:
            raise ValueError('dataset_size must provide enough distinct prompts for one complete global batch')
        self.sample_weights = None
        if sample_weights is not None:
            weights = torch.as_tensor(sample_weights, dtype=torch.float64)
            if (weights.shape != (dataset_size,) or not torch.isfinite(weights).all()
                    or (weights < 0).any() or int((weights > 0).sum()) < self.m):
                raise ValueError('Invalid sample_weights or too few positive-weight prompts')
            self.sample_weights = weights
        self.strata = None
        if strata is not None:
            if sample_weights is not None:
                raise ValueError('Use either sample_weights or strata, not both')
            labels = list(strata)
            if len(labels) != dataset_size or any(not isinstance(s, str) or not s for s in labels):
                raise ValueError('strata needs one nonempty label per dataset row')
            groups = {}
            for index, label in enumerate(labels):
                groups.setdefault(label, []).append(index)
            if self.m % len(groups):
                raise ValueError(f'{self.m} prompts per draw cannot be split evenly over {len(groups)} strata')
            self.per_stratum = self.m // len(groups)
            if any(len(rows) < self.per_stratum for rows in groups.values()):
                raise ValueError('Every stratum needs at least M / num_strata rows')
            self.strata = [torch.tensor(groups[label]) for label in sorted(groups)]
        self.epoch = 0

    def __iter__(self):
        """Yield one batch of indices for this rank (infinite generator)."""
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            if self.strata is not None:
                indices = [int(rows[i]) for rows in self.strata
                           for i in torch.randperm(len(rows), generator=g)[: self.per_stratum].tolist()]
            elif self.sample_weights is None:
                indices = torch.randperm(self.dataset_size, generator=g)[: self.m].tolist()
            else:
                indices = torch.multinomial(self.sample_weights, self.m, replacement=False, generator=g).tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            if self.strata is not None:
                mine = repeated_indices[self.rank::self.num_replicas]
                order = torch.randperm(len(mine), generator=g).tolist()
                yield [mine[i] for i in order]
                continue
            shuffled_order = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_order]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling."""
        self.epoch = epoch
