"""Finite and resumable batches of repeated text prompts."""
import math

import torch
from torch.utils.data import DataLoader, Sampler

from src.data.prompt_dataset import PromptDataset
from src.data.grouped_rollout import DistributedKRepeatSampler


class GroupedPromptSampler(Sampler):
    def __init__(self, dataset_size, batch_size, k, num_replicas, rank, seed=42, strata=None):
        total = batch_size*num_replicas
        if type(k) is not int or k < 2 or total % k:
            raise ValueError('global batch (train.batch_size * world_size) must be divisible by num_image_per_prompt >= 2')
        if dataset_size < total//k:
            raise ValueError('dataset must contain enough distinct rows for one complete global group batch')
        self.repeat = DistributedKRepeatSampler(dataset_size, batch_size, k, num_replicas, rank, seed, strata=strata)
        self.batches = math.ceil(dataset_size/(total//k))
        self.batch_size, self.epoch = batch_size, 0

    def __len__(self):
        return self.batches*self.batch_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        for batch in range(self.batches):
            self.repeat.set_epoch(self.epoch*self.batches+batch)
            yield from next(iter(self.repeat))


def _collate_groups(rows):
    return {'prompts': [row['prompt'] for row in rows], 'metadata': [row['metadata'] for row in rows]}


def create_grouped_dataloader(prompt_path, batch_size, k, seed=42, stratify_sources=False):
    """K-repeat prompt groups; ``stratify_sources`` draws equally from every dataset_source."""
    dataset = PromptDataset(prompt_path)
    distributed = torch.distributed.is_initialized()
    replicas = torch.distributed.get_world_size() if distributed else 1
    rank = torch.distributed.get_rank() if distributed else 0
    strata = None
    if stratify_sources:
        strata = [row.get('dataset_source') for row in dataset.metadatas or []]
        if len(strata) != len(dataset) or any(not isinstance(s, str) or not s for s in strata):
            raise ValueError('Source-stratified sampling requires dataset_source on every prompt row')
    sampler = GroupedPromptSampler(len(dataset), batch_size, k, replicas, rank, seed, strata=strata)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
        generator=torch.Generator().manual_seed(seed), num_workers=0, drop_last=True,
        collate_fn=_collate_groups)
    return loader, sampler
