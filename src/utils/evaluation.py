"""Bounded reward evaluation over images on their generation device."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def score_details_to_metric_stats(
    score_details: dict[str, Any],
) -> tuple[dict[str, float], dict[str, int]]:
    """Reduce finite scores, counting only applicable samples for each metric."""
    metrics: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, scores in score_details.items():
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu()
            if scores.dtype == torch.bfloat16:
                scores = scores.float()
            scores = scores.numpy()
        elif isinstance(scores, list):
            scores = np.array([float(score) for score in scores])
        else:
            scores = np.asarray(scores)
        mask = np.isfinite(scores)
        if name != 'geneval':
            mask &= scores != -10
        valid = scores[mask]
        if valid.size:
            metrics[name] = float(np.mean(valid))
            counts[name] = int(valid.size)
    return metrics, counts


def reward_evaluation_routes(eval_cfg) -> list[tuple[str | None, str, float, str]]:
    """Return (dataset override, reward, weight, metric prefix) for each panel."""
    mapping = getattr(eval_cfg, 'reward_dataset_map', {}) or {}
    if not isinstance(mapping, dict):
        raise TypeError('eval.reward_dataset_map must be a dict')
    routes = []
    for name, weight in eval_cfg.reward_fn.items():
        dataset = mapping.get(name)
        dataset = dataset.strip() if isinstance(dataset, str) and dataset.strip() else None
        routes.append((dataset, name, weight, ''))
    benchmarks = getattr(eval_cfg, 'benchmark_reward_fn', {}) or {}
    if not isinstance(benchmarks, dict):
        raise TypeError('eval.benchmark_reward_fn must be a dict')
    for dataset, rewards in benchmarks.items():
        if not isinstance(dataset, str) or not dataset.strip() or dataset != dataset.strip():
            raise ValueError('eval.benchmark_reward_fn requires nonempty dataset names')
        if not isinstance(rewards, dict) or not rewards:
            raise ValueError(f'eval.benchmark_reward_fn[{dataset!r}] requires a nonempty reward dict')
        for name, weight in rewards.items():
            if not isinstance(weight, (int, float)) or not np.isfinite(weight):
                raise ValueError(f'Invalid benchmark reward weight for {name!r}')
            routes.append((dataset, name, weight, f'{dataset}/'))
    return routes


def _cpu_score_vector(values: Any, count: int) -> list:
    """Retain scalar dtypes while releasing a scorer's device-side outputs."""
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or len(values) != count:
            raise ValueError(f'Eval scorer must return {count} per-sample scores')
        return list(values.detach().cpu().unbind())
    if isinstance(values, np.ndarray):
        if values.shape != (count,):
            raise ValueError(f'Eval scorer must return {count} per-sample scores')
        return list(values)
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f'Eval scorer must return {count} per-sample scores')
    if values and all(isinstance(value, torch.Tensor) for value in values):
        return list(torch.stack(list(values)).detach().cpu().unbind())
    return list(values)


def score_eval_images(
    scorer,
    images: torch.Tensor | None,
    prompts: list[str],
    *,
    metadata: list[dict[str, Any]] | None,
    batch_size: int,
    device: torch.device | str,
) -> dict[str, Any]:
    """Score one reward in bounded batches and collect its per-sample details."""
    if images is None or not len(images):
        return {}
    if batch_size <= 0:
        raise ValueError('Evaluation batch_size must be positive')
    if len(images) != len(prompts):
        raise ValueError('Evaluation images and prompts must be aligned')
    if metadata is not None and len(metadata) != len(prompts):
        raise ValueError('Evaluation metadata and prompts must be aligned')

    details = {}
    formats = {}
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        batch_metadata = metadata[start:end] if metadata is not None else None
        batch_details = scorer(
            images[start:end].to(device), prompts[start:end], metadata=batch_metadata, only_strict=False,
        )
        for name, values in batch_details.items():
            if name not in formats:
                formats[name] = (type(values), getattr(values, 'dtype', None))
            details.setdefault(name, []).extend(_cpu_score_vector(values, end - start))
    for name, values in details.items():
        if len(values) != len(prompts):
            raise ValueError(f'Incomplete evaluation score vector for {name!r}')
        kind, dtype = formats[name]
        if issubclass(kind, torch.Tensor):
            details[name] = torch.stack(values)
        elif issubclass(kind, np.ndarray):
            details[name] = np.asarray(values, dtype=dtype)
    return details
