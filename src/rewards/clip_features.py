"""Extract projected CLIP features as tensors."""
import torch


def projected_features(output):
    if isinstance(output, torch.Tensor):
        return output

    pooled = getattr(output, 'pooler_output', None)
    if not isinstance(pooled, torch.Tensor):
        raise TypeError('CLIP features require a tensor or a projected pooler_output')
    return pooled
