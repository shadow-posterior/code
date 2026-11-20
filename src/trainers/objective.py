"""Detached SPM labels with RMS and MAE normalization."""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
from src.utils.scalars import TensorChecks


def _work(value):
    tensor = torch.as_tensor(value).detach()
    if tensor.ndim < 2:
        raise ValueError('SPM fields must have separate batch and latent axes')
    return tensor.to(dtype=torch.float64 if tensor.dtype == torch.float64 else torch.float32)


def _positive_scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _sample_value(value, reference, name, checks):
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype).detach()
    if result.ndim == 1 and result.numel() in (1, reference.shape[0]):
        result = result.reshape((-1,) + (1,)*(reference.ndim-1))
    if result.ndim and (result.ndim != reference.ndim or result.shape[0] not in (1, reference.shape[0])
                       or any(size != 1 for size in result.shape[1:])):
        raise ValueError(f'{name} must be scalar or contain one scalar per sample')
    checks.require(torch.isfinite(result).all(), f'{name} must be finite')
    return result


def _times(student_sigma, query_sigma, reference, checks):
    s = _sample_value(student_sigma, reference, 'Student sigma', checks)
    tau = _sample_value(query_sigma, reference, 'KL query sigma', checks)
    checks.require(((s > 0) & (s <= 1)).all(), 'Student sigma must be in (0, 1]')
    checks.require(((tau > 0) & (tau < 1)).all(), 'KL query sigma must be strictly between zero and one')
    return s, tau


def _field(value, reference):
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype).detach()
    if value.shape != reference.shape:
        raise ValueError('SPM velocity fields must have the same shape')
    return value


MAE_NORMALIZER_FLOOR = 1e-8
RMS_NORMALIZER_FLOOR = 1e-8


def _velocity_mae(teacher_output, query_target, query_sigma, checks):
    """Convert the clean-space MAE floor to velocity units by dividing by sigma."""
    target = _work(query_target)
    teacher = _field(teacher_output, target)
    tau = _sample_value(query_sigma, target, 'KL query sigma', checks)
    checks.require(((tau > 0) & (tau < 1)).all(), 'KL query sigma must be strictly between zero and one')
    mae = (teacher-target).abs().mean(dim=tuple(range(1, target.ndim)), keepdim=True)
    checks.require(torch.isfinite(mae).all(), 'Velocity MAE is not finite', FloatingPointError)
    return torch.maximum(mae, MAE_NORMALIZER_FLOOR/tau)


@torch.no_grad()
def matching_force(f0_input, f1_output, teacher_output, u, advantage, *, alpha,
              student_sigma, query_sigma, gamma, query_target, f0_output=None):
    """Return the total, reward, teacher and consistency terms in velocity units.

    The reward uses inverse per-sample residual mean square. Both output
    terms use s divided by the teacher's velocity MAE against query_target.
    All labels are detached.
    """
    alpha = _positive_scalar(alpha, 'alpha')
    u = _work(u)
    f0, f1, teacher = (_field(field, u) for field in (f0_input, f1_output, teacher_output))
    checks = TensorChecks()
    s, tau = _times(student_sigma, query_sigma, u, checks)
    advantage = _sample_value(advantage, u, 'advantage', checks)
    gamma = _sample_value(gamma, u, 'gamma', checks)
    checks.require((gamma >= 0).all(), 'gamma must be nonnegative')
    residual = u-f0
    mean_square = residual.square().mean(dim=tuple(range(1, u.ndim)), keepdim=True)
    checks.require(torch.isfinite(mean_square).all(), 'SPM RMS normalizer is not finite', FloatingPointError)
    precision = mean_square.clamp_min(RMS_NORMALIZER_FLOOR).reciprocal()
    reward = precision*advantage*residual
    coefficient = s/_velocity_mae(teacher, query_target, tau, checks)
    teacher_force = coefficient*alpha*(teacher-f1)
    if f0_output is None:
        checks.require((gamma == 0).all(), 'positive gamma requires F0 at the output query')
        consistency = torch.zeros_like(u)
    else:
        enabled = (gamma > 0).any()
        output = torch.as_tensor(f0_output, device=u.device, dtype=u.dtype).detach()
        if output.shape != u.shape:
            checks.require(~enabled, 'SPM velocity fields must have the same shape')
            consistency = torch.zeros_like(u)
        else:
            consistency = torch.where(enabled, coefficient*gamma*(output-f1), torch.zeros_like(u))
    parts = (reward+teacher_force+consistency, reward, teacher_force, consistency)
    checks.require(torch.stack([torch.isfinite(part).all() for part in parts]).all(),
                   'SPM force contains non-finite values', FloatingPointError)
    checks.check()
    return parts


def _per_sample(scale, field):
    return scale.reshape((-1,) + (1,) * (field.ndim - 1)).to(field)


def clean_prediction_loss(prediction, force, sigma):
    """||x1 - sg(x1 - s G)||^2 / (2 s^2): dL/dx1 = G/s, hence dL/dv = -G.

    Subtract the identical detached prediction before adding the force, so small fields are not
    rounded away by a large clean sample. Only the current Student prediction retains a graph.
    """
    work = prediction.float()
    scale = torch.as_tensor(sigma, device=work.device, dtype=work.dtype).detach()
    if scale.ndim == 1:
        scale = _per_sample(scale, work)
    checks = TensorChecks()
    checks.require((torch.isfinite(scale) & (scale > 0)).all(), 'clean prediction sigma must be finite and positive')
    residual = (work - work.detach()) / scale + force.detach().to(work)
    loss = residual.square().flatten(1).sum(1).mean() / 2
    checks.require(torch.isfinite(loss.detach()), 'SPM loss is not finite', FloatingPointError)
    checks.check()
    return loss


@torch.no_grad()
def average_gradients(model):
    """Average accumulated gradients across ranks once, after every local backward."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    parameters = [p for p in model.parameters() if p.requires_grad]
    flat = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).float().reshape(-1) for p in parameters])
    dist.all_reduce(flat)
    flat /= dist.get_world_size()
    offset = 0
    for p in parameters:
        p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype).clone()
        offset += p.numel()


def flow_matching_loss(prediction, target):
    """Average the per-sample velocity mean squared error."""
    if (not isinstance(prediction, torch.Tensor) or not prediction.is_floating_point()
            or prediction.ndim < 2 or not prediction.numel()):
        raise ValueError('prediction must be a nonempty floating field [batch, ...]')
    if (not isinstance(target, torch.Tensor) or not target.is_floating_point()
            or target.shape != prediction.shape or target.device != prediction.device):
        raise ValueError('target must match the prediction field shape and device')
    dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    error = (prediction.to(dtype)-target.detach().to(dtype)).square()
    loss = error.flatten(1).mean(1).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Fake loss contains non-finite values')
    return loss
