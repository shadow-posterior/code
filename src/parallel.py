"""Distributed process groups, model wrapping and collective helpers."""

from __future__ import annotations

import os
import functools
from typing import Callable, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.device_mesh import init_device_mesh


def setup_distributed() -> None:
    """Initialize the default distributed process group.

    Expects standard torchrun environment variables (RANK, WORLD_SIZE,
    LOCAL_RANK, MASTER_ADDR, MASTER_PORT) to be set.
    """
    if dist.is_initialized():
        return

    from datetime import timedelta
    timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SEC", "10800"))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def cleanup_distributed() -> None:
    """Destroy the default process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


_ScoreResult = TypeVar('_ScoreResult')


def synchronized_reward_scoring(score: Callable[[], _ScoreResult], *, context: str) -> _ScoreResult:
    """Finish local scoring on every rank before any reward-data collective.

    The callback must only perform local work, never distributed collectives.
    Standalone calls retain their original result and exception semantics. With
    multiple ranks, every worker receives the same scoring error before callers
    may proceed to reward gathering or credit computation.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return score()
    result, local_error, failure = None, None, None
    try:
        result = score()
    except Exception as error:
        local_error = error
        failure = f'{type(error).__name__}: {error}'
    failures = [None] * dist.get_world_size()

    dist.all_gather_object(failures, failure)
    errors = [f'rank {rank}: {error}' for rank, error in enumerate(failures) if error is not None]
    if errors:
        raise RuntimeError(f'{context} failed; ' + '; '.join(errors)) from local_error
    return result


_STRATEGY_MAP = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "hybrid": ShardingStrategy.HYBRID_SHARD,
    "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    "no_shard": ShardingStrategy.NO_SHARD,
}


def get_sharding_strategy(name: str) -> ShardingStrategy:
    """Map a string name to a torch FSDP ShardingStrategy enum.

    Args:
        name: One of "full_shard", "shard_grad_op", "hybrid", "hybrid_zero2", "no_shard".

    Returns:
        Corresponding ShardingStrategy enum value.
    """
    if name not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown sharding strategy: {name}. "
            f"Choose from: {list(_STRATEGY_MAP.keys())}"
        )
    return _STRATEGY_MAP[name]


_DTYPE_ALIASES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def get_mixed_precision_policy(fsdp_precision: str) -> MixedPrecision | None:
    """Create an FSDP MixedPrecision policy from a string identifier.

    Args:
        fsdp_precision: "bf16", "fp16", "fp32", or "no" (full fp32).
            Also accepts PyTorch-style names ("bfloat16", "float16", "float32").

    Returns:
        MixedPrecision policy or None for fp32.
    """
    fsdp_precision = _DTYPE_ALIASES.get(fsdp_precision, fsdp_precision)
    if fsdp_precision == "bf16":
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif fsdp_precision == "fp16":
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    elif fsdp_precision in {"no", "fp32"}:
        return None
    else:
        raise ValueError(
            f"Unknown precision: {fsdp_precision}. Choose from: bf16, fp16, fp32, no"
        )


def fsdp_wrap_model(
    model: torch.nn.Module,
    sharding_strategy: str = "full_shard",
    fsdp_precision: str = "bf16",
    auto_wrap_policy: Callable | None = None,
    device_id: int | None = None,
    use_orig_params: bool = True,
    ignored_modules: list | None = None,
    cpu_offload: bool = False,
) -> FSDP:
    """Wrap a model with FSDP.

    Args:
        model: The PyTorch module to wrap.
        sharding_strategy: FSDP sharding strategy name.
        fsdp_precision: FSDP precision policy name (param/reduce/buffer dtype).
            This is intentionally decoupled from autocast precision to keep
            gradient reduction stable (e.g., bf16 reduce with fp16 autocast).
        auto_wrap_policy: Custom FSDP auto-wrap policy. If None, uses
            size-based policy wrapping modules with >1M parameters.
        device_id: CUDA device ID. Defaults to LOCAL_RANK.
        use_orig_params: If True, FSDP preserves original parameter structure
            instead of flattening into FlatParameters. Required for correct
            behavior with LoRA (mixed frozen/trainable parameters) and
            recommended for PyTorch 2.0+.
        ignored_modules: List of submodules to exclude from FSDP sharding.
            Useful for modules containing Conv2d with 4D weights (e.g.,
            PatchEmbed.proj) which can cause shape mismatch in FSDP
            use_orig_params mode.
        cpu_offload: If True, keep FSDP parameters on CPU and copy to GPU per
            forward. Recommended for frozen inference-only models (e.g. teacher)
            to avoid overlapping unshard peaks with the trainable generator.

    Returns:
        FSDP-wrapped model.
    """
    if device_id is None:
        device_id = get_local_rank()

    strategy = get_sharding_strategy(sharding_strategy)
    mp_policy = get_mixed_precision_policy(fsdp_precision)

    if auto_wrap_policy is None:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1_000_000
        )

    fsdp_extra_kwargs: dict = {}
    if strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
        try:
            mesh = get_device_mesh(use_hybrid=True)
            fsdp_extra_kwargs["device_mesh"] = mesh
        except Exception:
            pass

    if ignored_modules:
        fsdp_extra_kwargs["ignored_modules"] = ignored_modules

    cpu_offload_policy = CPUOffload(offload_params=True) if cpu_offload else None

    wrapped = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.device("cuda", device_id),
        limit_all_gathers=True,
        use_orig_params=use_orig_params,
        cpu_offload=cpu_offload_policy,
        **fsdp_extra_kwargs,
    )
    return wrapped


def free_fsdp_unsharded_params(module: torch.nn.Module | None) -> None:
    """Release FSDP1 unsharded parameters after an inference-only forward.

    PyTorch FSDP1 does not expose a public ``reshard()`` on the wrapper (calling
    ``module.reshard()`` delegates to the inner nn.Module and fails). Use the
    internal reshard helper with ``free_unsharded_flat_param=True`` so a second
    large FSDP model can unshard without peak OOM.
    """
    if module is None or not isinstance(module, FSDP):
        return
    handle = getattr(module, "_handle", None)
    if handle is None:
        return
    from torch.distributed.fsdp._runtime_utils import _reshard as _fsdp_internal_reshard

    _fsdp_internal_reshard(module, handle,
                          free_unsharded_flat_param=handle.uses_sharded_strategy)


@torch.no_grad()
def release_fsdp_caches(model, *, no_shard_only=False):
    """Invalidate inference caches before mutating local original-parameter shards.

    FSDP1 can retain a root full-parameter cache after a no-gradient forward;
    SHARD_GRAD_OP may retain nested caches as well. Updating only the shards
    would otherwise leave subsequent forwards using the previous weights.
    ``no_shard_only`` restores original parameter views after backward in
    single-rank FSDP without discarding multi-rank SHARD_GRAD_OP caches.
    """
    if isinstance(model, FSDP):
        for module in reversed(FSDP.fsdp_modules(model)):
            handle = getattr(module, '_handle', None)

            if handle is not None and hasattr(handle.flat_param, '_local_shard'):
                if no_shard_only and handle.uses_sharded_strategy:
                    continue
                free_fsdp_unsharded_params(module)


def get_transformer_wrap_policy(block_type: type) -> Callable:
    """Wrap each transformer block as an FSDP unit."""
    return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={block_type})


def ddp_wrap_model(
    model: torch.nn.Module,
    device_id: int | None = None,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
) -> torch.nn.parallel.DistributedDataParallel:
    """Wrap a model with DDP.

    Args:
        model: The PyTorch module to wrap (should already be on the correct device).
        device_id: CUDA device ID. Defaults to LOCAL_RANK.
        find_unused_parameters: Whether DDP should find unused parameters.
        broadcast_buffers: Whether DDP broadcasts module buffers before forward.
            Keep True by default for behavior parity with vanilla DDP.

    Returns:
        DDP-wrapped model.
    """
    if device_id is None:
        device_id = get_local_rank()
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device_id],
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )


def get_device_mesh(use_hybrid: bool = False) -> torch.distributed.device_mesh.DeviceMesh:
    """Initialize a device mesh for FSDP hybrid sharding.

    For multi-node training, hybrid sharding shards within a node and
    replicates across nodes, reducing inter-node communication.

    Args:
        use_hybrid: If True, creates a 2D mesh (replicate x shard).
            If False, creates a 1D mesh (shard only).

    Returns:
        DeviceMesh instance.
    """
    world_size = get_world_size()
    if use_hybrid:
        n_nodes = world_size // get_local_world_size()
        local_size = get_local_world_size()
        mesh = init_device_mesh(
            "cuda",
            (n_nodes, local_size),
            mesh_dim_names=("replicate", "shard"),
        )
    else:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))
    return mesh
