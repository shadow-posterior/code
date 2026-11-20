"""Training loop, evaluation, and checkpoint lifecycle."""

from __future__ import annotations

from src.rewards.geneval_metrics import with_geneval_overall
from src.rewards.scorer import RewardScorer

from src.paths import create_run_directory, project_path

import json
import logging
import os
import random
import time
import traceback
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from src.config import TrainingConfig
from src.parallel import (
    setup_distributed,
    release_fsdp_caches,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
)
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.evaluation import score_details_to_metric_stats, score_eval_images, reward_evaluation_routes
from src.utils.run_state import (
    resume_run_directory, select_resume_checkpoint, write_checkpoint_manifest,
)
from src.utils.logging import (
    SwanlabLogger,
    Timer,
    get_gpu_memory_gb,
    setup_logging,
)

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer with distributed training infrastructure.

    Args:
        config: The root training configuration.
    """

    _AUTOCAST_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}

    def __init__(self, config: TrainingConfig):
        self.config = config
        config.output_dir = project_path(config.output_dir)
        config.prompt_path = project_path(config.prompt_path)
        config.eval.dataset_path = project_path(config.eval.dataset_path)
        config.eval.eval_prompt_path = project_path(config.eval.eval_prompt_path)
        if config.train.resume_from:
            config.train.resume_from = project_path(config.train.resume_from)
        self.global_step = 0
        self._student_microstep = 0
        self._resume_rng_state = None
        self.models: dict[str, torch.nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, Any] = {}
        self.timer = Timer()
        self.swanlab_logger: SwanlabLogger | None = None
        self._autocast_dtype = self._AUTOCAST_DTYPE.get(config.train.autocast_dtype)

        self._eval_prompts: list[str] | None = None

        self._eval_prompts_cache: dict[str, list[str]] = {}

        self._eval_metadata_cache: dict[str, list[dict[str, Any]] | None] = {}

        self._eval_scorers: dict[str, RewardScorer] = {}

        self._eval_media_groups: dict[str, tuple[torch.Tensor, list[str]]] = {}

        self._eval_samples: dict[str, dict[str, Any]] = {}

        self._defer_scorer_offload = False

    def _autocast(self):
        """Return a torch.autocast context manager using the configured precision."""
        return torch.autocast(
            device_type="cuda",
            dtype=self._autocast_dtype,
            enabled=self._autocast_dtype is not None,
        )

    def train(self) -> None:
        """Main training entry point. Sets up everything and runs the loop."""
        self._setup_environment()
        self.setup_models()
        self.setup_optimizers()
        self._maybe_resume()
        self._setup_logging()
        self._train_loop()
        self._cleanup()

    def _setup_environment(self) -> None:
        """Initialize distributed backend, set seeds, and prepare output dir."""
        setup_distributed()
        setup_logging()
        if self.config.distributed.strategy == "fsdp" and get_world_size() == 1:
            logger.warning("distributed.strategy=fsdp with world_size=1: using ddp (FSDP NO_SHARD writeback bug).")
            self.config.distributed.strategy = "ddp"
        self._set_seed(self.config.train.seed)

        self.config.output_dir = self._shared_main_result(lambda:
            str(resume_run_directory(self.config.train.resume_from))
            if self.config.train.resume_from else create_run_directory(self.config.output_dir))

        logger.info(
            f"Environment ready: rank={get_rank()}, world_size={get_world_size()}, "
            f"device=cuda:{torch.cuda.current_device()}, output_dir={self.config.output_dir}"
        )

    @staticmethod
    def _shared_main_result(operation: Callable[[], Any]) -> Any:
        """Resolve paths once and propagate both the value and any failure."""
        result = [None]
        if is_main_process():
            try:
                result[0] = operation()
            except Exception as error:
                result[0] = error
        if dist.is_initialized() and get_world_size() > 1:
            dist.broadcast_object_list(result, src=0)
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility.

        TF32 is always enabled for performance. Full deterministic mode
        (cudnn.benchmark=False, deterministic algorithms) is only activated
        when TORCH_DETERMINISTIC=1 env var is set — useful for resume
        verification but too slow for normal training.
        """
        seed = seed + get_rank()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if os.environ.get("TORCH_DETERMINISTIC", "0") == "1":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            logger.info("Full deterministic mode enabled (TORCH_DETERMINISTIC=1)")

    def _setup_logging(self) -> None:
        """Initialize SwanLab logger on the main process."""
        log_cfg = self.config.logging
        self.swanlab_logger = SwanlabLogger(
            project=log_cfg.project,
            run_name=log_cfg.run_name,
            workspace=log_cfg.workspace,
            config=self.config.to_dict(),
            tags=log_cfg.tags,
            enabled=log_cfg.enabled,
            output_dir=self.config.output_dir,
            resume=bool(self.config.train.resume_from),
        )

        if is_main_process():
            os.makedirs(self.config.output_dir, exist_ok=True)
            config_path = os.path.join(self.config.output_dir, 'config.yaml')
            if self.config.train.resume_from and os.path.exists(config_path):
                config_path = os.path.join(self.config.output_dir, 'resume_config.yaml')
            self.config.save_yaml(config_path)

    def _maybe_resume(self) -> None:
        """Restore training state and defer the RNG restore until after dataloader setup."""
        self._resume_rng_state = None
        resume_path = self.config.train.resume_from
        if resume_path:
            resume_path = project_path(resume_path)
            resume_path = self._shared_main_result(lambda:
                select_resume_checkpoint(resume_path, self._required_checkpoint_files()))
            self.config.train.resume_from = resume_path
            extra = load_checkpoint(
                resume_path,
                models=self.models,
                optimizers=self.optimizers,
            )
            self._restore_extra_checkpoint_state(extra)
            self.global_step = extra['step']
            scheduler_states = extra['schedulers']
            if scheduler_states.keys() != self.schedulers.keys():
                raise ValueError('Checkpoint schedulers differ from the current run')
            for name, scheduler in self.schedulers.items():
                scheduler.load_state_dict(scheduler_states[name])
            self._resume_rng_state = extra['rng_state']

            logger.info(f"Resumed training from step {self.global_step}")

    def _required_checkpoint_files(self) -> list[str]:
        """Files that must exist before an automatically selected checkpoint loads."""
        return [*(f'{name}.pt' for name in self.models),
                *(f'optimizer_{name}.pt' for name in self.optimizers)]

    @abstractmethod
    def _restore_extra_checkpoint_state(self, extra: dict) -> None:
        """Validate the training contract and restore the source offsets."""
        ...

    def _restore_rng_state(self) -> None:
        """Restore RNG states saved during resume.

        Called after dataloader creation and fast-forward so that the training
        loop sees exactly the same RNG state as the original run.
        """
        rng_state = self._resume_rng_state
        if rng_state is None:
            return

        rank_states = rng_state['rank_states']
        cuda_states = rng_state['torch_cuda_all_ranks']
        if len(rank_states) != get_world_size() or len(cuda_states) != get_world_size():
            raise ValueError('Checkpoint RNG world size differs from the current run')
        rank = get_rank()
        local_rng = rank_states[rank]
        random.setstate(local_rng['python'])
        np.random.set_state(local_rng['numpy'])
        torch.random.set_rng_state(local_rng['torch_cpu'])
        torch.cuda.set_rng_state(cuda_states[rank])
        self._resume_rng_state = None

    def _cleanup(self) -> None:
        """Clean up distributed resources and finalize logging."""
        if self.swanlab_logger:
            self.swanlab_logger.finish()
        cleanup_distributed()

    def _train_loop(self) -> None:
        """Main training loop with gradient accumulation and logging."""
        cfg = self.config.train
        iterations = cfg.gradient_accumulation_steps
        dataloader, sampler = self._create_dataloader()

        logger.info(
            f"Starting training: student_updates={cfg.total_steps}, "
            f"batch_size={cfg.batch_size}, grad_accum={cfg.gradient_accumulation_steps}, "
            f"effective_batch_size={cfg.batch_size * cfg.gradient_accumulation_steps * get_world_size()}"
        )

        epoch, step_in_epoch = divmod(self.global_step * iterations, len(dataloader))
        sampler.set_epoch(epoch)
        data_iter = iter(dataloader)
        if step_in_epoch:
            logger.info(f"Fast-forwarding dataloader: skipping {step_in_epoch} batches (epoch={epoch})")
            for _ in range(step_in_epoch):
                next(data_iter)

        self._restore_rng_state()

        eval_cfg = self.config.eval
        if eval_cfg.enabled and eval_cfg.eval_at_start and self.global_step == 0:
            with self.timer.measure("eval"):
                self._evaluate()

        last_saved_step = None
        while self.global_step < cfg.total_steps:
            with self.timer.measure("train_step"):
                for micro in range(iterations):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        epoch += 1
                        sampler.set_epoch(epoch)
                        data_iter = iter(dataloader)
                        batch = next(data_iter)
                    self._student_microstep = micro
                    metrics = self.train_step(batch)
            self._student_microstep = 0

            self.global_step += 1

            if self.global_step % cfg.log_interval == 0:
                self._log_step(metrics)

            if eval_cfg.enabled and (eval_cfg.eval_interval > 0 and self.global_step % eval_cfg.eval_interval == 0
                                     or self.global_step == cfg.total_steps):
                with self.timer.measure("eval"):
                    self._evaluate()

            if cfg.save_interval > 0 and self.global_step % cfg.save_interval == 0:
                self._save_checkpoint()
                last_saved_step = self.global_step

        logger.info(f"Training complete at step {self.global_step}")

        if last_saved_step != self.global_step:
            self._save_checkpoint()

        self._save_final_transformer()

    def _log_step(self, metrics: dict[str, Any]) -> None:
        """Log training metrics via SwanLab and console.

        Averages scalar metrics across all ranks before logging to ensure
        consistent reporting in distributed training.

        Args:
            metrics: Dict of section -> {metric_name: value} from train_step.
        """

        if dist.is_initialized() and get_world_size() > 1:
            metrics = self._reduce_metrics(metrics)

        from src.utils.reward_logging import with_training_reward_section
        metrics = with_training_reward_section(metrics)

        profile_metrics = {
            "gpu_mem_gb": get_gpu_memory_gb(),
        }
        profile_metrics.update(self.timer.get_metrics())

        all_sections = dict(metrics)
        all_sections["profile"] = profile_metrics

        if self.swanlab_logger:
            self.swanlab_logger.log_multi_section(all_sections, step=self.global_step)

        if is_main_process():
            self._append_jsonl("metrics.jsonl", {"step": self.global_step, **all_sections})
            parts = [f"step={self.global_step}"]
            for section, section_metrics in all_sections.items():
                for k, v in section_metrics.items():
                    if isinstance(v, float):
                        parts.append(f"{section}/{k}={v:.4g}")
                    else:
                        parts.append(f"{section}/{k}={v}")
            logger.info(" | ".join(parts))

    def _save_checkpoint(self) -> None:
        """Save the adapters, training state and per-rank RNG state."""
        scheduler_states = {
            name: sched.state_dict() for name, sched in self.schedulers.items()
        }

        local_cuda_rng = torch.cuda.get_rng_state()
        if dist.is_initialized() and get_world_size() > 1:
            local_rng_cuda = local_cuda_rng.to(torch.cuda.current_device())
            all_cuda_rng = [torch.empty_like(local_rng_cuda) for _ in range(get_world_size())]
            dist.all_gather(all_cuda_rng, local_rng_cuda)
            all_cuda_rng = [t.cpu() for t in all_cuda_rng]
        else:
            all_cuda_rng = [local_cuda_rng]

        saved_python_rng = random.getstate()
        saved_numpy_rng = np.random.get_state()
        saved_torch_cpu_rng = torch.random.get_rng_state()

        local_rng = {"python": saved_python_rng, "numpy": saved_numpy_rng,
                     "torch_cpu": saved_torch_cpu_rng}
        rank_states = [local_rng]
        if dist.is_initialized() and get_world_size() > 1:
            rank_states = [None] * get_world_size()
            dist.all_gather_object(rank_states, local_rng)

        extra_state = {
            **self._get_extra_checkpoint_state(),
            'schedulers': scheduler_states,
            'rng_state': {'rank_states': rank_states, 'torch_cuda_all_ranks': all_cuda_rng},
        }
        checkpoint_dir = Path(self.config.output_dir) / f'checkpoint-{self.global_step}'
        files = ['meta.pt', 'config.yaml', *self._required_checkpoint_files()]
        try:
            self._shared_main_result(lambda: write_checkpoint_manifest(
                checkpoint_dir, self.global_step, files, complete=False))
            save_checkpoint(
                output_dir=self.config.output_dir,
                step=self.global_step,
                models=self.models,
                optimizers=self.optimizers,
                generator_lora=self.config.model.generator_lora,
                pretrained_path=self.config.model.pretrained_path,
                extra_state=extra_state,
            )

            def commit_checkpoint():
                self.config.save_yaml(str(checkpoint_dir / 'config.yaml'))
                write_checkpoint_manifest(checkpoint_dir, self.global_step, files, complete=True)

            self._shared_main_result(commit_checkpoint)
        finally:
            random.setstate(saved_python_rng)
            np.random.set_state(saved_numpy_rng)
            torch.random.set_rng_state(saved_torch_cpu_rng)
            torch.cuda.set_rng_state(local_cuda_rng)

    @abstractmethod
    def _get_extra_checkpoint_state(self) -> dict:
        """Collect the training contract and source offsets for the checkpoint."""
        ...

    def _save_final_transformer(self) -> None:
        """Export the final generator adapter, using EMA weights when available."""
        from src.utils.checkpoint import _gather_fsdp_state_dict, save_lora_peft_format

        generator = self.models.get('generator_ema', self.models['generator'])
        state_dict = _gather_fsdp_state_dict(generator)
        directory = os.path.join(self.config.output_dir, 'transformer')
        self._shared_main_result(lambda: save_lora_peft_format(
            state_dict, self.config.model.generator_lora, directory,
            pretrained_path=self.config.model.pretrained_path))

    def _load_eval_prompts_and_metadata(
        self,
        dataset_override: str | None = None,
    ) -> tuple[list[str], list[dict[str, Any]] | None]:
        """Load evaluation prompts and optional metadata from file or dataset directory."""
        eval_cfg = self.config.eval

        use_dataset = dataset_override is not None or eval_cfg.eval_prompt_source == "dataset"
        if use_dataset:
            dataset_name = dataset_override if dataset_override is not None else eval_cfg.dataset
            dataset_dir = os.path.join(eval_cfg.dataset_path, dataset_name)

            for fname in [
                "test.txt",
                "test.jsonl",
                "test_metadata.jsonl",
                "prompts.json",
                "prompts.txt",
                "metadata.jsonl",
            ]:
                fpath = os.path.join(dataset_dir, fname)
                if os.path.exists(fpath):
                    return self._read_prompt_file_with_metadata(fpath)
            raise FileNotFoundError(
                f"No prompt file found in dataset directory: {dataset_dir}"
            )
        else:
            return self._read_prompt_file_with_metadata(eval_cfg.eval_prompt_path)

    @staticmethod
    def _read_prompt_file_with_metadata(
        path: str,
    ) -> tuple[list[str], list[dict[str, Any]] | None]:
        """Read prompts and optional metadata from a JSON/JSONL/text file."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], str):
                    return data, None
                elif len(data) > 0 and isinstance(data[0], dict):
                    prompts = [item.get("prompt", item.get("text", "")) for item in data]
                    return prompts, data
            raise ValueError(f"Cannot parse prompts from JSON: {path}")
        elif ext == ".jsonl":
            prompts = []
            metadata: list[dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            prompts.append(item.get("prompt", item.get("text", "")))
                            metadata.append(item)
                        else:
                            prompts.append(str(item))
                            metadata.append({"text": str(item)})
            return prompts, metadata
        else:
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()], None

    @staticmethod
    def _cache_key_for_dataset(dataset_override: str | None = None) -> str:
        return "__default__" if dataset_override is None else f"dataset::{dataset_override}"

    def _get_cached_eval_prompts(self, dataset_override: str | None = None) -> list[str]:
        """Get cached eval prompts for default or reward-specific dataset."""
        cache_key = self._cache_key_for_dataset(dataset_override=dataset_override)
        prompts = self._eval_prompts_cache.get(cache_key)
        if prompts is None:
            prompts, metadata = self._load_eval_prompts_and_metadata(
                dataset_override=dataset_override
            )
            self._eval_prompts_cache[cache_key] = prompts
            self._eval_metadata_cache[cache_key] = metadata
            if dataset_override is None:
                self._eval_prompts = prompts
            source = (
                f"dataset={dataset_override}"
                if dataset_override is not None
                else f"{self.config.eval.eval_prompt_source}"
            )
            logger.info(f"[Eval] Loaded {len(prompts)} prompts ({source})")
        return prompts

    def _get_cached_eval_metadata(
        self,
        dataset_override: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Get cached optional metadata aligned with eval prompts."""
        cache_key = self._cache_key_for_dataset(dataset_override=dataset_override)
        if cache_key not in self._eval_prompts_cache:
            self._get_cached_eval_prompts(dataset_override=dataset_override)
        return self._eval_metadata_cache.get(cache_key)

    def _resolve_eval_dataset_name(self, dataset_override: str | None = None) -> str:
        """Resolve the effective dataset name used for evaluation."""
        if dataset_override is not None:
            return dataset_override
        if self.config.eval.eval_prompt_source == "dataset":
            return str(self.config.eval.dataset)
        return "prompt_file"

    def _maybe_cache_eval_media_group(
        self,
        dataset_name: str,
        images: torch.Tensor | None,
        prompts: list[str],
    ) -> None:
        """Cache per-dataset media samples for later SwanLab logging."""
        if (
            not is_main_process()
            or self.config.eval.num_media_images <= 0
            or dataset_name in self._eval_media_groups
            or images is None
            or len(images) == 0
        ):
            return

        num_images = min(self.config.eval.num_media_images, len(images))
        self._eval_media_groups[dataset_name] = (
            images[:num_images].detach().cpu().clone(),
            prompts[:num_images],
        )

    @staticmethod
    def _sanitize_media_key_token(name: str) -> str:
        """Make dataset names safe as SwanLab media key suffixes."""
        token = (name or "").strip()
        if not token:
            return "default"
        token = token.replace("/", "_").replace(" ", "_")
        token = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in token)
        return token or "default"

    def _log_eval_media_groups(
        self,
        *,
        step: int,
        base_key: str,
        fallback_images: torch.Tensor | None = None,
        fallback_prompts: list[str] | None = None,
    ) -> None:
        """Log evaluation media; one panel per dataset when multiple are used."""
        if (
            self.swanlab_logger is None
            or self.config.eval.num_media_images <= 0
            or not is_main_process()
        ):
            return

        media_groups = dict(self._eval_media_groups)
        if not media_groups and fallback_images is not None and len(fallback_images) > 0:
            dataset_name = self._resolve_eval_dataset_name(dataset_override=None)
            num_images = min(self.config.eval.num_media_images, len(fallback_images))
            media_groups[dataset_name] = (
                fallback_images[:num_images].detach().cpu(),
                (fallback_prompts or [])[:num_images],
            )

        if not media_groups:
            return

        multi_dataset = len(media_groups) > 1
        for dataset_name, (images, prompts) in media_groups.items():
            if images is None or len(images) == 0:
                continue
            log_pil = [
                Image.fromarray(
                    (img.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                )
                for img in images
            ]
            key = (
                f"{base_key}/{self._sanitize_media_key_token(dataset_name)}"
                if multi_dataset
                else base_key
            )
            self.swanlab_logger.log_images(
                images=log_pil,
                captions=prompts[:len(log_pil)],
                step=step,
                key=key,
            )

    def _get_eval_scorer(self, reward_name: str) -> RewardScorer:
        scorer = self._eval_scorers.get(reward_name)
        if scorer is None:
            scorer = RewardScorer(reward_name, torch.device("cpu"), allow_unavailable=True)
            self._eval_scorers[reward_name] = scorer
        return scorer

    def _offload_eval_scorers(self) -> None:
        """Move evaluation scorers to the CPU and release cached GPU memory."""
        for scorer in self._eval_scorers.values():
            scorer.to(torch.device("cpu"))
        torch.cuda.empty_cache()

    @staticmethod
    def _score_details_to_metric_stats(
        score_details: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, int]]:
        return score_details_to_metric_stats(score_details)

    @staticmethod
    def _collect_cross_rank_error_messages(
        local_failed: bool,
        local_error: str | None,
    ) -> list[str]:
        """Gather unique error messages from all ranks for clearer eval diagnostics."""
        if not local_failed:
            return []
        world_size = get_world_size()
        payload = local_error.strip() if local_error else None
        if payload and len(payload) > 8000:
            payload = payload[:8000] + "\n...[truncated-before-gather]"
        if dist.is_initialized() and world_size > 1:
            gathered_errors: list[str | None] = [None] * world_size
            dist.all_gather_object(gathered_errors, payload)
            dedup: list[str] = []
            seen: set[str] = set()
            for item in gathered_errors:
                if isinstance(item, str):
                    text = item.strip()
                    if text and text not in seen:
                        dedup.append(text)
                        seen.add(text)
            return dedup
        return [payload] if payload else []

    @staticmethod
    def _truncate_error_message(message: str, max_chars: int = 1600) -> str:
        text = message.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"

    @staticmethod
    def _aggregate_eval_metrics_across_ranks(
        eval_metrics: dict[str, float],
        metric_counts: dict[str, int],
    ) -> dict[str, float]:
        """Cross-rank weighted average aggregation for eval metrics.

        Supports per-rank metric key mismatch safely (e.g., some ranks only have
        invalid values for a reward and thus emit no local metric key).
        """
        world_size = get_world_size()
        if not dist.is_initialized() or world_size <= 1:
            return with_geneval_overall(eval_metrics)

        device = (torch.device("cuda", torch.cuda.current_device())
                  if dist.get_backend() == "nccl" else torch.device("cpu"))

        local_keys = list(eval_metrics.keys())
        gathered_keys: list[list[str] | None] = [None] * world_size
        dist.all_gather_object(gathered_keys, local_keys)
        all_keys = sorted({k for keys in gathered_keys if keys is not None for k in keys})
        aggregated = {}
        for k in all_keys:
            has_local = k in eval_metrics
            count = metric_counts.get(k, 0) if has_local else 0
            local_num = torch.tensor(
                float(eval_metrics[k] * count) if has_local else 0.0,
                device=device,
            )
            local_den = torch.tensor(
                float(count),
                device=device,
            )
            dist.all_reduce(local_num, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_den, op=dist.ReduceOp.SUM)
            if local_den.item() > 0:
                aggregated[k] = local_num.item() / local_den.item()
        return with_geneval_overall(aggregated)

    def _generate_eval_images_for_prompts(
        self,
        all_eval_prompts: list[str],
        batch_size: int,
        batch_generate_fn: Callable[[list[str]], torch.Tensor],
    ) -> tuple[torch.Tensor | None, list[str]]:
        """Generate eval images for a prompt list with distributed-safe batching.

        Args:
            all_eval_prompts: Full prompt list before sharding.
            batch_size: Per-rank eval batch size.
            batch_generate_fn: Receives one padded batch of prompts and returns
                decoded image tensor [B, C, H, W] in [0,1].

        Returns:
            (images_on_generation_device_or_none, all_prompts_local_non_padded)
        """
        rank = get_rank()
        world_size = get_world_size()

        if len(all_eval_prompts) == 0:
            return None, []

        rank_prompts = all_eval_prompts[rank::world_size]
        max_prompts_per_rank = (len(all_eval_prompts) + world_size - 1) // world_size
        max_batches = (max_prompts_per_rank + batch_size - 1) // batch_size

        all_images: list[torch.Tensor] = []
        all_prompts_local: list[str] = []

        for batch_idx in range(max_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(rank_prompts))

            if start_idx < len(rank_prompts):
                batch_prompts = rank_prompts[start_idx:end_idx]
                if len(batch_prompts) < batch_size:
                    batch_prompts = batch_prompts + [batch_prompts[-1]] * (batch_size - len(batch_prompts))
                actual_count = end_idx - start_idx
            else:
                batch_prompts = [rank_prompts[0] if rank_prompts else all_eval_prompts[0]] * batch_size
                actual_count = 0

            image_tensors = batch_generate_fn(batch_prompts)
            if actual_count > 0:
                all_images.append(image_tensors[:actual_count].detach())
                all_prompts_local.extend(batch_prompts[:actual_count])
            del image_tensors

        if not all_images:
            return None, []
        return torch.cat(all_images, dim=0), all_prompts_local

    def _run_eval_for_reward_sets(
        self,
        batch_size: int,
        batch_generate_fn: Callable[[list[str]], torch.Tensor],
    ) -> tuple[dict[str, float], torch.Tensor | None, list[str]]:
        """Generate each dataset once and score its rewards independently."""
        eval_cfg = self.config.eval
        shared_prompts = not (eval_cfg.reward_dataset_map or eval_cfg.benchmark_reward_fn)
        self._eval_media_groups = {}
        self._eval_samples = {}
        gpu_device = (torch.device("cuda", torch.cuda.current_device())
                      if torch.cuda.is_available() else torch.device("cpu"))
        eval_metrics: dict[str, float] = {}
        weighted_means: dict[str, float] = {}

        dataset_to_rewards: dict[str, dict[str, list[tuple[str, float]]]] = {}
        for dataset_override, reward_name, weight, prefix in reward_evaluation_routes(eval_cfg):
            dataset_name = self._resolve_eval_dataset_name(dataset_override=dataset_override)
            dataset_to_rewards.setdefault(dataset_name, {}).setdefault(reward_name, []).append(
                (prefix, weight))

        if shared_prompts:
            dataset_to_rewards.setdefault(self._resolve_eval_dataset_name(dataset_override=None), {})

        for dataset_name, reward_names in dataset_to_rewards.items():
            prompt_dataset_override = None if shared_prompts or dataset_name == "prompt_file" else dataset_name
            all_eval_prompts = self._get_cached_eval_prompts(
                dataset_override=prompt_dataset_override
            )
            all_eval_metadata = self._get_cached_eval_metadata(
                dataset_override=prompt_dataset_override
            )
            all_images_tensor, all_prompts_local = self._generate_eval_images_for_prompts(
                all_eval_prompts=all_eval_prompts,
                batch_size=batch_size,
                batch_generate_fn=batch_generate_fn,
            )
            self._maybe_cache_eval_media_group(
                dataset_name=dataset_name,
                images=all_images_tensor,
                prompts=all_prompts_local,
            )

            num_local = len(all_images_tensor) if all_images_tensor is not None else 0
            self._record_eval_samples(dataset_name, all_prompts_local, {}, num_local)
            combined_scores = []
            if is_main_process():
                logger.info(
                    f"[Eval] Dataset {dataset_name!r}: generated {num_local} local samples; "
                    f"running rewards {reward_names}"
                )
            metadata_local = (
                all_eval_metadata[get_rank()::get_world_size()]
                if isinstance(all_eval_metadata, list)
                else None
            )
            if isinstance(metadata_local, list) and num_local > 0 and len(metadata_local) != num_local:
                metadata_local = metadata_local[:num_local]
            cpu_device = torch.device("cpu")
            for reward_name in reward_names:
                scorer = self._get_eval_scorer(reward_name)
                scorer.to(gpu_device)
                reward_failed_local = False
                reward_error_local: str | None = None
                reward_t0 = time.perf_counter()
                if is_main_process():
                    logger.info(
                        f"[Eval] Start reward {reward_name!r} on dataset {dataset_name!r}"
                    )
                if all_images_tensor is not None and num_local > 0:
                    try:
                        score_details = score_eval_images(
                            scorer, all_images_tensor, all_prompts_local,
                            metadata=metadata_local, batch_size=batch_size, device=gpu_device,
                        )
                    except Exception as e:
                        reward_failed_local = True
                        reward_error_local = (
                            f"{type(e).__name__}: {e}\n"
                            f"{traceback.format_exc(limit=8)}"
                        )
                        score_details = {}
                else:
                    score_details = {}
                if dist.is_initialized() and get_world_size() > 1:
                    fail_flag = torch.tensor(
                        1 if reward_failed_local else 0,
                        device=gpu_device,
                        dtype=torch.int32,
                    )
                    dist.all_reduce(fail_flag, op=dist.ReduceOp.MAX)
                    reward_failed_local = bool(fail_flag.item())
                reward_error_messages = self._collect_cross_rank_error_messages(
                    local_failed=reward_failed_local,
                    local_error=reward_error_local,
                )
                if reward_failed_local:
                    if is_main_process():
                        elapsed = time.perf_counter() - reward_t0
                        logger.warning(
                            f"[Eval] Skipped reward {reward_name!r} on dataset {dataset_name!r}: "
                            f"failed on at least one rank (elapsed={elapsed:.2f}s)."
                        )
                        if reward_error_messages:
                            logger.warning(
                                f"[Eval] Failure example for reward {reward_name!r}:\n"
                                + self._truncate_error_message(reward_error_messages[0])
                            )

                    if not self._defer_scorer_offload:
                        scorer.to(cpu_device)
                        torch.cuda.empty_cache()
                    continue

                if is_main_process():
                    elapsed = time.perf_counter() - reward_t0
                    logger.info(
                        f"[Eval] Finished reward {reward_name!r} on dataset {dataset_name!r} "
                        f"(elapsed={elapsed:.2f}s)"
                    )

                self._record_eval_samples(dataset_name, all_prompts_local, score_details, num_local,
                                          reward_name=reward_name)
                local_metrics, metric_counts = self._score_details_to_metric_stats(score_details)
                aggregated = self._aggregate_eval_metrics_across_ranks(local_metrics, metric_counts)

                reward_value = aggregated.get(reward_name)
                if reward_value is None and is_main_process():
                    logger.warning(f"[Eval] Reward {reward_name!r} has no valid metric and was skipped.")
                if shared_prompts:
                    if reward_name in score_details:
                        weight = reward_names[reward_name][0][1]
                        scores = [weight * value for value in score_details[reward_name]]
                        combined_scores = ([a + b for a, b in zip(combined_scores, scores)]
                                           if combined_scores else scores)
                elif reward_value is not None:
                    for prefix, weight in reward_names[reward_name]:
                        weighted_means[prefix] = weighted_means.get(prefix, 0.0) + weight * reward_value

                for prefix, _weight in reward_names[reward_name]:
                    for key, value in aggregated.items():
                        out_key = f"{prefix}{key}"
                        if out_key in eval_metrics and key != reward_name:
                            out_key = f"{prefix}{reward_name}_{key}"
                        eval_metrics[out_key] = value

                if not self._defer_scorer_offload:
                    scorer.to(cpu_device)
                    torch.cuda.empty_cache()

            if shared_prompts:
                if combined_scores and isinstance(combined_scores[0], torch.Tensor):
                    combined_scores = torch.stack(combined_scores)
                local_metrics, counts = self._score_details_to_metric_stats({"mean": combined_scores})
                eval_metrics.update(self._aggregate_eval_metrics_across_ranks(local_metrics, counts))
                return eval_metrics, all_images_tensor, all_prompts_local

        for prefix, weighted_mean in weighted_means.items():
            eval_metrics[f"{prefix}mean"] = weighted_mean

        if self._eval_media_groups:
            first_dataset = next(iter(self._eval_media_groups))
            preview_images, preview_prompts = self._eval_media_groups[first_dataset]
            return eval_metrics, preview_images, preview_prompts
        return eval_metrics, None, []

    def _append_jsonl(self, name: str, record: dict[str, Any]) -> None:
        """Append one JSON record to output_dir/name (call on the main process)."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(os.path.join(self.config.output_dir, name), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")

    def _record_eval_samples(
        self,
        dataset_name: str,
        prompts_local: list[str],
        score_details: dict[str, Any],
        num_local: int,
        reward_name: str | None = None,
    ) -> None:
        """Keep this rank's per-sample scores of one dataset (invalid = -10 kept)."""
        entry = self._eval_samples.setdefault(dataset_name, {"prompts": list(prompts_local), "scores": {}})
        for key, values in score_details.items():
            if key == "mean":
                continue
            if isinstance(values, torch.Tensor):
                values = values.detach().flatten().cpu().tolist()
            elif isinstance(values, np.ndarray):
                values = values.flatten().tolist()
            if not isinstance(values, (list, tuple)) or len(values) != num_local:
                continue
            out_key = key
            if reward_name is not None and key in entry["scores"] and key != reward_name:
                out_key = f"{reward_name}_{key}"
            entry["scores"][out_key] = [float(v) for v in values]

    def _write_eval_records(self, step: int, model_name: str, metrics: dict[str, float]) -> None:
        """Append the latest eval pass to output_dir/eval_results.jsonl.

        Prompts are sharded rank::world, so global index = rank + world * i.
        Fixed eval seeds make records of the same prompts paired across steps
        (and across runs with the same world size and eval settings).
        """
        gathered = [self._eval_samples]
        if dist.is_initialized() and get_world_size() > 1:
            gathered = [None] * get_world_size()
            dist.all_gather_object(gathered, self._eval_samples)
        if not is_main_process():
            return
        world = len(gathered)
        samples = {}
        for dataset in gathered[0]:
            order = sorted((rank + world * i, rank, i)
                           for rank, part in enumerate(gathered)
                           for i in range(len(part.get(dataset, {}).get("prompts", []))))
            prompts = [gathered[r][dataset]["prompts"][i] for _, r, i in order]

            score_parts = [set(part[dataset]["scores"]) for part in gathered
                           if part.get(dataset, {}).get("prompts")]
            keys = set.intersection(*score_parts) if score_parts else set()
            samples[dataset] = {"prompts": prompts, "scores": {
                key: [gathered[r][dataset]["scores"][key][i] for _, r, i in order] for key in sorted(keys)}}
        self._append_jsonl("eval_results.jsonl", {"step": step, "model": model_name,
                                                   "world_size": world, "metrics": metrics,
                                                   "samples": samples})

    def _eval_models(self) -> list[tuple[str, torch.nn.Module]]:
        """(name, model) pairs that eval.models selects (default: EMA, else generator)."""
        names = list(self.config.eval.models) or [
            "generator_ema" if self.models.get("generator_ema") is not None else "generator"]
        missing = [name for name in names if self.models.get(name) is None]
        if missing:
            raise ValueError(f"eval.models names unknown models: {missing}")
        return [(name, self.models[name]) for name in names]

    def _warmup_eval_reward_resources(self) -> None:
        """Load evaluation prompts and scorers before setting the evaluation seed."""
        routes = reward_evaluation_routes(self.config.eval)
        if not routes:
            self._get_cached_eval_prompts()
        for dataset, reward_name, _, _ in routes:
            self._get_cached_eval_prompts(dataset_override=dataset)
            self._get_eval_scorer(reward_name)

    @torch.no_grad()
    def _decode_latents_to_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to float image tensors in [0, 1]."""
        from src.utils.image import decode_latents_to_tensor
        return decode_latents_to_tensor(self.vae, latents)

    @torch.no_grad()
    def _evaluate(self) -> None:
        """Periodic evaluation: generate images → reward scoring → SwanLab logging.

        Distributed-aware design:
        - Prompts are sharded across ranks so each rank generates different images
        - All ranks must call _generate_latents the same number of times (FSDP requirement)
        - Scores are all_reduced across ranks for aggregation
        - A barrier at the end ensures all ranks finish eval before resuming training

        Guarantees evaluation consistency via:
        1. Fixed eval_seed per rank (eval_seed + rank for data diversity)
        2. Fixed prompt order (no shuffle, deterministic sharding)
        3. Fixed inference steps
        4. RNG isolation (save/restore training RNG state)
        """
        eval_cfg = self.config.eval
        rank = get_rank()
        logger.info(f"[Eval] Starting evaluation at step {self.global_step} (rank={rank})")

        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(),
        }

        try:
            self._warmup_eval_reward_resources()

            eval_models = self._eval_models()
            for index, (model_name, generator_model) in enumerate(eval_models):
                eval_seed = eval_cfg.eval_seed + rank
                random.seed(eval_seed)
                np.random.seed(eval_seed)
                torch.manual_seed(eval_seed)
                torch.cuda.manual_seed_all(eval_seed)

                was_training = generator_model.training
                generator_model.eval()

                def _batch_generate(batch_prompts: list[str], _model=generator_model) -> torch.Tensor:
                    prompt_embeds, pooled_embeds = self._encode_prompts(batch_prompts)
                    latents, _ = self._generate_latents(
                        prompt_embeds, pooled_embeds, num_steps=eval_cfg.eval_num_steps, model=_model,
                    )

                    return self._decode_latents_to_tensor(latents)

                eval_metrics, all_images_tensor, all_prompts_local = self._run_eval_for_reward_sets(
                    batch_size=eval_cfg.eval_batch_size,
                    batch_generate_fn=_batch_generate,
                )
                generator_model.train(was_training)
                self._write_eval_records(self.global_step, model_name, eval_metrics)

                section = f"eval/{model_name}" if eval_cfg.models else "eval"
                if self.swanlab_logger:
                    self.swanlab_logger.log(eval_metrics, step=self.global_step, section=section)
                    if index == 0:
                        self._log_eval_media_groups(
                            step=self.global_step,
                            base_key="media/eval",
                            fallback_images=all_images_tensor,
                            fallback_prompts=all_prompts_local,
                        )
                if is_main_process():
                    parts = [f"[Eval] step={self.global_step} model={model_name}"]
                    for k, v in eval_metrics.items():
                        parts.append(f"{k}={v:.4g}")
                    logger.info(" | ".join(parts))

            if not self._defer_scorer_offload:
                self._offload_eval_scorers()
        finally:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.random.set_rng_state(rng_state["torch_cpu"])
            torch.cuda.set_rng_state(rng_state["torch_cuda"])

            if dist.is_initialized():
                dist.barrier()

    @staticmethod
    def _reduce_metrics(
        metrics: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Average scalar metrics across all ranks via all_reduce.

        Non-float values (e.g., int, str) are left as-is on the current rank.

        Args:
            metrics: Nested dict of section -> {metric_name: value}.

        Returns:
            New metrics dict with float values averaged across world_size.
        """
        reduced = {section: dict(values) for section, values in metrics.items()}
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return reduced

        keys = sorted((section, key) for section, values in metrics.items()
                      for key, value in values.items() if isinstance(value, float))
        if not keys:
            return reduced
        device = (torch.device("cuda", torch.cuda.current_device())
                  if dist.get_backend() == "nccl" else torch.device("cpu"))
        packed = torch.tensor([metrics[section][key] for section, key in keys],
                              dtype=torch.float64, device=device)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        values = (packed / dist.get_world_size()).cpu().tolist()
        for (section, key), value in zip(keys, values):
            reduced[section][key] = value
        return reduced

    def _fp32_master(self, *models) -> None:
        """Keep the trainable weights in float32 (model.fp32_master_weights); the compute dtype is unchanged.

        FSDP flattens every wrapped unit into one tensor of a single dtype, so under FSDP the whole model is
        cast (sharded; its MixedPrecision policy still computes in model.dtype). Otherwise only the weights
        that are trainable in any of `models` are cast, matched by name so that a frozen EMA copy follows its
        source, and the frozen weights keep their memory footprint.
        """
        if not getattr(self.config.model, "fp32_master_weights", False):
            return
        models = [m for m in models if m is not None]
        if self.config.distributed.strategy == "fsdp":
            for m in models:
                m.float()
            return
        names = {n for m in models for n, p in m.named_parameters() if p.requires_grad}
        for m in models:
            for n, p in m.named_parameters():
                if n in names:
                    p.data = p.data.float()

    @staticmethod
    @torch.no_grad()
    def ema_update(
        src_model: torch.nn.Module,
        tgt_model: torch.nn.Module,
        decay: float,
        *,
        trainable_only: bool = False,
    ) -> None:
        """Exponential Moving Average update: tgt = decay * tgt + (1 - decay) * src.

        Casts source params to float32 for numerical stability
        (standard EMA update).

        Args:
            src_model: Source model (latest parameters).
            tgt_model: Target model (EMA parameters).
            decay: EMA decay rate (e.g., 0.9999). Higher = slower update.
            trainable_only: Skip frozen source parameters, preserving the fixed
                backbone exactly and avoiding unnecessary updates.
        """
        release_fsdp_caches(src_model)
        release_fsdp_caches(tgt_model)
        def parameters_by_name(model):
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            return {name.replace('_fsdp_wrapped_module.', ''): parameter
                    for name, parameter in model.named_parameters()}

        targets = parameters_by_name(tgt_model)
        sources = parameters_by_name(src_model)
        active_sources = {name: param for name, param in sources.items()
                          if not trainable_only or param.requires_grad}
        if active_sources.keys() <= targets.keys():
            pairs = [(param, targets[name]) for name, param in active_sources.items()]
        else:
            pairs = list(zip(sources.values(), targets.values()))
            if len(sources) != len(targets) or any(a.shape != b.shape for a, b in pairs):
                raise ValueError('EMA source and target parameter layouts differ')
        for src_param, tgt_param in pairs:
            if trainable_only and not src_param.requires_grad:
                continue
            tgt_param.data.mul_(decay).add_(
                src_param.data.to(device=tgt_param.device, dtype=torch.float32), alpha=1.0 - decay
            )

    @staticmethod
    def create_warmup_constant_scheduler(
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Create a learning rate scheduler: linear warmup then constant.

        Args:
            optimizer: The optimizer to schedule.
            warmup_steps: Number of warmup steps.

        Returns:
            LambdaLR scheduler.
        """
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @abstractmethod
    def setup_models(self) -> None:
        """Load, configure, and wrap all models (FSDP/DDP).

        Must populate self.models dict with all named models.
        """
        ...

    @abstractmethod
    def setup_optimizers(self) -> None:
        """Create optimizers and LR schedulers for all trainable models.

        Must populate self.optimizers and self.schedulers dicts.
        """
        ...

    @abstractmethod
    def train_step(self, batch: Any) -> dict[str, dict[str, float]]:
        """Execute one training step.

        Args:
            batch: A batch from the dataloader (e.g., list of prompt strings).

        Returns:
            Dict of section_name -> {metric_name: value} for logging.
            Example: {"generator": {"loss_velocity": 0.5}, "fake_score": {"loss_fake": 0.3}}
        """
        ...

    @abstractmethod
    def _create_dataloader(self):
        """Create the grouped dataloader and its sampler.

        Returns:
            Tuple of (DataLoader, GroupedPromptSampler).
        """
        ...
