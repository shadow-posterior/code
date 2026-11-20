"""Training metrics, image logging and timing for SPM."""

from __future__ import annotations

import logging
import copy
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch

from src.parallel import is_main_process
from src.utils.reward_logging import with_training_reward_section
from src.utils.run_state import load_swanlab_run, save_swanlab_run, swanlab_api_keys

logger = logging.getLogger(__name__)


class SwanlabLogger:
    """Structured SwanLab logger with section-based metric organization.

    All metrics are logged with section prefixes (e.g., "generator/loss_velocity",
    "profile/gpu_mem_gb") to keep the SwanLab dashboard organized.

    Args:
        project: SwanLab project name.
        run_name: SwanLab run name.
        workspace: SwanLab workspace (team/user).
        config: Config dict to log as SwanLab config.
        tags: List of tags for the run.
        enabled: Whether to actually log (False on worker ranks).
    """

    def __init__(
        self,
        project: str = "SPM",
        run_name: str = "",
        workspace: str = "",
        config: dict | None = None,
        tags: list[str] | None = None,
        enabled: bool = True,
        output_dir: str = "",
        resume: bool = False,
    ):
        mode = os.environ.get('SWANLAB_MODE', 'online')
        self.enabled = enabled and is_main_process() and mode != 'disabled'
        self._run = None
        auth_config = config
        config = copy.deepcopy(config)
        if config is not None and 'logging' in config:
            config['logging'].pop('api_key', None)

        if self.enabled:
            import swanlab

            self._swanlab = swanlab
            identity = None
            init_resume = {}
            if resume:
                identity = load_swanlab_run(output_dir, project=project,
                                            workspace=workspace, run_name=run_name)
                project, workspace, run_name = (identity['project'], identity['workspace'], identity['run_name'])
                init_resume = {'id': identity['run_id'],
                               'resume': 'must' if identity.get('cloud_created', True) else 'allow'}
            log_dir = os.path.join(output_dir, "swanlog") if output_dir else None
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            if mode == 'online':
                credentials = swanlab_api_keys(auth_config)
                started = time.monotonic()
                authenticated = False
                for key in [*credentials, None]:
                    try:
                        authenticated = swanlab.login(api_key=key, relogin=bool(key), save=False, timeout=2)
                        if authenticated:
                            break
                    except Exception:
                        pass

                    if time.monotonic() - started >= 10:
                        break
                if not authenticated:
                    mode = 'offline'
                    logger.warning('SwanLab login unavailable; continuing offline with the same run ID.')
            from swanlab import Settings
            self._run = swanlab.init(
                project=project,
                name=run_name or None,
                workspace=workspace or None,
                config=config,
                tags=tags,

                logdir=log_dir,

                mode=mode,
                settings=Settings(interactive=False),
                **init_resume,
            )
            if identity is not None and self._run.id != identity['run_id']:
                self._run.finish()
                raise RuntimeError('SwanLab returned a different run ID during resume')
            if output_dir:
                cloud_created = (bool(identity and identity.get('cloud_created', True))
                                 or getattr(self._run, 'mode', mode) == 'online')
                identity = {'run_id': self._run.id, 'project': project,
                            'workspace': workspace, 'run_name': run_name,
                            'cloud_created': cloud_created}

                try:
                    owner, cloud_project, _ = self._run.path.strip('/').split('/')
                    identity.update(workspace=owner.lstrip('@'), project=cloud_project)
                except (AttributeError, ValueError):
                    pass
                save_swanlab_run(output_dir, identity)
            logger.info(
                "SwanLab initialized (project=%s, run_name=%s)", project, run_name
            )

    def log(self, metrics: dict[str, Any], step: int, section: str = "") -> None:
        """Log metrics to SwanLab.

        Args:
            metrics: Dict of metric_name -> value.
            step: Global training step.
            section: Optional section prefix (e.g., "generator", "fake_score").
                If provided, metrics are logged as "section/metric_name".
        """
        if not self.enabled:
            return

        if section:
            metrics = {f"{section}/{k}": v for k, v in metrics.items()}

        self._swanlab.log(metrics, step=step)

    def log_multi_section(self, sections: dict[str, dict[str, Any]], step: int) -> None:
        """Log metrics from multiple sections in a single SwanLab call.

        Args:
            sections: Dict of section_name -> {metric_name: value}.
            step: Global training step.

        Example:
            logger.log_multi_section({
                "generator": {"loss_velocity": 0.5, "model_grad_norm": 1.2},
                "fake_score": {"loss_fake": 0.3},
                "profile": {"gpu_mem_gb": 24.5},
            }, step=100)
        """
        if not self.enabled:
            return

        sections = with_training_reward_section(sections)
        flat = {}
        for section, metrics in sections.items():
            for k, v in metrics.items():
                flat[f"{section}/{k}"] = v
        self._swanlab.log(flat, step=step)

    def log_images(
        self,
        images: list,
        captions: list[str],
        step: int,
        key: str = "media/eval",
    ) -> None:
        """Log images to SwanLab as a media panel.

        Args:
            images: List of PIL.Image instances.
            captions: List of caption strings (one per image).
            step: Global training step.
            key: SwanLab log key.
        """
        if not self.enabled:
            return
        swanlab = self._swanlab
        swanlab_images = [
            swanlab.Image(img, caption=cap) for img, cap in zip(images, captions)
        ]
        self._swanlab.log({key: swanlab_images}, step=step)

    def finish(self) -> None:
        """Finish the SwanLab run."""
        if self.enabled and self._run is not None:
            self._run.finish()


def synchronize_for_profiling(enabled: bool | None = None) -> None:
    """Drain CUDA only for explicitly synchronized wall-clock profiling."""
    if enabled is None:
        enabled = os.environ.get('SYNC_CUDA_TIMING', '0') == '1'
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Constant-space host wall-clock timing for training stages.

    CUDA work is asynchronous, so default measurements include host dispatch
    and naturally occurring waits rather than isolated GPU execution time.
    Set SYNC_CUDA_TIMING=1 (or synchronize_cuda=True) for synchronized profiling.

    Usage:
        timer = Timer()
        with timer.measure("forward_pass"):
            output = model(input)
        print(timer.get_metrics())
    """

    def __init__(self, synchronize_cuda: bool | None = None):
        self.synchronize_cuda = synchronize_cuda
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._last: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        """Context manager to measure wall-clock time of a code block.

        Args:
            name: Name of the operation being timed.
        """
        synchronize_for_profiling(self.synchronize_cuda)
        start = time.perf_counter()
        yield
        synchronize_for_profiling(self.synchronize_cuda)
        elapsed = time.perf_counter() - start
        self._last[name] = elapsed
        self._totals[name] += elapsed
        self._counts[name] += 1

    def get_last(self, name: str) -> float:
        """Get the most recent measurement for a named operation."""
        return self._last.get(name, 0.0)

    def get_average(self, name: str) -> float:
        """Get the average time across all measurements for a named operation."""
        count = self._counts.get(name, 0)
        return self._totals[name] / count if count else 0.0

    def get_metrics(self) -> dict[str, float]:
        """Get average times for all tracked operations, suitable for logging."""
        return {f"time_{k}_s": self.get_average(k) for k in self._counts}

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self._last.clear()
        self._totals.clear()
        self._counts.clear()


def get_gpu_memory_gb() -> float:
    """Peak PyTorch allocated GiB on this rank since the last peak reset.

    This excludes reserved-but-unused allocator blocks and other CUDA users;
    it is neither current device usage nor a peak aggregated across ranks.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format.

    Only the main process logs at INFO level; other ranks log at WARNING.
    Forces reconfiguration even if basicConfig was already called (e.g., by torchrun).
    """
    fmt = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    target_level = level if is_main_process() else logging.WARNING

    root = logging.getLogger()
    root.setLevel(target_level)

    for handler in root.handlers[:]:
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setLevel(target_level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
