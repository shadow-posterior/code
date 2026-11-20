"""SPM configuration dataclasses and YAML loading."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from src.paths import portable_config_paths


def _apply_flat_overrides(d: dict, overrides: dict[str, Any]) -> dict:
    """Apply dot-notation overrides like {"train.seed": 123} to a nested dict."""
    for key, value in overrides.items():
        parts = key.split(".")
        if not all(parts):
            raise ValueError(f'invalid configuration override: {key}')
        target = d
        for part in parts[:-1]:
            if not isinstance(target, dict):
                raise ValueError(f'invalid configuration override path: {key}')
            target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(f'invalid configuration override path: {key}')
        target[parts[-1]] = value
    return d


def _dataclass_from_dict(cls, data: Any):
    """Recursively instantiate a dataclass, rejecting unknown schema fields."""
    if not isinstance(data, dict):
        return data
    field_names = {f.name for f in fields(cls)}
    field_types = {f.name: f.type for f in fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f'unknown {cls.__name__} configuration settings: {sorted(unknown)}')
    kwargs = {}
    for name in field_names:
        if name not in data:
            continue
        value = data[name]
        ftype = field_types[name]
        if isinstance(ftype, str):
            ftype = eval(ftype)
        if isinstance(value, dict) and hasattr(ftype, "__dataclass_fields__"):
            kwargs[name] = _dataclass_from_dict(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


@dataclass
class LoRAConfig:
    """Settings and initial weights for the trainable adapter."""

    rank: int = 32
    lora_alpha: int = 8
    target_modules: list = field(default_factory=lambda: [
        "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    ])
    init_lora_weights: str = "gaussian"
    load_path: str = ""


@dataclass
class ModelConfig:
    """Configuration for the SD3.5 Medium backbone."""

    pretrained_path: str = ""
    generator_init_path: str = ""
    fake_score_init_path: str = ""
    teacher_init_path: str = ""
    gradient_checkpointing: bool = True
    dtype: str = "bf16"
    fp32_master_weights: bool = True
    image_resolution: int = 1024
    generator_lora: LoRAConfig = field(default_factory=LoRAConfig)
    fake_score_lora: LoRAConfig = field(default_factory=LoRAConfig)
    fsdp_patch_embed_linear: bool = False


@dataclass
class SamplingConfig:
    """Four-step generation and the teacher query schedule."""

    denoising_step_list: list[int] = field(default_factory=lambda: [1000, 750, 500, 250])
    num_train_timesteps: int = 1000
    min_step: int = 20
    max_step: int = 980
    timestep_sampling: str = "uniform"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    teacher_guidance_scale: float = 4.5

    def __post_init__(self) -> None:
        steps = self.denoising_step_list
        if (type(self.num_train_timesteps) is not int
                or not isinstance(steps, list) or len(steps) != 4
                or any(isinstance(s, bool) or not isinstance(s, (int, float))
                       or not math.isfinite(s) or s <= 0 for s in steps)
                or steps[0] != self.num_train_timesteps
                or any(a <= b for a, b in zip(steps, steps[1:]))):
            raise ValueError('sampling.denoising_step_list must have four strictly decreasing positive levels '
                             'starting at num_train_timesteps')
        if (type(self.min_step) is not int or type(self.max_step) is not int
                or not 1 <= self.min_step <= self.max_step < self.num_train_timesteps - 1):
            raise ValueError('sampling.min_step/max_step must select interior query times')
        if self.timestep_sampling not in ('uniform', 'logit_normal'):
            raise ValueError('sampling.timestep_sampling must be uniform or logit_normal')
        values = (self.logit_mean, self.logit_std, self.teacher_guidance_scale)
        if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
            raise ValueError('sampling logit parameters and teacher guidance must be finite numbers')
        if self.logit_std <= 0 or self.teacher_guidance_scale < 1:
            raise ValueError('sampling.logit_std must be positive and teacher_guidance_scale at least 1')


@dataclass
class ObjectiveConfig:
    """Shadow Posterior Matching with RMS and MAE normalization."""
    reward_fn: dict[str, float] = field(default_factory=lambda: {'pickscore': 1.0})
    reward_batch_size: int = 4
    num_image_per_prompt: int = 24
    alpha: float = 1.0
    gamma: float = 1.0
    adv_clip_max: float = 5.0
    per_prompt_global_std: bool = True
    reward_source_weights: dict = field(default_factory=dict)
    source_stratified_sampling: bool = False

    def __post_init__(self):
        if type(self.per_prompt_global_std) is not bool:
            raise ValueError('objective.per_prompt_global_std must be a boolean')
        if type(self.num_image_per_prompt) is not int or self.num_image_per_prompt < 2:
            raise ValueError('num_image_per_prompt must be an integer >= 2')
        if type(self.reward_batch_size) is not int or self.reward_batch_size <= 0:
            raise ValueError('reward_batch_size must be a positive integer')
        if not isinstance(self.reward_fn, dict) or not self.reward_fn:
            raise ValueError('reward_fn must be a nonempty fixed weight mapping')
        for name, value in self.reward_fn.items():
            if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('reward_fn weights must be finite numbers')
        if not any(self.reward_fn.values()):
            raise ValueError('reward_fn needs at least one nonzero weight')
        for name in ('alpha', 'adv_clip_max'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a fixed finite positive number')
        if (isinstance(self.gamma, bool) or not isinstance(self.gamma, (int, float))
                or not math.isfinite(self.gamma) or self.gamma < 0):
            raise ValueError('gamma must be a fixed finite nonnegative number')
        from src.rewards.source_rewards import validate_source_weights
        validate_source_weights(self.reward_fn, self.reward_source_weights,
                                self.source_stratified_sampling, 'objective')
        if self.reward_source_weights and self.per_prompt_global_std:
            raise ValueError('objective.reward_source_weights normalizes each reward by its group std; '
                             'set objective.per_prompt_global_std=false')


@dataclass
class OptimizerConfig:
    """AdamW settings for a trainable network."""

    lr: float = 1e-6
    warmup_steps: int = 0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class FakeScoreOptimizerConfig(OptimizerConfig):
    warmup_steps: int = 500


@dataclass
class SolverConfig:
    """Combined solver configuration for all trainable components."""

    generator: OptimizerConfig = field(default_factory=OptimizerConfig)
    fake_score: FakeScoreOptimizerConfig = field(default_factory=FakeScoreOptimizerConfig)


@dataclass
class TrainConfig:
    """Budgets and intervals count completed student optimizer updates."""
    total_steps: int = 100000
    save_interval: int = 1000
    log_interval: int = 10
    seed: int = 42
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    autocast_dtype: str = "bf16"
    resume_from: str = ""
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_every: int = 1


@dataclass
class DistributedConfig:
    """Distributed training parameters.

    Supports FSDP and DDP. FSDP is recommended for large models (SD3.5 2B+).
    """

    strategy: str = "fsdp"
    fsdp_sharding: str = "full_shard"
    fsdp_cpu_offload_frozen: bool = False


@dataclass
class EvalConfig:
    """Evaluation configuration for periodic reward-based assessment.

    During training, the evaluator generates images from fixed prompts,
    computes reward scores, and logs results to SwanLab.
    """
    enabled: bool = True
    eval_interval: int = 5000
    eval_at_start: bool = True
    eval_prompt_source: str = "prompt_file"
    eval_prompt_path: str = "prompts.json"
    dataset: str = "pickscore"
    dataset_path: str = "dataset"
    reward_fn: dict = field(default_factory=lambda: {"pickscore": 1.0})
    reward_dataset_map: dict = field(default_factory=dict)
    benchmark_reward_fn: dict = field(default_factory=dict)
    eval_num_steps: int = 4
    eval_batch_size: int = 4
    num_media_images: int = 8
    eval_seed: int = 12345
    models: list = field(default_factory=list)


@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration.

    Uses SwanLab for experiment tracking with structured metric sections:
    - fake_score/*: Fake score network training metrics
    - generator/*: Generator training metrics
    - system/*: GPU memory, throughput, etc.
    """
    project: str = "SPM"
    run_name: str = ""
    workspace: str = ""
    api_key: str = field(default="", repr=False)
    tags: list = field(default_factory=list)
    enabled: bool = True


@dataclass
class TrainingConfig:
    """Root configuration for SPM training."""

    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    prompt_path: str = "prompts.json"
    output_dir: str = "./outputs"

    @classmethod
    def from_yaml(cls, path: str, overrides: dict[str, Any] | None = None) -> "TrainingConfig":
        """Load a strict recipe, or recognize a saved run's config.yaml.

        Args:
            path: Path to the YAML config file.
            overrides: Optional dict of dot-notation overrides, e.g.,
                {"train.seed": 123, "objective.alpha": 0.5}.

        Returns:
            Fully instantiated TrainingConfig.
        """
        return cls._load_yaml(path, overrides, snapshot=False)

    @classmethod
    def _load_yaml(cls, path: str, overrides: dict[str, Any] | None, *, snapshot: bool) -> "TrainingConfig":
        from src.paths import project_path
        from src.utils.run_state import resume_config_path
        path = Path(project_path(path))
        if path.is_dir():
            overrides = {'train.resume_from': str(path), **(overrides or {})}
            path = resume_config_path(path)
            snapshot = True
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a mapping")
        snapshot = snapshot or path.name == 'resume_config.yaml'
        checkpoint = path.resolve().parent
        if (path.name == 'config.yaml'
                and checkpoint.name.startswith('checkpoint-')
                and checkpoint.name.removeprefix('checkpoint-').isdigit()
                and (checkpoint / 'meta.pt').is_file()):
            overrides = {'train.resume_from': str(checkpoint), **(overrides or {})}
            snapshot = True
        resume = (overrides or {}).get('train.resume_from',
                    raw.get('train', {}).get('resume_from', ''))
        if resume and (not snapshot or 'train.resume_from' in (overrides or {})):
            saved = resume_config_path(Path(project_path(resume)))
            if saved.is_file() and saved.resolve() != path.resolve():
                effective = {'train.resume_from': str(project_path(resume)), **(overrides or {})}

                launch_key = raw.get('logging', {}).get('api_key')
                if launch_key:
                    effective.setdefault('logging.api_key', launch_key)
                return cls._load_yaml(str(saved), effective, snapshot=True)
        if overrides:
            raw = _apply_flat_overrides(raw, overrides)
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'unsupported configuration sections: {sorted(unknown)}')
        if 'objective' in raw and not isinstance(raw['objective'], dict):
            raise ValueError('objective configuration must be a mapping')
        return _dataclass_from_dict(cls, raw)

    def to_dict(self) -> dict:
        """Serialize config to a plain dict (for SwanLab logging, etc.)."""
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        """Save config to a YAML file for reproducibility."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = self.to_dict()

        data['logging']['api_key'] = ''
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(path) or '.',
                    prefix='.config-', suffix='.yaml', delete=False) as handle:
                temporary = Path(handle.name)
                yaml.dump(portable_config_paths(data), handle, default_flow_style=False, sort_keys=False)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
