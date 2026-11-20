"""SPM training for few-step text-to-image models."""

from src.config import LoRAConfig, ModelConfig, TrainingConfig, ObjectiveConfig
from src.trainers.trainer import Trainer

__version__ = "0.1.0"

__all__ = [
    "LoRAConfig",
    "ModelConfig",
    "TrainingConfig",
    "ObjectiveConfig",
    "Trainer",
    "__version__",
]
