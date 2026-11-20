"""Train or resume SPM from a YAML recipe or saved run directory."""

import argparse

from src.config import TrainingConfig
from src.trainers.trainer import Trainer
from src.utils.cli import parse_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPM Training")
    parser.add_argument("config", type=str, help="Path to a YAML config or saved run directory")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key=value format, e.g., train.seed=123 objective.alpha=0.5",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = parse_overrides(args.override) if args.override else None
    config = TrainingConfig.from_yaml(args.config, overrides=overrides)

    Trainer(config).train()


if __name__ == "__main__":
    main()
