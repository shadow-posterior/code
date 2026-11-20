"""Paths derived from this checkout, so the complete directory can be moved."""
from pathlib import Path
from datetime import datetime, timedelta
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUN_TIMESTAMP = re.compile(r"^\d{8}_\d{6}(?:_|$)")


def project_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def local_model_path(value: str | Path) -> str:
    """Resolve a local model; hub IDs map to weights/<repository name>."""
    path = Path(project_path(value))
    if not path.exists() and not Path(value).is_absolute() and not str(value).startswith(("weights/", "./weights/")):
        path = WEIGHTS_DIR / Path(value).name
    if not path.exists():
        raise FileNotFoundError(f"Local model is missing: {path}. Copy the model into weights/ first.")
    return str(path)


def create_run_directory(output_root: str | Path) -> str:
    """Reserve outputs/configuration/timestamp without nested timestamp folders."""
    base = Path(project_path(output_root))
    if RUN_TIMESTAMP.match(base.name):
        base = base.parent
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().replace(microsecond=0)
    while True:
        run = base / stamp.strftime("%Y%m%d_%H%M%S")
        try:
            run.mkdir()
        except FileExistsError:
            stamp += timedelta(seconds=1)
        else:
            return str(run)


def portable_config_paths(value):
    """Serialize paths inside this checkout relative to the checkout."""
    if isinstance(value, dict):
        return {key: portable_config_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_config_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(portable_config_paths(item) for item in value)
    if isinstance(value, str) and Path(value).is_absolute():
        resolved = project_path(value)
        if resolved.startswith(str(PROJECT_ROOT) + "/"):
            return str(Path(resolved).relative_to(PROJECT_ROOT))
    return value
