"""Locate a saved training run and recover its experiment-tracking identity."""
from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

from src.paths import project_path


_CHECKPOINT = re.compile(r'^checkpoint-(\d+)$')
SWANLAB_STATE = 'swanlab_run.json'
CHECKPOINT_STATE = 'checkpoint_state.json'


def resume_run_directory(source: str | Path) -> Path:
    """Return the run folder for a run or checkpoint directory."""
    path = Path(project_path(source)).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'Resume source not found: {path}')
    return path.parent if _CHECKPOINT.fullmatch(path.name) else path


def resume_config_path(source: str | Path) -> Path:
    """Resolve the effective configuration for a checkpoint or run."""
    directory = Path(source)
    if _CHECKPOINT.fullmatch(directory.name):
        return directory / 'config.yaml'
    latest = directory / 'resume_config.yaml'
    return latest if latest.is_file() else directory / 'config.yaml'


def _serialized_checkpoint(path: Path) -> dict | None:
    import torch
    try:
        if not path.is_file() or not path.stat().st_size:
            return None
        with path.open('rb') as handle:
            header = handle.read(2)
        if header == b'PK' and not zipfile.is_zipfile(path):
            return None

        state = torch.load(path, map_location='meta', weights_only=False)
        return state if isinstance(state, dict) else None
    except Exception:
        return None


def write_checkpoint_manifest(directory: str | Path, step: int, files, *, complete: bool) -> None:
    """Publish completion only after every rank's state and effective config exist."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(set(files))
    sizes = {name: (directory / name).stat().st_size for name in names} if complete else {}
    if complete and any(size == 0 for size in sizes.values()):
        raise ValueError('Cannot commit a checkpoint with empty state files')
    _atomic_json(directory / CHECKPOINT_STATE,
                 {'version': 1, 'step': step, 'complete': complete, 'files': sizes})


def checkpoint_is_complete(source: str | Path, required_files=()) -> bool:
    """Validate a split checkpoint before training resumes.

    Required state files must be readable and match the checkpoint manifest when present.
    """
    path = Path(source)
    try:
        required = set(required_files)
        if path.is_dir() and (path / CHECKPOINT_STATE).exists():
            manifest = json.loads((path / CHECKPOINT_STATE).read_text())
            files = manifest.get('files')
            match = _CHECKPOINT.fullmatch(path.name)
            if (manifest.get('version') != 1 or manifest.get('complete') is not True
                    or not isinstance(files, dict) or not files
                    or match and manifest.get('step') != int(match[1])):
                return False
            for name, size in files.items():
                if (Path(name).name != name or not isinstance(size, int) or size <= 0
                        or not (path / name).is_file() or (path / name).stat().st_size != size):
                    return False
            required.update(files)
        if path.is_dir() and (path / 'meta.pt').is_file():
            if _serialized_checkpoint(path / 'meta.pt') is None:
                return False
            for name in required:
                if name == 'meta.pt':
                    continue
                file = path / name
                if name.endswith('.pt'):
                    if _serialized_checkpoint(file) is None:
                        return False
                elif not file.is_file() or not file.stat().st_size:
                    return False
        else:
            return False
        return True
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def select_resume_checkpoint(source: str | Path, required_files=()) -> str:
    """Choose by numeric step, skipping incomplete split-format checkpoints."""
    path = Path(project_path(source)).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'Resume source not found: {path}')
    if _CHECKPOINT.fullmatch(path.name):
        if (path / CHECKPOINT_STATE).exists() and not checkpoint_is_complete(path, required_files):
            raise FileNotFoundError(f'Incomplete checkpoint: {path}')
        return str(path)

    candidates = []
    for child in path.iterdir():
        match = _CHECKPOINT.fullmatch(child.name)
        if match and checkpoint_is_complete(child, required_files):
            candidates.append((int(match[1]), child))
    if candidates:
        return str(max(candidates, key=lambda item: item[0])[1])
    if checkpoint_is_complete(path, required_files):
        return str(path)
    raise FileNotFoundError(f'No complete training checkpoint in {path}')


def load_swanlab_run(directory: str | Path, *, project='', workspace='', run_name='') -> dict:
    directory = Path(directory)
    state_path = directory / SWANLAB_STATE
    requested_id = os.environ.get('SWANLAB_RUN_ID', '').strip()
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if not isinstance(state, dict) or state.get('version') != 1 or not state.get('run_id'):
            raise ValueError(f'Invalid SwanLab run identity: {state_path}')
        if requested_id and requested_id != state['run_id']:
            raise ValueError('SWANLAB_RUN_ID differs from the saved training run')
        return state

    if not requested_id:
        raise ValueError(f'Missing SwanLab run identity: {state_path}')
    return {'version': 1, 'run_id': requested_id, 'project': project,
            'workspace': workspace, 'run_name': run_name, 'cloud_created': True}


def save_swanlab_run(directory: str | Path, identity: dict) -> None:
    """Persist only identity fields, without authentication credentials."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {key: identity.get(key, '') for key in ('run_id', 'project', 'workspace', 'run_name')}
    payload['version'] = 1
    payload['cloud_created'] = identity.get('cloud_created', True)
    _atomic_json(directory / SWANLAB_STATE, payload)


def _atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=f'.{path.stem}-',
                                     suffix='.tmp', delete=False) as handle:
        json.dump(payload, handle)
        temporary = Path(handle.name)
    temporary.replace(path)


def swanlab_api_keys(config: dict | None) -> list[str]:
    """Read credentials for the current launch."""
    logging = (config or {}).get('logging', {})
    values = [os.environ.get('SWANLAB_API_KEY', ''), logging.get('api_key', '')]
    return list(dict.fromkeys(value.strip() for value in values if isinstance(value, str) and value.strip()))
