"""Load prompts and reward metadata from JSON, JSONL, or text."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_DATASET_PROMPT_FILES = (
    'train_metadata.jsonl', 'train.txt', 'train.jsonl', 'prompts.json', 'prompts.txt',
)


class PromptDataset(Dataset):
    def __init__(self, path: str):
        path = Path(path)
        if path.is_dir():
            directory = path
            path = next((directory / name for name in _DATASET_PROMPT_FILES
                         if (directory / name).is_file()), None)
            if path is None:
                raise FileNotFoundError(f'No prompt file found in {directory}. Expected one of {_DATASET_PROMPT_FILES}')

        with path.open(encoding='utf-8') as handle:
            match path.suffix.lower():
                case '.json':
                    rows = json.load(handle)
                    if not isinstance(rows, list) or not rows:
                        raise ValueError(f'JSON file must contain a non-empty list: {path}')
                case '.jsonl':
                    rows = [json.loads(line) for line in handle if line.strip()]
                case '.txt' | '.text':
                    rows = [line.strip() for line in handle if line.strip()]
                case suffix:
                    raise ValueError(f'Unsupported prompt file format: {suffix}. Use .json, .jsonl, or .txt')

        self.prompts = []
        self.metadatas = []
        for row in rows:
            if isinstance(row, str):
                row = {'prompt': row}
            if not isinstance(row, dict):
                raise ValueError(f'Prompt entries must be strings or mappings: {path}')
            prompt = row.get('prompt', row.get('text'))
            if not isinstance(prompt, str):
                raise ValueError(f'Each mapping must contain a string prompt or text field: {path}')
            self.prompts.append(prompt)
            self.metadatas.append(row)

        logger.info('Loaded %s prompts from %s', len(self.prompts), path)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict:
        return {'prompt': self.prompts[index], 'metadata': self.metadatas[index]}
