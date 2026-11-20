"""Shared parsing for training and inference command-line overrides."""

import yaml


def parse_overrides(override_list: list[str]) -> dict:
    """Parse key=value pairs, including scientific notation and YAML containers."""
    overrides = {}
    for item in override_list:
        if '=' not in item:
            raise ValueError(f'Invalid override format: {item}. Expected key=value.')
        key, value = item.split('=', 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        else:
            try:
                value = yaml.safe_load(value) if value else value
            except yaml.YAMLError:
                pass
        overrides[key] = value
    return overrides
