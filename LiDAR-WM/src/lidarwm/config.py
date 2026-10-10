from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    return os.path.expandvars(value) if isinstance(value, str) else value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with Path(path).open() as handle:
        cfg = _expand_environment(yaml.safe_load(handle))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        value = _expand_environment(yaml.safe_load(raw))
        node = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value
    return cfg


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(copy.deepcopy(config), handle, sort_keys=False)
