"""Strict configuration, provenance and atomic JSON helpers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def config(path: str | Path) -> dict:
    """Relative paths are relative to the benchmark repo, not the caller's cwd."""
    path = Path(path).resolve()
    cfg = yaml.safe_load(os.path.expandvars(path.read_text()))
    root = path.parent.parent
    for key in ['hm3d_root', 'dataset_config', 'output', 'audio_manifest', 'category_config']:
        value = Path(cfg[key])
        cfg[key] = str(value if value.is_absolute() else root / value)
    if cfg.get('source_plan'):
        value = Path(cfg['source_plan'])
        cfg['source_plan'] = str(value if value.is_absolute() else root / value)
    cfg['config_path'] = str(path)
    cfg['config_sha256'] = sha256(path)
    return cfg
