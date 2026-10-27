"""Complete, resumable training-state checkpoints for SAID."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


TRAINING_CHECKPOINT_SCHEMA = "said-training-state-v3"
LEGACY_TRAINING_CHECKPOINT_SCHEMA = "said-training-state-v2"


@dataclass(frozen=True)
class TrainingCheckpointReport:
    """Result of saving or restoring a complete training state."""

    path: Path
    step: int
    class_feature_encoder: str | None
    config_sha256: str
    restored_random_state: bool
    run_state: dict[str, float | int | None]


def _module_device(module: nn.Module) -> torch.device:
    value = next(module.parameters(), None)
    if value is None:
        value = next(module.buffers(), None)
    return torch.device("cpu") if value is None else value.device


def _rng_state(
    data_generator: torch.Generator | None,
    *,
    cuda_device: torch.device | None,
) -> dict[str, Any]:
    python_version, python_state, python_gauss = random.getstate()
    (
        numpy_algorithm,
        numpy_state,
        numpy_position,
        numpy_has_gauss,
        numpy_cached_gaussian,
    ) = np.random.get_state()
    state: dict[str, Any] = {
        "python": {
            "version": int(python_version),
            "state": torch.tensor(python_state, dtype=torch.int64),
            "gauss_next": python_gauss,
        },
        "numpy": {
            "algorithm": str(numpy_algorithm),
            "state": torch.from_numpy(
                np.asarray(numpy_state, dtype=np.int64)
            ),
            "position": int(numpy_position),
            "has_gauss": int(numpy_has_gauss),
            "cached_gaussian": float(numpy_cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(cuda_device)
            if cuda_device is not None
            else None
        ),
        "data_generator": (
            data_generator.get_state() if data_generator is not None else None
        ),
    }
    return state


def _restore_rng_state(
    state: Mapping[str, Any],
    data_generator: torch.Generator | None,
    *,
    cuda_device: torch.device | None,
) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "data_generator"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"training checkpoint RNG state is missing {sorted(missing)}")
    python_state = state["python"]
    numpy_state = state["numpy"]
    if not isinstance(python_state, Mapping) or not isinstance(
        numpy_state, Mapping
    ):
        raise ValueError("training checkpoint contains an unsafe RNG schema")
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(value) for value in python_state["state"].tolist()),
            python_state["gauss_next"],
        )
    )
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            numpy_state["state"].numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if cuda_device is None or not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA random states but CUDA is unavailable"
            )
        torch.cuda.set_rng_state(cuda_state, cuda_device)
    saved_data_state = state["data_generator"]
    if saved_data_state is not None:
        if data_generator is None:
            raise ValueError(
                "checkpoint contains a data-generator state; supply data_generator"
            )
        data_generator.set_state(saved_data_state)


def _configuration_sha256(configuration: bytes | str) -> str:
    value = configuration.encode("utf-8") if isinstance(configuration, str) else configuration
    return hashlib.sha256(value).hexdigest()


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    class_feature_encoder: str | None,
    configuration: bytes | str,
    data_generator: torch.Generator | None = None,
    run_state: Mapping[str, float | int | None] | None = None,
    replace: bool = False,
) -> TrainingCheckpointReport:
    """Atomically save all state required to continue an interrupted run."""

    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.exists() and not replace:
        raise FileExistsError(f"training checkpoint already exists: {checkpoint_path}")
    if int(step) < 0:
        raise ValueError("step must be non-negative")
    if class_feature_encoder not in {"passt", "audiomae", None}:
        raise ValueError("class_feature_encoder must be 'passt', 'audiomae', or null")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config_sha256 = _configuration_sha256(configuration)
    serialized_run_state = dict(run_state or {})
    if any(
        not isinstance(key, str)
        or not isinstance(value, (int, float, type(None)))
        or isinstance(value, bool)
        for key, value in serialized_run_state.items()
    ):
        raise ValueError("run_state must map strings to numeric scalars or null")
    payload = {
        "schema": TRAINING_CHECKPOINT_SCHEMA,
        "step": int(step),
        "class_feature_encoder": class_feature_encoder,
        "config_sha256": config_sha256,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": _rng_state(
            data_generator,
            cuda_device=(
                _module_device(model)
                if _module_device(model).type == "cuda"
                else None
            ),
        ),
        "run_state": serialized_run_state,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        dir=checkpoint_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return TrainingCheckpointReport(
        path=checkpoint_path,
        step=int(step),
        class_feature_encoder=class_feature_encoder,
        config_sha256=config_sha256,
        restored_random_state=False,
        run_state=serialized_run_state,
    )


def save_model_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    replace: bool = False,
) -> Path:
    """Atomically save a flat, inference-ready public-runtime state dict."""

    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.exists() and not replace:
        raise FileExistsError(f"model checkpoint already exists: {checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        dir=checkpoint_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(model.state_dict(), temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return checkpoint_path


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    class_feature_encoder: str | None,
    configuration: bytes | str,
    data_generator: torch.Generator | None = None,
    restore_random_state: bool = True,
) -> TrainingCheckpointReport:
    """Strictly restore a complete training state and its provenance gate."""

    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("training checkpoint must contain a mapping")
    required = {
        "schema",
        "step",
        "class_feature_encoder",
        "config_sha256",
        "model",
        "ema_model",
        "optimizer",
        "scheduler",
        "rng_state",
    }
    schema = payload.get("schema")
    if schema == TRAINING_CHECKPOINT_SCHEMA:
        required.add("run_state")
    missing = required - set(payload)
    unknown = set(payload) - required
    if missing or unknown:
        raise ValueError(
            f"training checkpoint fields mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if schema not in {
        TRAINING_CHECKPOINT_SCHEMA,
        LEGACY_TRAINING_CHECKPOINT_SCHEMA,
    }:
        raise ValueError(f"unsupported training checkpoint schema: {payload['schema']!r}")
    if payload["class_feature_encoder"] != class_feature_encoder:
        raise ValueError(
            "training checkpoint Class Feature Encoder does not match the run"
        )
    config_sha256 = _configuration_sha256(configuration)
    if payload["config_sha256"] != config_sha256:
        raise ValueError(
            "training checkpoint configuration fingerprint does not match"
        )
    model.load_state_dict(payload["model"], strict=True)
    ema_model.load_state_dict(payload["ema_model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    saved_scheduler = payload["scheduler"]
    if (scheduler is None) != (saved_scheduler is None):
        raise ValueError("training checkpoint scheduler presence does not match")
    if scheduler is not None:
        scheduler.load_state_dict(saved_scheduler)
    if restore_random_state:
        model_device = _module_device(model)
        _restore_rng_state(
            payload["rng_state"],
            data_generator,
            cuda_device=model_device if model_device.type == "cuda" else None,
        )
    run_state = payload.get("run_state", {})
    if not isinstance(run_state, Mapping) or any(
        not isinstance(key, str)
        or not isinstance(value, (int, float, type(None)))
        or isinstance(value, bool)
        for key, value in run_state.items()
    ):
        raise ValueError("training checkpoint has an unsafe run_state")
    return TrainingCheckpointReport(
        path=checkpoint_path,
        step=int(payload["step"]),
        class_feature_encoder=class_feature_encoder,
        config_sha256=config_sha256,
        restored_random_state=bool(restore_random_state),
        run_state=dict(run_state),
    )
