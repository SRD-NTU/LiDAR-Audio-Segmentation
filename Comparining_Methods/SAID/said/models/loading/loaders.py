"""Strict loader for fingerprinted SAID paper checkpoints."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .compatibility import (
    AUDIO2SPH_PAPER_CHECKPOINT,
    PAPER_CHECKPOINTS,
    CheckpointCompatibilityError,
    normalize_audio2sph_pretraining_state_dict,
    normalize_submitted_state_dict,
    submitted_class_feature_encoder_state_dict,
    validate_checkpoint_file,
    validate_audio2sph_checkpoint_summary,
    validate_state_dict_compatibility,
    validate_submitted_checkpoint_summary,
)


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Auditable result of strict checkpoint loading."""

    path: Path
    model_name: str
    class_feature_encoder: str
    source_schema_id: str
    runtime_schema_id: str
    weights_kind: str
    size_bytes: int
    sha256: str
    source_key_count: int
    loaded_key_count: int
    compatibility_only_key_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def checkpoint_sha256(path: str | Path) -> str:
    """Return the SHA256 identity of a local checkpoint file."""

    path = Path(path).expanduser().resolve()
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_submitted_state_dict(path: str | Path) -> Mapping[str, torch.Tensor]:
    """Read the paper-checkpoint format and reject nested training payloads."""

    checkpoint_path = Path(path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except Exception as error:
        raise CheckpointCompatibilityError(
            f"failed to read checkpoint {checkpoint_path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise CheckpointCompatibilityError(
            f"checkpoint payload must be a flat state dict, got {type(payload).__name__}"
        )
    training_state_keys = {
        "model",
        "state_dict",
        "model_state_dict",
        "ema",
        "ema_state_dict",
        "optimizer",
        "optimizer_state_dict",
        "scheduler",
        "scaler",
        "step",
        "rng_state",
    }
    if any(key in training_state_keys for key in payload):
        raise CheckpointCompatibilityError(
            "This file contains a training-state payload. Use "
            "`said train --resume PATH` to restore training, or provide a flat "
            "model state dict through `load_said_ckpt` or `load_audio2sph_ckpt`."
        )
    for key, value in payload.items():
        if not isinstance(key, str):
            raise CheckpointCompatibilityError(
                f"checkpoint keys must be strings, got {type(key).__name__}"
            )
        if not isinstance(value, torch.Tensor):
            raise CheckpointCompatibilityError(
                f"checkpoint value is not a tensor: {key}"
            )
    return payload


def load_paper_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    class_feature_encoder: str,
) -> CheckpointLoadReport:
    """Verify and strictly load one of the checkpoints reported in the paper."""

    validate_checkpoint_file(path, class_feature_encoder)
    submitted_state = _read_submitted_state_dict(path)
    validate_submitted_checkpoint_summary(submitted_state, class_feature_encoder)
    runtime_state, compatibility_only = normalize_submitted_state_dict(
        submitted_state, class_feature_encoder
    )
    validate_state_dict_compatibility(runtime_state, model.state_dict())
    result = model.load_state_dict(runtime_state, strict=True)
    missing = tuple(result.missing_keys)
    unexpected = tuple(result.unexpected_keys)
    if missing or unexpected:
        raise CheckpointCompatibilityError(
            f"strict load returned missing={missing}, unexpected={unexpected}"
        )
    spec = PAPER_CHECKPOINTS[class_feature_encoder]
    return CheckpointLoadReport(
        path=Path(path).resolve(),
        model_name=spec.model_name,
        class_feature_encoder=spec.class_feature_encoder,
        source_schema_id="said-paper-flat-ema-v1",
        runtime_schema_id="said-runtime-v1",
        weights_kind="ema",
        size_bytes=spec.size_bytes,
        sha256=spec.sha256,
        source_key_count=len(submitted_state),
        loaded_key_count=len(runtime_state),
        compatibility_only_key_count=len(compatibility_only),
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def load_said_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    class_feature_encoder: str,
) -> CheckpointLoadReport:
    """Load a paper checkpoint or an exact public-runtime state dictionary.

    Fingerprinted paper files pass through the submitted-schema compatibility
    boundary. Checkpoints emitted by the public trainer already use the
    paper-aligned runtime names and therefore require an exact key, shape, and
    dtype match without key rewriting.
    """

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise CheckpointCompatibilityError(
            f"checkpoint file not found or not a regular file: {checkpoint_path}"
        )
    size = checkpoint_path.stat().st_size
    digest = checkpoint_sha256(checkpoint_path)
    paper = PAPER_CHECKPOINTS.get(class_feature_encoder)
    if paper is not None and size == paper.size_bytes and digest == paper.sha256:
        return load_paper_checkpoint(
            model,
            checkpoint_path,
            class_feature_encoder=class_feature_encoder,
        )

    runtime_state = _read_submitted_state_dict(checkpoint_path)
    validate_state_dict_compatibility(runtime_state, model.state_dict())
    result = model.load_state_dict(runtime_state, strict=True)
    missing = tuple(result.missing_keys)
    unexpected = tuple(result.unexpected_keys)
    if missing or unexpected:
        raise CheckpointCompatibilityError(
            f"strict load returned missing={missing}, unexpected={unexpected}"
        )
    return CheckpointLoadReport(
        path=checkpoint_path,
        model_name=f"SAID ({class_feature_encoder})",
        class_feature_encoder=class_feature_encoder,
        source_schema_id="said-runtime-flat-v1",
        runtime_schema_id="said-runtime-v1",
        weights_kind="model",
        size_bytes=size,
        sha256=digest,
        source_key_count=len(runtime_state),
        loaded_key_count=len(runtime_state),
        compatibility_only_key_count=0,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def load_audio2sph_pretraining_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    class_feature_encoder: str | None = None,
) -> CheckpointLoadReport:
    """Load a paper, historical-pretraining, or public Audio2Sph checkpoint."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise CheckpointCompatibilityError(
            f"checkpoint file not found or not a regular file: {checkpoint_path}"
        )
    size = checkpoint_path.stat().st_size
    digest = checkpoint_sha256(checkpoint_path)
    paper = PAPER_CHECKPOINTS.get(str(class_feature_encoder))
    audio2sph_paper = AUDIO2SPH_PAPER_CHECKPOINT
    source_state = _read_submitted_state_dict(checkpoint_path)
    compatibility_only: tuple[str, ...] = ()
    if (
        size == audio2sph_paper.size_bytes
        and digest == audio2sph_paper.sha256
    ):
        validate_audio2sph_checkpoint_summary(source_state)
        runtime_state, compatibility_only = (
            normalize_audio2sph_pretraining_state_dict(source_state)
        )
        source_schema = "audio2sph-paper-flat-ema-v1"
        weights_kind = "ema"
        encoder_name = "none"
        model_name = audio2sph_paper.model_name
    elif paper is not None and size == paper.size_bytes and digest == paper.sha256:
        validate_submitted_checkpoint_summary(
            source_state, str(class_feature_encoder)
        )
        runtime_state, compatibility_only = (
            normalize_audio2sph_pretraining_state_dict(source_state)
        )
        source_schema = "said-paper-flat-ema-v1"
        weights_kind = "ema"
        encoder_name = paper.class_feature_encoder
        model_name = "Audio2Sph + Panoramic Decoder"
    elif set(source_state) == set(model.state_dict()):
        runtime_state = dict(source_state)
        source_schema = "audio2sph-pretraining-runtime-flat-v1"
        weights_kind = "model"
        encoder_name = "none"
        model_name = "Audio2Sph + Panoramic Decoder"
    else:
        try:
            runtime_state, compatibility_only = (
                normalize_audio2sph_pretraining_state_dict(source_state)
            )
        except CheckpointCompatibilityError as error:
            raise CheckpointCompatibilityError(
                "Audio2Sph checkpoint is neither a public-runtime state dict "
                "nor the historical pretraining schema"
            ) from error
        source_schema = "audio2sph-historical-flat-v1"
        weights_kind = "ema"
        encoder_name = "none"
        model_name = "Audio2Sph + Panoramic Decoder"
    validate_state_dict_compatibility(runtime_state, model.state_dict())
    result = model.load_state_dict(runtime_state, strict=True)
    missing = tuple(result.missing_keys)
    unexpected = tuple(result.unexpected_keys)
    if missing or unexpected:
        raise CheckpointCompatibilityError(
            f"strict load returned missing={missing}, unexpected={unexpected}"
        )
    return CheckpointLoadReport(
        path=checkpoint_path,
        model_name=model_name,
        class_feature_encoder=encoder_name,
        source_schema_id=source_schema,
        runtime_schema_id="audio2sph-pretraining-runtime-v1",
        weights_kind=weights_kind,
        size_bytes=size,
        sha256=digest,
        source_key_count=len(source_state),
        loaded_key_count=len(runtime_state),
        compatibility_only_key_count=len(compatibility_only),
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def load_audio2sph_component_checkpoint(
    model: nn.Module,
    path: str | Path,
) -> CheckpointLoadReport:
    """Strictly initialize complete SAID's Audio2Sph component.

    Accepted files are a flat Audio2Sph state dictionary, a public
    ``Audio2SphPretrainingModel`` state dictionary, or the historical flat
    pretraining state dictionary.  Panoramic Decoder tensors are verified and
    then excluded because complete SAID does not contain that decoder.
    """

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise CheckpointCompatibilityError(
            f"checkpoint file not found or not a regular file: {checkpoint_path}"
        )
    source_state = _read_submitted_state_dict(checkpoint_path)
    expected = model.state_dict()
    compatibility_only: tuple[str, ...] = ()
    if set(source_state) == set(expected):
        runtime_state = dict(source_state)
        source_schema = "audio2sph-component-runtime-flat-v1"
    elif any(key.startswith("audio2sph.") for key in source_state):
        runtime_state = {
            key.removeprefix("audio2sph."): value
            for key, value in source_state.items()
            if key.startswith("audio2sph.")
        }
        compatibility_only = tuple(
            key for key in source_state if not key.startswith("audio2sph.")
        )
        source_schema = "audio2sph-pretraining-runtime-flat-v1"
    else:
        try:
            pretraining_state, ignored = (
                normalize_audio2sph_pretraining_state_dict(source_state)
            )
        except CheckpointCompatibilityError as error:
            raise CheckpointCompatibilityError(
                "unrecognized Audio2Sph component checkpoint schema"
            ) from error
        runtime_state = {
            key.removeprefix("audio2sph."): value
            for key, value in pretraining_state.items()
            if key.startswith("audio2sph.")
        }
        compatibility_only = tuple(
            list(ignored)
            + [
                key
                for key in pretraining_state
                if not key.startswith("audio2sph.")
            ]
        )
        source_schema = "audio2sph-historical-flat-v1"
    validate_state_dict_compatibility(runtime_state, expected)
    result = model.load_state_dict(runtime_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise CheckpointCompatibilityError(
            "strict Audio2Sph component load returned missing or unexpected keys"
        )
    return CheckpointLoadReport(
        path=checkpoint_path,
        model_name="Audio2Sph",
        class_feature_encoder="none",
        source_schema_id=source_schema,
        runtime_schema_id="audio2sph-runtime-v1",
        weights_kind="model",
        size_bytes=checkpoint_path.stat().st_size,
        sha256=checkpoint_sha256(checkpoint_path),
        source_key_count=len(source_state),
        loaded_key_count=len(runtime_state),
        compatibility_only_key_count=len(compatibility_only),
        missing_keys=(),
        unexpected_keys=(),
    )


def _read_class_feature_state(
    path: Path,
    *,
    source_suffix: str,
) -> tuple[Mapping[str, torch.Tensor], str, int]:
    if source_suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:  # pragma: no cover - dependency guard
            raise ImportError(
                "loading AudioMAE safetensors requires the safetensors package"
            ) from error
        state = load_file(str(path), device="cpu")
        return state, "upstream-safetensors-v1", 0
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(payload, Mapping):
        raise CheckpointCompatibilityError(
            "Class Feature Encoder checkpoint must contain a mapping"
        )
    adapter = payload.get("adapter")
    metadata = payload.get("meta")
    if isinstance(adapter, Mapping) and isinstance(metadata, Mapping):
        if metadata.get("backend") != "passt":
            raise CheckpointCompatibilityError(
                "a recognition-adapter checkpoint without embedded backbone "
                "weights cannot initialize the Class Feature Encoder"
            )
        backbone = adapter.get("backbone")
        domain_embedding = adapter.get("domain_emb")
        if not isinstance(backbone, Mapping) or not isinstance(
            domain_embedding, Mapping
        ):
            raise CheckpointCompatibilityError(
                "PaSST adapter must contain backbone and domain_emb states"
            )
        state: dict[str, torch.Tensor] = {}
        for key, value in backbone.items():
            state[f"backbone.{key}"] = value
        for key, value in domain_embedding.items():
            state[f"domain_embedding.{key}"] = value
        classifier = adapter.get("head")
        ignored = len(classifier) if isinstance(classifier, Mapping) else 0
        return state, "passt-dcase-adapter-v1", ignored
    state_value: Any = payload
    for container_key in ("model", "state_dict", "model_state_dict"):
        candidate = payload.get(container_key)
        if isinstance(candidate, Mapping):
            state_value = candidate
            break
    if not isinstance(state_value, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in state_value.items()
    ):
        raise CheckpointCompatibilityError(
            "Class Feature Encoder state must map string keys to tensors"
        )
    return state_value, "class-feature-runtime-flat-v1", 0


def load_class_feature_encoder_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    class_feature_encoder: str,
) -> CheckpointLoadReport:
    """Strictly load PaSST or AudioMAE Class Feature Encoder weights.

    The DCASE classifier head from a PaSST adapter is intentionally excluded:
    SAID consumes temporal backbone features before that classifier.  AudioMAE
    accepts the explicit upstream timm-compatible state dictionary, including
    ``model.safetensors`` from the paper-cited revision.
    """

    if class_feature_encoder not in {"passt", "audiomae"}:
        raise ValueError("class_feature_encoder must be 'passt' or 'audiomae'")
    provided_path = Path(path).expanduser()
    checkpoint_path = provided_path.resolve()
    if not checkpoint_path.is_file():
        raise CheckpointCompatibilityError(
            f"checkpoint file not found or not a regular file: {checkpoint_path}"
        )
    size = checkpoint_path.stat().st_size
    digest = checkpoint_sha256(checkpoint_path)
    paper = PAPER_CHECKPOINTS[class_feature_encoder]
    if size == paper.size_bytes and digest == paper.sha256:
        submitted_state = _read_submitted_state_dict(checkpoint_path)
        validate_submitted_checkpoint_summary(
            submitted_state, class_feature_encoder
        )
        source_state = submitted_class_feature_encoder_state_dict(
            submitted_state, class_feature_encoder
        )
        source_schema = "said-paper-class-feature-encoder-v1"
        ignored = len(submitted_state) - len(source_state)
    else:
        source_state, source_schema, ignored = _read_class_feature_state(
            checkpoint_path,
            source_suffix=provided_path.suffix,
        )
    expected = model.state_dict()
    if set(source_state) == set(expected):
        runtime_state = dict(source_state)
    else:
        prefixes = (
            "sph2imaging.class_feature_encoder.",
            "class_feature_encoder.",
        )
        runtime_state = dict(source_state)
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in source_state):
                runtime_state = {
                    key.removeprefix(prefix): value
                    for key, value in source_state.items()
                }
                break
        if (
            class_feature_encoder == "audiomae"
            and set(runtime_state) != set(expected)
        ):
            prefix = "backbone.audio_transformer."
            runtime_state = {
                prefix + key: value for key, value in source_state.items()
            }
    validate_state_dict_compatibility(runtime_state, expected)
    result = model.load_state_dict(runtime_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise CheckpointCompatibilityError(
            "strict Class Feature Encoder load returned missing or unexpected keys"
        )
    return CheckpointLoadReport(
        path=checkpoint_path,
        model_name=f"{class_feature_encoder} Class Feature Encoder",
        class_feature_encoder=class_feature_encoder,
        source_schema_id=source_schema,
        runtime_schema_id=(
            f"{class_feature_encoder}-class-feature-encoder-runtime-v1"
        ),
        weights_kind="model",
        size_bytes=size,
        sha256=digest,
        source_key_count=len(source_state) + ignored,
        loaded_key_count=len(runtime_state),
        compatibility_only_key_count=ignored,
        missing_keys=(),
        unexpected_keys=(),
    )
