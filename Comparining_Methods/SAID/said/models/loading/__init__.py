"""Strict model loading and paper-checkpoint compatibility."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from .compatibility import (
    AUDIO2SPH_PAPER_CHECKPOINT,
    PAPER_CHECKPOINTS,
    CheckpointCompatibilityError,
    PaperCheckpoint,
    checkpoint_component,
    normalize_audio2sph_pretraining_state_dict,
    normalize_submitted_state_dict,
    submitted_class_feature_encoder_state_dict,
    summarize_checkpoint_components,
    validate_checkpoint_file,
    validate_audio2sph_checkpoint_summary,
    validate_state_dict_compatibility,
    validate_submitted_checkpoint_summary,
)
from .loaders import (
    CheckpointLoadReport,
    checkpoint_sha256,
    load_audio2sph_component_checkpoint,
    load_audio2sph_pretraining_checkpoint,
    load_class_feature_encoder_checkpoint,
    load_paper_checkpoint,
    load_said_checkpoint,
)

__all__ = [
    "PAPER_CHECKPOINTS",
    "AUDIO2SPH_PAPER_CHECKPOINT",
    "CheckpointCompatibilityError",
    "CheckpointLoadReport",
    "PaperCheckpoint",
    "checkpoint_component",
    "checkpoint_sha256",
    "load_audio2sph_component_checkpoint",
    "load_audio2sph_pretraining_checkpoint",
    "load_class_feature_encoder_checkpoint",
    "load_paper_checkpoint",
    "load_said_checkpoint",
    "normalize_audio2sph_pretraining_state_dict",
    "normalize_submitted_state_dict",
    "submitted_class_feature_encoder_state_dict",
    "summarize_checkpoint_components",
    "validate_checkpoint_file",
    "validate_audio2sph_checkpoint_summary",
    "validate_state_dict_compatibility",
    "validate_submitted_checkpoint_summary",
]
