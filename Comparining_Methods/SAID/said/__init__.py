"""Semantic Acoustic Imaging Detector (SAID)."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from .utils.load_config import SAIDConfig, load_config
from .models import (
    CheckpointLoadReport,
    LabeledAcousticMaps,
    RecordingDomain,
    SAID,
    SAIDOutput,
    load_audio2sph_pretraining_checkpoint,
    load_paper_checkpoint,
)
from .data.dcase2026 import (
    DCASE2026_CLASS_NAMES,
    DCASE2026_CLASS_TO_ID,
    class_name,
)

__all__ = [
    "SAID",
    "SAIDOutput",
    "LabeledAcousticMaps",
    "RecordingDomain",
    "CheckpointLoadReport",
    "load_paper_checkpoint",
    "load_audio2sph_pretraining_checkpoint",
    "SAIDConfig",
    "load_config",
    "DCASE2026_CLASS_NAMES",
    "DCASE2026_CLASS_TO_ID",
    "class_name",
]
__version__ = "1.0.0"
