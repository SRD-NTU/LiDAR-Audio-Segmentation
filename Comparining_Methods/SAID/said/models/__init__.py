"""Paper-aligned model namespace for SAID."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from .loading import (
    AUDIO2SPH_PAPER_CHECKPOINT,
    PAPER_CHECKPOINTS,
    CheckpointCompatibilityError,
    CheckpointLoadReport,
    checkpoint_sha256,
    load_audio2sph_component_checkpoint,
    load_audio2sph_pretraining_checkpoint,
    load_class_feature_encoder_checkpoint,
    load_paper_checkpoint,
    load_said_checkpoint,
    normalize_audio2sph_pretraining_state_dict,
)
from .class_feature_encoders import (
    AudioMAEClassFeatureEncoder,
    PaSSTClassFeatureEncoder,
)
from .audio2sph import (
    Audio2Sph,
    Audio2SphOutput,
    SphericalCrossAttention,
    TimeAlignment,
)
from .panoramic_decoder import (
    Audio2SphPretrainingModel,
    Audio2SphPretrainingOutput,
    PanoramicDecoder,
    PanoramicDecoderOutput,
)
from .sph2imaging_blocks import (
    ClassHead,
    ClassHeadOutput,
    MapHead,
    MaskDecoder,
    MaskDecoderOutput,
    MultiScaleFeatureOutput,
    MultiScaleFeaturePath,
    RefineBlock,
    RefineBlockOutput,
)
from .said import LabeledAcousticMaps, RecordingDomain, SAID, SAIDOutput
from .sph2imaging import Sph2Imaging, Sph2ImagingOutput

__all__ = [
    "PaSSTClassFeatureEncoder",
    "AudioMAEClassFeatureEncoder",
    "Audio2Sph",
    "Audio2SphOutput",
    "SphericalCrossAttention",
    "TimeAlignment",
    "MultiScaleFeaturePath",
    "MultiScaleFeatureOutput",
    "MaskDecoder",
    "MaskDecoderOutput",
    "MapHead",
    "PanoramicDecoder",
    "PanoramicDecoderOutput",
    "Audio2SphPretrainingModel",
    "Audio2SphPretrainingOutput",
    "RefineBlock",
    "RefineBlockOutput",
    "CheckpointLoadReport",
    "checkpoint_sha256",
    "load_paper_checkpoint",
    "load_said_checkpoint",
    "load_audio2sph_pretraining_checkpoint",
    "load_audio2sph_component_checkpoint",
    "load_class_feature_encoder_checkpoint",
    "PAPER_CHECKPOINTS",
    "AUDIO2SPH_PAPER_CHECKPOINT",
    "CheckpointCompatibilityError",
    "normalize_audio2sph_pretraining_state_dict",
    "ClassHead",
    "ClassHeadOutput",
    "Sph2Imaging",
    "Sph2ImagingOutput",
    "SAID",
    "SAIDOutput",
    "LabeledAcousticMaps",
    "RecordingDomain",
]
