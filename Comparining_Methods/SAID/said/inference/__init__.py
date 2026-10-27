"""Inference interfaces for complete recordings."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from .load_audio import (
    EIGENMIKE_CAPSULES_1BASED,
    load_eigenmike_audio,
    select_eigenmike_channels,
)
from .save import (
    write_class_agnostic_archive_json,
    write_class_agnostic_prediction_archive,
    write_class_agnostic_prediction_json,
    write_prediction_archive,
)
from .demo import (
    DEMO_SCENES,
    DemoScene,
    load_demo_audio,
    write_demo_audio,
    write_demo_class_agnostic_ground_truth_archive,
    write_demo_ground_truth_archive,
)
from .infer import (
    Audio2SphPredictor,
    ClassAgnosticPredictionChunk,
    PredictionChunk,
    SAIDPredictor,
    run_audio2sph_inference,
    run_inference,
)
from .visualization import (
    combine_comparison_videos,
    render_prediction_video,
)

__all__ = [
    "EIGENMIKE_CAPSULES_1BASED",
    "load_eigenmike_audio",
    "select_eigenmike_channels",
    "write_prediction_archive",
    "write_class_agnostic_archive_json",
    "write_class_agnostic_prediction_archive",
    "write_class_agnostic_prediction_json",
    "DEMO_SCENES",
    "DemoScene",
    "load_demo_audio",
    "write_demo_audio",
    "write_demo_class_agnostic_ground_truth_archive",
    "write_demo_ground_truth_archive",
    "PredictionChunk",
    "ClassAgnosticPredictionChunk",
    "SAIDPredictor",
    "Audio2SphPredictor",
    "run_inference",
    "run_audio2sph_inference",
    "combine_comparison_videos",
    "render_prediction_video",
]
