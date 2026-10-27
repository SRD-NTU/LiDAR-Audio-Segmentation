"""Paper-aligned training components for SAID."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from .checkpoint import (
    TRAINING_CHECKPOINT_SCHEMA,
    TrainingCheckpointReport,
    load_training_checkpoint,
    save_model_checkpoint,
    save_training_checkpoint,
)
from .losses import (
    Audio2SphPretrainingCriterion,
    SAIDCriterion,
    SAIDLosses,
)
from .matcher import HungarianAssignments, SAIDHungarianMatcher
from ..data.targets import FrameAlignment, SourceMapTargets
from .train import (
    DeterministicSceneBatchSampler,
    DeterministicStepBatchSampler,
    TrainingRunReport,
    run_training,
)
from .steps import (
    ExponentialMovingAverage,
    PaperTrainingPhase,
    TrainingStepResult,
    build_optimizer_and_scheduler,
    train_audio2sph_step,
    train_said_step,
)

__all__ = [
    "TRAINING_CHECKPOINT_SCHEMA",
    "TrainingCheckpointReport",
    "load_training_checkpoint",
    "save_model_checkpoint",
    "save_training_checkpoint",
    "Audio2SphPretrainingCriterion",
    "SAIDCriterion",
    "SAIDLosses",
    "HungarianAssignments",
    "SAIDHungarianMatcher",
    "FrameAlignment",
    "SourceMapTargets",
    "DeterministicSceneBatchSampler",
    "DeterministicStepBatchSampler",
    "TrainingRunReport",
    "run_training",
    "ExponentialMovingAverage",
    "PaperTrainingPhase",
    "TrainingStepResult",
    "build_optimizer_and_scheduler",
    "train_audio2sph_step",
    "train_said_step",
]
