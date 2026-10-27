"""Optimization steps shared by Audio2Sph and complete SAID."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch
from torch import Tensor, nn

from ..models import Audio2SphPretrainingModel, SAID
from .losses import Audio2SphPretrainingCriterion, SAIDCriterion, SAIDLosses
from ..data.targets import SourceMapTargets


class PaperTrainingPhase(str, Enum):
    """Public names for the three training phases described in the paper."""

    AUDIO2SPH_PRETRAINING = "audio2sph_pretraining"
    SOURCEBANK_TRAINING = "sourcebank_training"
    DCASE_FINE_TUNING = "dcase_fine_tuning"


class ExponentialMovingAverage:
    """Float-precision EMA used for paper checkpoint selection."""

    def __init__(self, model: nn.Module, *, decay: float = 0.999) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must lie in [0,1)")
        self.decay = float(decay)
        self.model = deepcopy(model)
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_parameters = dict(self.model.named_parameters())
        model_parameters = dict(model.named_parameters())
        if ema_parameters.keys() != model_parameters.keys():
            raise ValueError("EMA and training model parameter schemas differ")
        for name, value in ema_parameters.items():
            source = model_parameters[name].detach().to(
                device=value.device, dtype=value.dtype
            )
            value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
        ema_buffers = dict(self.model.named_buffers())
        model_buffers = dict(model.named_buffers())
        if ema_buffers.keys() != model_buffers.keys():
            raise ValueError("EMA and training model buffer schemas differ")
        for name, value in ema_buffers.items():
            source = model_buffers[name].detach().to(device=value.device)
            if value.is_floating_point() or value.is_complex():
                value.mul_(self.decay).add_(
                    source.to(dtype=value.dtype), alpha=1.0 - self.decay
                )
            else:
                value.copy_(source)


class _WarmupCosine:
    def __init__(self, *, warmup_steps: int, total_steps: int, minimum_ratio: float) -> None:
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.minimum_ratio = float(minimum_ratio)
        if self.warmup_steps < 0 or self.total_steps <= 0:
            raise ValueError("scheduler steps must be non-negative with total_steps > 0")
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")
        if not 0.0 <= self.minimum_ratio <= 1.0:
            raise ValueError("minimum_ratio must lie in [0,1]")

    def __call__(self, step: int) -> float:
        if self.warmup_steps > 0 and int(step) <= self.warmup_steps:
            return float(step) / float(self.warmup_steps)
        progress = min(
            1.0,
            max(0.0, float(int(step) - self.warmup_steps))
            / float(self.total_steps - self.warmup_steps),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_ratio + (1.0 - self.minimum_ratio) * cosine


def _trainable(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    return [parameter for parameter in parameters if parameter.requires_grad]


def build_optimizer_and_scheduler(
    model: nn.Module,
    *,
    phase: PaperTrainingPhase,
    total_steps: int,
    warmup_steps: int = 1_000,
    learning_rate: float | None = None,
    audio2sph_learning_rate: float = 2.5e-6,
    minimum_learning_rate_ratio: float = 0.1,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """Build the audited AdamW and warmup-cosine training policy."""

    if isinstance(model, SAID):
        frozen = list(model.sph2imaging.class_feature_encoder.parameters())
        if any(parameter.requires_grad for parameter in frozen):
            raise ValueError("the paper Class Feature Encoder must remain frozen")
    if learning_rate is None:
        learning_rate = (
            1.0e-4
            if phase in {
                PaperTrainingPhase.AUDIO2SPH_PRETRAINING,
                PaperTrainingPhase.SOURCEBANK_TRAINING,
            }
            else 1.5e-6
        )
    if phase is PaperTrainingPhase.SOURCEBANK_TRAINING:
        if not isinstance(model, SAID):
            raise TypeError("SourceBank training requires a complete SAID model")
        audio2sph_parameters = _trainable(model.audio2sph.parameters())
        sph2imaging_parameters = _trainable(model.sph2imaging.parameters())
        parameter_groups: list[dict[str, object]] = [
            {"params": sph2imaging_parameters, "lr": float(learning_rate)},
            {
                "params": audio2sph_parameters,
                "lr": float(audio2sph_learning_rate),
            },
        ]
    else:
        parameter_groups = [
            {"params": _trainable(model.parameters()), "lr": float(learning_rate)}
        ]
    if any(len(group["params"]) == 0 for group in parameter_groups):  # type: ignore[arg-type]
        raise ValueError("an optimizer parameter group is empty")
    optimizer = torch.optim.AdamW(parameter_groups, lr=float(learning_rate))
    schedule = _WarmupCosine(
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        minimum_ratio=minimum_learning_rate_ratio,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


@dataclass(frozen=True)
class TrainingStepResult:
    """Detached values emitted by one optimizer update."""

    loss: float
    metrics: dict[str, float | int]


def _finish_step(
    loss: Tensor,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ema: ExponentialMovingAverage,
) -> None:
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("training loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    ema.update(model)
    if scheduler is not None:
        scheduler.step()


def train_audio2sph_step(
    model: Audio2SphPretrainingModel,
    *,
    audio: Tensor,
    target_maps: Tensor,
    criterion: Audio2SphPretrainingCriterion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ema: ExponentialMovingAverage,
) -> TrainingStepResult:
    model.train()
    output = model(audio)
    loss = criterion(output, target_maps)
    _finish_step(
        loss,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
    )
    value = float(loss.detach().cpu())
    return TrainingStepResult(loss=value, metrics={"loss": value})


def train_said_step(
    model: SAID,
    *,
    audio: Tensor,
    targets: SourceMapTargets,
    criterion: SAIDCriterion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ema: ExponentialMovingAverage,
    recording_domain: str | list[str] | None,
    generator: torch.Generator | None = None,
) -> TrainingStepResult:
    model.train()
    output = model(audio, recording_domain=recording_domain)
    losses: SAIDLosses = criterion(output, targets, generator=generator)
    _finish_step(
        losses.total,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
    )
    detached: dict[str, float | int] = {}
    for name, value in losses.as_dict().items():
        detached[name] = (
            int(value) if isinstance(value, int) else float(value.detach().cpu())
        )
    return TrainingStepResult(loss=float(losses.total.detach().cpu()), metrics=detached)
