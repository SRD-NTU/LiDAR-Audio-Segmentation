"""Objectives reported for Audio2Sph pretraining and complete SAID training."""

# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2022 Meta, Inc.
# SAID adaptation and modifications: Copyright (c) 2026 Runbang Wang
#
# Dice, point sampling, and uncertainty-guided sampling are adapted from
# Detectron2 PointRend and Mask2Former:
# https://github.com/facebookresearch/detectron2/tree/9604f5995cc628619f0e4fd913453b4d7d61db3f
# https://github.com/facebookresearch/Mask2Former/tree/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a
# See THIRD_PARTY_NOTICES.md for attribution and license details.

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..models import Audio2SphPretrainingOutput, SAIDOutput
from .matcher import HungarianAssignments, SAIDHungarianMatcher, _point_sample
from ..data.targets import FrameAlignment, SourceMapTargets


def _dice_loss(logits: Tensor, targets: Tensor, normalizer: float) -> Tensor:
    probabilities = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2.0 * (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (numerator + 1.0) / (denominator + 1.0)).sum() / normalizer


def _uncertainty_guided_points(
    logits: Tensor,
    *,
    sample_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
    generator: torch.Generator | None,
) -> Tensor:
    maps = int(logits.shape[0])
    candidate_count = int(sample_points * oversample_ratio)
    important_count = int(sample_points * importance_sample_ratio)
    random_count = int(sample_points) - important_count
    if candidate_count < sample_points:
        raise ValueError("oversample_ratio must produce at least sample_points")
    if important_count < 0 or random_count < 0:
        raise ValueError("importance_sample_ratio must lie in [0,1]")
    candidates = torch.rand(
        maps,
        candidate_count,
        2,
        device=logits.device,
        generator=generator,
    )
    sampled_logits = _point_sample(logits[:, None], candidates)[:, 0]
    uncertainty = -sampled_logits.abs()
    selected = torch.topk(uncertainty, k=important_count, dim=1).indices
    offsets = candidate_count * torch.arange(
        maps, dtype=torch.long, device=logits.device
    )
    important = candidates.reshape(-1, 2)[
        (selected + offsets[:, None]).reshape(-1)
    ].reshape(maps, important_count, 2)
    if random_count == 0:
        return important
    random_points = torch.rand(
        maps,
        random_count,
        2,
        device=logits.device,
        generator=generator,
    )
    return torch.cat((important, random_points), dim=1)


def _align_output_frames(
    output: SAIDOutput, targets: SourceMapTargets
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    tuple[Tensor, ...],
    tuple[Tensor, ...],
]:
    model_output = output.sph2imaging
    prediction_frames = int(model_output.active_head_logits.shape[1])
    target_frames = targets.frames
    if targets.frame_alignment is FrameAlignment.EXACT:
        if prediction_frames != target_frames:
            raise ValueError(
                "exact target alignment requires identical frame counts; "
                f"got predictions={prediction_frames}, targets={target_frames}"
            )
    elif targets.frame_alignment is FrameAlignment.DCASE_HALF_OPEN:
        if prediction_frames != target_frames + 1:
            raise ValueError(
                "DCASE half-open alignment requires one detector endpoint frame; "
                f"got predictions={prediction_frames}, targets={target_frames}"
            )
    else:  # pragma: no cover - enum construction prevents this path
        raise ValueError(f"unsupported frame alignment: {targets.frame_alignment}")
    time_slice = slice(0, target_frames)
    return (
        model_output.active_head_logits[:, time_slice],
        model_output.slot_map_logits[:, time_slice],
        model_output.refined_map_logits[:, time_slice],
        model_output.slot_class_logits[:, time_slice],
        tuple(value[:, time_slice] for value in model_output.auxiliary_active_logits),
        tuple(value[:, time_slice] for value in model_output.auxiliary_slot_map_logits),
    )


def _matched_maps(
    prediction: Tensor,
    target: Tensor,
    assignments: HungarianAssignments,
) -> tuple[Tensor, Tensor]:
    predicted: list[Tensor] = []
    expected: list[Tensor] = []
    batch_size, frames = prediction.shape[:2]
    for flat_index, (slot_indices, target_indices) in enumerate(assignments.frames):
        if slot_indices.numel() == 0:
            continue
        batch_index = flat_index // int(frames)
        frame_index = flat_index % int(frames)
        predicted.append(prediction[batch_index, frame_index, slot_indices])
        expected.append(target[batch_index, frame_index, target_indices].to(prediction))
    if not predicted:
        return (
            prediction.new_zeros((0, *prediction.shape[-2:])),
            prediction.new_zeros((0, *target.shape[-2:])),
        )
    return torch.cat(predicted, dim=0), torch.cat(expected, dim=0)


def _active_head_cross_entropy(
    logits: Tensor,
    targets: SourceMapTargets,
    assignments: HungarianAssignments,
    *,
    no_object_weight: float,
) -> Tensor:
    batch_size, frames, slots, classes_with_no_object = logits.shape
    no_object = int(classes_with_no_object) - 1
    labels = torch.full(
        (batch_size, frames, slots),
        no_object,
        dtype=torch.long,
        device=logits.device,
    )
    for flat_index, (slot_indices, target_indices) in enumerate(assignments.frames):
        if slot_indices.numel() == 0:
            continue
        batch_index = flat_index // int(frames)
        frame_index = flat_index % int(frames)
        labels[batch_index, frame_index, slot_indices] = targets.class_ids[
            batch_index, frame_index, target_indices
        ].to(device=logits.device, dtype=torch.long)
    weights = logits.new_ones(classes_with_no_object)
    weights[no_object] = float(no_object_weight)
    return F.cross_entropy(logits.reshape(-1, classes_with_no_object).float(), labels.reshape(-1), weight=weights.float())


def _sampled_map_losses(
    logits: Tensor,
    target_maps: Tensor,
    assignments: HungarianAssignments,
    *,
    sample_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    predicted, expected = _matched_maps(logits, target_maps, assignments)
    if predicted.shape[0] == 0:
        zero = logits.sum() * 0.0
        return zero, zero
    coordinates = _uncertainty_guided_points(
        predicted,
        sample_points=sample_points,
        oversample_ratio=oversample_ratio,
        importance_sample_ratio=importance_sample_ratio,
        generator=generator,
    )
    point_logits = _point_sample(predicted[:, None], coordinates)[:, 0]
    with torch.no_grad():
        point_targets = _point_sample(expected[:, None], coordinates)[:, 0]
    normalizer = float(max(1, assignments.matched_count))
    map_bce = (
        F.binary_cross_entropy_with_logits(
            point_logits, point_targets, reduction="none"
        ).mean(dim=1).sum()
        / normalizer
    )
    map_dice = _dice_loss(point_logits, point_targets, normalizer)
    return map_bce, map_dice


def _refine_losses(
    logits: Tensor,
    targets: SourceMapTargets,
    assignments: HungarianAssignments,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    predicted, expected = _matched_maps(
        logits, targets.refined_map_targets, assignments
    )
    if predicted.shape[0] == 0:
        zero = logits.sum() * 0.0
        return zero, zero, zero, zero
    count = float(predicted.shape[0])
    refine_bce = (
        F.binary_cross_entropy_with_logits(
            predicted, expected, reduction="none"
        ).mean(dim=(-2, -1)).sum()
        / count
    )
    refine_dice = _dice_loss(predicted, expected, count)
    probability = predicted.sigmoid().flatten(1).float()
    expected_flat = expected.flatten(1).float()
    epsilon = 1.0e-6
    probability_centered = probability - probability.mean(dim=1, keepdim=True)
    expected_centered = expected_flat - expected_flat.mean(dim=1, keepdim=True)
    correlation = (
        (probability_centered * expected_centered).sum(dim=1)
        / (
            probability_centered.square().sum(dim=1).clamp_min(epsilon).sqrt()
            * expected_centered.square().sum(dim=1).clamp_min(epsilon).sqrt()
        ).clamp_min(epsilon)
    ).clamp(-1.0, 1.0)
    intersection = (probability * expected_flat).sum(dim=1)
    union = (
        probability + expected_flat - probability * expected_flat
    ).sum(dim=1).clamp_min(epsilon)
    correlation_loss = 1.0 - correlation.mean()
    soft_iou_loss = 1.0 - (intersection / union).clamp(0.0, 1.0).mean()
    return refine_bce, refine_dice, correlation_loss, soft_iou_loss


def _class_head_focal_loss(
    logits: Tensor,
    targets: SourceMapTargets,
    assignments: HungarianAssignments,
    *,
    class_weights: Tensor,
    gamma: float,
) -> Tensor:
    predictions: list[Tensor] = []
    labels: list[Tensor] = []
    _, frames = logits.shape[:2]
    for flat_index, (slot_indices, target_indices) in enumerate(assignments.frames):
        if slot_indices.numel() == 0:
            continue
        batch_index = flat_index // int(frames)
        frame_index = flat_index % int(frames)
        predictions.append(logits[batch_index, frame_index, slot_indices])
        labels.append(
            targets.class_ids[batch_index, frame_index, target_indices].to(
                device=logits.device, dtype=torch.long
            )
        )
    if not predictions:
        return logits.sum() * 0.0
    prediction = torch.cat(predictions, dim=0).float()
    label = torch.cat(labels, dim=0)
    cross_entropy = F.cross_entropy(
        prediction,
        label,
        weight=class_weights.to(device=prediction.device, dtype=prediction.dtype),
        reduction="none",
    )
    probability = prediction.softmax(dim=-1).gather(1, label[:, None])[:, 0]
    return (((1.0 - probability) ** float(gamma)) * cross_entropy).mean()


@dataclass(frozen=True)
class SAIDLosses:
    """Named terms of the paper's complete SAID objective."""

    total: Tensor
    active_head_cross_entropy: Tensor
    map_bce: Tensor
    map_dice: Tensor
    refine_bce: Tensor
    refine_dice: Tensor
    correlation: Tensor
    soft_iou: Tensor
    class_head_focal: Tensor
    matched_sources: int

    @property
    def refine(self) -> Tensor:
        return (
            self.refine_bce
            + self.refine_dice
            + 0.5 * (self.correlation + self.soft_iou)
        )

    def as_dict(self) -> dict[str, Tensor | int]:
        return {
            "loss": self.total,
            "active_head_cross_entropy": self.active_head_cross_entropy,
            "map_bce": self.map_bce,
            "map_dice": self.map_dice,
            "refine_bce": self.refine_bce,
            "refine_dice": self.refine_dice,
            "correlation": self.correlation,
            "soft_iou": self.soft_iou,
            "refine": self.refine,
            "class_head_focal": self.class_head_focal,
            "matched_sources": self.matched_sources,
        }


class Audio2SphPretrainingCriterion(nn.Module):
    """Unweighted mean BCE for class-agnostic Audio2Sph pretraining."""

    def forward(
        self, output: Audio2SphPretrainingOutput, target_maps: Tensor
    ) -> Tensor:
        logits = output.panoramic_decoder.class_agnostic_map_logits
        if tuple(logits.shape) != tuple(target_maps.shape):
            raise ValueError(
                "Audio2Sph target shape must equal class-agnostic map shape; "
                f"got logits={tuple(logits.shape)}, targets={tuple(target_maps.shape)}"
            )
        if not bool(torch.isfinite(target_maps).all()):
            raise ValueError("Audio2Sph target maps must be finite")
        if bool(((target_maps < 0) | (target_maps > 1)).any()):
            raise ValueError("Audio2Sph target maps must lie in [0,1]")
        return F.binary_cross_entropy_with_logits(
            logits.float(), target_maps.to(logits).float(), reduction="mean"
        )


class SAIDCriterion(nn.Module):
    """Ten-group matching and complete objective defined in the paper."""

    def __init__(
        self,
        *,
        class_counts: Sequence[float],
        sample_points: int = 12_544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        no_object_weight: float = 0.1,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        counts = torch.as_tensor(tuple(class_counts), dtype=torch.float32)
        if tuple(counts.shape) != (13,) or bool((counts <= 0).any()):
            raise ValueError("class_counts must contain 13 positive values")
        weights = torch.rsqrt(counts)
        weights = (weights / weights.mean()).clamp(0.25, 4.0)
        self.register_buffer("class_weights", weights)
        self.matcher = SAIDHungarianMatcher(sample_points=sample_points)
        self.sample_points = int(sample_points)
        self.oversample_ratio = float(oversample_ratio)
        self.importance_sample_ratio = float(importance_sample_ratio)
        self.no_object_weight = float(no_object_weight)
        self.focal_gamma = float(focal_gamma)

    def forward(
        self,
        output: SAIDOutput,
        targets: SourceMapTargets,
        *,
        generator: torch.Generator | None = None,
    ) -> SAIDLosses:
        (
            final_active,
            final_maps,
            refined_maps,
            class_logits,
            auxiliary_active,
            auxiliary_maps,
        ) = _align_output_frames(output, targets)
        if len(auxiliary_active) != 9 or len(auxiliary_maps) != 9:
            raise ValueError(
                "the paper objective requires the initial prediction plus "
                "nine Mask Decoder layers"
            )
        # Preserve the audited random-sampling order: final prediction first,
        # followed by the initial prediction and the eight intermediate
        # decoder predictions stored as auxiliary outputs.
        active_groups = (final_active, *auxiliary_active)
        map_groups = (final_maps, *auxiliary_maps)
        active_cross_entropy = final_active.sum() * 0.0
        map_bce = final_maps.sum() * 0.0
        map_dice = final_maps.sum() * 0.0
        final_assignments: HungarianAssignments | None = None
        for group_index, (active_logits, map_logits) in enumerate(
            zip(active_groups, map_groups)
        ):
            assignments = self.matcher(
                active_logits,
                map_logits,
                targets.valid_sources,
                targets.class_ids,
                targets.map_targets,
                generator=generator,
            )
            active_cross_entropy = active_cross_entropy + _active_head_cross_entropy(
                active_logits,
                targets,
                assignments,
                no_object_weight=self.no_object_weight,
            )
            group_bce, group_dice = _sampled_map_losses(
                map_logits,
                targets.map_targets,
                assignments,
                sample_points=self.sample_points,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
                generator=generator,
            )
            map_bce = map_bce + group_bce
            map_dice = map_dice + group_dice
            if group_index == 0:
                final_assignments = assignments
        if final_assignments is None:  # pragma: no cover - ten groups are required
            raise RuntimeError("final prediction group was not evaluated")
        refine_bce, refine_dice, correlation, soft_iou = _refine_losses(
            refined_maps, targets, final_assignments
        )
        class_focal = _class_head_focal_loss(
            class_logits,
            targets,
            final_assignments,
            class_weights=self.class_weights,
            gamma=self.focal_gamma,
        )
        refine = refine_bce + refine_dice + 0.5 * (correlation + soft_iou)
        total = (
            2.0 * active_cross_entropy
            + 5.0 * map_bce
            + 5.0 * map_dice
            + refine
            + class_focal
        )
        return SAIDLosses(
            total=total,
            active_head_cross_entropy=active_cross_entropy,
            map_bce=map_bce,
            map_dice=map_dice,
            refine_bce=refine_bce,
            refine_dice=refine_dice,
            correlation=correlation,
            soft_iou=soft_iou,
            class_head_focal=class_focal,
            matched_sources=final_assignments.matched_count,
        )
