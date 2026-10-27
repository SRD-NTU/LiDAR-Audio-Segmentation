"""Per-frame Hungarian matching for the SAID prediction groups."""

# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2022 Meta, Inc.
# SAID adaptation and modifications: Copyright (c) 2026 Runbang Wang
#
# The point sampling and Hungarian map-matching flow is adapted from
# Detectron2 PointRend and Mask2Former:
# https://github.com/facebookresearch/detectron2/tree/9604f5995cc628619f0e4fd913453b4d7d61db3f
# https://github.com/facebookresearch/Mask2Former/tree/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a
# See THIRD_PARTY_NOTICES.md for attribution and license details.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn


def _point_sample(values: Tensor, coordinates: Tensor) -> Tensor:
    """Bilinearly sample ``[N,C,H,W]`` values at normalized xy points."""

    grid = coordinates.mul(2.0).sub(1.0).unsqueeze(2)
    return F.grid_sample(
        values, grid, mode="bilinear", align_corners=False
    ).squeeze(-1)


def _pairwise_map_bce(logits: Tensor, targets: Tensor) -> Tensor:
    pixels = int(logits.shape[1])
    positive = F.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits), reduction="none"
    )
    negative = F.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits), reduction="none"
    )
    return (
        torch.einsum("qp,mp->qm", positive, targets)
        + torch.einsum("qp,mp->qm", negative, 1.0 - targets)
    ) / float(pixels)


def _pairwise_map_dice(logits: Tensor, targets: Tensor) -> Tensor:
    probabilities = logits.sigmoid()
    numerator = 2.0 * torch.einsum("qp,mp->qm", probabilities, targets)
    denominator = (
        probabilities.sum(dim=1)[:, None] + targets.sum(dim=1)[None]
    )
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


@dataclass(frozen=True)
class HungarianAssignments:
    """Matched slot and target indices for every flattened batch-time frame."""

    frames: tuple[tuple[Tensor, Tensor], ...]

    @property
    def matched_count(self) -> int:
        return sum(int(slots.numel()) for slots, _ in self.frames)


class SAIDHungarianMatcher(nn.Module):
    """Match source slots using Active Head class and Map Head agreement."""

    def __init__(
        self,
        *,
        class_cost: float = 2.0,
        map_bce_cost: float = 5.0,
        map_dice_cost: float = 5.0,
        sample_points: int = 12_544,
    ) -> None:
        super().__init__()
        self.class_cost = float(class_cost)
        self.map_bce_cost = float(map_bce_cost)
        self.map_dice_cost = float(map_dice_cost)
        self.sample_points = int(sample_points)
        if self.sample_points <= 0:
            raise ValueError("sample_points must be positive")
        if self.class_cost == self.map_bce_cost == self.map_dice_cost == 0.0:
            raise ValueError("at least one Hungarian matching cost must be nonzero")

    @torch.no_grad()
    def forward(
        self,
        active_head_logits: Tensor,
        slot_map_logits: Tensor,
        valid_sources: Tensor,
        class_ids: Tensor,
        map_targets: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> HungarianAssignments:
        if active_head_logits.ndim != 4:
            raise ValueError("active_head_logits must have shape [B,T,N,14]")
        if slot_map_logits.ndim != 5:
            raise ValueError("slot_map_logits must have shape [B,T,N,H,W]")
        batch_size, frames, slots, classes_with_no_object = active_head_logits.shape
        if int(classes_with_no_object) != 14:
            raise ValueError("Active Head must predict 13 classes plus no-object")
        expected_prefix = (int(batch_size), int(frames))
        if tuple(slot_map_logits.shape[:2]) != expected_prefix:
            raise ValueError("Active Head and Map Head frame dimensions differ")
        if tuple(valid_sources.shape[:2]) != expected_prefix:
            raise ValueError("prediction and target frame dimensions differ")
        assignments: list[tuple[Tensor, Tensor]] = []
        for batch_index in range(int(batch_size)):
            for frame_index in range(int(frames)):
                keep = valid_sources[batch_index, frame_index].to(dtype=torch.bool)
                target_indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
                if target_indices.numel() == 0:
                    empty = torch.empty(
                        0, dtype=torch.long, device=active_head_logits.device
                    )
                    assignments.append((empty, empty))
                    continue
                labels = class_ids[
                    batch_index, frame_index, target_indices
                ].to(device=active_head_logits.device, dtype=torch.long)
                target_maps = map_targets[
                    batch_index, frame_index, target_indices
                ].to(slot_map_logits)
                predicted_maps = slot_map_logits[batch_index, frame_index]
                coordinates = torch.rand(
                    1,
                    self.sample_points,
                    2,
                    device=predicted_maps.device,
                    generator=generator,
                )
                predicted_points = _point_sample(
                    predicted_maps[:, None],
                    coordinates.repeat(int(slots), 1, 1),
                )[:, 0]
                target_points = _point_sample(
                    target_maps[:, None],
                    coordinates.repeat(int(target_maps.shape[0]), 1, 1),
                )[:, 0]
                active_probabilities = active_head_logits[
                    batch_index, frame_index
                ].softmax(dim=-1)
                class_cost = -active_probabilities[:, labels]
                map_bce_cost = _pairwise_map_bce(
                    predicted_points.float(), target_points.float()
                )
                map_dice_cost = _pairwise_map_dice(
                    predicted_points.float(), target_points.float()
                )
                total_cost = (
                    self.class_cost * class_cost
                    + self.map_bce_cost * map_bce_cost
                    + self.map_dice_cost * map_dice_cost
                )
                slot_index, local_target_index = linear_sum_assignment(
                    total_cost.detach().cpu().numpy()
                )
                assignments.append(
                    (
                        torch.as_tensor(
                            slot_index,
                            dtype=torch.long,
                            device=active_head_logits.device,
                        ),
                        target_indices[
                            torch.as_tensor(
                                local_target_index,
                                dtype=torch.long,
                                device=target_indices.device,
                            )
                        ],
                    )
                )
        return HungarianAssignments(tuple(assignments))


__all__ = [
    "HungarianAssignments",
    "SAIDHungarianMatcher",
    "_point_sample",
]
