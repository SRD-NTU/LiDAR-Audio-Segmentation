"""Paper-aligned target contract shared by recorded and simulated data."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import torch
from torch import Tensor


class FrameAlignment(str, Enum):
    """Declared relationship between targets and SAID output frames."""

    EXACT = "exact"
    DCASE_HALF_OPEN = "dcase_half_open"


@dataclass(frozen=True)
class SourceMapTargets:
    """Per-source classes and acoustic maps for a batch of recordings.

    All tensors use batch-major order. ``valid_sources`` and ``class_ids``
    have shape ``[B,T,M]``; ``map_targets`` and ``refined_map_targets`` have
    shapes ``[B,T,M,45,90]`` and ``[B,T,M,180,360]`` respectively.
    """

    valid_sources: Tensor
    class_ids: Tensor
    map_targets: Tensor
    refined_map_targets: Tensor
    frame_alignment: FrameAlignment = FrameAlignment.EXACT

    def __post_init__(self) -> None:
        valid = self.valid_sources
        classes = self.class_ids
        maps = self.map_targets
        refined = self.refined_map_targets
        if valid.ndim != 3:
            raise ValueError("valid_sources must have shape [B,T,M]")
        if tuple(classes.shape) != tuple(valid.shape):
            raise ValueError("class_ids must have the same shape as valid_sources")
        if maps.ndim != 5 or tuple(maps.shape[:3]) != tuple(valid.shape):
            raise ValueError("map_targets must have shape [B,T,M,H,W]")
        if refined.ndim != 5 or tuple(refined.shape[:3]) != tuple(valid.shape):
            raise ValueError(
                "refined_map_targets must have shape [B,T,M,H,W]"
            )
        if tuple(maps.shape[-2:]) != (45, 90):
            raise ValueError("map_targets must use the paper's 45x90 grid")
        if tuple(refined.shape[-2:]) != (180, 360):
            raise ValueError(
                "refined_map_targets must use the paper's 180x360 grid"
            )
        valid_mask = valid.to(dtype=torch.bool)
        if bool(valid_mask.any()):
            selected = classes[valid_mask]
            if bool(((selected < 0) | (selected >= 13)).any()):
                raise ValueError(
                    "valid class_ids must be in the zero-based range [0,12]"
                )
        for name, value in (
            ("map_targets", maps),
            ("refined_map_targets", refined),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f"{name} values must lie in [0,1]")

    @property
    def batch_size(self) -> int:
        return int(self.valid_sources.shape[0])

    @property
    def frames(self) -> int:
        return int(self.valid_sources.shape[1])

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SourceMapTargets":
        required = {
            "valid_sources",
            "class_ids",
            "map_targets",
            "refined_map_targets",
            "frame_alignment",
        }
        missing = required - set(value)
        unknown = set(value) - required
        if missing or unknown:
            raise ValueError(
                f"source-map target fields mismatch: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        tensors = {}
        for key in required - {"frame_alignment"}:
            item = value[key]
            if not isinstance(item, Tensor):
                raise TypeError(f"{key} must be a torch.Tensor")
            tensors[key] = item
        alignment_value = value["frame_alignment"]
        alignment = (
            alignment_value
            if isinstance(alignment_value, FrameAlignment)
            else FrameAlignment(str(alignment_value))
        )
        return cls(**tensors, frame_alignment=alignment)

    def to(self, device: torch.device | str) -> "SourceMapTargets":
        return SourceMapTargets(
            valid_sources=self.valid_sources.to(device),
            class_ids=self.class_ids.to(device),
            map_targets=self.map_targets.to(device),
            refined_map_targets=self.refined_map_targets.to(device),
            frame_alignment=self.frame_alignment,
        )
