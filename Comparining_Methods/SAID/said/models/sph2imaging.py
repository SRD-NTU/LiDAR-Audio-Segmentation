"""Sph2Imaging: SAID's labeled acoustic-map decoder."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .sph2imaging_blocks import (
    ClassHead,
    MaskDecoder,
    MultiScaleFeatureOutput,
    MultiScaleFeaturePath,
    RefineBlock,
)


@dataclass(frozen=True)
class Sph2ImagingOutput:
    """Paper outputs and training logits produced by Sph2Imaging."""

    active_head_logits: Tensor
    slot_active: Tensor
    slot_class_logits: Tensor
    slot_class_probabilities: Tensor
    slot_class_ids: Tensor
    slot_map_logits: Tensor
    slot_maps: Tensor
    refined_map_logits: Tensor
    refined_maps: Tensor
    slot_confidence: Tensor
    slot_queries: Tensor
    class_features: Tensor
    multi_scale_features: MultiScaleFeatureOutput
    auxiliary_active_logits: tuple[Tensor, ...]
    auxiliary_slot_map_logits: tuple[Tensor, ...]


class Sph2Imaging(nn.Module):
    """Decode Panoramic Features into labeled acoustic maps."""

    def __init__(self, class_feature_encoder: nn.Module) -> None:
        super().__init__()
        output_dimension = int(
            getattr(class_feature_encoder, "output_dim", 768)
        )
        if output_dimension != 768:
            raise ValueError(
                "the paper Class Feature Encoder must output 768 dimensions"
            )
        self.multi_scale_feature_path = MultiScaleFeaturePath()
        self.mask_decoder = MaskDecoder()
        self.class_feature_encoder = class_feature_encoder
        self.class_feature_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.LayerNorm(256),
        )
        self.class_embedding = nn.Embedding(13, 256)
        self.refine_block = RefineBlock()
        self.class_head = ClassHead()

    def forward(
        self,
        panoramic_features: Tensor,
        *,
        audio: Tensor,
        x2: Tensor,
        x3: Tensor,
        domain_id: Tensor | None = None,
    ) -> Sph2ImagingOutput:
        if panoramic_features.ndim != 5:
            raise ValueError(
                "panoramic_features must have shape [B,T,H,W,D]"
            )
        batch_size, frames = panoramic_features.shape[:2]
        multi_scale_features = self.multi_scale_feature_path(panoramic_features)

        class_features = self.class_feature_encoder(
            audio, domain_id=domain_id, target_frames=int(frames)
        )
        expected_shape = (int(batch_size), int(frames), 768)
        if tuple(class_features.shape) != expected_shape:
            raise ValueError(
                f"Class Feature Encoder returned {tuple(class_features.shape)}; "
                f"expected {expected_shape}"
            )
        projected_class_features = self.class_feature_projection(
            class_features.to(
                device=multi_scale_features.map_features.device,
                dtype=multi_scale_features.map_features.dtype,
            )
        )
        class_feature_memory = (
            projected_class_features[:, :, None, :]
            + self.class_embedding.weight[None, None].to(
                device=projected_class_features.device,
                dtype=projected_class_features.dtype,
            )
        ).reshape(batch_size * frames, 13, 256)

        decoded = self.mask_decoder(
            large_features=multi_scale_features.large_features,
            medium_features=multi_scale_features.medium_features,
            small_features=multi_scale_features.small_features,
            map_features=multi_scale_features.map_features,
            class_feature_memory=class_feature_memory,
        )
        refined = self.refine_block(
            multi_scale_features.map_features, decoded.slot_queries
        )

        active_logits = decoded.active_logits.reshape(
            batch_size, frames, *decoded.active_logits.shape[1:]
        )
        slot_map_logits = decoded.slot_map_logits.reshape(
            batch_size, frames, *decoded.slot_map_logits.shape[1:]
        )
        refined_map_logits = refined.refined_map_logits.reshape(
            batch_size, frames, *refined.refined_map_logits.shape[1:]
        )
        slot_queries = decoded.slot_queries.reshape(
            batch_size, frames, *decoded.slot_queries.shape[1:]
        )
        class_output = self.class_head(slot_queries, x2=x2, x3=x3)
        slot_active = 1.0 - active_logits.softmax(dim=-1)[..., -1]
        slot_class_probabilities = class_output.slot_class_logits.softmax(dim=-1)
        slot_class_probabilities_max, slot_class_ids = (
            slot_class_probabilities.max(dim=-1)
        )
        refined_maps = refined_map_logits.sigmoid()
        foreground = refined_maps > 0.5
        foreground_sum = (refined_maps * foreground).sum(dim=(-2, -1))
        foreground_count = foreground.sum(dim=(-2, -1))
        foreground_mean = torch.where(
            foreground_count > 0,
            foreground_sum / foreground_count.clamp_min(1),
            torch.zeros_like(foreground_sum),
        )
        slot_confidence = (
            slot_active * slot_class_probabilities_max * foreground_mean
        )

        return Sph2ImagingOutput(
            active_head_logits=active_logits,
            slot_active=slot_active,
            slot_class_logits=class_output.slot_class_logits,
            slot_class_probabilities=slot_class_probabilities,
            slot_class_ids=slot_class_ids,
            slot_map_logits=slot_map_logits,
            slot_maps=slot_map_logits.sigmoid(),
            refined_map_logits=refined_map_logits,
            refined_maps=refined_maps,
            slot_confidence=slot_confidence,
            slot_queries=slot_queries,
            class_features=class_features,
            multi_scale_features=multi_scale_features,
            auxiliary_active_logits=tuple(
                value.reshape(batch_size, frames, *value.shape[1:])
                for value in decoded.auxiliary_active_logits
            ),
            auxiliary_slot_map_logits=tuple(
                value.reshape(batch_size, frames, *value.shape[1:])
                for value in decoded.auxiliary_slot_map_logits
            ),
        )
