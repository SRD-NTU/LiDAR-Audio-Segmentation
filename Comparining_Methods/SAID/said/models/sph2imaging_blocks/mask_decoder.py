"""Mask Decoder used by Sph2Imaging."""

# SPDX-License-Identifier: MIT
# Copyright (c) Facebook, Inc. and its affiliates.
# Modifications for SAID: Copyright (c) 2026 Runbang Wang
#
# Upstream Mask2Former revision:
# https://github.com/facebookresearch/Mask2Former/tree/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .multi_scale_features import _SinePositionEmbedding


class SelfAttention(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=0.0)
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(0.0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, value: Tensor, query_position: Tensor) -> Tensor:
        query = value + query_position
        update = self.attention(query, query, value=value)[0]
        return self.norm(value + self.dropout(update))


class CrossAttention(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=0.0)
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(0.0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        value: Tensor,
        memory: Tensor,
        *,
        memory_mask: Tensor | None,
        memory_position: Tensor | None,
        query_position: Tensor,
    ) -> Tensor:
        key = memory if memory_position is None else memory + memory_position
        update = self.attention(
            query=value + query_position,
            key=key,
            value=memory,
            attn_mask=memory_mask,
        )[0]
        return self.norm(value + self.dropout(update))


class FeedForwardNetwork(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dimension, hidden_dimension)
        self.dropout = nn.Dropout(0.0)
        self.linear2 = nn.Linear(hidden_dimension, dimension)
        self.norm = nn.LayerNorm(dimension)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, value: Tensor) -> Tensor:
        update = self.linear2(self.dropout(F.relu(self.linear1(value))))
        return self.norm(value + self.dropout(update))


class MapHead(nn.Module):
    """Three-layer MLP that maps each Slot Query to a map embedding."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dimension, dimension) for _ in range(3)]
        )

    def forward(self, value: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index < len(self.layers) - 1:
                value = F.relu(value)
        return value


@dataclass(frozen=True)
class MaskDecoderOutput:
    """Slot Active, Slot Map, and Slot Query predictions for every frame."""

    active_logits: Tensor
    slot_map_logits: Tensor
    slot_queries: Tensor
    auxiliary_active_logits: tuple[Tensor, ...]
    auxiliary_slot_map_logits: tuple[Tensor, ...]


class MaskDecoder(nn.Module):
    """Update source slots with spatial and Class Features cross-attention."""

    def __init__(
        self,
        *,
        dimension: int = 256,
        num_heads: int = 4,
        feed_forward_dimension: int = 2048,
        num_layers: int = 9,
        num_slots: int = 16,
        num_classes: int = 13,
        num_feature_levels: int = 3,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.num_feature_levels = int(num_feature_levels)
        if self.num_feature_levels != 3:
            raise ValueError("MaskDecoder requires Large, Medium, and Small Features")

        self.position_embedding = _SinePositionEmbedding(self.dimension // 2)
        self.self_attention_layers = nn.ModuleList(
            [
                SelfAttention(self.dimension, self.num_heads)
                for _ in range(self.num_layers)
            ]
        )
        self.masked_cross_attention_layers = nn.ModuleList(
            [
                CrossAttention(self.dimension, self.num_heads)
                for _ in range(self.num_layers)
            ]
        )
        self.class_feature_cross_attention_layers = nn.ModuleList(
            [
                CrossAttention(self.dimension, self.num_heads)
                for _ in range(self.num_layers)
            ]
        )
        self.feed_forward_layers = nn.ModuleList(
            [
                FeedForwardNetwork(
                    self.dimension, int(feed_forward_dimension)
                )
                for _ in range(self.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.dimension)
        self.slot_queries = nn.Embedding(int(num_slots), self.dimension)
        self.slot_query_positions = nn.Embedding(int(num_slots), self.dimension)
        self.feature_level_embedding = nn.Embedding(
            self.num_feature_levels, self.dimension
        )
        self.active_head = nn.Linear(self.dimension, int(num_classes) + 1)
        self.map_head = MapHead(self.dimension)

    def _prediction_heads(
        self,
        slots: Tensor,
        map_features: Tensor,
        attention_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        normalized_slots = self.output_norm(slots).transpose(0, 1)
        active_logits = self.active_head(normalized_slots)
        map_embeddings = self.map_head(normalized_slots)
        slot_map_logits = torch.einsum(
            "bqc,bchw->bqhw", map_embeddings, map_features
        )
        attention_mask = F.interpolate(
            slot_map_logits,
            size=attention_size,
            mode="bilinear",
            align_corners=False,
        )
        attention_mask = (
            attention_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        return active_logits, slot_map_logits, attention_mask.detach()

    def forward(
        self,
        *,
        large_features: Tensor,
        medium_features: Tensor,
        small_features: Tensor,
        map_features: Tensor,
        class_feature_memory: Tensor,
    ) -> MaskDecoderOutput:
        features = [large_features, medium_features, small_features]
        batch_frames = int(features[0].shape[0])
        if any(int(feature.shape[0]) != batch_frames for feature in features):
            raise ValueError("all multi-scale feature levels must share a batch")
        if (
            class_feature_memory.ndim != 3
            or int(class_feature_memory.shape[0]) != batch_frames
            or int(class_feature_memory.shape[2]) != self.dimension
        ):
            raise ValueError(
                "class_feature_memory must have shape [B*T, classes, dimension]"
            )

        spatial_memory: list[Tensor] = []
        spatial_positions: list[Tensor] = []
        spatial_sizes: list[tuple[int, int]] = []
        for level, feature in enumerate(features):
            spatial_sizes.append(tuple(int(x) for x in feature.shape[-2:]))
            position = self.position_embedding(feature).flatten(2).permute(2, 0, 1)
            memory = feature.flatten(2)
            memory = memory + self.feature_level_embedding.weight[level][
                None, :, None
            ]
            spatial_memory.append(memory.permute(2, 0, 1))
            spatial_positions.append(position)

        class_memory = class_feature_memory.transpose(0, 1)
        query_positions = self.slot_query_positions.weight[:, None].repeat(
            1, batch_frames, 1
        )
        slots = self.slot_queries.weight[:, None].repeat(1, batch_frames, 1)

        active_predictions: list[Tensor] = []
        map_predictions: list[Tensor] = []
        active_logits, slot_map_logits, attention_mask = self._prediction_heads(
            slots, map_features, spatial_sizes[0]
        )
        active_predictions.append(active_logits)
        map_predictions.append(slot_map_logits)

        for index in range(self.num_layers):
            level = index % self.num_feature_levels
            attention_mask = attention_mask.clone()
            fully_masked = attention_mask.sum(-1) == attention_mask.shape[-1]
            attention_mask[torch.where(fully_masked)] = False
            slots = self.masked_cross_attention_layers[index](
                slots,
                spatial_memory[level],
                memory_mask=attention_mask,
                memory_position=spatial_positions[level],
                query_position=query_positions,
            )
            slots = self.class_feature_cross_attention_layers[index](
                slots,
                class_memory,
                memory_mask=None,
                memory_position=None,
                query_position=query_positions,
            )
            slots = self.self_attention_layers[index](slots, query_positions)
            slots = self.feed_forward_layers[index](slots)
            next_level = (index + 1) % self.num_feature_levels
            active_logits, slot_map_logits, attention_mask = self._prediction_heads(
                slots, map_features, spatial_sizes[next_level]
            )
            active_predictions.append(active_logits)
            map_predictions.append(slot_map_logits)

        return MaskDecoderOutput(
            active_logits=active_predictions[-1],
            slot_map_logits=map_predictions[-1],
            slot_queries=self.output_norm(slots).transpose(0, 1),
            auxiliary_active_logits=tuple(active_predictions[:-1]),
            auxiliary_slot_map_logits=tuple(map_predictions[:-1]),
        )
