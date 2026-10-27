"""Class Head used by Sph2Imaging."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttention(nn.Module):
    """Cross-Attention and FFN block used by the paper's Class Head."""

    def __init__(self, dimension: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            int(dimension), int(num_heads), dropout=0.0, batch_first=True
        )
        self.norm_attention = nn.LayerNorm(int(dimension))
        self.norm_feed_forward = nn.LayerNorm(int(dimension))
        self.dropout = nn.Dropout(0.0)
        self.feed_forward = nn.Sequential(
            nn.Linear(int(dimension), 4 * int(dimension)),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(4 * int(dimension), int(dimension)),
        )

    def forward(self, slot_queries: Tensor, audio_tokens: Tensor) -> Tensor:
        update, _ = self.attention(
            query=slot_queries,
            key=audio_tokens,
            value=audio_tokens,
            need_weights=False,
        )
        slot_queries = self.norm_attention(
            slot_queries + self.dropout(update)
        )
        return self.norm_feed_forward(
            slot_queries + self.dropout(self.feed_forward(slot_queries))
        )


@dataclass(frozen=True)
class ClassHeadOutput:
    """Slot Class predictions and the Class Head's updated Slot Queries."""

    slot_class_logits: Tensor
    class_head_slot_queries: Tensor


class ClassHead(nn.Module):
    """Predict Slot Class scores from Audio2Sph X2/X3 features."""

    def __init__(
        self,
        *,
        dimension: int = 256,
        num_classes: int = 13,
        frequency_tokens_per_level: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.frequency_tokens_per_level = int(frequency_tokens_per_level)
        self.feature_projection = nn.ModuleDict(
            {
                "x2": nn.Linear(384, self.dimension),
                "x3": nn.Linear(768, self.dimension),
            }
        )
        self.feature_level_embedding = nn.Parameter(torch.zeros(2, self.dimension))
        self.cross_attention_layers = nn.ModuleList(
            [
                CrossAttention(self.dimension, int(num_heads))
                for _ in range(int(num_layers))
            ]
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.dimension),
            nn.Linear(self.dimension, self.dimension),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(self.dimension, int(num_classes)),
        )
        nn.init.normal_(self.feature_level_embedding, std=0.02)

    def _frequency_position(
        self, count: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        position = torch.linspace(
            0.0, 1.0, steps=int(count), device=device, dtype=dtype
        )
        half_dimension = max(1, self.dimension // 2)
        frequencies = torch.arange(
            half_dimension, device=device, dtype=dtype
        )
        frequencies = torch.pow(
            torch.full_like(frequencies, 10000.0),
            2.0
            * torch.div(frequencies, 2, rounding_mode="floor")
            / float(half_dimension),
        )
        embedding = position[:, None] / frequencies[None]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(
                embedding, (0, self.dimension - embedding.shape[1])
            )
        return embedding[:, : self.dimension]

    def _audio_tokens(
        self,
        name: str,
        features: Tensor,
        output_frames: int,
        level: int,
    ) -> Tensor:
        if features.ndim != 4:
            raise ValueError(f"{name} must have shape [B,C,T,F]")
        features = F.interpolate(
            features.float(),
            size=(int(output_frames), self.frequency_tokens_per_level),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=features.dtype)
        tokens = features.permute(0, 2, 3, 1).contiguous()
        tokens = self.feature_projection[name](tokens)
        position = self._frequency_position(
            self.frequency_tokens_per_level, tokens.device, tokens.dtype
        )
        return (
            tokens
            + position[None, None]
            + self.feature_level_embedding[level]
            .to(device=tokens.device, dtype=tokens.dtype)
            .view(1, 1, 1, -1)
        )

    def forward(
        self, slot_queries: Tensor, *, x2: Tensor, x3: Tensor
    ) -> ClassHeadOutput:
        if slot_queries.ndim != 4:
            raise ValueError("slot_queries must have shape [B,T,N,S]")
        batch_size, frames, num_slots, dimension = slot_queries.shape
        if int(dimension) != self.dimension:
            raise ValueError(
                f"slot query dimension {dimension} does not match {self.dimension}"
            )
        tokens = torch.cat(
            (
                self._audio_tokens("x2", x2, frames, 0),
                self._audio_tokens("x3", x3, frames, 1),
            ),
            dim=2,
        ).reshape(batch_size * frames, -1, self.dimension)
        flat_queries = slot_queries.reshape(
            batch_size * frames, num_slots, self.dimension
        )
        updated_queries = flat_queries
        for layer in self.cross_attention_layers:
            updated_queries = layer(updated_queries, tokens)
        class_conditioned_queries = (
            flat_queries + self.residual_scale * updated_queries
        )
        class_logits = self.classifier(class_conditioned_queries)
        return ClassHeadOutput(
            slot_class_logits=class_logits.reshape(
                batch_size, frames, num_slots, -1
            ),
            class_head_slot_queries=class_conditioned_queries.reshape(
                batch_size, frames, num_slots, self.dimension
            ),
        )
