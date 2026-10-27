"""Refine Block for full-resolution labeled acoustic maps."""

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications for SAID: Copyright (c) 2026 Runbang Wang
#
# Upstream SAM 2 revision:
# https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


class _LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, value: Tensor) -> Tensor:
        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        value = (value - mean) * torch.rsqrt(variance + 1.0e-6)
        return self.weight[:, None, None] * value + self.bias[:, None, None]


@dataclass(frozen=True)
class RefineBlockOutput:
    """Full-resolution map logits and their generating features."""

    refined_map_logits: Tensor
    high_resolution_map_features: Tensor
    high_resolution_map_embeddings: Tensor


class RefineBlock(nn.Module):
    """Upsample Map Features and read one refined map per Slot Query."""

    def __init__(
        self,
        *,
        dimension: int = 256,
        map_dimension: int = 256,
        map_embedding_layers: int = 3,
        residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        intermediate_dimension = max(1, int(map_dimension) // 4)
        output_dimension = max(1, int(map_dimension) // 8)
        residual_layers: list[nn.Module] = []
        for _ in range(int(residual_blocks)):
            residual_layers.extend(
                [
                    nn.Conv2d(
                        output_dimension,
                        output_dimension,
                        kernel_size=3,
                        padding=1,
                    ),
                    _LayerNorm2d(output_dimension),
                    nn.GELU(),
                ]
            )
        self.high_resolution_refinement = nn.Sequential(*residual_layers)
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(
                int(map_dimension),
                intermediate_dimension,
                kernel_size=2,
                stride=2,
            ),
            _LayerNorm2d(intermediate_dimension),
            nn.GELU(),
            nn.ConvTranspose2d(
                intermediate_dimension,
                output_dimension,
                kernel_size=2,
                stride=2,
            ),
            nn.GELU(),
        )
        dimensions = (
            [int(dimension)]
            + [int(dimension)] * (max(1, int(map_embedding_layers)) - 1)
            + [output_dimension]
        )
        embedding_layers: list[nn.Module] = []
        for index in range(len(dimensions) - 1):
            embedding_layers.append(
                nn.Linear(dimensions[index], dimensions[index + 1])
            )
            if index < len(dimensions) - 2:
                embedding_layers.append(nn.GELU())
        self.map_embedding = nn.Sequential(*embedding_layers)

    def forward(
        self, map_features: Tensor, slot_queries: Tensor
    ) -> RefineBlockOutput:
        high_resolution_map_features = self.upsampling(map_features)
        if self.high_resolution_refinement:
            high_resolution_map_features = (
                high_resolution_map_features
                + self.high_resolution_refinement(high_resolution_map_features)
            )
        high_resolution_map_embeddings = self.map_embedding(slot_queries)
        refined_map_logits = torch.einsum(
            "bqc,bchw->bqhw",
            high_resolution_map_embeddings,
            high_resolution_map_features,
        )
        return RefineBlockOutput(
            refined_map_logits=refined_map_logits,
            high_resolution_map_features=high_resolution_map_features,
            high_resolution_map_embeddings=high_resolution_map_embeddings,
        )
