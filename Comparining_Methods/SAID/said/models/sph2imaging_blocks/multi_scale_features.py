"""Multi-scale spatial feature processing used by Sph2Imaging."""

# SPDX-License-Identifier: Apache-2.0 AND MIT
# Multi-scale deformable attention portions:
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Mask2Former pixel-decoder portions:
# Copyright (c) Facebook, Inc. and its affiliates.
# Modifications for SAID: Copyright (c) 2026 Runbang Wang
#
# Upstream Deformable DETR revision:
# https://github.com/fundamentalvision/Deformable-DETR/tree/11169a60c33333af00a4849f1808023eba96a931
# Upstream Mask2Former revision:
# https://github.com/facebookresearch/Mask2Former/tree/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class MultiScaleFeatureOutput:
    """Per-frame Sph2Imaging features using the paper's scale names."""

    map_features: Tensor
    small_features: Tensor
    medium_features: Tensor
    large_features: Tensor


class _SinePositionEmbedding(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.features = int(features)

    def forward(self, value: Tensor) -> Tensor:
        batch_size, _, height, width = value.shape
        y = torch.arange(
            1, height + 1, dtype=value.dtype, device=value.device
        )[None, :, None].expand(batch_size, height, width)
        x = torch.arange(
            1, width + 1, dtype=value.dtype, device=value.device
        )[None, None, :].expand(batch_size, height, width)
        scale = 2.0 * math.pi
        y = y / (y[:, -1:, :] + 1.0e-6) * scale
        x = x / (x[:, :, -1:] + 1.0e-6) * scale
        frequencies = torch.arange(
            self.features, dtype=value.dtype, device=value.device
        )
        frequencies = 10000 ** (
            2
            * torch.div(frequencies, 2, rounding_mode="floor")
            / self.features
        )
        x = x[:, :, :, None] / frequencies
        y = y[:, :, :, None] / frequencies
        x = torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=4).flatten(3)
        y = torch.stack((y[..., 0::2].sin(), y[..., 1::2].cos()), dim=4).flatten(3)
        return torch.cat((y, x), dim=3).permute(0, 3, 1, 2)


def _sample_multi_scale(
    value: Tensor,
    spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    batch_size, _, heads, head_dim = value.shape
    _, query_count, _, level_count, point_count, _ = sampling_locations.shape
    split_sizes = [int(height * width) for height, width in spatial_shapes.tolist()]
    values = value.split(split_sizes, dim=1)
    grids = 2.0 * sampling_locations - 1.0
    samples = []
    for level, (height, width) in enumerate(spatial_shapes.tolist()):
        level_value = values[level].flatten(2).transpose(1, 2).reshape(
            batch_size * heads, head_dim, int(height), int(width)
        )
        level_grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        samples.append(
            F.grid_sample(
                level_value,
                level_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = attention_weights.transpose(1, 2).reshape(
        batch_size, heads, 1, query_count, level_count * point_count
    ).flatten(0, 1)
    output = (torch.stack(samples, dim=-2).flatten(-2) * weights).sum(-1)
    output = output.view(batch_size, heads * head_dim, query_count)
    return output.transpose(1, 2).contiguous()


class _DeformableAttention(nn.Module):
    def __init__(
        self,
        dimension: int = 256,
        levels: int = 3,
        heads: int = 4,
        points: int = 4,
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        self.dimension = int(dimension)
        self.levels = int(levels)
        self.heads = int(heads)
        self.points = int(points)
        self.sampling_offsets = nn.Linear(
            dimension, heads * levels * points * 2
        )
        self.attention_weights = nn.Linear(
            dimension, heads * levels * points
        )
        self.value_proj = nn.Linear(dimension, dimension)
        self.output_proj = nn.Linear(dimension, dimension)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        angles = torch.arange(self.heads, dtype=torch.float32) * (
            2.0 * math.pi / self.heads
        )
        grid = torch.stack((angles.cos(), angles.sin()), dim=-1)
        grid = grid / grid.abs().max(dim=-1, keepdim=True).values
        grid = grid.view(self.heads, 1, 1, 2).repeat(
            1, self.levels, self.points, 1
        )
        for point in range(self.points):
            grid[:, :, point] *= point + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.flatten())
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        flattened_input: Tensor,
        spatial_shapes: Tensor,
        padding_mask: Tensor | None,
    ) -> Tensor:
        batch_size, query_count, _ = query.shape
        input_count = flattened_input.shape[1]
        if int(spatial_shapes.prod(dim=1).sum().item()) != input_count:
            raise ValueError("spatial_shapes do not match flattened input")
        value = self.value_proj(flattened_input)
        if padding_mask is not None:
            value = value.masked_fill(padding_mask[..., None], 0.0)
        value = value.view(
            batch_size,
            input_count,
            self.heads,
            self.dimension // self.heads,
        )
        offsets = self.sampling_offsets(query).view(
            batch_size,
            query_count,
            self.heads,
            self.levels,
            self.points,
            2,
        )
        weights = self.attention_weights(query).view(
            batch_size,
            query_count,
            self.heads,
            self.levels * self.points,
        )
        weights = weights.softmax(dim=-1).view(
            batch_size,
            query_count,
            self.heads,
            self.levels,
            self.points,
        )
        normalizer = torch.stack(
            (spatial_shapes[:, 1], spatial_shapes[:, 0]), dim=-1
        ).to(query.dtype)
        locations = (
            reference_points[:, :, None, :, None]
            + offsets / normalizer[None, None, None, :, None]
        )
        sampled = _sample_multi_scale(value, spatial_shapes, locations, weights)
        return self.output_proj(sampled)


class _MultiScaleEncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deformable_attention = _DeformableAttention()
        self.dropout1 = nn.Dropout(0.0)
        self.norm1 = nn.LayerNorm(256)
        self.linear1 = nn.Linear(256, 2048)
        self.dropout2 = nn.Dropout(0.0)
        self.linear2 = nn.Linear(2048, 256)
        self.dropout3 = nn.Dropout(0.0)
        self.norm2 = nn.LayerNorm(256)

    def forward(
        self,
        source: Tensor,
        position: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        update = self.deformable_attention(
            source + position,
            reference_points,
            source,
            spatial_shapes,
            padding_mask,
        )
        source = self.norm1(source + self.dropout1(update))
        update = self.linear2(self.dropout2(F.relu(self.linear1(source))))
        return self.norm2(source + self.dropout3(update))


class _MultiScaleEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_MultiScaleEncoderLayer()])
        self.feature_level_embedding = nn.Parameter(torch.empty(3, 256))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_level_embedding)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        for module in self.modules():
            if isinstance(module, _DeformableAttention):
                module.reset_parameters()

    @staticmethod
    def _reference_points(spatial_shapes: Tensor, batch_size: int) -> Tensor:
        references = []
        device = spatial_shapes.device
        for height, width in spatial_shapes.tolist():
            y, x = torch.meshgrid(
                torch.linspace(
                    0.5, float(height) - 0.5, int(height), device=device
                ),
                torch.linspace(
                    0.5, float(width) - 0.5, int(width), device=device
                ),
                indexing="ij",
            )
            references.append(
                torch.stack(
                    (x.flatten() / float(width), y.flatten() / float(height)),
                    dim=-1,
                )[None].expand(batch_size, -1, -1)
            )
        return torch.cat(references, dim=1)[:, :, None].expand(-1, -1, 3, -1)

    def forward(
        self, sources: list[Tensor], positions: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        flattened_sources = []
        flattened_masks = []
        flattened_positions = []
        shapes = []
        for level, (source, position) in enumerate(zip(sources, positions)):
            batch_size, _, height, width = source.shape
            mask = torch.zeros(
                (batch_size, height, width), dtype=torch.bool, device=source.device
            )
            shapes.append((height, width))
            flattened_sources.append(source.flatten(2).transpose(1, 2))
            flattened_masks.append(mask.flatten(1))
            flattened_positions.append(
                position.flatten(2).transpose(1, 2)
                + self.feature_level_embedding[level].view(1, 1, -1)
            )
        source = torch.cat(flattened_sources, dim=1)
        padding_mask = torch.cat(flattened_masks, dim=1)
        position = torch.cat(flattened_positions, dim=1)
        spatial_shapes = torch.as_tensor(
            shapes, dtype=torch.long, device=source.device
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros(1),
                spatial_shapes.prod(dim=1).cumsum(0)[:-1],
            )
        )
        reference_points = self._reference_points(
            spatial_shapes, source.shape[0]
        ).to(source.dtype)
        for layer in self.layers:
            source = layer(
                source,
                position,
                reference_points,
                spatial_shapes,
                padding_mask,
            )
        return source, spatial_shapes, level_start_index


class MultiScaleFeaturePath(nn.Module):
    """Build Small, Medium, Large, and Map Features from panoramic features."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(16, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
        )
        self.input_projection = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1),
                    nn.GroupNorm(32, 256),
                )
                for _ in range(3)
            ]
        )
        self.multi_scale_encoder = _MultiScaleEncoder()
        self.position_embedding = _SinePositionEmbedding(128)
        self.lateral_mid = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral_high = nn.Conv2d(256, 256, kernel_size=1)
        self.output_mid = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
        )
        self.output_high = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
        )
        self.map_features = nn.Conv2d(256, 256, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self, panoramic_features: Tensor
    ) -> MultiScaleFeatureOutput:
        batch_size, frames, height, width, dimension = panoramic_features.shape
        value = panoramic_features.permute(0, 1, 4, 2, 3).reshape(
            batch_size * frames, dimension, height, width
        )
        high_input = self.stem(value)
        mid_input = self.down1(high_input)
        low_input = self.down2(mid_input)
        encoder_inputs = [low_input, mid_input, high_input]
        sources = [
            projection(feature.float())
            for projection, feature in zip(self.input_projection, encoder_inputs)
        ]
        positions = [
            self.position_embedding(feature).to(feature.dtype)
            for feature in sources
        ]
        memory, spatial_shapes, level_start_index = self.multi_scale_encoder(
            sources, positions
        )
        split_sizes = []
        for level in range(3):
            if level < 2:
                split_sizes.append(
                    int(
                        (
                            level_start_index[level + 1]
                            - level_start_index[level]
                        ).item()
                    )
                )
            else:
                split_sizes.append(
                    int(memory.shape[1] - level_start_index[level].item())
                )
        split_memory = torch.split(memory, split_sizes, dim=1)
        small_features, medium_features, large_features = [
            item.transpose(1, 2).reshape(
                batch_size * frames,
                256,
                int(spatial_shapes[level, 0]),
                int(spatial_shapes[level, 1]),
            )
            for level, item in enumerate(split_memory)
        ]
        medium_features = self.output_mid(
            self.lateral_mid(mid_input.float())
            + F.interpolate(
                small_features,
                size=mid_input.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        large_features = self.output_high(
            self.lateral_high(high_input.float())
            + F.interpolate(
                medium_features,
                size=high_input.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        return MultiScaleFeatureOutput(
            map_features=self.map_features(large_features),
            small_features=small_features,
            medium_features=medium_features,
            large_features=large_features,
        )
