"""Audio2Sph: SAID's audio encoder."""

# SPDX-License-Identifier: MIT
# ConvNeXt components: Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications and Audio2Sph integration: Copyright (c) 2026 Runbang Wang
# Upstream ConvNeXt revision:
# https://github.com/facebookresearch/ConvNeXt/tree/048efcea897d999aed302f2639b6270aedf8d4c8

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, value: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return value
        survival_probability = 1.0 - self.probability
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        survives = torch.rand(shape, device=value.device) < survival_probability
        return value * survives.to(value.dtype) / survival_probability


class _ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int, *, channels_first: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.channels_first = bool(channels_first)
        self.normalized_shape = (int(channels),)

    def forward(self, value: Tensor) -> Tensor:
        if not self.channels_first:
            return F.layer_norm(
                value, self.normalized_shape, self.weight, self.bias, 1.0e-6
            )
        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        normalized = (value - mean) / torch.sqrt(variance + 1.0e-6)
        return self.weight[:, None, None] * normalized + self.bias[:, None, None]


class _ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, drop_path: float) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = _ChannelLayerNorm(channels, channels_first=False)
        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(1.0e-6 * torch.ones(channels))
        self.drop_path = (
            _DropPath(drop_path) if float(drop_path) > 0.0 else nn.Identity()
        )

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = self.dwconv(value)
        value = value.permute(0, 2, 3, 1)
        value = self.norm(value)
        value = self.pwconv1(value)
        value = self.act(value)
        value = self.pwconv2(value)
        value = self.gamma * value
        value = value.permute(0, 3, 1, 2)
        return residual + self.drop_path(value)


class SphericalCrossAttention(nn.Module):
    """Map one ConvNeXt frequency scale onto the shared spherical queries."""

    def __init__(
        self,
        query_dim: int,
        feature_dim: int,
        attention_dim: int,
        num_heads: int,
        ffn_multiplier: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(feature_dim)
        self.q_proj = nn.Linear(query_dim, attention_dim)
        self.k_proj = nn.Linear(feature_dim, attention_dim)
        self.v_proj = nn.Linear(feature_dim, attention_dim)
        self.out_proj = nn.Linear(attention_dim, query_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(attention_dim) // int(num_heads)
        self.scale = self.head_dim**-0.5
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        hidden_dim = int(query_dim * ffn_multiplier)
        self.ffn_norm = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: Tensor, features: Tensor) -> Tensor:
        normalized_query = self.norm_q(query)
        normalized_features = self.norm_kv(features)
        batch_size, query_count, _ = normalized_query.shape
        feature_count = normalized_features.shape[1]
        q = self.q_proj(normalized_query).view(
            batch_size, query_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(normalized_features).view(
            batch_size, feature_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(normalized_features).view(
            batch_size, feature_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        weights = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        weights = self.attn_drop(weights)
        update = weights @ v
        update = update.transpose(1, 2).contiguous().view(
            batch_size, query_count, self.num_heads * self.head_dim
        )
        query = query + self.proj_drop(self.out_proj(update))
        return query + self.ffn(self.ffn_norm(query))


@dataclass(frozen=True)
class Audio2SphOutput:
    """Audio2Sph outputs consumed by Sph2Imaging."""

    panoramic_features: Tensor
    y0: Tensor
    x1: Tensor
    x2: Tensor
    x3: Tensor
    stft_frame_count: int


class TimeAlignment(nn.Module):
    """Align Y0 to the output frame rate defined in the paper."""

    def __init__(self, output_stride: int) -> None:
        super().__init__()
        self.output_stride = int(output_stride)

    def forward(self, y0: Tensor, stft_frame_count: int) -> Tensor:
        target_frames = max(
            1,
            (int(stft_frame_count) + self.output_stride - 1)
            // self.output_stride,
        )
        if int(y0.shape[1]) == target_frames:
            return y0
        batch_size, frames, height, width, channels = y0.shape
        sequence = y0.permute(0, 2, 3, 4, 1).contiguous().view(
            batch_size * height * width * channels, 1, frames
        )
        mean = F.adaptive_avg_pool1d(sequence, target_frames)
        maximum = F.adaptive_max_pool1d(sequence, target_frames)
        sequence = 0.5 * (mean + maximum)
        return sequence.view(
            batch_size, height, width, channels, target_frames
        ).permute(0, 4, 1, 2, 3).contiguous()


class Audio2Sph(nn.Module):
    """Convert four-channel audio into frame-aligned panoramic features."""

    def __init__(
        self,
        *,
        audio_channels: int = 4,
        n_fft: int = 2048,
        hop_length: int = 480,
        output_stride: int = 10,
        sphere_height: int = 45,
        sphere_width: int = 90,
        panoramic_dim: int = 16,
        encoder_drop_path: float = 0.2,
    ) -> None:
        super().__init__()
        self.audio_channels = int(audio_channels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.output_stride = int(output_stride)
        self.sphere_height = int(sphere_height)
        self.sphere_width = int(sphere_width)
        self.panoramic_dim = int(panoramic_dim)

        dimensions = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.audio_channels * 3,
                dimensions[0],
                kernel_size=(3, 7),
                stride=(1, 4),
                padding=(1, 3),
            ),
            _ChannelLayerNorm(dimensions[0], channels_first=True),
        )
        self.downsample_layers = nn.ModuleList(
            [
                nn.Identity(),
                nn.Sequential(
                    _ChannelLayerNorm(dimensions[0], channels_first=True),
                    nn.Conv2d(
                        dimensions[0], dimensions[1], kernel_size=2, stride=2
                    ),
                ),
                nn.Sequential(
                    _ChannelLayerNorm(dimensions[1], channels_first=True),
                    nn.Conv2d(
                        dimensions[1],
                        dimensions[2],
                        kernel_size=(1, 2),
                        stride=(1, 2),
                    ),
                ),
                nn.Sequential(
                    _ChannelLayerNorm(dimensions[2], channels_first=True),
                    nn.Conv2d(
                        dimensions[2],
                        dimensions[3],
                        kernel_size=(1, 2),
                        stride=(1, 2),
                    ),
                ),
            ]
        )
        drop_rates = torch.linspace(
            0.0, float(encoder_drop_path), sum(depths)
        ).tolist()
        self.stages = nn.ModuleList()
        offset = 0
        for dimension, depth in zip(dimensions, depths):
            self.stages.append(
                nn.Sequential(
                    *[
                        _ConvNeXtBlock(dimension, drop_rates[offset + index])
                        for index in range(depth)
                    ]
                )
            )
            offset += depth

        self.grid_queries = nn.Parameter(
            torch.zeros(self.sphere_height * self.sphere_width, self.panoramic_dim)
        )
        nn.init.trunc_normal_(self.grid_queries, std=0.02)
        self.spherical_cross_attention = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SphericalCrossAttention(
                            self.panoramic_dim, feature_dim, 128, 4, 4.0, 0.0
                        )
                    ]
                )
                for feature_dim in dimensions[1:]
            ]
        )
        self.time_alignment = TimeAlignment(self.output_stride)

    def _time_frequency_input(self, audio: Tensor) -> tuple[Tensor, int]:
        if audio.ndim != 3:
            raise ValueError(f"Audio2Sph expects audio [B,C,L], got {tuple(audio.shape)}")
        if int(audio.shape[1]) != self.audio_channels:
            raise ValueError(
                f"Audio2Sph expects {self.audio_channels} channels, got {audio.shape[1]}"
            )
        batch_size, channels, samples = audio.shape
        flat_audio = audio.reshape(batch_size * channels, samples)
        window = torch.hann_window(self.n_fft).to(audio.device)
        spectrum = torch.stft(
            flat_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            normalized=True,
            return_complex=True,
        )
        spectrum = spectrum.reshape(
            batch_size, channels, spectrum.shape[-2], spectrum.shape[-1]
        ).permute(0, 1, 3, 2)
        frame_count = int(spectrum.shape[2])
        magnitude = spectrum.abs().clamp_min(1.0e-8)
        phase = torch.angle(spectrum)
        features = torch.cat(
            [torch.log(magnitude), torch.sin(phase), torch.cos(phase)], dim=1
        )
        features = features[..., :-1]
        pad_frames = math.ceil(frame_count / 2) * 2 - frame_count
        if pad_frames:
            features = F.pad(features, (0, 0, 0, pad_frames))
        return features, frame_count

    def _cross_attention(self, features: Tensor, level: int) -> Tensor:
        batch_size, channels, frames, frequencies = features.shape
        tokens = features.permute(0, 2, 3, 1).contiguous().view(
            batch_size * frames, frequencies, channels
        )
        query = self.grid_queries[None].repeat(batch_size * frames, 1, 1)
        for layer in self.spherical_cross_attention[level]:
            query = layer(query, tokens)
        return query.view(
            batch_size,
            frames,
            self.sphere_height * self.sphere_width,
            self.panoramic_dim,
        )

    def forward(self, audio: Tensor) -> Audio2SphOutput:
        time_frequency, stft_frame_count = self._time_frequency_input(audio)
        stem_features = self.stages[0](self.stem(time_frequency))
        x1 = self.stages[1](self.downsample_layers[1](stem_features))
        x2 = self.stages[2](self.downsample_layers[2](x1))
        x3 = self.stages[3](self.downsample_layers[3](x2))

        aligned_x1, aligned_x2, aligned_x3 = x1, x2, x3
        reference_frames = int(aligned_x3.shape[2])
        if int(aligned_x2.shape[2]) != reference_frames:
            aligned_x2 = F.interpolate(
                aligned_x2,
                size=(reference_frames, aligned_x2.shape[3]),
                mode="nearest",
            )
        if int(aligned_x1.shape[2]) != reference_frames:
            aligned_x1 = F.interpolate(
                aligned_x1,
                size=(reference_frames, aligned_x1.shape[3]),
                mode="nearest",
            )

        y1 = self._cross_attention(aligned_x1, 0)
        y2 = self._cross_attention(aligned_x2, 1)
        y3 = self._cross_attention(aligned_x3, 2)
        y0 = (y3 + y2 + y1).view(
            audio.shape[0],
            reference_frames,
            self.sphere_height,
            self.sphere_width,
            self.panoramic_dim,
        )
        panoramic_features = self.time_alignment(y0, stft_frame_count)
        return Audio2SphOutput(
            panoramic_features=panoramic_features,
            y0=y0,
            x1=x1,
            x2=x2,
            x3=x3,
            stft_frame_count=stft_frame_count,
        )
