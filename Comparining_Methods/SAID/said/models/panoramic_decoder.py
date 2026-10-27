"""Panoramic Decoder used for class-agnostic Audio2Sph pretraining."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .audio2sph import Audio2Sph, Audio2SphOutput


class _SphericalConvolution(nn.Module):
    """Convolution with circular azimuth and replicated elevation padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.padding = int(kernel_size) // 2
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=0
        )

    def forward(self, value: Tensor) -> Tensor:
        if self.padding:
            value = F.pad(
                value,
                (self.padding, self.padding, 0, 0),
                mode="circular",
            )
            value = F.pad(
                value,
                (0, 0, self.padding, self.padding),
                mode="replicate",
            )
        return self.convolution(value)


class _PanoramicConvolutionBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, *, normalize: bool
    ) -> None:
        super().__init__()
        self.convolution = _SphericalConvolution(
            in_channels, out_channels, kernel_size=3
        )
        if normalize:
            groups = min(8, int(out_channels))
            while groups > 1 and int(out_channels) % groups:
                groups -= 1
            self.normalization: nn.Module = nn.GroupNorm(groups, out_channels)
        else:
            self.normalization = nn.Identity()
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(self.normalization(self.convolution(value)))


@dataclass(frozen=True)
class PanoramicDecoderOutput:
    """Class-agnostic maps predicted during Audio2Sph pretraining."""

    class_agnostic_map_logits: Tensor
    class_agnostic_maps: Tensor


class PanoramicDecoder(nn.Module):
    """Decode Y0 into 180-by-360 class-agnostic acoustic maps."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution1 = _PanoramicConvolutionBlock(16, 16, normalize=True)
        self.convolution2 = _PanoramicConvolutionBlock(16, 8, normalize=True)
        self.convolution3 = _PanoramicConvolutionBlock(8, 4, normalize=False)
        self.output_convolution = nn.Conv2d(4, 1, kernel_size=1)

    @staticmethod
    def _align_time(value: Tensor, target_frames: int) -> Tensor:
        if int(value.shape[1]) == int(target_frames):
            return value
        batch_size, _, channels, height, width = value.shape
        value = value.permute(0, 2, 3, 4, 1).contiguous()
        value = F.interpolate(
            value,
            size=(height, width, int(target_frames)),
            mode="nearest",
        )
        return value.permute(0, 4, 1, 2, 3).contiguous().view(
            batch_size, int(target_frames), channels, height, width
        )

    def forward(self, y0: Tensor, *, target_frames: int) -> PanoramicDecoderOutput:
        if y0.ndim != 5 or int(y0.shape[-1]) != 16:
            raise ValueError("Y0 must have shape [B,T,H,W,16]")
        batch_size, frames, height, width, channels = y0.shape
        value = y0.permute(0, 1, 4, 2, 3).contiguous().view(
            batch_size * frames, channels, height, width
        )
        value = self.convolution1(value)
        value = F.interpolate(
            value, scale_factor=2, mode="bilinear", align_corners=False
        )
        value = self.convolution2(value)
        value = F.interpolate(
            value, scale_factor=2, mode="bilinear", align_corners=False
        )
        value = self.convolution3(value)
        logits = self.output_convolution(value).view(
            batch_size, frames, 1, value.shape[-2], value.shape[-1]
        )
        maps = logits.sigmoid()
        logits = self._align_time(logits, int(target_frames))[:, :, 0]
        maps = self._align_time(maps, int(target_frames))[:, :, 0]
        return PanoramicDecoderOutput(
            class_agnostic_map_logits=logits,
            class_agnostic_maps=maps,
        )


@dataclass(frozen=True)
class Audio2SphPretrainingOutput:
    """Audio2Sph features and their class-agnostic pretraining prediction."""

    audio2sph: Audio2SphOutput
    panoramic_decoder: PanoramicDecoderOutput


class Audio2SphPretrainingModel(nn.Module):
    """Paper pretraining model: Audio2Sph followed by Panoramic Decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.audio2sph = Audio2Sph()
        self.panoramic_decoder = PanoramicDecoder()

    def forward(self, audio: Tensor) -> Audio2SphPretrainingOutput:
        encoded = self.audio2sph(audio)
        decoded = self.panoramic_decoder(
            encoded.y0,
            target_frames=encoded.stft_frame_count,
        )
        return Audio2SphPretrainingOutput(
            audio2sph=encoded,
            panoramic_decoder=decoded,
        )
