from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .audio import AudioSpatialEncoder


def circular_pad(x: torch.Tensor, padding: int = 1) -> torch.Tensor:
    x = F.pad(x, (padding, padding, 0, 0), mode="circular")
    return F.pad(x, (0, 0, padding, padding), mode="replicate")


class CircularConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, **kwargs) -> None:
        if kernel_size % 2 != 1:
            raise ValueError("CircularConv2d requires an odd kernel")
        self.circular_padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=0, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(circular_pad(x, self.circular_padding))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.conv1 = CircularConv2d(in_channels, out_channels, stride=stride, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = CircularConv2d(out_channels, out_channels, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.skip = (
            nn.Identity() if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        return F.gelu(self.norm2(self.conv2(x)) + residual)


class AudioFiLM(nn.Module):
    def __init__(self, audio_dim: int, channels: int) -> None:
        super().__init__()
        self.affine = nn.Linear(audio_dim, channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(audio).chunk(2, dim=-1)
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, audio_dim: int) -> None:
        super().__init__()
        self.block = ResidualBlock(in_channels + skip_channels, out_channels)
        self.refine = ResidualBlock(out_channels, out_channels)
        self.film = AudioFiLM(audio_dim, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.block(torch.cat((x, skip), dim=1))
        return self.refine(self.film(x, audio))


class AcousticWorldModel(nn.Module):
    """Circular LiDAR U-Net with learned/physical audio fusion and latent rendering.

    The counterfactual renderer maps every bottleneck LiDAR region to the audio
    latent it would be expected to emit. Cosine agreement with observed audio is
    both exposed for supervision and fed back into the point-mask decoder.
    """

    def __init__(
        self,
        base_channels: int = 32,
        audio_dim: int = 160,
        num_categories: int = 16,
        lidar_width: int = 1800,
        audio_sample_rate: int = 14700,
        n_fft: int = 256,
        hop_length: int = 128,
        attention_heads: int = 5,
    ) -> None:
        super().__init__()
        channels = (base_channels, base_channels * 2, base_channels * 3, audio_dim)
        bottleneck_width = math.ceil(math.ceil(math.ceil(lidar_width / 2) / 2) / 2)
        self.bottleneck_width = bottleneck_width
        self.audio_encoder = AudioSpatialEncoder(
            dim=audio_dim, sample_rate=audio_sample_rate, n_fft=n_fft,
            hop_length=hop_length, doa_bins=bottleneck_width,
        )

        # This branch is deliberately audio-free so counterfactual rendering
        # cannot copy the observation it is supposed to explain.
        self.stem = nn.Sequential(ResidualBlock(8, channels[0]), ResidualBlock(channels[0], channels[0]))
        self.down1 = ResidualBlock(channels[0], channels[1], stride=2)
        self.down2 = ResidualBlock(channels[1], channels[2], stride=2)
        self.down3 = ResidualBlock(channels[2], channels[3], stride=2)

        self.query_norm = nn.LayerNorm(audio_dim)
        self.audio_norm = nn.LayerNorm(audio_dim)
        self.cross_attention = nn.MultiheadAttention(
            audio_dim, attention_heads, dropout=0.1, batch_first=True,
        )
        self.attention_scale = nn.Parameter(torch.tensor(0.1))

        self.renderer = nn.Sequential(
            nn.Linear(audio_dim, audio_dim), nn.GELU(), nn.Linear(audio_dim, audio_dim),
        )
        self.observation_projection = nn.Linear(audio_dim, audio_dim)
        self.position_head = nn.Sequential(
            nn.Linear(audio_dim, audio_dim), nn.GELU(), nn.Linear(audio_dim, 3), nn.Tanh(),
        )
        self.render_temperature = nn.Parameter(torch.tensor(math.log(10.0)))
        self.render_fusion = ResidualBlock(audio_dim + 3, audio_dim)

        self.up2 = DecoderBlock(channels[3], channels[2], channels[2], audio_dim)
        self.up1 = DecoderBlock(channels[2], channels[1], channels[1], audio_dim)
        self.up0 = DecoderBlock(channels[1], channels[0], channels[0], audio_dim)
        self.mask_head = nn.Sequential(
            ResidualBlock(channels[0], channels[0]), nn.Conv2d(channels[0], 1, 1),
        )
        self.semantic_head = nn.Sequential(
            ResidualBlock(channels[0], channels[0]), nn.Conv2d(channels[0], num_categories + 1, 1),
        )
        self.semantic_match_scale = nn.Parameter(torch.tensor(0.1))
        self.category_head = nn.Linear(audio_dim, num_categories)
        self.doa_residual = nn.Linear(audio_dim, bottleneck_width)

    def forward(
        self, lidar: torch.Tensor, audio: torch.Tensor, valid: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if valid is None:
            valid = torch.ones_like(lidar[:, :1])
        audio_state = self.audio_encoder(audio)
        audio_global = audio_state["global"]
        category_logits = self.category_head(audio_global)
        source_position = self.position_head(audio_global)
        doa_bottleneck = audio_state["doa"] + self.doa_residual(audio_global)
        x0 = self.stem(torch.cat((lidar * valid, valid), dim=1))
        semantic_logits = self.semantic_head(x0)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        shape = x3.shape
        geometry_tokens = x3.flatten(2).transpose(1, 2)
        # Render before observation conditioning: each region independently
        # predicts an acoustic latent from scene geometry alone.
        rendered_audio = F.normalize(self.renderer(geometry_tokens), dim=-1)
        observed_audio = F.normalize(self.observation_projection(audio_global), dim=-1)
        render_logits = torch.einsum("bnd,bd->bn", rendered_audio, observed_audio)
        render_logits = render_logits * self.render_temperature.exp().clamp(1.0, 50.0)
        render_map = render_logits.reshape(shape[0], shape[2], shape[3])

        attended, _ = self.cross_attention(
            self.query_norm(geometry_tokens), self.audio_norm(audio_state["tokens"]),
            self.audio_norm(audio_state["tokens"]), need_weights=False,
        )
        geometry_tokens = geometry_tokens + self.attention_scale.tanh() * attended
        x3 = geometry_tokens.transpose(1, 2).reshape(shape)
        doa_map = doa_bottleneck[:, None, None].expand(-1, -1, shape[2], -1)
        coarse_valid = F.adaptive_avg_pool2d(valid, shape[-2:])
        coarse_xyz = F.adaptive_avg_pool2d(lidar[:, :3] * valid, shape[-2:]) / coarse_valid.clamp_min(1e-3)
        position_map = -(coarse_xyz - source_position[:, :, None, None]).square().sum(dim=1, keepdim=True).sqrt() / 0.1
        x3 = self.render_fusion(torch.cat((x3, render_map[:, None], doa_map, position_map), dim=1))

        x = self.up2(x3, x2, audio_global)
        x = self.up1(x, x1, audio_global)
        x = self.up0(x, x0, audio_global)
        category_probability = category_logits.softmax(dim=-1)
        semantic_match = torch.einsum("bchw,bc->bhw", semantic_logits[:, 1:], category_probability)[:, None]
        mask_logits = (self.mask_head(x) + self.semantic_match_scale * semantic_match) * valid
        return {
            "mask_logits": mask_logits,
            "render_logits": render_map,
            "rendered_audio": rendered_audio,
            "observed_audio": observed_audio,
            "category_logits": category_logits,
            "semantic_logits": semantic_logits,
            "source_position": source_position,
            "doa_logits": doa_bottleneck,
            "physics_doa": audio_state["doa"],
        }
