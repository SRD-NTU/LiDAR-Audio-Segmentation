from __future__ import annotations

import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F


MIC_POSITIONS = torch.tensor(
    [[0.04, 0.04, 0.0], [0.04, -0.04, 0.0], [-0.04, 0.04, 0.0], [-0.04, -0.04, 0.0]],
    dtype=torch.float32,
)
MIC_PAIRS = tuple(itertools.combinations(range(4), 2))


class AudioSpatialEncoder(nn.Module):
    """Learns audio semantics while retaining explicit IPD, ILD and GCC-PHAT cues."""

    def __init__(
        self,
        dim: int = 160,
        sample_rate: int = 14700,
        n_fft: int = 256,
        hop_length: int = 128,
        doa_bins: int = 225,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.doa_bins = doa_bins
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("mic_positions", MIC_POSITIONS.clone(), persistent=True)
        in_channels = 4 + 3 * len(MIC_PAIRS)  # magnitude + sin/cos IPD + ILD
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 40, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 40), nn.GELU(),
            nn.Conv2d(40, 80, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 80), nn.GELU(),
            nn.Conv2d(80, dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(),
        )
        self.global_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.level_head = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        batch, channels, samples = audio.shape
        spec = torch.stft(
            audio.reshape(batch * channels, samples), n_fft=self.n_fft,
            hop_length=self.hop_length, window=self.window.to(audio),
            center=True, return_complex=True,
        )
        return spec.reshape(batch, channels, spec.shape[-2], spec.shape[-1])

    def forward(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        # FFT kernels are substantially more stable in fp32 even under AMP.
        with torch.autocast(device_type=audio.device.type, enabled=False):
            audio = audio.float()
            rms = audio.square().mean(dim=(-2, -1), keepdim=True).add(1e-10).sqrt()
            log_level = rms.log10().flatten(1).clamp(-6.0, 1.0)
            spec = self._stft(audio / rms.clamp_min(1e-5))
            magnitude = spec.abs().clamp_min(1e-5)
            logmag = torch.log1p(magnitude)
            # Per-example centering retains spectral shape and channel differences.
            logmag = logmag - logmag.mean(dim=(-2, -1), keepdim=True)
            features = [logmag]
            for left, right in MIC_PAIRS:
                cross = spec[:, left] * spec[:, right].conj()
                unit = cross / cross.abs().clamp_min(1e-5)
                ild = (magnitude[:, left].log() - magnitude[:, right].log()).clamp(-6.0, 6.0)
                features.extend((unit.real[:, None], unit.imag[:, None], ild[:, None]))
            spatial_features = torch.cat(features, dim=1)
            doa = self.gcc_phat(spec)

        feature_map = self.encoder(spatial_features)
        tokens = feature_map.flatten(2).transpose(1, 2)
        global_embedding = self.global_head(feature_map.mean(dim=(-2, -1))) + self.level_head(log_level)
        return {"tokens": tokens, "global": global_embedding, "doa": doa, "spectrum": spec}

    def gcc_phat(self, spec: torch.Tensor) -> torch.Tensor:
        """Far-field steered response over LiDAR azimuth (x-forward, y-left)."""
        device = spec.device
        dtype = spec.real.dtype
        frequencies = torch.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate).to(device=device, dtype=dtype)
        # Spatial aliasing becomes severe above ~4 kHz for an 8 cm aperture.
        freq_mask = (frequencies >= 200.0) & (frequencies <= 4000.0)
        frequencies = frequencies[freq_mask]
        angles = torch.arange(self.doa_bins, device=device, dtype=dtype) * (2.0 * math.pi / self.doa_bins)
        directions = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), dim=-1)
        scores = []
        for left, right in MIC_PAIRS:
            cross = spec[:, left, freq_mask] * spec[:, right, freq_mask].conj()
            phat = cross / cross.abs().clamp_min(1e-5)
            phat = phat.mean(dim=-1)
            baseline = (self.mic_positions[right] - self.mic_positions[left]).to(dtype)
            delay = directions @ baseline / 343.0
            steering = torch.exp(2j * math.pi * frequencies[:, None] * delay[None])
            scores.append(torch.einsum("bf,fa->ba", phat, steering).real / frequencies.numel())
        score = torch.stack(scores).mean(dim=0)
        score = (score - score.mean(dim=-1, keepdim=True)) / (score.std(dim=-1, keepdim=True) + 1e-5)
        return score
