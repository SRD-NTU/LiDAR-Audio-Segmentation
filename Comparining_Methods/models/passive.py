"""Trainable passive LiDAR/audio variants using published repository components.

All four models take only raw XYZ, four measured microphone signals and fixed
array calibration. They are new adaptations, never claimed as unchanged paper
models. Original repository files remain untouched.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

# Component imports point into the original repositories. Keep those trees
# byte-for-byte free of generated import caches even in ad hoc Python runs.
sys.dont_write_bytecode = True

import torch
import torchaudio.functional as AF
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
PASSIVE = ("echosegnet", "jaeger", "said", "sound3dvdet")


def _load_class(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(f"adapted_{path.stem}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, name)


def downsample_audio(wave: torch.Tensor, output_samples: int = 32000) -> torch.Tensor:
    """Bandlimited 44.1-to-16 kHz resampling for adapted encoders."""
    if wave.ndim != 3 or wave.shape[-1] != 4:
        raise ValueError("Audio must be [B,T,4] in tetra_a,b,c,d order")
    # torchaudio's sinc kernel uses view() internally and needs contiguous input.
    x = wave.transpose(1, 2).contiguous()
    x = AF.resample(x, orig_freq=44100, new_freq=16000)
    return F.pad(x[..., :output_samples], (0, max(0, output_samples - x.shape[-1])))


def spectral_ten(wave_16k: torch.Tensor) -> torch.Tensor:
    """4 log magnitudes and 6 inter-microphone phase-coherence maps."""
    b, c, t = wave_16k.shape
    flat = wave_16k.reshape(b * c, t)
    window = torch.hann_window(256, device=flat.device, dtype=flat.dtype)
    spectrum = torch.stft(flat, n_fft=256, hop_length=256, window=window,
                          return_complex=True).reshape(b, c, 129, -1)[:, :, :128, :128]
    if spectrum.shape[-1] < 128:
        spectrum = F.pad(spectrum, (0, 128 - spectrum.shape[-1]))
    mag = torch.log1p(spectrum.abs())
    unit = spectrum / spectrum.abs().clamp_min(1e-6)
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    phase = [torch.real(unit[:, i] * unit[:, j].conj()).unsqueeze(1) for i, j in pairs]
    return torch.cat([mag, *phase], dim=1)


def tetra_to_lowband_foa(wave_16k: torch.Tensor, microphone_positions_m: torch.Tensor,
                         sample_rate: int = 16000, cutoff_hz: float = 1500.0) -> torch.Tensor:
    """First-order pressure-gradient approximation, *not* measured FOA.

    Fit p(r,t)=p0(t)+r·grad_p(t) at the four calibrated point microphones.
    For a low-frequency plane wave, the source-facing FOA direction is
    approximately +c ∫ grad_p dt (opposite wave propagation velocity).
    Frequencies above 1.5 kHz are suppressed; the inverse is
    regularized below 80 Hz. The output convention is [W,X,Y,Z] in LiDAR axes
    and must be trained on this dataset; published FOA weights are not reused.
    """
    if wave_16k.shape[1] != 4 or microphone_positions_m.shape != (4, 3):
        raise ValueError("Expected four audio channels and calibrated 4x3 array")
    matrix = torch.cat([torch.ones((4, 1), device=wave_16k.device,
                                    dtype=wave_16k.dtype), microphone_positions_m], dim=1)
    components = torch.linalg.pinv(matrix) @ wave_16k
    p0, gradient = components[:, :1], components[:, 1:]
    n = wave_16k.shape[-1]
    frequencies = torch.fft.rfftfreq(n, 1 / sample_rate).to(wave_16k.device)
    omega = 2 * torch.pi * frequencies
    regularizer = (2 * torch.pi * 80.0) ** 2
    inverse_derivative = (-1j * omega / (omega.square() + regularizer))[None, None]
    passband = (frequencies <= cutoff_hz)[None, None]
    directional = torch.fft.irfft(343.0 * torch.fft.rfft(gradient) * inverse_derivative * passband,
                                  n=n).real
    return torch.cat([p0, directional], dim=1).contiguous()


def point_features(xyz: torch.Tensor) -> torch.Tensor:
    radius = xyz.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    direction = xyz / radius
    return torch.cat([xyz / 20.0, direction, radius / 20.0], dim=-1)


class PointContext(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(7, 64), nn.ReLU(), nn.Linear(64, width), nn.ReLU())

    def encode_context(self, context_xyz: torch.Tensor):
        context = self.point(point_features(context_xyz))
        return torch.cat([context.mean(1), context.amax(1)], dim=-1)

    def encode_query(self, query_xyz: torch.Tensor):
        return self.point(point_features(query_xyz))


class AudioConv(nn.Module):
    def __init__(self, input_channels: int = 10, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class PassiveAdapter(nn.Module):
    """Method-specific audio mechanism plus a shared LiDAR point-mask decoder."""
    def __init__(self, method: str, microphone_positions_m, said_grid=(9, 18)):
        super().__init__()
        if method not in PASSIVE:
            raise ValueError(f"Unknown passive adaptation: {method}")
        self.method = method
        self.register_buffer("microphones", torch.as_tensor(microphone_positions_m, dtype=torch.float32))
        self.geometry = PointContext()
        self.audio_projection = None
        self.audio_queries_projection = None
        self.cross_attention = None
        self.map_head = None
        if method == "echosegnet":
            self.audio_encoder = AudioConv()
            self.film = nn.Linear(128, 256)
        elif method == "jaeger":
            NeuralIV = _load_class("JAEGER/models/FOA_IV_encoder.py", "NeuralIV")
            self.audio_encoder = NeuralIV(4, 32, [(9, 5), (5, 4), (5, 4), (5, 4)])
            self.audio_projection = nn.Linear(96, 128)
        elif method == "said":
            Audio2Sph = _load_class("SAID/said/models/audio2sph.py", "Audio2Sph")
            self.audio_encoder = Audio2Sph(n_fft=256, hop_length=160, output_stride=10,
                                            sphere_height=said_grid[0], sphere_width=said_grid[1],
                                            encoder_drop_path=0.0)
            self.map_head = nn.Conv2d(16, 1, 1)
            self.audio_projection = nn.Linear(16, 128)
        else:
            MicArraySSDetector = _load_class("Sound3DVDet/MicArraySSDet.py", "MicArraySSDetector")
            # Reuse the six released microphone convolution blocks; bypass the
            # original forward's interactive breakpoint and broken mapper.
            original = MicArraySSDetector(TF_map_height=128, TF_map_width=128)
            self.audio_encoder = nn.Sequential(original.conv_encoder0, original.conv_encoder1,
                                               original.conv_encoder2, original.conv_encoder3,
                                               original.conv_encoder4, original.conv_encoder5)
            self.audio_queries_projection = nn.Linear(512, 128)
            self.cross_attention = nn.MultiheadAttention(128, 4, batch_first=True)
            self.audio_projection = nn.Linear(512, 128)
        extra = 1 if method == "said" else 0
        self.mask_head = nn.Sequential(nn.Linear(128 + 256 + 128 + extra, 128), nn.ReLU(),
                                       nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    @staticmethod
    def _sample_map(map_logits: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
        radius = query_xyz.norm(dim=-1).clamp_min(1e-6)
        azimuth = torch.atan2(query_xyz[..., 1], query_xyz[..., 0])
        elevation = torch.asin((query_xyz[..., 2] / radius).clamp(-1, 1))
        # Pad azimuth cyclically to make the -pi/pi seam continuous.
        padded = F.pad(map_logits, (1, 1, 0, 0), mode="circular")
        w = map_logits.shape[-1]
        column = (azimuth / (2 * torch.pi) + .5) * (w - 1) + 1
        grid_x = 2 * column / (w + 1) - 1
        grid_y = -2 * elevation / torch.pi
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)
        return F.grid_sample(padded, grid, mode="bilinear", align_corners=True,
                             padding_mode="border")[:, 0, :, 0].unsqueeze(-1)

    def encode_observation(self, context_xyz: torch.Tensor, audio: torch.Tensor) -> dict:
        wave = downsample_audio(audio)
        context = self.geometry.encode_context(context_xyz)
        map_logits = None
        queries = None
        film = None
        if self.method == "echosegnet":
            acoustic = self.audio_encoder(spectral_ten(wave))
            film = self.film(acoustic).chunk(2, dim=-1)
        elif self.method == "jaeger":
            foa = tetra_to_lowband_foa(wave, self.microphones)
            acoustic = self.audio_projection(self.audio_encoder(foa).mean(1))
        elif self.method == "said":
            panoramic = self.audio_encoder(wave).y0.mean(dim=1)
            map_logits = self.map_head(panoramic.permute(0, 3, 1, 2))
            acoustic = self.audio_projection(panoramic.mean(dim=(1, 2)))
        else:
            feature = self.audio_encoder(spectral_ten(wave))
            acoustic = self.audio_projection(feature.mean(dim=(-1, -2)))
            queries = self.audio_queries_projection(feature.flatten(2).transpose(1, 2))
        return {"context": context, "acoustic": acoustic, "map_logits": map_logits,
                "audio_queries": queries, "film": film}

    def score_queries(self, state: dict, query_xyz: torch.Tensor) -> torch.Tensor:
        point = self.geometry.encode_query(query_xyz)
        if state["film"] is not None:
            scale, shift = state["film"]
            point = point * (1 + torch.tanh(scale[:, None])) + shift[:, None]
        if state["audio_queries"] is not None:
            queries = state["audio_queries"]
            point = point + self.cross_attention(point, queries, queries, need_weights=False)[0]
        features = [point, state["context"][:, None].expand(-1, point.shape[1], -1),
                    state["acoustic"][:, None].expand(-1, point.shape[1], -1)]
        if state["map_logits"] is not None:
            features.append(self._sample_map(state["map_logits"], query_xyz))
        return self.mask_head(torch.cat(features, dim=-1)).squeeze(-1)

    def forward(self, context_xyz: torch.Tensor, query_xyz: torch.Tensor,
                audio: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.encode_observation(context_xyz, audio)
        return {"point_logits": self.score_queries(state, query_xyz),
                "map_logits": state["map_logits"]}


def make_passive(method: str, microphone_positions_m) -> PassiveAdapter:
    return PassiveAdapter(method, microphone_positions_m)
