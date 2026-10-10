"""Compact physics-conditioned range-image segmentation network."""
import torch
from torch import nn
from torch.nn import functional as F

def block(ch_in, ch_out):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
        nn.GroupNorm(8, ch_out), nn.SiLU(),
        nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
        nn.GroupNorm(8, ch_out), nn.SiLU(),
    )


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            block(13, 16), nn.AvgPool2d(2),
            block(16, 32), nn.AvgPool2d(2),
            block(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 64), nn.SiLU(), nn.Dropout(.4),
        )

    def forward(self, wave):
        batch = wave.shape[0]
        spectrum = torch.stft(wave.reshape(batch * 4, -1), n_fft=512,
                              hop_length=256, window=torch.hann_window(512, device=wave.device),
                              return_complex=True).reshape(batch, 4, 257, -1)
        magnitude = spectrum.abs().clamp_min(1e-6)
        log_magnitude = torch.log1p(magnitude)
        cross = spectrum[:, 1:] * spectrum[:, :1].conj()
        phase = cross / cross.abs().clamp_min(1e-6)
        ild = log_magnitude[:, 1:] - log_magnitude[:, :1]
        return self.net(torch.cat((log_magnitude, ild, phase.real, phase.imag), dim=1))


class AudioLidarV2(nn.Module):
    def __init__(self, audio_branch=True):
        super().__init__()
        self.audio_branch = audio_branch
        self.audio = AudioEncoder() if audio_branch else None
        self.enc1 = block(9, 24)
        self.enc2 = block(24, 48)
        self.enc3 = block(48, 96)
        self.center = block(96, 128)
        self.dec3 = block(128 + 96, 96)
        self.dec2 = block(96 + 48, 48)
        self.dec1 = block(48 + 24, 24)
        self.film = nn.ModuleList([nn.Linear(64, 2 * c) for c in (128, 96, 48, 24)]) if audio_branch else None
        if self.film is not None:
            for layer in self.film:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.head = nn.Conv2d(24, 1, 1)

    def condition(self, x, audio, layer):
        if audio is None:
            return x
        scale, shift = self.film[layer](audio).chunk(2, dim=1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, grid, wave):
        audio = self.audio(wave) if self.audio_branch else None
        x1 = self.enc1(grid)
        x2 = self.enc2(F.avg_pool2d(x1, 2))
        x3 = self.enc3(F.avg_pool2d(x2, 2))
        x = self.condition(self.center(F.avg_pool2d(x3, 2)), audio, 0)
        x = self.condition(self.dec3(torch.cat((F.interpolate(x, size=x3.shape[-2:], mode="bilinear", align_corners=False), x3), 1)), audio, 1)
        x = self.condition(self.dec2(torch.cat((F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False), x2), 1)), audio, 2)
        x = self.condition(self.dec1(torch.cat((F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1), 1)), audio, 3)
        return self.head(x).squeeze(1)
