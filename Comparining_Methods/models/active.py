"""MAGNet scene-memory component adapted to a finite recorded pose menu."""
from __future__ import annotations

import math

import torch
from torch import nn

from models.passive import _load_class, AudioConv, PointContext, downsample_audio, spectral_ten


class MAGNetFiniteMenu(nn.Module):
    """New policy objective; reuses MAGNet's released scene-memory encoder.

    The policy receives only the initial waveform/cloud and pose-only future
    menu. No future audio tensor is accepted by forward().
    """
    def __init__(self):
        super().__init__()
        StateEncoder = _load_class(
            "SAVN-CE/savnce_baselines/magnet/models/savnce_state_encoder.py",
            "SAVNCE_StateEncoder")
        self.audio = AudioConv()
        self.geometry = PointContext()
        self.memory = StateEncoder(feature_size=384, embedding_size=128, nhead=4,
                                   num_encoder_layers=1, num_decoder_layers=1)
        self.pose = nn.Sequential(nn.Linear(13, 64), nn.ReLU(), nn.Linear(64, 128))
        self.action = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, context_xyz: torch.Tensor, initial_audio: torch.Tensor,
                pose_features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if pose_features.ndim != 3 or pose_features.shape[-1] != 13:
            raise ValueError("Candidate poses must be [B,K,13]")
        if valid.shape != pose_features.shape[:2] or not valid.any(dim=1).all():
            raise ValueError("Every episode needs at least one valid action")
        acoustic = self.audio(spectral_ten(downsample_audio(initial_audio)))
        geometry = self.geometry.encode_context(context_xyz)
        current = torch.cat([acoustic, geometry], dim=-1)
        empty_memory = current.new_empty((0, len(current), current.shape[-1]))
        empty_mask = current.new_empty((len(current), 0))
        state = self.memory(current, empty_memory, empty_mask)
        actions = self.pose(pose_features)
        score = self.action(torch.cat([actions, state[:, None].expand_as(actions)], dim=-1)).squeeze(-1)
        return score.masked_fill(~valid, torch.finfo(score.dtype).min)
