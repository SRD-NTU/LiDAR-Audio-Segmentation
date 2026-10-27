"""Load Eigenmike audio for SAID inference."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
from torch import Tensor


EIGENMIKE_CAPSULES_1BASED = (6, 10, 26, 22)


def select_eigenmike_channels(audio: Tensor) -> Tensor:
    """Return the four Eigenmike capsule signals used by the paper."""

    if audio.ndim != 2:
        raise ValueError(f"audio must have shape [channels,samples], got {tuple(audio.shape)}")
    channels = int(audio.shape[0])
    if channels == 4:
        return audio
    if channels == 32:
        indices = torch.tensor(
            [index - 1 for index in EIGENMIKE_CAPSULES_1BASED],
            dtype=torch.long,
            device=audio.device,
        )
        return audio.index_select(0, indices)
    raise ValueError(
        "SAID expects four selected Eigenmike capsule signals or a complete "
        f"32-channel Eigenmike recording, got {channels} channels"
    )


def load_eigenmike_audio(
    path: str | Path,
    *,
    sample_rate: int = 48_000,
    offset_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> Tensor:
    """Load, resample, and select the paper's four Eigenmike capsules."""

    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio file not found: {audio_path}")
    if offset_seconds < 0:
        raise ValueError("offset_seconds must be non-negative")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    array, _ = librosa.load(
        audio_path,
        sr=int(sample_rate),
        mono=False,
        dtype=np.float32,
        offset=float(offset_seconds),
        duration=None if duration_seconds is None else float(duration_seconds),
    )
    if array.ndim == 1:
        array = array[None, :]
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    return select_eigenmike_channels(tensor).contiguous()
