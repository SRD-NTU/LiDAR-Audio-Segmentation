"""The only intended inference loader; evaluator metadata is never returned."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Observation:
    xyz: np.ndarray
    beam_id: np.ndarray
    audio: np.ndarray
    sample_rate: int


def load_observation(sample: Path) -> Observation:
    with np.load(Path(sample)/'lidar.npz',allow_pickle=False) as p:
        xyz=p['xyz'].copy(); beam=p['beam_id'].copy()
    audio,rate=sf.read(Path(sample)/'audio.wav',always_2d=True,dtype='float32')
    return Observation(xyz,beam,audio,rate)
