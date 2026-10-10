"""Load the portable AudioLiDAR package without evaluator metadata as inputs."""
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset

BEAMS = 64
STEPS = 1800


def load_sample(root, sample, labeled=True):
    path = Path(root) / sample
    with np.load(path / "lidar.npz", allow_pickle=False) as archive:
        xyz = archive["xyz"].astype(np.float32)
        ray_id = archive["ray_id"].astype(np.int64)
        labels = archive["sound_mask"].astype(np.float32) if labeled else None
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(ray_id) != len(xyz):
        raise ValueError(f"Invalid LiDAR arrays: {path}")
    if len(np.unique(ray_id)) != len(ray_id) or np.any((ray_id < 0) | (ray_id >= BEAMS * STEPS)):
        raise ValueError(f"Invalid ray_id mapping: {path}")
    if not np.isfinite(xyz).all():
        raise ValueError(f"Nonfinite LiDAR: {path}")
    grid = np.zeros((5, STEPS * BEAMS), dtype=np.float32)
    grid[:3, ray_id] = np.clip(xyz.T / 10.0, -5.0, 5.0)
    grid[3, ray_id] = np.log1p(np.linalg.norm(xyz, axis=1)) / 4.0
    grid[4, ray_id] = 1.0
    grid = grid.reshape(5, STEPS, BEAMS).transpose(0, 2, 1).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        rate, audio = wavfile.read(path / "audio.wav")
    if rate != 44100 or audio.shape != (88200, 4) or audio.dtype != np.float32:
        raise ValueError(f"Expected 2 s, 44.1 kHz, four-channel float32 audio: {path}")
    # Synchronous stride-2 sampling preserves the four-channel sample clock.
    audio = np.ascontiguousarray(audio[::2].T)
    if not np.isfinite(audio).all():
        raise ValueError(f"Nonfinite audio: {path}")
    result = {"grid": grid, "audio": audio, "ray_id": ray_id, "sample": sample}
    if labeled:
        target = np.zeros(STEPS * BEAMS, dtype=np.float32)
        target[ray_id] = labels
        result["target"] = target.reshape(STEPS, BEAMS).T.copy()
    return result


class AudioLidarDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.samples = json.loads((self.root / "metadata" / f"{split}.json").read_text())
        if not isinstance(self.samples, list) or not self.samples:
            raise ValueError(f"Empty {split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = load_sample(self.root, self.samples[index])
        # Fixed size tensors are stackable, while ray_id varies by sample.
        return item["grid"], item["audio"], item["target"], item["sample"]
