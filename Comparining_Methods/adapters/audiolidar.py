"""Read-only observations and explicit, evaluation-only targets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_ROOT = Path("/projects/soumyaratna/LiDAR_Segmentation/dataset/v2-tetra4-4channel-full")


class AudioLiDAR:
    def __init__(self, root: Path = DEFAULT_ROOT):
        self.root = Path(root).resolve()
        self.calibration = json.loads((self.root / "metadata/microphone_array.json").read_text())
        if self.calibration["frame"] != "x-forward_y-left_z-up":
            raise ValueError("Unsupported LiDAR frame")
        self._splits = {
            split: json.loads((self.root / f"metadata/{split}.json").read_text())
            for split in ("train", "val", "test")
        }
        self._samples = {sample for paths in self._splits.values() for sample in paths}

    def samples(self, split: str) -> list[str]:
        return list(self._splits[split])

    def folder(self, sample: str) -> Path:
        if sample not in self._samples:
            raise ValueError(f"Sample is absent from split manifests: {sample}")
        return self.root / sample

    def receiver_pose(self, sample: str) -> np.ndarray:
        """Rigid transform from LiDAR coordinates to the world frame.

        Only receiver and fixed array calibration fields are read. The rest of
        metadata.json, including source position, must never reach a model.
        """
        metadata = json.loads((self.folder(sample) / "metadata.json").read_text())
        center = np.asarray(metadata["listener"]["position_world"], dtype=np.float64)
        local = np.asarray(self.calibration["positions_lidar_m"], dtype=np.float64)
        world = np.asarray(metadata["audio"]["microphone_array"]["positions_world_m"], dtype=np.float64)
        rotation_t, _, rank, _ = np.linalg.lstsq(local, world - center, rcond=None)
        rotation = rotation_t.T
        if rank != 3 or not np.allclose(local @ rotation.T + center, world, atol=1e-5):
            raise ValueError(f"Invalid array pose: {sample}")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError(f"Non-rigid receiver transform: {sample}")
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3], pose[:3, 3] = rotation, center
        return pose

    def observation(self, sample: str) -> dict:
        folder = self.folder(sample)
        with np.load(folder / "lidar.npz", allow_pickle=False) as data:
            xyz = data["xyz"].astype(np.float32)
        audio, rate = sf.read(folder / "audio.wav", dtype="float32", always_2d=True)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or audio.shape != (88200, 4) or rate != 44100:
            raise ValueError(f"Unexpected observation shape: {sample}")
        return {"xyz": xyz, "audio": audio, "sample_rate": rate,
                "receiver_to_world": self.receiver_pose(sample),
                "microphone_positions_lidar_m": np.asarray(self.calibration["positions_lidar_m"], dtype=np.float64)}

    def target_for_evaluation(self, sample: str) -> dict:
        """Offline scorer only. Never pass this result into model inference."""
        with np.load(self.folder(sample) / "lidar.npz", allow_pickle=False) as data:
            mask = data["sound_mask"].astype(bool)
            instance = data["instance_id"].astype(np.int32)
        return {"sound_mask": mask, "instance_id": instance}

    def recorded_views(self, sample: str) -> list[str]:
        """Same-source, same-recording views; source-conditioned protocol only."""
        folder = self.folder(sample)
        initial = json.loads((folder / "metadata.json").read_text())
        recording = initial["audio"]["dry_audio_sha256"]
        prefix = "/".join(sample.split("/")[:3])
        result = []
        for other in sorted(folder.parent.iterdir()):
            if not other.is_dir() or other == folder or not other.name.isdigit():
                continue
            metadata = json.loads((other / "metadata.json").read_text())
            if metadata["audio"]["dry_audio_sha256"] == recording:
                result.append(prefix + "/" + other.name)
        return result
