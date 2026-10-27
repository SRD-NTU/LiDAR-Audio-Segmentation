"""Pose-only menus and TRAIN/VAL-only parallax teacher for MAGNet adaptation."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from adapters.audiolidar import AudioLiDAR


def pose_only_menu(data: AudioLiDAR, sample: str, max_distance_m: float = 3.0) -> list[dict]:
    """This returns no future audio or target field to the policy."""
    initial = data.receiver_pose(sample)
    output = []
    for future in data.recorded_views(sample):
        pose = data.receiver_pose(future)
        translation = initial[:3, :3].T @ (pose[:3, 3] - initial[:3, 3])
        distance = float(np.linalg.norm(translation))
        if distance > max_distance_m or distance < 1e-5:
            continue
        rotation = initial[:3, :3].T @ pose[:3, :3]
        features = np.concatenate([translation / 3.0, rotation.reshape(-1), [distance / 3.0]])
        output.append({"sample": future, "pose": pose, "features": features.astype(np.float32),
                       "distance_m": distance})
    return output


def train_parallax_teacher(data: AudioLiDAR, sample: str, menu: list[dict]) -> int:
    """Offline TRAIN/VAL imitation target. Never call on TEST/inference."""
    if sample.split("/")[0] not in ("train", "val"):
        raise ValueError("Parallax teacher cannot read TEST source positions")
    if not menu:
        raise ValueError("No valid menu entries")
    metadata = json.loads((data.folder(sample) / "metadata.json").read_text())
    source = np.asarray(metadata["source"]["position_world"], dtype=np.float64)
    initial = data.receiver_pose(sample)
    initial_ray = source - initial[:3, 3]
    initial_ray /= np.linalg.norm(initial_ray)
    scores = []
    for item in menu:
        future_ray = source - item["pose"][:3, 3]
        future_ray /= np.linalg.norm(future_ray)
        angle = math.acos(float(np.clip(initial_ray @ future_ray, -1, 1)))
        scores.append(angle - .02 * item["distance_m"])
    return int(np.argmax(scores))
