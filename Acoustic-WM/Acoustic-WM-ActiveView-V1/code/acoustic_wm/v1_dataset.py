"""Recorded-branch menus built from calibrated poses, without source filtering."""
import json
from pathlib import Path

import numpy as np


def calibrated_pose(metadata):
    center = np.asarray(metadata["listener"]["position_world"], np.float64)
    microphones = metadata["audio"]["microphone_array"]
    local = np.asarray(microphones["positions_lidar_m"], np.float64)
    world = np.asarray(microphones["positions_world_m"], np.float64)
    rotation_t, _, rank, _ = np.linalg.lstsq(local, world - center, rcond=None)
    if rank != 3 or not np.allclose(local @ rotation_t + center, world, atol=1e-5):
        raise ValueError("Microphone calibration does not define a rigid receiver pose")
    pose = np.eye(4)
    pose[:3, :3] = rotation_t.T
    pose[:3, 3] = center
    return pose


def episode_records(data_root, sample):
    data_root = Path(data_root).resolve()
    reference = data_root / sample
    all_folders = sorted(p for p in reference.parent.iterdir() if p.is_dir() and p.name.isdigit())
    if not all_folders or reference not in all_folders:
        raise ValueError(f"Incomplete or missing episode: {sample}")
    reference_metadata = json.loads((reference / "metadata.json").read_text())
    activity = reference_metadata["audio"]["dry_audio_sha256"]
    paired = [(p, json.loads((p / "metadata.json").read_text())) for p in all_folders]
    paired = [(p, m) for p, m in paired if m["audio"]["dry_audio_sha256"] == activity]
    folders = [p for p, _ in paired]
    metadata = [m for _, m in paired]
    source = metadata[0]["source"]["instance_id"]
    if any(m["source"]["instance_id"] != source or m["audio"]["dry_audio_sha256"] != activity
           for m in metadata):
        raise ValueError("Recorded branches do not share source and activity")
    if len(folders) < 2:
        raise ValueError(f"No same-activity future observation is available: {sample}")
    return folders, metadata, folders.index(reference)


def make_menu(data_root, sample):
    folders, metadata, index = episode_records(data_root, sample)
    poses = [calibrated_pose(m) for m in metadata]

    def entry(j):
        return {"id": folders[j].name,
                "receiver_to_world": poses[j].tolist(),
                "lidar": str((folders[j] / "lidar.npz").resolve()),
                "audio": str((folders[j] / "audio.wav").resolve()),
                "source_id": str(metadata[j]["source"]["instance_id"]),
                "activity_id": metadata[j]["audio"]["dry_audio_sha256"],
                "valid": True,
                "movement_cost": float(np.linalg.norm(poses[j][:3, 3] - poses[index][:3, 3]))}

    basis = metadata[index].get("availability_basis")
    if not basis:
        basis = "recorded pose menu; source independence has not been certified"
    menu = {"reference": entry(index),
            "observations": [entry(j) for j in range(len(folders)) if j != index],
            "availability_basis": basis}
    return menu
