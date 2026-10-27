#!/usr/bin/env python3
"""Evaluate the source-conditioned finite-menu policy with one new recording.

This is a diagnostic of an adapted MAGNet policy, not a matched active-method
comparison. The frozen passive backend is shared by the initial/final passes.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("TMPDIR", str(ROOT / "cache/tmp"))
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache/torch"))

import numpy as np
import soundfile as sf
import torch

from adapters.active_menu import pose_only_menu
from adapters.audiolidar import AudioLiDAR, DEFAULT_ROOT
from evaluation.scoring import aggregate, foreground_iou
from models.active import MAGNetFiniteMenu
from models.passive import PASSIVE, make_passive
from scripts.evaluate_passive import file_sha256, predict


def selected_action(policy, observed, menu, sample, device, context_points):
    """Choose using only initial audio/XYZ and known receiver poses."""
    xyz = observed["xyz"]
    seed = int.from_bytes(sha256(sample.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    index = rng.choice(len(xyz), min(context_points, len(xyz)), replace=False)
    poses = np.zeros((1, 11, 13), dtype=np.float32)
    valid = np.zeros((1, 11), dtype=bool)
    for j, option in enumerate(menu):
        poses[0, j] = option["features"]
        valid[0, j] = True
    with torch.inference_mode():
        logits = policy(torch.from_numpy(xyz[index].copy())[None].to(device),
                        torch.from_numpy(observed["audio"].copy())[None].to(device),
                        torch.from_numpy(poses).to(device),
                        torch.from_numpy(valid).to(device))
    return int(logits.argmax(1).item())


def run(args):
    if args.backend_method not in PASSIVE:
        raise ValueError("Unknown passive backend")
    if args.context_points <= 0 or args.chunk_points <= 0:
        raise ValueError("Point counts must be positive")
    data = AudioLiDAR(args.data_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU inference requires a Slurm allocation")
    policy_ckpt = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=False)
    backend_ckpt = torch.load(args.backend_checkpoint, map_location="cpu", weights_only=False)
    if policy_ckpt.get("variant") != "magnet-finite-menu-parallax-imitation":
        raise ValueError("Not an adapted finite-menu policy checkpoint")
    if backend_ckpt.get("method") != args.backend_method:
        raise ValueError("Passive backend checkpoint/method mismatch")
    if args.split == "test" and (policy_ckpt["development_limited"] or
                                 backend_ckpt["development_limited"]):
        raise ValueError("Refusing TEST evaluation with a development-limited checkpoint")
    policy = MAGNetFiniteMenu().to(device).eval()
    policy.load_state_dict(policy_ckpt["model"], strict=True)
    backend = make_passive(args.backend_method, data.calibration["positions_lidar_m"])
    backend.load_state_dict(backend_ckpt["model"], strict=True)
    backend.to(device).eval()
    all_samples = data.samples(args.split)
    eligible = [(sample, menu) for sample in all_samples
                if (menu := pose_only_menu(data, sample))]
    selected = eligible[:args.limit or None]
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(ROOT):
        raise ValueError("Output must stay inside Comparing_Methods")
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance = {
        "method": "magnet", "variant": policy_ckpt["variant"],
        "policy_sha256": file_sha256(args.policy_checkpoint),
        "backend_method": args.backend_method,
        "backend_sha256": file_sha256(args.backend_checkpoint),
        "split": args.split, "total_split_examples": len(all_samples),
        "eligible_examples": len(eligible), "evaluated_examples": len(selected),
        "full_eligible_split": len(selected) == len(eligible),
        "source_conditioned_menu": True,
        "action_protocol": "pose-only decision followed by exactly one recorded future waveform",
        "fusion": "mean initial/future probabilities using same frozen passive backend",
        "input_modalities": ["initial LiDAR XYZ", "initial four-channel audio", "candidate receiver poses"],
        "inference_command": " ".join(sys.argv),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    rows = []
    for i, (sample, menu) in enumerate(selected):
        started = time.monotonic()
        initial = data.observation(sample)
        action = selected_action(policy, initial, menu, sample, device, args.context_points)
        chosen = menu[action]
        # The chosen waveform is first opened here, after the action is fixed.
        future_audio, rate = sf.read(data.folder(chosen["sample"]) / "audio.wav",
                                     dtype="float32", always_2d=True)
        if future_audio.shape != (88200, 4) or rate != 44100:
            raise ValueError(f"Invalid acquired waveform: {chosen['sample']}")
        initial_pose = initial["receiver_to_world"]
        future_pose = chosen["pose"]
        xyz_world = initial["xyz"].astype(np.float64) @ initial_pose[:3, :3].T + initial_pose[:3, 3]
        xyz_future = (xyz_world - future_pose[:3, 3]) @ future_pose[:3, :3]
        _, initial_p = predict(backend, initial, sample, device, args.context_points,
                               args.chunk_points, args.threshold)
        future_observation = {"xyz": xyz_future.astype(np.float32), "audio": future_audio}
        _, future_p = predict(backend, future_observation, chosen["sample"], device,
                              args.context_points, args.chunk_points, args.threshold)
        probability = (initial_p + future_p) / 2
        mask = probability >= args.threshold
        output = output_dir / (sample + ".npz")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, sound_mask=mask, point_probability=probability,
                            selected_future_sample=chosen["sample"],
                            selected_action=np.int32(action))
        # The target is used only after the complete action and mask are saved.
        target = data.target_for_evaluation(sample)["sound_mask"]
        rows.append({"sample": sample, "scene": sample.split("/")[1],
                     "selected_future": chosen["sample"],
                     "foreground_iou": foreground_iou(mask, target),
                     "inference_seconds": time.monotonic() - started})
        if (i + 1) % 25 == 0:
            print(json.dumps({"completed": i + 1, "total": len(selected)}), flush=True)
    with (output_dir / "per_example.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else
                                ["sample", "scene", "selected_future", "foreground_iou", "inference_seconds"])
        writer.writeheader()
        writer.writerows(rows)
    report = {"provenance": provenance, "metrics": aggregate(rows) if rows else None}
    (output_dir / "aggregate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"evaluated": len(rows), "eligible": len(eligible),
                      "mean_iou": report["metrics"]["mean_iou"] if rows else None}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--backend-checkpoint", type=Path, required=True)
    parser.add_argument("--backend-method", choices=PASSIVE, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-points", type=int, default=2048)
    parser.add_argument("--chunk-points", type=int, default=8192)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--limit", type=int, default=0, help="Development limit")
    run(parser.parse_args())
