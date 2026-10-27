#!/usr/bin/env python3
"""Run a trained adapted model on complete point clouds, then score saved masks."""
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
import torch

from adapters.audiolidar import AudioLiDAR, DEFAULT_ROOT
from evaluation.io import write_unchanged_or_new
from evaluation.scoring import aggregate, foreground_iou
from models.passive import PASSIVE, make_passive


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def predict(model, observed: dict, sample: str, device: torch.device,
            context_points: int, chunk_points: int, threshold: float):
    xyz = observed["xyz"]
    seed = int.from_bytes(sha256(sample.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    context_indices = rng.choice(len(xyz), min(context_points, len(xyz)), replace=False)
    context = torch.from_numpy(xyz[context_indices].copy())[None].to(device)
    audio = torch.from_numpy(observed["audio"].copy())[None].to(device)
    state = model.encode_observation(context, audio)
    values = np.empty(len(xyz), dtype=np.float32)
    for start in range(0, len(xyz), chunk_points):
        chunk = torch.from_numpy(xyz[start:start + chunk_points].copy())[None].to(device)
        values[start:start + len(chunk[0])] = model.score_queries(state, chunk)[0].sigmoid().cpu().numpy()
    return values >= threshold, values


def run(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != args.method:
        raise ValueError("Checkpoint method does not match requested method")
    if args.split == "test" and checkpoint.get("development_limited") and not args.allow_development_checkpoint:
        raise ValueError("Refusing TEST evaluation of a development-limited checkpoint")
    data = AudioLiDAR(args.data_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU inference requires a Slurm allocation")
    model = make_passive(args.method, data.calibration["positions_lidar_m"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(ROOT):
        raise ValueError("Evaluation outputs must stay inside Comparing_Methods")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace an existing evaluation: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = data.samples(args.split)[:args.limit or None]
    provenance = {"method": args.method, "variant": checkpoint["variant"],
                  "checkpoint_sha256": file_sha256(args.checkpoint), "trained_on": "train",
                  "input_modalities": ["LiDAR XYZ", "tetrahedral four-channel audio", "fixed calibration"],
                  "inference_command": " ".join(sys.argv), "split": args.split,
                  "evaluated_examples": len(samples), "full_split": len(samples) == len(data.samples(args.split)),
                  "development_limited_checkpoint": bool(checkpoint["development_limited"]),
                  "threshold": args.threshold}
    write_unchanged_or_new(output_dir / "provenance.json", json.dumps(provenance, indent=2) + "\n")
    rows = []
    for i, sample in enumerate(samples):
        start = time.monotonic()
        observed = data.observation(sample)
        mask, probability = predict(model, observed, sample, device,
                                    args.context_points, args.chunk_points, args.threshold)
        target_path = output_dir / (sample + ".npz")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target_path, sound_mask=mask, point_probability=probability)
        # Evaluation-only target is opened after the prediction has been saved.
        target = data.target_for_evaluation(sample)["sound_mask"]
        row = {"sample": sample, "scene": sample.split("/")[1],
               "foreground_iou": foreground_iou(mask, target),
               "inference_seconds": time.monotonic() - start}
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(json.dumps({"completed": i + 1, "total": len(samples)}), flush=True)
    report = {"provenance": provenance, "metrics": aggregate(rows) if rows else None}
    write_unchanged_or_new(output_dir / "aggregate.json", json.dumps(report, indent=2) + "\n")
    with (output_dir / "per_example.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"method": args.method, "split": args.split,
                      "samples": len(rows), "mean_iou": report["metrics"]["mean_iou"],
                      "full_split": provenance["full_split"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=PASSIVE, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-points", type=int, default=2048)
    parser.add_argument("--chunk-points", type=int, default=8192)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--limit", type=int, default=0, help="Development limit; zero means complete split")
    parser.add_argument("--allow-development-checkpoint", action="store_true")
    run(parser.parse_args())
