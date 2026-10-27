#!/usr/bin/env python3
"""Train an explicitly adapted passive method on AudioLiDAR TRAIN, select on VAL."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
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
from torch import nn
from torch.utils.data import Dataset, DataLoader

from adapters.audiolidar import AudioLiDAR, DEFAULT_ROOT
from models.passive import PASSIVE, make_passive


class SampledPointDataset(Dataset):
    def __init__(self, data: AudioLiDAR, split: str, context_points: int,
                 query_points: int, seed: int, limit: int = 0):
        self.data, self.split = data, split
        self.samples = data.samples(split)[:limit or None]
        self.context_points = context_points
        self.query_points = query_points
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        observed = self.data.observation(sample)
        target = self.data.target_for_evaluation(sample)["sound_mask"]
        xyz = observed["xyz"]
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + index)
        context = rng.choice(len(xyz), size=self.context_points, replace=len(xyz) < self.context_points)
        positive = np.flatnonzero(target)
        negative = np.flatnonzero(~target)
        if not len(positive) or not len(negative):
            raise ValueError(f"Source/background missing: {sample}")
        n_positive = self.query_points // 4
        query = np.concatenate([
            rng.choice(positive, n_positive, replace=len(positive) < n_positive),
            rng.choice(negative, self.query_points - n_positive,
                       replace=len(negative) < self.query_points - n_positive)])
        rng.shuffle(query)
        return {"context": torch.from_numpy(xyz[context].copy()),
                "query": torch.from_numpy(xyz[query].copy()),
                "audio": torch.from_numpy(observed["audio"].copy()),
                "target": torch.from_numpy(target[query].astype(np.float32)),
                "sample": sample}


def angular_target(query: torch.Tensor, labels: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Training-only object angular extent for the SAID panoramic head."""
    az = (torch.arange(width, device=query.device, dtype=query.dtype) + .5) / width * 2 * torch.pi - torch.pi
    el = torch.pi / 2 - (torch.arange(height, device=query.device, dtype=query.dtype) + .5) / height * torch.pi
    elev, azim = torch.meshgrid(el, az, indexing="ij")
    sphere = torch.stack([torch.cos(elev) * torch.cos(azim),
                          torch.cos(elev) * torch.sin(azim), torch.sin(elev)], dim=-1).reshape(-1, 3)
    directions = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cosine = sphere[None] @ directions.transpose(1, 2)
    angle = torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6))
    likelihood = torch.exp(-.5 * (angle / (15 * torch.pi / 180)) ** 2)
    likelihood = likelihood * labels[:, None]
    return likelihood.amax(dim=-1).reshape(-1, 1, height, width)


def loss_for(output: dict, batch: dict) -> torch.Tensor:
    logits, target = output["point_logits"], batch["target"]
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probabilities = logits.sigmoid()
    dice = 1 - (2 * (probabilities * target).sum(1) + 1) / (
        probabilities.sum(1) + target.sum(1) + 1)
    loss = loss + dice.mean()
    if output["map_logits"] is not None:
        shape = output["map_logits"].shape
        heat = angular_target(batch["query"], target, shape[-2], shape[-1])
        loss = loss + .5 * nn.functional.binary_cross_entropy_with_logits(
            output["map_logits"], heat, pos_weight=torch.tensor(4.0, device=logits.device))
    return loss


def run(args):
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    data = AudioLiDAR(args.data_root)
    model = make_passive(args.method, data.calibration["positions_lidar_m"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU training requires a Slurm CUDA allocation")
    model.to(device)
    train = SampledPointDataset(data, "train", args.context_points, args.query_points,
                                args.seed, args.max_train)
    val = SampledPointDataset(data, "val", args.context_points, args.query_points,
                              args.seed + 9000000, args.max_val)
    workers = args.workers
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=workers,
                              pin_memory=device.type == "cuda", persistent_workers=False)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda", persistent_workers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.smoke:
        batch = next(iter(train_loader))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = model(batch["context"], batch["query"], batch["audio"])
        loss = loss_for(output, batch)
        loss.backward()
        print(json.dumps({"method": args.method, "sample": batch["sample"][0],
                          "loss": float(loss.detach()), "finite_gradients": all(
                              torch.isfinite(p.grad).all().item() for p in model.parameters()
                              if p.grad is not None)}))
        return
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT):
        raise ValueError("--run-dir must stay inside Comparing_Methods")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "data_root": str(args.data_root), "run_dir": str(run_dir),
              "variant": f"{args.method}-tetra-lidar-train-adaptation",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "train_examples": len(train), "val_examples": len(val)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    best = float("inf")
    for epoch in range(args.epochs):
        started = time.monotonic()
        train.epoch = epoch
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["context"], batch["query"], batch["audio"])
            loss = loss_for(output, batch)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}, step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch["target"])
        model.eval()
        validation = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output = model(batch["context"], batch["query"], batch["audio"])
                validation += float(loss_for(output, batch)) * len(batch["target"])
        validation /= len(val)
        row = {"epoch": epoch + 1, "train_loss": total / len(train), "val_loss": validation,
               "seconds": time.monotonic() - started}
        with (run_dir / "history.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if validation < best:
            best = validation
            checkpoint = {"model": model.state_dict(), "method": args.method,
                          "variant": config["variant"], "epoch": epoch + 1,
                          "val_loss": validation, "train_examples": len(train),
                          "val_examples": len(val), "seed": args.seed,
                          "development_limited": bool(args.max_train or args.max_val),
                          "calibration": data.calibration}
            temp = run_dir / "best.pt.tmp"
            torch.save(checkpoint, temp)
            temp.replace(run_dir / "best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=PASSIVE, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/adapted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-points", type=int, default=2048)
    parser.add_argument("--query-points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-train", type=int, default=0,
                        help="Development-only TRAIN limit; zero means full split")
    parser.add_argument("--max-val", type=int, default=0,
                        help="Development-only VAL limit; zero means full split")
    parser.add_argument("--smoke", action="store_true", help="One real TRAIN backward pass, no checkpoint")
    run(parser.parse_args())
