#!/usr/bin/env python3
"""Train MAGNet finite-menu imitation on TRAIN, select checkpoint on VAL."""
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
from torch.utils.data import DataLoader, Dataset

from adapters.audiolidar import AudioLiDAR, DEFAULT_ROOT
from adapters.active_menu import pose_only_menu, train_parallax_teacher
from models.active import MAGNetFiniteMenu


class MenuDataset(Dataset):
    def __init__(self, data: AudioLiDAR, split: str, context_points: int, seed: int, limit: int = 0):
        self.data, self.split = data, split
        self.context_points, self.seed = context_points, seed
        self.episodes = []
        for sample in data.samples(split):
            menu = pose_only_menu(data, sample)
            if not menu:
                continue
            teacher = train_parallax_teacher(data, sample, menu)
            self.episodes.append((sample, menu, teacher))
            if limit and len(self.episodes) >= limit:
                break
        if not self.episodes:
            raise ValueError(f"No valid recorded actions in {split}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        sample, menu, teacher = self.episodes[index]
        observed = self.data.observation(sample)
        rng = np.random.default_rng(self.seed + index)
        xyz = observed["xyz"]
        context_index = rng.choice(len(xyz), size=self.context_points, replace=len(xyz) < self.context_points)
        poses = np.zeros((11, 13), dtype=np.float32)
        valid = np.zeros(11, dtype=bool)
        for j, item in enumerate(menu):
            poses[j] = item["features"]
            valid[j] = True
        return {"context": torch.from_numpy(xyz[context_index].copy()),
                "audio": torch.from_numpy(observed["audio"].copy()),
                "poses": torch.from_numpy(poses), "valid": torch.from_numpy(valid),
                "teacher": torch.tensor(teacher, dtype=torch.long), "sample": sample}


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = AudioLiDAR(args.data_root)
    train = MenuDataset(data, "train", args.context_points, args.seed, args.max_train)
    val = MenuDataset(data, "val", args.context_points, args.seed + 9000000, args.max_val)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU training needs a Slurm allocation")
    model = MAGNetFiniteMenu().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    def move(batch):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    if args.smoke:
        batch = move(next(iter(train_loader)))
        logits = model(batch["context"], batch["audio"], batch["poses"], batch["valid"])
        loss = nn.functional.cross_entropy(logits, batch["teacher"])
        loss.backward()
        print(json.dumps({"variant": "magnet-finite-menu-parallax-imitation",
                          "sample": batch["sample"][0], "loss": float(loss.detach()),
                          "finite_gradients": all(torch.isfinite(p.grad).all().item()
                                                  for p in model.parameters() if p.grad is not None)}))
        return
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT):
        raise ValueError("Output must remain inside Comparing_Methods")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "data_root": str(args.data_root), "run_dir": str(run_dir),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "variant": "magnet-finite-menu-parallax-imitation",
              "train_examples": len(train), "val_examples": len(val),
              "menu_protocol": "original-v2-source-conditioned-same-recording-3m"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    best = float("inf")
    for epoch in range(args.epochs):
        start = time.monotonic()
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = move(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["context"], batch["audio"], batch["poses"], batch["valid"])
            loss = nn.functional.cross_entropy(logits, batch["teacher"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite policy loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(batch["teacher"])
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = move(batch)
                logits = model(batch["context"], batch["audio"], batch["poses"], batch["valid"])
                val_loss += float(nn.functional.cross_entropy(logits, batch["teacher"])) * len(batch["teacher"])
                correct += int((logits.argmax(1) == batch["teacher"]).sum())
        val_loss /= len(val)
        row = {"epoch": epoch + 1, "train_loss": train_loss / len(train),
               "val_loss": val_loss, "val_teacher_accuracy": correct / len(val),
               "seconds": time.monotonic() - start}
        with (run_dir / "history.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if val_loss < best:
            best = val_loss
            checkpoint = {"model": model.state_dict(), "variant": config["variant"],
                          "epoch": epoch + 1, "train_examples": len(train),
                          "val_examples": len(val), "val_loss": val_loss,
                          "development_limited": bool(args.max_train or args.max_val),
                          "menu_protocol": config["menu_protocol"], "seed": args.seed}
            temp = run_dir / "best.pt.tmp"
            torch.save(checkpoint, temp)
            temp.replace(run_dir / "best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/magnet_finite_menu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-points", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-train", type=int, default=0, help="Development limit")
    parser.add_argument("--max-val", type=int, default=0, help="Development limit")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
