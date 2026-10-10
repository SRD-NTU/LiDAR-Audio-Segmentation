#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lidarwm.config import load_config
from lidarwm.data import AudioLidarDataset, collate_samples
from lidarwm.engine import build_model, load_checkpoint, to_device, forward_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a sounding-object mask for one dataset sample")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sample", required=True, help="Manifest-relative path or integer split index")
    parser.add_argument("--output", required=True, help="Output .npz file")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = AudioLidarDataset(cfg["data"]["root"], args.split, cfg["data"].get("audio_stride", 3),
      feature_root=cfg['data'].get('feature_root'),antialias=cfg['data'].get('antialias',False))
    if args.sample.isdigit():
        index = int(args.sample)
    else:
        matches = [i for i, record in enumerate(dataset.records) if record.relative_path == args.sample]
        if not matches:
            raise ValueError(f"Sample not found in {args.split}: {args.sample}")
        index = matches[0]
    raw = dataset[index]
    batch = collate_samples([raw])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model)
    model.eval()
    with torch.inference_mode():
        moved = to_device(batch, device)
        probability_grid = forward_batch(model,moved)["mask_logits"].sigmoid()[0, 0]
    ray_id = raw["ray_id"].numpy()
    azimuth_index, beam_index = np.divmod(ray_id, 64)
    probability = probability_grid.cpu().numpy()[beam_index, azimuth_index]
    threshold = args.threshold
    if threshold is None:
        threshold = float(checkpoint.get("validation", {}).get("threshold", 0.5))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, probability=probability.astype(np.float32), mask=(probability >= threshold).astype(np.uint8),
        ray_id=ray_id, threshold=np.float32(threshold), sample=np.asarray(raw["path"]),
    )
    print(f"Wrote {len(probability)} point predictions to {output} (threshold={threshold:.2f})")


if __name__ == "__main__":
    main()
