"""Prepare V1 training branches directly from raw recorded audio and LiDAR.

The future recordings and source labels are read only in this offline TRAIN/VAL
preparation. The deployment planner never uses this procedure.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "code"))
for key, subfolder in [("XDG_CACHE_HOME", "xdg"), ("TORCH_HOME", "torch"),
                       ("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba")]:
    os.environ.setdefault(key, str(HERE / "runtime" / subfolder))
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", USE_TF="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")

import numpy as np
import soundfile as sf
import torch

from acoustic_wm.action_protocol import actions_from_poses, valid_actions
from acoustic_wm.audio.predictive_features import covariance48_from_wave
from acoustic_wm.frontend import RawObservationEncoder
from acoustic_wm.model import ReleasedModel
from acoustic_wm.v1_dataset import episode_records, calibrated_pose


def save_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        np.savez_compressed(stream, **values)
        temporary = Path(stream.name)
    temporary.replace(path)


def prepare_one(sample, data_root, cache_root, encoder, model):
    reference_path = cache_root / "reference" / (sample + ".npz")
    controlled_path = cache_root / "controlled" / (sample + ".npz")
    skipped_path = cache_root / "skipped" / (sample + ".json")
    if reference_path.is_file() and controlled_path.is_file():
        return "existing"
    if skipped_path.is_file():
        return "skipped"
    folders, metadata, reference_index = episode_records(data_root, sample)
    reference = folders[reference_index]
    poses = [calibrated_pose(m) for m in metadata]
    other = [j for j in range(len(folders)) if j != reference_index]
    actions = actions_from_poses(poses[reference_index], [poses[j] for j in other])
    actions, valid, _, _ = valid_actions(actions, max_translation=3.0)
    if not valid.any():
        skipped_path.parent.mkdir(parents=True, exist_ok=True)
        skipped_path.write_text(json.dumps({"sample": sample, "reason": "no same-activity action within 3 m"}) + "\n")
        return "skipped"
    with np.load(reference / "lidar.npz", allow_pickle=False) as z:
        xyz = z["xyz"].astype(np.float32)
        truth = z["sound_mask"].astype(bool)
    wave, rate = sf.read(reference / "audio.wav", dtype="float32", always_2d=True)
    if rate != 44100 or wave.shape[1] != 4:
        raise ValueError("Expected four-channel 44.1 kHz reference audio")
    row = encoder.encode_arrays(xyz, wave, rate)
    observed, weight0 = covariance48_from_wave(wave)
    state = model.current(row, observed, weight0)
    future = []
    weights = []
    for j in other:
        # Offline branching target; these files are never passed to the planner.
        second, second_rate = sf.read(folders[j] / "audio.wav", dtype="float32", always_2d=True)
        if second_rate != 44100 or second.shape[1] != 4:
            raise ValueError("Invalid recorded branch audio")
        covariance, reliability = covariance48_from_wave(second)
        future.append(covariance)
        weights.append(reliability)
    masks = np.asarray(row["masks"], bool)
    inverse = np.asarray(row["inverse"], np.int64)
    positive = np.bincount(inverse[truth], minlength=masks.shape[1])
    negative = np.bincount(inverse[~truth], minlength=masks.shape[1])
    intersection = masks @ positive
    union = truth.sum() + masks @ negative
    iou = np.divide(intersection, union, out=np.ones(len(masks)), where=union > 0).astype(np.float32)
    reference_values = {"masks": masks, "inverse": inverse,
                        "hypotheses": np.asarray(row["hypotheses"], np.float32)}
    controlled_values = {"base": state["base"].cpu().numpy(),
                         "sites": state["log_site"].cpu().numpy(),
                         "iou": iou, "actions": actions, "valid": valid,
                         "observed": observed, "future_observed": np.stack(future),
                         "weight": np.stack(weights),
                         "reference_world_points": state["geometry"][0].cpu().numpy(),
                         "reference_world_features": state["geometry"][1].cpu().numpy(),
                         "reference_world_valid": state["geometry"][2].cpu().numpy()}
    save_npz(reference_path, reference_values)
    save_npz(controlled_path, controlled_values)
    return "prepared"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=PROJECT / ".importants/Experiments_20261024/generated/common")
    parser.add_argument("--output-root", type=Path, default=HERE / "prepared/common")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Development limit; zero prepares the full split")
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error("Invalid shard index or count")
    samples = sorted(str(p.parent.relative_to(args.data_root))
                     for p in (args.data_root / args.split).rglob("audio.wav"))
    if not samples:
        raise FileNotFoundError("No raw recorded observations found")
    manifest = args.output_root / "manifests" / f"{args.split}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        manifest.write_text(json.dumps(samples, indent=2) + "\n")
    elif json.loads(manifest.read_text()) != samples:
        raise ValueError("Existing manifest differs from the dataset inventory")
    work = samples[args.shard_index::args.shards]
    if args.limit:
        work = work[:args.limit]
    config = json.loads((HERE / "config.json").read_text())
    config["paths"]["dataset"] = str(HERE / "assets/sensor")
    torch.set_num_threads(1)
    encoder = RawObservationEncoder(config, args.device)
    model = ReleasedModel(HERE, args.device)
    count = {"prepared": 0, "existing": 0, "skipped": 0}
    for index, sample in enumerate(work, 1):
        status = prepare_one(sample, args.data_root, args.output_root / "cache", encoder, model)
        count[status] += 1
        if index % 25 == 0:
            print(json.dumps({"progress": index, "assigned": len(work), **count}), flush=True)
    print(json.dumps({"split": args.split, "shard": args.shard_index,
                      "assigned": len(work), **count}), flush=True)


if __name__ == "__main__":
    main()
