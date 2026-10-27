"""Full-split raw-input evaluation; labels are opened only after inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))

import numpy as np
import torch

from acoustic_wm.frontend import RawObservationEncoder
from acoustic_wm.v1_dataset import make_menu
from acoustic_wm.v1_pipeline import ActiveViewV1
from infer import run as infer_one
from audit_menus import audit as audit_shared_menus
from freeze import verify_source_manifest


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_interval(rows, key, seed=20261027, repeats=2000):
    scenes = sorted({row["scene"] for row in rows})
    totals = np.asarray([sum(float(row[key]) for row in rows if row["scene"] == scene)
                         for scene in scenes])
    counts = np.asarray([sum(row["scene"] == scene for row in rows) for scene in scenes])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(scenes), (repeats, len(scenes)))
    means = totals[draw].sum(1) / counts[draw].sum(1)
    return {"mean": float(totals.sum() / counts.sum()),
            "scene_macro_mean": float(np.mean(totals / counts)),
            "ci95": np.quantile(means, [.025, .975]).tolist(),
            "episodes": len(rows), "scenes": len(scenes),
            "resampling_unit": "scene", "seed": seed, "draws": repeats}


def score_prediction(sample, data_root, prediction_path):
    with np.load(prediction_path, allow_pickle=False) as result:
        mask = result["sound_mask"].astype(bool)
        probability = result["point_probability"].astype(np.float32)
        actions = result["chosen_actions"].astype(int).tolist()
        budget = int(result["budget_used"])
        distance = float(result["total_distance_m"])
        selected = int(result["selected_object"])
        group = int(result["selected_group"])
        checkpoint_identity = str(result["checkpoint_identity"])
        checkpoint_sha256 = str(result["checkpoint_sha256"])
    # Ground truth enters only after the prediction file has been written.
    with np.load(data_root / sample / "lidar.npz", allow_pickle=False) as z:
        truth = z["sound_mask"].astype(bool)
    if mask.shape != truth.shape or probability.shape != truth.shape:
        raise ValueError(f"Reference output and label lengths differ for {sample}")
    union = (mask | truth).sum()
    iou = float((mask & truth).sum() / union) if union else 1.0
    metadata = json.loads((data_root / sample / "metadata.json").read_text())
    return {"sample": sample, "scene": sample.split("/")[1],
            "source_group": "/".join(sample.split("/")[:3]),
            "iou": iou, "point_brier": float(np.mean((probability - truth) ** 2)),
            "reference_source_points": int(truth.sum()),
            "selected_candidate": selected, "selected_geometric_group": group,
            "chosen_actions": actions, "budget_used": budget, "total_distance_m": distance,
            "menu_sha256": metadata.get("menu_sha256"),
            "availability_basis": metadata.get("availability_basis", "unspecified"),
            "original_recording_id": metadata["audio"].get("dry_audio_id"),
            "original_recording_sha256": metadata["audio"].get("dry_audio_sha256"),
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_sha256": checkpoint_sha256,
            "prediction_sha256": sha(prediction_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=PROJECT / ".importants/Experiments_20261024/generated/common")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-queries", type=int, default=1)
    parser.add_argument("--require-shared-menu", action="store_true",
                        help="Fail unless receiver poses and local validity agree across scene sources")
    parser.add_argument("--limit", type=int, default=0,
                        help="Development limit; zero evaluates every observation in the split")
    parser.add_argument("--allow-development-checkpoint", action="store_true",
                        help="Permit a limited TRAIN/VAL checkpoint for a full-split diagnostic")
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if (HERE / "SOURCE_MANIFEST.json").exists():
        verify_source_manifest()
    checkpoint_hash = sha(args.checkpoint)
    saved_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training_scope = saved_checkpoint.get("training_scope", {"development_limited": True,
                                                               "reason": "Checkpoint has no full-split training record"})
    if not args.limit and training_scope["development_limited"] and not args.allow_development_checkpoint:
        raise ValueError("Full-split evaluation requires a full TRAIN/VAL V1 checkpoint")
    samples = sorted(str(p.parent.relative_to(args.data_root))
                     for p in (args.data_root / args.split).rglob("audio.wav"))
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        raise FileNotFoundError("No recorded observations found")
    menu_audit = audit_shared_menus(args.data_root, args.split) if args.require_shared_menu else None
    if menu_audit is not None and not menu_audit["source_independent"]:
        raise ValueError("The selected split does not have a source-independent shared receiver menu")
    config = json.loads((HERE / "config.json").read_text())
    config["paths"]["dataset"] = str(HERE / "assets/sensor")
    torch.set_num_threads(1)
    encoder = RawObservationEncoder(config, args.device)
    model = ActiveViewV1(HERE, args.device, checkpoint=args.checkpoint)
    rows = []
    for number, sample in enumerate(samples, 1):
        menu_path = args.output_dir / "menus" / (sample + ".json")
        prediction_path = args.output_dir / "predictions" / (sample + ".npz")
        row_path = args.output_dir / "rows" / (sample + ".json")
        expected_menu = make_menu(args.data_root, sample)
        if not menu_path.exists():
            menu_path.parent.mkdir(parents=True, exist_ok=True)
            menu_path.write_text(json.dumps(expected_menu, indent=2) + "\n")
        elif json.loads(menu_path.read_text()) != expected_menu:
            raise ValueError(f"Saved action menu differs from the selected dataset: {sample}")
        if not prediction_path.exists():
            infer_one(menu_path, prediction_path, checkpoint=args.checkpoint,
                      device=args.device, max_queries=args.max_queries,
                      encoder=encoder, model=model)
        if not row_path.exists():
            row = score_prediction(sample, args.data_root, prediction_path)
            row["menu_file_sha256"] = sha(menu_path)
            row_path.parent.mkdir(parents=True, exist_ok=True)
            row_path.write_text(json.dumps(row, indent=2) + "\n")
        else:
            row = json.loads(row_path.read_text())
        if (row["checkpoint_identity"] != str(args.checkpoint) or
                row.get("checkpoint_sha256") != checkpoint_hash or
                row["prediction_sha256"] != sha(prediction_path)):
            raise ValueError(f"Saved prediction does not match this checkpoint/run: {sample}")
        if row.get("menu_file_sha256") != sha(menu_path):
            raise ValueError(f"Saved result does not match its action menu: {sample}")
        rows.append(row)
        if number % 25 == 0:
            print(json.dumps({"completed": number, "total": len(samples),
                              "mean_iou_so_far": float(np.mean([r["iou"] for r in rows]))}), flush=True)
    report = {"schema": "activeview-v1-evaluation-v1", "split": args.split,
              "data_root": str(args.data_root), "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": checkpoint_hash, "max_queries": args.max_queries,
              "v1_source_manifest_sha256": (sha(HERE / "SOURCE_MANIFEST.json")
                                             if (HERE / "SOURCE_MANIFEST.json").exists() else None),
              "shared_menu_audited": bool(args.require_shared_menu),
              "training_scope": training_scope,
              "limited_development_run": bool(args.limit),
              "iou": scene_interval(rows, "iou", args.seed),
              "point_brier": scene_interval(rows, "point_brier", args.seed),
              "source_visible_episodes": sum(r["reference_source_points"] > 0 for r in rows),
              "total_distance_m": scene_interval(rows, "total_distance_m", args.seed),
              "samples": rows}
    output = args.output_dir / f"{args.split}_summary.json"
    if sha(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": str(output), "episodes": len(rows),
                      "mean_iou": report["iou"]["mean"]}))


if __name__ == "__main__":
    main()
