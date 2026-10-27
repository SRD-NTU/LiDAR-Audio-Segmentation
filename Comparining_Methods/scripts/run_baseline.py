#!/usr/bin/env python3
"""Honest baseline gate, data smoke test, and complete prediction scoring.

No published method currently has a validated inference adapter. The default
command exits with a specific blocker instead of producing untrained scores.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.audiolidar import AudioLiDAR, DEFAULT_ROOT
from evaluation.scoring import aggregate, foreground_iou, selected_candidate_accuracy
from evaluation.io import write_unchanged_or_new

STATUS = json.loads((ROOT / "configs/method_status.json").read_text())


def score_predictions(data: AudioLiDAR, method: str, split: str, predictions: Path, output: Path) -> None:
    """Score externally produced masks. This does not certify model fidelity."""
    provenance_file = predictions / "provenance.json"
    if not provenance_file.is_file():
        raise ValueError("Missing provenance.json: variant, checkpoint hash and inference protocol are required")
    provenance = json.loads(provenance_file.read_text())
    required = ("method", "variant", "checkpoint_sha256", "trained_on", "input_modalities", "inference_command")
    if any(not provenance.get(key) for key in required) or provenance["method"] != method:
        raise ValueError(f"Incomplete or mismatched provenance; required: {', '.join(required)}")
    if provenance["trained_on"] == "test":
        raise ValueError("Test-trained predictions are invalid")
    rows = []
    for sample in data.samples(split):
        prediction_path = predictions / (sample + ".npz")
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Incomplete split: {prediction_path}")
        # Prediction is loaded before the target to keep inference and scoring distinct.
        with np.load(prediction_path, allow_pickle=False) as record:
            predicted_mask = record["sound_mask"].astype(bool)
            candidate_masks = record["candidate_masks"].astype(bool) if "candidate_masks" in record else None
            selected = int(record["selected_candidate"]) if "selected_candidate" in record else None
        target = data.target_for_evaluation(sample)["sound_mask"]
        iou = foreground_iou(predicted_mask, target)
        accuracy = None
        if candidate_masks is not None and selected is not None:
            if not np.array_equal(predicted_mask, candidate_masks[selected]):
                raise ValueError(f"Selected candidate differs from exported mask: {sample}")
            accuracy = selected_candidate_accuracy(candidate_masks, selected, target)
        rows.append({"sample": sample, "scene": sample.split("/")[1],
                     "foreground_iou": iou, "candidate_selection_accuracy": accuracy})
    summary = aggregate(rows)
    table = StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    aggregate_text = json.dumps({"provenance": provenance, "metrics": summary}, indent=2) + "\n"
    for path, content in ((output / "per_example.csv", table.getvalue()),
                          (output / "aggregate.json", aggregate_text)):
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise FileExistsError(f"Refusing to overwrite changed evidence: {path}")
    write_unchanged_or_new(output / "per_example.csv", table.getvalue())
    write_unchanged_or_new(output / "aggregate.json", aggregate_text)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=STATUS)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--mode", choices=("passive", "active"), default="passive")
    parser.add_argument("--action", choices=("run", "check", "smoke-data", "evaluate-predictions"), default="run")
    parser.add_argument("--variant", choices=("published", "adapted"), default="published")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Adapted passive checkpoint or MAGNet policy checkpoint")
    parser.add_argument("--backend-checkpoint", type=Path, help="Frozen passive checkpoint for MAGNet")
    parser.add_argument("--backend-method", choices=("echosegnet", "jaeger", "said", "sound3dvdet"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Development-only evaluation limit")
    args = parser.parse_args()
    status = STATUS[args.method]
    if args.action == "check":
        if args.variant == "published":
            print(json.dumps(status, indent=2))
        else:
            modules = {module: importlib.util.find_spec(module) is not None
                       for module in ("torch", "torchaudio", "numpy", "soundfile")}
            files = {"checkpoint": args.checkpoint.is_file() if args.checkpoint else False}
            if args.method == "magnet":
                files["backend_checkpoint"] = (args.backend_checkpoint.is_file()
                                               if args.backend_checkpoint else False)
            print(json.dumps({"method": args.method, "variant": "adapted",
                              "train_entry": "scripts/train_magnet.py" if args.method == "magnet"
                              else "scripts/train_passive.py",
                              "dependencies": modules, "ready_to_train": all(modules.values()),
                              "checkpoints": files,
                              "ready_to_evaluate": all(modules.values()) and all(files.values()),
                              "published_model_equivalent": False}, indent=2))
        return
    if args.mode != status["protocol"]:
        parser.error(f"{status['published_name']} belongs to the {status['protocol']} protocol")
    data = AudioLiDAR(args.data_root)
    if args.action == "smoke-data":
        sample = data.samples(args.split)[0]
        observation = data.observation(sample)
        print(json.dumps({"sample": sample, "xyz_shape": list(observation["xyz"].shape),
                          "audio_shape": list(observation["audio"].shape),
                          "sample_rate": observation["sample_rate"],
                          "receiver_to_world": observation["receiver_to_world"].tolist()}, indent=2))
        return
    if args.action == "evaluate-predictions":
        if args.mode == "active":
            parser.error("Active scoring needs verified source-independent menus, pre-action choice traces, exactly one acquired observation, and a shared frozen backend; this evaluator accepts passive saved masks only")
        if args.predictions_dir is None:
            parser.error("--predictions-dir is required")
        output = (args.output_dir or ROOT / "results" / args.method / args.split).resolve()
        if not output.is_relative_to(ROOT):
            parser.error("--output-dir must be inside Comparing_Methods")
        score_predictions(data, args.method, args.split, args.predictions_dir.resolve(), output)
        return
    if args.variant == "adapted":
        if args.checkpoint is None or args.output_dir is None:
            parser.error("Adapted evaluation needs --checkpoint and --output-dir")
        if args.method == "magnet":
            if args.backend_checkpoint is None or args.backend_method is None:
                parser.error("MAGNet also needs --backend-checkpoint and --backend-method")
            command = [sys.executable, str(ROOT / "scripts/evaluate_magnet.py"),
                       "--policy-checkpoint", str(args.checkpoint),
                       "--backend-checkpoint", str(args.backend_checkpoint),
                       "--backend-method", args.backend_method]
        else:
            command = [sys.executable, str(ROOT / "scripts/evaluate_passive.py"),
                       "--method", args.method, "--checkpoint", str(args.checkpoint)]
        command += ["--data-root", str(args.data_root), "--split", args.split,
                    "--output-dir", str(args.output_dir), "--device", args.device,
                    "--limit", str(args.limit)]
        subprocess.run(command, check=True)
        return
    print(json.dumps({"method": args.method, "split": args.split, "mode": args.mode,
                      "status": status["status"], "reason": status["reason"]}, indent=2), file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
