"""Parallel orchestration around the separately downloaded DCASE evaluator."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any

from tqdm import tqdm

from .class_f1 import prediction_files, prediction_sequence_name


_OFFICIAL: ModuleType | None = None


def _load_official(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "said_external_dcase_evaluator", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "MAX_JSON_BYTES"):
        module.MAX_JSON_BYTES = sys.maxsize
    return module


def _supports_parallel_api(module: ModuleType) -> bool:
    callables = (
        "EvalAccumulator",
        "evaluate_sequence",
        "find_gt_json",
        "load_gt_json",
        "load_pred_json",
        "plot_ap_curves",
        "print_results",
    )
    return hasattr(module, "DEFAULT_IOU_THRESHOLDS") and all(
        callable(getattr(module, name, None)) for name in callables
    )


def _plain_frame_iou(value: Any) -> dict[str, dict[int, dict[int, float]]]:
    return {
        str(sequence): {
            int(frame): {
                int(class_id): float(iou)
                for class_id, iou in classes.items()
            }
            for frame, classes in frames.items()
        }
        for sequence, frames in value.items()
    }


def _evaluate_one(job: tuple[int, str, str, str, bool]):
    index, sequence, prediction_path, ground_truth_path, collapse_classes = job
    assert _OFFICIAL is not None
    predictions = _OFFICIAL.load_pred_json(prediction_path)
    ground_truth = _OFFICIAL.load_gt_json(ground_truth_path)
    if collapse_classes:
        for annotation in predictions:
            annotation["category_id"] = 0
        for annotation in ground_truth:
            annotation["category_id"] = 0
    accumulator = _OFFICIAL.EvalAccumulator()
    _OFFICIAL.evaluate_sequence(
        sequence,
        predictions,
        ground_truth,
        accumulator,
        _OFFICIAL.DEFAULT_IOU_THRESHOLDS,
    )
    payload = {
        "cls_records": {
            int(class_id): records
            for class_id, records in accumulator.cls_records.items()
        },
        "cls_n_gt": {
            int(class_id): int(count)
            for class_id, count in accumulator.cls_n_gt.items()
        },
        "micro_records": accumulator.micro_records,
        "micro_n_gt": int(accumulator.micro_n_gt),
        "frame_iou": _plain_frame_iou(accumulator.frame_iou),
    }
    return index, sequence, len(predictions), len(ground_truth), payload


def _merge_payload(accumulator: Any, payload: dict[str, Any]) -> None:
    for class_id, records in payload["cls_records"].items():
        accumulator.cls_records[int(class_id)].extend(records)
    for class_id, count in payload["cls_n_gt"].items():
        accumulator.cls_n_gt[int(class_id)] += int(count)
    accumulator.micro_records.extend(payload["micro_records"])
    accumulator.micro_n_gt += int(payload["micro_n_gt"])
    for sequence, frames in payload["frame_iou"].items():
        for frame, classes in frames.items():
            accumulator.frame_iou[sequence][int(frame)].update(classes)


def _write_collapsed(source: Path, destination: Path) -> None:
    with source.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    annotations = document.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(f"annotation document has no list field: {source}")
    for annotation in annotations:
        annotation["category_id"] = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")


def _run_fallback(module: ModuleType, args: argparse.Namespace) -> None:
    if not callable(getattr(module, "main", None)):
        raise TypeError("official evaluator has no callable main()")
    prediction_root = Path(args.pred_dir)
    ground_truth_root = Path(args.gt_dir)
    with tempfile.TemporaryDirectory(prefix="said-official-fallback-") as work:
        normalized_predictions = Path(work) / "predictions"
        for source in tqdm(
            prediction_files(prediction_root),
            desc=f"prepare {args.stage} predictions",
            unit="recording",
            dynamic_ncols=True,
        ):
            destination = (
                normalized_predictions
                / f"{prediction_sequence_name(source)}.json"
            )
            if args.collapse_classes:
                _write_collapsed(source, destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        prediction_root = normalized_predictions

        if args.collapse_classes:
            collapsed_ground_truth = Path(work) / "ground_truth"
            ground_truth_files = sorted(ground_truth_root.rglob("*_std.json"))
            for source in tqdm(
                ground_truth_files,
                desc=f"prepare {args.stage} ground truth",
                unit="recording",
                dynamic_ncols=True,
            ):
                relative = source.relative_to(ground_truth_root)
                _write_collapsed(source, collapsed_ground_truth / relative)
            ground_truth_root = collapsed_ground_truth
        print(
            f"[metrics] {args.stage}: evaluator API changed; using its serial main()",
            flush=True,
        )
        original_argv = sys.argv
        try:
            sys.argv = [
                str(args.script),
                "--pred_dir",
                str(prediction_root),
                "--gt_dir",
                str(ground_truth_root),
                "--frames_base",
                str(args.frames_base),
                "--output_dir",
                str(args.output_dir),
                "--n_comp_seqs",
                "0",
            ]
            module.main()
        finally:
            sys.argv = original_argv


def _run_parallel(module: ModuleType, args: argparse.Namespace) -> None:
    files = prediction_files(args.pred_dir)
    jobs: list[tuple[int, str, str, str, bool]] = []
    skipped: list[str] = []
    for index, prediction in enumerate(files):
        sequence = prediction_sequence_name(prediction)
        ground_truth = module.find_gt_json(sequence, str(args.gt_dir))
        if ground_truth is None:
            skipped.append(sequence)
            continue
        jobs.append(
            (
                index,
                sequence,
                str(prediction),
                str(ground_truth),
                bool(args.collapse_classes),
            )
        )
    if not jobs:
        raise RuntimeError("official evaluator found no matched recordings")

    worker_count = min(max(1, int(args.workers)), len(jobs))
    print(
        f"[metrics] {args.stage}: {len(jobs)} recordings, "
        f"{worker_count} workers",
        flush=True,
    )
    global _OFFICIAL
    _OFFICIAL = module
    results: dict[int, tuple[str, int, int, dict[str, Any]]] = {}
    if worker_count == 1:
        iterator = map(_evaluate_one, jobs)
        for index, sequence, prediction_count, gt_count, payload in tqdm(
            iterator,
            total=len(jobs),
            desc=f"official {args.stage}",
            unit="recording",
            dynamic_ncols=True,
        ):
            results[index] = (sequence, prediction_count, gt_count, payload)
    else:
        context = multiprocessing.get_context("fork")
        with context.Pool(processes=worker_count) as pool:
            iterator = pool.imap_unordered(_evaluate_one, jobs, chunksize=1)
            for index, sequence, prediction_count, gt_count, payload in tqdm(
                iterator,
                total=len(jobs),
                desc=f"official {args.stage}",
                unit="recording",
                dynamic_ncols=True,
            ):
                results[index] = (
                    sequence,
                    prediction_count,
                    gt_count,
                    payload,
                )

    accumulator = module.EvalAccumulator()
    for index in sorted(results):
        _merge_payload(accumulator, results[index][3])
    output_context = (
        redirect_stdout(io.StringIO()) if args.quiet_results else nullcontext()
    )
    with output_context:
        metrics = module.print_results(
            accumulator, module.DEFAULT_IOU_THRESHOLDS, None
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    if not args.quiet_results:
        module.plot_ap_curves(metrics, str(output / "00_ap_curves.png"))
    if skipped:
        print(f"[metrics] skipped recordings without labels: {skipped}")
    print(f"[metrics] {args.stage}: complete", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--frames-base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--collapse-classes", action="store_true")
    parser.add_argument("--quiet-results", action="store_true")
    args = parser.parse_args()
    module = _load_official(args.script.resolve())
    if _supports_parallel_api(module):
        _run_parallel(module, args)
    else:
        _run_fallback(module, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
