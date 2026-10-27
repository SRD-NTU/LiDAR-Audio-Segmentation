"""Parallel runner for the frozen four-metric SAID paper protocol.

The mask implementation is imported from the DCASE evaluator revision used by
the paper, while Macro Pearson r is imported from the later official revision
that introduced the class-macro correlation definition.  Upstream sources are
downloaded separately and are never copied into the SAID distribution.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any

import numpy as np
from tqdm import tqdm

from .class_f1 import (
    find_ground_truth,
    match_spatial_pairs,
    prediction_files,
    prediction_sequence_name,
)


_MASK_OFFICIAL: ModuleType | None = None
_PEARSON_OFFICIAL: ModuleType | None = None


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _group_by_frame(annotations: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        grouped[int(annotation["metadata_frame_index"])].append(annotation)
    return grouped


def _pearson_payload(accumulator: Any) -> dict[str, Any]:
    return {
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
    }


def _evaluate_recording(job: tuple[int, str, str, str]) -> dict[str, Any]:
    index, sequence, prediction_path, ground_truth_path = job
    assert _MASK_OFFICIAL is not None
    assert _PEARSON_OFFICIAL is not None

    predictions = _PEARSON_OFFICIAL.load_pred_json(prediction_path)
    ground_truth = _PEARSON_OFFICIAL.load_gt_json(ground_truth_path)

    pearson_accumulator = _PEARSON_OFFICIAL.EvalAccumulator()
    _PEARSON_OFFICIAL.evaluate_sequence(
        sequence,
        predictions,
        ground_truth,
        pearson_accumulator,
        _PEARSON_OFFICIAL.DEFAULT_IOU_THRESHOLDS,
    )

    predictions_by_frame = _group_by_frame(predictions)
    ground_truth_by_frame = _group_by_frame(ground_truth)
    class_count = int(_PEARSON_OFFICIAL.N_CLASSES)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    gt_class_counts = np.zeros(class_count, dtype=np.int64)

    sigma_rad = math.radians(float(_MASK_OFFICIAL.DEFAULT_SIGMA_DEG))
    inverse_two_sigma_squared = 0.5 / (sigma_rad * sigma_rad)
    cutoff_rad = float(_MASK_OFFICIAL._BLOB_CUTOFF_SIGMA) * sigma_rad

    images: list[dict[str, Any]] = []
    gt_annotations: list[dict[str, Any]] = []
    prediction_annotations: list[dict[str, Any]] = []
    next_gt_id = 1
    frame_indices = sorted(set(predictions_by_frame) | set(ground_truth_by_frame))
    for local_image_id, frame_index in enumerate(frame_indices, start=1):
        images.append(
            {
                "id": local_image_id,
                "width": int(_MASK_OFFICIAL.IMG_W),
                "height": int(_MASK_OFFICIAL.IMG_H),
            }
        )
        frame_predictions = predictions_by_frame.get(frame_index, [])
        frame_ground_truth = ground_truth_by_frame.get(frame_index, [])

        for annotation in frame_ground_truth:
            class_id = int(annotation["category_id"])
            if 0 <= class_id < class_count:
                gt_class_counts[class_id] += 1
        for gt_index, prediction_index in match_spatial_pairs(
            frame_ground_truth,
            frame_predictions,
        ):
            gt_class = int(frame_ground_truth[gt_index]["category_id"])
            predicted_class = int(
                frame_predictions[prediction_index]["category_id"]
            )
            if 0 <= gt_class < class_count and 0 <= predicted_class < class_count:
                confusion[gt_class, predicted_class] += 1

        rendered_gt = [
            _MASK_OFFICIAL.preprocess_annotation(
                annotation,
                sigma_rad,
                inverse_two_sigma_squared,
                cutoff_rad,
                float(_MASK_OFFICIAL.DEFAULT_ETH),
            )
            for annotation in frame_ground_truth
        ]
        rendered_predictions = [
            _MASK_OFFICIAL.preprocess_annotation(
                annotation,
                sigma_rad,
                inverse_two_sigma_squared,
                cutoff_rad,
                float(_MASK_OFFICIAL.DEFAULT_ETH),
            )
            for annotation in frame_predictions
        ]
        for item in rendered_gt:
            if not item["valid"]:
                continue
            gt_annotations.append(
                {
                    "id": next_gt_id,
                    "image_id": local_image_id,
                    "category_id": int(item["cid"]) + 1,
                    "segmentation": item["rle"],
                    "area": int(item["area"]),
                    "bbox": item["bbox"],
                    "iscrowd": 0,
                }
            )
            next_gt_id += 1
        for item in rendered_predictions:
            if not item["valid"]:
                continue
            prediction_annotations.append(
                {
                    "image_id": local_image_id,
                    "category_id": int(item["cid"]) + 1,
                    "segmentation": item["rle"],
                    "score": float(item["score"]),
                }
            )

    return {
        "index": index,
        "sequence": sequence,
        "images": images,
        "gt_annotations": gt_annotations,
        "prediction_annotations": prediction_annotations,
        "pearson": _pearson_payload(pearson_accumulator),
        "confusion": confusion.tolist(),
        "gt_class_counts": gt_class_counts.tolist(),
    }


def _merge_pearson(accumulator: Any, payload: dict[str, Any]) -> None:
    for class_id, records in payload["cls_records"].items():
        accumulator.cls_records[int(class_id)].extend(records)
    for class_id, count in payload["cls_n_gt"].items():
        accumulator.cls_n_gt[int(class_id)] += int(count)
    accumulator.micro_records.extend(payload["micro_records"])
    accumulator.micro_n_gt += int(payload["micro_n_gt"])


def _macro_pearson(module: ModuleType, accumulator: Any) -> float:
    class_means: list[float] = []
    pearson_stats = getattr(module, "_pearson_stats", None)
    if callable(pearson_stats):
        for class_id in range(int(module.N_CLASSES)):
            if int(accumulator.cls_n_gt[class_id]) <= 0:
                continue
            value, count = pearson_stats(accumulator.cls_records[class_id])
            if count > 0 and math.isfinite(float(value)):
                class_means.append(float(value))
        return float(np.mean(class_means)) if class_means else float("nan")
    with redirect_stdout(io.StringIO()):
        metrics = module.print_results(
            accumulator, module.DEFAULT_IOU_THRESHOLDS, None
        )
    return float(metrics["macro_pearson_r"])


def _coco_mask_metrics(
    module: ModuleType,
    images: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    class_agnostic: bool,
) -> dict[str, Any]:
    if not bool(getattr(module, "HAS_COCO", False)):
        raise ImportError(
            "pycocotools is required for the frozen DCASE Mask AP protocol"
        )
    if class_agnostic:
        categories = [
            {"id": 1, "name": "All classes", "supercategory": "sound"}
        ]
        gt_input = [dict(annotation, category_id=1) for annotation in ground_truth]
        prediction_input = [
            dict(annotation, category_id=1) for annotation in predictions
        ]
    else:
        categories = [dict(category) for category in module.COCO_CATS]
        gt_input = [dict(annotation) for annotation in ground_truth]
        prediction_input = [dict(annotation) for annotation in predictions]

    coco_gt = module.COCO()
    coco_gt.dataset = {
        "images": [dict(image) for image in images],
        "annotations": gt_input,
        "categories": categories,
    }
    with redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_predictions = coco_gt.loadRes(prediction_input)
        evaluator = module.COCOeval(coco_gt, coco_predictions, iouType="segm")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    precision = evaluator.eval["precision"]
    per_category: dict[str, float] = {}
    for category_index, category in enumerate(categories):
        values = precision[:, :, category_index, 0, 2]
        valid = values[values > -1]
        per_category[str(category["name"])] = (
            float(np.mean(valid)) if valid.size else float("nan")
        )
    return {
        "ap": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
        "per_category": per_category,
    }


def _matched_class_macro_f1(
    confusion: np.ndarray, gt_class_counts: np.ndarray
) -> float:
    values: list[float] = []
    for class_id in np.flatnonzero(gt_class_counts):
        true_positive = int(confusion[class_id, class_id])
        predicted = int(confusion[:, class_id].sum())
        support = int(confusion[class_id, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        values.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(values)) if values else 0.0


def _run(args: argparse.Namespace) -> dict[str, Any]:
    global _MASK_OFFICIAL, _PEARSON_OFFICIAL
    _MASK_OFFICIAL = _load_module(
        args.mask_script.resolve(), "said_external_dcase_mask_evaluator"
    )
    _PEARSON_OFFICIAL = _load_module(
        args.pearson_script.resolve(), "said_external_dcase_pearson_evaluator"
    )
    if hasattr(_PEARSON_OFFICIAL, "MAX_JSON_BYTES"):
        _PEARSON_OFFICIAL.MAX_JSON_BYTES = sys.maxsize

    files = prediction_files(args.pred_dir)
    jobs: list[tuple[int, str, str, str]] = []
    for index, prediction in enumerate(files):
        sequence = prediction_sequence_name(prediction)
        ground_truth = find_ground_truth(
            sequence, args.gt_dir, split=args.split
        )
        jobs.append((index, sequence, str(prediction), str(ground_truth)))

    workers = min(max(1, int(args.workers)), len(jobs))
    print(
        "[metrics] (4.2) reconstruct per-recording masks",
        flush=True,
    )
    print(
        f"[metrics] recordings: {len(jobs)}, workers: {workers}", flush=True
    )
    results: dict[int, dict[str, Any]] = {}
    if workers == 1:
        iterator = map(_evaluate_recording, jobs)
        for result in tqdm(
            iterator,
            total=len(jobs),
            desc="paper per-recording",
            unit="recording",
            dynamic_ncols=True,
        ):
            results[int(result["index"])] = result
    else:
        context = multiprocessing.get_context("fork")
        with context.Pool(processes=workers) as pool:
            iterator = pool.imap_unordered(_evaluate_recording, jobs, chunksize=1)
            for result in tqdm(
                iterator,
                total=len(jobs),
                desc="paper per-recording",
                unit="recording",
                dynamic_ncols=True,
            ):
                results[int(result["index"])] = result

    images: list[dict[str, Any]] = []
    gt_annotations: list[dict[str, Any]] = []
    prediction_annotations: list[dict[str, Any]] = []
    pearson_accumulator = _PEARSON_OFFICIAL.EvalAccumulator()
    class_count = int(_PEARSON_OFFICIAL.N_CLASSES)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    gt_class_counts = np.zeros(class_count, dtype=np.int64)
    next_image_id = 1
    next_gt_id = 1
    for index in sorted(results):
        result = results[index]
        image_offset = next_image_id - 1
        gt_offset = next_gt_id - 1
        for image in result["images"]:
            copied = dict(image)
            copied["id"] = int(copied["id"]) + image_offset
            images.append(copied)
        for annotation in result["gt_annotations"]:
            copied = dict(annotation)
            copied["id"] = int(copied["id"]) + gt_offset
            copied["image_id"] = int(copied["image_id"]) + image_offset
            gt_annotations.append(copied)
        for annotation in result["prediction_annotations"]:
            copied = dict(annotation)
            copied["image_id"] = int(copied["image_id"]) + image_offset
            prediction_annotations.append(copied)
        _merge_pearson(pearson_accumulator, result["pearson"])
        confusion += np.asarray(result["confusion"], dtype=np.int64)
        gt_class_counts += np.asarray(
            result["gt_class_counts"], dtype=np.int64
        )
        next_image_id += len(result["images"])
        next_gt_id += len(result["gt_annotations"])

    print(
        "[metrics] (4.2) class-aware COCO Mask mAP "
        "(IoU 0.50:0.95)",
        flush=True,
    )
    started = time.monotonic()
    aware = _coco_mask_metrics(
        _MASK_OFFICIAL,
        images,
        gt_annotations,
        prediction_annotations,
        class_agnostic=False,
    )
    print(
        f"[metrics] class-aware complete in {time.monotonic() - started:.1f}s",
        flush=True,
    )

    print(
        "[metrics] (4.2) class-agnostic COCO Mask AP",
        flush=True,
    )
    started = time.monotonic()
    agnostic = _coco_mask_metrics(
        _MASK_OFFICIAL,
        images,
        gt_annotations,
        prediction_annotations,
        class_agnostic=True,
    )
    pearson = _macro_pearson(_PEARSON_OFFICIAL, pearson_accumulator)
    class_f1 = _matched_class_macro_f1(confusion, gt_class_counts)
    print(
        f"[metrics] class-agnostic complete in "
        f"{time.monotonic() - started:.1f}s",
        flush=True,
    )
    return {
        "official_macro_map": aware["ap"],
        "official_macro_pearson_r": pearson,
        "class_agnostic_mask_ap": agnostic["ap"],
        "matched_class_macro_f1": class_f1,
        "details": {
            "class_aware": aware,
            "class_agnostic": agnostic,
            "counts": {
                "recordings": len(jobs),
                "frames_with_annotations_or_predictions": len(images),
                "ground_truth_annotations": len(gt_annotations),
                "prediction_annotations": len(prediction_annotations),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-script", type=Path, required=True)
    parser.add_argument("--pearson-script", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--split", default="dev-test")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    metrics = _run(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
