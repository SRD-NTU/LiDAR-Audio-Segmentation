"""Class-F1 and class-collapsing utilities for DCASE predictions."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ...data.dcase2026 import DCASE2026_CLASS_NAMES


IMAGE_WIDTH = 360
IMAGE_HEIGHT = 180
MATCH_THRESHOLD_DEGREES = 20.0


def prediction_sequence_name(path: str | Path) -> str:
    """Return the recording identity represented by a prediction JSON."""

    name = Path(path).name
    if name.endswith("_inference.json"):
        return name[: -len("_inference.json")]
    if name.endswith(".json"):
        return name[:-5]
    raise ValueError(f"prediction is not a JSON file: {path}")


def prediction_files(directory: str | Path) -> list[Path]:
    """List recording prediction files while excluding generated metadata."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"prediction directory not found: {root}")
    excluded = {
        "compression_manifest.json",
        "evaluation_manifest.json",
        "evaluator.json",
        "manifest.json",
        "metrics.json",
    }
    files = sorted(
        path
        for path in root.glob("*.json")
        if not path.name.startswith(".") and path.name not in excluded
    )
    if not files:
        raise FileNotFoundError(f"no prediction JSON files found: {root}")
    identities = [prediction_sequence_name(path) for path in files]
    if len(identities) != len(set(identities)):
        raise ValueError("prediction directory contains duplicate recording identities")
    return files


def find_ground_truth(
    sequence: str,
    ground_truth: str | Path,
    *,
    split: str,
) -> Path:
    """Resolve one official label without assuming Sony or TAU subdirectory."""

    root = Path(ground_truth).expanduser().resolve()
    candidates = sorted(root.rglob(f"{sequence}_std.json"))
    if split:
        split_candidates = [
            path
            for path in candidates
            if split in path.parts or split == "dev-test"
        ]
        if split_candidates:
            candidates = split_candidates
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one ground-truth JSON for {sequence}, found {len(candidates)}"
        )
    return candidates[0]


def load_annotations(path: str | Path) -> list[dict[str, Any]]:
    """Load and minimally validate a DCASE annotation document."""

    with Path(path).open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    annotations = document.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(f"annotation document has no list field 'annotations': {path}")
    if not all(isinstance(annotation, dict) for annotation in annotations):
        raise ValueError(f"annotation list contains a non-object value: {path}")
    return annotations


def _peak(annotation: dict[str, Any]) -> tuple[float, float]:
    peak_x = 0.0
    peak_y = 0.0
    peak_energy = -math.inf
    for segment in annotation.get("segmentation", []):
        for point in segment:
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise ValueError("segmentation points must be [x, y, intensity] triplets")
            x, y, energy = map(float, point)
            if not all(math.isfinite(value) for value in (x, y, energy)):
                raise ValueError("segmentation points must contain finite values")
            if energy > peak_energy:
                peak_x, peak_y, peak_energy = x, y, energy
    if peak_energy == -math.inf:
        raise ValueError("annotation segmentation contains no points")
    return peak_x, peak_y


def angular_distance_matrix_degrees(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> np.ndarray:
    """Compute great-circle distances between annotation energy peaks."""

    if not ground_truth or not predictions:
        return np.empty((len(ground_truth), len(predictions)), dtype=np.float64)
    gt_peaks = np.asarray([_peak(item) for item in ground_truth], dtype=np.float64)
    pred_peaks = np.asarray([_peak(item) for item in predictions], dtype=np.float64)

    gt_azimuth = np.radians((gt_peaks[:, 0] / IMAGE_WIDTH) * 360.0 - 180.0)
    gt_elevation = np.radians(
        90.0 - (gt_peaks[:, 1] / (IMAGE_HEIGHT - 1)) * 180.0
    )
    pred_azimuth = np.radians(
        (pred_peaks[:, 0] / IMAGE_WIDTH) * 360.0 - 180.0
    )
    pred_elevation = np.radians(
        90.0 - (pred_peaks[:, 1] / (IMAGE_HEIGHT - 1)) * 180.0
    )

    elevation_delta = gt_elevation[:, None] - pred_elevation[None, :]
    azimuth_delta = gt_azimuth[:, None] - pred_azimuth[None, :]
    haversine = (
        np.sin(elevation_delta / 2.0) ** 2
        + np.cos(gt_elevation)[:, None]
        * np.cos(pred_elevation)[None, :]
        * np.sin(azimuth_delta / 2.0) ** 2
    )
    return np.degrees(
        2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    )


def match_spatial_pairs(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    threshold_degrees: float = MATCH_THRESHOLD_DEGREES,
) -> list[tuple[int, int]]:
    """Match sources class-agnostically within the angular threshold."""

    if not ground_truth or not predictions:
        return []
    distances = angular_distance_matrix_degrees(ground_truth, predictions)
    gated = distances.copy()
    gated[distances > float(threshold_degrees)] = 1.0e9
    gt_indices, prediction_indices = linear_sum_assignment(gated)
    return [
        (int(gt_index), int(prediction_index))
        for gt_index, prediction_index in zip(gt_indices, prediction_indices)
        if distances[gt_index, prediction_index] <= float(threshold_degrees)
    ]


def _recording_class_counts(
    job: tuple[str, str, float],
) -> tuple[np.ndarray, np.ndarray]:
    prediction_path, ground_truth_path, threshold_degrees = job
    class_count = len(DCASE2026_CLASS_NAMES)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    gt_counts = np.zeros(class_count, dtype=np.int64)
    prediction_annotations = load_annotations(prediction_path)
    gt_annotations = load_annotations(ground_truth_path)
    predictions_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ground_truth_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in prediction_annotations:
        predictions_by_frame[int(annotation["metadata_frame_index"])].append(
            annotation
        )
    for annotation in gt_annotations:
        class_id = int(annotation["category_id"])
        if not 0 <= class_id < class_count:
            raise ValueError(f"ground-truth class ID is out of range: {class_id}")
        gt_counts[class_id] += 1
        ground_truth_by_frame[int(annotation["metadata_frame_index"])].append(
            annotation
        )

    for frame_index in set(predictions_by_frame) | set(ground_truth_by_frame):
        frame_gt = ground_truth_by_frame.get(frame_index, [])
        frame_predictions = predictions_by_frame.get(frame_index, [])
        for gt_index, prediction_index in match_spatial_pairs(
            frame_gt,
            frame_predictions,
            threshold_degrees=threshold_degrees,
        ):
            gt_class = int(frame_gt[gt_index]["category_id"])
            predicted_class = int(
                frame_predictions[prediction_index]["category_id"]
            )
            if not 0 <= predicted_class < class_count:
                raise ValueError(
                    f"prediction class ID is out of range: {predicted_class}"
                )
            confusion[gt_class, predicted_class] += 1
    return confusion, gt_counts


def matched_class_macro_f1(
    predictions: str | Path,
    ground_truth: str | Path,
    *,
    split: str = "dev-test",
    threshold_degrees: float = MATCH_THRESHOLD_DEGREES,
    workers: int = 1,
    show_progress: bool = False,
) -> float:
    """Compute macro Class-F1 over class-agnostically matched source pairs."""

    class_count = len(DCASE2026_CLASS_NAMES)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    gt_counts = np.zeros(class_count, dtype=np.int64)
    jobs = []
    for prediction_path in prediction_files(predictions):
        sequence = prediction_sequence_name(prediction_path)
        ground_truth_path = find_ground_truth(
            sequence, ground_truth, split=split
        )
        jobs.append(
            (
                str(prediction_path),
                str(ground_truth_path),
                float(threshold_degrees),
            )
        )
    worker_count = min(max(1, int(workers)), len(jobs))
    if show_progress:
        print(
            f"[metrics] class-f1: {len(jobs)} recordings, "
            f"{worker_count} workers",
            flush=True,
        )

    if worker_count == 1:
        iterator = map(_recording_class_counts, jobs)
        for local_confusion, local_gt_counts in tqdm(
            iterator,
            total=len(jobs),
            desc="class-f1",
            unit="recording",
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            confusion += local_confusion
            gt_counts += local_gt_counts
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context
        ) as executor:
            futures = [
                executor.submit(_recording_class_counts, job) for job in jobs
            ]
            with tqdm(
                total=len(futures),
                desc="class-f1",
                unit="recording",
                dynamic_ncols=True,
                disable=not show_progress,
            ) as progress:
                for future in as_completed(futures):
                    local_confusion, local_gt_counts = future.result()
                    confusion += local_confusion
                    gt_counts += local_gt_counts
                    progress.update(1)

    per_class_f1: list[float] = []
    for class_id in np.flatnonzero(gt_counts):
        true_positive = int(confusion[class_id, class_id])
        predicted = int(confusion[:, class_id].sum())
        support = int(confusion[class_id, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        per_class_f1.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(per_class_f1)) if per_class_f1 else 0.0


def collapsed_annotations(
    source: str | Path,
    destination: str | Path,
) -> Path:
    """Write a class-agnostic copy for localization-only official evaluation."""

    source_path = Path(source)
    destination_path = Path(destination)
    with source_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    annotations = document.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(f"annotation document has no list field 'annotations': {source}")
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"annotation list contains a non-object value: {source}")
        annotation["category_id"] = 0
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    return destination_path


def selected_ground_truth_files(
    predictions: Iterable[Path],
    ground_truth: str | Path,
    *,
    split: str,
) -> dict[str, Path]:
    """Resolve the exact label set paired with a prediction directory."""

    return {
        prediction_sequence_name(path): find_ground_truth(
            prediction_sequence_name(path), ground_truth, split=split
        )
        for path in predictions
    }
