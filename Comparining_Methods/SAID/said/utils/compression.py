"""Reproducible compression for SAID DCASE prediction JSON files."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


COMPRESSION_MANIFEST_SCHEMA = "said-prediction-compression-v1"
DECIMAL_JSON_LIMIT_BYTES = 20_000_000


@dataclass(frozen=True)
class CompressionPreset:
    """Complete, serializable point-selection specification."""

    name: str
    relative_floor: float
    low_confidence_grid_pixels: float
    low_confidence_boundary_offset_pixels: float
    boundary_energy_relative_to_peak: float
    boundary_band_relative_to_peak: float
    high_confidence_threshold: float
    high_confidence_grid_pixels: float
    high_confidence_boundary_offset_pixels: float
    point_budget: int
    unlimited_budget_class_ids: tuple[int, ...]
    xy_decimals: int
    energy_decimals: int
    integer_xy: bool

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["unlimited_budget_class_ids"] = list(
            self.unlimited_budget_class_ids
        )
        return payload


# This preset records the executable parameters used for the final Table 1
# artifact. The paper describes the relative floor, confidence split, grids,
# boundary offset, and boundary energy. The manifest also records the point
# cap and numeric serialization used by the evaluated artifact.
PAPER_TABLE1_PRESET = CompressionPreset(
    name="paper-table1",
    relative_floor=0.10,
    low_confidence_grid_pixels=6.0,
    low_confidence_boundary_offset_pixels=2.0,
    boundary_energy_relative_to_peak=0.12,
    boundary_band_relative_to_peak=0.22,
    high_confidence_threshold=0.20,
    high_confidence_grid_pixels=2.0,
    high_confidence_boundary_offset_pixels=0.0,
    point_budget=88,
    unlimited_budget_class_ids=(2, 3, 5, 10),
    xy_decimals=3,
    energy_decimals=4,
    integer_xy=True,
)

# The first fallback follows the recorded stricter per-recording setting:
# high confidence starts at 0.25 and low-confidence boundary seeds are
# coarsened with a four-pixel offset. Applying the 88-point cap to every class
# makes this preset strictly size-bounded relative to the Table 1 preset.
STRICT_88_PRESET = replace(
    PAPER_TABLE1_PRESET,
    name="strict-88",
    low_confidence_boundary_offset_pixels=4.0,
    high_confidence_threshold=0.25,
    unlimited_budget_class_ids=(),
)

# A 64-point cap was the final recorded size fallback. If an unusually long
# recording remains above the limit, compression stops with an explicit error
# instead of deleting detections.
STRICT_64_PRESET = replace(
    STRICT_88_PRESET,
    name="strict-64",
    point_budget=64,
)

STRICT_48_PRESET = replace(
    STRICT_88_PRESET,
    name="strict-48",
    point_budget=48,
)

STRICT_32_PRESET = replace(
    STRICT_88_PRESET,
    name="strict-32",
    point_budget=32,
)

DEFAULT_FALLBACK_PRESETS = (
    STRICT_88_PRESET,
    STRICT_64_PRESET,
    STRICT_48_PRESET,
    STRICT_32_PRESET,
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _flatten_triplets(annotation: dict[str, Any]) -> np.ndarray:
    rows: list[list[float]] = []
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list):
        raise ValueError("annotation segmentation must be a list")
    for component in segmentation:
        if not isinstance(component, list):
            raise ValueError("annotation segmentation components must be lists")
        for triplet in component:
            if not isinstance(triplet, list) or len(triplet) < 3:
                raise ValueError("segmentation points must be [x, y, energy] triplets")
            x, y, energy = map(float, triplet[:3])
            if not all(math.isfinite(value) for value in (x, y, energy)):
                raise ValueError("segmentation points must contain finite values")
            if energy > 0.0:
                rows.append([x, y, energy])
    if not rows:
        raise ValueError("each retained detection must contain a positive-energy point")
    return np.asarray(rows, dtype=np.float32)


def _best_per_grid(points: np.ndarray, grid_pixels: float) -> np.ndarray:
    if grid_pixels <= 0.0 or not math.isfinite(float(grid_pixels)):
        raise ValueError("grid spacing must be finite and positive")
    if len(points) <= 1:
        return points
    order = np.argsort(points[:, 2])[::-1]
    occupied: set[tuple[int, int]] = set()
    selected: list[int] = []
    for index in order.tolist():
        x, y = float(points[index, 0]), float(points[index, 1])
        cell = (int(x // grid_pixels), int(y // grid_pixels))
        if cell in occupied:
            continue
        occupied.add(cell)
        selected.append(index)
    return points[np.asarray(selected, dtype=np.int64)]


def _append_boundary_support(
    points: np.ndarray,
    *,
    relative_floor: float,
    offset_pixels: float,
    boundary_energy_relative_to_peak: float,
    boundary_band_relative_to_peak: float,
) -> np.ndarray:
    if offset_pixels <= 0.0:
        return points
    peak = float(points[:, 2].max())
    band_high = max(boundary_band_relative_to_peak, relative_floor)
    boundary = points[
        (points[:, 2] >= relative_floor * peak)
        & (points[:, 2] <= band_high * peak)
    ]
    if len(boundary) == 0:
        boundary = points[points[:, 2] >= relative_floor * peak]
    boundary = _best_per_grid(
        boundary,
        grid_pixels=max(2.0, float(offset_pixels)),
    )
    energy = max(
        boundary_energy_relative_to_peak * peak,
        relative_floor * peak,
    )
    offsets = np.asarray(
        [
            [offset_pixels, 0.0],
            [-offset_pixels, 0.0],
            [0.0, offset_pixels],
            [0.0, -offset_pixels],
            [0.7071 * offset_pixels, 0.7071 * offset_pixels],
            [0.7071 * offset_pixels, -0.7071 * offset_pixels],
            [-0.7071 * offset_pixels, 0.7071 * offset_pixels],
            [-0.7071 * offset_pixels, -0.7071 * offset_pixels],
        ],
        dtype=np.float32,
    )
    rings: list[np.ndarray] = []
    for offset in offsets:
        coordinates = boundary[:, :2] + offset[None, :]
        coordinates[:, 0] = np.mod(coordinates[:, 0], 360.0)
        coordinates[:, 1] = np.clip(coordinates[:, 1], 0.0, 179.0)
        energies = np.full((len(coordinates), 1), energy, dtype=np.float32)
        rings.append(np.concatenate((coordinates, energies), axis=1))
    return np.concatenate((points, *rings), axis=0)


def _wrapped_coordinates(points: np.ndarray) -> np.ndarray:
    coordinates = points[:, :2].astype(np.float32, copy=True)
    coordinates[:, 0] = np.mod(coordinates[:, 0], 360.0)
    coordinates[:, 1] = np.clip(coordinates[:, 1], 0.0, 179.0)
    return coordinates


def _minimum_wrapped_distance_squared(
    candidates: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    if len(selected) == 0:
        return np.full((len(candidates),), np.inf, dtype=np.float32)
    delta_x = np.abs(candidates[:, None, 0] - selected[None, :, 0])
    delta_x = np.minimum(delta_x, 360.0 - delta_x)
    delta_y = candidates[:, None, 1] - selected[None, :, 1]
    return (delta_x * delta_x + delta_y * delta_y).min(axis=1)


def _spread_pick(
    candidates: np.ndarray,
    count: int,
    selected: list[list[float]],
) -> list[list[float]]:
    if count <= 0 or len(candidates) == 0:
        return []
    if len(candidates) <= count:
        return candidates[np.argsort(candidates[:, 2])[::-1]].tolist()
    candidate_coordinates = _wrapped_coordinates(candidates)
    selected_coordinates = (
        _wrapped_coordinates(np.asarray(selected, dtype=np.float32))
        if selected
        else np.zeros((0, 2), dtype=np.float32)
    )
    minimum_distance_squared = _minimum_wrapped_distance_squared(
        candidate_coordinates,
        selected_coordinates,
    )
    used = np.zeros((len(candidates),), dtype=bool)
    picked: list[list[float]] = []
    peak = max(float(candidates[:, 2].max()), 1e-9)
    for _ in range(count):
        score = (
            np.minimum(minimum_distance_squared, 36.0**2) / (36.0**2)
            + 0.35 * candidates[:, 2] / peak
        )
        score[used] = -1.0
        index = int(score.argmax())
        if score[index] < 0.0:
            break
        used[index] = True
        picked.append(candidates[index].tolist())
        delta_x = np.abs(
            candidate_coordinates[:, 0] - candidate_coordinates[index, 0]
        )
        delta_x = np.minimum(delta_x, 360.0 - delta_x)
        delta_y = (
            candidate_coordinates[:, 1] - candidate_coordinates[index, 1]
        )
        minimum_distance_squared = np.minimum(
            minimum_distance_squared,
            delta_x * delta_x + delta_y * delta_y,
        )
    return picked


def _apply_point_budget(
    points: np.ndarray,
    *,
    point_budget: int,
    grid_pixels: float,
) -> np.ndarray:
    if point_budget <= 0 or len(points) <= point_budget:
        return points[np.argsort(points[:, 2])[::-1]]
    candidates = _best_per_grid(points, grid_pixels=grid_pixels)
    if len(candidates) < point_budget:
        candidates = points
    peak_index = int(candidates[:, 2].argmax())
    selected: list[list[float]] = [candidates[peak_index].tolist()]
    peak = float(candidates[:, 2].max())
    bands = (
        candidates[candidates[:, 2] >= 0.70 * peak],
        candidates[
            (candidates[:, 2] >= 0.30 * peak)
            & (candidates[:, 2] < 0.70 * peak)
        ],
        candidates[candidates[:, 2] < 0.30 * peak],
    )
    remaining = point_budget - 1
    quotas = (round(remaining * 0.20), round(remaining * 0.35), remaining)
    for band, quota in zip(bands, quotas, strict=True):
        count = min(max(0, int(quota)), point_budget - len(selected))
        selected.extend(_spread_pick(band, count, selected))
        if len(selected) >= point_budget:
            break
    if len(selected) < point_budget:
        selected.extend(
            _spread_pick(candidates, point_budget - len(selected), selected)
        )
    unique: dict[tuple[int, int], list[float]] = {}
    for x, y, energy in selected:
        key = (int(round(float(x) * 1000)), int(round(float(y) * 1000)))
        previous = unique.get(key)
        if previous is None or float(energy) > float(previous[2]):
            unique[key] = [float(x), float(y), float(energy)]
    output = np.asarray(
        sorted(
            unique.values(),
            key=lambda row: (
                -float(row[2]),
                float(row[0]),
                float(row[1]),
            ),
        ),
        dtype=np.float32,
    )
    return output[:point_budget]


def compress_annotation(
    annotation: dict[str, Any],
    preset: CompressionPreset = PAPER_TABLE1_PRESET,
) -> dict[str, Any]:
    """Compress one detection while preserving its metadata and identity."""

    try:
        confidence = float(annotation["score"])
        category_id = int(annotation["category_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("each annotation requires numeric score and category_id") from error
    if not math.isfinite(confidence):
        raise ValueError("annotation score must be finite")
    points = _flatten_triplets(annotation)
    peak = float(points[:, 2].max())
    points = points[points[:, 2] >= preset.relative_floor * peak]
    if len(points) == 0:
        raise RuntimeError("relative-floor selection removed every point")
    if confidence >= preset.high_confidence_threshold:
        grid_pixels = preset.high_confidence_grid_pixels
        boundary_offset_pixels = (
            preset.high_confidence_boundary_offset_pixels
        )
    else:
        grid_pixels = preset.low_confidence_grid_pixels
        boundary_offset_pixels = preset.low_confidence_boundary_offset_pixels
    points = _append_boundary_support(
        points,
        relative_floor=preset.relative_floor,
        offset_pixels=boundary_offset_pixels,
        boundary_energy_relative_to_peak=(
            preset.boundary_energy_relative_to_peak
        ),
        boundary_band_relative_to_peak=(
            preset.boundary_band_relative_to_peak
        ),
    )
    budget = (
        0
        if category_id in preset.unlimited_budget_class_ids
        else preset.point_budget
    )
    if budget > 0 and len(points) > budget:
        points = _apply_point_budget(
            points,
            point_budget=budget,
            grid_pixels=grid_pixels,
        )
    elif budget <= 0:
        points = _best_per_grid(points, grid_pixels=grid_pixels)
        points = points[np.argsort(points[:, 2])[::-1]]
    else:
        points = points[np.argsort(points[:, 2])[::-1]]

    def format_coordinate(value: float) -> int | float:
        rounded = round(float(value), preset.xy_decimals)
        if preset.integer_xy and float(rounded).is_integer():
            return int(rounded)
        return rounded

    output = deepcopy(annotation)
    output["score"] = round(confidence, 6)
    output["segmentation"] = [
        [
            [
                format_coordinate(x),
                format_coordinate(y),
                round(float(energy), preset.energy_decimals),
            ]
            for x, y, energy in points.tolist()
        ]
    ]
    return output


def _write_compressed_file(
    source: Path,
    destination: Path,
    preset: CompressionPreset,
) -> dict[str, Any]:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read prediction JSON: {source}") from error
    if not isinstance(payload, dict) or not isinstance(
        payload.get("annotations"), list
    ):
        raise ValueError(f"prediction JSON has no annotations list: {source}")
    if payload.get("schema") == "audio2sph-prediction-json-v1":
        raise ValueError(
            "Audio2Sph JSON is class-agnostic and is not a DCASE submission "
            f"prediction: {source}"
        )
    output = deepcopy(payload)
    if not all(isinstance(annotation, dict) for annotation in payload["annotations"]):
        raise ValueError(f"prediction annotations must be objects: {source}")
    output["annotations"] = [
        compress_annotation(annotation, preset)
        for annotation in payload["annotations"]
    ]
    if len(output["annotations"]) != len(payload["annotations"]):
        raise RuntimeError("compression must retain every detection")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                output,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "input_detection_count": len(payload["annotations"]),
        "output_detection_count": len(output["annotations"]),
        "output_bytes": destination.stat().st_size,
        "output_sha256": _sha256(destination),
    }


def _compress_task(
    task: tuple[str, str, CompressionPreset, int, tuple[CompressionPreset, ...]],
) -> dict[str, Any]:
    source_value, destination_value, initial, limit, fallbacks = task
    source = Path(source_value)
    destination = Path(destination_value)
    attempts: list[dict[str, Any]] = []
    selected = initial
    result = _write_compressed_file(source, destination, selected)
    attempts.append(
        {
            "preset": selected.name,
            "output_bytes": result["output_bytes"],
        }
    )
    for fallback in fallbacks:
        if int(result["output_bytes"]) <= limit:
            break
        selected = fallback
        result = _write_compressed_file(source, destination, selected)
        attempts.append(
            {
                "preset": selected.name,
                "output_bytes": result["output_bytes"],
            }
        )
    if int(result["output_bytes"]) > limit:
        raise RuntimeError(
            f"{source.name} remains above {limit} bytes after preset "
            f"{selected.name}: {result['output_bytes']} bytes"
        )
    return {
        "file": source.name,
        "input_bytes": source.stat().st_size,
        "input_sha256": _sha256(source),
        **result,
        "preset": selected.name,
        "fallback_applied": selected.name != initial.name,
        "attempts": attempts,
    }


def _prediction_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    nested = resolved / "inference_outputs"
    if nested.is_dir():
        return nested
    if not resolved.is_dir():
        raise FileNotFoundError(f"prediction directory not found: {resolved}")
    return resolved


def compress_prediction_directory(
    predictions: str | Path,
    output_directory: str | Path,
    *,
    workers: int | None = None,
    maximum_file_bytes: int = DECIMAL_JSON_LIMIT_BYTES,
    preset: CompressionPreset = PAPER_TABLE1_PRESET,
    fallback_presets: tuple[CompressionPreset, ...] = DEFAULT_FALLBACK_PRESETS,
) -> Path:
    """Compress a prediction directory and atomically publish its manifest."""

    if maximum_file_bytes <= 0:
        raise ValueError("maximum_file_bytes must be positive")
    source_directory = _prediction_directory(Path(predictions))
    sources = sorted(source_directory.glob("*_inference.json"))
    if not sources:
        raise FileNotFoundError(
            f"no *_inference.json files found under {source_directory}"
        )
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"compression output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    if workers is None:
        workers = max(1, math.floor((os.cpu_count() or 1) * 0.75))
    workers = max(1, min(int(workers), len(sources)))
    try:
        tasks = [
            (
                str(source),
                str(temporary / source.name),
                preset,
                int(maximum_file_bytes),
                fallback_presets,
            )
            for source in sources
        ]
        if workers == 1:
            records = [_compress_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                records = list(executor.map(_compress_task, tasks))
        records.sort(key=lambda record: str(record["file"]))
        total_bytes = sum(int(record["output_bytes"]) for record in records)
        manifest = {
            "schema": COMPRESSION_MANIFEST_SCHEMA,
            "source_directory": str(source_directory),
            "file_count": len(records),
            "maximum_file_bytes": int(maximum_file_bytes),
            "all_prediction_files_within_limit": all(
                int(record["output_bytes"]) <= maximum_file_bytes
                for record in records
            ),
            "max_prediction_file_bytes": max(
                int(record["output_bytes"]) for record in records
            ),
            "average_prediction_file_bytes": total_bytes / len(records),
            "total_prediction_bytes": total_bytes,
            "initial_preset": preset.to_manifest(),
            "fallback_presets": [
                fallback.to_manifest() for fallback in fallback_presets
            ],
            "fallback_file_count": sum(
                bool(record["fallback_applied"]) for record in records
            ),
            "files": records,
        }
        manifest_path = temporary / "compression_manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "compression_manifest.json"


__all__ = [
    "COMPRESSION_MANIFEST_SCHEMA",
    "DECIMAL_JSON_LIMIT_BYTES",
    "CompressionPreset",
    "DEFAULT_FALLBACK_PRESETS",
    "PAPER_TABLE1_PRESET",
    "STRICT_64_PRESET",
    "STRICT_48_PRESET",
    "STRICT_32_PRESET",
    "STRICT_88_PRESET",
    "compress_annotation",
    "compress_prediction_directory",
]
