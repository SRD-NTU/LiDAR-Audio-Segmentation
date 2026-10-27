"""Convert SAID predictions to standard DCASE2026 JSON."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from ..inference import PredictionChunk
from ..data.dcase2026 import DCASE2026_CLASS_NAMES


def dense_map_to_dcase_points(
    refined_map: Tensor | np.ndarray,
    *,
    relative_threshold: float = 0.10,
) -> list[list[float]]:
    """Convert one paper-resolution map to DCASE ``[x,y,intensity]`` points."""

    if not math.isfinite(float(relative_threshold)) or not 0.0 <= float(relative_threshold) <= 1.0:
        raise ValueError("relative_threshold must be finite and between zero and one")
    if isinstance(refined_map, Tensor):
        values = refined_map.detach().cpu().numpy()
    else:
        values = np.asarray(refined_map)
    if values.shape != (180, 360):
        raise ValueError(f"refined_map must have shape (180,360), got {values.shape}")
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("refined_map must contain finite values")
    if values.size and (float(values.min()) < 0.0 or float(values.max()) > 1.0):
        raise ValueError("refined_map values must lie between zero and one")
    flat = values.reshape(-1)
    peak = float(flat.max())
    if peak <= 0.0:
        return []
    selected = np.flatnonzero(flat >= float(relative_threshold) * peak)
    if selected.size == 0:
        selected = np.asarray([int(flat.argmax())], dtype=np.int64)
    selected = selected[np.argsort(flat[selected])[::-1]]
    y = selected // 360
    x_spherical = selected % 360
    x_dcase = (179 - x_spherical) % 360
    energy = np.round(flat[selected].astype(np.float64), decimals=6)
    return np.column_stack((x_dcase, y, energy)).astype(np.float64).tolist()


def _annotations(
    chunks: Iterable[PredictionChunk],
    *,
    confidence_threshold: float,
    max_sources_per_frame: int,
    relative_map_threshold: float,
) -> Iterable[dict]:
    expected_frame_start = 0
    for chunk in chunks:
        if int(chunk.frame_start) != expected_frame_start:
            raise ValueError("prediction chunks must be contiguous and begin at frame zero")
        maps = chunk.refined_maps
        class_ids = chunk.slot_class_ids
        confidence = chunk.slot_confidence
        if maps.ndim != 4 or tuple(maps.shape[-2:]) != (180, 360):
            raise ValueError("refined_maps must have shape [frames,slots,180,360]")
        if tuple(class_ids.shape) != tuple(maps.shape[:2]) or tuple(confidence.shape) != tuple(maps.shape[:2]):
            raise ValueError("prediction chunk tensors have inconsistent frame or slot dimensions")
        for local_frame in range(int(maps.shape[0])):
            candidates = torch.nonzero(
                confidence[local_frame] >= float(confidence_threshold),
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                continue
            order = torch.argsort(confidence[local_frame, candidates], descending=True)
            selected_slots = candidates[order[: int(max_sources_per_frame)]]
            for export_rank, slot_tensor in enumerate(selected_slots):
                slot = int(slot_tensor)
                class_id = int(class_ids[local_frame, slot])
                if not 0 <= class_id < len(DCASE2026_CLASS_NAMES):
                    raise ValueError("slot class ID is outside the public taxonomy")
                points = dense_map_to_dcase_points(
                    maps[local_frame, slot],
                    relative_threshold=relative_map_threshold,
                )
                if not points:
                    continue
                yield {
                    "metadata_frame_index": int(chunk.frame_start) + local_frame,
                    "instance_id": slot + 1,
                    "category_id": class_id,
                    "score": round(float(confidence[local_frame, slot]), 6),
                    "segmentation": [points],
                }
        expected_frame_start = int(chunk.frame_stop)


def write_dcase_prediction_json(
    chunks: Iterable[PredictionChunk],
    output_file: str | Path,
    *,
    confidence_threshold: float = 0.05,
    max_sources_per_frame: int = 4,
    relative_map_threshold: float = 0.10,
) -> Path:
    """Write one complete recording without retaining all annotations in memory."""

    if not math.isfinite(float(confidence_threshold)) or not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence_threshold must be finite and between zero and one")
    if max_sources_per_frame <= 0:
        raise ValueError("max_sources_per_frame must be positive")
    return _write_annotations_json(
        _annotations(
            chunks,
            confidence_threshold=confidence_threshold,
            max_sources_per_frame=max_sources_per_frame,
            relative_map_threshold=relative_map_threshold,
        ),
        output_file,
    )


def _write_annotations_json(
    annotations: Iterable[dict],
    output_file: str | Path,
) -> Path:
    """Atomically write a streaming sequence of DCASE annotations."""

    output = Path(output_file)
    if output.exists():
        raise FileExistsError(f"prediction file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    annotation_count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write('{"annotations":[\n')
            for annotation in annotations:
                if annotation_count:
                    stream.write(",\n")
                json.dump(annotation, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                annotation_count += 1
            stream.write("\n]}\n")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _archive_annotations(
    manifest_path: str | Path,
    *,
    relative_map_threshold: float,
) -> Iterable[dict]:
    """Yield DCASE annotations from an already selected lossless archive."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prediction manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "said-prediction-archive-v1":
        raise ValueError("DCASE JSON export requires a complete SAID archive")
    archive_directory = manifest_path.parent
    covered_frames = 0
    for record in manifest.get("chunks", []):
        frame_start = int(record["frame_start"])
        frame_stop = int(record["frame_stop"])
        if frame_start != covered_frames or frame_stop < frame_start:
            raise ValueError("prediction chunks must be ordered and contiguous")
        chunk_path = archive_directory / str(record["file"])
        with np.load(chunk_path, allow_pickle=False) as chunk:
            frame_indices = np.asarray(chunk["frame_index"], dtype=np.int64)
            slot_indices = np.asarray(chunk["slot_index"], dtype=np.int64)
            class_ids = np.asarray(chunk["class_id"], dtype=np.int64)
            confidence = np.asarray(chunk["confidence"], dtype=np.float32)
            maps = np.asarray(chunk["refined_map"], dtype=np.float32)
            if not (
                frame_indices.shape
                == slot_indices.shape
                == class_ids.shape
                == confidence.shape
                == maps.shape[:1]
            ):
                raise ValueError("prediction chunk arrays have inconsistent lengths")
            for index in range(int(frame_indices.size)):
                frame_index = int(frame_indices[index])
                class_id = int(class_ids[index])
                if not frame_start <= frame_index < frame_stop:
                    raise ValueError("prediction frame lies outside its chunk")
                if not 0 <= class_id < len(DCASE2026_CLASS_NAMES):
                    raise ValueError("slot class ID is outside the public taxonomy")
                points = dense_map_to_dcase_points(
                    maps[index],
                    relative_threshold=relative_map_threshold,
                )
                if not points:
                    continue
                yield {
                    "metadata_frame_index": frame_index,
                    "instance_id": int(slot_indices[index]) + 1,
                    "category_id": class_id,
                    "score": round(float(confidence[index]), 6),
                    "segmentation": [points],
                }
        covered_frames = frame_stop
    if covered_frames != int(manifest.get("frames", -1)):
        raise ValueError("prediction archive frame count does not match its manifest")


def write_dcase_archive_json(
    manifest_path: str | Path,
    output_file: str | Path,
    *,
    relative_map_threshold: float = 0.10,
) -> Path:
    """Convert a complete SAID archive to standard DCASE JSON."""

    if (
        not math.isfinite(float(relative_map_threshold))
        or not 0.0 <= float(relative_map_threshold) <= 1.0
    ):
        raise ValueError(
            "relative_map_threshold must be finite and between zero and one"
        )
    return _write_annotations_json(
        _archive_annotations(
            manifest_path,
            relative_map_threshold=relative_map_threshold,
        ),
        output_file,
    )
