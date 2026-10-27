"""Portable, chunked prediction archive for complete-recording inference."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import numpy as np
import torch

from ..data.dcase2026 import DCASE2026_CLASS_NAMES
from .infer import ClassAgnosticPredictionChunk, PredictionChunk


def _selected_slots(
    confidence: torch.Tensor,
    *,
    threshold: float,
    max_sources_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_indices: list[torch.Tensor] = []
    slot_indices: list[torch.Tensor] = []
    for frame_index in range(int(confidence.shape[0])):
        candidates = torch.nonzero(
            confidence[frame_index] >= float(threshold), as_tuple=False
        ).flatten()
        if candidates.numel() == 0:
            continue
        order = torch.argsort(
            confidence[frame_index, candidates], descending=True
        )
        selected = candidates[order[: int(max_sources_per_frame)]]
        frame_indices.append(
            torch.full_like(selected, frame_index, dtype=torch.long)
        )
        slot_indices.append(selected)
    if not frame_indices:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    return torch.cat(frame_indices), torch.cat(slot_indices)


def write_prediction_archive(
    chunks: Iterable[PredictionChunk],
    output_directory: str | Path,
    *,
    recording: str | Path,
    model_name: str,
    checkpoint_sha256: str,
    class_feature_encoder: str,
    recording_domain: str,
    threshold: float = 0.05,
    max_sources_per_frame: int = 4,
    sample_rate: int = 48_000,
    segment_duration: float = 2.0,
    output_fps: int = 10,
) -> Path:
    """Write selected refined maps losslessly in independent compressed chunks."""

    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be finite and between zero and one")
    if max_sources_per_frame <= 0:
        raise ValueError("max_sources_per_frame must be positive")
    if sample_rate <= 0 or output_fps <= 0 or not math.isfinite(float(segment_duration)) or segment_duration <= 0:
        raise ValueError("sample rate, segment duration, and output fps must be positive")
    if class_feature_encoder not in {"passt", "audiomae"}:
        raise ValueError("class_feature_encoder must be 'passt' or 'audiomae'")
    if recording_domain not in {"sony", "tau"}:
        raise ValueError("recording_domain must be 'sony' or 'tau'")
    digest = str(checkpoint_sha256).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal digest")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        chunk_directory = temporary / "chunks"
        chunk_directory.mkdir()
        records = []
        total_frames = 0
        total_predictions = 0
        for chunk_index, chunk in enumerate(chunks):
            maps = chunk.refined_maps
            class_ids_all = chunk.slot_class_ids
            confidence_all = chunk.slot_confidence
            if maps.ndim != 4 or tuple(maps.shape[-2:]) != (180, 360):
                raise ValueError(
                    "refined_maps must have shape [frames,slots,180,360]"
                )
            expected = tuple(maps.shape[:2])
            if tuple(class_ids_all.shape) != expected or tuple(confidence_all.shape) != expected:
                raise ValueError(
                    "refined_maps, slot_class_ids, and slot_confidence must share "
                    "their frame and slot dimensions"
                )
            if int(chunk.frame_start) != total_frames:
                raise ValueError("prediction chunks must be contiguous and begin at frame zero")
            if not torch.isfinite(maps).all() or not torch.isfinite(confidence_all).all():
                raise ValueError("prediction chunks must contain finite map and confidence values")
            if maps.numel() and (float(maps.min()) < 0.0 or float(maps.max()) > 1.0):
                raise ValueError("refined map values must lie between zero and one")
            if confidence_all.numel() and (
                float(confidence_all.min()) < 0.0 or float(confidence_all.max()) > 1.0
            ):
                raise ValueError("slot confidence values must lie between zero and one")
            if class_ids_all.numel() and (
                int(class_ids_all.min()) < 0
                or int(class_ids_all.max()) >= len(DCASE2026_CLASS_NAMES)
            ):
                raise ValueError("slot class IDs must use the public zero-based taxonomy")

            local_frames, slots = _selected_slots(
                confidence_all,
                threshold=threshold,
                max_sources_per_frame=max_sources_per_frame,
            )
            frame_indices = local_frames + int(chunk.frame_start)
            selected_maps = maps[local_frames, slots]
            class_ids = class_ids_all[local_frames, slots]
            confidence = confidence_all[local_frames, slots]
            filename = f"chunk_{chunk_index:06d}.npz"
            path = chunk_directory / filename
            np.savez_compressed(
                path,
                frame_index=frame_indices.numpy().astype(np.int32, copy=False),
                slot_index=slots.numpy().astype(np.int16, copy=False),
                class_id=class_ids.numpy().astype(np.int16, copy=False),
                confidence=confidence.numpy().astype(np.float32, copy=False),
                refined_map=selected_maps.numpy().astype(np.float32, copy=False),
            )
            predictions = int(local_frames.numel())
            records.append(
                {
                    "file": f"chunks/{filename}",
                    "frame_start": int(chunk.frame_start),
                    "frame_stop": int(chunk.frame_stop),
                    "predictions": predictions,
                }
            )
            total_frames = int(chunk.frame_stop)
            total_predictions += predictions
        manifest = {
            "schema": "said-prediction-archive-v1",
            "recording": str(Path(recording)),
            "model": str(model_name),
            "checkpoint_sha256": digest,
            "class_feature_encoder": class_feature_encoder,
            "recording_domain": recording_domain,
            "sample_rate": int(sample_rate),
            "segment_duration_seconds": float(segment_duration),
            "output_fps": int(output_fps),
            "frame_time_seconds": "frame_index / output_fps",
            "last_frame_rule": "frame start is earlier than recording duration",
            "map_height": 180,
            "map_width": 360,
            "map_dtype": "float32",
            "map_quantization": "none",
            "class_id_base": 0,
            "class_names": list(DCASE2026_CLASS_NAMES),
            "selection": {
                "confidence_threshold": float(threshold),
                "max_sources_per_frame": int(max_sources_per_frame),
            },
            "frames": total_frames,
            "predictions": total_predictions,
            "chunks": records,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest, indent=2, ensure_ascii=False, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "manifest.json"


def write_class_agnostic_prediction_archive(
    chunks: Iterable[ClassAgnosticPredictionChunk],
    output_directory: str | Path,
    *,
    recording: str | Path,
    model_name: str,
    checkpoint_sha256: str,
    sample_rate: int = 48_000,
    segment_duration: float = 2.0,
    output_fps: int = 100,
) -> Path:
    """Write dense class-agnostic panoramic maps in compressed chunks."""

    if (
        sample_rate <= 0
        or output_fps <= 0
        or not math.isfinite(float(segment_duration))
        or segment_duration <= 0
    ):
        raise ValueError(
            "sample rate, segment duration, and output fps must be positive"
        )
    digest = str(checkpoint_sha256).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(
            "checkpoint_sha256 must be a 64-character hexadecimal digest"
        )
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        chunk_directory = temporary / "chunks"
        chunk_directory.mkdir()
        records = []
        total_frames = 0
        for chunk_index, chunk in enumerate(chunks):
            maps = chunk.maps
            if maps.ndim != 3 or tuple(maps.shape[-2:]) != (180, 360):
                raise ValueError(
                    "class-agnostic maps must have shape [frames,180,360]"
                )
            if int(chunk.frame_start) != total_frames:
                raise ValueError(
                    "prediction chunks must be contiguous and begin at frame zero"
                )
            if not torch.isfinite(maps).all():
                raise ValueError("class-agnostic maps must contain finite values")
            if maps.numel() and (
                float(maps.min()) < 0.0 or float(maps.max()) > 1.0
            ):
                raise ValueError(
                    "class-agnostic map values must lie between zero and one"
                )
            filename = f"chunk_{chunk_index:06d}.npz"
            np.savez_compressed(
                chunk_directory / filename,
                map=maps.numpy().astype(np.float32, copy=False),
            )
            records.append(
                {
                    "file": f"chunks/{filename}",
                    "frame_start": int(chunk.frame_start),
                    "frame_stop": int(chunk.frame_stop),
                    "maps": int(maps.shape[0]),
                }
            )
            total_frames = int(chunk.frame_stop)
        manifest = {
            "schema": "audio2sph-prediction-archive-v1",
            "recording": str(Path(recording)),
            "model": str(model_name),
            "checkpoint_sha256": digest,
            "sample_rate": int(sample_rate),
            "segment_duration_seconds": float(segment_duration),
            "output_fps": int(output_fps),
            "frame_time_seconds": "frame_index / output_fps",
            "last_frame_rule": "frame start is earlier than recording duration",
            "map_height": 180,
            "map_width": 360,
            "map_dtype": "float32",
            "map_quantization": "none",
            "class_agnostic": True,
            "frames": total_frames,
            "chunks": records,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest, indent=2, ensure_ascii=False, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "manifest.json"


def _write_class_agnostic_json(
    frames: Iterable[tuple[int, torch.Tensor | np.ndarray]],
    output_file: str | Path,
    *,
    metadata: dict,
    relative_map_threshold: float,
) -> Path:
    """Atomically stream class-agnostic spherical maps to readable JSON."""

    from ..evaluation.output2dcase import dense_map_to_dcase_points

    if (
        not math.isfinite(float(relative_map_threshold))
        or not 0.0 <= float(relative_map_threshold) <= 1.0
    ):
        raise ValueError(
            "relative_map_threshold must be finite and between zero and one"
        )
    output = Path(output_file)
    if output.exists():
        raise FileExistsError(f"prediction file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    frame_count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            header = json.dumps(
                metadata,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            stream.write(header[:-1] + ',"annotations":[\n')
            for frame_index, acoustic_map in frames:
                if int(frame_index) != frame_count:
                    raise ValueError(
                        "class-agnostic frames must be contiguous and begin at zero"
                    )
                points = dense_map_to_dcase_points(
                    acoustic_map,
                    relative_threshold=relative_map_threshold,
                )
                annotation = {
                    "metadata_frame_index": frame_count,
                    "segmentation": [points] if points else [],
                }
                if frame_count:
                    stream.write(",\n")
                json.dump(
                    annotation,
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                frame_count += 1
            stream.write(f'\n],"frame_count":{frame_count}}}\n')
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _class_agnostic_json_metadata(
    *,
    recording: str | Path,
    model_name: str,
    checkpoint_sha256: str,
    sample_rate: int,
    segment_duration: float,
    output_fps: int,
    relative_map_threshold: float,
) -> dict:
    digest = str(checkpoint_sha256).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(
            "checkpoint_sha256 must be a 64-character hexadecimal digest"
        )
    if (
        sample_rate <= 0
        or output_fps <= 0
        or not math.isfinite(float(segment_duration))
        or segment_duration <= 0
    ):
        raise ValueError(
            "sample rate, segment duration, and output fps must be positive"
        )
    return {
        "schema": "audio2sph-prediction-json-v1",
        "recording": str(Path(recording)),
        "model": str(model_name),
        "checkpoint_sha256": digest,
        "sample_rate": int(sample_rate),
        "segment_duration_seconds": float(segment_duration),
        "output_fps": int(output_fps),
        "frame_time_seconds": "metadata_frame_index / output_fps",
        "last_frame_rule": "frame start is earlier than recording duration",
        "map_height": 180,
        "map_width": 360,
        "class_agnostic": True,
        "coordinate_system": {
            "x": "DCASE equirectangular column in [0,359]",
            "y": "DCASE equirectangular row in [0,179]",
        },
        "relative_map_threshold": float(relative_map_threshold),
        "map_encoding": "points at or above relative_map_threshold times frame peak",
    }


def write_class_agnostic_prediction_json(
    chunks: Iterable[ClassAgnosticPredictionChunk],
    output_file: str | Path,
    *,
    recording: str | Path,
    model_name: str,
    checkpoint_sha256: str,
    sample_rate: int = 48_000,
    segment_duration: float = 2.0,
    output_fps: int = 100,
    relative_map_threshold: float = 0.10,
) -> Path:
    """Write Audio2Sph predictions as one class-agnostic JSON file."""

    metadata = _class_agnostic_json_metadata(
        recording=recording,
        model_name=model_name,
        checkpoint_sha256=checkpoint_sha256,
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        output_fps=output_fps,
        relative_map_threshold=relative_map_threshold,
    )

    def frames() -> Iterable[tuple[int, torch.Tensor]]:
        expected_frame = 0
        for chunk in chunks:
            maps = chunk.maps
            if maps.ndim != 3 or tuple(maps.shape[-2:]) != (180, 360):
                raise ValueError(
                    "class-agnostic maps must have shape [frames,180,360]"
                )
            if int(chunk.frame_start) != expected_frame:
                raise ValueError(
                    "prediction chunks must be contiguous and begin at frame zero"
                )
            for local_frame in range(int(maps.shape[0])):
                yield expected_frame + local_frame, maps[local_frame]
            expected_frame = int(chunk.frame_stop)

    return _write_class_agnostic_json(
        frames(),
        output_file,
        metadata=metadata,
        relative_map_threshold=relative_map_threshold,
    )


def write_class_agnostic_archive_json(
    manifest_path: str | Path,
    output_file: str | Path,
    *,
    relative_map_threshold: float = 0.10,
) -> Path:
    """Convert a lossless Audio2Sph archive to one class-agnostic JSON file."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prediction manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "audio2sph-prediction-archive-v1":
        raise ValueError("class-agnostic JSON export requires an Audio2Sph archive")
    metadata = _class_agnostic_json_metadata(
        recording=str(manifest.get("recording", "")),
        model_name=str(manifest.get("model", "")),
        checkpoint_sha256=str(manifest.get("checkpoint_sha256", "")),
        sample_rate=int(manifest.get("sample_rate", 0)),
        segment_duration=float(manifest.get("segment_duration_seconds", 0.0)),
        output_fps=int(manifest.get("output_fps", 0)),
        relative_map_threshold=relative_map_threshold,
    )
    archive_directory = manifest_path.parent

    def frames() -> Iterable[tuple[int, np.ndarray]]:
        covered_frames = 0
        for record in manifest.get("chunks", []):
            frame_start = int(record["frame_start"])
            frame_stop = int(record["frame_stop"])
            if frame_start != covered_frames or frame_stop < frame_start:
                raise ValueError("prediction chunks must be ordered and contiguous")
            with np.load(
                archive_directory / str(record["file"]), allow_pickle=False
            ) as chunk:
                maps = np.asarray(chunk["map"], dtype=np.float32)
                if tuple(maps.shape) != (frame_stop - frame_start, 180, 360):
                    raise ValueError(
                        "class-agnostic prediction chunk has an invalid shape"
                    )
                for local_frame in range(int(maps.shape[0])):
                    yield frame_start + local_frame, maps[local_frame]
            covered_frames = frame_stop
        if covered_frames != int(manifest.get("frames", -1)):
            raise ValueError(
                "prediction archive frame count does not match its manifest"
            )

    return _write_class_agnostic_json(
        frames(),
        output_file,
        metadata=metadata,
        relative_map_threshold=relative_map_threshold,
    )
