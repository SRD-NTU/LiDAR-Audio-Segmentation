"""DCASE2026 scenes used by the four reproducible SAID demonstrations."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from dataclasses import dataclass
import gzip
from importlib import resources
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch import Tensor

from ..models import RecordingDomain
from .load_audio import load_eigenmike_audio


@dataclass(frozen=True)
class DemoScene:
    """One fixed paper-demo excerpt in the official DCASE2026 layout."""

    output_name: str
    subset: str
    recording: str
    start_seconds: int
    duration_seconds: int
    recording_domain: RecordingDomain
    legend_class_ids: tuple[int, ...]

    def paths(self) -> tuple[Path, Path, Path]:
        directory = resources.files("said").joinpath(
            "demo_data", self.output_name.removesuffix(".mp4")
        )
        audio = Path(str(directory.joinpath("recording.wav")))
        labels = Path(str(directory.joinpath("ground_truth.json.gz")))
        video = Path(str(directory.joinpath("panorama.mp4")))
        for kind, path in (("audio", audio), ("labels", labels), ("video", video)):
            if not path.is_file():
                raise FileNotFoundError(
                    f"DCASE2026 demo {kind} is missing: {path}"
                )
        return audio, labels, video


DEMO_SCENES: tuple[DemoScene, ...] = (
    DemoScene(
        output_name="said_demo_1.mp4",
        subset="dev-test-tau",
        recording="fold4_room8_mix003",
        start_seconds=150,
        duration_seconds=20,
        recording_domain=RecordingDomain.TAU,
        legend_class_ids=(0, 1, 8, 9),
    ),
    DemoScene(
        output_name="said_demo_2.mp4",
        subset="dev-test-tau",
        recording="fold4_room10_mix007",
        start_seconds=50,
        duration_seconds=20,
        recording_domain=RecordingDomain.TAU,
        legend_class_ids=(0, 1, 3, 4, 6, 7, 8, 9, 10),
    ),
    DemoScene(
        output_name="said_demo_3.mp4",
        subset="dev-test-sony",
        recording="fold4_room24_mix002",
        start_seconds=20,
        duration_seconds=20,
        recording_domain=RecordingDomain.SONY,
        legend_class_ids=(0, 1, 4, 5, 8, 11),
    ),
    DemoScene(
        output_name="said_demo_4.mp4",
        subset="dev-test-sony",
        recording="fold4_room23_mix003",
        start_seconds=0,
        duration_seconds=20,
        recording_domain=RecordingDomain.SONY,
        legend_class_ids=(0, 1, 4, 5, 6, 7, 8, 10),
    ),
)


def load_demo_audio(
    scene: DemoScene,
    source_audio: str | Path,
    *,
    sample_rate: int = 48_000,
) -> Tensor:
    """Load exactly one four-channel demo excerpt for model inference."""

    audio = load_eigenmike_audio(
        source_audio,
        sample_rate=int(sample_rate),
        offset_seconds=0.0,
        duration_seconds=float(scene.duration_seconds),
    )
    expected = int(sample_rate) * int(scene.duration_seconds)
    if int(audio.shape[-1]) < expected:
        raise ValueError(
            f"packaged demo audio is shorter than {scene.duration_seconds}s: "
            f"{source_audio}"
        )
    return audio[:, :expected].contiguous()


def write_demo_audio(audio: Tensor, path: str | Path, *, sample_rate: int = 48_000) -> Path:
    """Write the selected Eigenmike capsules for synchronized demo playback."""

    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"demo audio already exists: {destination}")
    if audio.ndim != 2 or int(audio.shape[0]) != 4:
        raise ValueError("demo audio must have shape [4,samples]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        destination,
        audio.detach().cpu().numpy().T,
        int(sample_rate),
        subtype="PCM_16",
    )
    return destination


def _annotation_map(annotation: dict[str, Any]) -> np.ndarray | None:
    dense = np.zeros((180, 360), dtype=np.float32)
    for component in annotation.get("segmentation", []):
        points = np.asarray(component, dtype=np.float64)
        if points.size == 0:
            continue
        if (
            points.ndim != 2
            or int(points.shape[1]) < 3
            or not np.isfinite(points[:, :3]).all()
        ):
            raise ValueError("DCASE2026 demo label contains invalid segmentation points")
        dcase_x = np.rint(points[:, 0]).astype(np.int64).clip(0, 359)
        row = np.rint(points[:, 1]).astype(np.int64).clip(0, 179)
        spherical_x = (179 - dcase_x) % 360
        energy = np.maximum(points[:, 2], 0.0).astype(np.float32)
        np.maximum.at(dense, (row, spherical_x), energy)
    peak = float(dense.max(initial=0.0))
    if peak <= 0.0:
        return None
    return dense / peak


def _load_demo_annotations(label_path: Path) -> list[dict[str, Any]]:
    try:
        if label_path.suffix == ".gz":
            with gzip.open(label_path, "rt", encoding="utf-8") as file:
                payload = json.load(file)
        else:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read DCASE2026 demo labels: {label_path}") from error
    annotations = payload.get("annotations") if isinstance(payload, dict) else None
    if not isinstance(annotations, list):
        raise ValueError(f"DCASE2026 labels contain no annotations list: {label_path}")
    if any(not isinstance(annotation, dict) for annotation in annotations):
        raise ValueError("DCASE2026 demo annotation must be an object")
    return annotations


def _annotations_by_excerpt_frame(
    scene: DemoScene,
    annotations: list[dict[str, Any]],
    *,
    output_fps: int,
) -> tuple[dict[int, list[dict[str, Any]]], int]:
    first_frame = int(scene.start_seconds) * int(output_fps)
    total_frames = int(scene.duration_seconds) * int(output_fps)
    last_frame = first_frame + total_frames
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        frame = annotation.get("metadata_frame_index")
        category = annotation.get("category_id")
        if (
            isinstance(frame, bool)
            or not isinstance(frame, int)
            or isinstance(category, bool)
            or not isinstance(category, int)
        ):
            raise ValueError("DCASE2026 demo frame and category IDs must be integers")
        if first_frame <= frame < last_frame:
            if not 0 <= category < 13:
                raise ValueError(f"DCASE2026 category ID is outside [0,12]: {category}")
            by_frame.setdefault(frame - first_frame, []).append(annotation)
    return by_frame, total_frames


def write_demo_ground_truth_archive(
    scene: DemoScene,
    label_path: str | Path,
    output_directory: str | Path,
    *,
    output_fps: int = 10,
    frames_per_chunk: int = 20,
) -> Path:
    """Convert one official label excerpt to the visualization archive schema."""

    label_path = Path(label_path).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Ground Truth archive already exists: {output}")
    if int(output_fps) <= 0 or int(frames_per_chunk) <= 0:
        raise ValueError("demo frame rate and chunk size must be positive")
    annotations = _load_demo_annotations(label_path)
    by_frame, total_frames = _annotations_by_excerpt_frame(
        scene, annotations, output_fps=int(output_fps)
    )

    output.mkdir(parents=True)
    chunk_directory = output / "chunks"
    chunk_directory.mkdir()
    records: list[dict[str, int | str]] = []
    total_predictions = 0
    for chunk_index, frame_start in enumerate(
        range(0, total_frames, int(frames_per_chunk))
    ):
        frame_stop = min(total_frames, frame_start + int(frames_per_chunk))
        frame_indices: list[int] = []
        slot_indices: list[int] = []
        class_ids: list[int] = []
        maps: list[np.ndarray] = []
        for local_frame in range(frame_start, frame_stop):
            slot = 0
            for annotation in by_frame.get(local_frame, []):
                dense = _annotation_map(annotation)
                if dense is None:
                    continue
                frame_indices.append(local_frame)
                slot_indices.append(slot)
                class_ids.append(int(annotation["category_id"]))
                maps.append(dense)
                slot += 1
        filename = f"chunk_{chunk_index:06d}.npz"
        np.savez_compressed(
            chunk_directory / filename,
            frame_index=np.asarray(frame_indices, dtype=np.int32),
            slot_index=np.asarray(slot_indices, dtype=np.int16),
            class_id=np.asarray(class_ids, dtype=np.int16),
            confidence=np.ones(len(frame_indices), dtype=np.float32),
            refined_map=(
                np.stack(maps).astype(np.float32, copy=False)
                if maps
                else np.empty((0, 180, 360), dtype=np.float32)
            ),
        )
        count = len(frame_indices)
        records.append(
            {
                "file": f"chunks/{filename}",
                "frame_start": frame_start,
                "frame_stop": frame_stop,
                "predictions": count,
            }
        )
        total_predictions += count

    manifest = {
        "schema": "said-prediction-archive-v1",
        "recording": scene.recording,
        "model": "DCASE2026 Ground Truth",
        "source_labels": str(label_path),
        "output_fps": int(output_fps),
        "map_height": 180,
        "map_width": 360,
        "map_dtype": "float32",
        "class_id_base": 0,
        "frames": total_frames,
        "predictions": total_predictions,
        "excerpt_start_seconds": int(scene.start_seconds),
        "excerpt_duration_seconds": int(scene.duration_seconds),
        "chunks": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def write_demo_class_agnostic_ground_truth_archive(
    scene: DemoScene,
    label_path: str | Path,
    output_directory: str | Path,
    *,
    output_fps: int = 10,
    frames_per_chunk: int = 20,
) -> Path:
    """Merge official per-source maps into the Audio2Sph target convention."""

    label_path = Path(label_path).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"class-agnostic Ground Truth archive already exists: {output}"
        )
    if int(output_fps) <= 0 or int(frames_per_chunk) <= 0:
        raise ValueError("demo frame rate and chunk size must be positive")
    annotations = _load_demo_annotations(label_path)
    by_frame, total_frames = _annotations_by_excerpt_frame(
        scene, annotations, output_fps=int(output_fps)
    )

    output.mkdir(parents=True)
    chunk_directory = output / "chunks"
    chunk_directory.mkdir()
    records: list[dict[str, int | str]] = []
    for chunk_index, frame_start in enumerate(
        range(0, total_frames, int(frames_per_chunk))
    ):
        frame_stop = min(total_frames, frame_start + int(frames_per_chunk))
        maps: list[np.ndarray] = []
        for local_frame in range(frame_start, frame_stop):
            source_maps = [
                dense
                for annotation in by_frame.get(local_frame, [])
                if (dense := _annotation_map(annotation)) is not None
            ]
            maps.append(
                np.maximum.reduce(source_maps).astype(np.float32, copy=False)
                if source_maps
                else np.zeros((180, 360), dtype=np.float32)
            )
        filename = f"chunk_{chunk_index:06d}.npz"
        np.savez_compressed(
            chunk_directory / filename,
            map=np.stack(maps).astype(np.float32, copy=False),
        )
        records.append(
            {
                "file": f"chunks/{filename}",
                "frame_start": frame_start,
                "frame_stop": frame_stop,
                "maps": len(maps),
            }
        )

    manifest = {
        "schema": "audio2sph-prediction-archive-v1",
        "recording": scene.recording,
        "model": "Class-agnostic Ground Truth",
        "source_labels": str(label_path),
        "output_fps": int(output_fps),
        "map_height": 180,
        "map_width": 360,
        "map_dtype": "float32",
        "class_agnostic": True,
        "source_merge": "pixelwise maximum",
        "frames": total_frames,
        "excerpt_start_seconds": int(scene.start_seconds),
        "excerpt_duration_seconds": int(scene.duration_seconds),
        "chunks": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path
