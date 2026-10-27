"""DCASE2026 training segments and development-test recordings."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

import json
import math
import operator
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from ..models import RecordingDomain
from .targets import FrameAlignment, SourceMapTargets


DCASE_ROTATION_VIEWS = (
    ("rot0", 0, (6, 10, 26, 22)),
    ("rot90", 90, (28, 8, 24, 12)),
    ("rot180", 180, (22, 26, 10, 6)),
    ("rot270", 270, (12, 24, 8, 28)),
)

DCASE2026_CLASS_NAMES: tuple[str, ...] = (
    "Female speech",
    "Male speech",
    "Clapping",
    "Telephone",
    "Laughter",
    "Domestic sounds",
    "Walk/footsteps",
    "Door open/close",
    "Music",
    "Musical instr.",
    "Water tap/shower",
    "Bell",
    "Knock",
)

DCASE2026_CLASS_TO_ID = {
    name: class_id for class_id, name in enumerate(DCASE2026_CLASS_NAMES)
}


def class_name(class_id: int) -> str:
    """Return the class name for a zero-based DCASE2026 class identifier."""

    if isinstance(class_id, bool):
        raise ValueError(
            f"class_id must be an integer in [0,{len(DCASE2026_CLASS_NAMES) - 1}]"
        )
    try:
        normalized = operator.index(class_id)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"class_id must be an integer in [0,{len(DCASE2026_CLASS_NAMES) - 1}]"
        ) from error
    if normalized < 0 or normalized >= len(DCASE2026_CLASS_NAMES):
        raise ValueError(
            f"class_id must be an integer in [0,{len(DCASE2026_CLASS_NAMES) - 1}]"
        )
    return DCASE2026_CLASS_NAMES[normalized]


def _linear_resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if int(source_rate) == int(target_rate):
        return np.asarray(audio, dtype=np.float32)
    source_samples = int(audio.shape[-1])
    if source_samples <= 1:
        return np.zeros((audio.shape[0], 0), dtype=np.float32)
    target_samples = int(round(source_samples * target_rate / source_rate))
    source_time = np.linspace(0.0, 1.0, source_samples, endpoint=False)
    target_time = np.linspace(0.0, 1.0, target_samples, endpoint=False)
    return np.stack(
        [
            np.interp(target_time, source_time, channel.astype(np.float64)).astype(
                np.float32
            )
            for channel in audio
        ],
        axis=0,
    )


def _split_subsets(split: str) -> tuple[str, ...]:
    choices = {
        "dev_train": ("dev-train-sony", "dev-train-tau"),
        "dev_test": ("dev-test-sony", "dev-test-tau"),
    }
    try:
        return choices[str(split)]
    except KeyError as error:
        raise ValueError(
            "prepared DCASE split must be 'dev_train' or 'dev_test'"
        ) from error


class DCASE2026TrainingDataset(Dataset[dict[str, Any]]):
    """Two-second DCASE segments with the paper's four rotation views."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "dev_train",
        sample_rate: int = 48_000,
        segment_duration: float = 2.0,
        label_fps: int = 10,
        maximum_sources: int = 6,
        rotation_views: bool = True,
        normalize_maps: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.sample_rate = int(sample_rate)
        self.segment_duration = float(segment_duration)
        self.label_fps = int(label_fps)
        self.maximum_sources = int(maximum_sources)
        self.normalize_maps = bool(normalize_maps)
        self.segment_samples = int(round(self.segment_duration * self.sample_rate))
        label_frames = self.segment_duration * self.label_fps
        self.target_frames = int(round(label_frames))
        if self.sample_rate <= 0 or self.segment_samples <= 0:
            raise ValueError("sample rate and segment duration must be positive")
        if not math.isclose(label_frames, self.target_frames, abs_tol=1.0e-8):
            raise ValueError("segment_duration * label_fps must be integral")
        if self.maximum_sources <= 0:
            raise ValueError("maximum_sources must be positive")
        self.views = DCASE_ROTATION_VIEWS if rotation_views else DCASE_ROTATION_VIEWS[:1]
        self.recordings: list[dict[str, Any]] = []
        self.items: list[tuple[int, int, int]] = []
        # Keep a bounded sparse working set in each DataLoader worker.  Dense
        # 180x360 maps are materialized only for the requested two-second
        # segment, so memory cannot grow with the full annotation corpus.
        self._annotation_cache_limit = 2
        self._annotation_cache: OrderedDict[
            int, dict[int, list[dict[str, Any]]]
        ] = OrderedDict()
        self._class_metadata_cache: dict[
            int, dict[int, tuple[int, ...]]
        ] = {}

        for subset in _split_subsets(split):
            domain = (
                RecordingDomain.SONY
                if subset.endswith("sony")
                else RecordingDomain.TAU
            )
            audio_directory = self.root / "eigen_dev" / subset
            label_directory = self.root / "labels_dev" / subset
            if not audio_directory.is_dir() or not label_directory.is_dir():
                raise FileNotFoundError(
                    f"expected DCASE directories {audio_directory} and {label_directory}"
                )
            audio_files = {
                path.stem: path for path in sorted(audio_directory.glob("*.wav"))
            }
            label_files = {
                path.name.removesuffix("_std.json"): path
                for path in sorted(label_directory.glob("*_std.json"))
            }
            if set(audio_files) != set(label_files):
                raise ValueError(f"DCASE {subset} audio and label stems differ")
            for name in sorted(audio_files):
                information = sf.info(str(audio_files[name]))
                if int(information.channels) != 32:
                    raise ValueError(
                        f"DCASE training audio must contain 32 Eigenmike channels: {audio_files[name]}"
                    )
                recording_index = len(self.recordings)
                self.recordings.append(
                    {
                        "name": name,
                        "subset": subset,
                        "domain": domain,
                        "audio_path": audio_files[name],
                        "label_path": label_files[name],
                        "source_rate": int(information.samplerate),
                        "source_samples": int(information.frames),
                    }
                )
                target_samples = int(
                    round(
                        int(information.frames)
                        * self.sample_rate
                        / int(information.samplerate)
                    )
                )
                for start_sample in range(0, target_samples, self.segment_samples):
                    if start_sample + self.segment_samples > target_samples:
                        continue
                    for view_index in range(len(self.views)):
                        self.items.append(
                            (recording_index, start_sample, view_index)
                        )
        if not self.items:
            raise RuntimeError(f"no complete DCASE training segments found under {self.root}")

    def __len__(self) -> int:
        return len(self.items)

    def _annotations(self, recording_index: int) -> dict[int, list[dict[str, Any]]]:
        cached = self._annotation_cache.get(recording_index)
        if cached is not None:
            self._annotation_cache.move_to_end(recording_index)
            return cached
        label_path = self.recordings[recording_index]["label_path"]
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read DCASE labels: {label_path}") from error
        annotations = payload.get("annotations") if isinstance(payload, dict) else None
        if not isinstance(annotations, list):
            raise ValueError(f"DCASE labels contain no annotations list: {label_path}")
        by_frame: dict[int, list[dict[str, Any]]] = {}
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError(f"DCASE annotation must be an object: {label_path}")
            frame = annotation.get("metadata_frame_index")
            instance = annotation.get("instance_id")
            category = annotation.get("category_id")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (frame, instance, category)):
                raise ValueError(f"DCASE frame, instance, and category IDs must be integers: {label_path}")
            if frame < 0 or not 0 <= category < 13:
                raise ValueError(f"DCASE annotation index is outside the public schema: {label_path}")
            sparse_indices: list[np.ndarray] = []
            sparse_values: list[np.ndarray] = []
            for component in annotation.get("segmentation", []):
                array = np.asarray(component, dtype=np.float64)
                if array.size == 0:
                    continue
                if array.ndim != 2 or array.shape[1] < 3 or not np.isfinite(array[:, :3]).all():
                    raise ValueError(f"invalid DCASE segmentation points: {label_path}")
                dcase_x = np.rint(array[:, 0]).astype(np.int64).clip(0, 359)
                y = np.rint(array[:, 1]).astype(np.int64).clip(0, 179)
                spherical_x = (179 - dcase_x) % 360
                energy = np.maximum(array[:, 2], 0.0).astype(np.float32)
                sparse_indices.append((y * 360 + spherical_x).astype(np.int32))
                sparse_values.append(energy)
            if not sparse_indices:
                continue
            flat_indices = np.concatenate(sparse_indices)
            values = np.concatenate(sparse_values)
            order = np.argsort(flat_indices, kind="stable")
            ordered_indices = flat_indices[order]
            ordered_values = values[order]
            unique_indices, first = np.unique(
                ordered_indices, return_index=True
            )
            values = np.maximum.reduceat(ordered_values, first).astype(
                np.float32, copy=False
            )
            peak = float(values.max(initial=0.0))
            if peak <= 0.0:
                continue
            if self.normalize_maps:
                values = values / peak
            by_frame.setdefault(int(frame), []).append(
                {
                    "instance_id": int(instance),
                    "class_id": int(category),
                    "flat_indices": unique_indices.astype(np.int32, copy=False),
                    "values": values.astype(np.float32, copy=False),
                    "peak": peak,
                }
            )
        self._annotation_cache[recording_index] = by_frame
        self._annotation_cache.move_to_end(recording_index)
        while len(self._annotation_cache) > self._annotation_cache_limit:
            self._annotation_cache.popitem(last=False)
        return by_frame

    def _load_audio(
        self, recording: dict[str, Any], start_sample: int, capsules: tuple[int, ...]
    ) -> Tensor:
        source_rate = int(recording["source_rate"])
        ratio = self.sample_rate / source_rate
        source_start = int(math.floor(start_sample / ratio))
        source_stop = int(
            math.ceil((start_sample + self.segment_samples) / ratio)
        )
        array, _ = sf.read(
            str(recording["audio_path"]),
            start=source_start,
            frames=max(1, source_stop - source_start),
            always_2d=True,
            dtype="float32",
        )
        selected = array[:, [index - 1 for index in capsules]].T
        selected = _linear_resample(selected, source_rate, self.sample_rate)
        resampled_start = int(round(source_start * ratio))
        offset = max(0, int(start_sample) - resampled_start)
        selected = selected[:, offset : offset + self.segment_samples]
        if selected.shape[1] < self.segment_samples:
            selected = np.pad(
                selected,
                ((0, 0), (0, self.segment_samples - selected.shape[1])),
            )
        return torch.from_numpy(np.asarray(selected, dtype=np.float32)).contiguous()

    def _targets(
        self,
        recording_index: int,
        start_sample: int,
        yaw_degrees: int,
    ) -> dict[str, Tensor | str]:
        start_position = start_sample * self.label_fps / self.sample_rate
        start_frame = int(round(start_position))
        if not math.isclose(start_position, start_frame, abs_tol=1.0e-8):
            raise ValueError("DCASE segments must begin on the official 10 Hz grid")
        frames = range(start_frame, start_frame + self.target_frames)
        annotations = self._annotations(recording_index)
        frame_annotations = [annotations.get(frame, []) for frame in frames]
        ownership: dict[int, tuple[int, float]] = {}
        for values in frame_annotations:
            for annotation in values:
                instance = int(annotation["instance_id"])
                count, energy = ownership.get(instance, (0, 0.0))
                ownership[instance] = (
                    count + 1,
                    energy + float(annotation["peak"]),
                )
        selected_instances = [
            instance
            for instance, _ in sorted(
                ownership.items(),
                key=lambda item: (-item[1][0], -item[1][1], item[0]),
            )[: self.maximum_sources]
        ]
        slot = {instance: index for index, instance in enumerate(selected_instances)}
        valid = torch.zeros(
            self.target_frames, self.maximum_sources, dtype=torch.bool
        )
        classes = torch.full(
            (self.target_frames, self.maximum_sources), -1, dtype=torch.long
        )
        refined = torch.zeros(
            self.target_frames, self.maximum_sources, 180, 360
        )
        horizontal_shift = int(round((yaw_degrees % 360) * 360 / 360)) % 360
        for frame_index, values in enumerate(frame_annotations):
            for annotation in values:
                source_slot = slot.get(int(annotation["instance_id"]))
                if source_slot is None:
                    continue
                valid[frame_index, source_slot] = True
                classes[frame_index, source_slot] = int(annotation["class_id"])
                indices = torch.from_numpy(annotation["flat_indices"]).long()
                row = torch.div(indices, 360, rounding_mode="floor")
                column = torch.remainder(indices, 360)
                rotated_indices = row * 360 + torch.remainder(
                    column - horizontal_shift, 360
                )
                refined[frame_index, source_slot].view(-1).scatter_reduce_(
                    0,
                    rotated_indices,
                    torch.from_numpy(annotation["values"]),
                    reduce="amax",
                    include_self=True,
                )
        maps = F.interpolate(
            refined.reshape(-1, 1, 180, 360),
            size=(45, 90),
            mode="bilinear",
            align_corners=False,
        ).reshape(self.target_frames, self.maximum_sources, 45, 90)
        return {
            "valid_sources": valid,
            "class_ids": classes,
            "map_targets": maps,
            "refined_map_targets": refined,
            "frame_alignment": FrameAlignment.DCASE_HALF_OPEN.value,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        recording_index, start_sample, view_index = self.items[int(index)]
        recording = self.recordings[recording_index]
        view_name, yaw_degrees, capsules = self.views[view_index]
        return {
            "audio": self._load_audio(recording, start_sample, capsules),
            "targets": self._targets(
                recording_index, start_sample, yaw_degrees
            ),
            "recording_domain": recording["domain"].value,
            "recording": recording["name"],
            "subset": recording["subset"],
            "rotation_view": view_name,
            "start_sample": int(start_sample),
        }

    def class_balance_metadata(self) -> tuple[dict[str, Any], ...]:
        """Return per-segment classes without materializing dense maps.

        Official label files can be hundreds of megabytes because every
        annotation stores a dense list of map points. Class-balanced sampling
        needs only the frame and category fields, so this scanner reads those
        small integer headers directly from a bounded byte buffer.
        """

        header = re.compile(
            rb'"metadata_frame_index"\s*:\s*(-?\d+)\s*,\s*'
            rb'"instance_id"\s*:\s*(-?\d+)\s*,\s*'
            rb'"category_id"\s*:\s*(-?\d+)'
        )

        def recording_classes(recording_index: int) -> dict[int, tuple[int, ...]]:
            cached = self._class_metadata_cache.get(recording_index)
            if cached is not None:
                return cached
            path = self.recordings[recording_index]["label_path"]
            classes: dict[int, set[int]] = {}
            carry = b""
            read_offset = 0
            last_match_offset = -1
            with path.open("rb") as stream:
                while chunk := stream.read(4 * 1024 * 1024):
                    buffer = carry + chunk
                    base_offset = read_offset - len(carry)
                    for match in header.finditer(buffer):
                        absolute_offset = base_offset + match.start()
                        if absolute_offset <= last_match_offset:
                            continue
                        frame = int(match.group(1))
                        category = int(match.group(3))
                        if frame < 0 or not 0 <= category < 13:
                            raise ValueError(
                                f"DCASE annotation index is outside the public schema: {path}"
                            )
                        classes.setdefault(frame, set()).add(category)
                        last_match_offset = absolute_offset
                    read_offset += len(chunk)
                    carry = buffer[-1024:]
            result = {
                frame: tuple(sorted(values))
                for frame, values in classes.items()
            }
            self._class_metadata_cache[recording_index] = result
            return result

        metadata: list[dict[str, Any]] = []
        for index, (recording_index, start_sample, _) in enumerate(self.items):
            start_frame = int(round(start_sample * self.label_fps / self.sample_rate))
            classes_by_frame = recording_classes(recording_index)
            class_ids = {
                int(class_id)
                for frame in range(start_frame, start_frame + self.target_frames)
                for class_id in classes_by_frame.get(frame, ())
            }
            metadata.append(
                {"index": index, "class_ids": tuple(sorted(class_ids))}
            )
        return tuple(metadata)


def collate_source_map_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate prepared or rendered examples into the public training schema."""

    if not items:
        raise ValueError("cannot collate an empty training batch")
    alignments = {item["targets"]["frame_alignment"] for item in items}
    if len(alignments) != 1:
        raise ValueError("all items in a batch must use one frame-alignment contract")
    targets = SourceMapTargets(
        valid_sources=torch.stack(
            [item["targets"]["valid_sources"] for item in items]
        ),
        class_ids=torch.stack([item["targets"]["class_ids"] for item in items]),
        map_targets=torch.stack(
            [item["targets"]["map_targets"] for item in items]
        ),
        refined_map_targets=torch.stack(
            [item["targets"]["refined_map_targets"] for item in items]
        ),
        frame_alignment=FrameAlignment(next(iter(alignments))),
    )
    domains = [item["recording_domain"] for item in items]
    if all(domain is None for domain in domains):
        recording_domain: list[str] | None = None
    elif any(domain is None for domain in domains):
        raise ValueError("a batch cannot mix defined and undefined recording domains")
    else:
        recording_domain = domains
    return {
        "audio": torch.stack([item["audio"] for item in items]),
        "targets": targets,
        "recording_domain": recording_domain,
        "metadata": [
            {
                key: item[key]
                for key in ("recording", "subset", "rotation_view", "start_sample")
                if key in item
            }
            for item in items
        ],
    }


@dataclass(frozen=True)
class DCASESequence:
    """One labeled development-test recording."""

    subset: str
    name: str
    audio_path: Path
    label_path: Path
    recording_domain: RecordingDomain
    frame_count: int


_FRAME_FIELD = re.compile(
    rb'"metadata_frame_index"\s*:\s*([^,}\]\s]+)'
)
_JSON_INTEGER = re.compile(rb"-?(?:0|[1-9]\d*)")


def _frame_count(label_path: Path) -> int:
    """Read frame indices without materializing multi-gigabyte map payloads."""

    maximum: int | None = None
    carry = b""
    read_offset = 0
    last_match_offset = -1
    try:
        with label_path.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                buffer = carry + chunk
                base_offset = read_offset - len(carry)
                for match in _FRAME_FIELD.finditer(buffer):
                    absolute_offset = base_offset + match.start()
                    if absolute_offset <= last_match_offset:
                        continue
                    token = match.group(1)
                    if _JSON_INTEGER.fullmatch(token) is None:
                        raise ValueError(
                            f"invalid metadata_frame_index in {label_path}"
                        )
                    frame_index = int(token)
                    if frame_index < 0:
                        raise ValueError(
                            f"negative metadata_frame_index in {label_path}"
                        )
                    maximum = (
                        frame_index
                        if maximum is None
                        else max(maximum, frame_index)
                    )
                    last_match_offset = absolute_offset
                read_offset += len(chunk)
                carry = buffer[-1024:]
    except OSError as error:
        raise ValueError(f"cannot read DCASE label JSON: {label_path}") from error
    if maximum is None:
        raise ValueError(
            f"DCASE label JSON contains no indexed annotations: {label_path}"
        )
    return maximum + 1


def discover_development_test(
    dataset_root: str | Path,
    *,
    require_paper_split: bool = True,
) -> tuple[DCASESequence, ...]:
    """Discover the Sony and TAU development-test recordings used in the paper."""

    root = Path(dataset_root).expanduser().resolve()
    subsets = (
        ("dev-test-sony", RecordingDomain.SONY),
        ("dev-test-tau", RecordingDomain.TAU),
    )
    sequences: list[DCASESequence] = []
    discovered_counts: dict[str, int] = {}
    for subset, domain in subsets:
        audio_directory = root / "eigen_dev" / subset
        label_directory = root / "labels_dev" / subset
        if not audio_directory.is_dir() or not label_directory.is_dir():
            raise FileNotFoundError(
                f"expected DCASE directories {audio_directory} and {label_directory}"
            )
        audio_files = {
            path.stem: path for path in sorted(audio_directory.glob("*.wav"))
        }
        label_files: dict[str, Path] = {}
        for path in sorted(label_directory.glob("*_std.json")):
            name = path.name.removesuffix("_std.json")
            if name in label_files:
                raise ValueError(
                    f"duplicate DCASE labels for {subset}/{name}: "
                    f"{label_files[name]} and {path}"
                )
            label_files[name] = path
        missing_labels = sorted(set(audio_files) - set(label_files))
        missing_audio = sorted(set(label_files) - set(audio_files))
        if missing_labels or missing_audio:
            raise ValueError(
                f"DCASE {subset} audio/label mismatch: "
                f"missing_labels={missing_labels}, missing_audio={missing_audio}"
            )
        for name in sorted(audio_files):
            label_path = label_files[name]
            sequences.append(
                DCASESequence(
                    subset=subset,
                    name=name,
                    audio_path=audio_files[name],
                    label_path=label_path,
                    recording_domain=domain,
                    frame_count=_frame_count(label_path),
                )
            )
        discovered_counts[subset] = len(audio_files)
    if require_paper_split:
        expected_counts = {"dev-test-sony": 30, "dev-test-tau": 48}
        if discovered_counts != expected_counts:
            raise ValueError(
                "the paper development-test split contains 30 Sony and 48 TAU "
                f"recordings; discovered {discovered_counts} under {root}"
            )
    return tuple(sequences)
