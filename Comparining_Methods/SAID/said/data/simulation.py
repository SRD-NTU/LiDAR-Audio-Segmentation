"""Online Scene Generation datasets for the published SAID training recipes."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

import math
import json
from collections import defaultdict
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .sourcebank import SourceBankRecord, load_sourcebank_manifest

if TYPE_CHECKING:
    from ..rendering import AcousticSceneRenderer


def _linear_resample(
    audio: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    """Resample one mono waveform without introducing an optional dependency."""

    value = np.asarray(audio, dtype=np.float32).reshape(-1)
    if int(source_rate) == int(target_rate):
        return value
    if value.size <= 1:
        return np.zeros(0, dtype=np.float32)
    target_samples = int(round(value.size * target_rate / source_rate))
    source_time = np.linspace(0.0, 1.0, value.size, endpoint=False)
    target_time = np.linspace(0.0, 1.0, target_samples, endpoint=False)
    return np.interp(target_time, source_time, value).astype(np.float32)


def _probability_table(
    raw: Mapping[Any, Any],
    *,
    allowed: Sequence[int],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tuple(int(value) for value in allowed), dtype=np.int64)
    normalized: dict[int, float] = {}
    for key, probability in raw.items():
        try:
            integer_key = int(key)
            numeric = float(probability)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must map integers to probabilities") from error
        if integer_key not in set(values.tolist()):
            raise ValueError(f"{field} contains unsupported key {integer_key}")
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{field} probabilities must be finite and non-negative")
        normalized[integer_key] = numeric
    if set(normalized) != set(values.tolist()):
        raise ValueError(f"{field} must define exactly {values.tolist()}")
    probabilities = np.asarray(
        [normalized[int(value)] for value in values], dtype=np.float64
    )
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError(f"{field} probabilities must have positive total mass")
    return values, probabilities / total


def validate_vctk_v080(root: str | Path) -> tuple[Path, ...]:
    """Validate a local VCTK v0.80 tree and return its WAV recordings.

    The corpus itself is not distributed with SAID.  The checks intentionally
    bind the paper recipe to the v0.80 release and its ODC-By 1.0 notice.
    """

    directory = Path(root).expanduser().resolve()
    readme = directory / "README"
    copying = directory / "COPYING"
    if not readme.is_file() or not copying.is_file():
        raise FileNotFoundError(
            "VCTK v0.80 requires README and COPYING at the configured root"
        )
    readme_text = readme.read_text(encoding="utf-8", errors="replace")
    license_text = copying.read_text(encoding="utf-8", errors="replace")
    if "Version 0.80" not in readme_text or "109 English" not in readme_text:
        raise ValueError("the configured speech corpus is not VCTK v0.80")
    if "ODC Attribution License" not in license_text:
        raise ValueError("VCTK v0.80 COPYING does not declare ODC-By 1.0")
    audio_root = directory / "wav48"
    paths = tuple(sorted(audio_root.glob("*/*.wav")))
    if not paths:
        raise RuntimeError(f"no VCTK v0.80 wav48 recordings found under {directory}")
    return paths


_VCTK_PREPARATION_SCHEMA = "said-vctk-v080-preparation-v1"
_VCTK_PREPARATION_FILE = "SAID_VCTK_PREPARATION.json"


def _remove_silence_rms(
    audio: np.ndarray,
    *,
    sample_rate: int,
    threshold: float,
    window_seconds: float,
    hop_ratio: float,
) -> np.ndarray:
    """Apply the RMS activity rule used for the paper's VCTK clips."""

    value = np.asarray(audio, dtype=np.float32).reshape(-1)
    window = max(1, int(round(sample_rate * window_seconds)))
    hop = max(1, int(round(window * hop_ratio)))
    if value.size < window:
        rms = float(np.sqrt(np.mean(value**2))) if value.size else 0.0
        return value if rms > threshold else np.zeros(0, dtype=value.dtype)
    frames = librosa.util.frame(
        value, frame_length=window, hop_length=hop
    ).T
    inactive = np.flatnonzero(np.sqrt(np.mean(frames**2, axis=-1)) <= threshold)
    if inactive.size == 0:
        return value
    remove = np.zeros(value.size, dtype=np.bool_)
    for start in inactive * hop:
        remove[int(start) : min(int(start) + window, value.size)] = True
    return value[~remove]


def _repeat_to_samples(audio: np.ndarray, samples: int) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32).reshape(-1)
    if value.size == 0:
        return value
    repeats = samples // value.size + 1
    return np.tile(value, repeats)[:samples]


def prepare_vctk_v080(
    source_root: str | Path,
    output_root: str | Path,
    *,
    sample_rate: int = 48_000,
    segment_duration: float = 2.0,
    silence_threshold: float = 0.0055,
    activity_window_seconds: float = 0.05,
    activity_hop_ratio: float = 0.1,
) -> dict[str, object]:
    """Prepare the exact VCTK speech interface used by Audio2Sph training."""

    source_directory = Path(source_root).expanduser().resolve()
    audio_paths = validate_vctk_v080(source_directory)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"VCTK preparation output must be empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "train").mkdir()
    (destination / "test").mkdir()
    shutil.copy2(source_directory / "README", destination / "README")
    shutil.copy2(source_directory / "COPYING", destination / "COPYING")

    speakers = sorted({path.parent.name for path in audio_paths})
    if len(speakers) != 109:
        raise ValueError(
            f"VCTK v0.80 preparation requires 109 speaker directories, found {len(speakers)}"
        )
    split_speakers = {
        "train": speakers[:-10],
        "test": speakers[-10:],
    }
    segment_samples = int(round(sample_rate * segment_duration))
    counts: dict[str, int] = {"train": 0, "test": 0}
    skipped: dict[str, int] = {"train": 0, "test": 0}
    for split, selected_speakers in split_speakers.items():
        selected = set(selected_speakers)
        for path in audio_paths:
            if path.parent.name not in selected:
                continue
            value, _ = librosa.load(
                path=path, sr=sample_rate, mono=True
            )
            active = _remove_silence_rms(
                value,
                sample_rate=sample_rate,
                threshold=silence_threshold,
                window_seconds=activity_window_seconds,
                hop_ratio=activity_hop_ratio,
            )
            if active.size == 0:
                skipped[split] += 1
                continue
            prepared = _repeat_to_samples(active, segment_samples)
            sf.write(destination / split / path.name, prepared, sample_rate)
            counts[split] += 1
    report: dict[str, object] = {
        "schema": _VCTK_PREPARATION_SCHEMA,
        "source_release": "VCTK-0.80",
        "source_license": "ODC-By-1.0",
        "sample_rate": int(sample_rate),
        "segment_duration": float(segment_duration),
        "activity_mode": "rms",
        "activity_window_seconds": float(activity_window_seconds),
        "activity_hop_ratio": float(activity_hop_ratio),
        "silence_threshold": float(silence_threshold),
        "speaker_split": {
            "train": split_speakers["train"],
            "test": split_speakers["test"],
        },
        "written_recordings": counts,
        "skipped_silent_recordings": skipped,
    }
    (destination / _VCTK_PREPARATION_FILE).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def validate_prepared_vctk_v080(
    root: str | Path, *, split: str
) -> tuple[Path, ...]:
    """Validate the prepared paper interface and return one split's clips."""

    if split not in {"train", "test"}:
        raise ValueError("prepared VCTK split must be 'train' or 'test'")
    directory = Path(root).expanduser().resolve()
    metadata_path = directory / _VCTK_PREPARATION_FILE
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"prepared VCTK metadata is missing: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema": _VCTK_PREPARATION_SCHEMA,
        "source_release": "VCTK-0.80",
        "source_license": "ODC-By-1.0",
        "sample_rate": 48_000,
        "segment_duration": 2.0,
        "activity_mode": "rms",
        "activity_window_seconds": 0.05,
        "activity_hop_ratio": 0.1,
        "silence_threshold": 0.0055,
    }
    mismatched = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatched:
        raise ValueError(
            f"prepared VCTK metadata does not match the paper recipe: {mismatched}"
        )
    if not (directory / "README").is_file() or not (directory / "COPYING").is_file():
        raise FileNotFoundError(
            "prepared VCTK must retain the source README and COPYING notices"
        )
    readme_text = (directory / "README").read_text(
        encoding="utf-8", errors="replace"
    )
    license_text = (directory / "COPYING").read_text(
        encoding="utf-8", errors="replace"
    )
    if "Version 0.80" not in readme_text or "109 English" not in readme_text:
        raise ValueError("prepared VCTK README does not identify VCTK v0.80")
    if "ODC Attribution License" not in license_text:
        raise ValueError("prepared VCTK COPYING does not declare ODC-By 1.0")

    speaker_split = metadata.get("speaker_split")
    if not isinstance(speaker_split, dict) or set(speaker_split) != {
        "train",
        "test",
    }:
        raise ValueError("prepared VCTK metadata has no complete speaker split")
    train_speakers = speaker_split["train"]
    test_speakers = speaker_split["test"]
    if (
        not isinstance(train_speakers, list)
        or not isinstance(test_speakers, list)
        or len(train_speakers) != 99
        or len(test_speakers) != 10
        or any(not isinstance(value, str) for value in train_speakers + test_speakers)
        or train_speakers != sorted(train_speakers)
        or test_speakers != sorted(test_speakers)
        or set(train_speakers) & set(test_speakers)
        or len(set(train_speakers + test_speakers)) != 109
    ):
        raise ValueError(
            "prepared VCTK speaker split must contain 99 sorted training "
            "speakers and 10 disjoint sorted test speakers"
        )

    written_recordings = metadata.get("written_recordings")
    if not isinstance(written_recordings, dict) or set(written_recordings) != {
        "train",
        "test",
    }:
        raise ValueError(
            "prepared VCTK metadata has no complete recording counts"
        )
    split_paths: dict[str, tuple[Path, ...]] = {}
    for split_name, speakers in (
        ("train", train_speakers),
        ("test", test_speakers),
    ):
        paths = tuple(sorted((directory / split_name).glob("*.wav")))
        expected_count = written_recordings[split_name]
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count <= 0
            or len(paths) != expected_count
        ):
            raise ValueError(
                f"prepared VCTK {split_name} recording count does not match metadata"
            )
        allowed_speakers = set(speakers)
        for path in paths:
            speaker = path.stem.split("_", maxsplit=1)[0]
            if speaker not in allowed_speakers:
                raise ValueError(
                    f"prepared VCTK recording is assigned to the wrong split: {path}"
                )
            information = sf.info(path)
            if (
                information.samplerate != 48_000
                or information.frames != 96_000
                or information.channels != 1
            ):
                raise ValueError(
                    "prepared VCTK recordings must be mono, 48000 Hz, and "
                    f"96000 samples: {path}"
                )
        split_paths[split_name] = paths
    return split_paths[split]


def _sample_activity(
    random: np.random.Generator,
    *,
    source_count: int,
    samples: int,
    frames: int,
    duration_seconds: float,
    always_active_probability: float,
    minimum_intervals: int,
    maximum_intervals: int,
    minimum_interval_seconds: float,
    maximum_interval_seconds: float,
    minimum_gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    sample_activity = np.zeros((source_count, samples), dtype=np.float32)
    frame_activity = np.zeros((source_count, frames), dtype=np.float32)
    for source_index in range(source_count):
        if float(random.random()) < always_active_probability:
            sample_activity[source_index] = 1.0
            frame_activity[source_index] = 1.0
            continue
        interval_count = int(
            random.integers(minimum_intervals, maximum_intervals + 1)
        )
        intervals: list[tuple[float, float]] = []
        for _ in range(interval_count):
            for _ in range(64):
                interval_duration = float(
                    random.uniform(
                        minimum_interval_seconds, maximum_interval_seconds
                    )
                )
                interval_duration = min(duration_seconds, interval_duration)
                latest_start = max(0.0, duration_seconds - interval_duration)
                start = float(random.uniform(0.0, latest_start))
                end = start + interval_duration
                if all(
                    end + minimum_gap_seconds <= previous_start
                    or start >= previous_end + minimum_gap_seconds
                    for previous_start, previous_end in intervals
                ):
                    intervals.append((start, end))
                    break
        if not intervals:
            interval_duration = min(
                duration_seconds,
                0.5 * (minimum_interval_seconds + maximum_interval_seconds),
            )
            latest_start = max(0.0, duration_seconds - interval_duration)
            start = float(random.uniform(0.0, latest_start))
            intervals.append((start, start + interval_duration))
        for start, end in intervals:
            sample_start = min(
                samples - 1,
                max(0, int(math.floor(start / duration_seconds * samples))),
            )
            sample_end = min(
                samples,
                max(sample_start + 1, int(math.ceil(end / duration_seconds * samples))),
            )
            sample_activity[source_index, sample_start:sample_end] = 1.0
            if frames == 1:
                frame_start, frame_end = 0, 1
            else:
                scale = (frames - 1) / duration_seconds
                frame_start = min(frames - 1, max(0, int(math.ceil(start * scale))))
                last_frame = min(
                    frames - 1,
                    max(frame_start, int(math.floor(end * scale))),
                )
                frame_end = last_frame + 1
            frame_activity[source_index, frame_start:frame_end] = 1.0
    return sample_activity, frame_activity


class SimulatedSceneDataset(Dataset[dict[str, Any]]):
    """Generate paper-configured Audio2Sph or complete-SAID scenes online."""

    def __init__(
        self,
        *,
        recipe: str,
        scene_count: int,
        seed: int,
        sample_rate: int,
        segment_duration: float,
        renderer: AcousticSceneRenderer | None = None,
        prepared_vctk_root: str | Path | None = None,
        vctk_split: str = "train",
        sourcebank_manifest: str | Path | None = None,
        source_audio_root: str | Path | None = None,
        require_commercial_use: bool = False,
        source_count_probabilities: Mapping[Any, Any] | None = None,
        class_probabilities: Mapping[Any, Any] | None = None,
        source_rms: float = 0.08,
        always_active_probability: float = 0.3,
        minimum_intervals: int = 1,
        maximum_intervals: int = 3,
        minimum_interval_seconds: float = 0.2,
        maximum_interval_seconds: float = 1.0,
        minimum_gap_seconds: float = 0.05,
    ) -> None:
        if recipe not in {"audio2sph_pretraining", "sourcebank_training"}:
            raise ValueError(
                "recipe must be 'audio2sph_pretraining' or 'sourcebank_training'"
            )
        self.recipe = recipe
        self.scene_count = int(scene_count)
        self.seed = int(seed)
        self.sample_rate = int(sample_rate)
        self.segment_duration = float(segment_duration)
        self.samples = int(round(self.sample_rate * self.segment_duration))
        self.frames_100hz = int(round(self.segment_duration * 100.0)) + 1
        self.source_rms = float(source_rms)
        if self.scene_count <= 0 or self.samples <= 0:
            raise ValueError("scene_count and audio duration must be positive")
        if not 0.0 <= always_active_probability <= 1.0:
            raise ValueError("always_active_probability must lie in [0,1]")
        if not 1 <= int(minimum_intervals) <= int(maximum_intervals):
            raise ValueError("activity interval counts are invalid")
        if not 0.0 < minimum_interval_seconds <= maximum_interval_seconds:
            raise ValueError("activity interval durations are invalid")
        if maximum_interval_seconds > self.segment_duration:
            raise ValueError("activity intervals must fit within the scene")
        if minimum_gap_seconds < 0.0:
            raise ValueError("minimum activity gap must be non-negative")
        self.activity_parameters = {
            "always_active_probability": float(always_active_probability),
            "minimum_intervals": int(minimum_intervals),
            "maximum_intervals": int(maximum_intervals),
            "minimum_interval_seconds": float(minimum_interval_seconds),
            "maximum_interval_seconds": float(maximum_interval_seconds),
            "minimum_gap_seconds": float(minimum_gap_seconds),
        }
        if renderer is None:
            from ..rendering import AcousticSceneRenderer

            renderer = AcousticSceneRenderer(
                sample_rate=self.sample_rate,
                segment_duration=self.segment_duration,
            )
        self.renderer = renderer

        self.vctk_paths: tuple[Path, ...] = ()
        self.sourcebank_records: tuple[SourceBankRecord, ...] = ()
        self.records_by_class: dict[int, tuple[SourceBankRecord, ...]] = {}
        if self.recipe == "audio2sph_pretraining":
            if prepared_vctk_root is None:
                raise ValueError(
                    "Audio2Sph pretraining requires prepared_vctk_root"
                )
            self.vctk_paths = validate_prepared_vctk_v080(
                prepared_vctk_root, split=vctk_split
            )
            self.source_counts = np.arange(0, 5, dtype=np.int64)
            self.source_count_probabilities = np.full(5, 0.2, dtype=np.float64)
            self.class_values = np.arange(13, dtype=np.int64)
            self.class_probabilities = np.full(13, 1.0 / 13.0)
        else:
            if sourcebank_manifest is None or source_audio_root is None:
                raise ValueError(
                    "SourceBank training requires its manifest and source_audio_root"
                )
            self.sourcebank_records = load_sourcebank_manifest(
                sourcebank_manifest,
                audio_root=source_audio_root,
                require_commercial_use=require_commercial_use,
            )
            grouped: defaultdict[int, list[SourceBankRecord]] = defaultdict(list)
            for record in self.sourcebank_records:
                grouped[record.class_id].append(record)
            if set(grouped) != set(range(13)):
                raise ValueError(
                    "the SourceBank paper recipe requires authorized rows for all 13 classes"
                )
            self.records_by_class = {
                class_id: tuple(records) for class_id, records in grouped.items()
            }
            if source_count_probabilities is None or class_probabilities is None:
                raise ValueError(
                    "SourceBank paper probabilities must be explicit in the data config"
                )
            self.source_counts, self.source_count_probabilities = _probability_table(
                source_count_probabilities,
                allowed=range(1, 7),
                field="source_count_probabilities",
            )
            self.class_values, self.class_probabilities = _probability_table(
                class_probabilities,
                allowed=range(13),
                field="class_probabilities",
            )

    def __len__(self) -> int:
        return self.scene_count

    def _load_audio_range(
        self,
        path: Path,
        *,
        random: np.random.Generator,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
        normalize_rms: bool,
    ) -> np.ndarray:
        information = sf.info(str(path))
        source_rate = int(information.samplerate)
        file_duration = float(information.frames) / source_rate
        range_end = file_duration if end_seconds is None else min(
            file_duration, float(end_seconds)
        )
        range_start = max(0.0, float(start_seconds))
        authorized_start = int(math.ceil(range_start * source_rate))
        authorized_stop = min(
            int(information.frames),
            int(math.ceil(range_end * source_rate)),
        )
        if authorized_start >= authorized_stop:
            raise ValueError(
                f"audio range [{range_start}, {range_end}] is empty for {path}"
            )
        requested_frames = int(math.ceil(self.segment_duration * source_rate))
        available_frames = authorized_stop - authorized_start
        if available_frames > requested_frames:
            latest_start = authorized_stop - requested_frames
            source_start = int(
                random.integers(authorized_start, latest_start + 1)
            )
            source_frames = requested_frames
        else:
            source_start = authorized_start
            source_frames = available_frames
        audio, _ = sf.read(
            str(path),
            start=source_start,
            frames=source_frames,
            always_2d=True,
            dtype="float32",
        )
        mono = audio.mean(axis=1)
        mono = _linear_resample(mono, source_rate, self.sample_rate)
        if mono.size < self.samples:
            mono = np.pad(mono, (0, self.samples - mono.size))
        else:
            mono = mono[: self.samples]
        if normalize_rms:
            rms = float(np.sqrt(np.mean(np.square(mono)) + 1.0e-12))
            if rms >= 1.0e-5 and self.source_rms > 0.0:
                mono = mono * (self.source_rms / rms)
        return mono.astype(np.float32, copy=False)

    def _sample_sources(
        self, random: np.random.Generator
    ) -> tuple[list[np.ndarray], list[int]]:
        source_count = int(
            random.choice(self.source_counts, p=self.source_count_probabilities)
        )
        audio: list[np.ndarray] = []
        class_ids: list[int] = []
        if self.recipe == "audio2sph_pretraining":
            for _ in range(source_count):
                path = self.vctk_paths[int(random.integers(0, len(self.vctk_paths)))]
                audio.append(
                    self._load_audio_range(
                        path,
                        random=random,
                        normalize_rms=False,
                    )
                )
                class_ids.append(0)
            return audio, class_ids
        for _ in range(source_count):
            class_id = int(
                random.choice(self.class_values, p=self.class_probabilities)
            )
            candidates = self.records_by_class[class_id]
            record = candidates[int(random.integers(0, len(candidates)))]
            audio.append(
                self._load_audio_range(
                    record.audio_path,
                    random=random,
                    start_seconds=record.start_seconds,
                    end_seconds=record.end_seconds,
                    normalize_rms=True,
                )
            )
            class_ids.append(class_id)
        return audio, class_ids

    def __getitem__(self, index: int) -> dict[str, Any]:
        logical_index = int(index)
        if not 0 <= logical_index < self.scene_count:
            raise IndexError(logical_index)
        random = np.random.default_rng(self.seed + logical_index * 1009)
        sources, class_ids = self._sample_sources(random)
        sample_activity, frame_activity = _sample_activity(
            random,
            source_count=len(sources),
            samples=self.samples,
            frames=self.frames_100hz,
            duration_seconds=self.segment_duration,
            **self.activity_parameters,
        )
        rendered = self.renderer.render(
            sources,
            class_ids,
            random=random,
            extended_sources=self.recipe == "sourcebank_training",
            activity_samples=sample_activity,
            activity_frames_100hz=frame_activity,
            add_noise=self.recipe == "sourcebank_training",
            target_kind=(
                "audio2sph"
                if self.recipe == "audio2sph_pretraining"
                else "said"
            ),
        )
        if self.recipe == "audio2sph_pretraining":
            return {
                "audio": rendered.audio,
                "target_maps": rendered.audio2sph_target_180x360,
                "scene_index": logical_index,
                "source_count": len(sources),
            }
        return {
            "audio": rendered.audio,
            "targets": {
                "valid_sources": rendered.valid_sources,
                "class_ids": rendered.said_class_ids,
                "map_targets": rendered.said_map_targets,
                "refined_map_targets": rendered.said_refined_map_targets,
                "frame_alignment": rendered.frame_alignment.value,
            },
            "recording_domain": None,
            "scene_index": logical_index,
            "source_count": len(sources),
        }


def collate_audio2sph_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate Online Scene Generation examples for Audio2Sph pretraining."""

    if not items:
        raise ValueError("cannot collate an empty Audio2Sph batch")
    return {
        "audio": torch.stack([item["audio"] for item in items]),
        "target_maps": torch.stack([item["target_maps"] for item in items]),
        "metadata": [
            {
                "scene_index": int(item["scene_index"]),
                "source_count": int(item["source_count"]),
            }
            for item in items
        ],
    }
