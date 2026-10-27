"""Auditable orchestration for the published SAID training recipes."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

import json
import os
import random
import time
import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sized

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from ..utils.download_checkpoints import (
    checkpoint_cache_path,
    ensure_paper_checkpoint,
)
from ..utils.load_config import ConfigError, SAIDConfig
from ..data import (
    DCASE2026TrainingDataset,
    SimulatedSceneDataset,
    collate_audio2sph_batch,
    collate_source_map_batch,
)
from ..models import (
    Audio2SphPretrainingModel,
    SAID,
    load_audio2sph_component_checkpoint,
    load_audio2sph_pretraining_checkpoint,
    load_class_feature_encoder_checkpoint,
    load_said_checkpoint,
)
from ..rendering import AcousticSceneRenderer
from .checkpoint import (
    load_training_checkpoint,
    save_model_checkpoint,
    save_training_checkpoint,
)
from .losses import Audio2SphPretrainingCriterion, SAIDCriterion
from .steps import (
    ExponentialMovingAverage,
    PaperTrainingPhase,
    build_optimizer_and_scheduler,
    train_audio2sph_step,
    train_said_step,
)


@dataclass(frozen=True)
class TrainingRunReport:
    """Files and final step produced by a completed training invocation."""

    output_directory: Path
    completed_step: int
    latest_training_state: Path
    latest_ema_checkpoint: Path
    metrics_path: Path
    best_ema_checkpoint: Path | None = None
    best_step: int | None = None
    best_metric: float | None = None


class DeterministicStepBatchSampler(Sampler[list[int]]):
    """Map each optimizer step to reproducible dataset indices.

    The mapping is stateless, so DataLoader prefetch does not alter which
    examples are consumed after resuming at a saved optimizer step.
    """

    def __init__(
        self,
        dataset: Sized,
        *,
        start_step: int,
        total_steps: int,
        batch_size: int,
        seed: int,
        sampling_weights: tuple[float, ...] | None = None,
    ) -> None:
        self.dataset_size = int(len(dataset))
        self.start_step = int(start_step)
        self.total_steps = int(total_steps)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.cumulative_weights: tuple[float, ...] | None = None
        if self.dataset_size <= 0:
            raise ValueError("training dataset must not be empty")
        if not 0 <= self.start_step <= self.total_steps:
            raise ValueError("start_step must lie in [0,total_steps]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sampling_weights is not None:
            if len(sampling_weights) != self.dataset_size:
                raise ValueError(
                    "sampling_weights must contain one value per dataset item"
                )
            cumulative: list[float] = []
            total = 0.0
            for weight in sampling_weights:
                value = float(weight)
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        "sampling weights must be finite and positive"
                    )
                total += value
                cumulative.append(total)
            self.cumulative_weights = tuple(cumulative)

    @staticmethod
    def _splitmix64(value: int) -> int:
        mask = (1 << 64) - 1
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return (value ^ (value >> 31)) & mask

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.total_steps):
            batch: list[int] = []
            for batch_offset in range(self.batch_size):
                random_bits = self._splitmix64(
                    self.seed ^ (step * self.batch_size + batch_offset)
                )
                if self.cumulative_weights is None:
                    index = random_bits % self.dataset_size
                else:
                    position = (
                        random_bits / float(1 << 64)
                    ) * self.cumulative_weights[-1]
                    index = min(
                        bisect.bisect_right(
                            self.cumulative_weights, position
                        ),
                        self.dataset_size - 1,
                    )
                batch.append(index)
            yield batch

    def __len__(self) -> int:
        return self.total_steps - self.start_step


class DeterministicSceneBatchSampler(Sampler[list[int]]):
    """Assign one unique, reproducible Online Scene Generation ID per draw."""

    def __init__(
        self,
        dataset: Sized,
        *,
        start_step: int,
        total_steps: int,
        batch_size: int,
    ) -> None:
        self.dataset_size = int(len(dataset))
        self.start_step = int(start_step)
        self.total_steps = int(total_steps)
        self.batch_size = int(batch_size)
        if not 0 <= self.start_step <= self.total_steps:
            raise ValueError("start_step must lie in [0,total_steps]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.total_steps * self.batch_size > self.dataset_size:
            raise ValueError(
                "simulated scene_count must cover every configured training draw"
            )

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.total_steps):
            first = step * self.batch_size
            yield list(range(first, first + self.batch_size))

    def __len__(self) -> int:
        return self.total_steps - self.start_step


def _acquire_canonical_paper_asset(path: Path, model: str) -> Path:
    """Acquire a configured paper asset only at its canonical project path.

    Explicit user paths remain strictly local.  The reference recipes name the
    project checkpoint directory, so their Class Feature Encoder or complete-model
    initialization is downloaded and verified by the same path used by
    ``said demo``, ``said infer``, and ``said evaluate``.
    """

    configured = path.expanduser().resolve()
    canonical = checkpoint_cache_path(model)
    if configured == canonical:
        return ensure_paper_checkpoint(model, destination=configured)
    return configured


def _read_data_configuration(
    config: SAIDConfig,
) -> tuple[Mapping[str, Any], Mapping[str, Any], bytes]:
    config.validate_data_config()
    raw_bytes = config.data_config.read_bytes()
    raw = yaml.safe_load(raw_bytes)
    if not isinstance(raw, Mapping):  # guarded by validate_data_config
        raise ConfigError("data configuration root must be a mapping")
    section = raw[config.data]
    if not isinstance(section, Mapping):  # guarded by validate_data_config
        raise ConfigError(f"data section {config.data!r} must be a mapping")
    audio = raw["audio"]
    if not isinstance(audio, Mapping):  # guarded by validate_data_config
        raise ConfigError("audio data contract must be a mapping")
    return section, audio, raw_bytes


def _dcase_dataset(
    config: SAIDConfig,
    section: Mapping[str, Any],
    audio: Mapping[str, Any],
):
    required = {
        "dataset",
        "root",
        "train_split",
        "validation_split",
        "rotation_views",
        "maximum_sources",
        "class_counts",
        "class_balancing",
    }
    unknown = set(section) - required
    missing = required - set(section)
    if unknown or missing:
        raise ConfigError(
            "dcase_recordings fields mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if section["dataset"] != "dcase2026_task3_track_a":
        raise ConfigError(
            "the paper recipe requires dataset='dcase2026_task3_track_a'"
        )
    root_value = section["root"]
    if not isinstance(root_value, str) or not root_value.strip():
        raise ConfigError("dcase_recordings.root must be a filesystem path")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (config.data_config.parent / root).resolve()
    split = section["train_split"]
    if split != "dev_train":
        raise ConfigError("the paper fine-tuning recipe requires train_split='dev_train'")
    if section["validation_split"] != "dev_test":
        raise ConfigError(
            "the paper fine-tuning recipe requires "
            "validation_split='dev_test'"
        )
    rotation_views = section["rotation_views"]
    if not isinstance(rotation_views, bool):
        raise ConfigError("dcase_recordings.rotation_views must be true or false")
    maximum_sources = section["maximum_sources"]
    if isinstance(maximum_sources, bool) or not isinstance(maximum_sources, int):
        raise ConfigError("dcase_recordings.maximum_sources must be an integer")
    class_counts = section["class_counts"]
    if (
        not isinstance(class_counts, list)
        or len(class_counts) != 13
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) <= 0.0
            for value in class_counts
        )
    ):
        raise ConfigError(
            "dcase_recordings.class_counts must contain 13 positive values"
        )
    dataset = DCASE2026TrainingDataset(
        root,
        split=str(split),
        sample_rate=int(audio["sample_rate"]),
        segment_duration=float(audio["segment_duration"]),
        rotation_views=rotation_views,
        maximum_sources=int(maximum_sources),
    )
    balancing = section["class_balancing"]
    required_balancing = {
        "exponent",
        "minimum_weight",
        "maximum_weight",
        "no_class_weight",
    }
    if not isinstance(balancing, Mapping):
        raise ConfigError("dcase_recordings.class_balancing must be a mapping")
    if set(balancing) != required_balancing:
        raise ConfigError(
            "dcase_recordings.class_balancing fields mismatch: "
            f"missing={sorted(required_balancing - set(balancing))}, "
            f"unknown={sorted(set(balancing) - required_balancing)}"
        )
    numeric: dict[str, float] = {}
    for key, value in balancing.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError("DCASE class-balancing values must be numbers")
        numeric[key] = float(value)
    if not 0.0 < numeric["minimum_weight"] <= numeric["maximum_weight"]:
        raise ConfigError("DCASE class-balancing weight bounds are invalid")
    if numeric["exponent"] <= 0.0 or numeric["no_class_weight"] <= 0.0:
        raise ConfigError("DCASE class-balancing weights must be positive")

    metadata = dataset.class_balance_metadata()
    item_counts = [0] * 13
    for item in metadata:
        for class_id in item["class_ids"]:
            item_counts[int(class_id)] += 1
    nonzero_counts = [count for count in item_counts if count > 0]
    reference = max(nonzero_counts) if nonzero_counts else 1
    class_weights = tuple(
        min(
            numeric["maximum_weight"],
            max(
                numeric["minimum_weight"],
                (
                    (reference / count) ** numeric["exponent"]
                    if count > 0
                    else numeric["maximum_weight"]
                ),
            ),
        )
        for count in item_counts
    )
    sampling_weights = tuple(
        max(class_weights[class_id] for class_id in item["class_ids"])
        if item["class_ids"]
        else numeric["no_class_weight"]
        for item in metadata
    )
    sampling_report = {
        "strategy": "class_balanced_repeat",
        **numeric,
        "class_item_counts": item_counts,
        "class_weights": class_weights,
    }
    return (
        dataset,
        tuple(float(value) for value in class_counts),
        sampling_weights,
        sampling_report,
    )


def _dcase_validation_dataset(
    config: SAIDConfig,
    section: Mapping[str, Any],
    audio: Mapping[str, Any],
) -> DCASE2026TrainingDataset:
    root = _data_path(config, section["root"], "dcase_recordings.root")
    return DCASE2026TrainingDataset(
        root,
        split=str(section["validation_split"]),
        sample_rate=int(audio["sample_rate"]),
        segment_duration=float(audio["segment_duration"]),
        rotation_views=False,
        maximum_sources=int(section["maximum_sources"]),
    )


def _data_path(config: SAIDConfig, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config.data_config.parent / path).resolve()
    return path


def _numeric_range(value: Any, field: str) -> tuple[float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        raise ConfigError(f"{field} must contain two numbers")
    result = (float(value[0]), float(value[1]))
    if result[0] > result[1]:
        raise ConfigError(f"{field} must be ordered [minimum,maximum]")
    return result


def _exact_mapping(
    value: Any, *, field: str, required: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    unknown = set(value) - required
    missing = required - set(value)
    if unknown or missing:
        raise ConfigError(
            f"{field} fields mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return value


def _simulated_dataset(
    config: SAIDConfig,
    section: Mapping[str, Any],
    audio: Mapping[str, Any],
    *,
    validation: bool = False,
):
    section = _exact_mapping(
        section,
        field="simulated_scenes",
        required={"scene_count", "seed", "vctk", "sourcebank", "room", "sources"},
    )
    scene_count = section["scene_count"]
    seed = section["seed"]
    if isinstance(scene_count, bool) or not isinstance(scene_count, int) or scene_count <= 0:
        raise ConfigError("simulated_scenes.scene_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("simulated_scenes.seed must be a non-negative integer")
    required_draws = config.training.total_steps * config.training.batch_size
    if scene_count < required_draws:
        raise ConfigError(
            "simulated_scenes.scene_count must cover total_steps * batch_size"
        )

    vctk = _exact_mapping(
        section["vctk"],
        field="simulated_scenes.vctk",
        required={
            "release",
            "license",
            "source_root",
            "prepared_root",
            "training_split",
            "preprocessing",
        },
    )
    if vctk["release"] != "0.80" or vctk["license"] != "ODC-By-1.0":
        raise ConfigError(
            "the paper recipe requires VCTK release 0.80 under ODC-By-1.0"
        )
    if vctk["training_split"] != "train":
        raise ConfigError("the paper Audio2Sph recipe requires VCTK training_split=train")
    vctk_preprocessing = _exact_mapping(
        vctk["preprocessing"],
        field="simulated_scenes.vctk.preprocessing",
        required={
            "sample_rate",
            "segment_duration",
            "activity_mode",
            "activity_window_seconds",
            "activity_hop_ratio",
            "silence_threshold",
            "held_out_speakers",
        },
    )
    expected_vctk_preprocessing = {
        "sample_rate": 48_000,
        "segment_duration": 2.0,
        "activity_mode": "rms",
        "activity_window_seconds": 0.05,
        "activity_hop_ratio": 0.1,
        "silence_threshold": 0.0055,
        "held_out_speakers": 10,
    }
    if dict(vctk_preprocessing) != expected_vctk_preprocessing:
        raise ConfigError(
            "simulated_scenes.vctk.preprocessing must match the published "
            f"Audio2Sph recipe: {expected_vctk_preprocessing}"
        )
    sourcebank = _exact_mapping(
        section["sourcebank"],
        field="simulated_scenes.sourcebank",
        required={"manifest", "source_audio_root", "require_commercial_use"},
    )
    if not isinstance(sourcebank["require_commercial_use"], bool):
        raise ConfigError(
            "simulated_scenes.sourcebank.require_commercial_use must be boolean"
        )

    room = _exact_mapping(
        section["room"],
        field="simulated_scenes.room",
        required={
            "width_m",
            "length_m",
            "height_m",
            "wall_absorption",
            "maximum_reflection_order",
        },
    )
    width = _numeric_range(room["width_m"], "simulated_scenes.room.width_m")
    length = _numeric_range(room["length_m"], "simulated_scenes.room.length_m")
    height = _numeric_range(room["height_m"], "simulated_scenes.room.height_m")
    absorption = _numeric_range(
        room["wall_absorption"], "simulated_scenes.room.wall_absorption"
    )
    maximum_order = room["maximum_reflection_order"]
    if (
        isinstance(maximum_order, bool)
        or not isinstance(maximum_order, int)
        or maximum_order < 0
    ):
        raise ConfigError("room.maximum_reflection_order must be non-negative")

    sources = _exact_mapping(
        section["sources"],
        field="simulated_scenes.sources",
        required={
            "audio2sph_count",
            "said_count_probabilities",
            "class_probabilities",
            "source_rms",
            "extent",
            "activity",
            "noise",
            "target_standard_deviation_degrees",
        },
    )
    if sources["audio2sph_count"] != [0, 4]:
        raise ConfigError("the paper Audio2Sph recipe requires 0--4 sources")
    source_rms = sources["source_rms"]
    target_sigma = sources["target_standard_deviation_degrees"]
    if (
        isinstance(source_rms, bool)
        or not isinstance(source_rms, (int, float))
        or float(source_rms) <= 0.0
    ):
        raise ConfigError("simulated source_rms must be positive")
    if (
        isinstance(target_sigma, bool)
        or not isinstance(target_sigma, (int, float))
        or float(target_sigma) <= 0.0
    ):
        raise ConfigError("target_standard_deviation_degrees must be positive")
    extent = _exact_mapping(
        sources["extent"],
        field="simulated_scenes.sources.extent",
        required={"point_spacing_degrees", "sizes_degrees", "probabilities"},
    )
    sizes = extent["sizes_degrees"]
    probabilities = extent["probabilities"]
    if (
        not isinstance(sizes, list)
        or len(sizes) != 3
        or not isinstance(probabilities, list)
        or len(probabilities) != 3
    ):
        raise ConfigError("source extent sizes and probabilities require three values")
    try:
        extent_sizes = tuple(float(value) for value in sizes)
        extent_probabilities = tuple(float(value) for value in probabilities)
        point_spacing = float(extent["point_spacing_degrees"])
    except (TypeError, ValueError) as error:
        raise ConfigError("source extent values must be numeric") from error
    if (
        point_spacing <= 0.0
        or any(value <= 0.0 for value in extent_sizes)
        or any(value < 0.0 for value in extent_probabilities)
        or sum(extent_probabilities) <= 0.0
    ):
        raise ConfigError("source extent values are invalid")

    activity = _exact_mapping(
        sources["activity"],
        field="simulated_scenes.sources.activity",
        required={
            "always_active_probability",
            "minimum_intervals",
            "maximum_intervals",
            "minimum_interval_seconds",
            "maximum_interval_seconds",
            "minimum_gap_seconds",
        },
    )
    noise = _exact_mapping(
        sources["noise"],
        field="simulated_scenes.sources.noise",
        required={"enabled_for_sourcebank", "snr_db"},
    )
    if noise["enabled_for_sourcebank"] is not True:
        raise ConfigError("the paper SourceBank recipe requires additive noise")
    noise_snr = _numeric_range(
        noise["snr_db"], "simulated_scenes.sources.noise.snr_db"
    )
    renderer = AcousticSceneRenderer(
        sample_rate=int(audio["sample_rate"]),
        segment_duration=float(audio["segment_duration"]),
        target_standard_deviation_degrees=float(target_sigma),
        maximum_sources=6,
        maximum_reflection_order=int(maximum_order),
        room_width_m=width,
        room_length_m=length,
        room_height_m=height,
        wall_absorption=absorption,
        source_extent_sizes_degrees=extent_sizes,  # type: ignore[arg-type]
        source_extent_probabilities=extent_probabilities,  # type: ignore[arg-type]
        source_point_spacing_degrees=point_spacing,
        noise_snr_db=noise_snr,
    )
    recipe = (
        "audio2sph_pretraining"
        if config.model == "audio2sph"
        else "sourcebank_training"
    )
    try:
        dataset = SimulatedSceneDataset(
            recipe=recipe,
            scene_count=int(scene_count),
            seed=int(seed) + (1_000_000_007 if validation else 0),
            sample_rate=int(audio["sample_rate"]),
            segment_duration=float(audio["segment_duration"]),
            renderer=renderer,
            prepared_vctk_root=(
                _data_path(
                    config,
                    vctk["prepared_root"],
                    "simulated_scenes.vctk.prepared_root",
                )
                if recipe == "audio2sph_pretraining"
                else None
            ),
            vctk_split=(
                "test"
                if validation and recipe == "audio2sph_pretraining"
                else str(vctk["training_split"])
            ),
            sourcebank_manifest=(
                _data_path(
                    config,
                    sourcebank["manifest"],
                    "simulated_scenes.sourcebank.manifest",
                )
                if recipe == "sourcebank_training"
                else None
            ),
            source_audio_root=(
                _data_path(
                    config,
                    sourcebank["source_audio_root"],
                    "simulated_scenes.sourcebank.source_audio_root",
                )
                if recipe == "sourcebank_training"
                else None
            ),
            require_commercial_use=bool(sourcebank["require_commercial_use"]),
            source_count_probabilities=sources["said_count_probabilities"],
            class_probabilities=sources["class_probabilities"],
            source_rms=float(source_rms),
            always_active_probability=float(activity["always_active_probability"]),
            minimum_intervals=int(activity["minimum_intervals"]),
            maximum_intervals=int(activity["maximum_intervals"]),
            minimum_interval_seconds=float(activity["minimum_interval_seconds"]),
            maximum_interval_seconds=float(activity["maximum_interval_seconds"]),
            minimum_gap_seconds=float(activity["minimum_gap_seconds"]),
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid simulated-scene recipe: {error}") from error
    class_counts = tuple(
        float(sources["class_probabilities"][class_id])
        for class_id in range(13)
    )
    report = {
        "strategy": "unique_indexed_online_scenes",
        "scene_count": int(scene_count),
        "seed": int(seed) + (1_000_000_007 if validation else 0),
        "recipe": recipe,
        "validation": bool(validation),
    }
    return dataset, class_counts, report


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {name}")
    return device


def _configuration_fingerprint_input(
    config_path: Path, data_configuration: bytes
) -> bytes:
    return (
        config_path.read_bytes()
        + b"\n--said-data-configuration--\n"
        + data_configuration
    )


def _prepare_output(
    directory: Path, *, resume: Path | None
) -> tuple[Path | None, Path | None]:
    """Resolve automatic resume or preserve an interrupted pre-checkpoint run."""

    if directory.exists() and not directory.is_dir():
        raise FileExistsError(
            f"training output path is not a directory: {directory}"
        )
    if resume is not None:
        directory.mkdir(parents=True, exist_ok=True)
        return resume, None

    automatic_resume = directory / "latest.training.pt"
    if automatic_resume.is_file():
        return automatic_resume.resolve(), None

    archived: Path | None = None
    if directory.is_dir() and any(directory.iterdir()):
        archive_root = directory.parent / "_interrupted"
        archive_root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        archived = archive_root / f"{directory.name}-{timestamp}"
        suffix = 1
        while archived.exists():
            archived = archive_root / (
                f"{directory.name}-{timestamp}-{suffix}"
            )
            suffix += 1
        directory.rename(archived)
    directory.mkdir(parents=True, exist_ok=True)
    return None, archived


def _append_metric(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _format_training_metrics(record: Mapping[str, Any]) -> str:
    """Format the periodic human-readable line shown beside the progress bar."""

    fields = [
        f"step={int(record['step'])}",
        f"loss={float(record['loss']):.6f}",
    ]
    metric_labels = (
        ("map_bce", "map_bce"),
        ("map_dice", "map_dice"),
        ("refine", "refine"),
        ("soft_iou", "soft_iou"),
        ("correlation", "corr"),
        ("class_head_focal", "class_focal"),
        ("active_head_cross_entropy", "active_ce"),
    )
    for key, label in metric_labels:
        if key in record:
            fields.append(f"{label}={float(record[key]):.6f}")
    if "matched_sources" in record:
        fields.append(f"matched={int(record['matched_sources'])}")
    learning_rates = record.get("learning_rates", ())
    learning_rate_text = ",".join(
        f"{float(value):.3e}" for value in learning_rates
    )
    fields.append(f"lr={learning_rate_text}")
    return " ".join(fields)


def _truncate_metrics_after(path: Path, completed_step: int) -> None:
    """Discard records newer than the state being resumed."""

    if not path.is_file():
        return
    kept: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            record = json.loads(line)
            step = int(record["step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid training metric at {path}:{line_number}"
            ) from error
        if step <= int(completed_step):
            kept.append(json.dumps(record, sort_keys=True))
    temporary = path.with_name(f".{path.name}.resume.tmp")
    temporary.write_text(
        "" if not kept else "\n".join(kept) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_training_outputs(
    *,
    output_directory: Path,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    class_feature_encoder: str | None,
    configuration: bytes,
    point_generator: torch.Generator,
    keep_milestone: bool,
    run_state: Mapping[str, float | int | None],
) -> tuple[Path, Path]:
    latest_ema = save_model_checkpoint(
        output_directory / "latest.ema.ckpt",
        model=ema.model,
        replace=True,
    )
    latest_training = save_training_checkpoint(
        output_directory / "latest.training.pt",
        model=model,
        ema_model=ema.model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        class_feature_encoder=class_feature_encoder,
        configuration=configuration,
        data_generator=point_generator,
        run_state=run_state,
        replace=True,
    ).path
    if keep_milestone:
        milestones = (
            (latest_ema, output_directory / f"step-{step:07d}.ema.ckpt"),
            (
                latest_training,
                output_directory / f"step-{step:07d}.training.pt",
            ),
        )
        for source, milestone in milestones:
            if milestone.exists():
                continue
            os.link(source, milestone)
    return latest_training, latest_ema


@torch.no_grad()
def _validate_model(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    *,
    config: SAIDConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    validation_generator = torch.Generator(device=device).manual_seed(
        config.training.seed + 3
    )
    for batch in loader:
        if config.model == "audio2sph":
            assert isinstance(model, Audio2SphPretrainingModel)
            assert isinstance(criterion, Audio2SphPretrainingCriterion)
            loss = criterion(
                model(batch["audio"].to(device, non_blocking=True)),
                batch["target_maps"].to(device, non_blocking=True),
            )
            values: Mapping[str, Any] = {"loss": loss}
        else:
            assert isinstance(model, SAID)
            assert isinstance(criterion, SAIDCriterion)
            losses = criterion(
                model(
                    batch["audio"].to(device, non_blocking=True),
                    recording_domain=batch["recording_domain"],
                ),
                batch["targets"].to(device),
                generator=validation_generator,
            )
            values = losses.as_dict()
        batches += 1
        for name, value in values.items():
            if isinstance(value, int):
                totals[name] = totals.get(name, 0.0) + float(value)
            else:
                numeric = float(value.detach().cpu())
                if not np.isfinite(numeric):
                    raise FloatingPointError(
                        f"validation metric is not finite: {name}={numeric}"
                    )
                totals[name] = totals.get(name, 0.0) + numeric
    if batches == 0:
        raise RuntimeError("validation loader produced no batches")
    return {
        f"validation_{name}": total / batches
        for name, total in totals.items()
    }


def run_training(
    config: SAIDConfig,
    *,
    config_path: str | Path,
    resume: str | Path | None = None,
) -> TrainingRunReport:
    """Run one published training recipe selected by ``config``."""

    config_path = Path(config_path).expanduser().resolve()
    runtime = config.training
    output_directory = runtime.output_directory.expanduser().resolve()
    requested_resume = (
        None if resume is None else Path(resume).expanduser().resolve()
    )
    if requested_resume is not None and not requested_resume.is_file():
        raise FileNotFoundError(
            f"training state not found: {requested_resume}"
        )
    section, audio_configuration, data_configuration = (
        _read_data_configuration(config)
    )
    if config.data == "dcase_recordings":
        dataset, class_counts, sampling_weights, sampling_report = _dcase_dataset(
            config, section, audio_configuration
        )
        phase = PaperTrainingPhase.DCASE_FINE_TUNING
        collate = collate_source_map_batch
    else:
        dataset, class_counts, sampling_report = _simulated_dataset(
            config, section, audio_configuration
        )
        sampling_weights = None
        phase = (
            PaperTrainingPhase.AUDIO2SPH_PRETRAINING
            if config.model == "audio2sph"
            else PaperTrainingPhase.SOURCEBANK_TRAINING
        )
        collate = (
            collate_audio2sph_batch
            if config.model == "audio2sph"
            else collate_source_map_batch
        )
    validation_dataset = None
    validation_sampling_report = None
    if runtime.validate_every_steps is not None:
        if config.data == "dcase_recordings":
            validation_dataset = _dcase_validation_dataset(
                config, section, audio_configuration
            )
            validation_sampling_report = {
                "strategy": "fixed_dev_test_batches",
                "split": str(
                    section.get("validation_split", "dev_test")
                ),
            }
        else:
            (
                validation_dataset,
                _,
                validation_sampling_report,
            ) = _simulated_dataset(
                config,
                section,
                audio_configuration,
                validation=True,
            )
    resume_path, archived_output = _prepare_output(
        output_directory, resume=requested_resume
    )
    if requested_resume is None and resume_path is not None:
        print(f"[resume] {resume_path}", flush=True)
    if archived_output is not None:
        print(
            "[restart] no resumable checkpoint was found; preserved the "
            f"interrupted output at {archived_output}",
            flush=True,
        )
    random.seed(runtime.seed)
    np.random.seed(runtime.seed)
    torch.manual_seed(runtime.seed)
    device = _device(runtime.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(runtime.seed)
    torch.backends.cudnn.enabled = False
    if config.model == "audio2sph":
        model: nn.Module = Audio2SphPretrainingModel().to(device)
    else:
        assert config.class_feature_encoder is not None
        model = SAID(class_feature_encoder=config.class_feature_encoder).to(device)
    configuration = _configuration_fingerprint_input(
        config_path, data_configuration
    )

    initialization_reports = []
    if resume_path is None:
        if config.model == "audio2sph":
            if config.load_audio2sph_ckpt is not None:
                initialization_reports.append(
                    load_audio2sph_pretraining_checkpoint(
                        model,
                        _acquire_canonical_paper_asset(
                            config.load_audio2sph_ckpt,
                            "audio2sph",
                        ),
                    )
                )
        elif config.data == "simulated_scenes":
            assert isinstance(model, SAID)
            assert config.class_feature_encoder is not None
            assert config.load_audio2sph_ckpt is not None
            assert config.load_class_feature_encoder_ckpt is not None
            initialization_reports.append(
                load_audio2sph_component_checkpoint(
                    model.audio2sph,
                    _acquire_canonical_paper_asset(
                        config.load_audio2sph_ckpt,
                        "audio2sph",
                    ),
                )
            )
            initialization_reports.append(
                load_class_feature_encoder_checkpoint(
                    model.sph2imaging.class_feature_encoder,
                    _acquire_canonical_paper_asset(
                        config.load_class_feature_encoder_ckpt,
                        f"said_{config.class_feature_encoder}",
                    ),
                    class_feature_encoder=config.class_feature_encoder,
                )
            )
        else:
            assert isinstance(model, SAID)
            assert config.class_feature_encoder is not None
            assert config.load_said_ckpt is not None
            initialization_reports.append(
                load_said_checkpoint(
                    model,
                    _acquire_canonical_paper_asset(
                        config.load_said_ckpt,
                        f"said_{config.class_feature_encoder}",
                    ),
                    class_feature_encoder=config.class_feature_encoder,
                )
            )
        start_step = 0
    else:
        start_step = 0

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        phase=phase,
        total_steps=runtime.total_steps,
        warmup_steps=runtime.warmup_steps,
        learning_rate=runtime.learning_rate,
        audio2sph_learning_rate=runtime.audio2sph_learning_rate,
        minimum_learning_rate_ratio=runtime.minimum_learning_rate_ratio,
    )
    ema = ExponentialMovingAverage(model, decay=runtime.ema_decay)
    point_generator = torch.Generator(device=device).manual_seed(
        runtime.seed + 1
    )
    best_metric: float | None = None
    best_step: int | None = None
    if resume_path is not None:
        report = load_training_checkpoint(
            resume_path,
            model=model,
            ema_model=ema.model,
            optimizer=optimizer,
            scheduler=scheduler,
            class_feature_encoder=config.class_feature_encoder,
            configuration=configuration,
            data_generator=point_generator,
        )
        start_step = report.step
        saved_best_metric = report.run_state.get("best_metric")
        saved_best_step = report.run_state.get("best_step")
        best_metric = (
            None
            if saved_best_metric is None
            else float(saved_best_metric)
        )
        best_step = (
            None if saved_best_step is None else int(saved_best_step)
        )
    if start_step >= runtime.total_steps:
        raise ValueError(
            f"training state step {start_step} has reached total_steps={runtime.total_steps}"
        )

    if config.data == "simulated_scenes":
        sampler: Sampler[list[int]] = DeterministicSceneBatchSampler(
            dataset,
            start_step=start_step,
            total_steps=runtime.total_steps,
            batch_size=runtime.batch_size,
        )
    else:
        sampler = DeterministicStepBatchSampler(
            dataset,
            start_step=start_step,
            total_steps=runtime.total_steps,
            batch_size=runtime.batch_size,
            seed=runtime.seed,
            sampling_weights=sampling_weights,
        )
    data_loader_generator = torch.Generator(device="cpu").manual_seed(
        runtime.seed + 2 + start_step
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=runtime.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=runtime.num_workers > 0,
        generator=data_loader_generator,
    )
    validation_loader = None
    if validation_dataset is not None:
        assert runtime.validate_every_steps is not None
        if config.data == "simulated_scenes":
            validation_sampler: Sampler[list[int]] = (
                DeterministicSceneBatchSampler(
                    validation_dataset,
                    start_step=0,
                    total_steps=runtime.validation_batches,
                    batch_size=runtime.batch_size,
                )
            )
        else:
            validation_sampler = DeterministicStepBatchSampler(
                validation_dataset,
                start_step=0,
                total_steps=runtime.validation_batches,
                batch_size=runtime.batch_size,
                seed=runtime.seed + 3,
            )
        validation_loader = DataLoader(
            validation_dataset,
            batch_sampler=validation_sampler,
            num_workers=runtime.num_workers,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
            persistent_workers=runtime.num_workers > 0,
            generator=torch.Generator(device="cpu").manual_seed(
                runtime.seed + 4
            ),
        )
    criterion: nn.Module
    if config.model == "audio2sph":
        criterion = Audio2SphPretrainingCriterion().to(device)
    else:
        criterion = SAIDCriterion(class_counts=class_counts).to(device)
    metrics_path = output_directory / "metrics.jsonl"
    if resume_path is not None:
        _truncate_metrics_after(metrics_path, start_step)
    run_manifest = {
        "schema": "said-training-run-v1",
        "model": config.model,
        "class_feature_encoder": config.class_feature_encoder,
        "data": config.data,
        "start_step": start_step,
        "total_steps": runtime.total_steps,
        "phase": phase.value,
        "initialization": [
            {
                "path": str(report.path),
                "sha256": report.sha256,
                "source_schema": report.source_schema_id,
                "loaded_key_count": report.loaded_key_count,
                "compatibility_only_key_count": (
                    report.compatibility_only_key_count
                ),
            }
            for report in initialization_reports
        ],
        "resume": None if resume_path is None else str(resume_path),
        "sampling": sampling_report,
        "validation": {
            "every_steps": runtime.validate_every_steps,
            "batches": runtime.validation_batches,
            "selection_metric": runtime.selection_metric,
            "sampling": validation_sampling_report,
        },
    }
    manifest_path = (
        output_directory / "run.json"
        if resume_path is None
        else output_directory / f"resume-from-{start_step:07d}.json"
    )
    serialized_manifest = json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != serialized_manifest:
            raise FileExistsError(
                f"training manifest already exists with different content: "
                f"{manifest_path}"
            )
    else:
        manifest_path.write_text(serialized_manifest, encoding="utf-8")

    if config.model == "audio2sph":
        model_name = "Audio2Sph"
    else:
        assert config.class_feature_encoder is not None
        encoder_name = {
            "passt": "PaSST",
            "audiomae": "AudioMAE",
        }[config.class_feature_encoder]
        model_name = f"SAID ({encoder_name})"
    print(
        f"Training {model_name} | phase {phase.value} | device {device}\n"
        f"Steps {start_step:,} -> {runtime.total_steps:,} | "
        f"output {output_directory}",
        flush=True,
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if not parameter.requires_grad
    )
    trainable_tensors = sum(
        1 for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_tensors = sum(
        1 for parameter in model.parameters() if not parameter.requires_grad
    )
    print(
        f"[params] trainable={trainable_parameters} frozen={frozen_parameters} "
        f"trainable_tensors={trainable_tensors} "
        f"frozen_tensors={frozen_tensors}",
        flush=True,
    )
    started = time.monotonic()
    latest_training = output_directory / "latest.training.pt"
    latest_ema = output_directory / "latest.ema.ckpt"
    best_ema = output_directory / "best.ema.ckpt"
    progress = tqdm(
        loader,
        initial=start_step,
        total=runtime.total_steps,
        desc="train",
        unit="step",
        dynamic_ncols=True,
    )
    for completed_step, batch in enumerate(progress, start=start_step + 1):
        if config.model == "audio2sph":
            assert isinstance(model, Audio2SphPretrainingModel)
            assert isinstance(criterion, Audio2SphPretrainingCriterion)
            result = train_audio2sph_step(
                model,
                audio=batch["audio"].to(device, non_blocking=True),
                target_maps=batch["target_maps"].to(
                    device, non_blocking=True
                ),
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
            )
        else:
            assert isinstance(model, SAID)
            assert isinstance(criterion, SAIDCriterion)
            result = train_said_step(
                model,
                audio=batch["audio"].to(device, non_blocking=True),
                targets=batch["targets"].to(device),
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                recording_domain=batch["recording_domain"],
                generator=point_generator,
            )
        learning_rate_text = ",".join(
            f"{float(group['lr']):.3e}" for group in optimizer.param_groups
        )
        progress.set_postfix(
            loss=f"{result.loss:.4f}",
            lr=learning_rate_text,
            refresh=False,
        )
        if (
            completed_step % runtime.log_every_steps == 0
            or completed_step == runtime.total_steps
        ):
            record = {
                "step": completed_step,
                "elapsed_seconds": time.monotonic() - started,
                "learning_rates": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
                **result.metrics,
            }
            _append_metric(metrics_path, record)
            tqdm.write(_format_training_metrics(record))
        should_validate = (
            validation_loader is not None
            and runtime.validate_every_steps is not None
            and (
                completed_step % runtime.validate_every_steps == 0
                or completed_step == runtime.total_steps
            )
        )
        if should_validate:
            validation_metrics = _validate_model(
                ema.model,
                criterion,
                validation_loader,
                config=config,
                device=device,
            )
            selection_value = float(
                validation_metrics[runtime.selection_metric]
            )
            improved = best_metric is None or selection_value < best_metric
            if improved:
                best_metric = selection_value
                best_step = completed_step
                save_model_checkpoint(
                    best_ema,
                    model=ema.model,
                    replace=True,
                )
                _write_json_atomic(
                    output_directory / "best.json",
                    {
                        "schema": "said-best-checkpoint-v1",
                        "checkpoint": best_ema.name,
                        "step": best_step,
                        "selection_metric": runtime.selection_metric,
                        "value": best_metric,
                        "mode": "min",
                    },
                )
            validation_record = {
                "step": completed_step,
                "event": "validation",
                "improved": improved,
                "best_step": best_step,
                "best_metric": best_metric,
                **validation_metrics,
            }
            _append_metric(metrics_path, validation_record)
            tqdm.write(
                f"[validation] step {completed_step:,}/{runtime.total_steps:,} | "
                f"loss {selection_value:.4f} | "
                f"best {float(best_metric):.4f} at step {int(best_step):,}"
            )
        should_save = (
            completed_step % runtime.save_every_steps == 0
            or completed_step == runtime.total_steps
        )
        if should_save:
            latest_training, latest_ema = _save_training_outputs(
                output_directory=output_directory,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                step=completed_step,
                class_feature_encoder=config.class_feature_encoder,
                configuration=configuration,
                point_generator=point_generator,
                keep_milestone=(
                    completed_step % runtime.keep_every_steps == 0
                    or completed_step == runtime.total_steps
                ),
                run_state={
                    "best_metric": best_metric,
                    "best_step": best_step,
                },
            )
            tqdm.write(
                f"[checkpoint] step {completed_step:,} | "
                f"EMA {latest_ema.name} | state {latest_training.name}"
            )

    return TrainingRunReport(
        output_directory=output_directory,
        completed_step=runtime.total_steps,
        latest_training_state=latest_training,
        latest_ema_checkpoint=latest_ema,
        metrics_path=metrics_path,
        best_ema_checkpoint=(best_ema if best_ema.is_file() else None),
        best_step=best_step,
        best_metric=best_metric,
    )
