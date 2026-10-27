"""Load and validate the public SAID configuration."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


_TOP_LEVEL_FIELDS = {
    "schema",
    "preset",
    "model",
    "class_feature_encoder",
    "data",
    "data_config",
    "load_said_ckpt",
    "load_audio2sph_ckpt",
    "load_class_feature_encoder_ckpt",
    "training",
}

_TRAINING_FIELDS = {
    "output_directory",
    "total_steps",
    "batch_size",
    "num_workers",
    "seed",
    "device",
    "learning_rate",
    "audio2sph_learning_rate",
    "warmup_steps",
    "minimum_learning_rate_ratio",
    "ema_decay",
    "save_every_steps",
    "keep_every_steps",
    "log_every_steps",
}

_OPTIONAL_TRAINING_FIELDS = {
    "validate_every_steps",
    "validation_batches",
    "selection_metric",
}


class ConfigError(ValueError):
    """Raised when a public SAID configuration violates the schema."""


def _positive_integer(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparison = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{field} must be {comparison}")
    return int(value)


def _positive_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number")
    if float(value) <= 0.0:
        raise ConfigError(f"{field} must be positive")
    return float(value)


def _optional_path(value: Any, field: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a filesystem path or null")
    return Path(value).expanduser()


@dataclass(frozen=True)
class TrainingConfig:
    """Explicit runtime settings shared by the published training recipes."""

    output_directory: Path
    total_steps: int
    batch_size: int
    num_workers: int
    seed: int
    device: str
    learning_rate: float | None
    audio2sph_learning_rate: float
    warmup_steps: int
    minimum_learning_rate_ratio: float
    ema_decay: float
    save_every_steps: int
    keep_every_steps: int
    log_every_steps: int
    validate_every_steps: int | None
    validation_batches: int
    selection_metric: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TrainingConfig":
        unknown = set(raw) - _TRAINING_FIELDS - _OPTIONAL_TRAINING_FIELDS
        missing = _TRAINING_FIELDS - set(raw)
        if unknown or missing:
            raise ConfigError(
                "training fields mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        output_directory = _optional_path(
            raw["output_directory"], "training.output_directory"
        )
        assert output_directory is not None
        device = raw["device"]
        if not isinstance(device, str) or not device.strip():
            raise ConfigError("training.device must be a device name or 'auto'")
        learning_rate_value = raw["learning_rate"]
        learning_rate = (
            None
            if learning_rate_value is None
            else _positive_float(learning_rate_value, "training.learning_rate")
        )
        minimum_ratio = _positive_float(
            raw["minimum_learning_rate_ratio"],
            "training.minimum_learning_rate_ratio",
        )
        if minimum_ratio > 1.0:
            raise ConfigError(
                "training.minimum_learning_rate_ratio must not exceed 1"
            )
        ema_decay_value = raw["ema_decay"]
        if isinstance(ema_decay_value, bool) or not isinstance(
            ema_decay_value, (int, float)
        ):
            raise ConfigError("training.ema_decay must be a number")
        ema_decay = float(ema_decay_value)
        if not 0.0 <= ema_decay < 1.0:
            raise ConfigError("training.ema_decay must lie in [0,1)")
        result = cls(
            output_directory=output_directory,
            total_steps=_positive_integer(
                raw["total_steps"], "training.total_steps"
            ),
            batch_size=_positive_integer(raw["batch_size"], "training.batch_size"),
            num_workers=_positive_integer(
                raw["num_workers"], "training.num_workers", allow_zero=True
            ),
            seed=_positive_integer(raw["seed"], "training.seed", allow_zero=True),
            device=device,
            learning_rate=learning_rate,
            audio2sph_learning_rate=_positive_float(
                raw["audio2sph_learning_rate"],
                "training.audio2sph_learning_rate",
            ),
            warmup_steps=_positive_integer(
                raw["warmup_steps"], "training.warmup_steps", allow_zero=True
            ),
            minimum_learning_rate_ratio=minimum_ratio,
            ema_decay=ema_decay,
            save_every_steps=_positive_integer(
                raw["save_every_steps"], "training.save_every_steps"
            ),
            keep_every_steps=_positive_integer(
                raw["keep_every_steps"], "training.keep_every_steps"
            ),
            log_every_steps=_positive_integer(
                raw["log_every_steps"], "training.log_every_steps"
            ),
            validate_every_steps=(
                None
                if raw.get("validate_every_steps") is None
                else _positive_integer(
                    raw["validate_every_steps"],
                    "training.validate_every_steps",
                )
            ),
            validation_batches=_positive_integer(
                raw.get("validation_batches", 0),
                "training.validation_batches",
                allow_zero=True,
            ),
            selection_metric=str(
                raw.get("selection_metric", "validation_loss")
            ),
        )
        if result.warmup_steps >= result.total_steps:
            raise ConfigError(
                "training.warmup_steps must be smaller than training.total_steps"
            )
        if result.keep_every_steps % result.save_every_steps != 0:
            raise ConfigError(
                "training.keep_every_steps must be a multiple of "
                "training.save_every_steps"
            )
        if result.selection_metric != "validation_loss":
            raise ConfigError(
                "training.selection_metric must be 'validation_loss'"
            )
        validation_enabled = result.validate_every_steps is not None
        if validation_enabled != (result.validation_batches > 0):
            raise ConfigError(
                "training.validate_every_steps and validation_batches must "
                "enable validation together"
            )
        if (
            result.validate_every_steps is not None
            and result.validate_every_steps % result.save_every_steps != 0
        ):
            raise ConfigError(
                "training.validate_every_steps must be a multiple of "
                "training.save_every_steps so best state is resumable"
            )
        return result

    def resolved(self, base: Path) -> "TrainingConfig":
        output = self.output_directory
        if not output.is_absolute():
            output = (base / output).resolve()
        return TrainingConfig(
            output_directory=output,
            total_steps=self.total_steps,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=self.seed,
            device=self.device,
            learning_rate=self.learning_rate,
            audio2sph_learning_rate=self.audio2sph_learning_rate,
            warmup_steps=self.warmup_steps,
            minimum_learning_rate_ratio=self.minimum_learning_rate_ratio,
            ema_decay=self.ema_decay,
            save_every_steps=self.save_every_steps,
            keep_every_steps=self.keep_every_steps,
            log_every_steps=self.log_every_steps,
            validate_every_steps=self.validate_every_steps,
            validation_batches=self.validation_batches,
            selection_metric=self.selection_metric,
        )


@dataclass(frozen=True)
class SAIDConfig:
    """Validated choices shared by inference, evaluation, and training."""

    schema: str
    preset: str
    model: str
    class_feature_encoder: str | None
    data: str
    data_config: Path
    load_said_ckpt: Path | None
    load_audio2sph_ckpt: Path | None
    load_class_feature_encoder_ckpt: Path | None
    training: TrainingConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SAIDConfig":
        unknown = set(raw) - _TOP_LEVEL_FIELDS
        missing = _TOP_LEVEL_FIELDS - set(raw)
        if unknown:
            raise ConfigError(f"unknown top-level fields: {sorted(unknown)}")
        if missing:
            raise ConfigError(f"missing top-level fields: {sorted(missing)}")

        schema = raw["schema"]
        if schema != "said-config-v1":
            raise ConfigError("schema must be 'said-config-v1'")

        preset = raw["preset"]
        if preset != "paper":
            raise ConfigError("preset must be 'paper'")

        model = raw["model"]
        if model not in {"said", "audio2sph"}:
            raise ConfigError("model must be 'said' or 'audio2sph'")

        class_feature_encoder = raw["class_feature_encoder"]
        if class_feature_encoder not in {"passt", "audiomae", None}:
            raise ConfigError(
                "class_feature_encoder must be 'passt', 'audiomae', or null"
            )

        data = raw["data"]
        if data not in {"dcase_recordings", "simulated_scenes"}:
            raise ConfigError(
                "data must be 'dcase_recordings' or 'simulated_scenes'"
            )

        data_config = raw["data_config"]
        if not isinstance(data_config, str) or not data_config.strip():
            raise ConfigError("data_config must be a filesystem path")

        load_said_ckpt = _optional_path(raw["load_said_ckpt"], "load_said_ckpt")
        load_audio2sph_ckpt = _optional_path(
            raw["load_audio2sph_ckpt"], "load_audio2sph_ckpt"
        )
        load_class_feature_encoder_ckpt = _optional_path(
            raw["load_class_feature_encoder_ckpt"],
            "load_class_feature_encoder_ckpt",
        )
        training_raw = raw["training"]
        if not isinstance(training_raw, Mapping):
            raise ConfigError("training must be a mapping")
        training = TrainingConfig.from_mapping(training_raw)

        if load_said_ckpt is not None and (
            load_audio2sph_ckpt is not None
            or load_class_feature_encoder_ckpt is not None
        ):
            raise ConfigError(
                "load_said_ckpt is mutually exclusive with component initialization"
            )

        if model == "audio2sph":
            if class_feature_encoder is not None:
                raise ConfigError(
                    "model='audio2sph' requires class_feature_encoder=null"
                )
            if load_said_ckpt is not None:
                raise ConfigError("model='audio2sph' requires load_said_ckpt=null")
            if load_class_feature_encoder_ckpt is not None:
                raise ConfigError(
                    "model='audio2sph' requires "
                    "load_class_feature_encoder_ckpt=null"
                )
            if data != "simulated_scenes":
                raise ConfigError(
                    "model='audio2sph' with preset='paper' requires "
                    "data='simulated_scenes'"
                )
        else:
            if class_feature_encoder is None:
                raise ConfigError(
                    "model='said' with preset='paper' requires "
                    "class_feature_encoder='passt' or 'audiomae'"
                )
            if load_said_ckpt is None and load_audio2sph_ckpt is None:
                raise ConfigError(
                    "model='said' requires load_said_ckpt or load_audio2sph_ckpt"
                )
            if data == "simulated_scenes":
                if load_audio2sph_ckpt is None:
                    raise ConfigError(
                        "SourceBank SAID training requires load_audio2sph_ckpt"
                    )
                if load_class_feature_encoder_ckpt is None:
                    raise ConfigError(
                        "SourceBank SAID training requires "
                        "load_class_feature_encoder_ckpt"
                    )
            if data == "dcase_recordings":
                if load_said_ckpt is None:
                    raise ConfigError(
                        "DCASE SAID fine-tuning requires load_said_ckpt"
                    )
                if load_class_feature_encoder_ckpt is not None:
                    raise ConfigError(
                        "DCASE SAID fine-tuning loads Class Features from the "
                        "complete SAID checkpoint"
                    )

        if load_said_ckpt is not None:
            canonical_names = {
                "passt": "said_passt.ckpt",
                "audiomae": "said_audiomae.ckpt",
            }
            canonical = canonical_names.get(class_feature_encoder)
            recognized = set(canonical_names.values())
            if load_said_ckpt.name in recognized and load_said_ckpt.name != canonical:
                raise ConfigError(
                    "load_said_ckpt filename is inconsistent with "
                    f"class_feature_encoder={class_feature_encoder!r}"
                )

        return cls(
            schema=schema,
            preset=preset,
            model=model,
            class_feature_encoder=class_feature_encoder,
            data=data,
            data_config=Path(data_config),
            load_said_ckpt=load_said_ckpt,
            load_audio2sph_ckpt=load_audio2sph_ckpt,
            load_class_feature_encoder_ckpt=load_class_feature_encoder_ckpt,
            training=training,
        )

    def resolved(self, config_path: str | Path) -> "SAIDConfig":
        """Resolve relative file fields against the top-level config directory."""

        base = Path(config_path).expanduser().resolve().parent

        def resolve(path: Path | None) -> Path | None:
            if path is None or path.is_absolute():
                return path
            return (base / path).resolve()

        return SAIDConfig(
            schema=self.schema,
            preset=self.preset,
            model=self.model,
            class_feature_encoder=self.class_feature_encoder,
            data=self.data,
            data_config=resolve(self.data_config),  # type: ignore[arg-type]
            load_said_ckpt=resolve(self.load_said_ckpt),
            load_audio2sph_ckpt=resolve(self.load_audio2sph_ckpt),
            load_class_feature_encoder_ckpt=resolve(
                self.load_class_feature_encoder_ckpt
            ),
            training=self.training.resolved(base),
        )

    def validate_for_command(self, command: str) -> None:
        """Validate model capabilities required by a public command."""

        if command not in {
            "infer",
            "evaluate",
            "prepare",
            "train",
        }:
            raise ConfigError(f"unknown command: {command}")
        if command in {"infer", "evaluate"}:
            if self.model != "said":
                raise ConfigError(f"said {command} requires model='said'")
            if self.load_said_ckpt is None:
                raise ConfigError(f"said {command} requires load_said_ckpt")
        if command in {"infer", "evaluate"} and self.data != "dcase_recordings":
            raise ConfigError(f"said {command} requires data='dcase_recordings'")
        if command == "prepare" and (
            self.model != "audio2sph" or self.data != "simulated_scenes"
        ):
            raise ConfigError(
                "said prepare requires model='audio2sph' and data='simulated_scenes'"
            )

    def validate_data_config(self) -> None:
        """Validate the selected data adapter and the published audio contract."""

        with self.data_config.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
        if not isinstance(raw, Mapping):
            raise ConfigError("data configuration root must be a mapping")
        required = {"schema", "dcase_recordings", "simulated_scenes", "audio"}
        unknown = set(raw) - required
        missing = required - set(raw)
        if unknown or missing:
            raise ConfigError(
                f"data configuration fields mismatch: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        if raw["schema"] != "said-data-config-v1":
            raise ConfigError("data schema must be 'said-data-config-v1'")
        if not isinstance(raw[self.data], Mapping):
            raise ConfigError(f"data section {self.data!r} must be a mapping")
        audio = raw["audio"]
        if not isinstance(audio, Mapping):
            raise ConfigError("audio data contract must be a mapping")
        if audio.get("array") != "eigenmike32":
            raise ConfigError("audio.array must be 'eigenmike32'")
        if audio.get("capsule_indices_1based") != [6, 10, 26, 22]:
            raise ConfigError("Eigenmike capsule_indices_1based must be [6, 10, 26, 22]")
        if audio.get("sample_rate") != 48000:
            raise ConfigError("audio.sample_rate must be 48000")
        if not isinstance(audio.get("segment_duration"), (int, float)) or audio["segment_duration"] <= 0:
            raise ConfigError("audio.segment_duration must be positive")


def load_config(path: str | Path, *, resolve_paths: bool = True) -> SAIDConfig:
    """Load and validate a SAID YAML configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be a mapping")
    config = SAIDConfig.from_mapping(raw)
    return config.resolved(config_path) if resolve_paths else config


def default_inference_config(
    class_feature_encoder: str,
    load_said_ckpt: str | Path,
) -> SAIDConfig:
    """Create the built-in paper-model configuration used by public inference."""

    if class_feature_encoder not in {"passt", "audiomae"}:
        raise ConfigError(
            "class_feature_encoder must be 'passt' or 'audiomae'"
        )
    return SAIDConfig(
        schema="said-config-v1",
        preset="paper",
        model="said",
        class_feature_encoder=class_feature_encoder,
        data="dcase_recordings",
        data_config=Path("configs/data.yaml"),
        load_said_ckpt=Path(load_said_ckpt).expanduser(),
        load_audio2sph_ckpt=None,
        load_class_feature_encoder_ckpt=None,
        training=TrainingConfig(
            output_directory=Path("runs/said_dcase_fine_tuning"),
            total_steps=500_000,
            batch_size=1,
            num_workers=8,
            seed=260_916,
            device="auto",
            learning_rate=1.5e-6,
            audio2sph_learning_rate=2.5e-6,
            warmup_steps=1_000,
            minimum_learning_rate_ratio=0.1,
            ema_decay=0.999,
            save_every_steps=10_000,
            keep_every_steps=100_000,
            log_every_steps=100,
            validate_every_steps=None,
            validation_batches=0,
            selection_metric="validation_loss",
        ),
    )
