"""License-aware public SourceBank manifest contract."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path

import soundfile as sf


SOURCEBANK_FIELDS = (
    "source_dataset",
    "source_id",
    "local_audio_path",
    "class_id",
    "start_seconds",
    "end_seconds",
    "license_spdx_or_uri",
    "attribution",
    "source_page",
    "redistribution_allowed",
    "commercial_use_allowed",
    "training_use_allowed",
)


def _strict_boolean(value: str, field: str, line: int) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} must be true or false at manifest line {line}")


@dataclass(frozen=True)
class SourceBankRecord:
    """One locally held source clip and its permission metadata."""

    source_dataset: str
    source_id: str
    audio_path: Path
    class_id: int
    start_seconds: float
    end_seconds: float
    license_spdx_or_uri: str
    attribution: str
    source_page: str
    redistribution_allowed: bool
    commercial_use_allowed: bool
    training_use_allowed: bool


def load_sourcebank_manifest(
    path: str | Path,
    *,
    audio_root: str | Path,
    require_commercial_use: bool = False,
) -> tuple[SourceBankRecord, ...]:
    """Read a SourceBank manifest with default-deny permission checks."""

    manifest_path = Path(path).expanduser().resolve()
    root = Path(audio_root).expanduser().resolve()
    delimiter = "\t" if manifest_path.suffix.lower() == ".tsv" else ","
    records: list[SourceBankRecord] = []
    seen: set[str] = set()
    audio_shapes: dict[Path, tuple[int, int]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        if reader.fieldnames is None or set(reader.fieldnames) != set(
            SOURCEBANK_FIELDS
        ):
            raise ValueError(
                "SourceBank manifest fields must be exactly "
                f"{list(SOURCEBANK_FIELDS)}"
            )
        for line, row in enumerate(reader, start=2):
            source_id = row["source_id"].strip()
            source_dataset = row["source_dataset"].strip()
            license_value = row["license_spdx_or_uri"].strip()
            attribution = row["attribution"].strip()
            source_page = row["source_page"].strip()
            if not all(
                (source_id, source_dataset, license_value, attribution, source_page)
            ):
                raise ValueError(
                    f"SourceBank identity and license fields must not be empty "
                    f"at manifest line {line}"
                )
            if source_id in seen:
                raise ValueError(f"duplicate SourceBank source_id {source_id!r}")
            seen.add(source_id)
            relative = Path(row["local_audio_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"local_audio_path must be relative at manifest line {line}"
                )
            audio_path = (root / relative).resolve()
            if not audio_path.is_relative_to(root):
                raise ValueError(
                    f"local_audio_path escapes source_audio_root at line {line}"
                )
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"SourceBank audio is missing at manifest line {line}: "
                    f"{audio_path}"
                )
            try:
                class_id = int(row["class_id"])
                start = float(row["start_seconds"])
                end = float(row["end_seconds"])
            except ValueError as error:
                raise ValueError(
                    f"invalid SourceBank class or time at manifest line {line}"
                ) from error
            if not 0 <= class_id < 13:
                raise ValueError(
                    f"class_id must lie in [0,12] at manifest line {line}"
                )
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError(
                    f"SourceBank time range must be finite at manifest line {line}"
                )
            if start < 0.0 or end <= start:
                raise ValueError(
                    f"SourceBank time range is invalid at manifest line {line}"
                )
            if audio_path not in audio_shapes:
                information = sf.info(str(audio_path))
                audio_shapes[audio_path] = (
                    int(information.frames),
                    int(information.samplerate),
                )
            audio_frames, audio_sample_rate = audio_shapes[audio_path]
            if audio_frames <= 0 or audio_sample_rate <= 0:
                raise ValueError(
                    f"SourceBank audio is empty or invalid at manifest line {line}"
                )
            authorized_start = int(math.ceil(start * audio_sample_rate))
            authorized_stop = int(math.ceil(end * audio_sample_rate))
            if authorized_stop <= authorized_start:
                raise ValueError(
                    "SourceBank time range contains no audio sample at "
                    f"manifest line {line}"
                )
            if (
                authorized_start >= audio_frames
                or authorized_stop > audio_frames
            ):
                duration = audio_frames / audio_sample_rate
                raise ValueError(
                    "SourceBank time range exceeds the audio duration at "
                    f"manifest line {line}: end={end}, duration={duration}"
                )
            redistribution = _strict_boolean(
                row["redistribution_allowed"],
                "redistribution_allowed",
                line,
            )
            commercial = _strict_boolean(
                row["commercial_use_allowed"],
                "commercial_use_allowed",
                line,
            )
            training = _strict_boolean(
                row["training_use_allowed"],
                "training_use_allowed",
                line,
            )
            if not training:
                raise PermissionError(
                    f"training use is not authorized for source_id={source_id!r}"
                )
            if require_commercial_use and not commercial:
                raise PermissionError(
                    f"commercial use is not authorized for source_id={source_id!r}"
                )
            records.append(
                SourceBankRecord(
                    source_dataset=source_dataset,
                    source_id=source_id,
                    audio_path=audio_path,
                    class_id=class_id,
                    start_seconds=start,
                    end_seconds=end,
                    license_spdx_or_uri=license_value,
                    attribution=attribution,
                    source_page=source_page,
                    redistribution_allowed=redistribution,
                    commercial_use_allowed=commercial,
                    training_use_allowed=training,
                )
            )
    if not records:
        raise RuntimeError(f"SourceBank manifest contains no authorized rows: {path}")
    return tuple(records)
