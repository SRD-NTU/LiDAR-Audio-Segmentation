"""Run full-recording DCASE development-test prediction and evaluation."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import nn

from ..data import discover_development_test
from ..inference import (
    EIGENMIKE_CAPSULES_1BASED,
    SAIDPredictor,
    load_eigenmike_audio,
)
from .metrics import (
    METRIC_FIELDS as DEFAULT_METRIC_FIELDS,
    PAPER_AP_FIELDS,
    ensure_official_evaluator,
    evaluate_dcase_metrics,
    official_evaluator_identity,
)
from .output2dcase import write_dcase_prediction_json

_MANIFEST_SCHEMA = "said-dcase2026-development-evaluation-v1"
_PROGRESS_SCHEMA = "said-dcase2026-development-evaluation-progress-v1"
_PROGRESS_FILENAME = "evaluation_progress.json"
_SOFTWARE_IDENTITY = {
    "package": "said-acoustic-imaging",
    "version": "1.0.0",
    "evaluation_protocol_revision": "paper-v1",
}


def _paper_protocol() -> dict[str, Any]:
    return {
        "sample_rate": 48_000,
        "output_fps": 10,
        "segment_duration_seconds": 2.0,
        "internal_positions_per_segment": 21,
        "retained_positions_per_segment": 20,
        "frame_count_source": "max label metadata_frame_index plus one",
        "input_audio_scope": "complete recording",
        "output_frame_limit": "label frame count",
        "eigenmike_capsule_indices_1based": list(EIGENMIKE_CAPSULES_1BASED),
        "confidence_threshold": 0.05,
        "max_sources_per_frame": 4,
        "relative_map_threshold": 0.10,
        "energy_top_k": 0,
        "map_coordinates": {
            "shape": [180, 360],
            "export_x": "(179 - sph_x) mod 360",
            "export_y": "sph_y",
        },
    }


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dataset_fingerprint(records: list[dict[str, Any]]) -> str:
    identity = [
        {
            "subset": record["subset"],
            "name": record["name"],
            "recording_domain": record["recording_domain"],
            "label_frames": record["label_frames"],
            "audio_bytes": record["audio_bytes"],
            "audio_sha256": record["audio_sha256"],
            "label_sha256": record["label_sha256"],
        }
        for record in records
    ]
    serialized = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(serialized).hexdigest()


def _sequence_record(
    *,
    index: int,
    sequence: Any,
    dataset_root: Path,
    output_directory: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    """Build the provenance record for one atomically completed prediction."""

    return {
        "index": index,
        "subset": sequence.subset,
        "name": sequence.name,
        "recording_domain": sequence.recording_domain.value,
        "label_frames": sequence.frame_count,
        "audio": str(sequence.audio_path.relative_to(dataset_root)),
        "audio_bytes": sequence.audio_path.stat().st_size,
        "audio_sha256": _sha256(sequence.audio_path),
        "label": str(sequence.label_path.relative_to(dataset_root)),
        "label_sha256": _sha256(sequence.label_path),
        "prediction": str(prediction_path.relative_to(output_directory)),
        "prediction_sha256": _sha256(prediction_path),
        "prediction_bytes": prediction_path.stat().st_size,
    }


def _progress_document(
    *,
    model_name: str,
    checkpoint_sha256: str,
    class_feature_encoder: str,
    dataset_root: Path,
    sequences: tuple[Any, ...],
    protocol: dict[str, Any],
    records: list[dict[str, Any]] | None = None,
    active_recording: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed = [] if records is None else records
    return {
        "schema": _PROGRESS_SCHEMA,
        "software": _SOFTWARE_IDENTITY,
        "model": model_name,
        "checkpoint_sha256": checkpoint_sha256,
        "class_feature_encoder": class_feature_encoder,
        "dataset_root": str(dataset_root),
        "split": "development-test",
        "sequences": len(sequences),
        "subsets": {
            "dev-test-sony": sum(
                sequence.subset == "dev-test-sony" for sequence in sequences
            ),
            "dev-test-tau": sum(
                sequence.subset == "dev-test-tau" for sequence in sequences
            ),
        },
        "protocol": protocol,
        "recordings": completed,
        "active_recording": active_recording,
    }


def _read_progress(path: Path) -> dict[str, Any]:
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read evaluation progress record: {path}") from error
    if not isinstance(progress, dict):
        raise ValueError(f"evaluation progress record must be an object: {path}")
    return progress


def _validate_record(
    record: Any,
    *,
    expected_index: int,
    sequence: Any,
    dataset_root: Path,
    output_directory: Path,
) -> None:
    if not isinstance(record, dict):
        raise ValueError("evaluation progress contains a non-object recording entry")
    expected_prediction = (
        Path("inference_outputs") / f"{sequence.name}_inference.json"
    )
    expected_metadata = {
        "index": expected_index,
        "subset": sequence.subset,
        "name": sequence.name,
        "recording_domain": sequence.recording_domain.value,
        "label_frames": sequence.frame_count,
        "audio": str(sequence.audio_path.relative_to(dataset_root)),
        "audio_bytes": sequence.audio_path.stat().st_size,
        "audio_sha256": _sha256(sequence.audio_path),
        "label": str(sequence.label_path.relative_to(dataset_root)),
        "label_sha256": _sha256(sequence.label_path),
        "prediction": str(expected_prediction),
    }
    for field, expected in expected_metadata.items():
        if record.get(field) != expected:
            raise ValueError(
                f"evaluation progress {sequence.subset}/{sequence.name} "
                f"has mismatched {field}"
            )
    prediction = (output_directory / expected_prediction).resolve()
    if not prediction.is_relative_to(output_directory) or not prediction.is_file():
        raise ValueError(f"evaluation prediction is missing: {expected_prediction}")
    if record.get("prediction_bytes") != prediction.stat().st_size:
        raise ValueError(f"evaluation prediction size changed: {expected_prediction}")
    if record.get("prediction_sha256") != _sha256(prediction):
        raise ValueError(f"evaluation prediction hash changed: {expected_prediction}")


def _validate_progress(
    progress: dict[str, Any],
    *,
    model_name: str,
    checkpoint_sha256: str,
    class_feature_encoder: str,
    dataset_root: Path,
    output_directory: Path,
    sequences: tuple[Any, ...],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_fields = {
        "schema": _PROGRESS_SCHEMA,
        "software": _SOFTWARE_IDENTITY,
        "model": model_name,
        "checkpoint_sha256": checkpoint_sha256,
        "class_feature_encoder": class_feature_encoder,
        "dataset_root": str(dataset_root),
        "split": "development-test",
        "sequences": len(sequences),
        "subsets": {
            "dev-test-sony": sum(
                sequence.subset == "dev-test-sony" for sequence in sequences
            ),
            "dev-test-tau": sum(
                sequence.subset == "dev-test-tau" for sequence in sequences
            ),
        },
        "protocol": protocol,
    }
    for field, expected in expected_fields.items():
        if progress.get(field) != expected:
            raise ValueError(
                f"existing evaluation progress uses a different {field}"
            )
    records = progress.get("recordings")
    if not isinstance(records, list) or len(records) > len(sequences):
        raise ValueError("evaluation progress has an invalid recording list")
    for index, record in enumerate(records, start=1):
        _validate_record(
            record,
            expected_index=index,
            sequence=sequences[index - 1],
            dataset_root=dataset_root,
            output_directory=output_directory,
        )
    expected_files = {
        f"{sequence.name}_inference.json" for sequence in sequences[: len(records)]
    }
    actual_files = {
        path.name
        for path in (output_directory / "inference_outputs").glob(
            "*_inference.json"
        )
    }
    active = progress.get("active_recording")
    if active is not None:
        if len(records) >= len(sequences) or not isinstance(active, dict):
            raise ValueError("evaluation progress has an invalid active recording")
        next_sequence = sequences[len(records)]
        expected_active = {
            "index": len(records) + 1,
            "subset": next_sequence.subset,
            "name": next_sequence.name,
        }
        if active != expected_active:
            raise ValueError("evaluation progress has a mismatched active recording")
        active_name = f"{next_sequence.name}_inference.json"
        if active_name in actual_files:
            prediction_path = (
                output_directory / "inference_outputs" / active_name
            )
            records.append(
                _sequence_record(
                    index=len(records) + 1,
                    sequence=next_sequence,
                    dataset_root=dataset_root,
                    output_directory=output_directory,
                    prediction_path=prediction_path,
                )
            )
            expected_files.add(active_name)
        progress["active_recording"] = None
    if actual_files != expected_files:
        unexpected = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        raise ValueError(
            "evaluation predictions do not match the progress record: "
            f"unexpected={unexpected}, missing={missing}"
        )
    return records


def validate_completed_inference(
    manifest_path: str | Path,
    *,
    checkpoint_sha256: str,
    class_feature_encoder: str,
    dataset_root: str | Path,
) -> Path:
    """Validate a completed JSON-export manifest before evaluator-only reuse."""

    path = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read evaluation manifest: {path}") from error
    if manifest.get("schema") != _MANIFEST_SCHEMA:
        raise ValueError(f"unsupported evaluation manifest schema: {path}")
    if manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("existing evaluation uses a different checkpoint")
    if manifest.get("class_feature_encoder") != class_feature_encoder:
        raise ValueError("existing evaluation uses a different Class Feature Encoder")
    if manifest.get("software") != _SOFTWARE_IDENTITY:
        raise ValueError("existing evaluation uses a different software revision")
    if manifest.get("protocol") != _paper_protocol():
        raise ValueError("existing evaluation uses a different evaluation protocol")
    if manifest.get("subsets") != {"dev-test-sony": 30, "dev-test-tau": 48}:
        raise ValueError("existing evaluation does not contain 30 Sony and 48 TAU recordings")
    records = manifest.get("recordings")
    if manifest.get("sequences") != 78 or not isinstance(records, list) or len(records) != 78:
        raise ValueError("existing evaluation does not contain the 78-recording paper split")
    root = path.parent
    dataset = Path(dataset_root).expanduser().resolve()
    sequences = discover_development_test(dataset)
    current = {(sequence.subset, sequence.name): sequence for sequence in sequences}
    seen: set[tuple[str, str]] = set()
    verified_records: list[dict[str, Any]] = []
    for expected_index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError("existing evaluation contains a non-object recording entry")
        key = (record.get("subset"), record.get("name"))
        if not all(isinstance(item, str) for item in key) or key not in current:
            raise ValueError(f"existing evaluation contains an unknown recording: {key}")
        typed_key = (str(key[0]), str(key[1]))
        if typed_key in seen:
            raise ValueError(f"existing evaluation contains duplicate recording: {typed_key}")
        seen.add(typed_key)
        sequence = current[typed_key]
        if record.get("index") != expected_index:
            raise ValueError("existing evaluation recording indices are not contiguous")
        expected_metadata = {
            "recording_domain": sequence.recording_domain.value,
            "label_frames": sequence.frame_count,
            "audio": str(sequence.audio_path.relative_to(dataset)),
            "audio_bytes": sequence.audio_path.stat().st_size,
            "audio_sha256": _sha256(sequence.audio_path),
            "label": str(sequence.label_path.relative_to(dataset)),
            "label_sha256": _sha256(sequence.label_path),
        }
        for field, expected in expected_metadata.items():
            if record.get(field) != expected:
                raise ValueError(
                    f"existing evaluation {typed_key} has mismatched {field}"
                )
        prediction_value = record.get("prediction")
        if not isinstance(prediction_value, str):
            raise ValueError(f"existing evaluation {typed_key} has no prediction path")
        prediction = (root / prediction_value).resolve()
        if not prediction.is_relative_to(root) or not prediction.is_file():
            raise ValueError(f"existing evaluation prediction is missing: {prediction_value}")
        if record.get("prediction_bytes") != prediction.stat().st_size:
            raise ValueError(f"existing evaluation prediction size changed: {prediction_value}")
        if record.get("prediction_sha256") != _sha256(prediction):
            raise ValueError(f"existing evaluation prediction hash changed: {prediction_value}")
        verified_records.append(record)
    if seen != set(current):
        raise ValueError("existing evaluation recording identities are incomplete")
    if manifest.get("dataset_fingerprint_sha256") != _dataset_fingerprint(
        verified_records
    ):
        raise ValueError("existing evaluation dataset fingerprint does not match")
    return path


def run_development_test_inference(
    model: nn.Module,
    dataset_root: str | Path,
    output_directory: str | Path,
    *,
    device: str | torch.device,
    batch_size: int,
    model_name: str,
    checkpoint_sha256: str,
    class_feature_encoder: str,
    sample_rate: int = 48_000,
    output_fps: int = 10,
    confidence_threshold: float = 0.05,
    max_sources_per_frame: int = 4,
    relative_map_threshold: float = 0.10,
) -> Path:
    """Infer the paper's 78 full development-test recordings and export JSON."""

    protocol = _paper_protocol()
    requested = {
        "sample_rate": sample_rate,
        "output_fps": output_fps,
        "confidence_threshold": confidence_threshold,
        "max_sources_per_frame": max_sources_per_frame,
        "relative_map_threshold": relative_map_threshold,
    }
    mismatched = {
        key: value
        for key, value in requested.items()
        if value != protocol[key]
    }
    if mismatched:
        raise ValueError(
            f"paper evaluation protocol parameters differ: {mismatched}"
        )
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    prediction_directory = output / "inference_outputs"
    sequences = discover_development_test(root)
    progress_path = output / _PROGRESS_FILENAME
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(
                f"evaluation output path is not a directory: {output}"
            )
        if not prediction_directory.is_dir() or not progress_path.is_file():
            raise FileExistsError(
                "evaluation output exists without a resumable progress record: "
                f"{output}"
            )
        progress = _read_progress(progress_path)
        records = _validate_progress(
            progress,
            model_name=model_name,
            checkpoint_sha256=checkpoint_sha256,
            class_feature_encoder=class_feature_encoder,
            dataset_root=root,
            output_directory=output,
            sequences=sequences,
            protocol=protocol,
        )
        progress["recordings"] = records
        _write_json_atomic(progress_path, progress)
        print(
            f"[resume] verified {len(records)}/{len(sequences)} completed recordings",
            flush=True,
        )
    else:
        prediction_directory.mkdir(parents=True)
        records = []
        progress = _progress_document(
            model_name=model_name,
            checkpoint_sha256=checkpoint_sha256,
            class_feature_encoder=class_feature_encoder,
            dataset_root=root,
            sequences=sequences,
            protocol=protocol,
        )
        _write_json_atomic(progress_path, progress)

    predictor = SAIDPredictor(
        model,
        device=device,
        sample_rate=sample_rate,
        output_fps=output_fps,
        batch_size=batch_size,
    )
    for index, sequence in enumerate(
        sequences[len(records) :], start=len(records) + 1
    ):
        progress["active_recording"] = {
            "index": index,
            "subset": sequence.subset,
            "name": sequence.name,
        }
        _write_json_atomic(progress_path, progress)
        audio = load_eigenmike_audio(
            sequence.audio_path,
            sample_rate=sample_rate,
        )
        required_samples = sequence.frame_count * sample_rate // output_fps
        if int(audio.shape[-1]) < required_samples:
            raise ValueError(
                f"recording {sequence.audio_path} is shorter than its labels"
            )
        prediction_path = prediction_directory / f"{sequence.name}_inference.json"
        write_dcase_prediction_json(
            predictor.predict(
                audio,
                recording_domain=sequence.recording_domain,
                output_frame_limit=sequence.frame_count,
            ),
            prediction_path,
            confidence_threshold=confidence_threshold,
            max_sources_per_frame=max_sources_per_frame,
            relative_map_threshold=relative_map_threshold,
        )
        records.append(
            _sequence_record(
                index=index,
                sequence=sequence,
                dataset_root=root,
                output_directory=output,
                prediction_path=prediction_path,
            )
        )
        progress["recordings"] = records
        progress["active_recording"] = None
        _write_json_atomic(progress_path, progress)
        print(
            f"[{index:02d}/{len(sequences)}] {sequence.subset}/{sequence.name} "
            f"frames={sequence.frame_count}",
            flush=True,
        )
    manifest = {
        "schema": _MANIFEST_SCHEMA,
        "software": _SOFTWARE_IDENTITY,
        "model": model_name,
        "checkpoint_sha256": checkpoint_sha256,
        "class_feature_encoder": class_feature_encoder,
        "dataset_root": str(root),
        "split": "development-test",
        "sequences": len(records),
        "subsets": {
            "dev-test-sony": sum(record["subset"] == "dev-test-sony" for record in records),
            "dev-test-tau": sum(record["subset"] == "dev-test-tau" for record in records),
        },
        "protocol": protocol,
        "recordings": records,
    }
    manifest["dataset_fingerprint_sha256"] = _dataset_fingerprint(records)
    manifest_path = output / "evaluation_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    progress_path.unlink()
    return manifest_path


def run_metrics(
    *,
    evaluation_directory: str | Path,
    dataset_root: str | Path,
    prediction_directory: str | Path | None = None,
    paper_ap: bool = False,
) -> Path:
    """Compute current official metrics and optional paper COCO APs."""

    evaluation = Path(evaluation_directory).expanduser().resolve()
    evaluation.mkdir(parents=True, exist_ok=True)
    predictions = (
        evaluation / "inference_outputs"
        if prediction_directory is None
        else Path(prediction_directory).expanduser().resolve()
    )
    if not predictions.is_dir():
        raise FileNotFoundError(f"prediction directory not found: {predictions}")
    metrics_path = evaluation / "metrics.json"
    official = ensure_official_evaluator()
    metrics = evaluate_dcase_metrics(
        predictions,
        Path(dataset_root).expanduser().resolve() / "labels_dev",
        split="dev-test",
        official_script=official,
        include_paper_ap=paper_ap,
    )
    _write_json_atomic(metrics_path, metrics)
    evaluator_record = {
        "official_evaluator": official_evaluator_identity(
            official, include_paper_ap=paper_ap
        ),
        "predictions": str(predictions),
        "paper_ap_enabled": bool(paper_ap),
        "metrics": dict(metrics),
    }
    _write_json_atomic(evaluation / "evaluator.json", evaluator_record)
    return metrics_path
