"""Download the official sources used by the frozen SAID paper metrics."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .class_f1 import (
    matched_class_macro_f1,
    prediction_files,
    selected_ground_truth_files,
)


OFFICIAL_EVALUATOR_REPOSITORY = (
    "https://github.com/iranroman/DCASE2026_Task3_SAISELD_baseline"
)
OFFICIAL_MASK_EVALUATOR_COMMIT = "84b2cd1"
OFFICIAL_EVALUATOR_URL = (
    "https://raw.githubusercontent.com/iranroman/"
    "DCASE2026_Task3_SAISELD_baseline/"
    "main/evaluate.py"
)
OFFICIAL_MASK_EVALUATOR_URL = (
    "https://raw.githubusercontent.com/iranroman/"
    "DCASE2026_Task3_SAISELD_baseline/"
    f"{OFFICIAL_MASK_EVALUATOR_COMMIT}/evaluate.py"
)
METRIC_FIELDS = (
    "official_macro_map",
    "official_macro_pearson_r",
    "official_class_agnostic_ap",
    "matched_class_macro_f1",
)
PAPER_AP_FIELDS = (
    "paper_coco_macro_map",
    "paper_coco_mask_ap",
)
_MAXIMUM_DOWNLOAD_BYTES = 5 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def official_evaluator_path() -> Path:
    """Return the package-local path for the official Pearson evaluator."""

    return Path(__file__).resolve().parent / "official" / "evaluate.py"


def official_mask_evaluator_path() -> Path:
    """Return the package-local path for the official paper Mask mAP source."""

    return (
        Path(__file__).resolve().parent
        / "official"
        / "evaluate_mask_coco_84b2cd1.py"
    )


def _ensure_download(
    *,
    path: Path,
    source: str,
    label: str,
) -> Path:
    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".evaluate.", suffix=".download", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        downloaded = 0
        with os.fdopen(descriptor, "wb") as stream:
            request = Request(
                source,
                headers={"User-Agent": "SAID/1.0 DCASE2026 evaluator acquisition"},
            )
            try:
                with urlopen(request, timeout=60) as response:
                    while block := response.read(1024 * 1024):
                        downloaded += len(block)
                        if downloaded > _MAXIMUM_DOWNLOAD_BYTES:
                            raise RuntimeError(
                                f"official {label} evaluator download is unexpectedly large"
                            )
                        stream.write(block)
            except (HTTPError, URLError, TimeoutError) as error:
                raise RuntimeError(
                    f"could not download the official DCASE {label} evaluator. "
                    "The first evaluation requires network access; later runs "
                    "reuse the local copy."
                ) from error
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(
        f"Downloaded official DCASE {label} evaluator to {path}.",
        flush=True,
    )
    return path


def ensure_official_evaluator(destination: str | Path | None = None) -> Path:
    """Download the pinned official Macro Pearson evaluator when absent."""

    path = (
        official_evaluator_path()
        if destination is None
        else Path(destination).expanduser().resolve()
    )
    return _ensure_download(
        path=path,
        source=OFFICIAL_EVALUATOR_URL,
        label="Macro Pearson",
    )


def ensure_official_mask_evaluator(
    destination: str | Path | None = None,
) -> Path:
    """Download the official COCO Mask mAP revision used by the paper."""

    path = (
        official_mask_evaluator_path()
        if destination is None
        else Path(destination).expanduser().resolve()
    )
    return _ensure_download(
        path=path,
        source=OFFICIAL_MASK_EVALUATOR_URL,
        label="COCO Mask mAP",
    )


def metric_worker_count() -> int:
    """Use half of the machine's logical CPUs for metric computation."""

    return max(1, (os.cpu_count() or 1) // 2)


def _run_paper_metrics(
    *,
    mask_script: Path,
    pearson_script: Path,
    predictions: Path,
    ground_truth: Path,
    split: str,
    output_json: Path,
    workers: int,
) -> Mapping[str, Any]:
    command = [
        sys.executable,
        "-m",
        "said.evaluation.metrics._paper_metrics_runner",
        "--mask-script",
        str(mask_script),
        "--pearson-script",
        str(pearson_script),
        "--pred-dir",
        str(predictions),
        "--gt-dir",
        str(ground_truth),
        "--split",
        split,
        "--workers",
        str(workers),
        "--output-json",
        str(output_json),
    ]
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    result = subprocess.run(command, check=False, env=environment)
    if result.returncode != 0:
        raise RuntimeError(
            "frozen SAID paper metric evaluation failed with exit code "
            f"{result.returncode}"
        )
    if not output_json.is_file():
        raise RuntimeError("paper metric runner did not produce its output JSON")
    with output_json.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)
    if not isinstance(metrics, Mapping):
        raise TypeError("paper metric output JSON must contain an object")
    return metrics


def _run_official(
    script: Path,
    *,
    predictions: Path,
    ground_truth: Path,
    frames: Path,
    output: Path,
    workers: int,
    stage: str,
    collapse_classes: bool = False,
) -> Mapping[str, Any]:
    output.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "said.evaluation.metrics._official_parallel_runner",
        str(script),
        "--pred-dir",
        str(predictions),
        "--gt-dir",
        str(ground_truth),
        "--frames-base",
        str(frames),
        "--output-dir",
        str(output),
        "--workers",
        str(workers),
        "--stage",
        stage,
    ]
    if collapse_classes:
        command.append("--collapse-classes")
    command.append("--quiet-results")
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    result = subprocess.run(command, check=False, env=environment)
    if result.returncode != 0:
        raise RuntimeError(
            f"official DCASE evaluator failed during {stage} "
            f"with exit code {result.returncode}"
        )
    metrics_path = output / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError("official DCASE evaluator did not produce metrics.json")
    with metrics_path.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)
    if not isinstance(metrics, Mapping):
        raise TypeError("official DCASE metrics.json must contain an object")
    return metrics


def _validated_metrics(
    value: Mapping[str, Any],
    *,
    include_paper_ap: bool,
) -> dict[str, float]:
    required = METRIC_FIELDS + (PAPER_AP_FIELDS if include_paper_ap else ())
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"metric evaluation omitted fields: {missing}")
    metrics: dict[str, float] = {}
    for field in required:
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"metric must be numeric: {field}")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError(f"metric must be finite: {field}")
        metrics[field] = numeric
    return metrics


def evaluate_dcase_metrics(
    predictions: str | Path,
    ground_truth: str | Path,
    *,
    split: str = "dev-test",
    official_script: str | Path | None = None,
    official_mask_script: str | Path | None = None,
    include_paper_ap: bool = False,
) -> dict[str, float]:
    """Evaluate current official metrics and optionally the paper's COCO APs."""

    prediction_root = Path(predictions).expanduser().resolve()
    ground_truth_root = Path(ground_truth).expanduser().resolve()
    current_script = (
        ensure_official_evaluator()
        if official_script is None
        else Path(official_script).expanduser().resolve()
    )
    if not current_script.is_file():
        raise FileNotFoundError(
            f"current official DCASE evaluator not found: {current_script}"
        )
    files = prediction_files(prediction_root)
    selected_ground_truth_files(files, ground_truth_root, split=split)
    workers = metric_worker_count()
    print(
        f"[metrics] CPU workers: {workers}/{os.cpu_count() or 1} "
        "(50% of logical CPUs)",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="said-dcase-metrics-") as temporary:
        workspace = Path(temporary)
        print("[metrics] stage 1/3: (6.30) class-aware", flush=True)
        aware = _run_official(
            current_script,
            predictions=prediction_root,
            ground_truth=ground_truth_root,
            frames=ground_truth_root.parent / "frames_dev",
            output=workspace / "current_class_aware",
            workers=workers,
            stage="(6.30) class-aware",
        )
        print("[metrics] stage 2/3: (6.30) class-agnostic", flush=True)
        agnostic = _run_official(
            current_script,
            predictions=prediction_root,
            ground_truth=ground_truth_root,
            frames=ground_truth_root.parent / "frames_dev",
            output=workspace / "current_class_agnostic",
            workers=workers,
            stage="(6.30) class-agnostic",
            collapse_classes=True,
        )
        print("[metrics] stage 3/3: Macro Class-F1", flush=True)
        class_f1 = matched_class_macro_f1(
            prediction_root,
            ground_truth_root,
            split=split,
            workers=workers,
            show_progress=True,
        )
        metrics: dict[str, Any] = {
            "official_macro_map": aware["macro_mAP"],
            "official_macro_pearson_r": aware["macro_pearson_r"],
            "official_class_agnostic_ap": agnostic["macro_mAP"],
            "matched_class_macro_f1": class_f1,
        }
        if include_paper_ap:
            mask_script = (
                ensure_official_mask_evaluator()
                if official_mask_script is None
                else Path(official_mask_script).expanduser().resolve()
            )
            if not mask_script.is_file():
                raise FileNotFoundError(
                    f"paper COCO Mask AP evaluator not found: {mask_script}"
                )
            print(
                "[metrics] optional: (4.2) paper COCO APs",
                flush=True,
            )
            paper = _run_paper_metrics(
                mask_script=mask_script,
                pearson_script=current_script,
                predictions=prediction_root,
                ground_truth=ground_truth_root,
                split=split,
                output_json=workspace / "paper_metrics.json",
                workers=workers,
            )
            metrics.update(
                paper_coco_macro_map=paper["official_macro_map"],
                paper_coco_mask_ap=paper["class_agnostic_mask_ap"],
            )

    validated = _validated_metrics(
        metrics, include_paper_ap=include_paper_ap
    )
    print("\nEvaluation metrics", flush=True)
    print(f"  (6.30) Macro mAP                 {validated['official_macro_map']:.9f}")
    print(
        "  (6.30) Macro Pearson r           "
        f"{validated['official_macro_pearson_r']:.9f}"
    )
    print(
        "  (6.30) Class-agnostic AP         "
        f"{validated['official_class_agnostic_ap']:.9f}"
    )
    print(
        "  Macro Class-F1                   "
        f"{validated['matched_class_macro_f1']:.9f}"
    )
    if include_paper_ap:
        print(f"  (4.2) Paper Macro mAP             {validated['paper_coco_macro_map']:.9f}")
        print(f"  (4.2) Paper Mask AP               {validated['paper_coco_mask_ap']:.9f}")
    return validated


def official_evaluator_identity(
    path: str | Path,
    *,
    include_paper_ap: bool = False,
) -> dict[str, Any]:
    """Return provenance for the current evaluator and optional paper AP."""

    evaluator = Path(path).expanduser().resolve()
    identity: dict[str, Any] = {
        "repository": OFFICIAL_EVALUATOR_REPOSITORY,
        "source": OFFICIAL_EVALUATOR_URL,
        "revision": "main",
        "sha256": _sha256(evaluator),
        "local_path": str(evaluator),
        "role": "current (6.30) Macro mAP and Macro Pearson r",
    }
    mask_evaluator = official_mask_evaluator_path()
    if include_paper_ap and mask_evaluator.is_file():
        identity["paper_ap_evaluator"] = {
            "source": OFFICIAL_MASK_EVALUATOR_URL,
            "commit": OFFICIAL_MASK_EVALUATOR_COMMIT,
            "sha256": _sha256(mask_evaluator),
            "local_path": str(mask_evaluator),
            "role": "paper (4.2) COCO Macro mAP and Mask AP",
        }
    return identity
