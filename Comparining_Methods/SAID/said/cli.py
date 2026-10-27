"""Command-line interface for the SAID release."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence

import yaml

from . import __version__
from .utils.load_config import (
    ConfigError,
    SAIDConfig,
    default_inference_config,
    load_config,
)

_COMPLETE_SAID_MODELS = {
    "said_passt": "passt",
    "said_audiomae": "audiomae",
}


def _class_feature_encoder_for_model(model: str) -> str:
    try:
        return _COMPLETE_SAID_MODELS[model]
    except KeyError as error:
        raise ConfigError(
            "complete SAID model must be 'said_passt' or 'said_audiomae'"
        ) from error


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Top-level SAID YAML configuration (default: paper model settings).",
    )


def _add_checkpoint_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Local published checkpoint (overrides the configured path).",
    )


def _read_config(
    path: Path | None,
    model: str | None = None,
    checkpoint: Path | None = None,
) -> SAIDConfig:
    selected_encoder = (
        None if model is None else _class_feature_encoder_for_model(model)
    )
    if path is None:
        from .utils.download_checkpoints import checkpoint_cache_path

        selected = selected_encoder or "passt"
        config = default_inference_config(
            selected,
            checkpoint_cache_path(f"said_{selected}"),
        )
    else:
        config = load_config(path)
    if (
        path is not None
        and selected_encoder is not None
        and selected_encoder != config.class_feature_encoder
    ):
        if config.model != "said":
            raise ConfigError("--model selects a complete SAID checkpoint")
        from .utils.download_checkpoints import checkpoint_cache_path

        config = replace(
            config,
            class_feature_encoder=selected_encoder,
            load_said_ckpt=checkpoint_cache_path(
                f"said_{selected_encoder}"
            ),
            load_audio2sph_ckpt=None,
            load_class_feature_encoder_ckpt=None,
        )
    if checkpoint is not None:
        local_checkpoint = checkpoint.expanduser().resolve()
        if not local_checkpoint.is_file():
            raise FileNotFoundError(
                f"local SAID checkpoint not found: {local_checkpoint}"
            )
        config = replace(config, load_said_ckpt=local_checkpoint)
    return config


def _ensure_config_checkpoint(config: SAIDConfig) -> Path:
    if config.class_feature_encoder is None or config.load_said_ckpt is None:
        raise ConfigError("a complete SAID checkpoint is required")
    checkpoint = config.load_said_ckpt.expanduser().resolve()
    if checkpoint.is_file():
        print(f"Using checkpoint: {checkpoint}", file=sys.stderr, flush=True)
        return checkpoint
    from .utils.download_checkpoints import (
        CHECKPOINT_ASSETS,
        ensure_paper_checkpoint,
    )

    model_id = f"said_{config.class_feature_encoder}"
    asset = CHECKPOINT_ASSETS[model_id]
    if checkpoint.name != asset.filename:
        raise FileNotFoundError(
            f"custom SAID checkpoint not found: {checkpoint}"
        )
    checkpoint = ensure_paper_checkpoint(
        model_id,
        destination=checkpoint,
    )
    print(f"Using checkpoint: {checkpoint}", file=sys.stderr, flush=True)
    return checkpoint


def _ensure_audio2sph_checkpoint(checkpoint_override: Path | None) -> Path:
    from .utils.download_checkpoints import (
        checkpoint_cache_path,
        ensure_paper_checkpoint,
    )

    if checkpoint_override is not None:
        checkpoint = checkpoint_override.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"local Audio2Sph checkpoint not found: {checkpoint}"
            )
    else:
        checkpoint = checkpoint_cache_path("audio2sph")
        if not checkpoint.is_file():
            checkpoint = ensure_paper_checkpoint(
                "audio2sph", destination=checkpoint
            )
    print(f"Using checkpoint: {checkpoint}", file=sys.stderr, flush=True)
    return checkpoint


def _run_infer(args: argparse.Namespace) -> int:
    output_format = getattr(args, "output_format", None)
    visual = getattr(args, "visual", None)
    if visual is not None:
        visual = visual.expanduser().resolve()
        if not visual.is_file():
            raise FileNotFoundError(f"visual input not found: {visual}")
    if getattr(args, "model", None) == "audio2sph":
        if args.config is not None:
            raise ConfigError(
                "--model audio2sph selects the published Audio2Sph checkpoint "
                "directly; omit --config"
            )
        checkpoint = _ensure_audio2sph_checkpoint(
            getattr(args, "checkpoint", None)
        )
        from .inference import run_audio2sph_inference

        manifest = run_audio2sph_inference(
            args.recording,
            args.output,
            checkpoint=checkpoint,
            device=args.device,
            batch_size=args.batch_size,
            output_format=output_format or "json",
            visual=visual,
        )
        print(manifest)
        return 0

    config = _read_config(
        args.config,
        getattr(args, "model", None),
        getattr(args, "checkpoint", None),
    )
    config.validate_for_command("infer")
    if config.class_feature_encoder is None or config.load_said_ckpt is None:
        raise ConfigError("said infer requires a complete SAID checkpoint")
    checkpoint = _ensure_config_checkpoint(config)

    from .inference import run_inference

    manifest = run_inference(
        args.recording,
        args.output,
        checkpoint=checkpoint,
        class_feature_encoder=config.class_feature_encoder,
        recording_domain=args.recording_domain,
        device=args.device,
        batch_size=args.batch_size,
        score_threshold=args.score_threshold,
        max_sources_per_frame=args.max_sources_per_frame,
        output_format=output_format or "json",
        visual=visual,
    )
    print(manifest)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    config = _read_config(
        args.config,
        getattr(args, "model", None),
        getattr(args, "checkpoint", None),
    )
    config.validate_for_command("evaluate")
    if config.class_feature_encoder is None or config.load_said_ckpt is None:
        raise ConfigError("said evaluate requires a complete SAID checkpoint")

    from .evaluation import (
        run_development_test_inference,
        run_metrics,
        validate_completed_inference,
    )
    from .models import PAPER_CHECKPOINTS, checkpoint_sha256

    predictions_only = bool(getattr(args, "predictions_only", False))
    supplied_predictions = getattr(args, "predictions", None)
    if predictions_only and supplied_predictions is not None:
        raise ConfigError(
            "--predictions-only cannot be combined with --predictions"
        )
    if supplied_predictions is not None:
        prediction_directory = Path(supplied_predictions).expanduser().resolve()
        if not prediction_directory.is_dir():
            raise FileNotFoundError(
                f"prediction directory not found: {prediction_directory}"
            )
        metrics = run_metrics(
            evaluation_directory=args.output,
            dataset_root=args.dataset,
            prediction_directory=prediction_directory,
            paper_ap=bool(getattr(args, "paper_ap", False)),
        )
        print(metrics)
        return 0
    manifest_candidate = Path(args.output).expanduser().resolve() / "evaluation_manifest.json"
    if manifest_candidate.is_file():
        if config.load_said_ckpt.is_file():
            expected_sha256 = checkpoint_sha256(config.load_said_ckpt)
        else:
            expected = PAPER_CHECKPOINTS[config.class_feature_encoder]
            if config.load_said_ckpt.name != (
                "said_passt.ckpt"
                if config.class_feature_encoder == "passt"
                else "said_audiomae.ckpt"
            ):
                raise FileNotFoundError(
                    "the custom checkpoint is required to validate reusable "
                    f"predictions: {config.load_said_ckpt}"
                )
            expected_sha256 = expected.sha256
        manifest = validate_completed_inference(
            manifest_candidate,
            checkpoint_sha256=expected_sha256,
            class_feature_encoder=config.class_feature_encoder,
            dataset_root=args.dataset,
        )
        if predictions_only:
            print(manifest)
            return 0
        metrics = run_metrics(
            evaluation_directory=args.output,
            dataset_root=args.dataset,
            paper_ap=bool(getattr(args, "paper_ap", False)),
        )
        print(metrics)
        return 0

    checkpoint = _ensure_config_checkpoint(config)

    import torch

    from .models import SAID, load_said_checkpoint

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.backends.cudnn.enabled = False
    model = SAID(class_feature_encoder=config.class_feature_encoder)
    report = load_said_checkpoint(
        model,
        checkpoint,
        class_feature_encoder=config.class_feature_encoder,
    )
    manifest = run_development_test_inference(
        model,
        args.dataset,
        args.output,
        device=device,
        batch_size=args.batch_size,
        model_name=report.model_name,
        checkpoint_sha256=report.sha256,
        class_feature_encoder=config.class_feature_encoder,
    )
    if predictions_only:
        print(manifest)
        return 0
    metrics = run_metrics(
        evaluation_directory=args.output,
        dataset_root=args.dataset,
        paper_ap=bool(getattr(args, "paper_ap", False)),
    )
    print(metrics)
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    selected_model = args.model or "said_passt"
    output = (args.output or Path("said_demo_output")).expanduser().resolve()
    if selected_model == "audio2sph":
        return _run_audio2sph_demo(args, output)
    config = _read_config(args.config, args.model, args.checkpoint)
    config.validate_for_command("infer")
    if config.class_feature_encoder is None:
        raise ConfigError("said demo requires a complete SAID checkpoint")
    output_names = tuple(
        f"said_{config.class_feature_encoder}_demo_{index}.mp4"
        for index in range(1, 5)
    )
    output_paths = _validate_demo_output_paths(output, output_names)
    checkpoint = _ensure_config_checkpoint(config)

    import torch

    from .inference import (
        DEMO_SCENES,
        SAIDPredictor,
        combine_comparison_videos,
        load_demo_audio,
        render_prediction_video,
        write_demo_audio,
        write_demo_ground_truth_archive,
        write_prediction_archive,
    )
    from .models import SAID, load_said_checkpoint

    for scene in DEMO_SCENES:
        scene.paths()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    torch.backends.cudnn.enabled = False
    model = SAID(class_feature_encoder=config.class_feature_encoder)
    total_scenes = len(DEMO_SCENES)

    def show_progress(completed: int, message: str) -> None:
        width = 20
        filled = round(width * completed / max(total_scenes, 1))
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"[{bar}] {completed}/{total_scenes} {message}",
            flush=True,
        )

    show_progress(0, f"Loading {config.class_feature_encoder} checkpoint")
    report = load_said_checkpoint(
        model,
        checkpoint,
        class_feature_encoder=config.class_feature_encoder,
    )
    predictor = SAIDPredictor(
        model,
        device=device,
        batch_size=args.batch_size,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        work = staging / ".work"
        work.mkdir()
        for index, scene in enumerate(DEMO_SCENES, start=1):
            output_name = output_names[index - 1]
            show_progress(index - 1, f"Generating {output_name}")
            source_audio, source_labels, source_video = scene.paths()
            audio = load_demo_audio(scene, source_audio)
            excerpt_audio = write_demo_audio(
                audio, work / f"scene_{index}.wav"
            )
            prediction_directory = work / f"prediction_{index}"
            prediction_manifest = write_prediction_archive(
                predictor.predict(
                    audio,
                    recording_domain=scene.recording_domain,
                ),
                prediction_directory,
                recording=source_audio,
                model_name=report.model_name,
                checkpoint_sha256=report.sha256,
                class_feature_encoder=config.class_feature_encoder,
                recording_domain=scene.recording_domain.value,
                threshold=0.05,
                max_sources_per_frame=4,
            )
            ground_truth_manifest = write_demo_ground_truth_archive(
                scene,
                source_labels,
                work / f"ground_truth_{index}",
            )
            prediction_video = work / f"prediction_{index}.mp4"
            ground_truth_video = work / f"ground_truth_{index}.mp4"
            common_render_options = {
                "header_height": 120,
                "show_confidence": False,
                "legend_class_ids": scene.legend_class_ids,
                "video_offset_seconds": 0.0,
            }
            render_prediction_video(
                prediction_manifest,
                source_video,
                excerpt_audio,
                prediction_video,
                title=f"{report.model_name}  |  Prediction",
                **common_render_options,
            )
            render_prediction_video(
                ground_truth_manifest,
                source_video,
                excerpt_audio,
                ground_truth_video,
                title="Ground Truth",
                **common_render_options,
            )
            combine_comparison_videos(
                ground_truth_video,
                prediction_video,
                staging / output_name,
            )
            show_progress(index, f"Completed {output_name}")
        shutil.rmtree(work)
        _publish_demo_outputs(staging, output_paths)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for path in output_paths:
        print(path)
    return 0


def _validate_demo_output_paths(
    output: Path,
    output_names: Sequence[str],
) -> tuple[Path, ...]:
    """Return non-conflicting destinations in the shared demo directory."""

    if output.exists() and not output.is_dir():
        raise FileExistsError(f"demo output is not a directory: {output}")
    destinations = tuple(output / name for name in output_names)
    existing = tuple(path for path in destinations if path.exists())
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"demo output files already exist: {names}")
    return destinations


def _publish_demo_outputs(
    staging: Path,
    destinations: Sequence[Path],
) -> None:
    """Publish one model's completed videos without replacing prior demos."""

    if not destinations:
        raise ValueError("at least one demo destination is required")
    output = destinations[0].parent
    output.mkdir(parents=True, exist_ok=True)
    existing = tuple(path for path in destinations if path.exists())
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"demo output files already exist: {names}")
    for destination in destinations:
        os.replace(staging / destination.name, destination)
    staging.rmdir()


def _run_audio2sph_demo(args: argparse.Namespace, output: Path) -> int:
    if args.config is not None:
        raise ConfigError(
            "--model audio2sph selects the published Audio2Sph checkpoint "
            "directly; omit --config"
        )
    output_names = tuple(
        f"audio2sph_demo_{index}.mp4" for index in range(1, 5)
    )
    output_paths = _validate_demo_output_paths(output, output_names)
    checkpoint = _ensure_audio2sph_checkpoint(args.checkpoint)

    import torch

    from .inference import (
        Audio2SphPredictor,
        DEMO_SCENES,
        combine_comparison_videos,
        load_demo_audio,
        render_prediction_video,
        write_class_agnostic_prediction_archive,
        write_demo_audio,
        write_demo_class_agnostic_ground_truth_archive,
    )
    from .models import (
        Audio2SphPretrainingModel,
        load_audio2sph_pretraining_checkpoint,
    )

    for scene in DEMO_SCENES:
        scene.paths()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    torch.backends.cudnn.enabled = False
    model = Audio2SphPretrainingModel()
    total_scenes = len(DEMO_SCENES)

    def show_progress(completed: int, message: str) -> None:
        width = 20
        filled = round(width * completed / max(total_scenes, 1))
        bar = "#" * filled + "-" * (width - filled)
        print(f"[{bar}] {completed}/{total_scenes} {message}", flush=True)

    show_progress(0, "Loading Audio2Sph checkpoint")
    report = load_audio2sph_pretraining_checkpoint(model, checkpoint)
    predictor = Audio2SphPredictor(
        model,
        device=device,
        batch_size=args.batch_size,
        output_fps=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        work = staging / ".work"
        work.mkdir()
        for index, scene in enumerate(DEMO_SCENES, start=1):
            output_name = output_names[index - 1]
            show_progress(index - 1, f"Generating {output_name}")
            source_audio, source_labels, source_video = scene.paths()
            audio = load_demo_audio(scene, source_audio)
            excerpt_audio = write_demo_audio(
                audio, work / f"scene_{index}.wav"
            )
            prediction_manifest = write_class_agnostic_prediction_archive(
                predictor.predict(audio),
                work / f"prediction_{index}",
                recording=source_audio,
                model_name=report.model_name,
                checkpoint_sha256=report.sha256,
                output_fps=10,
            )
            ground_truth_manifest = (
                write_demo_class_agnostic_ground_truth_archive(
                    scene,
                    source_labels,
                    work / f"ground_truth_{index}",
                )
            )
            prediction_video = work / f"prediction_{index}.mp4"
            ground_truth_video = work / f"ground_truth_{index}.mp4"
            render_prediction_video(
                prediction_manifest,
                source_video,
                excerpt_audio,
                prediction_video,
                title="Audio2Sph + Panoramic Decoder  |  Prediction",
                header_height=88,
                show_confidence=False,
                video_offset_seconds=0.0,
            )
            render_prediction_video(
                ground_truth_manifest,
                source_video,
                excerpt_audio,
                ground_truth_video,
                title="Class-agnostic Ground Truth",
                header_height=88,
                show_confidence=False,
                video_offset_seconds=0.0,
            )
            combine_comparison_videos(
                ground_truth_video,
                prediction_video,
                staging / output_name,
            )
            show_progress(index, f"Completed {output_name}")
        shutil.rmtree(work)
        _publish_demo_outputs(staging, output_paths)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for path in output_paths:
        print(path)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    config = _read_config(args.config)
    config.validate_for_command("train")
    from .training import run_training

    output_directory = config.training.output_directory.expanduser().resolve()
    with _training_output_lock(output_directory):
        report = run_training(
            config,
            config_path=args.config,
            resume=args.resume,
        )
    print(report.latest_ema_checkpoint)
    return 0


@contextmanager
def _training_output_lock(output_directory: Path):
    """Prevent two CLI training processes from writing to one run directory."""

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_directory.parent / f".{output_directory.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another training process is already using output directory: "
                f"{output_directory}"
            ) from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _run_prepare(args: argparse.Namespace) -> int:
    config = _read_config(args.config)
    config.validate_for_command("prepare")
    config.validate_data_config()
    with config.data_config.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)
    vctk = data_config["simulated_scenes"]["vctk"]

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return (config.data_config.parent / path).resolve()

    from .data import prepare_vctk_v080

    preprocessing = vctk["preprocessing"]
    report = prepare_vctk_v080(
        resolve(vctk["source_root"]),
        resolve(vctk["prepared_root"]),
        sample_rate=int(preprocessing["sample_rate"]),
        segment_duration=float(preprocessing["segment_duration"]),
        silence_threshold=float(preprocessing["silence_threshold"]),
        activity_window_seconds=float(
            preprocessing["activity_window_seconds"]
        ),
        activity_hop_ratio=float(preprocessing["activity_hop_ratio"]),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _run_compress(args: argparse.Namespace) -> int:
    from .utils.compression import compress_prediction_directory

    manifest = compress_prediction_directory(
        args.predictions,
        args.output,
        workers=args.workers,
        maximum_file_bytes=args.maximum_file_bytes,
    )
    print(manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="said",
        description="Semantic Acoustic Imaging Detector",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser(
        "infer", help="Predict labeled acoustic maps from four-channel audio."
    )
    infer.add_argument("recording", type=Path, help="Four-channel or 32-channel Eigenmike WAV file.")
    _add_config_argument(infer)
    _add_checkpoint_argument(infer)
    infer.add_argument(
        "--model",
        choices=("said_passt", "said_audiomae", "audio2sph"),
        default=None,
        help="Published model (default: said_passt).",
    )
    infer.add_argument("--output", type=Path, default=Path("said_output"), help="Output directory.")
    infer.add_argument(
        "--recording-domain",
        choices=("sony", "tau"),
        default="sony",
        help=(
            "SAID (PaSST) recording-domain embedding "
            "(default: sony; unused by SAID (AudioMAE) and audio2sph)."
        ),
    )
    infer.add_argument(
        "--device", default="auto", help="Inference device (default: auto)."
    )
    infer.add_argument(
        "--batch-size", type=int, default=1, help="Two-second windows per batch."
    )
    infer.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        help=(
            "Minimum confidence stored for complete SAID models "
            "(unused by audio2sph)."
        ),
    )
    infer.add_argument(
        "--max-sources-per-frame",
        type=int,
        default=4,
        help=(
            "Maximum stored slots per frame for complete SAID models "
            "(unused by audio2sph)."
        ),
    )
    infer.add_argument(
        "--format",
        dest="output_format",
        choices=("json", "npz"),
        default=None,
        help=(
            "Output format (default: JSON; use npz for lossless float32 maps)."
        ),
    )
    infer.add_argument(
        "--visual",
        type=Path,
        default=None,
        metavar="PANORAMA",
        help="Also overlay predictions on a synchronized 360-degree video.",
    )
    infer.set_defaults(handler=_run_infer)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate SAID on the DCASE development test set.",
    )
    evaluate.add_argument(
        "dataset",
        type=Path,
        help="Directory containing the DCASE2026 eigen_dev/ and labels_dev/ folders.",
    )
    _add_config_argument(evaluate)
    _add_checkpoint_argument(evaluate)
    evaluate.add_argument(
        "--model",
        choices=("said_passt", "said_audiomae"),
        default=None,
        help="Complete paper model (default: said_passt).",
    )
    evaluate.add_argument("--output", type=Path, default=Path("said_evaluation"), help="Output directory.")
    evaluate.add_argument(
        "--predictions-only",
        action="store_true",
        help="Write or validate predictions without running an evaluator.",
    )
    evaluate.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Score an existing prediction directory without running inference.",
    )
    evaluate.add_argument(
        "--add-previous-metrics",
        dest="paper_ap",
        action="store_true",
        help="Add the previous (4.2) COCO AP metrics to the current results.",
    )
    evaluate.add_argument(
        "--device", default="auto", help="Inference device (default: auto)."
    )
    evaluate.add_argument(
        "--batch-size", type=int, default=1, help="Two-second windows per batch."
    )
    evaluate.set_defaults(handler=_run_evaluate)

    demo = subparsers.add_parser(
        "demo", help="Generate four DCASE2026 audio-visual demonstrations."
    )
    _add_config_argument(demo)
    _add_checkpoint_argument(demo)
    demo.add_argument(
        "--model",
        choices=("said_passt", "said_audiomae", "audio2sph"),
        default=None,
        help="Published model (default: said_passt).",
    )
    demo.add_argument(
        "--output", type=Path, default=None, help="Output directory."
    )
    demo.add_argument(
        "--device", default="auto", help="Inference device (default: auto)."
    )
    demo.add_argument(
        "--batch-size", type=int, default=1, help="Two-second windows per batch."
    )
    demo.set_defaults(handler=_run_demo)

    prepare = subparsers.add_parser(
        "prepare",
        help="Prepare VCTK for Audio2Sph training.",
        description=(
            "Prepare VCTK v0.80 for the paper's Audio2Sph training recipe.\n\n"
            "Set simulated_scenes.vctk.source_root and prepared_root in "
            "configs/data.yaml before running this command."
        ),
        epilog=(
            "Example:\n"
            "  said prepare --config configs/training/audio2sph.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    prepare.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Audio2Sph training configuration.",
    )
    prepare.set_defaults(handler=_run_prepare)

    train = subparsers.add_parser(
        "train",
        help="Run Audio2Sph pretraining or complete SAID training.",
        description=(
            "Train Audio2Sph or complete SAID with a paper-aligned recipe.\n\n"
            "Set the corresponding local dataset paths in configs/data.yaml "
            "before training."
        ),
        epilog=(
            "Paper recipes:\n"
            "  Audio2Sph:\n"
            "    said train --config configs/training/audio2sph.yaml\n\n"
            "  SAID (PaSST):\n"
            "    said train --config configs/training/sourcebank_passt.yaml\n"
            "    said train --config configs/training/dcase_passt.yaml\n\n"
            "  SAID (AudioMAE):\n"
            "    said train --config configs/training/sourcebank_audiomae.yaml\n"
            "    said train --config configs/training/dcase_audiomae.yaml\n\n"
            "Each recipe uses its configured initialization independently; "
            "edit the next recipe's load_*_ckpt field to pass newly trained "
            "weights between recipes.\n\n"
            "Rerun the same command to resume the latest saved state.\n"
            "Use --resume CHECKPOINT only for a training state with the same "
            "training and data configuration fingerprint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train.add_argument("--config", type=Path, required=True, help="Top-level SAID YAML configuration.")
    train.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Explicit .training.pt state with the same configuration "
            "fingerprint; latest.training.pt is resumed automatically."
        ),
    )
    train.set_defaults(handler=_run_train)

    compress = subparsers.add_parser(
        "compress", help="Compress DCASE acoustic-map predictions."
    )
    compress.add_argument("predictions", type=Path, help="Prediction directory.")
    compress.add_argument("--output", type=Path, default=Path("said_compressed"), help="Output directory.")
    compress.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes (default: 75%% of available CPU cores).",
    )
    compress.add_argument(
        "--maximum-file-bytes",
        type=int,
        default=20_000_000,
        help="Per-recording JSON limit in bytes (default: 20000000).",
    )
    compress.set_defaults(handler=_run_compress)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        ConfigError,
        FileExistsError,
        FileNotFoundError,
        ImportError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
