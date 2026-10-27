"""Segmented complete-recording prediction for SAID."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..models import (
    Audio2SphPretrainingModel,
    RecordingDomain,
    SAID,
    load_audio2sph_pretraining_checkpoint,
    load_said_checkpoint,
)


@dataclass(frozen=True)
class PredictionChunk:
    """Consecutive 10 fps predictions from one complete recording."""

    frame_start: int
    refined_maps: Tensor
    slot_class_ids: Tensor
    slot_confidence: Tensor

    @property
    def frame_stop(self) -> int:
        return int(self.frame_start) + int(self.refined_maps.shape[0])


@dataclass(frozen=True)
class ClassAgnosticPredictionChunk:
    """Consecutive Audio2Sph panoramic maps from one complete recording."""

    frame_start: int
    maps: Tensor

    @property
    def frame_stop(self) -> int:
        return int(self.frame_start) + int(self.maps.shape[0])


class SAIDPredictor:
    """Apply a paper checkpoint to a complete recording in two-second windows."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device,
        sample_rate: int = 48_000,
        segment_duration: float = 2.0,
        output_fps: int = 10,
        batch_size: int = 1,
        cudnn_enabled: bool = False,
    ) -> None:
        torch.backends.cudnn.enabled = bool(cudnn_enabled)
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.sample_rate = int(sample_rate)
        self.segment_samples = int(round(float(segment_duration) * self.sample_rate))
        self.output_fps = int(output_fps)
        self.frames_per_segment = int(round(float(segment_duration) * self.output_fps))
        self.batch_size = int(batch_size)
        if self.segment_samples <= 0 or self.frames_per_segment <= 0:
            raise ValueError("segment duration, sample rate, and output fps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def _segments(self, audio: Tensor) -> Iterator[Tensor]:
        if audio.ndim != 2 or int(audio.shape[0]) != 4:
            raise ValueError(f"audio must have shape [4,samples], got {tuple(audio.shape)}")
        for start in range(0, int(audio.shape[-1]), self.segment_samples):
            yield F.pad(
                audio[:, start : start + self.segment_samples],
                (0, max(0, start + self.segment_samples - int(audio.shape[-1]))),
            )

    def predict(
        self,
        audio: Tensor,
        *,
        recording_domain: RecordingDomain | str = RecordingDomain.SONY,
        output_frame_limit: int | None = None,
    ) -> Iterator[PredictionChunk]:
        """Yield chronological predictions without retaining full outputs in RAM.

        ``output_frame_limit`` restricts emitted frames while keeping complete
        two-second input windows. Evaluation uses this to preserve real audio
        context after the final annotated frame.
        """

        # A frame at index k represents the 100 ms interval beginning at
        # k / output_fps.  Keep every frame whose start lies in the recording.
        samples = int(audio.shape[-1])
        audio_frames = (
            samples * self.output_fps + self.sample_rate - 1
        ) // self.sample_rate
        if output_frame_limit is None:
            total_frames = audio_frames
        else:
            if isinstance(output_frame_limit, bool) or not isinstance(
                output_frame_limit, int
            ):
                raise TypeError("output_frame_limit must be an integer or null")
            if output_frame_limit < 0:
                raise ValueError("output_frame_limit must be non-negative")
            if output_frame_limit > audio_frames:
                raise ValueError(
                    "output_frame_limit exceeds the frames covered by the recording"
                )
            total_frames = int(output_frame_limit)
        if total_frames <= 0:
            return
        required_segments = (
            total_frames + self.frames_per_segment - 1
        ) // self.frames_per_segment
        segments = islice(self._segments(audio), required_segments)
        frame_start = 0
        with torch.inference_mode():
            while True:
                batch_segments = list(islice(segments, self.batch_size))
                if not batch_segments:
                    break
                batch = torch.stack(batch_segments).to(
                    device=self.device, dtype=torch.float32
                )
                domains = [recording_domain] * int(batch.shape[0])
                output = self.model(batch, recording_domain=domains)
                for batch_index in range(int(batch.shape[0])):
                    remaining = total_frames - frame_start
                    frame_count = min(self.frames_per_segment, remaining)
                    if frame_count <= 0:
                        return
                    yield PredictionChunk(
                        frame_start=frame_start,
                        refined_maps=output.refined_maps[
                            batch_index, :frame_count
                        ].detach().cpu(),
                        slot_class_ids=output.slot_class_ids[
                            batch_index, :frame_count
                        ].detach().cpu(),
                        slot_confidence=output.slot_confidence[
                            batch_index, :frame_count
                        ].detach().cpu(),
                    )
                    frame_start += frame_count


class Audio2SphPredictor:
    """Apply Audio2Sph and the Panoramic Decoder in two-second windows."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device,
        sample_rate: int = 48_000,
        segment_duration: float = 2.0,
        output_fps: int = 100,
        native_output_fps: int = 100,
        batch_size: int = 1,
        cudnn_enabled: bool = False,
    ) -> None:
        torch.backends.cudnn.enabled = bool(cudnn_enabled)
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.sample_rate = int(sample_rate)
        self.segment_samples = int(round(float(segment_duration) * self.sample_rate))
        self.output_fps = int(output_fps)
        self.native_output_fps = int(native_output_fps)
        self.frames_per_segment = int(round(float(segment_duration) * self.output_fps))
        self.batch_size = int(batch_size)
        if self.segment_samples <= 0 or self.frames_per_segment <= 0:
            raise ValueError("segment duration, sample rate, and output fps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (
            self.native_output_fps <= 0
            or self.output_fps > self.native_output_fps
            or self.native_output_fps % self.output_fps != 0
        ):
            raise ValueError("output_fps must be a positive divisor of 100")
        self.frame_stride = self.native_output_fps // self.output_fps

    def _segments(self, audio: Tensor) -> Iterator[Tensor]:
        if audio.ndim != 2 or int(audio.shape[0]) != 4:
            raise ValueError(f"audio must have shape [4,samples], got {tuple(audio.shape)}")
        for start in range(0, int(audio.shape[-1]), self.segment_samples):
            yield F.pad(
                audio[:, start : start + self.segment_samples],
                (0, max(0, start + self.segment_samples - int(audio.shape[-1]))),
            )

    def predict(
        self,
        audio: Tensor,
        *,
        output_frame_limit: int | None = None,
    ) -> Iterator[ClassAgnosticPredictionChunk]:
        """Yield maps whose frame starts lie inside the input recording."""

        samples = int(audio.shape[-1])
        audio_frames = (
            samples * self.output_fps + self.sample_rate - 1
        ) // self.sample_rate
        if output_frame_limit is None:
            total_frames = audio_frames
        else:
            if isinstance(output_frame_limit, bool) or not isinstance(
                output_frame_limit, int
            ):
                raise TypeError("output_frame_limit must be an integer or null")
            if output_frame_limit < 0:
                raise ValueError("output_frame_limit must be non-negative")
            if output_frame_limit > audio_frames:
                raise ValueError(
                    "output_frame_limit exceeds the frames covered by the recording"
                )
            total_frames = int(output_frame_limit)
        if total_frames <= 0:
            return
        required_segments = (
            total_frames + self.frames_per_segment - 1
        ) // self.frames_per_segment
        segments = islice(self._segments(audio), required_segments)
        frame_start = 0
        with torch.inference_mode():
            while True:
                batch_segments = list(islice(segments, self.batch_size))
                if not batch_segments:
                    break
                batch = torch.stack(batch_segments).to(
                    device=self.device, dtype=torch.float32
                )
                output = self.model(batch)
                maps = output.panoramic_decoder.class_agnostic_maps
                sampled = maps[:, :: self.frame_stride]
                if int(sampled.shape[1]) < self.frames_per_segment:
                    raise RuntimeError(
                        "Panoramic Decoder returned fewer frames than the "
                        "declared Audio2Sph time base"
                    )
                for batch_index in range(int(batch.shape[0])):
                    remaining = total_frames - frame_start
                    frame_count = min(self.frames_per_segment, remaining)
                    if frame_count <= 0:
                        return
                    yield ClassAgnosticPredictionChunk(
                        frame_start=frame_start,
                        maps=sampled[
                            batch_index, :frame_count
                        ].detach().cpu(),
                    )
                    frame_start += frame_count


def run_inference(
    recording: str | Path,
    output_directory: str | Path,
    *,
    checkpoint: str | Path,
    class_feature_encoder: str,
    recording_domain: RecordingDomain | str = RecordingDomain.SONY,
    device: str | torch.device = "auto",
    batch_size: int = 1,
    score_threshold: float = 0.05,
    max_sources_per_frame: int = 4,
    output_format: str = "json",
    visual: str | Path | None = None,
) -> Path:
    """Run complete SAID inference and write DCASE JSON or a lossless NPZ archive."""

    from .load_audio import load_eigenmike_audio
    from .save import write_prediction_archive

    if output_format not in {"json", "npz"}:
        raise ValueError("output_format must be 'json' or 'npz'")
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output path already exists: {output}")
    visual_path = None if visual is None else Path(visual).expanduser().resolve()
    if visual_path is not None:
        if not visual_path.is_file():
            raise FileNotFoundError(f"visual input not found: {visual_path}")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("SAID visualization requires ffmpeg on PATH")

    resolved_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device) == "auto"
        else torch.device(device)
    )
    model = SAID(class_feature_encoder=class_feature_encoder)
    report = load_said_checkpoint(
        model,
        checkpoint,
        class_feature_encoder=class_feature_encoder,
    )
    audio = load_eigenmike_audio(recording)
    predictor = SAIDPredictor(
        model,
        device=resolved_device,
        batch_size=batch_size,
    )
    domain = (
        recording_domain.value
        if isinstance(recording_domain, RecordingDomain)
        else str(recording_domain)
    )
    chunks = predictor.predict(audio, recording_domain=recording_domain)
    if output_format == "npz" and visual_path is None:
        return write_prediction_archive(
            chunks,
            output,
            recording=recording,
            model_name=report.model_name,
            checkpoint_sha256=report.sha256,
            class_feature_encoder=class_feature_encoder,
            recording_domain=domain,
            threshold=score_threshold,
            max_sources_per_frame=max_sources_per_frame,
        )

    from ..evaluation.output2dcase import (
        write_dcase_archive_json,
        write_dcase_prediction_json,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    filename = f"{Path(recording).stem}_inference.json"
    video_filename = f"{Path(recording).stem}_prediction.mp4"
    try:
        if visual_path is None:
            write_dcase_prediction_json(
                chunks,
                staging / filename,
                confidence_threshold=score_threshold,
                max_sources_per_frame=max_sources_per_frame,
                relative_map_threshold=0.10,
            )
            os.replace(staging, output)
            return output / filename

        publish = staging / "publish"
        archive = publish if output_format == "npz" else staging / "archive"
        manifest = write_prediction_archive(
            chunks,
            archive,
            recording=recording,
            model_name=report.model_name,
            checkpoint_sha256=report.sha256,
            class_feature_encoder=class_feature_encoder,
            recording_domain=domain,
            threshold=score_threshold,
            max_sources_per_frame=max_sources_per_frame,
        )
        if output_format == "json":
            publish.mkdir()
            write_dcase_archive_json(
                manifest,
                publish / filename,
                relative_map_threshold=0.10,
            )
        from .demo import write_demo_audio
        from .visualization import render_prediction_video
        from ..data.dcase2026 import DCASE2026_CLASS_NAMES

        selected_audio = write_demo_audio(audio, staging / "selected_audio.wav")
        render_prediction_video(
            manifest,
            visual_path,
            selected_audio,
            publish / video_filename,
            title=f"{report.model_name}  |  Prediction",
            header_height=120,
            show_confidence=False,
            legend_class_ids=tuple(range(len(DCASE2026_CLASS_NAMES))),
            video_fps=10,
        )
        os.replace(publish, output)
        shutil.rmtree(staging, ignore_errors=True)
        return output / ("manifest.json" if output_format == "npz" else filename)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_audio2sph_inference(
    recording: str | Path,
    output_directory: str | Path,
    *,
    checkpoint: str | Path,
    device: str | torch.device = "auto",
    batch_size: int = 1,
    output_fps: int = 100,
    output_format: str = "json",
    visual: str | Path | None = None,
) -> Path:
    """Run class-agnostic Audio2Sph inference for one complete recording."""

    from .load_audio import load_eigenmike_audio
    from .save import (
        write_class_agnostic_archive_json,
        write_class_agnostic_prediction_archive,
        write_class_agnostic_prediction_json,
    )

    if output_format not in {"json", "npz"}:
        raise ValueError("output_format must be 'json' or 'npz'")
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output path already exists: {output}")
    visual_path = None if visual is None else Path(visual).expanduser().resolve()
    if visual_path is not None:
        if not visual_path.is_file():
            raise FileNotFoundError(f"visual input not found: {visual_path}")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("SAID visualization requires ffmpeg on PATH")

    resolved_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device) == "auto"
        else torch.device(device)
    )
    model = Audio2SphPretrainingModel()
    report = load_audio2sph_pretraining_checkpoint(model, checkpoint)
    audio = load_eigenmike_audio(recording)
    predictor = Audio2SphPredictor(
        model,
        device=resolved_device,
        batch_size=batch_size,
        output_fps=output_fps,
    )
    chunks = predictor.predict(audio)
    if output_format == "npz" and visual_path is None:
        return write_class_agnostic_prediction_archive(
            chunks,
            output,
            recording=recording,
            model_name=report.model_name,
            checkpoint_sha256=report.sha256,
            output_fps=output_fps,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    filename = f"{Path(recording).stem}_inference.json"
    video_filename = f"{Path(recording).stem}_prediction.mp4"
    try:
        if visual_path is None:
            write_class_agnostic_prediction_json(
                chunks,
                staging / filename,
                recording=recording,
                model_name=report.model_name,
                checkpoint_sha256=report.sha256,
                output_fps=output_fps,
                relative_map_threshold=0.10,
            )
            os.replace(staging, output)
            return output / filename

        publish = staging / "publish"
        archive = publish if output_format == "npz" else staging / "archive"
        manifest = write_class_agnostic_prediction_archive(
            chunks,
            archive,
            recording=recording,
            model_name=report.model_name,
            checkpoint_sha256=report.sha256,
            output_fps=output_fps,
        )
        if output_format == "json":
            publish.mkdir()
            write_class_agnostic_archive_json(
                manifest,
                publish / filename,
                relative_map_threshold=0.10,
            )
        from .demo import write_demo_audio
        from .visualization import render_prediction_video

        selected_audio = write_demo_audio(audio, staging / "selected_audio.wav")
        rendered_fps = 10 if output_fps >= 10 and output_fps % 10 == 0 else output_fps
        render_prediction_video(
            manifest,
            visual_path,
            selected_audio,
            publish / video_filename,
            title=f"{report.model_name}  |  Prediction",
            show_confidence=False,
            video_fps=rendered_fps,
        )
        os.replace(publish, output)
        shutil.rmtree(staging, ignore_errors=True)
        return output / ("manifest.json" if output_format == "npz" else filename)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
