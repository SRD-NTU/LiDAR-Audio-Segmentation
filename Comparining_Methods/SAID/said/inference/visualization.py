"""Render SAID prediction archives over synchronized 360-degree video."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import soundfile as sf

from ..data.dcase2026 import DCASE2026_CLASS_NAMES


_CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 92, 138),
    (45, 140, 240),
    (255, 214, 10),
    (0, 190, 190),
    (50, 200, 95),
    (255, 145, 30),
    (145, 190, 45),
    (160, 95, 55),
    (235, 190, 30),
    (220, 65, 220),
    (70, 100, 210),
    (145, 75, 230),
    (220, 50, 55),
)


def _video_columns_from_model_columns(width: int = 360) -> np.ndarray:
    """Return model-column indices in standard equirectangular video order."""

    if int(width) <= 0:
        raise ValueError("map width must be positive")
    columns = np.arange(int(width), dtype=np.int64)
    return ((int(width) // 2 - 1) - columns) % int(width)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", int(size))
    except OSError:  # pragma: no cover - platform font fallback
        return ImageFont.load_default()


def _overlay_predictions(
    frame: np.ndarray,
    maps: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
) -> np.ndarray:
    if frame.ndim != 3 or int(frame.shape[2]) != 3:
        raise ValueError("video frame must have shape [height,width,3]")
    if maps.ndim != 3 or tuple(maps.shape[-2:]) != (180, 360):
        raise ValueError("prediction maps must have shape [sources,180,360]")
    if class_ids.shape != confidences.shape or class_ids.shape != maps.shape[:1]:
        raise ValueError("maps, class IDs, and confidences must describe the same sources")

    output = frame.astype(np.float32) / 255.0
    video_columns = _video_columns_from_model_columns()
    for prediction_map, class_id in zip(maps, class_ids):
        normalized_class_id = int(class_id)
        if normalized_class_id < 0 or normalized_class_id >= len(_CLASS_COLORS):
            raise ValueError(f"class ID is outside the SAID taxonomy: {class_id}")
        display_map = np.asarray(prediction_map[:, video_columns], dtype=np.float32)
        peak = float(np.max(display_map))
        if peak > 0.0:
            display_map = display_map / peak
        else:
            display_map = np.zeros_like(display_map)
        map_image = Image.fromarray(
            np.rint(np.clip(display_map, 0.0, 1.0) * 255.0).astype(np.uint8),
            mode="L",
        ).resize(
            (int(frame.shape[1]), int(frame.shape[0])),
            resample=Image.Resampling.BILINEAR,
        )
        strength = np.asarray(map_image, dtype=np.float32) / 255.0
        alpha = (np.clip((strength - 0.10) / 0.90, 0.0, 1.0) * 0.68)[..., None]
        color = np.asarray(
            _CLASS_COLORS[normalized_class_id], dtype=np.float32
        )[None, None, :] / 255.0
        output = output * (1.0 - alpha) + color * alpha
    return np.rint(np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)


def _overlay_class_agnostic_map(
    frame: np.ndarray,
    acoustic_map: np.ndarray,
) -> np.ndarray:
    """Overlay one class-agnostic map with a fixed zero-to-one display scale."""

    if frame.ndim != 3 or int(frame.shape[2]) != 3:
        raise ValueError("video frame must have shape [height,width,3]")
    if tuple(acoustic_map.shape) != (180, 360):
        raise ValueError("class-agnostic map must have shape [180,360]")
    if not np.isfinite(acoustic_map).all():
        raise ValueError("class-agnostic map must contain finite values")
    if acoustic_map.size and (
        float(acoustic_map.min()) < 0.0 or float(acoustic_map.max()) > 1.0
    ):
        raise ValueError("class-agnostic map values must lie in [0,1]")

    video_columns = _video_columns_from_model_columns()
    display_map = np.asarray(acoustic_map[:, video_columns], dtype=np.float32)
    map_image = Image.fromarray(
        np.rint(display_map * 255.0).astype(np.uint8), mode="L"
    ).resize(
        (int(frame.shape[1]), int(frame.shape[0])),
        resample=Image.Resampling.BILINEAR,
    )
    strength = np.asarray(map_image, dtype=np.float32) / 255.0
    # Keep an absolute zero-to-one scale while making moderate responses legible
    # after video encoding and README-sized playback.  The same transfer curve
    # is used for Ground Truth and Prediction; no frame-wise normalization is
    # introduced.
    alpha = (
        np.power(np.clip((strength - 0.05) / 0.75, 0.0, 1.0), 0.60) * 0.78
    )[..., None]
    # A fixed amber-to-white ramp keeps Ground Truth and Prediction directly
    # comparable without introducing class colors or per-frame peak scaling.
    low = np.asarray((1.0, 0.34, 0.02), dtype=np.float32)[None, None, :]
    high = np.asarray((1.0, 1.0, 0.82), dtype=np.float32)[None, None, :]
    color = low * (1.0 - strength[..., None]) + high * strength[..., None]
    output = frame.astype(np.float32) / 255.0
    output = output * (1.0 - alpha) + color * alpha
    return np.rint(np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)


def _compose_frame(
    video_frame: np.ndarray,
    maps: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    *,
    frame_index: int,
    fps: int,
    title: str,
    show_confidence: bool,
    legend_class_ids: np.ndarray | None,
    header_height: int,
) -> np.ndarray:
    overlay = _overlay_predictions(video_frame, maps, class_ids, confidences)
    height, width = overlay.shape[:2]
    canvas = Image.new("RGB", (int(width), int(height + header_height)), (14, 18, 24))
    canvas.paste(Image.fromarray(overlay, mode="RGB"), (0, int(header_height)))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22)
    detail_font = _font(16)
    draw.text(
        (18, 10),
        title,
        fill=(245, 247, 250),
        font=title_font,
    )
    time_seconds = float(frame_index) / float(fps)
    draw.text(
        (int(width) - 105, 13),
        f"{time_seconds:05.1f} s",
        fill=(210, 216, 224),
        font=detail_font,
    )
    x = 18
    y = 52
    fixed_legend = legend_class_ids is not None
    displayed_class_ids = legend_class_ids if fixed_legend else class_ids
    if int(displayed_class_ids.size) == 0:
        draw.text(
            (x, y),
            "No source above the display threshold",
            fill=(180, 188, 198),
            font=detail_font,
        )
    else:
        for legend_index, class_id in enumerate(displayed_class_ids):
            normalized_class_id = int(class_id)
            color = _CLASS_COLORS[normalized_class_id]
            label = DCASE2026_CLASS_NAMES[normalized_class_id]
            if show_confidence and not fixed_legend:
                confidence = confidences[legend_index]
                label += f" {float(confidence):.2f}"
            text_width = int(draw.textlength(label, font=detail_font))
            item_width = 34 + text_width
            if x > 18 and x + item_width > int(width) - 18:
                x = 18
                y += 28
            draw.ellipse((x, y + 3, x + 11, y + 14), fill=color)
            draw.text((x + 16, y), label, fill=(226, 230, 236), font=detail_font)
            x += item_width
    return np.asarray(canvas, dtype=np.uint8)


def _compose_class_agnostic_frame(
    video_frame: np.ndarray,
    acoustic_map: np.ndarray,
    *,
    frame_index: int,
    fps: int,
    title: str,
    header_height: int,
) -> np.ndarray:
    overlay = _overlay_class_agnostic_map(video_frame, acoustic_map)
    height, width = overlay.shape[:2]
    canvas = Image.new(
        "RGB", (int(width), int(height + header_height)), (14, 18, 24)
    )
    canvas.paste(Image.fromarray(overlay, mode="RGB"), (0, int(header_height)))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22)
    detail_font = _font(16)
    draw.text((18, 10), title, fill=(245, 247, 250), font=title_font)
    time_seconds = float(frame_index) / float(fps)
    draw.text(
        (int(width) - 105, 13),
        f"{time_seconds:05.1f} s",
        fill=(210, 216, 224),
        font=detail_font,
    )
    draw.text(
        (18, 48),
        "Acoustic energy",
        fill=(210, 216, 224),
        font=detail_font,
    )
    return np.asarray(canvas, dtype=np.uint8)


def _read_raw_frame(stream, *, width: int, height: int) -> np.ndarray:
    expected = int(width) * int(height) * 3
    payload = stream.read(expected)
    if len(payload) != expected:
        raise RuntimeError(
            f"demo video ended early: expected {expected} bytes, received {len(payload)}"
        )
    return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)


def render_prediction_video(
    manifest_path: str | Path,
    source_video: str | Path,
    source_audio: str | Path,
    output_path: str | Path,
    *,
    width: int = 960,
    height: int = 480,
    header_height: int = 88,
    title: str | None = None,
    show_confidence: bool = True,
    legend_class_ids: Sequence[int] | None = None,
    video_offset_seconds: float = 0.0,
    video_fps: int | None = None,
) -> Path:
    """Render a synchronized MP4 from a lossless SAID prediction archive."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    source_video = Path(source_video).expanduser().resolve()
    source_audio = Path(source_audio).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output video already exists: {output_path}")
    for required in (manifest_path, source_video, source_audio):
        if not required.is_file():
            raise FileNotFoundError(f"demo input is missing: {required}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("said demo video rendering requires ffmpeg on PATH")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if schema not in {
        "said-prediction-archive-v1",
        "audio2sph-prediction-archive-v1",
    }:
        raise ValueError("unsupported prediction archive schema")
    class_agnostic = schema == "audio2sph-prediction-archive-v1"
    if (int(manifest.get("map_height", 0)), int(manifest.get("map_width", 0))) != (
        180,
        360,
    ):
        raise ValueError("demo visualization requires 180-by-360 prediction maps")
    archive_fps = int(manifest.get("output_fps", 0))
    total_frames = int(manifest.get("frames", 0))
    if archive_fps <= 0 or total_frames <= 0:
        raise ValueError("prediction archive must contain positive fps and frame count")
    rendered_fps = archive_fps if video_fps is None else int(video_fps)
    if (
        rendered_fps <= 0
        or rendered_fps > archive_fps
        or archive_fps % rendered_fps
    ):
        raise ValueError("video_fps must be a positive divisor of the archive fps")
    frame_stride = archive_fps // rendered_fps
    expected_rendered_frames = (total_frames + frame_stride - 1) // frame_stride
    if width <= 0 or height <= 0 or header_height <= 0:
        raise ValueError("video dimensions must be positive")
    if not np.isfinite(float(video_offset_seconds)) or video_offset_seconds < 0:
        raise ValueError("video_offset_seconds must be finite and non-negative")
    display_title = title or f"{manifest.get('model', 'SAID')}  |  Prediction"
    fixed_legend = None
    if class_agnostic and legend_class_ids is not None:
        raise ValueError("class-agnostic visualization does not use a class legend")
    if not class_agnostic and legend_class_ids is not None:
        fixed_legend = np.asarray(tuple(legend_class_ids), dtype=np.int64)
        if fixed_legend.ndim != 1 or np.unique(fixed_legend).size != fixed_legend.size:
            raise ValueError("legend class IDs must be a unique one-dimensional sequence")
        if fixed_legend.size and (
            int(fixed_legend.min()) < 0
            or int(fixed_legend.max()) >= len(_CLASS_COLORS)
        ):
            raise ValueError("legend class ID is outside the SAID taxonomy")
    output_height = int(height) + int(header_height)
    if int(width) % 2 or output_height % 2:
        raise ValueError("H.264 demo dimensions must be even")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.partial.mp4")
    decoder_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(float(video_offset_seconds)),
        "-i",
        str(source_video),
        "-vf",
        f"fps={rendered_fps},scale={int(width)}:{int(height)}:flags=lanczos",
        "-frames:v",
        str(expected_rendered_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    encoder_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{int(width)}x{output_height}",
        "-r",
        str(rendered_fps),
        "-i",
        "-",
        "-i",
        str(source_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-filter:a",
        "pan=stereo|c0=0.5*c0+0.5*c2|c1=0.5*c1+0.5*c3",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(temporary_output),
    ]
    decoder = subprocess.Popen(
        decoder_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        encoder_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert decoder.stdout is not None
    assert encoder.stdin is not None
    archive_directory = manifest_path.parent
    covered_frames = 0
    rendered_frames = 0
    try:
        for record in manifest.get("chunks", []):
            frame_start = int(record["frame_start"])
            frame_stop = int(record["frame_stop"])
            if frame_start != covered_frames or frame_stop < frame_start:
                raise ValueError("prediction chunks must be ordered and contiguous")
            chunk_path = archive_directory / str(record["file"])
            with np.load(chunk_path, allow_pickle=False) as chunk:
                if class_agnostic:
                    dense_maps = np.asarray(chunk["map"], dtype=np.float32)
                    expected_shape = (frame_stop - frame_start, 180, 360)
                    if tuple(dense_maps.shape) != expected_shape:
                        raise ValueError(
                            "class-agnostic prediction chunk has an invalid shape"
                        )
                else:
                    frame_indices = np.asarray(chunk["frame_index"], dtype=np.int64)
                    class_ids = np.asarray(chunk["class_id"], dtype=np.int64)
                    confidences = np.asarray(chunk["confidence"], dtype=np.float32)
                    maps = np.asarray(chunk["refined_map"], dtype=np.float32)
                    if not (
                        frame_indices.shape
                        == class_ids.shape
                        == confidences.shape
                        == maps.shape[:1]
                    ):
                        raise ValueError(
                            "prediction chunk arrays have inconsistent lengths"
                        )
                for frame_index in range(frame_start, frame_stop):
                    if frame_index % frame_stride:
                        continue
                    video_frame = _read_raw_frame(
                        decoder.stdout, width=int(width), height=int(height)
                    )
                    if class_agnostic:
                        composed = _compose_class_agnostic_frame(
                            video_frame,
                            dense_maps[frame_index - frame_start],
                            frame_index=frame_index,
                            fps=archive_fps,
                            title=display_title,
                            header_height=int(header_height),
                        )
                    else:
                        selected = frame_indices == frame_index
                        composed = _compose_frame(
                            video_frame,
                            maps[selected],
                            class_ids[selected],
                            confidences[selected],
                            frame_index=frame_index,
                            fps=archive_fps,
                            title=display_title,
                            show_confidence=bool(show_confidence),
                            legend_class_ids=fixed_legend,
                            header_height=int(header_height),
                        )
                    encoder.stdin.write(composed.tobytes())
                    rendered_frames += 1
            covered_frames = frame_stop
        if covered_frames != total_frames:
            raise ValueError(
                "prediction archive frame count does not match its manifest"
            )
        if rendered_frames != expected_rendered_frames:
            raise ValueError(
                f"archive rendered {rendered_frames} frames; "
                f"expected {expected_rendered_frames}"
            )
        encoder.stdin.close()
        decoder.stdout.close()
        decoder_stderr = decoder.stderr.read().decode("utf-8", errors="replace")
        encoder_stderr = encoder.stderr.read().decode("utf-8", errors="replace")
        decoder_returncode = decoder.wait()
        encoder_returncode = encoder.wait()
        if decoder_returncode != 0:
            raise RuntimeError(f"ffmpeg video decode failed: {decoder_stderr.strip()}")
        if encoder_returncode != 0:
            raise RuntimeError(f"ffmpeg video encode failed: {encoder_stderr.strip()}")
        os.replace(temporary_output, output_path)
    except BaseException:
        if encoder.stdin is not None and not encoder.stdin.closed:
            encoder.stdin.close()
        decoder.kill()
        encoder.kill()
        decoder.wait()
        encoder.wait()
        temporary_output.unlink(missing_ok=True)
        raise
    return output_path


def combine_comparison_videos(
    ground_truth_video: str | Path,
    prediction_video: str | Path,
    output_path: str | Path,
) -> Path:
    """Place synchronized Ground Truth and Prediction videos side by side."""

    ground_truth_video = Path(ground_truth_video).expanduser().resolve()
    prediction_video = Path(prediction_video).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output video already exists: {output_path}")
    for required in (ground_truth_video, prediction_video):
        if not required.is_file():
            raise FileNotFoundError(f"comparison video input is missing: {required}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("SAID comparison rendering requires ffmpeg on PATH")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.partial.mp4")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(ground_truth_video),
        "-i",
        str(prediction_video),
        "-filter_complex",
        "[0:v:0][1:v:0]hstack=inputs=2:shortest=1[v]",
        "-map",
        "[v]",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(temporary_output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg comparison rendering failed: {completed.stderr.strip()}"
        )
    os.replace(temporary_output, output_path)
    return output_path
