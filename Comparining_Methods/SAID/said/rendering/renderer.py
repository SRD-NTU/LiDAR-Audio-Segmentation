"""Directional ISM renderer and paper target construction for SAID."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 CUHK
# Original NESD renderer: Qiuqiang Kong
# SAID adaptation and modifications: Copyright (c) 2026 Runbang Wang
# Upstream NESD revision:
# https://github.com/qiuqiangkong/nesd/tree/7475da725dcfc8d60761c5891644ca326a7226e7

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve
from torch import Tensor

from ..data.targets import FrameAlignment
from .eigenmike import load_eigenmike_positions
from .rigid_sphere import rigid_sphere_response_table
from .shoebox_ism import RoomSample, ShoeboxImageSourceModel


@dataclass(frozen=True)
class RenderedScene:
    """Rendered four-channel audio and paper-aligned supervision."""

    audio: Tensor
    source_images: Tensor
    activity_samples: Tensor
    activity_frames_100hz: Tensor
    class_ids: Tensor
    source_maps_45x90: Tensor
    source_maps_180x360: Tensor
    audio2sph_target_180x360: Tensor
    valid_sources: Tensor
    said_class_ids: Tensor
    said_map_targets: Tensor
    said_refined_map_targets: Tensor
    frame_alignment: FrameAlignment
    room: RoomSample


def _fractional_filter(fraction: float, taps: int = 99) -> np.ndarray:
    origin = (taps - 1) // 2
    positions = np.arange(-origin, origin + 1, dtype=np.float64) - fraction
    # torch.blackman_window(taps), used by the frozen training renderer, is
    # periodic. Dropping the final sample of a symmetric taps+1 window gives
    # the same coefficient convention in NumPy.
    response = np.sinc(positions) * np.blackman(taps + 1)[:-1]
    response /= response.sum()
    return response


def _grid_directions(height: int, width: int) -> Tensor:
    elevation = torch.linspace(0.0, torch.pi, height)
    azimuth = torch.linspace(0.0, 2.0 * torch.pi, width)
    elevation, azimuth = torch.meshgrid(elevation, azimuth, indexing="ij")
    return torch.stack(
        [
            torch.sin(elevation) * torch.cos(azimuth),
            torch.sin(elevation) * torch.sin(azimuth),
            torch.cos(elevation),
        ],
        dim=-1,
    )


def _sphere_pad(value: Tensor) -> Tensor:
    value = F.pad(value, (1, 1, 0, 0), mode="circular")
    return F.pad(value, (0, 0, 1, 1), mode="replicate")


def _source_map(
    directions: np.ndarray,
    *,
    height: int,
    width: int,
    sigma_degrees: float,
    extended: bool,
) -> Tensor:
    values = torch.from_numpy(np.asarray(directions, dtype=np.float32))
    values = F.normalize(values, dim=-1)
    grid = F.normalize(_grid_directions(height, width), dim=-1)
    cosine = torch.einsum("kd,hwd->khw", values, grid).clamp(
        -1.0 + 1.0e-6, 1.0 - 1.0e-6
    )
    angular_distance = torch.acos(cosine)
    if not extended:
        sigma = torch.deg2rad(torch.tensor(float(sigma_degrees)))
        return torch.exp(-angular_distance[0].square() / (2.0 * sigma.square()))
    minimum = angular_distance.min(dim=0).values
    fill = torch.deg2rad(torch.tensor(2.0))
    edge = torch.deg2rad(torch.tensor(1.0))
    outside = (minimum - fill).clamp_min(0.0)
    support = torch.where(
        minimum <= fill,
        torch.ones_like(minimum),
        torch.exp(-outside.square() / (2.0 * edge.square())),
    )
    value = support[None, None]
    value = F.max_pool2d(_sphere_pad(value), kernel_size=3, stride=1)
    value = F.avg_pool2d(_sphere_pad(value), kernel_size=3, stride=1)
    return value[0, 0].clamp_(0.0, 1.0)


def _max_pool_time(value: Tensor, stride: int) -> Tensor:
    frames = int(value.shape[0])
    output_frames = (frames + stride - 1) // stride
    padding = output_frames * stride - frames
    if padding:
        value = torch.cat(
            [value, value.new_zeros((padding, *value.shape[1:]))], dim=0
        )
    return value.reshape(output_frames, stride, *value.shape[1:]).max(dim=1).values


class AcousticSceneRenderer:
    """Render the two-second Eigenmike scenes described in the paper."""

    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        segment_duration: float = 2.0,
        speed_of_sound_m_s: float = 343.0,
        target_standard_deviation_degrees: float = 4.0,
        maximum_sources: int = 6,
        maximum_reflection_order: int = 5,
        room_width_m: tuple[float, float] = (2.0, 10.0),
        room_length_m: tuple[float, float] = (2.0, 10.0),
        room_height_m: tuple[float, float] = (2.0, 4.0),
        wall_absorption: tuple[float, float] = (0.0, 0.5),
        source_extent_sizes_degrees: tuple[float, float, float] = (
            3.0,
            8.0,
            13.0,
        ),
        source_extent_probabilities: tuple[float, float, float] = (
            0.30,
            0.45,
            0.25,
        ),
        source_point_spacing_degrees: float = 4.0,
        noise_snr_db: tuple[float, float] = (10.0, 30.0),
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.segment_duration = float(segment_duration)
        self.samples = int(round(sample_rate * segment_duration))
        self.frames_100hz = int(round(segment_duration * 100.0)) + 1
        self.speed_of_sound_m_s = float(speed_of_sound_m_s)
        self.target_standard_deviation_degrees = float(
            target_standard_deviation_degrees
        )
        self.maximum_sources = int(maximum_sources)
        self.noise_snr_db = tuple(float(value) for value in noise_snr_db)
        if (
            len(self.noise_snr_db) != 2
            or self.noise_snr_db[0] > self.noise_snr_db[1]
        ):
            raise ValueError("noise_snr_db must contain an ordered [min,max] range")
        microphones = load_eigenmike_positions()
        self.room_model = ShoeboxImageSourceModel(
            microphones,
            room_width_m=room_width_m,
            room_length_m=room_length_m,
            room_height_m=room_height_m,
            wall_absorption=wall_absorption,
            maximum_reflection_order=maximum_reflection_order,
            sample_rate=sample_rate,
            point_spacing_degrees=source_point_spacing_degrees,
            extent_sizes_degrees=source_extent_sizes_degrees,
            extent_probabilities=source_extent_probabilities,
        )
        self.sphere_responses = rigid_sphere_response_table(
            sample_rate=sample_rate,
            speed_of_sound_m_s=speed_of_sound_m_s,
        )
        self.sphere_origin = self.sphere_responses.shape[-1] // 2

    def _path_rir(
        self,
        room: RoomSample,
        *,
        source_index: int,
        subsource_index: int,
        microphone_index: int,
    ) -> np.ndarray:
        images = room.image_positions_m[source_index][subsource_index][
            microphone_index
        ]
        orders = room.image_orders[source_index][subsource_index][
            microphone_index
        ]
        microphone = room.microphone_positions_m[microphone_index]
        direction = images - microphone[None]
        distance = np.linalg.norm(direction, axis=-1).clip(1.0e-8)
        unit = direction / distance[:, None]
        capsule = room.microphone_directions[microphone_index]
        angle = np.rint(
            np.rad2deg(
                np.arccos(np.clip(unit @ capsule, -1.0, 1.0))
            )
        ).astype(np.int64)
        sphere = self.sphere_responses[angle]
        delay = (
            distance / self.speed_of_sound_m_s * self.sample_rate
            + room.subsource_delays_ms[source_index][subsource_index]
            / 1000.0
            * self.sample_rate
        )
        integer = np.floor(delay).astype(np.int64)
        gain = (
            1.0
            / distance
            * np.sqrt(1.0 - room.wall_absorption) ** orders
            / np.sqrt(max(1, images.shape[0]))
            * room.subsource_gains[source_index][subsource_index]
        )
        rir = np.zeros(self.sample_rate, dtype=np.float64)
        fractional_origin = 49
        for path_index in range(images.shape[0]):
            fractional = _fractional_filter(
                float(delay[path_index] - integer[path_index])
            )
            response = np.convolve(
                sphere[path_index], fractional, mode="full"
            )[fractional_origin:]
            response *= float(gain[path_index])
            start = int(integer[path_index])
            if start >= rir.shape[0]:
                continue
            count = min(response.shape[0], rir.shape[0] - start)
            rir[start : start + count] += response[:count]
        return rir.astype(np.float32)

    def render(
        self,
        dry_sources: list[np.ndarray],
        class_ids: list[int],
        *,
        random: np.random.Generator,
        extended_sources: bool,
        activity_samples: np.ndarray | None = None,
        activity_frames_100hz: np.ndarray | None = None,
        add_noise: bool = False,
        target_kind: str = "both",
    ) -> RenderedScene:
        """Render one scene and construct Audio2Sph and SAID targets."""

        if target_kind not in {"audio2sph", "said", "both"}:
            raise ValueError("target_kind must be 'audio2sph', 'said', or 'both'")
        source_count = len(dry_sources)
        if source_count > self.maximum_sources:
            raise ValueError("source count exceeds the paper maximum")
        if len(class_ids) != source_count:
            raise ValueError("class_ids must contain one value per source")
        sources = np.zeros((source_count, self.samples), dtype=np.float32)
        for index, audio in enumerate(dry_sources):
            value = np.asarray(audio, dtype=np.float32).reshape(-1)
            if value.size < self.samples:
                repeats = (self.samples + value.size - 1) // max(1, value.size)
                value = np.tile(value, repeats)
            sources[index] = value[: self.samples]
        if activity_samples is None:
            activity_samples = np.ones_like(sources, dtype=np.float32)
        if activity_frames_100hz is None:
            activity_frames_100hz = np.ones(
                (source_count, self.frames_100hz), dtype=np.float32
            )
        activity_samples = np.asarray(activity_samples, dtype=np.float32)
        activity_frames_100hz = np.asarray(
            activity_frames_100hz, dtype=np.float32
        )
        if activity_samples.shape != sources.shape:
            raise ValueError("activity_samples must match dry source audio")
        if activity_frames_100hz.shape != (
            source_count,
            self.frames_100hz,
        ):
            raise ValueError("activity_frames_100hz has the wrong shape")
        room = self.room_model.sample(
            random,
            source_count=source_count,
            extended_sources=extended_sources,
        )
        source_images = np.zeros(
            (source_count, 4, self.samples), dtype=np.float32
        )
        for source_index in range(source_count):
            subsource_images = np.zeros(
                (
                    len(room.subsource_positions_m[source_index]),
                    4,
                    self.samples,
                ),
                dtype=np.float32,
            )
            for subsource_index in range(subsource_images.shape[0]):
                for microphone_index in range(4):
                    rir = self._path_rir(
                        room,
                        source_index=source_index,
                        subsource_index=subsource_index,
                        microphone_index=microphone_index,
                    )
                    convolved = fftconvolve(
                        sources[source_index], rir, mode="full"
                    )
                    subsource_images[subsource_index, microphone_index] = (
                        convolved[
                            self.sphere_origin : self.sphere_origin + self.samples
                        ]
                    )
            source_energy = float(np.mean(sources[source_index] ** 2))
            rendered_energy = float(np.mean(subsource_images**2))
            if source_energy > 0.0 and rendered_energy > 0.0:
                subsource_images *= np.sqrt(source_energy / rendered_energy)
            source_images[source_index] = (
                subsource_images.sum(axis=0)
                * activity_samples[source_index][None]
            )
        audio = source_images.sum(axis=0)
        if add_noise and audio.size:
            signal_rms = float(np.sqrt(np.mean(audio**2) + 1.0e-8))
            snr = float(random.uniform(*self.noise_snr_db))
            noise_rms = signal_rms / (10.0 ** (snr / 20.0))
            audio += random.normal(0.0, noise_rms, audio.shape).astype(np.float32)

        local_center = (
            room.source_centers_m - room.array_center_m
        ) @ room.array_rotation
        maps45: list[Tensor] = []
        maps180: list[Tensor] = []
        for source_index in range(source_count):
            source_is_extended = bool(room.source_is_extended[source_index])
            local_subsources = (
                room.subsource_positions_m[source_index] - room.array_center_m
            ) @ room.array_rotation
            if target_kind in {"said", "both"}:
                maps45.append(
                    _source_map(
                        local_subsources
                        if source_is_extended
                        else local_center[source_index : source_index + 1],
                        height=45,
                        width=90,
                        sigma_degrees=self.target_standard_deviation_degrees,
                        extended=source_is_extended,
                    )
                )
            maps180.append(
                _source_map(
                    local_subsources
                    if source_is_extended
                    else local_center[source_index : source_index + 1],
                    height=180,
                    width=360,
                    sigma_degrees=self.target_standard_deviation_degrees,
                    extended=source_is_extended,
                )
            )
        source_maps45 = (
            torch.stack(maps45) if maps45 else torch.zeros(0, 45, 90)
        )
        source_maps180 = (
            torch.stack(maps180)
            if maps180
            else torch.zeros(0, 180, 360)
        )
        activity100 = torch.from_numpy(activity_frames_100hz)
        if source_count and target_kind in {"audio2sph", "both"}:
            audio2sph_target = (
                source_maps180[:, None]
                * activity100[:, :, None, None]
            ).max(dim=0).values
        elif target_kind in {"audio2sph", "both"}:
            audio2sph_target = torch.zeros(
                self.frames_100hz, 180, 360
            )
        else:
            audio2sph_target = torch.zeros(0, 180, 360)
        if target_kind in {"said", "both"}:
            activity10 = _max_pool_time(
                activity100.transpose(0, 1), 10
            ).transpose(0, 1)
            output_frames = int(activity10.shape[1])
            valid = torch.zeros(
                output_frames, self.maximum_sources, dtype=torch.bool
            )
            output_classes = torch.full(
                (output_frames, self.maximum_sources), -1, dtype=torch.long
            )
            output_maps = torch.zeros(
                output_frames, self.maximum_sources, 45, 90
            )
            refined = torch.zeros(
                output_frames, self.maximum_sources, 180, 360
            )
            activity_score = activity10.sum(dim=1)
            order = torch.argsort(activity_score, descending=True, stable=True)
            for slot, source_index_tensor in enumerate(
                order[: self.maximum_sources]
            ):
                source_index = int(source_index_tensor)
                source_activity = activity10[source_index] > 0.5
                valid[:, slot] = source_activity
                output_classes[source_activity, slot] = int(
                    class_ids[source_index]
                )
                output_maps[:, slot] = (
                    source_maps45[source_index][None]
                    * source_activity[:, None, None]
                )
                refined[:, slot] = (
                    source_maps180[source_index][None]
                    * source_activity[:, None, None]
                )
        else:
            valid = torch.zeros(0, self.maximum_sources, dtype=torch.bool)
            output_classes = torch.full(
                (0, self.maximum_sources), -1, dtype=torch.long
            )
            output_maps = torch.zeros(0, self.maximum_sources, 45, 90)
            refined = torch.zeros(0, self.maximum_sources, 180, 360)
        return RenderedScene(
            audio=torch.from_numpy(audio),
            source_images=torch.from_numpy(source_images),
            activity_samples=torch.from_numpy(activity_samples),
            activity_frames_100hz=activity100,
            class_ids=torch.tensor(class_ids, dtype=torch.long),
            source_maps_45x90=source_maps45,
            source_maps_180x360=source_maps180,
            audio2sph_target_180x360=audio2sph_target,
            valid_sources=valid,
            said_class_ids=output_classes,
            said_map_targets=output_maps,
            said_refined_map_targets=refined,
            frame_alignment=FrameAlignment.EXACT,
            room=room,
        )
