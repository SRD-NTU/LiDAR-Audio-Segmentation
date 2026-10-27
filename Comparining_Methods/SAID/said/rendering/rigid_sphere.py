"""Rigid-sphere spatial impulse responses for the Eigenmike array.

The implementation follows the spherical-array formulation used by NESD and
Rafaely, *Fundamentals of Spherical Array Processing* (2015).
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 CUHK
# Original implementation contributors: Yin Cao and Qiuqiang Kong
# SAID adaptation and modifications: Copyright (c) 2026 Runbang Wang
# Upstream NESD revision:
# https://github.com/qiuqiangkong/nesd/tree/7475da725dcfc8d60761c5891644ca326a7226e7

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import lpmv, spherical_jn, spherical_yn


def _spherical_hankel_second_kind(
    order: int, value: np.ndarray, *, derivative: bool = False
) -> np.ndarray:
    return spherical_jn(order, value, derivative) - 1j * spherical_yn(
        order, value, derivative
    )


def _mode_strength(order: int, kr: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        hankel = _spherical_hankel_second_kind(order, kr)
        derivative = _spherical_hankel_second_kind(
            order, kr, derivative=True
        )
        correction = (
            spherical_jn(order, kr, derivative=True) / derivative
        ) * hankel
    correction[0] = 0.0
    return 4.0 * np.pi * (1j**order) * (
        spherical_jn(order, kr) - correction
    )


def _fractional_delay_filter(
    delay_samples: float, *, taps: int = 199
) -> tuple[int, np.ndarray, int]:
    if taps <= 0 or taps % 2 == 0:
        raise ValueError("fractional-delay taps must be a positive odd integer")
    integer = int(np.floor(delay_samples))
    fraction = float(delay_samples) - integer
    origin = (taps - 1) // 2
    positions = (
        np.arange(-origin, origin + 1, dtype=np.float32)
        - np.float32(fraction)
    )
    # torch.blackman_window, used by the frozen renderer, is periodic by
    # default.  Dropping the final sample of a symmetric N+1 window matches it.
    window = np.blackman(taps + 1)[:-1].astype(np.float32)
    impulse = np.sinc(positions).astype(np.float32) * window
    impulse /= impulse.sum()
    return integer, impulse, origin


def _delay_centered_response(
    response: np.ndarray, delay_samples: float
) -> np.ndarray:
    integer, fractional, origin = _fractional_delay_filter(delay_samples)
    response = np.asarray(response, dtype=np.float32)
    fractional = np.asarray(fractional, dtype=np.float32)
    convolved = np.convolve(response, fractional, mode="full")[origin:]
    convolved = convolved[: response.shape[0]]
    output = np.zeros_like(response, dtype=np.float32)
    start = max(0, integer)
    source_start = max(0, -integer)
    count = min(
        output.shape[0] - start,
        convolved.shape[0] - source_start,
    )
    if count > 0:
        output[start : start + count] = convolved[
            source_start : source_start + count
        ]
    return output


def rigid_sphere_impulse_response(
    angle_degrees: float,
    *,
    sample_rate: int = 48_000,
    radius_m: float = 0.042,
    speed_of_sound_m_s: float = 343.0,
    fft_size: int = 256,
    maximum_spherical_order: int = 30,
) -> np.ndarray:
    """Compute one far-field rigid-sphere response at an incidence angle."""

    if not 0.0 <= float(angle_degrees) <= 180.0:
        raise ValueError("angle_degrees must lie in [0,180]")
    frequency = np.linspace(
        0.0, sample_rate / 2.0, fft_size // 2 + 1, dtype=np.float64
    )
    kr = 2.0 * np.pi * frequency * radius_m / speed_of_sound_m_s
    angle = np.deg2rad(float(angle_degrees))
    spectrum = np.zeros(frequency.shape, dtype=np.complex128)
    for order in range(maximum_spherical_order + 1):
        spectrum += (
            _mode_strength(order, kr)
            / (4.0 * np.pi)
            * (2 * order + 1)
            * lpmv(0, order, np.cos(angle))
        )
    response = np.fft.fftshift(np.fft.irfft(spectrum, n=fft_size))
    geometric_delay = (
        radius_m
        * np.cos(angle)
        / speed_of_sound_m_s
        * sample_rate
    )
    return _delay_centered_response(response, geometric_delay).astype(
        np.float32
    )


@lru_cache(maxsize=4)
def rigid_sphere_response_table(
    *,
    sample_rate: int = 48_000,
    radius_m: float = 0.042,
    speed_of_sound_m_s: float = 343.0,
    fft_size: int = 256,
    maximum_spherical_order: int = 30,
) -> np.ndarray:
    """Return responses for rounded incidence angles from 0 to 180 degrees."""

    table = np.stack(
        [
            rigid_sphere_impulse_response(
                angle,
                sample_rate=sample_rate,
                radius_m=radius_m,
                speed_of_sound_m_s=speed_of_sound_m_s,
                fft_size=fft_size,
                maximum_spherical_order=maximum_spherical_order,
            )
            for angle in range(181)
        ]
    )
    table.setflags(write=False)
    return table


def responses_for_directions(
    microphone_directions: np.ndarray,
    arrival_directions: np.ndarray,
    response_table: np.ndarray,
) -> np.ndarray:
    """Select rigid-sphere filters by rounded capsule/path incidence angle."""

    microphone = np.asarray(microphone_directions, dtype=np.float64)
    arrival = np.asarray(arrival_directions, dtype=np.float64)
    microphone /= np.linalg.norm(microphone, axis=-1, keepdims=True).clip(1e-12)
    arrival /= np.linalg.norm(arrival, axis=-1, keepdims=True).clip(1e-12)
    cosine = np.sum(microphone * arrival, axis=-1).clip(-1.0, 1.0)
    angle = np.rint(np.rad2deg(np.arccos(cosine))).astype(np.int64)
    return np.asarray(response_table)[angle]
