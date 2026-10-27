"""Paper microphone-array geometry for Online Scene Generation."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 CUHK
# Original NESD implementation: Qiuqiang Kong
# SAID adaptation and modifications: Copyright (c) 2026 Runbang Wang
# Upstream NESD revision:
# https://github.com/qiuqiangkong/nesd/tree/7475da725dcfc8d60761c5891644ca326a7226e7

from __future__ import annotations

import csv
from importlib import resources
from pathlib import Path

import numpy as np


PAPER_CAPSULE_INDICES_1BASED = (6, 10, 26, 22)


def load_eigenmike_positions(
    path: str | Path | None = None,
    *,
    capsule_indices_1based: tuple[int, ...] = PAPER_CAPSULE_INDICES_1BASED,
) -> np.ndarray:
    """Load Eigenmike capsule positions as Cartesian coordinates in metres."""

    resource = resources.files("said").joinpath("arrays/eigenmike32.csv")
    if path is None:
        with resources.as_file(resource) as packaged:
            return load_eigenmike_positions(
                packaged,
                capsule_indices_1based=capsule_indices_1based,
            )
    rows: dict[int, tuple[float, float, float]] = {}
    with Path(path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            index = int(row["microphone"])
            rows[index] = (
                float(row["radius"]),
                float(row["theta"]),
                float(row["phi"]),
            )
    missing = set(capsule_indices_1based) - set(rows)
    if missing:
        raise ValueError(f"Eigenmike geometry is missing capsules {sorted(missing)}")
    spherical = np.asarray(
        [rows[index] for index in capsule_indices_1based], dtype=np.float64
    )
    radius = spherical[:, 0]
    theta = np.deg2rad(spherical[:, 1])
    phi = np.deg2rad(spherical[:, 2])
    return np.stack(
        [
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        ],
        axis=-1,
    )
