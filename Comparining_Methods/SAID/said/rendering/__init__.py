"""Online Scene Generation for the paper-aligned SAID training pipeline."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from .rigid_sphere import (
    responses_for_directions,
    rigid_sphere_impulse_response,
    rigid_sphere_response_table,
)
from .renderer import AcousticSceneRenderer, RenderedScene
from .eigenmike import PAPER_CAPSULE_INDICES_1BASED, load_eigenmike_positions
from .shoebox_ism import RoomSample, ShoeboxImageSourceModel

__all__ = [
    "RoomSample",
    "ShoeboxImageSourceModel",
    "responses_for_directions",
    "rigid_sphere_impulse_response",
    "rigid_sphere_response_table",
    "AcousticSceneRenderer",
    "RenderedScene",
    "PAPER_CAPSULE_INDICES_1BASED",
    "load_eigenmike_positions",
]
