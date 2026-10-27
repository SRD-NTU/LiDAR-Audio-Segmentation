"""Class Feature Encoders used by Sph2Imaging."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from .audiomae import AudioMAEClassFeatureEncoder, align_audiomae_patch_tokens
from .passt import PaSSTClassFeatureEncoder

__all__ = [
    "PaSSTClassFeatureEncoder",
    "AudioMAEClassFeatureEncoder",
    "align_audiomae_patch_tokens",
]
