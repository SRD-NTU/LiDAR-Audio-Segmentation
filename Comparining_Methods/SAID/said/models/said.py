"""Top-level Semantic Acoustic Imaging Detector."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .audio2sph import Audio2Sph, Audio2SphOutput
from .class_feature_encoders import (
    AudioMAEClassFeatureEncoder,
    PaSSTClassFeatureEncoder,
)
from .sph2imaging import Sph2Imaging, Sph2ImagingOutput


@dataclass(frozen=True)
class LabeledAcousticMaps:
    """Maps, zero-based class IDs, and confidence jointly forming predictions."""

    refined_maps: Tensor
    slot_class_ids: Tensor
    slot_confidence: Tensor


@dataclass(frozen=True)
class SAIDOutput:
    """Final labeled acoustic maps and inspectable model outputs."""

    labeled_acoustic_maps: LabeledAcousticMaps
    audio2sph: Audio2SphOutput
    sph2imaging: Sph2ImagingOutput

    @property
    def refined_maps(self) -> Tensor:
        return self.labeled_acoustic_maps.refined_maps

    @property
    def slot_class_ids(self) -> Tensor:
        return self.labeled_acoustic_maps.slot_class_ids

    @property
    def slot_confidence(self) -> Tensor:
        return self.labeled_acoustic_maps.slot_confidence


class RecordingDomain(str, Enum):
    """Recording domains represented by the paper's PaSST checkpoint."""

    SONY = "sony"
    TAU = "tau"


def _recording_domain_ids(
    recording_domain: RecordingDomain | str | Sequence[RecordingDomain | str] | None,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor | None:
    if recording_domain is None:
        return None
    if isinstance(recording_domain, (str, RecordingDomain)):
        values = [recording_domain] * int(batch_size)
    else:
        values = list(recording_domain)
        if len(values) != int(batch_size):
            raise ValueError(
                "recording_domain must contain one value per input recording"
            )
    ids = []
    for value in values:
        try:
            normalized = value.value if isinstance(value, RecordingDomain) else str(value).lower()
            domain = RecordingDomain(normalized)
        except ValueError as error:
            raise ValueError("recording_domain must be 'sony' or 'tau'") from error
        ids.append(0 if domain is RecordingDomain.SONY else 1)
    return torch.tensor(ids, dtype=torch.long, device=device)


class SAID(nn.Module):
    """Semantic Acoustic Imaging Detector reported in the paper."""

    def __init__(self, *, class_feature_encoder: str = "passt") -> None:
        super().__init__()
        encoder_name = str(class_feature_encoder).lower()
        if encoder_name == "passt":
            encoder = PaSSTClassFeatureEncoder()
        elif encoder_name == "audiomae":
            encoder = AudioMAEClassFeatureEncoder()
        else:
            raise ValueError(
                "class_feature_encoder must be 'passt' or 'audiomae'"
            )
        self.class_feature_encoder_name = encoder_name
        self.audio2sph = Audio2Sph()
        self.sph2imaging = Sph2Imaging(encoder)

    def forward(
        self,
        audio: Tensor,
        *,
        recording_domain: (
            RecordingDomain | str | Sequence[RecordingDomain | str] | None
        ) = RecordingDomain.SONY,
    ) -> SAIDOutput:
        domain_id = _recording_domain_ids(
            recording_domain,
            batch_size=int(audio.shape[0]),
            device=audio.device,
        )
        audio2sph_output = self.audio2sph(audio)
        sph2imaging_output = self.sph2imaging(
            audio2sph_output.panoramic_features,
            audio=audio,
            x2=audio2sph_output.x2,
            x3=audio2sph_output.x3,
            domain_id=domain_id,
        )
        return SAIDOutput(
            labeled_acoustic_maps=LabeledAcousticMaps(
                refined_maps=sph2imaging_output.refined_maps,
                slot_class_ids=sph2imaging_output.slot_class_ids,
                slot_confidence=sph2imaging_output.slot_confidence,
            ),
            audio2sph=audio2sph_output,
            sph2imaging=sph2imaging_output,
        )
