"""AudioMAE Class Feature Encoder for SAID."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang
# Exact AudioMAE initialization revision used by the paper model:
# https://huggingface.co/gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k/tree/49cfdfa4663a9c689e7ee5184df2d5660f730233
# Original AudioMAE project: https://github.com/facebookresearch/AudioMAE

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def align_audiomae_patch_tokens(
    tokens: Tensor,
    *,
    valid_fbank_frames: Tensor,
    waveform_duration_sec: Tensor,
    target_frames: int | None,
) -> Tensor:
    """Align valid AudioMAE patch rows to detector-frame coordinates.

    AudioMAE receives a 1024-by-128 fbank and produces a 64-by-8 patch grid.
    Rows intersecting real fbank frames are frequency-pooled and assigned their
    physical center time. A final partial row uses the center of its real frames.
    Detector-frame values are interpolated on those physical coordinates.
    """

    if tokens.ndim != 3:
        raise ValueError(
            f"AudioMAE patch tokens must be [B,P,C], got {tuple(tokens.shape)}"
        )
    time_rows = 64
    frequency_columns = 8
    patch_frames = 16
    expected_patches = time_rows * frequency_columns
    if int(tokens.shape[1]) != expected_patches:
        raise ValueError(
            f"AudioMAE expects {expected_patches} patch tokens, "
            f"got {int(tokens.shape[1])}"
        )
    batch_size = int(tokens.shape[0])
    if valid_fbank_frames.ndim != 1 or int(valid_fbank_frames.shape[0]) != batch_size:
        raise ValueError(
            "valid_fbank_frames must be [B] matching AudioMAE tokens, "
            f"got {tuple(valid_fbank_frames.shape)} for B={batch_size}"
        )
    if (
        waveform_duration_sec.ndim != 1
        or int(waveform_duration_sec.shape[0]) != batch_size
    ):
        raise ValueError(
            "waveform_duration_sec must be [B] matching AudioMAE tokens, "
            f"got {tuple(waveform_duration_sec.shape)} for B={batch_size}"
        )

    time_tokens = tokens.reshape(
        batch_size, time_rows, frequency_columns, tokens.shape[-1]
    ).mean(dim=2)
    if target_frames is None:
        return time_tokens
    if int(target_frames) <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")

    frame_center0_sec = 0.0125
    frame_hop_sec = 0.010
    aligned: list[Tensor] = []
    for batch_idx in range(batch_size):
        valid_count = int(valid_fbank_frames[batch_idx].item())
        valid_count = max(0, min(valid_count, time_rows * patch_frames))
        if valid_count <= 0:
            raise ValueError(
                "AudioMAE time alignment requires at least one valid fbank frame"
            )
        valid_rows = (valid_count + patch_frames - 1) // patch_frames
        row_ids = torch.arange(valid_rows, device=tokens.device, dtype=torch.long)
        row_starts = row_ids * patch_frames
        row_counts = torch.minimum(
            torch.full_like(row_starts, patch_frames),
            torch.full_like(row_starts, valid_count) - row_starts,
        )
        row_centers = float(frame_center0_sec) + float(frame_hop_sec) * (
            row_starts.to(dtype=torch.float32)
            + (row_counts.to(dtype=torch.float32) - 1.0) * 0.5
        )
        duration = float(waveform_duration_sec[batch_idx].item())
        if duration <= 0.0:
            raise ValueError(
                f"waveform_duration_sec must be positive, got {duration}"
            )
        target_times = torch.linspace(
            0.0,
            duration,
            steps=int(target_frames),
            device=tokens.device,
            dtype=torch.float32,
        )
        right = torch.searchsorted(row_centers, target_times, right=False)
        right = right.clamp(0, valid_rows - 1)
        left = (right - 1).clamp(0, valid_rows - 1)
        left_t = row_centers[left]
        right_t = row_centers[right]
        denominator = right_t - left_t
        weight = torch.where(
            denominator > 0,
            (target_times - left_t) / denominator.clamp_min(1.0e-12),
            torch.zeros_like(target_times),
        ).clamp(0.0, 1.0)
        source = time_tokens[batch_idx, :valid_rows]
        aligned.append(
            source[left] * (1.0 - weight[:, None])
            + source[right] * weight[:, None]
        )
    return torch.stack(aligned, dim=0)


class _AudioMAEBackbone(nn.Module):
    target_sample_rate = 16000
    target_frames = 1024
    num_mel_bins = 128
    audioset_mean = -4.2677393
    audioset_std = 4.5689974

    def __init__(self) -> None:
        super().__init__()
        try:
            from timm.models.vision_transformer import VisionTransformer
        except ImportError as error:  # pragma: no cover - dependency guard
            raise ImportError("AudioMAEClassFeatureEncoder requires timm") from error
        self.audio_transformer = VisionTransformer(
            img_size=(1024, 128),
            patch_size=16,
            in_chans=1,
            num_classes=527,
            global_pool="avg",
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
        )

    @staticmethod
    def _to_mono_16k(audio: Tensor, source_sample_rate: int) -> Tensor:
        if audio.ndim != 3:
            raise ValueError(f"AudioMAE expects audio [B,C,T], got {tuple(audio.shape)}")
        if int(source_sample_rate) <= 0:
            raise ValueError(f"source_sample_rate must be positive, got {source_sample_rate}")
        mono = audio.float().mean(dim=1)
        if int(source_sample_rate) == _AudioMAEBackbone.target_sample_rate:
            return mono
        target_length = max(
            1,
            int(
                round(
                    mono.shape[-1]
                    * _AudioMAEBackbone.target_sample_rate
                    / int(source_sample_rate)
                )
            ),
        )
        try:
            import torchaudio.functional as audio_functional

            return audio_functional.resample(
                mono,
                orig_freq=int(source_sample_rate),
                new_freq=_AudioMAEBackbone.target_sample_rate,
            )
        except (ImportError, RuntimeError):
            return F.interpolate(
                mono[:, None], size=target_length, mode="linear", align_corners=False
            )[:, 0]

    def _waveform_to_fbank(self, mono_16k: Tensor) -> tuple[Tensor, Tensor]:
        try:
            from torchaudio.compliance import kaldi
        except ImportError as error:  # pragma: no cover - dependency guard
            raise ImportError("AudioMAEClassFeatureEncoder requires torchaudio") from error
        rows: list[Tensor] = []
        valid_frame_counts: list[int] = []
        for waveform in mono_16k:
            if waveform.numel() < 400:
                waveform = F.pad(waveform, (0, 400 - int(waveform.numel())))
            fbank = kaldi.fbank(
                waveform.unsqueeze(0),
                htk_compat=True,
                sample_frequency=self.target_sample_rate,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=self.num_mel_bins,
                dither=0.0,
            )
            valid_frame_counts.append(min(int(fbank.shape[0]), self.target_frames))
            if fbank.shape[0] < self.target_frames:
                fbank = F.pad(fbank, (0, 0, 0, self.target_frames - fbank.shape[0]))
            else:
                fbank = fbank[: self.target_frames]
            rows.append((fbank - self.audioset_mean) / (self.audioset_std * 2.0))
        spectrogram = torch.stack(rows, dim=0)[:, None]
        valid_frames = torch.as_tensor(
            valid_frame_counts, device=spectrogram.device, dtype=torch.long
        )
        return spectrogram, valid_frames

    def patch_tokens(
        self, audio: Tensor, *, source_sample_rate: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        mono = self._to_mono_16k(audio, source_sample_rate)
        spectrogram, valid_frames = self._waveform_to_fbank(mono)
        model_device = next(self.audio_transformer.parameters()).device
        spectrogram = spectrogram.to(device=model_device)
        valid_frames = valid_frames.to(device=model_device)
        duration = torch.full(
            (mono.shape[0],),
            float(mono.shape[-1]) / float(self.target_sample_rate),
            device=model_device,
            dtype=torch.float32,
        )
        features = self.audio_transformer.forward_features(spectrogram)
        if not isinstance(features, Tensor) or features.ndim != 3:
            raise RuntimeError("AudioMAE forward_features must return [B,P,C] tokens")
        prefix_tokens = int(getattr(self.audio_transformer, "num_prefix_tokens", 1))
        return features[:, prefix_tokens:, :].float(), valid_frames, duration


class AudioMAEClassFeatureEncoder(nn.Module):
    """Produce physically aligned AudioMAE Class Features ``[B,T,768]``."""

    output_dim = 768
    input_sample_rate = 48000

    def __init__(self, *, freeze: bool = True) -> None:
        super().__init__()
        self.backbone = _AudioMAEBackbone()
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def frame_tokens_kind() -> str:
        return "frame_aligned"

    def frame_tokens(
        self,
        audio: Tensor,
        *,
        domain_id: Tensor | None = None,
        target_frames: int | None = None,
    ) -> Tensor:
        del domain_id
        was_training = self.training
        self.eval()
        grad_enabled = any(parameter.requires_grad for parameter in self.parameters())
        try:
            with torch.set_grad_enabled(grad_enabled):
                tokens, valid_frames, duration = self.backbone.patch_tokens(
                    audio, source_sample_rate=self.input_sample_rate
                )
                output = align_audiomae_patch_tokens(
                    tokens,
                    valid_fbank_frames=valid_frames,
                    waveform_duration_sec=duration,
                    target_frames=target_frames,
                ).float()
        finally:
            self.train(was_training)
        return output

    def forward(
        self,
        audio: Tensor,
        *,
        domain_id: Tensor | None = None,
        target_frames: int | None = None,
    ) -> Tensor:
        return self.frame_tokens(
            audio, domain_id=domain_id, target_frames=target_frames
        )
