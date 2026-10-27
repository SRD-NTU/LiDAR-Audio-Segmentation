"""PaSST Class Feature Encoder for SAID."""

# SPDX-License-Identifier: Apache-2.0
# PaSST feature-extraction flow adapted from kkoutini/PaSST.
# Modifications and SAID integration: Copyright (c) 2026 Runbang Wang
# Upstream PaSST revision:
# https://github.com/kkoutini/PaSST/tree/2a5c818afcc2a215b2a1aaf1ed8be71f89d43201
#
# Changed from upstream: exposes frequency-pooled temporal patch features,
# aligns them to SAID frames, and applies the checkpoint's recording-domain
# embedding. See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import contextlib
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PaSSTClassFeatureEncoder(nn.Module):
    """Produce frame-aligned PaSST Class Features with shape ``[B,T,768]``."""

    output_dim = 768
    input_sample_rate = 48000
    backbone_sample_rate = 32000

    def __init__(self, *, freeze: bool = True) -> None:
        super().__init__()
        try:
            from hear21passt.base import PasstBasicWrapper
            from hear21passt.models.passt import get_model
            from hear21passt.models.preprocess import AugmentMelSTFT
        except ImportError as error:  # pragma: no cover - dependency guard
            raise ImportError("PaSSTClassFeatureEncoder requires hear21passt") from error

        with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
            net = get_model(
                arch="passt_s_swa_p16_128_ap476",
                pretrained=False,
                n_classes=527,
                in_channels=1,
                fstride=10,
                tstride=10,
                input_fdim=128,
                input_tdim=998,
                u_patchout=0,
                s_patchout_t=0,
                s_patchout_f=0,
            )
        mel = AugmentMelSTFT(
            n_mels=128,
            sr=32000,
            win_length=800,
            hopsize=320,
            n_fft=1024,
            freqm=48,
            timem=192,
            htk=False,
            fmin=0.0,
            # This is exactly the value selected by hear21passt when fmax is
            # omitted, supplied explicitly to avoid its misleading warning.
            fmax=15000.0,
            norm=1,
            fmin_aug_range=10,
            fmax_aug_range=2000,
        )
        self.backbone = PasstBasicWrapper(mel=mel, net=net, mode="embed_only")
        self.domain_embedding = nn.Embedding(2, 768)
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def frame_tokens_kind() -> str:
        return "frame_aligned"

    def _resample(self, audio: Tensor) -> Tensor:
        target_length = max(
            1,
            int(round(audio.shape[-1] * self.backbone_sample_rate / self.input_sample_rate)),
        )
        flat = audio.reshape(-1, 1, audio.shape[-1]).float()
        output = F.interpolate(flat, size=target_length, mode="linear", align_corners=False)
        return output.reshape(audio.shape[0], audio.shape[1], target_length).to(dtype=audio.dtype)

    def frame_tokens(
        self,
        audio: Tensor,
        *,
        domain_id: Tensor | None = None,
        target_frames: int | None = None,
    ) -> Tensor:
        if audio.ndim != 3:
            raise ValueError(f"PaSST expects audio [B,C,T], got {tuple(audio.shape)}")
        was_training = self.training
        self.eval()
        grad_enabled = any(parameter.requires_grad for parameter in self.parameters())
        try:
            with torch.set_grad_enabled(grad_enabled):
                wrapper = self.backbone
                net = wrapper.net
                mono = self._resample(audio).mean(dim=1)
                try:
                    import hear21passt.models.passt as passt_module

                    passt_module.first_RUN = False
                except ImportError:  # pragma: no cover - dependency guard
                    pass
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"stft with return_complex=False is deprecated\..*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\..*",
                        category=FutureWarning,
                    )
                    with (
                        open(os.devnull, "w", encoding="utf-8") as sink,
                        contextlib.redirect_stdout(sink),
                    ):
                        spectrogram = wrapper.mel(mono).unsqueeze(1)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Input image size .* doesn't match model .*",
                        category=UserWarning,
                    )
                    x = net.patch_embed(spectrogram)
                batch_size, _, frequency_dim, time_dim = x.shape
                time_position = net.time_new_pos_embed
                if x.shape[-1] != time_position.shape[-1]:
                    time_position = time_position[:, :, :, : x.shape[-1]]
                x = x + time_position + net.freq_new_pos_embed
                x = x.flatten(2).transpose(1, 2)
                class_tokens = (
                    net.cls_token.expand(batch_size, -1, -1)
                    + net.new_pos_embed[:, :1, :]
                )
                if net.dist_token is None:
                    x = torch.cat((class_tokens, x), dim=1)
                    patch_offset = 1
                else:
                    distillation_token = (
                        net.dist_token.expand(batch_size, -1, -1)
                        + net.new_pos_embed[:, 1:, :]
                    )
                    x = torch.cat((class_tokens, distillation_token, x), dim=1)
                    patch_offset = 2
                x = net.pos_drop(x)
                x = net.blocks(x)
                x = net.norm(x)
                tokens = x[:, patch_offset:, :].reshape(
                    batch_size, frequency_dim, time_dim, self.output_dim
                ).mean(dim=1)
                if domain_id is not None:
                    tokens = tokens + self.domain_embedding(domain_id)[:, None, :]
                tokens = tokens.float()
                if target_frames is not None and int(tokens.shape[1]) != int(target_frames):
                    tokens = F.interpolate(
                        tokens.transpose(1, 2),
                        size=int(target_frames),
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2).contiguous()
        finally:
            self.train(was_training)
        return tokens

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
