"""Compatibility boundary for submitted and public-runtime weights."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch


_SUBMITTED_CHECKPOINT_PREFIXES: dict[str, tuple[str, ...]] = {
    "Audio2Sph": (
        "stem.",
        "downsample_layers.",
        "stages.",
        "decoder.cross_s2.",
        "decoder.cross_s3.",
        "decoder.cross_s4.",
        "decoder.time_layers.",
    ),
    "Panoramic Decoder": (
        "decoder.refine0.",
        "decoder.refine1.",
        "decoder.refine2.",
        "decoder.head.",
    ),
    "Sph2Imaging": ("mask2former_head.",),
    "Class Feature Encoder": ("external_sed_prior.",),
}

_EXACT_COMPONENT_KEYS = {"decoder.sphere_query": "Audio2Sph"}


@dataclass(frozen=True)
class PaperCheckpoint:
    """Immutable identity and schema summary for a submitted checkpoint."""

    model_name: str
    class_feature_encoder: str
    filename: str
    sha256: str
    size_bytes: int
    key_count: int
    component_counts: Mapping[str, int]


PAPER_CHECKPOINTS: dict[str, PaperCheckpoint] = {
    "passt": PaperCheckpoint(
        model_name="SAID (PaSST)",
        class_feature_encoder="passt",
        filename="said_passt.ckpt",
        sha256="85b72bd4a0749dd4ca080ddea30eba68a903af465d916408a03e21912609ca1b",
        size_bytes=547_886_573,
        key_count=757,
        component_counts={
            "Audio2Sph": 233,
            "Panoramic Decoder": 12,
            "Sph2Imaging": 347,
            "Class Feature Encoder": 165,
        },
    ),
    "audiomae": PaperCheckpoint(
        model_name="SAID (AudioMAE)",
        class_feature_encoder="audiomae",
        filename="said_audiomae.ckpt",
        sha256="018f9d05bfa681456334811abe6e62c59331a0a8f4313eb763e5c32b0148c626",
        size_bytes=549_046_019,
        key_count=750,
        component_counts={
            "Audio2Sph": 233,
            "Panoramic Decoder": 12,
            "Sph2Imaging": 347,
            "Class Feature Encoder": 158,
        },
    ),
}


AUDIO2SPH_PAPER_CHECKPOINT = PaperCheckpoint(
    model_name="Audio2Sph + Panoramic Decoder",
    class_feature_encoder="none",
    filename="audio2sph.ckpt",
    sha256="bd4c2c44e24d4c207785a46cfc95efffc3f9a7a9a357d4563553a886be0b917b",
    size_bytes=110_233_837,
    key_count=245,
    component_counts={
        "Audio2Sph": 233,
        "Panoramic Decoder": 12,
        "Sph2Imaging": 0,
        "Class Feature Encoder": 0,
    },
)


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint violates the submitted-model contract."""


def checkpoint_component(key: str) -> str:
    """Return the paper component that owns a submitted checkpoint key."""

    if not isinstance(key, str):
        raise CheckpointCompatibilityError(
            f"checkpoint keys must be strings, got {type(key).__name__}"
        )
    exact = _EXACT_COMPONENT_KEYS.get(key)
    if exact is not None:
        return exact
    for component, prefixes in _SUBMITTED_CHECKPOINT_PREFIXES.items():
        if key.startswith(prefixes):
            return component
    raise CheckpointCompatibilityError(f"unrecognized checkpoint key: {key}")


def summarize_checkpoint_components(state_dict: Mapping[str, Any]) -> dict[str, int]:
    """Count keys by paper component and reject unowned parameter paths."""

    counts = {
        "Audio2Sph": 0,
        "Panoramic Decoder": 0,
        "Sph2Imaging": 0,
        "Class Feature Encoder": 0,
    }
    unknown: list[str] = []
    for key in state_dict:
        if not isinstance(key, str):
            raise CheckpointCompatibilityError(
                f"checkpoint keys must be strings, got {type(key).__name__}"
            )
        try:
            counts[checkpoint_component(key)] += 1
        except CheckpointCompatibilityError:
            unknown.append(key)

    if unknown:
        preview = ", ".join(unknown[:5])
        raise CheckpointCompatibilityError(
            f"checkpoint contains {len(unknown)} unrecognized keys: {preview}"
        )

    return counts


def validate_submitted_checkpoint_summary(
    state_dict: Mapping[str, Any], class_feature_encoder: str
) -> dict[str, int]:
    """Validate published key counts before strict model compatibility checks."""

    try:
        expected = PAPER_CHECKPOINTS[class_feature_encoder]
    except KeyError as error:
        raise CheckpointCompatibilityError(
            f"unknown class feature encoder: {class_feature_encoder}"
        ) from error
    counts = summarize_checkpoint_components(state_dict)
    if len(state_dict) != expected.key_count:
        raise CheckpointCompatibilityError(
            f"checkpoint has {len(state_dict)} keys; expected {expected.key_count}"
        )
    if counts != dict(expected.component_counts):
        raise CheckpointCompatibilityError(
            f"checkpoint component counts {counts}; expected {dict(expected.component_counts)}"
        )
    return counts


def validate_audio2sph_checkpoint_summary(
    state_dict: Mapping[str, Any],
) -> dict[str, int]:
    """Validate the published Audio2Sph and Panoramic Decoder state schema."""

    expected = AUDIO2SPH_PAPER_CHECKPOINT
    counts = summarize_checkpoint_components(state_dict)
    if len(state_dict) != expected.key_count:
        raise CheckpointCompatibilityError(
            f"checkpoint has {len(state_dict)} keys; expected {expected.key_count}"
        )
    if counts != dict(expected.component_counts):
        raise CheckpointCompatibilityError(
            f"checkpoint component counts {counts}; "
            f"expected {dict(expected.component_counts)}"
        )
    return counts


def _runtime_key_for_submitted_key(
    key: str, class_feature_encoder: str
) -> str | None:
    """Translate one frozen serialized identifier into a paper-aligned name."""

    if key.startswith("stem."):
        return f"audio2sph.{key}"
    if key.startswith("downsample_layers."):
        return f"audio2sph.{key}"
    if key.startswith("stages."):
        return f"audio2sph.{key}"
    if key == "decoder.sphere_query":
        return "audio2sph.grid_queries"
    if key.startswith("decoder.cross_s2."):
        return "audio2sph.spherical_cross_attention.0." + key.removeprefix(
            "decoder.cross_s2."
        )
    if key.startswith("decoder.cross_s3."):
        return "audio2sph.spherical_cross_attention.1." + key.removeprefix(
            "decoder.cross_s3."
        )
    if key.startswith("decoder.cross_s4."):
        return "audio2sph.spherical_cross_attention.2." + key.removeprefix(
            "decoder.cross_s4."
        )
    if key.startswith("decoder.time_layers."):
        return "audio2sph.temporal_context_layers." + key.removeprefix(
            "decoder.time_layers."
        )
    if key.startswith(
        ("decoder.refine0.", "decoder.refine1.", "decoder.refine2.", "decoder.head.")
    ):
        # The paper removes the Panoramic Decoder before complete-SAID training.
        return None
    pixel_decoder = "mask2former_head.pixel_decoder."
    if key.startswith(pixel_decoder):
        remainder = key.removeprefix(pixel_decoder)
        multi_scale_feature_prefixes = (
            ("input_proj.", "input_projection."),
            (
                "transformer.level_embed",
                "multi_scale_encoder.feature_level_embedding",
            ),
            ("transformer.layers.", "multi_scale_encoder.layers."),
            ("mask_features.", "map_features."),
        )
        for submitted_prefix, runtime_prefix in multi_scale_feature_prefixes:
            if remainder.startswith(submitted_prefix):
                remainder = runtime_prefix + remainder.removeprefix(
                    submitted_prefix
                )
                break
        remainder = remainder.replace(
            ".self_attn.", ".deformable_attention."
        )
        return "sph2imaging.multi_scale_feature_path." + remainder
    predictor = "mask2former_head.predictor."
    if key.startswith(predictor):
        remainder = key.removeprefix(predictor)
        mask_decoder_prefixes = (
            (
                "transformer_self_attention_layers.",
                "self_attention_layers.",
            ),
            (
                "transformer_cross_attention_layers.",
                "masked_cross_attention_layers.",
            ),
            (
                "sed_token_cross_attention_layers.",
                "class_feature_cross_attention_layers.",
            ),
            ("transformer_ffn_layers.", "feed_forward_layers."),
            ("decoder_norm.", "output_norm."),
            ("query_feat.", "slot_queries."),
            ("query_embed.", "slot_query_positions."),
            ("level_embed.", "feature_level_embedding."),
            ("class_embed.", "active_head."),
            ("mask_embed.", "map_head."),
        )
        for submitted_prefix, runtime_prefix in mask_decoder_prefixes:
            if remainder.startswith(submitted_prefix):
                remainder = runtime_prefix + remainder.removeprefix(
                    submitted_prefix
                )
                break
        remainder = remainder.replace(".self_attn.", ".attention.")
        remainder = remainder.replace(".multihead_attn.", ".attention.")
        return "sph2imaging.mask_decoder." + remainder
    class_feature_projection = "mask2former_head.sed_token_proj."
    if key.startswith(class_feature_projection):
        return (
            "sph2imaging.class_feature_projection."
            + key.removeprefix(class_feature_projection)
        )
    class_embedding = "mask2former_head.sed_class_embed."
    if key.startswith(class_embedding):
        return (
            "sph2imaging.class_embedding."
            + key.removeprefix(class_embedding)
        )
    refine_block = "mask2former_head.refine360180_head."
    if key.startswith(refine_block):
        remainder = key.removeprefix(refine_block)
        refine_prefixes = (
            ("high_res_refine.", "high_resolution_refinement."),
            ("output_upscaling.", "upsampling."),
            ("highres_mask_embed_mlp.", "map_embedding."),
        )
        for submitted_prefix, runtime_prefix in refine_prefixes:
            if remainder.startswith(submitted_prefix):
                remainder = runtime_prefix + remainder.removeprefix(
                    submitted_prefix
                )
                break
        return "sph2imaging.refine_block." + remainder
    class_head = "mask2former_head.tf_semantic."
    if key.startswith(class_head):
        remainder = key.removeprefix(class_head)
        class_head_prefixes = (
            ("source_proj.s3.", "feature_projection.x2."),
            ("source_proj.s4.", "feature_projection.x3."),
            ("level_embed", "feature_level_embedding"),
            ("layers.", "cross_attention_layers."),
            ("scale", "residual_scale"),
            ("instance_cls_head.net.", "classifier."),
        )
        for submitted_prefix, runtime_prefix in class_head_prefixes:
            if remainder.startswith(submitted_prefix):
                remainder = runtime_prefix + remainder.removeprefix(
                    submitted_prefix
                )
                break
        remainder = remainder.replace(".attn.", ".attention.")
        remainder = remainder.replace(".norm_attn.", ".norm_attention.")
        remainder = remainder.replace(
            ".norm_ffn.", ".norm_feed_forward."
        )
        remainder = remainder.replace(".ffn.", ".feed_forward.")
        return "sph2imaging.class_head." + remainder
    if key.startswith("mask2former_head."):
        return "sph2imaging." + key.removeprefix("mask2former_head.")

    recognition_head = "external_sed_prior.model.head."
    if key.startswith(recognition_head):
        # These classifier tensors belong to checkpoint construction history.
        # SAID reads temporal Class Features before that classifier.
        return None

    passt_backbone = "external_sed_prior.model.backbone."
    if key.startswith(passt_backbone):
        if class_feature_encoder not in {"passt", "test"}:
            raise CheckpointCompatibilityError(
                "checkpoint Class Features do not match the selected encoder"
            )
        return "sph2imaging.class_feature_encoder.backbone." + key.removeprefix(
            passt_backbone
        )
    passt_domain_embedding = "external_sed_prior.model.domain_emb."
    if key.startswith(passt_domain_embedding):
        if class_feature_encoder not in {"passt", "test"}:
            raise CheckpointCompatibilityError(
                "checkpoint Class Features do not match the selected encoder"
            )
        return "sph2imaging.class_feature_encoder.domain_embedding." + key.removeprefix(
            passt_domain_embedding
        )

    audiomae_backbone = "external_sed_prior.model.backend.model."
    if key.startswith(audiomae_backbone):
        if class_feature_encoder not in {"audiomae", "test"}:
            raise CheckpointCompatibilityError(
                "checkpoint Class Features do not match the selected encoder"
            )
        return (
            "sph2imaging.class_feature_encoder.backbone.audio_transformer."
            + key.removeprefix(audiomae_backbone)
        )

    raise CheckpointCompatibilityError(f"unrecognized checkpoint key: {key}")


def normalize_submitted_state_dict(
    state_dict: Mapping[str, torch.Tensor], class_feature_encoder: str
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Map submitted identifiers to the public runtime without rewriting files."""

    runtime_state: dict[str, torch.Tensor] = {}
    compatibility_only: list[str] = []
    for source_key, value in state_dict.items():
        runtime_key = _runtime_key_for_submitted_key(
            source_key, class_feature_encoder
        )
        if runtime_key is None:
            compatibility_only.append(source_key)
            continue
        if runtime_key in runtime_state:
            raise CheckpointCompatibilityError(
                "checkpoint key normalization produced a collision at "
                f"{runtime_key}"
            )
        runtime_state[runtime_key] = value
    return runtime_state, tuple(compatibility_only)


def _panoramic_decoder_key(key: str) -> str | None:
    """Map the serialized pretraining decoder to the paper-aligned module."""

    prefixes = (
        (
            "decoder.refine0.conv.conv.",
            "panoramic_decoder.convolution1.convolution.convolution.",
        ),
        (
            "decoder.refine0.norm.",
            "panoramic_decoder.convolution1.normalization.",
        ),
        (
            "decoder.refine1.conv.conv.",
            "panoramic_decoder.convolution2.convolution.convolution.",
        ),
        (
            "decoder.refine1.norm.",
            "panoramic_decoder.convolution2.normalization.",
        ),
        (
            "decoder.refine2.conv.conv.",
            "panoramic_decoder.convolution3.convolution.convolution.",
        ),
        ("decoder.head.", "panoramic_decoder.output_convolution."),
    )
    for source_prefix, runtime_prefix in prefixes:
        if key.startswith(source_prefix):
            return runtime_prefix + key.removeprefix(source_prefix)
    return None


def normalize_audio2sph_pretraining_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Extract Audio2Sph and Panoramic Decoder tensors from a paper checkpoint."""

    runtime_state: dict[str, torch.Tensor] = {}
    compatibility_only: list[str] = []
    for source_key, value in state_dict.items():
        component = checkpoint_component(source_key)
        if component == "Panoramic Decoder":
            runtime_key = _panoramic_decoder_key(source_key)
        elif component == "Audio2Sph":
            runtime_key = _runtime_key_for_submitted_key(source_key, "test")
        else:
            runtime_key = None
        if runtime_key is None:
            compatibility_only.append(source_key)
            continue
        if runtime_key in runtime_state:
            raise CheckpointCompatibilityError(
                "checkpoint key normalization produced a collision at "
                f"{runtime_key}"
            )
        runtime_state[runtime_key] = value
    return runtime_state, tuple(compatibility_only)


def submitted_class_feature_encoder_state_dict(
    state_dict: Mapping[str, torch.Tensor], class_feature_encoder: str
) -> dict[str, torch.Tensor]:
    """Extract paper-aligned Class Feature Encoder tensors for validation."""

    runtime_state, _ = normalize_submitted_state_dict(
        state_dict, class_feature_encoder
    )
    prefix = "sph2imaging.class_feature_encoder."
    return {
        key.removeprefix(prefix): value
        for key, value in runtime_state.items()
        if key.startswith(prefix)
    }


def _shape(value: Any, key: str) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor):
        raise CheckpointCompatibilityError(f"checkpoint value is not a tensor: {key}")
    return tuple(int(x) for x in value.shape)


def validate_state_dict_compatibility(
    state_dict: Mapping[str, Any], expected_state_dict: Mapping[str, Any]
) -> None:
    """Require an exact key and tensor-shape match with the constructed model."""

    actual_keys = set(state_dict)
    expected_keys = set(expected_state_dict)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise CheckpointCompatibilityError(
            f"state-dict key mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(actual_keys):
        actual_shape = _shape(state_dict[key], key)
        expected_shape = _shape(expected_state_dict[key], key)
        if actual_shape != expected_shape:
            shape_mismatches.append((key, actual_shape, expected_shape))
    if shape_mismatches:
        raise CheckpointCompatibilityError(
            f"state-dict shape mismatch: {shape_mismatches[:5]}"
        )
    dtype_mismatches = [
        (key, str(state_dict[key].dtype), str(expected_state_dict[key].dtype))
        for key in sorted(actual_keys)
        if state_dict[key].dtype != expected_state_dict[key].dtype
    ]
    if dtype_mismatches:
        raise CheckpointCompatibilityError(
            f"state-dict dtype mismatch: {dtype_mismatches[:5]}"
        )


def validate_checkpoint_file(path: str | Path, class_feature_encoder: str) -> None:
    """Verify the byte length and SHA256 of a published checkpoint file."""

    try:
        expected = PAPER_CHECKPOINTS[class_feature_encoder]
    except KeyError as error:
        raise CheckpointCompatibilityError(
            f"unknown class feature encoder: {class_feature_encoder}"
        ) from error
    checkpoint_path = Path(path)
    try:
        if not checkpoint_path.is_file():
            raise CheckpointCompatibilityError(
                f"checkpoint file not found or not a regular file: {checkpoint_path}. "
                "Set the checkpoint field to an existing file path."
            )
        size = checkpoint_path.stat().st_size
    except CheckpointCompatibilityError:
        raise
    except OSError as error:
        raise CheckpointCompatibilityError(
            f"checkpoint file cannot be accessed: {checkpoint_path}: {error}"
        ) from error
    if size != expected.size_bytes:
        raise CheckpointCompatibilityError(
            f"checkpoint size is {size} bytes; expected {expected.size_bytes}"
        )
    digest = sha256()
    try:
        with checkpoint_path.open("rb") as file:
            for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CheckpointCompatibilityError(
            f"checkpoint file cannot be read: {checkpoint_path}: {error}"
        ) from error
    actual = digest.hexdigest()
    if actual != expected.sha256:
        raise CheckpointCompatibilityError(
            f"checkpoint SHA256 is {actual}; expected {expected.sha256}"
        )
