"""Official DCASE2026 metrics and Class-F1 used by SAID evaluation."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from .dcase2026_metrics_download import (
    METRIC_FIELDS,
    PAPER_AP_FIELDS,
    OFFICIAL_EVALUATOR_REPOSITORY,
    OFFICIAL_EVALUATOR_URL,
    OFFICIAL_MASK_EVALUATOR_URL,
    ensure_official_evaluator,
    ensure_official_mask_evaluator,
    evaluate_dcase_metrics,
    official_evaluator_identity,
    official_evaluator_path,
    official_mask_evaluator_path,
)
from .class_f1 import matched_class_macro_f1

__all__ = [
    "METRIC_FIELDS",
    "PAPER_AP_FIELDS",
    "OFFICIAL_EVALUATOR_REPOSITORY",
    "OFFICIAL_EVALUATOR_URL",
    "OFFICIAL_MASK_EVALUATOR_URL",
    "ensure_official_evaluator",
    "ensure_official_mask_evaluator",
    "evaluate_dcase_metrics",
    "matched_class_macro_f1",
    "official_evaluator_identity",
    "official_evaluator_path",
    "official_mask_evaluator_path",
]
