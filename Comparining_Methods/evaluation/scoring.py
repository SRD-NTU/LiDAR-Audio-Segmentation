"""Per-example foreground IoU and scene-cluster uncertainty."""
from __future__ import annotations

import numpy as np


def foreground_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError("Prediction and target must be aligned one-dimensional masks")
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / union) if union else 1.0


def selected_candidate_accuracy(candidates: np.ndarray, selected: int, target: np.ndarray) -> float:
    """1 if the selected proposal is tied for best target IoU in its saved bank."""
    candidates = np.asarray(candidates, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if candidates.ndim != 2 or candidates.shape[1:] != target.shape or not 0 <= selected < len(candidates):
        raise ValueError("Candidate bank, selected index, and target are inconsistent")
    scores = np.array([foreground_iou(mask, target) for mask in candidates])
    return float(scores[selected] >= scores.max() - 1e-12)


def aggregate(rows: list[dict], seed: int = 20261027, draws: int = 2000) -> dict:
    if not rows:
        raise ValueError("No scored examples")
    scenes = sorted({row["scene"] for row in rows})
    totals = np.array([sum(float(row["foreground_iou"]) for row in rows if row["scene"] == scene) for scene in scenes])
    counts = np.array([sum(row["scene"] == scene for row in rows) for scene in scenes])
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(scenes), (draws, len(scenes)))
    bootstrap = totals[selected].sum(axis=1) / counts[selected].sum(axis=1)
    result = {"mean_iou": float(totals.sum() / counts.sum()),
              "scene_macro_mean_iou": float(np.mean(totals / counts)),
              "mean_iou_ci95_scene_bootstrap": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
              "samples": len(rows), "scenes": len(scenes), "seed": seed, "bootstrap_draws": draws,
              "per_scene": {scene: float(total / count) for scene, total, count in zip(scenes, totals, counts)}}
    selection = [float(row["candidate_selection_accuracy"]) for row in rows
                 if row.get("candidate_selection_accuracy") is not None]
    result["candidate_selection_accuracy"] = float(np.mean(selection)) if len(selection) == len(rows) else None
    return result
