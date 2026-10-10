from __future__ import annotations

import torch


class BinarySegmentationMetrics:
    def __init__(self, thresholds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)) -> None:
        self.thresholds = thresholds
        self.reset()

    def reset(self) -> None:
        count = len(self.thresholds)
        self.intersection = torch.zeros(count, dtype=torch.float64)
        self.union = torch.zeros(count, dtype=torch.float64)
        self.predicted = torch.zeros(count, dtype=torch.float64)
        self.target = torch.zeros(count, dtype=torch.float64)
        self.sample_iou_sum = torch.zeros(count, dtype=torch.float64)
        self.samples = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        probability = logits.sigmoid()
        truth = (target > 0.5) & (valid > 0.5)
        for index, threshold in enumerate(self.thresholds):
            prediction = (probability >= threshold) & (valid > 0.5)
            intersection = (prediction & truth).sum(dim=(-3, -2, -1)).double().cpu()
            union = (prediction | truth).sum(dim=(-3, -2, -1)).double().cpu()
            self.intersection[index] += intersection.sum()
            self.union[index] += union.sum()
            self.predicted[index] += prediction.sum().double().cpu()
            self.target[index] += truth.sum().double().cpu()
            self.sample_iou_sum[index] += ((intersection + 1e-9) / (union + 1e-9)).sum()
        self.samples += logits.shape[0]

    def compute(self) -> dict[str, float]:
        micro_iou = self.intersection / self.union.clamp_min(1.0)
        sample_iou = self.sample_iou_sum / max(self.samples, 1)
        f1 = 2.0 * self.intersection / (self.predicted + self.target).clamp_min(1.0)
        best = int(sample_iou.argmax())
        return {
            "threshold": self.thresholds[best],
            "mean_iou": float(sample_iou[best]),
            "micro_iou": float(micro_iou[best]),
            "f1": float(f1[best]),
            "precision": float(self.intersection[best] / self.predicted[best].clamp_min(1.0)),
            "recall": float(self.intersection[best] / self.target[best].clamp_min(1.0)),
        }

