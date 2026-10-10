import torch

from lidarwm.metrics import BinarySegmentationMetrics


def test_perfect_metrics():
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    logits = torch.where(target > 0, torch.tensor(20.0), torch.tensor(-20.0))
    metrics = BinarySegmentationMetrics((0.5,))
    metrics.update(logits, target, torch.ones_like(target))
    result = metrics.compute()
    assert result["mean_iou"] == 1.0
    assert result["micro_iou"] == 1.0
    assert result["f1"] == 1.0

