import torch

from lidarwm.losses import WorldModelLoss
from lidarwm.model import AcousticWorldModel


def test_model_forward_and_backward():
    torch.manual_seed(0)
    model = AcousticWorldModel(
        base_channels=8, audio_dim=40, lidar_width=64, audio_sample_rate=8000,
        n_fft=64, hop_length=32, attention_heads=5,
    )
    batch = 2
    lidar = torch.randn(batch, 7, 16, 64)
    valid = torch.ones(batch, 1, 16, 64)
    audio = torch.randn(batch, 4, 1024)
    target = torch.zeros(batch, 1, 16, 64)
    target[:, :, 5:9, 20:27] = 1.0
    output = model(lidar, audio, valid)
    assert output["mask_logits"].shape == target.shape
    assert output["render_logits"].shape == (batch, 2, 8)
    losses = WorldModelLoss()(output, {
        "mask": target, "valid": valid, "lidar": lidar,
        "category": torch.tensor([0, 1]),
        "semantic": torch.zeros(batch, 16, 64, dtype=torch.long),
    })
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.mask_head[-1].weight.grad is not None
