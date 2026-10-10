import torch
from lidarwm.object_world import ObjectQueryWorldModel


def test_observed_anchors_raw_input_contract():
    model=ObjectQueryWorldModel(base_channels=4,audio_dim=32,n_fft=64,hop_length=32,
        max_queries=4,query_layers=1,anchor_mode='observed',source_acoustics='center',
        query_stride=4,local_geometry=True,query_graph=False).eval()
    lidar=torch.randn(1,7,16,33)*.01
    with torch.backends.mkldnn.flags(enabled=False),torch.autocast('cpu',dtype=torch.bfloat16):
        result=model(lidar,torch.randn(1,4,512),audio_embedding=torch.randn(1,512),text_logits=torch.randn(1,16))
        loss=result['mask_logits'].square().mean()+.01*result['object_scores'].square().mean()
        loss.backward()
    assert result['mask_logits'].shape==(1,1,16,33)
    assert torch.isfinite(result['mask_logits']).all()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.object_local.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
