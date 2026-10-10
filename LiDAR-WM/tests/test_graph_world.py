import torch
from lidarwm.graph_world import GraphLatentWorldModel


def test_directional_graph_world_backward():
    torch.manual_seed(42)
    model=GraphLatentWorldModel(base_channels=4,audio_dim=32,lidar_width=32,n_fft=64,hop_length=32)
    lidar=torch.randn(2,7,16,32)*.01
    with torch.backends.mkldnn.flags(enabled=False):
        output=model(lidar,torch.randn(2,4,512),audio_embedding=torch.randn(2,512),text_logits=torch.randn(2,16))
        loss=output['mask_logits'].square().mean()
        assert torch.isfinite(loss);loss.backward()
    assert model.geometry_graph.alpha.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model._graph_lidar is None and model._graph_valid is None
