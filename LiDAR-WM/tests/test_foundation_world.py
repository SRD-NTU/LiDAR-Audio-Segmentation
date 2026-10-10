import torch
from torch import nn
from lidarwm.engine import forward_batch
from lidarwm.losses import WorldModelLoss


def test_foundation_masks_remain_geometry_only_and_loss_backward(monkeypatch):
    import lidarwm.foundation_world as module
    class GeometryStub(nn.Module):
        def __init__(self,*args,**kwargs):
            super().__init__();self.frozen=True
        def forward(self,lidar,valid):
            return lidar[:,:1].expand(-1,256,-1,-1)*valid
    monkeypatch.setattr(module,'SonataGeometry',GeometryStub)
    torch.manual_seed(3)
    model=module.FoundationWorldModel(base_channels=8,audio_dim=32,lidar_width=32,n_fft=64,
        hop_length=32,max_queries=4,query_layers=1,query_graph=False,
        anchor_mode='observed',source_acoustics='center',local_geometry=True).eval()
    lidar=torch.randn(2,7,16,32)*.02
    batch=dict(lidar=lidar,valid=torch.ones(2,1,16,32),audio=torch.randn(2,4,512),
        audio_embedding=torch.randn(2,512),text_logits=torch.randn(2,16))
    instance=torch.zeros(2,16,32,dtype=torch.long);instance[:,4:8,8:12]=1
    batch.update(instance_map=instance,semantic=instance,mask=(instance>0).float()[:,None],category=torch.tensor([0,0]))
    with torch.backends.mkldnn.flags(enabled=False):
        output=forward_batch(model,batch)
        with torch.no_grad():
            swapped=forward_batch(model,{**batch,'audio':batch['audio'].flip(0),
                'audio_embedding':batch['audio_embedding'].flip(0),'text_logits':batch['text_logits'].flip(0)})
        torch.testing.assert_close(output['object_mask_logits'],swapped['object_mask_logits'],atol=2e-5,rtol=1e-4)
        assert not torch.allclose(output['mask_logits'],swapped['mask_logits'])
        criterion=WorldModelLoss(object_instances=1,object_source=1,geometry_foreground=.5,
            boundary=.1,hard_source=.2)
        losses=criterion(output,batch)
        assert all(torch.isfinite(loss) for loss in losses.values())
        losses['total'].backward()
    for head in (model.geometry_adapter,model.geometry_context,model.geometry_fine,model.phase_renderer):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters())
