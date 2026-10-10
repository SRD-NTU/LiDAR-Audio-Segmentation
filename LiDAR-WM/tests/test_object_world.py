import torch
from lidarwm.object_world import ObjectQueryWorldModel, geometry_seeds
from lidarwm.object_losses import lovasz_hinge, object_world_losses


def test_geometry_seeds_are_unique_and_bounded():
    xyz=torch.zeros(2,9,3)
    ids=geometry_seeds(xyz,torch.zeros(2,9),32)
    assert ids.shape==(2,9)
    assert all(len(row.unique())==9 for row in ids)


def test_lovasz_ignores_invalid_and_has_finite_gradient():
    logits=torch.tensor([[[[4.,-4.,100.]]]],requires_grad=True)
    target=torch.tensor([[[[1.,0.,0.]]]])
    valid=torch.tensor([[[[1.,1.,0.]]]])
    loss=lovasz_hinge(logits,target,valid)
    assert loss==0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0,0,0,2]==0
    assert lovasz_hinge(logits,target,torch.zeros_like(valid))==0


def test_object_queries_audio_independence_and_joint_backward():
    torch.manual_seed(42)
    model=ObjectQueryWorldModel(base_channels=4,audio_dim=32,lidar_width=32,
        n_fft=64,hop_length=32,max_queries=4,query_layers=1,query_graph=True).eval()
    lidar=torch.randn(2,7,16,32)*.02
    valid=torch.ones(2,1,16,32)
    audio=torch.randn(2,4,512)
    embedding=torch.randn(2,512);text=torch.randn(2,16)
    with torch.backends.mkldnn.flags(enabled=False):
        output=model(lidar,audio,valid,embedding,text)
        with torch.no_grad():
            other=model(lidar,audio.flip(0),valid,embedding.flip(0),text.flip(0))
        # no_grad may select a fused Transformer inference kernel on CPU.
        torch.testing.assert_close(output['object_mask_logits'],other['object_mask_logits'],atol=2e-5,rtol=1e-4)
        torch.testing.assert_close(output['object_rendered_audio'],other['object_rendered_audio'],atol=2e-5,rtol=1e-4)
        assert not torch.allclose(output['mask_logits'],other['mask_logits'])
        instance=torch.zeros(2,16,32,dtype=torch.long);instance[:,4:8,4:8]=1
        instance[:,8:12,20:24]=2
        mask=(instance==1).float()[:,None]
        semantic=instance.clone()
        batch=dict(instance_map=instance,valid=valid,mask=mask,semantic=semantic)
        losses=object_world_losses(output,batch)
        assert all(torch.isfinite(value) for value in losses.values())
        loss=sum(losses.values())+output['mask_logits'].square().mean()
        loss.backward()
    for module in (model.stem,model.object_decoder,model.object_renderer,model.audio_encoder,model.object_graph):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_batch_forward_never_passes_annotations_to_model():
    from lidarwm.engine import forward_batch
    class ObservationOnly:
        def __call__(self,lidar,audio,valid,**kwargs):
            assert set(kwargs)=={'audio_embedding','text_logits'}
            return {'mask_logits':lidar[:,:1]+audio.mean()}
    batch=dict(lidar=torch.randn(1,7,2,4),audio=torch.randn(1,4,32),valid=torch.ones(1,1,2,4),
        audio_embedding=torch.randn(1,512),text_logits=torch.randn(1,16),
        mask=torch.ones(1,1,2,4),semantic=torch.ones(1,2,4),category=torch.tensor([4]),
        instance_map=torch.ones(1,2,4))
    first=forward_batch(ObservationOnly(),batch)
    for key in ('mask','semantic','category','instance_map'):
        batch[key]=batch[key]*99
    other=forward_batch(ObservationOnly(),batch)
    assert torch.equal(first['mask_logits'],other['mask_logits'])
