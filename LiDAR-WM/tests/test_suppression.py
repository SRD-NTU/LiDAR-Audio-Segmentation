import numpy as np
import torch
from lidarwm.suppression import MaskSuppression,make_proposals,overlap_targets,choose_candidate,SuppressedWorldModel


def test_graph_suppression_forward_backward_and_padding():
    model=MaskSuppression(12,hidden=16)
    features=torch.randn(2,5,12);positions=torch.randn(2,5,3)
    valid=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
    logits,offset=model(features,positions,valid)
    loss=logits[valid].square().mean()+offset[valid].square().mean()
    assert torch.isfinite(loss);loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    alone=model(features[0,:3],positions[0,:3])[0]
    padded=model(features[:1],positions[:1],valid[:1])[0][0,:3]
    torch.testing.assert_close(alone,padded)


def test_candidate_zero_exactly_preserves_baseline_and_validity():
    lidar=np.zeros((7,4,16),dtype='float32');lidar[0]=.1
    valid=np.ones((4,16),dtype=bool);valid[0,0]=False
    probability=np.zeros((4,16),dtype='float32');probability[1:3,3:6]=.8;probability[0,0]=1
    bank=make_proposals(lidar,valid,probability,.5)
    masks=np.unpackbits(bank,axis=1,count=64,bitorder='little').reshape(-1,4,16)
    np.testing.assert_array_equal(masks[0],(probability>=.5)&valid)
    iou,intersection,target=overlap_targets(bank,((probability>=.5)&valid))
    assert iou[0]==1 and intersection[0]==6 and target==6


def test_empty_world_mask_produces_safe_fallback():
    bank=make_proposals(np.zeros((7,4,16)),np.ones((4,16),bool),np.zeros((4,16)),.5)
    assert bank.shape==(1,8)
    assert choose_candidate([.1,.11],margin=.02)==0
    assert choose_candidate([.1,.3],margin=.02)==1


def test_surface_components_wrap_azimuth_but_not_vertical_beams():
    lidar=np.zeros((7,4,16),dtype='float32');lidar[0]=.1
    valid=np.ones((4,16),bool);p=np.zeros((4,16),dtype='float32')
    p[1:3,[0,15]]=.8;p[1:3,7:9]=.7
    bank=make_proposals(lidar,valid,p,.5,thresholds=(.5,))
    masks=np.unpackbits(bank,axis=1,count=64,bitorder='little').reshape(-1,4,16).astype(bool)
    seam=np.zeros((4,16),bool);seam[1:3,[0,15]]=True
    assert any(np.array_equal(mask,seam) for mask in masks)


def test_compact_and_physics_graph_modes():
    for mode in ('compact','physics'):
        model=MaskSuppression(303,hidden=16,feature_mode=mode,residual_quality=True)
        logits,offset=model(torch.randn(2,4,303),torch.randn(2,4,3)*.05)
        loss=logits.square().mean()+offset.square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_integrated_model_preserves_fallback_and_serialized_predictions():
    torch.manual_seed(42)
    kwargs=dict(base_channels=4,audio_dim=128,lidar_width=32,n_fft=64,hop_length=32,
      suppression_margin=1.,baseline_threshold=.2,suppression_hidden=16)
    model=SuppressedWorldModel(**kwargs).eval()
    x=torch.randn(1,7,16,32)*.01;a=torch.randn(1,4,512);v=torch.ones(1,1,16,32)
    extra=dict(audio_embedding=torch.randn(1,512),text_logits=torch.randn(1,16))
    with torch.backends.mkldnn.flags(enabled=False),torch.inference_mode():
        base=model.base(x,a,v,**extra)['mask_logits'].sigmoid()>=.2
        result=model(x,a,v,**extra)['mask_logits']>0
        torch.testing.assert_close(base,result)
        restored=SuppressedWorldModel(**kwargs).eval();restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(x,a,v,**extra)['mask_logits']>0,result)
