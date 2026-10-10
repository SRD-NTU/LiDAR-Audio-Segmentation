import torch
from lidarwm.losses import WorldModelLoss


def test_semantic_log_odds_remain_finite_in_bfloat16():
    # Reproduce the failure: BF16 cannot distinguish 1-1e-4 from 1.
    p=torch.tensor([1.0],dtype=torch.bfloat16)
    assert not torch.isfinite(torch.logit(p.clamp(1e-4,1-1e-4))).all()
    logit=torch.logit(p.float().clamp(1e-4,1-1e-4))
    assert torch.isfinite(logit).all()


def test_balanced_semantic_option_is_explicit():
    assert not WorldModelLoss().balanced_semantic
    assert WorldModelLoss(balanced_semantic=True).balanced_semantic


def test_revised_model_saturated_semantics_under_autocast():
    from lidarwm.revised import RevisedWorldModel
    torch.manual_seed(42)
    model=RevisedWorldModel(base_channels=4,audio_dim=32,lidar_width=32,n_fft=32,hop_length=16).eval()
    with torch.no_grad():
        model.semantic_head.weight.zero_()
        model.semantic_head.bias.fill_(-1000)
        model.semantic_head.bias[1]=1000
    lidar=torch.randn(1,7,16,32)*.1
    text=torch.full((1,16),-1000.);text[:,0]=1000
    # Avoid machine-specific oneDNN BF16 convolution kernels in this CPU test;
    # production GPU AMP is additionally exercised by the profile job.
    with torch.backends.mkldnn.flags(enabled=False):
        with torch.autocast('cpu',dtype=torch.bfloat16):
            output=model(lidar,torch.randn(1,4,512),audio_embedding=torch.randn(1,512),text_logits=text)
            loss=output['mask_logits'].square().mean()
        assert torch.isfinite(loss)
        loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
