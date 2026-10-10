import torch
from lidarwm.inference import rotate_observations,predict_mask_logits


def test_square_array_rotation_round_trip():
    x=torch.randn(2,7,4,16);a=torch.randn(2,4,32);v=torch.rand(2,1,4,16)
    rotated=rotate_observations(x,a,v,3)
    recovered=rotate_observations(*rotated,1)
    for original,returned in zip((x,a,v),recovered):torch.testing.assert_close(original,returned)
    assert torch.equal(x[:,3].roll(4,-1),rotate_observations(x,a,v,1)[0][:,3])


def test_ensemble_inverse_rotation_preserves_equivariant_mask():
    class Dummy(torch.nn.Module):
        def forward(self,lidar,audio,valid):return {'mask_logits':lidar[:,3:4]}
    x=torch.randn(1,7,4,16);a=torch.randn(1,4,32);v=torch.ones(1,1,4,16)
    logits=predict_mask_logits(Dummy(),x,a,v,tta=True)
    torch.testing.assert_close(logits,x[:,3:4],atol=1e-6,rtol=1e-6)
