import torch
from lidarwm.reliable_acoustics import ReliablePhaseRenderer
from lidarwm.audio import MIC_POSITIONS
from lidarwm.sonata_geometry import geometry_normals


def test_phase_renderer_finite_single_query_and_gradients():
    renderer=ReliablePhaseRenderer(64,14700)
    xyz=torch.tensor([[[[2.]],[[1.]],[[.2]]]],requires_grad=True)
    spec=torch.randn(1,4,33,8,dtype=torch.complex64)
    # A single candidate should be neutral, not NaN from unbiased sample SD.
    assert torch.isfinite(renderer(xyz,spec,MIC_POSITIONS)).all()
    points=torch.randn(2,3,1,5,requires_grad=True)
    spec=torch.randn(2,4,33,8,dtype=torch.complex64)
    score=renderer(points,spec,MIC_POSITIONS)
    score.square().mean().backward()
    assert torch.isfinite(points.grad).all()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in renderer.parameters())


def test_normals_use_only_geometry_and_invalid_points_are_zero():
    xyz=torch.randn(2,3,8,16)*.03
    valid=torch.ones(2,1,8,16);valid[:,:,2:4,4:6]=0
    normal=geometry_normals(xyz,valid)
    assert torch.isfinite(normal).all()
    assert torch.equal(normal*(1-valid),torch.zeros_like(normal))
