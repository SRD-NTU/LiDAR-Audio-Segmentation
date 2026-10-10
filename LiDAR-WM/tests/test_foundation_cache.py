import torch
from lidarwm.foundation_world import FoundationWorldModel
from lidarwm.engine import to_device,forward_batch


def test_sparse_geometry_cache_decoding_and_nested_device_transfer():
    entries=[dict(features=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float16),
        inverse=torch.tensor([0,1,0],dtype=torch.int32),pixels=torch.tensor([0,3,5],dtype=torch.int32))]
    moved=to_device(dict(geometry_features=entries,path=['sample']),torch.device('cpu'))
    grid=FoundationWorldModel.decode_cache(moved['geometry_features'],(2,3))
    assert grid.shape==(1,2,2,3)
    torch.testing.assert_close(grid[0,:,0,0],torch.tensor([1.,2.]))
    torch.testing.assert_close(grid[0,:,1,0],torch.tensor([3.,4.]))
    assert grid[0,:,0,1].sum()==0
    assert moved['path']==['sample']


def test_observation_cache_is_allowed_but_labels_are_not_forwarded():
    cache=[dict(features=torch.ones(1,256),inverse=torch.tensor([0]),pixels=torch.tensor([0]))]
    class ObservationOnly:
        def __call__(self,lidar,audio,valid,**kwargs):
            assert set(kwargs)=={'geometry_features','audio_embedding','text_logits'}
            assert kwargs['geometry_features'] is cache
            return lidar.sum()
    batch=dict(lidar=torch.ones(1,7,2,4),audio=torch.ones(1,4,32),valid=torch.ones(1,1,2,4),
        geometry_features=cache,audio_embedding=torch.ones(1,512),text_logits=torch.ones(1,16),
        semantic=torch.ones(1,2,4),instance_map=torch.ones(1,2,4),mask=torch.ones(1,1,2,4))
    forward_batch(ObservationOnly(),batch)
