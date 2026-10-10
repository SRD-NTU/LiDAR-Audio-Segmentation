"""Low-capacity, physics-constrained band reliability from observed coherence."""
import numpy as np
import torch
from torch import nn

from .features import MICROPHONES, PAIRS


def direction_grid():
    # A coplanar array cannot determine the sign of elevation from phase alone.
    # Keep that ambiguity explicit rather than train an unidentifiable sign.
    azimuth = np.deg2rad(np.arange(-180, 180, 5))
    elevation = np.deg2rad(np.arange(0, 91, 5))
    azimuth, elevation = np.meshgrid(azimuth, elevation, indexing='ij')
    return np.stack((np.cos(elevation)*np.cos(azimuth),
                     np.cos(elevation)*np.sin(azimuth), np.sin(elevation)), -1).reshape(-1,3).astype(np.float32)


def grid_field(tables, radius=2.):
    centers = direction_grid()*radius
    distance = np.linalg.norm(centers[:,None]-MICROPHONES[None],axis=-1)
    delays = np.stack([(distance[:,r]-distance[:,l])/343 for l,r in PAIRS],-1)
    position = np.clip((delays+.00036)/.00072*160,0,159.999)
    lower = position.astype(np.int64)
    pair = np.arange(6)[None,:,None]
    band = np.arange(9)[None,None,:]
    left, right = tables[pair,band,lower[:,:,None]],tables[pair,band,lower[:,:,None]+1]
    return (left+(right-left)*(position-lower)[:,:,None]).reshape(len(centers),54).astype(np.float32)


class SpatialReliability(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(54,64),nn.GELU(),nn.Linear(64,55))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[-1] = -1.6

    def forward(self, coherence):
        values = self.head(coherence)
        return values[...,:54].softmax(-1),2+48*values[...,-1:].sigmoid()

    def logits(self, field, coherence):
        weights, temperature = self(coherence)
        return (field*weights[:,None]).sum(-1)*temperature


def scene_field(acoustic, tables, model):
    """Observed per-node phase score, with no geometry/source annotations."""
    coherence=np.sqrt((np.asarray(tables,dtype=np.float32)**2).mean(-1)).ravel()
    with torch.inference_mode():
        weight,_=model(torch.from_numpy(coherence))
    return np.asarray(acoustic,dtype=np.float32) @ weight.numpy()


def initialize_world_reliability(world, spatial):
    """Embed the trained audio-only weights into the world's 75-input gate."""
    with torch.no_grad():
        world.reliability[0].weight.zero_()
        world.reliability[0].weight[:,21:].copy_(spatial.head[0].weight)
        world.reliability[0].bias.copy_(spatial.head[0].bias)
        world.reliability[-1].weight.copy_(spatial.head[-1].weight[:54])
        world.reliability[-1].bias.copy_(spatial.head[-1].bias[:54])


def selected_scene_field(observation, selection, root):
    """Shared calibrated-world phase path for raw and cached deployment."""
    if not selection.get('spatial_head_checkpoint'):
        return np.asarray(observation['acoustic'],dtype=np.float32).mean(1)
    import hashlib
    path=root/selection['spatial_head_checkpoint']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=selection['spatial_head_sha256']:
        raise ValueError('Spatial audio head changed after selection')
    model=SpatialReliability().eval()
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    return scene_field(observation['acoustic'],observation['delay_tables'],model)
