"""Auditable XYZ-only adapter for the official pretrained Sonata encoder.

No observed RGB is read. The upstream nine-feature interface receives centered
XYZ, CONSTANT zero color, and normals computed from XYZ. Voxel selection and
unpooling use coordinates only. Fixed feature compression is differentiable;
encoder parameters can be fine-tuned end to end after frozen feature probing.
Weights: CC-BY-NC-4.0. Upstream HM3D pretraining overlap is NOT ruled out.
"""
import hashlib
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def geometry_normals(xyz,valid):
    """Central differences, rejecting neighbors across metric depth jumps."""
    dx=xyz.roll(-1,-1)-xyz.roll(1,-1)
    dy=torch.cat((xyz[:,:,1:],xyz[:,:,-1:]),-2)-torch.cat((xyz[:,:,:1],xyz[:,:,:-1]),-2)
    normal=F.normalize(torch.cross(dx,dy,dim=1),dim=1,eps=1e-6)
    reliable=valid*valid.roll(1,-1)*valid.roll(-1,-1)
    reliable=reliable*(dx.norm(dim=1,keepdim=True)<.5)*(dy.norm(dim=1,keepdim=True)<.5)
    # Orient normals towards the sensor; no hidden scene camera information.
    normal=normal*torch.where((normal*xyz).sum(1,keepdim=True)>0,-1.,1.)
    return normal*reliable


class SonataGeometry(nn.Module):
    def __init__(self,checkpoint='pretrained/sonata/sonata.pth',grid_size=.05,
                 feature_dim=256,patch_size=128,frozen=True):
        super().__init__()
        root=Path(__file__).resolve().parents[2]
        sys.path.insert(0,str(root/'third_party/sonata'))
        from sonata.model import load
        self.encoder=load(str(checkpoint),custom_config=dict(enable_flash=False,
            enc_patch_size=[patch_size]*5,shuffle_orders=False,drop_path=0.,
            upcast_attention=True,upcast_softmax=True))
        # Upstream GridPooling has its OWN shuffle_orders=True default, not
        # propagated from the top-level encoder option. Disable it everywhere
        # for deterministic caches and raw deployment equivalence.
        for module in self.encoder.modules():
            if hasattr(module,'shuffle_orders'):module.shuffle_orders=False
        self.grid_size=float(grid_size)
        self.feature_dim=int(feature_dim)
        self.frozen=bool(frozen)
        channels=1232
        generator=torch.Generator().manual_seed(443)
        # Preserve the finest 48 channels and compress the four contextual levels.
        projection=torch.randn(channels-48,feature_dim-48,generator=generator)/(feature_dim-48)**.5
        self.register_buffer('projection',projection)
        self.set_frozen(frozen)
        self.checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()

    def set_frozen(self,frozen):
        self.frozen=bool(frozen)
        self.encoder.requires_grad_(not frozen)

    def train(self,mode=True):
        super().train(mode)
        if self.frozen:self.encoder.eval()
        return self

    def _encode(self,lidar,valid,sparse=False):
        with torch.autocast(device_type=lidar.device.type,enabled=False):
            xyz=lidar[:,:3].float()*20
            normals=geometry_normals(xyz,valid.float())
            coordinates=[];features=[];grids=[];batches=[];inverses=[];pixels=[]
            for b in range(len(lidar)):
                indices=(valid[b,0].flatten()>0).nonzero().flatten()
                points=xyz[b].flatten(1).T[indices]
                if not len(points):raise ValueError('Sonata requires at least one valid LiDAR point')
                # CenterShift(apply_z=True), matching the pretrained interface.
                lower=points.amin(0);upper=points.amax(0)
                shift=torch.stack(((lower[0]+upper[0])/2,(lower[1]+upper[1])/2,lower[2]))
                centered=points-shift
                cell=torch.floor((centered-centered.amin(0))/self.grid_size).long()
                _,inverse=torch.unique(cell,dim=0,return_inverse=True)
                count=int(inverse.max())+1
                first=torch.full((count,),len(points),device=points.device,dtype=torch.long)
                first.scatter_reduce_(0,inverse,torch.arange(len(points),device=points.device),reduce='amin')
                position=centered[first]
                normal=normals[b].flatten(1).T[indices[first]]
                coordinates.append(position);grids.append(cell[first].int())
                features.append(torch.cat((position,torch.zeros_like(position),normal),-1))
                batches.append(torch.full((count,),b,device=points.device,dtype=torch.long))
                inverses.append(inverse);pixels.append(indices)
            point=self.encoder(dict(coord=torch.cat(coordinates),grid_coord=torch.cat(grids),
                feat=torch.cat(features),batch=torch.cat(batches)))
            # Unpool all five scales to the first voxel grid. No averaging across
            # range-image discontinuities or silent discarding of tiny objects.
            while 'pooling_parent' in point:
                parent=point.pop('pooling_parent');inverse=point.pop('pooling_inverse')
                parent.feat=torch.cat((parent.feat,point.feat[inverse]),-1)
                point=parent
            compressed=torch.cat((point.feat[:,:48],point.feat[:,48:]@self.projection),-1)
            dense=[];start=0
            for b,inverse in enumerate(inverses):
                count=len(coordinates[b]);local=compressed[start:start+count];start+=count
                if sparse:
                    dense.append(dict(features=local,inverse=inverse,pixels=pixels[b]))
                    continue
                grid=local.new_zeros(lidar.shape[-2]*lidar.shape[-1],self.feature_dim)
                grid=grid.index_copy(0,pixels[b],local[inverse])
                dense.append(grid.T.reshape(self.feature_dim,*lidar.shape[-2:]))
            return dense if sparse else torch.stack(dense)

    def forward(self,lidar,valid):
        # Upstream non-Flash attention chooses patch size from the smallest
        # scene in a batch. Per-scene encoding prevents batch-composition changes
        # to predictions and matches independently computed cached features.
        if len(lidar)>1:
            return torch.cat([self.forward(lidar[b:b+1],valid[b:b+1]) for b in range(len(lidar))])
        if self.frozen:
            with torch.no_grad():return self._encode(lidar,valid)
        return self._encode(lidar,valid)
