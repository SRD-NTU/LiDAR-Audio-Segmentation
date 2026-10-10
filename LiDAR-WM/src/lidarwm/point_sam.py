"""Geometry-only Point-SAM adapter with portable PyTorch sampling primitives.

Upstream code: https://github.com/zyc00/Point-SAM (MIT).
RGB features are filled with a fixed neutral value, never loaded from data.
"""
import sys
import types
from pathlib import Path
import torch

def batch_index_select(x,index,dim=1):
    if dim!=1: raise NotImplementedError('Point-SAM adapter supports dim=1')
    batch=torch.arange(x.shape[0],device=x.device).reshape(x.shape[0],*((1,)*(index.ndim-1)))
    return x[batch,index]

@torch.no_grad()
def sample_farthest_points(points,count):
    b,n,_=points.shape; result=torch.empty(b,count,dtype=torch.long,device=points.device)
    distances=torch.full((b,n),float('inf'),device=points.device)
    farthest=points.square().sum(-1).argmax(-1); batch=torch.arange(b,device=points.device)
    for i in range(count):
        result[:,i]=farthest
        center=points[batch,farthest][:,None]
        distances=torch.minimum(distances,(points-center).square().sum(-1))
        farthest=distances.argmax(-1)
    return result

def _register_primitives():
    modules={name:types.ModuleType(name) for name in ('torkit3d','torkit3d.nn','torkit3d.nn.functional',
      'torkit3d.ops','torkit3d.ops.sample_farthest_points','torkit3d.ops.chamfer_distance')}
    modules['torkit3d.nn.functional'].batch_index_select=batch_index_select
    modules['torkit3d.ops.sample_farthest_points'].sample_farthest_points=sample_farthest_points
    def chamfer_distance(x,y):
        distance=torch.cdist(x,y).square()
        d1,i1=distance.min(-1);d2,i2=distance.min(-2)
        return d1,d2,i1,i2
    modules['torkit3d.ops.chamfer_distance'].chamfer_distance=chamfer_distance
    sys.modules.update(modules)

def build_point_sam(device='cuda',groups=512,group_size=64):
    _register_primitives()
    sys.path.insert(0,str(Path('third_party/Point-SAM').resolve()))
    import timm
    from safetensors.torch import load_model
    from huggingface_hub import hf_hub_download
    from pc_sam.model.pc_encoder import PointCloudEncoder,PatchEmbed
    from pc_sam.model.prompt_encoder import MaskEncoder
    from pc_sam.model.mask_decoder import MaskDecoder
    from pc_sam.model.transformer import TwoWayTransformer
    from pc_sam.model.pc_sam import PointCloudSAM
    backbone=timm.create_model('eva02_large_patch14_448',pretrained=False)
    encoder=PointCloudEncoder(PatchEmbed(6,512,groups,group_size),backbone,256)
    decoder=MaskDecoder(256,TwoWayTransformer(depth=2,embedding_dim=256,num_heads=8,mlp_dim=2048))
    model=PointCloudSAM(encoder,MaskEncoder(256),decoder,prompt_iters=1)
    path=hf_hub_download('yuchen0187/Point-SAM','model.safetensors',cache_dir='pretrained/hub')
    load_model(model,path)
    return model.to(device).eval()
