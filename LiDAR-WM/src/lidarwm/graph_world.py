"""LS3D's actual directional GNN inside the geometry-only latent world encoder."""
import torch
from torch.nn import functional as F
from .revised import RevisedWorldModel
from .suppression import upstream_module


class GraphLatentWorldModel(RevisedWorldModel):
    def __init__(self,graph_neighbors=16,**kwargs):
        super().__init__(**kwargs)
        dim=kwargs.get('audio_dim',128)
        self.geometry_graph=upstream_module('directional_gnn').DirectionalGNN(
          dim,hidden_channels=dim,k_neighbors=graph_neighbors,dropout=.2,
          transform_channels=dim,gate_init=-5.,use_dedup=False)
        self.graph_neighbors=graph_neighbors;self._graph_lidar=None;self._graph_valid=None
        self.scene_context.register_forward_hook(self._refine_geometry)

    def _refine_geometry(self,module,args,tokens):
        height=(self._graph_lidar.shape[-2]+15)//16;width=(self._graph_lidar.shape[-1]+15)//16
        with torch.autocast(device_type=tokens.device.type,enabled=False):
            valid=self._graph_valid.float()
            xyz=F.adaptive_avg_pool2d(self._graph_lidar[:,:3].float()*valid,(height,width))
            mass=F.adaptive_avg_pool2d(valid,(height,width))
            xyz=(xyz/mass.clamp_min(1e-3)*20).flatten(2)
            points=xyz.transpose(1,2)
            distance=torch.cdist(points,points)
            eye=torch.eye(points.shape[1],device=points.device,dtype=torch.bool)[None]
            distance=distance.masked_fill(eye,float('inf'))
            knn=distance.topk(min(self.graph_neighbors,points.shape[1]-1),largest=False).indices
            refined=self.geometry_graph(xyz,tokens.float().transpose(1,2),knn_idx=knn)
            return refined.transpose(1,2)

    def forward(self,lidar,audio,valid=None,**kwargs):
        self._graph_lidar=lidar
        self._graph_valid=torch.ones_like(lidar[:,:1]) if valid is None else valid
        try:return super().forward(lidar,audio,valid,**kwargs)
        finally:self._graph_lidar=None;self._graph_valid=None
