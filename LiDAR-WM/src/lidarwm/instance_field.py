"""An object coordinate field for learning coherent instance geometry."""
import torch
from torch import nn
from .revised import RevisedWorldModel

class InstanceFieldWorldModel(RevisedWorldModel):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.center_head=nn.Sequential(nn.Conv2d(kwargs.get('base_channels',24),32,1),nn.GELU(),nn.Conv2d(32,3,1))
        self._geometry=None
        self.field_strength=nn.Parameter(torch.tensor(-3.0))
        self.semantic_head.register_forward_pre_hook(self._capture_geometry)

    def _capture_geometry(self,module,args):
        self._geometry=args[0]

    def forward(self,lidar,audio,valid=None,**kwargs):
        output=super().forward(lidar,audio,valid,**kwargs)
        output['instance_center']=lidar[:,:3]+self.center_head(self._geometry).tanh()*.2
        validity=torch.ones_like(lidar[:,:1]) if valid is None else valid
        # Infer a source-object center from the audio-conditioned mask posterior,
        # then spread support through the learned object coordinate field.
        weights=(output['mask_logits'].float()*2).masked_fill(validity<.5,-1e4).flatten(2).softmax(-1)
        anchor=(output['instance_center'].float().flatten(2)*weights).sum(-1)
        distance=(output['instance_center'].float()-anchor[:,:,None,None]).square().sum(1,keepdim=True).add(1e-8).sqrt()
        output['instance_anchor']=anchor
        output['mask_logits']=output['mask_logits']-torch.nn.functional.softplus(self.field_strength)*distance/.025
        self._geometry=None
        return output
