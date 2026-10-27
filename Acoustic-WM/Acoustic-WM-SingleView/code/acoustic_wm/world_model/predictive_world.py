"""Acoustic world state, observation assimilation, and learned pose transitions.

The initial acoustic prediction is made before seeing audio. Imagined receiver
motions consume actions and latent memory, never future observations/geometry.
The system models a static acoustic environment under receiver motion, not
physical motion of objects or sound-wave time stepping.
"""
import math
import torch
from torch import nn
from ..audio.predictive_features import POOL48
from .acoustic_transfer import DirectTransfer


def position_code(source):
    phase=source[...,None]*torch.tensor([1.,2.,4.,8.],device=source.device)/5
    return torch.cat([source/10,source.norm(dim=-1,keepdim=True)/10,phase.sin().flatten(-2),phase.cos().flatten(-2)],-1)


def move_source(source,action):
    """Row vectors: new source = (old source - local receiver delta) R_old.T R_new."""
    rotation=action[...,3:].reshape(*action.shape[:-1],3,3)
    return torch.einsum('...i,...ij->...j',source-action[...,:3],rotation)


class PredictiveAcousticWorld(nn.Module):
    def __init__(self,microphones,mode='world',hidden=128):
        super().__init__();self.mode=mode;self.hidden=hidden
        self.direct=DirectTransfer(microphones);self.direct.pool=torch.tensor(POOL48,dtype=torch.float32)
        self.point=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,64),nn.GELU())
        self.point_position=nn.Sequential(nn.Linear(4,64),nn.GELU(),nn.Linear(64,64))
        self.query=nn.Linear(28,64)
        self.prior=nn.Sequential(nn.Linear(284,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.Tanh())
        self.decoder=nn.Sequential(nn.Linear(hidden+28,hidden),nn.GELU(),nn.Linear(hidden,48*18))
        nn.init.normal_(self.decoder[-1].weight,std=.01);nn.init.zeros_(self.decoder[-1].bias)
        self.sensory=nn.Sequential(nn.Linear(48*4*4*2,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.assimilation=nn.GRUCell(hidden,hidden)
        self.action=nn.Sequential(nn.Linear(40,hidden),nn.GELU())
        self.transition=nn.GRUCell(hidden,hidden)

    def encode_geometry(self,points,features,valid):
        if self.mode=='no_geometry':valid=torch.zeros_like(valid)
        key=self.point(features.float())+self.point_position(torch.cat([points/10,points.norm(dim=-1,keepdim=True)/10],-1))
        mask=valid.float();mean=(key*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        maximum=key.masked_fill(~valid[...,None],-1e4).amax(1);maximum=torch.where(valid.any(1,keepdim=True),maximum,0.)
        return points,key,valid,torch.cat([mean,maximum],-1)

    def initialize(self,geometry,sources):
        points,key,valid,global_feature=geometry;code=position_code(sources)
        semantic=torch.einsum('bkh,bsh->bks',self.query(code),key)/8
        d2=(points[:,None]-sources[:,:,None]).square().sum(-1)
        along=(points[:,None]*sources[:,:,None]).sum(-1)/sources.square().sum(-1,keepdim=True).clamp_min(.01)
        nearest=along.clamp(0,1)[...,None]*sources[:,:,None]
        ray2=(points[:,None]-nearest).square().sum(-1)
        def pool(score):
            w=score.masked_fill(~valid[:,None],-1e4).softmax(-1)*valid[:,None]
            w=w/w.sum(-1,keepdim=True).clamp_min(1e-6)
            return w@key
        local=pool(semantic-d2/2);path=pool(semantic-ray2/.5)
        return self.prior(torch.cat([global_feature[:,None].expand(-1,len(sources[0]),-1),local,path,code],-1))

    def decode(self,state,source):
        raw=self.decoder(torch.cat([state,position_code(source)],-1)).reshape(*source.shape[:-1],48,18)
        factor=torch.view_as_complex(raw[...,:16].reshape(*raw.shape[:-1],4,2,2).contiguous())
        covariance=factor@factor.conj().transpose(-1,-2)+.01*torch.eye(4,device=source.device)
        covariance=covariance/covariance.diagonal(dim1=-2,dim2=-1).real.sum(-1)[...,None,None]
        mix=torch.sigmoid(raw[...,16]-1)[...,None,None]
        q=(1-mix)*self.direct(source)+mix*covariance
        logvar=-2+2*torch.tanh(raw[...,17]/2)
        return q,logvar

    def assimilate(self,state,prediction,observation):
        if self.mode=='static':return state
        sensory=self.sensory(torch.view_as_real(observation-prediction).flatten(-4))
        return self.assimilation(sensory.reshape(-1,self.hidden),state.reshape(-1,self.hidden)).reshape_as(state)

    def imagine(self,state,source,action):
        new_source=move_source(source,action)
        if self.mode!='static':
            encoded=self.action(torch.cat([action,position_code(new_source)],-1))
            state=self.transition(encoded.reshape(-1,self.hidden),state.reshape(-1,self.hidden)).reshape_as(state)
        q,logvar=self.decode(state,new_source)
        return state,new_source,q,logvar

    def forward(self,points,features,valid,sources):
        state=self.initialize(self.encode_geometry(points,features,valid),sources)
        q,logvar=self.decode(state,sources)
        return q,logvar,state
