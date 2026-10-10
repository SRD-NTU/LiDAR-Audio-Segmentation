"""Mask-candidate adaptation of the MIT-licensed LS3D suppression graph.

Upstream stays unmodified at learned-suppression-3d/. Proposals and node features
use ONLY geometry and world-model predictions, never annotation-derived regions.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import hashlib
from functools import lru_cache
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import torch
from torch import nn


@lru_cache(maxsize=2)
def upstream_module(name):
    root=Path(__file__).resolve().parents[2]/'learned-suppression-3d'/'ls3d'
    relative={'suppression':'models/suppression.py','candidates':'postprocess/candidates.py',
      'directional_gnn':'models/directional_gnn.py'}[name]
    import sys
    if name=='directional_gnn':
        import types
        package='_lidarwm_ls3d_models'
        if package not in sys.modules:
            parent=types.ModuleType(package);parent.__path__=[str(root/'models')];sys.modules[package]=parent
        module_name=package+'.directional_gnn'
    else:module_name='_lidarwm_ls3d_'+name
    spec=importlib.util.spec_from_file_location(module_name,root/relative)
    module=importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through the module registry.
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


class MaskSuppression(nn.Module):
    def __init__(self,in_dim,hidden=96,dropout=.25,suppress_radius=.08,feature_mode='all',residual_quality=False):
        super().__init__()
        self.feature_mode=feature_mode
        dimension={'compact':34,'physics':17}.get(feature_mode,in_dim)
        self.graph=upstream_module('suppression').LearnedSuppression(
            dimension,hidden=hidden,rel_hidden=48,dropout=dropout,
            suppress_radius=suppress_radius,n_rounds=2)
        self.quality=nn.Sequential(nn.Linear(dimension,hidden),nn.GELU(),nn.Dropout(dropout),
          nn.Linear(hidden,1)) if residual_quality else None

    def forward(self,features,positions,valid=None):
        # LS3D uses cdist, sigmoid/logit and displacement updates. Keep its small
        # graph in FP32 to avoid the BF16 probability saturation seen previously.
        with torch.autocast(device_type=features.device.type,enabled=False):
            features=features.float();positions=positions.float()
            if self.feature_mode=='compact':
                # Physically interpretable descriptors resist memorizing the
                # training scenes through high-dimensional geometry latents.
                indices=list(range(10))+[266,267,268,302]+list(range(286,302))
                semantic_match=(features[...,270:286]*features[...,286:302]).sum(-1,keepdim=True)/16
                features=torch.cat((features[...,indices],semantic_match,positions),-1)
            elif self.feature_mode=='physics':
                features=torch.cat((features[...,list(range(10))+[266,267,268,302]],positions),-1)
            logit,displacement=self.graph(features,positions,valid)
            if self.quality is not None:
                # Independent overlap quality avoids penalizing a correct object
                # merely because it generated many overlapping candidate masks.
                logit=.25*logit+self.quality(features).squeeze(-1)
            return logit,displacement


class WorldFeatures:
    """Expose frozen geometry tokens without changing world-model checkpoints."""
    def __init__(self,model):
        self.model=model;self.geometry=None
        self.handle=model.scene_context.register_forward_hook(self._capture)

    def _capture(self,module,args,output):self.geometry=output.detach()

    @torch.inference_mode()
    def __call__(self,lidar,audio,valid,**extra):
        output=self.model(lidar,audio,valid,**extra)
        output['geometry_tokens']=self.geometry
        self.geometry=None
        return output

    def close(self):self.handle.remove()


def surface_edges(xyz,valid):
    """Periodic range-image adjacency, gated by actual 3D distances in metres."""
    height,width=valid.shape;ids=np.arange(height*width).reshape(height,width)
    points=xyz.reshape(height,width,3)
    sources=[];targets=[]
    for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
        neighbor=np.roll(ids,(-dr,-dc),(0,1))
        other=np.roll(points,(-dr,-dc),(0,1))
        usable=valid&np.roll(valid,(-dr,-dc),(0,1))
        if dr:usable[-dr:]=False  # Vertical beams do not wrap; azimuth does.
        ranges=np.minimum(np.linalg.norm(points,axis=-1),np.linalg.norm(other,axis=-1))
        usable&=np.linalg.norm(points-other,axis=-1)<(.06+.02*ranges)
        sources.append(ids[usable]);targets.append(neighbor[usable])
    return np.concatenate(sources),np.concatenate(targets)


def make_proposals(lidar,valid,probability,baseline_threshold,thresholds=(.015,.03,.06,.12,.25,.5,.75,.9),
                   max_candidates=80,min_points=3,components_per_threshold=8):
    """A geometry-gated multithreshold mask bank, independent of ground truth.

    Candidate zero is the exact baseline mask, providing a measurable fallback.
    Subsequent masks are complete threshold masks and individual surface-connected
    components; no semantic/instance IDs or source coordinates are consulted.
    """
    height,width=valid.shape;n=height*width
    valid=valid.astype(bool);p=probability.reshape(-1);v=valid.reshape(-1)
    xyz=lidar[:3].transpose(1,2,0).reshape(n,3)*20
    edge_a,edge_b=surface_edges(xyz,valid)
    bank=[];seen=set()

    def add(mask):
        mask=mask&v
        packed=np.packbits(mask,bitorder='little');digest=hashlib.sha1(packed.tobytes()).digest()
        if digest not in seen and len(bank)<max_candidates:
            bank.append(packed);seen.add(digest)

    add(p>=baseline_threshold)
    for threshold in sorted(set(thresholds)|{float(baseline_threshold)}):
        active=(p>=threshold)&v
        if not active.any():continue
        add(active)
        use=active[edge_a]&active[edge_b]
        graph=coo_matrix((np.ones(int(use.sum()),dtype=np.uint8),(edge_a[use],edge_b[use])),shape=(n,n)).tocsr()
        _,labels=connected_components(graph,directed=False)
        counts=np.bincount(labels[active],minlength=n)
        mass=np.bincount(labels[active],weights=p[active],minlength=n)
        eligible=np.flatnonzero(counts>=min_points)
        ranked=eligible[np.argsort(-(mass[eligible]/np.sqrt(counts[eligible])))[:components_per_threshold]]
        for component in ranked:add(active&(labels==component))
    return np.stack(bank)


def proposal_features(lidar,valid,probability,packed_masks,geometry,semantic,category,render,physical,doa):
    """Voter statistics + world latents + neighbors + predicted acoustic evidence."""
    h,w=valid.shape;n=h*w
    masks=np.unpackbits(packed_masks,axis=1,count=n,bitorder='little').astype(bool)
    xyz=lidar[:3].transpose(1,2,0).reshape(n,3)
    p=probability.reshape(-1).astype('float32')
    gh,gw=render.shape
    row,col=np.indices((h,w));cell=(row*gh//h*gw+col*gw//w).reshape(-1)
    semantics=semantic.reshape(semantic.shape[0],n).T
    azimuth=np.minimum(col.reshape(-1)*len(doa)//w,len(doa)-1)
    stats=[];pooled=[];positions=[];acoustics=[];areas=[]
    for mask in masks:
        index=np.flatnonzero(mask);areas.append(len(index))
        if not len(index):
            stats.append(np.zeros(10));pooled.append(np.zeros(geometry.shape[-1]));positions.append(np.zeros(3))
            acoustics.append(np.r_[np.zeros(3),np.zeros(semantic.shape[0]),category,1.]);continue
        values=p[index];weights=values/values.sum().clip(1e-8)
        points=xyz[index];center=(points*weights[:,None]).sum(0)
        mass=np.bincount(cell[index],weights=weights,minlength=gh*gw)
        pooled.append(mass@geometry)
        positions.append(center)
        ranges=np.linalg.norm(points,axis=-1)
        extent=points.max(0)-points.min(0)
        stats.append(np.r_[np.log1p(len(index))/np.log1p(n),values.mean(),values.max(),values.std(),
          np.quantile(values,.75),ranges.mean(),ranges.std(),extent])
        acoustics.append(np.r_[mass@render.reshape(-1),mass@physical.reshape(-1),
          weights@doa[azimuth[index]],weights@semantics[index],category,0.])
    positions=np.asarray(positions,dtype='float32');pooled=np.asarray(pooled,dtype='float32')
    neighbors=upstream_module('candidates').neighbor_mean(positions,pooled,4)
    features=np.concatenate((np.asarray(stats),pooled,neighbors,np.asarray(acoustics)),axis=1).astype('float32')
    # Distinguish the full-baseline fallback independently of emptiness.
    features[:,-1]=0;features[0,-1]=1
    if not np.isfinite(features).all():raise FloatingPointError('Nonfinite candidate features')
    return features,positions,np.asarray(areas,dtype='int32')


def overlap_targets(packed_masks,truth):
    truth=truth.reshape(-1).astype(bool)
    masks=np.unpackbits(packed_masks,axis=1,count=len(truth),bitorder='little').astype(bool)
    intersection=(masks&truth[None]).sum(1)
    areas=masks.sum(1);target=int(truth.sum())
    union=areas+target-intersection
    iou=(intersection+1e-9)/(union+1e-9)
    return iou.astype('float32'),intersection.astype('int32'),target


def choose_candidate(scores,margin=0.):
    scores=np.asarray(scores)
    winner=int(scores.argmax())
    return winner if scores[winner]>scores[0]+margin else 0


def output_candidates(lidar,valid,output,index,baseline_threshold):
    """Shared cache/deployment implementation: no target or metadata arguments."""
    x=lidar[index].detach().float().cpu().numpy()
    v=valid[index,0].detach().cpu().numpy()>.5
    probability=output['mask_logits'][index,0].detach().float().sigmoid().cpu().numpy()
    masks=make_proposals(x,v,probability,baseline_threshold)
    def array(key):return output[key][index].detach().float().cpu().numpy()
    semantic=output['semantic_logits'][index].detach().float().softmax(0).cpu().numpy()
    category=output['category_logits'][index].detach().float().softmax(-1).cpu().numpy()
    features,positions,areas=proposal_features(x,v,probability,masks,array('geometry_tokens'),semantic,category,
      array('render_logits'),array('physical_render_logits'),array('doa_logits'))
    return features,positions,masks,areas


class SuppressedWorldModel(nn.Module):
    """Deployable frozen world model + LS3D candidate scorer + point-mask decoder.

    The refiner is trained separately on frozen-world candidates. Returning hard
    masks as +/-20 logits preserves the ordinary point-wise inference API.
    """
    def __init__(self,base_architecture='revised',suppression_dim=303,suppression_hidden=96,
                 suppression_dropout=.25,suppression_feature_mode='all',suppression_residual_quality=False,
                 baseline_threshold=.05,suppression_margin=0.,suppression_context_radius=0.,
                 suppression_movement_scale=0.,**kwargs):
        super().__init__()
        if base_architecture=='revised':
            from .revised import RevisedWorldModel
            self.base=RevisedWorldModel(**kwargs)
        elif base_architecture=='instance_field':
            from .instance_field import InstanceFieldWorldModel
            self.base=InstanceFieldWorldModel(**kwargs)
        elif base_architecture=='graph_world':
            from .graph_world import GraphLatentWorldModel
            self.base=GraphLatentWorldModel(**kwargs)
        else:raise ValueError('Suppression requires revised or instance_field geometry tokens')
        self.refiner=MaskSuppression(suppression_dim,hidden=suppression_hidden,dropout=suppression_dropout,
          feature_mode=suppression_feature_mode,residual_quality=suppression_residual_quality)
        self.register_buffer('feature_mean',torch.zeros(suppression_dim))
        self.register_buffer('feature_std',torch.ones(suppression_dim))
        self.baseline_threshold=baseline_threshold;self.suppression_margin=suppression_margin
        self.context_radius=suppression_context_radius;self.movement_scale=suppression_movement_scale
        self.extract=WorldFeatures(self.base)

    @torch.inference_mode()
    def forward(self,lidar,audio,valid=None,**extra):
        valid=torch.ones_like(lidar[:,:1]) if valid is None else valid
        # Match cache construction's GPU BF16 arithmetic exactly.
        with torch.autocast(device_type=lidar.device.type,dtype=torch.bfloat16,enabled=lidar.device.type=='cuda'):
            output=self.extract(lidar,audio,valid,**extra)
        selected=[]
        for i in range(lidar.shape[0]):
            features,positions,masks,_=output_candidates(lidar,valid,output,i,self.baseline_threshold)
            z=torch.from_numpy(features).to(lidar.device)
            z=((z-self.feature_mean)/self.feature_std).clamp(-6,6)
            c=torch.from_numpy(positions).to(lidar.device)
            logit,displacement=self.refiner(z,c)
            scores=logit.sigmoid().cpu().numpy()
            winner=choose_candidate(scores,self.suppression_margin)
            mask=np.unpackbits(masks[winner],count=valid.shape[-2]*valid.shape[-1],bitorder='little')
            if self.context_radius>0 and winner!=0:
                base_mask=np.unpackbits(masks[0],count=len(mask),bitorder='little')
                xyz=lidar[i,:3].detach().float().cpu().numpy().reshape(3,-1).T
                center=positions[winner]+self.movement_scale*displacement[winner].cpu().numpy()
                distance=np.linalg.norm(xyz-center,axis=-1)
                mask=((base_mask|mask)&(distance<=self.context_radius/20)).astype('uint8')
            selected.append(torch.from_numpy(mask.copy()).reshape(1,*valid.shape[-2:]))
        output['baseline_mask_logits']=output['mask_logits']
        output['mask_logits']=torch.stack(selected).to(lidar.device).float()*40-20
        return output
