"""Geometry-only object slots and inverse acoustic selection, trained jointly.

Not a reproduction of AQ3D/SeaVIS. This RGB-free adaptation adds metric query
anchors, full-resolution instance supervision and counterfactual source ranking
to the existing latent acoustic world. No labels enter forward or seeding.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .revised import RevisedWorldModel
from .suppression import upstream_module


def geometry_seeds(xyz, confidence, count):
    """Confidence-weighted farthest sampling from observable geometry only."""
    # Discrete selection is not differentiable; gathered features remain so.
    with torch.no_grad():
        batch, points, _ = xyz.shape
        count = min(count, points)
        chosen = confidence.argmax(-1)
        distance = torch.full_like(confidence, float('inf'))
        available = torch.ones_like(confidence, dtype=torch.bool)
        indices = []
        for _ in range(count):
            indices.append(chosen)
            anchor = xyz[torch.arange(batch, device=xyz.device), chosen]
            distance = torch.minimum(distance, (xyz-anchor[:, None]).square().sum(-1))
            available.scatter_(1, chosen[:, None], False)
            # Saturating distance avoids prioritizing far walls over tiny objects.
            priority = distance.clamp(max=4).sqrt() * (.05+confidence)
            chosen = priority.masked_fill(~available, -1).argmax(-1)
        return torch.stack(indices, 1)


class ObjectQueryWorldModel(RevisedWorldModel):
    def __init__(self, max_queries=32, query_layers=2, query_graph=True,
                 student_audio=False, anchor_mode='centroid', source_acoustics='pooled',
                 query_stride=8, local_geometry=False, **kwargs):
        super().__init__(**kwargs)
        dim=kwargs.get('audio_dim',128); channels=kwargs.get('base_channels',24)
        heads=kwargs.get('attention_heads',4)
        self.max_queries=max_queries
        if query_stride<1:
            raise ValueError('query_stride must be positive')
        self.query_stride=query_stride
        self.local_geometry=local_geometry
        self.student_audio=student_audio
        if anchor_mode not in ('centroid','observed'):
            raise ValueError('anchor_mode must be centroid or observed')
        self.anchor_mode=anchor_mode
        if source_acoustics not in ('pooled','center'):
            raise ValueError('source_acoustics must be pooled or center')
        self.source_acoustics=source_acoustics
        self.object_features=nn.Conv2d(channels,dim,1)
        self.object_local=None
        self.object_local_output=None
        if local_geometry:
            self.object_local=nn.Sequential(nn.Linear(dim+4,dim),nn.GELU(),nn.Linear(dim,dim),nn.LayerNorm(dim))
            self.object_local_output=nn.Sequential(nn.Linear(dim*2,dim),nn.GELU(),nn.Linear(dim,dim))
        self.object_position=nn.Sequential(nn.Linear(27,dim),nn.GELU(),nn.Linear(dim,dim))
        layer=nn.TransformerDecoderLayer(dim,heads,dim_feedforward=dim*3,
            dropout=.15,batch_first=True,norm_first=True)
        self.object_decoder=nn.TransformerDecoder(layer,query_layers)
        self.object_mask=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.object_class=nn.Linear(dim,17)
        self.object_offset=nn.Linear(dim,3)
        self.object_radius=nn.Linear(dim,1)
        nn.init.zeros_(self.object_offset.weight);nn.init.zeros_(self.object_offset.bias)
        nn.init.zeros_(self.object_radius.weight);nn.init.constant_(self.object_radius.bias,-.5)
        self.object_renderer=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,512))
        self.object_student=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,512))
        self.object_select_scale=nn.Parameter(torch.tensor([.25,.25,.25]))
        # Start with a conservative contribution to the already trained decoder.
        self.object_blend=nn.Parameter(torch.tensor(-2.))
        self.object_graph=None
        if query_graph:
            self.object_graph=upstream_module('directional_gnn').DirectionalGNN(
                dim,hidden_channels=dim,k_neighbors=6,dropout=.15,
                transform_channels=dim,gate_init=-4.,use_dedup=False)

    @staticmethod
    def position_features(xyz):
        # Relative metric coordinates + bounded Fourier features, no scene IDs.
        phases=xyz[...,None]*torch.tensor([.5,1.,2.,4.],device=xyz.device)
        return torch.cat((xyz/20,phases.sin().flatten(-2),phases.cos().flatten(-2)), -1)

    def forward(self,lidar,audio,valid=None,audio_embedding=None,text_logits=None):
        output=super().forward(lidar,audio,valid,audio_embedding,text_logits,return_features=True)
        valid=torch.ones_like(lidar[:,:1]) if valid is None else valid
        # Geometry features, seeds, slots and rendered audio are audio-independent.
        features=self.object_features(output.pop('geometry_features'))
        output.pop('geometry_tokens')
        stride=self.query_stride
        grid=(max(1,math.ceil(lidar.shape[-2]/stride)),max(1,math.ceil(lidar.shape[-1]/stride)))
        mass=F.adaptive_avg_pool2d(valid.float(),grid)
        xyz=F.adaptive_avg_pool2d(lidar[:,:3].float()*valid,grid)/mass.clamp_min(1e-5)*20
        xyz=xyz.flatten(2).transpose(1,2)
        memory=F.adaptive_avg_pool2d(features,grid).flatten(2).transpose(1,2)
        foreground=(1-output['semantic_logits'].float().softmax(1)[:,:1])
        confidence=(F.adaptive_max_pool2d(foreground,grid)*mass).flatten(1)
        if self.anchor_mode=='observed':
            # A mean across a depth discontinuity can lie in empty space, several
            # metres behind a narrow foreground object. Pick a *real observed*
            # point using predicted geometry-only foreground scores instead.
            peak,point_ids=F.max_pool2d(foreground*valid,stride,stride,ceil_mode=True,return_indices=True)
            point_ids=point_ids.flatten(2).transpose(1,2)
            xyz=(lidar[:,:3].float()*20).flatten(2).transpose(1,2).gather(1,point_ids.expand(-1,-1,3))
            memory=features.flatten(2).transpose(1,2).gather(1,point_ids.expand(-1,-1,features.shape[1]))
            confidence=peak.flatten(1)
        ids=geometry_seeds(xyz,confidence,self.max_queries)
        anchors=xyz.gather(1,ids[:,:,None].expand(-1,-1,3))
        query=memory.gather(1,ids[:,:,None].expand(-1,-1,memory.shape[-1]))
        if self.object_local is not None:
            with torch.autocast(device_type=lidar.device.type,enabled=False):
                distance=torch.cdist(anchors,xyz)
                distance=distance.masked_fill(mass.flatten(1)[:,None]<=0,float('inf'))
                neighbors=distance.topk(min(32,xyz.shape[1]),largest=False).indices
                batch_ids=torch.arange(xyz.shape[0],device=xyz.device)[:,None,None]
                relative=xyz[batch_ids,neighbors]-anchors[:,:,None]
                radius=relative.norm(dim=-1,keepdim=True)
                content=memory[batch_ids,neighbors].float()
                local=self.object_local(torch.cat((content,relative,radius),-1))
                # Same local metric scale regardless of sensor distance/view.
                weight=torch.exp(-radius.square()/2).clamp_min(1e-5)
                summary=torch.cat(((local*weight).sum(2)/weight.sum(2),
                    (local*weight).amax(2)), -1)
                query=query+self.object_local_output(summary)
        memory=memory+self.object_position(self.position_features(xyz))
        query=query+self.object_position(self.position_features(anchors))
        query=self.object_decoder(query,memory)
        if self.object_graph is not None and query.shape[1]>1:
            with torch.autocast(device_type=lidar.device.type,enabled=False):
                distance=torch.cdist(anchors,anchors)
                eye=torch.eye(anchors.shape[1],device=lidar.device,dtype=torch.bool)[None]
                knn=distance.masked_fill(eye,float('inf')).topk(min(6,anchors.shape[1]-1),largest=False).indices
                query=self.object_graph(anchors.transpose(1,2),query.float().transpose(1,2),knn_idx=knn).transpose(1,2)
        # FP32 log odds and metric distance prevent BF16 saturation and overflow.
        with torch.autocast(device_type=lidar.device.type,enabled=False):
            centers=anchors+self.object_offset(query.float()).tanh()*.75
            radii=.15+1.85*self.object_radius(query.float()).sigmoid()
            mask_feature=self.object_mask(query.float())
            mask_logits=torch.einsum('bqd,bdhw->bqhw',mask_feature,features.float())/math.sqrt(features.shape[1])
            full_xyz=lidar[:,:3].float()*20
            # Squared-distance identity avoids a [B,Q,3,H,W] allocation.
            dist2=(centers.square().sum(-1)[:,:,None,None]+full_xyz.square().sum(1)[:,None]
                   -2*torch.einsum('bqd,bdhw->bqhw',centers,full_xyz)).clamp_min(0)
            mask_logits=mask_logits+1-(dist2/(2*radii.square()[:,:,None])).clamp_max(30)
            masks=mask_logits.sigmoid()*valid
            pooled=torch.einsum('bqhw,bdhw->bqd',masks,features.float())/masks.sum((-2,-1)).clamp_min(1)[:,:,None]
        rendered=F.normalize(self.object_renderer(query+pooled),dim=-1)
        student=F.normalize(self.object_student(output.pop('spatial_audio_global')),dim=-1)
        observed=student if self.student_audio else output['observed_audio']
        class_logits=self.object_class(query+pooled)
        spectrum=output.pop('audio_spectrum')
        with torch.autocast(device_type=lidar.device.type,enabled=False):
            scores=torch.einsum('bqd,bd->bq',rendered.float(),observed.float())*10
            coarse_masks=F.adaptive_avg_pool2d(masks,output['physical_render_logits'].shape[-2:])
            acoustic=(coarse_masks*output['physical_render_logits'][:,None]).sum((-2,-1))/coarse_masks.sum((-2,-1)).clamp_min(1e-5)
            if self.source_acoustics=='center' and centers.shape[1]>1:
                # Counterfactual six-pair spherical IPD at each object's actual
                # hypothesized position, instead of a blurred range-image cell.
                acoustic=self.render_phase(centers.transpose(1,2).unsqueeze(-1),spectrum).squeeze(-1)
            semantic=torch.einsum('bqc,bc->bq',class_logits.float().softmax(-1)[...,1:],output['category_logits'].float().softmax(-1))
            objectness=1-class_logits.float().softmax(-1)[...,0]
            scales=F.softplus(self.object_select_scale.float())
            scores=scores+scales[0]*acoustic+scales[1]*semantic.clamp_min(1e-5).log()+scales[2]*objectness.clamp_min(1e-5).log()
            selected=torch.einsum('bq,bqhw->bhw',scores.softmax(-1),mask_logits.sigmoid())[:,None]
            mixture=torch.logit(selected.clamp(1e-5,1-1e-5))
            dense=output['mask_logits'].float()
            blend=self.object_blend.sigmoid()
            output['mask_logits']=(1-blend)*dense+blend*mixture
        output.pop('audio_global')
        output.update(object_mask_logits=mask_logits,object_scores=scores,
                      object_class_logits=class_logits,object_centers=centers,
                      object_rendered_audio=rendered,student_audio=student,
                      dense_mask_logits=dense,query_mask_logits=mixture)
        return output
