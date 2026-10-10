"""Pretrained metric-3D geometry + fine boundary path + acoustic object world."""
import torch
from torch import nn
from torch.nn import functional as F
from .object_world import ObjectQueryWorldModel
from .sonata_geometry import SonataGeometry
from .reliable_acoustics import ReliablePhaseRenderer


class FoundationWorldModel(ObjectQueryWorldModel):
    def __init__(self,geometry_checkpoint='pretrained/sonata/sonata.pth',geometry_grid_size=.05,
                 geometry_frozen=True,reliable_acoustics=True,geometry_feature_dropout=.1,**kwargs):
        super().__init__(**kwargs)
        c=kwargs.get('base_channels',24);dim=kwargs.get('audio_dim',128)
        self.geometry_foundation=SonataGeometry(geometry_checkpoint,grid_size=geometry_grid_size,frozen=geometry_frozen)
        self.geometry_adapter=nn.Sequential(nn.Conv2d(256,64,1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,c,1))
        self.geometry_context=nn.Sequential(nn.Conv2d(256,dim,1),nn.GroupNorm(8,dim),nn.GELU())
        self.geometry_fine=nn.Sequential(nn.Conv2d(256,c,1),nn.GroupNorm(8,c),nn.GELU())
        self.geometry_gate=nn.Parameter(torch.tensor(.1))
        self.geometry_feature_dropout=nn.Dropout2d(geometry_feature_dropout)
        self.object_audio_category=nn.Sequential(nn.Linear(dim*2,dim),nn.GELU(),nn.Dropout(.2),nn.Linear(dim,16))
        nn.init.zeros_(self.object_audio_category[-1].weight);nn.init.zeros_(self.object_audio_category[-1].bias)
        self.phase_renderer=ReliablePhaseRenderer(self.audio_encoder.n_fft,self.audio_encoder.sample_rate) if reliable_acoustics else None
        self._cached_geometry=None

    def acoustic_categories(self,semantic_audio,state,text_logits):
        # Trainable waveform-derived semantics complement the frozen CLAP prior;
        # no true category or category-dependent audio cropping is supplied.
        return super().acoustic_categories(semantic_audio,state,text_logits)+self.object_audio_category(torch.cat((semantic_audio,state['global']),-1))

    def encode_geometry(self,lidar,valid):
        features=self._cached_geometry
        if features is None:features=self.geometry_foundation(lidar,valid)
        elif isinstance(features,list):features=self.decode_cache(features,lidar.shape[-2:])
        features=features.float()*valid
        if features.shape[1:]!=(256,*lidar.shape[-2:]):raise ValueError('Invalid observation-only geometry feature shape')
        features=self.geometry_feature_dropout(features)
        stem=self.stem(torch.cat((lidar*valid,valid),1))
        # Full-resolution skip protects clocks/keyboards/thin appliances from
        # range-image downsampling. The main contextual state uses actual 3D PTv3.
        skips=[stem+self.geometry_adapter(features)*self.geometry_gate.tanh()]
        for down in self.down:skips.append(down(skips[-1]))
        mass=F.adaptive_avg_pool2d(valid,skips[-1].shape[-2:])
        contextual=F.adaptive_avg_pool2d(features*valid,skips[-1].shape[-2:])/mass.clamp_min(1e-5)
        scene=skips[-1]+self.geometry_context(contextual)
        geometry=self.scene_context(scene.flatten(2).transpose(1,2))
        geo=geometry.transpose(1,2).reshape(scene.shape)
        for decoder,skip in zip(self.geo_decoder,reversed(skips[:-1])):geo=decoder(geo,skip)
        geo=geo+self.geometry_fine(features)
        return skips,geometry,geo,self.semantic_head(geo)

    @staticmethod
    def decode_cache(entries,shape):
        grids=[]
        for entry in entries:
            feature=entry['features'].float()
            grid=feature.new_zeros(shape[0]*shape[1],feature.shape[-1])
            grid=grid.index_copy(0,entry['pixels'].long(),feature[entry['inverse'].long()])
            grids.append(grid.T.reshape(feature.shape[-1],*shape))
        return torch.stack(grids)

    def forward(self,lidar,audio,valid=None,audio_embedding=None,text_logits=None,geometry_features=None):
        if geometry_features is not None and not self.geometry_foundation.frozen:
            raise ValueError('Fine-tuning the encoder requires raw XYZ, not cached features')
        self._cached_geometry=geometry_features
        try:return super().forward(lidar,audio,valid,audio_embedding,text_logits)
        finally:self._cached_geometry=None

    def render_phase(self,xyz,spec):
        if self.phase_renderer is None:return super().render_phase(xyz,spec)
        return self.phase_renderer(xyz,spec,self.audio_encoder.mic_positions)
