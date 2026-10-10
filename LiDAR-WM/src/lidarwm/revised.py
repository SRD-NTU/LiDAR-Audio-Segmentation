"""Scene-context geometry and frozen semantic acoustic rendering."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import ResidualBlock, DecoderBlock, AudioFiLM
from .audio import AudioSpatialEncoder
from .audio import MIC_PAIRS

class ContextDecoder(nn.Module):
    def __init__(self, incoming, skip, outgoing):
        super().__init__()
        self.blocks=nn.Sequential(ResidualBlock(incoming+skip,outgoing),ResidualBlock(outgoing,outgoing))
    def forward(self,x,skip):
        x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        return self.blocks(torch.cat((x,skip),1))

class RevisedWorldModel(nn.Module):
    def __init__(self,base_channels=24,audio_dim=128,num_categories=16,lidar_width=1800,
                 audio_sample_rate=14700,n_fft=512,hop_length=192,attention_heads=4):
        super().__init__(); c=base_channels
        self.stem=nn.Sequential(ResidualBlock(8,c),ResidualBlock(c,c))
        self.down=nn.ModuleList([nn.Sequential(ResidualBlock(c,c*2,2),ResidualBlock(c*2,c*2)),
          nn.Sequential(ResidualBlock(c*2,c*4,2),ResidualBlock(c*4,c*4)),
          nn.Sequential(ResidualBlock(c*4,c*6,2),ResidualBlock(c*6,c*6)),
          nn.Sequential(ResidualBlock(c*6,audio_dim,2),ResidualBlock(audio_dim,audio_dim))])
        layer=nn.TransformerEncoderLayer(audio_dim,attention_heads,dim_feedforward=audio_dim*3,
          dropout=.15,batch_first=True,norm_first=True)
        self.scene_context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.geo_decoder=nn.ModuleList([ContextDecoder(audio_dim,c*6,c*6),ContextDecoder(c*6,c*4,c*4),
          ContextDecoder(c*4,c*2,c*2),ContextDecoder(c*2,c,c)])
        self.semantic_head=nn.Conv2d(c,num_categories+1,1)
        self.audio_encoder=AudioSpatialEncoder(audio_dim,audio_sample_rate,n_fft,hop_length,225)
        self.foundation=nn.Sequential(nn.Linear(512,audio_dim),nn.LayerNorm(audio_dim),nn.GELU(),nn.Dropout(.3))
        self.category_residual=nn.Linear(audio_dim,num_categories)
        nn.init.zeros_(self.category_residual.weight); nn.init.zeros_(self.category_residual.bias)
        self.renderer=nn.Sequential(nn.Linear(audio_dim,256),nn.GELU(),nn.Linear(256,512))
        self.attention=nn.MultiheadAttention(audio_dim,attention_heads,batch_first=True,dropout=.15)
        self.fusion=ResidualBlock(audio_dim+3,audio_dim)
        self.mask_decoder=nn.ModuleList([DecoderBlock(audio_dim,c*6,c*6,audio_dim),
          DecoderBlock(c*6,c*4,c*4,audio_dim),DecoderBlock(c*4,c*2,c*2,audio_dim),
          DecoderBlock(c*2,c,c,audio_dim)])
        self.mask_head=nn.Conv2d(c,1,1)
        self.semantic_scale=nn.Parameter(torch.tensor(.2))
        self.doa_residual=nn.Sequential(nn.Linear(audio_dim,225),nn.Tanh())
        self.position_head=nn.Sequential(nn.Linear(audio_dim,64),nn.GELU(),nn.Linear(64,3),nn.Tanh())

    def encode_geometry(self,lidar,valid):
        skips=[self.stem(torch.cat((lidar*valid,valid),1))]
        for down in self.down: skips.append(down(skips[-1]))
        shape=skips[-1].shape
        geometry=self.scene_context(skips[-1].flatten(2).transpose(1,2))
        scene=geometry.transpose(1,2).reshape(shape)
        geo=scene
        for decoder,skip in zip(self.geo_decoder,reversed(skips[:-1])): geo=decoder(geo,skip)
        semantic=self.semantic_head(geo)
        return skips,geometry,geo,semantic

    def acoustic_categories(self,semantic_audio,state,text_logits):
        return text_logits.detach()/2+self.category_residual(semantic_audio)

    def forward(self,lidar,audio,valid=None,audio_embedding=None,text_logits=None,return_features=False):
        if audio_embedding is None or text_logits is None:
            raise ValueError('RevisedWorldModel requires waveform-derived CLAP features. Run prepare_audio.py.')
        valid=torch.ones_like(lidar[:,:1]) if valid is None else valid
        skips,geometry,geo,semantic=self.encode_geometry(lidar,valid)
        shape=skips[-1].shape
        state=self.audio_encoder(audio)
        # Counterfactual direct-path rendering at every coarse 3D region:
        # spherical distance differences predict all six observable IPDs.
        geometry_xyz=F.adaptive_avg_pool2d(lidar[:,:3]*valid,shape[-2:])
        geometry_xyz=geometry_xyz/F.adaptive_avg_pool2d(valid,shape[-2:]).clamp_min(1e-3)*20
        physical=self.render_phase(geometry_xyz,state['spectrum'])
        semantic_audio=self.foundation(audio_embedding)
        observed=F.normalize(audio_embedding,dim=-1)
        rendered=F.normalize(self.renderer(geometry),dim=-1)
        scores=torch.einsum('bnd,bd->bn',rendered,observed)*10
        render_map=scores.reshape(shape[0],shape[2],shape[3])
        audio_global=state['global']+semantic_audio
        doa=state['doa']+self.doa_residual(state['global'])*2
        prior=F.interpolate(doa[:,None,None],size=shape[-2:],mode='bilinear',align_corners=False)
        att,_=self.attention(geometry,state['tokens'],state['tokens'],need_weights=False)
        x=(geometry+.2*att).transpose(1,2).reshape(shape)
        x=self.fusion(torch.cat((x,render_map[:,None],prior,physical[:,None]),1))
        for decoder,skip in zip(self.mask_decoder,reversed(skips[:-1])): x=decoder(x,skip,audio_global)
        categories=self.acoustic_categories(semantic_audio,state,text_logits)
        # BF16 rounds 1-1e-4 to 1, yielding infinite log-odds and gradients.
        # Disable autocast for both probability mixing and the logit transform.
        with torch.autocast(device_type=lidar.device.type,enabled=False):
            semantic_match=torch.einsum('bchw,bc->bhw',semantic.float().softmax(1)[:,1:],categories.float().softmax(-1))[:,None]
            semantic_logit=torch.logit(semantic_match.clamp(1e-4,1-1e-4))
        logits=self.mask_head(x)+self.semantic_scale*semantic_logit
        output=dict(mask_logits=logits,semantic_logits=semantic,render_logits=render_map,
          rendered_audio=rendered,observed_audio=observed,category_logits=categories,
          doa_logits=doa,physics_doa=state['doa'],source_position=self.position_head(state['global']),
          physical_render_logits=physical)
        if return_features:
            output.update(geometry_features=geo,geometry_tokens=geometry,
                          audio_global=audio_global,spatial_audio_global=state['global'],
                          audio_spectrum=state['spectrum'])
        return output

    def render_phase(self,xyz,spec):
        with torch.autocast(device_type=xyz.device.type,enabled=False):
            xyz=xyz.float(); positions=xyz.flatten(2).transpose(1,2)
            distance=(positions[:,:,None]-self.audio_encoder.mic_positions[None,None]).norm(dim=-1)
            frequency=torch.fft.rfftfreq(self.audio_encoder.n_fft,1/self.audio_encoder.sample_rate).to(xyz.device)
            keep=(frequency>=200)&(frequency<=3500); frequency=frequency[keep]
            scores=[]
            for left,right in MIC_PAIRS:
                cross=spec[:,left,keep]*spec[:,right,keep].conj()
                # Averaging before normalization suppresses unstable phase bins.
                average=cross.mean(-1)
                coherence=average.abs()/cross.abs().mean(-1).clamp_min(1e-8)
                observed=average/average.abs().clamp_min(1e-8)
                delta=(distance[:,:,right]-distance[:,:,left])/343
                predicted=torch.exp(2j*torch.pi*delta[:,:,None]*frequency[None,None])
                agreement=(predicted*observed[:,None].conj()).real
                scores.append((agreement*coherence[:,None]).sum(-1)/coherence.sum(-1,keepdim=True).clamp_min(1e-5))
            score=torch.stack(scores).mean(0)
            score=(score-score.mean(-1,keepdim=True))/(score.std(-1,keepdim=True)+1e-5)
            return score.reshape(xyz.shape[0],xyz.shape[2],xyz.shape[3])
