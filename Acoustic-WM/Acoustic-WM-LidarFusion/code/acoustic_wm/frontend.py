import json
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from .utils import input_path
from .audio.encoder import ReadOnlyClap,find_clap_snapshot
from .audio.spatial_features import covariance_from_wave
from .geometry.encoder import scene_tokens
from .geometry.object_proposals import CandidateProposals,source_hypotheses

class RawObservationEncoder:
    def __init__(self,config,device='cuda'):
        self.config=config;self.device=device
        self.proposals=CandidateProposals(config,device)
        D=self.proposals.D;root=Path(config['paths']['frontend_root'])
        self.calibration=json.loads((Path(config['paths']['dataset'])/'metadata/microphone_array.json').read_text())
        from scene_frontend.utonia_features import UtoniaFeatures
        projection=torch.load(root/'cache/utonia-xyz-v1/projection.pt',map_location=device,weights_only=True)
        self.geometry=UtoniaFeatures(root/'pretrained/utonia/utonia.pth',projection,device)
        self.clap=ReadOnlyClap(find_clap_snapshot(root),device)
        from scene_frontend.spatial import BandSRP,BandReliability
        self.srp=BandSRP(self.calibration['positions_lidar_m'],44100,device)
        ck=torch.load(Path(config['paths']['spatial'])/'adaptive.pt',map_location='cpu',weights_only=False)
        self.reliability=BandReliability(ck['adaptive']).to(device).eval();self.reliability.load_state_dict(ck['model'])

    @torch.inference_mode()
    def __call__(self,lidar,audio):
        with np.load(input_path(lidar),allow_pickle=False) as z:xyz=z['xyz'].astype(np.float32)
        wave,rate=sf.read(input_path(audio),dtype='float32',always_2d=True)
        return self.encode_arrays(xyz,wave,rate)

    @torch.inference_mode()
    def encode_arrays(self,xyz,wave,rate=44100):
        D=self.proposals.D
        observed,weight=covariance_from_wave(wave,rate)
        feat,pixels,inverse=self.geometry.encode(xyz)
        features=feat.half().cpu().numpy();inverse=D.align_inverse(xyz,pixels,inverse)
        centre,count=D.geometry_inputs(xyz,features,inverse)
        emb=self.clap([wave],rate)[0].cpu().numpy()
        score,coherence=self.srp(wave[None]);logits=self.reliability(torch.as_tensor(score,device=self.device),torch.as_tensor(coherence,device=self.device))
        posterior=(logits-logits.amax(1,keepdim=True)).exp()[0].cpu().numpy()
        silence=not bool(weight.any())
        if silence:posterior[:]=0
        x=dict(features=features,xyz=centre,count=count,inverse=inverse,audio=emb,acoustic=D.acoustic_features(centre,posterior),silence=silence)
        pred=self.proposals(x);points,valid=scene_tokens(centre)
        return dict(scene_points=points,scene_valid=valid,hypotheses=source_hypotheses(centre,count,pred['masks']),
            rank_features=pred['rank_features'],audio=emb,baseline=pred['baseline'],observed=observed,reliability=weight,
            xyz=centre,count=count,point_features=features[:,:16].astype(np.float32),full_point_features=features,inverse=inverse,masks=pred['masks'])
