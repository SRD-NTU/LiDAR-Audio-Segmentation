"""Deploy ORBIT from XYZ and four-channel WAV; annotations are never opened."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from apex import data as D
from apex.predict import range_grid
from .inference import load_models,propose_and_score


class ObservationEncoder:
    """Reusable XYZ/WAV extractor executing only the selected backbone."""
    def __init__(self,calibration,geometry='sonata',projection=None,device='cuda',spatial=None):
        self.calibration=calibration;self.geometry=geometry;self.device=device
        root=D.ROOT/'ORBIT'
        if geometry=='utonia':
            from .utonia_features import UtoniaFeatures
            if projection is None:raise ValueError('Utonia requires the TRAIN-fitted projection')
            p=torch.load(projection,map_location='cpu',weights_only=True)
            self.encoder=UtoniaFeatures(root/'pretrained/utonia/utonia.pth',p,device)
        elif geometry=='sonata':
            from lidarwm.sonata_geometry import SonataGeometry
            self.encoder=SonataGeometry(checkpoint=str(root/'pretrained/sonata/sonata.pth')).to(device).eval()
        else:raise ValueError(f'Unknown geometry backbone: {geometry}')
        from lidarwm.foundation import FrozenClap
        self.clap=FrozenClap(device)
        self.scorer=self.adapter=None
        if spatial is not None:
            from .spatial import BandSRP,BandReliability
            ck=torch.load(spatial,map_location='cpu',weights_only=False)
            self.scorer=BandSRP(calibration['positions_lidar_m'],44100,device)
            self.adapter=BandReliability(ck['adaptive']).to(device).eval()
            self.adapter.load_state_dict(ck['model'])

    @torch.inference_mode()
    def __call__(self,xyz,wave,rate):
        if rate!=44100 or wave.ndim!=2 or wave.shape[1]!=4:
            raise ValueError('Expected four physical 44100 Hz microphone channels')
        if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(wave).all():
            raise ValueError('Invalid or non-finite XYZ/WAV input')
        if self.geometry=='utonia':
            features,pixels,inverse=self.encoder.encode(xyz)
            features=features.half().cpu().numpy()
        else:
            grid,valid,_=range_grid(xyz)
            encoded=self.encoder._encode(torch.from_numpy(grid)[None].to(self.device),
                torch.from_numpy(valid)[None].to(self.device),sparse=True)[0]
            features=encoded['features'].half().cpu().numpy()
            pixels=encoded['pixels'].cpu().numpy();inverse=encoded['inverse'].cpu().numpy()
        inverse=D.align_inverse(xyz,pixels,inverse)
        centre,count=D.geometry_inputs(xyz,features,inverse)
        embedding,_=self.clap([wave],sample_rate=rate)
        silent=bool(np.max(np.abs(wave),initial=0)<=1e-12)
        if silent:posterior=np.zeros(D.acoustic.N_DIRS,np.float32)
        elif self.scorer is not None:
            scores,coherence=self.scorer(wave[None])
            logits=self.adapter(torch.as_tensor(scores,device=self.device),torch.as_tensor(coherence,device=self.device))[0]
            posterior=(logits-logits.max()).exp().cpu().numpy()
        else:
            posterior=D.acoustic.srp_phat_3d(wave.astype(np.float64),rate,np.asarray(self.calibration['positions_lidar_m']))
        return dict(features=features,xyz=centre,count=count,inverse=inverse,
                    acoustic=D.acoustic_features(centre,posterior),audio=embedding[0].cpu().numpy(),silence=silent)


def observation_inputs(xyz,wave,rate,calibration,geometry='sonata',projection=None,device='cuda',spatial=None):
    if rate!=44100 or wave.ndim!=2 or wave.shape[1]!=4 or not len(wave):
        raise ValueError('Expected nonempty four-channel 44100 Hz waveform')
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(wave).all():
        raise ValueError('Invalid or non-finite XYZ/WAV input')
    return ObservationEncoder(calibration,geometry,projection,device,spatial)(xyz,wave,rate)


def main():
    p=argparse.ArgumentParser();p.add_argument('--masks',type=Path,required=True)
    p.add_argument('--selector',type=Path,required=True);p.add_argument('--apex',type=Path,required=True)
    p.add_argument('--lidar',type=Path,required=True);p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=D.DATA/'metadata/microphone_array.json')
    p.add_argument('--geometry',choices=['sonata','utonia'],default='sonata');p.add_argument('--projection',type=Path)
    p.add_argument('--spatial',type=Path);p.add_argument('--reranker',type=Path)
    p.add_argument('--gate',type=Path);p.add_argument('--gate-weight',type=float,default=1.)
    p.add_argument('--gate2',type=Path);p.add_argument('--gate2-weight',type=float,default=0.)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rule',choices=['hybrid','selector','apex_quality','apex','audio_gate'],default='hybrid')
    p.add_argument('--device',default='cuda');args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.rule=='apex' and args.reranker:raise ValueError('The reranker applies only to proposal rules')
    if (args.rule=='audio_gate') != (args.gate is not None):
        raise ValueError('The audio_gate rule requires --gate, and --gate requires that rule')
    if bool(args.gate2) != bool(args.gate2_weight) or (args.gate2 and not args.gate):
        raise ValueError('gate2 requires a nonzero gate2-weight and the audio_gate rule')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    apex_config=torch.load(args.apex,map_location='cpu',weights_only=False)['config']
    expected=apex_config['train_provenance']['calibration_sha256']
    if D.sha(args.calibration)!=expected:raise ValueError('Microphone calibration differs from training')
    with np.load(args.lidar,allow_pickle=False) as z:xyz=z['xyz'].astype(np.float32)
    wave,rate=sf.read(args.audio,dtype='float32')
    if rate!=44100 or wave.ndim!=2 or wave.shape[1]!=4:raise ValueError('Expected four 44100 Hz microphone channels')
    calibration=json.loads(args.calibration.read_text())
    x=observation_inputs(xyz,wave,rate,calibration,args.geometry,args.projection,args.device,args.spatial)
    models=load_models(args.masks,args.selector,args.apex,args.device)
    result=propose_and_score(x,models,args.device)
    selected=None if args.rule in ('apex','audio_gate') else result['choices'][args.rule]
    rank_features=result['rank_features']
    baseline=result['scores']['hybrid']
    if args.gate:
        from .audio_gate import AudioSeedGate
        from .evaluate_audio_gate import candidate_gate_values
        ck=torch.load(args.gate,map_location='cpu',weights_only=False)
        gate=AudioSeedGate().to(args.device).eval();gate.load_state_dict(ck['model'])
        values=candidate_gate_values(x,models,gate,result,args.device)
        if args.gate2:
            ck2=torch.load(args.gate2,map_location='cpu',weights_only=False)
            gate2=AudioSeedGate().to(args.device).eval();gate2.load_state_dict(ck2['model'])
            values=values+args.gate2_weight*candidate_gate_values(x,models,gate2,result,args.device)
        baseline=baseline+args.gate_weight*values
        rank_features=np.column_stack([rank_features,values]).astype(np.float32)
        selected=int(baseline.argmax())
    if args.reranker:
        from .rerank import Reranker
        ck=torch.load(args.reranker,map_location='cpu',weights_only=False)
        head=Reranker(ck['config']['n_features']).to(args.device).eval();head.load_state_dict(ck['model'])
        with torch.inference_mode():
            scores=head(torch.tensor(rank_features,device=args.device),
                        torch.tensor(baseline,device=args.device))
        selected=int(scores.argmax())
    if args.rule=='apex':
        mask=(result['apex_probability']>=models[-1]['apex_threshold'])[x['inverse']]
        selected_origin='apex_direct'
    else:
        mask=result['masks'][selected][x['inverse']]
        selected_origin=result['origins'][selected]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    hashes={k+'_checkpoint_sha256':D.sha(getattr(args,k)) for k in ('masks','selector','apex')}
    if args.reranker:hashes['reranker_checkpoint_sha256']=D.sha(args.reranker)
    if args.gate:hashes['gate_checkpoint_sha256']=D.sha(args.gate)
    if args.gate2:hashes['gate2_checkpoint_sha256']=D.sha(args.gate2)
    if args.spatial:hashes['spatial_checkpoint_sha256']=D.sha(args.spatial)
    np.savez_compressed(args.output,method_version='orbit-consistent-v2',sound_mask=mask.astype(np.uint8),selected_origin=selected_origin,
        rule=('audio_gate_reranker' if args.gate else 'learned_reranker') if args.reranker else args.rule,
        gate_weight=args.gate_weight if args.gate else 0.,gate2_weight=args.gate2_weight,**hashes)
    print(json.dumps(dict(n_points=len(mask),foreground_points=int(mask.sum()),rule=args.rule,
        selected_origin=selected_origin,annotation_fields_opened=False)),flush=True)


if __name__=='__main__':main()
