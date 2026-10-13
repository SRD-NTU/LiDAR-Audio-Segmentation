"""Check observation-only raw feature regeneration against cached inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from apex import data as D
from .predict import observation_inputs
from .inference import load_models,propose_and_score


def main():
    p=argparse.ArgumentParser();p.add_argument('--masks',type=Path,required=True)
    p.add_argument('--selector',type=Path,required=True);p.add_argument('--apex',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--geometry',type=Path)
    p.add_argument('--projection',type=Path);p.add_argument('--spatial',type=Path)
    p.add_argument('--posterior',type=Path)
    p.add_argument('--gate',type=Path);p.add_argument('--gate2',type=Path)
    p.add_argument('--gate2-weight',type=float,default=0.);p.add_argument('--reranker',type=Path)
    p.add_argument('--gate-weight',type=float,default=1.)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if bool(args.gate2) != bool(args.gate2_weight) or (args.gate2 and not args.gate):
        raise ValueError('gate2 requires gate and a nonzero gate2-weight')
    if args.geometry:D.GEOMETRY=args.geometry
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    rs,post,emb,_=D.audio_bank('val');rel=rs[0]
    if args.posterior:post=np.load(args.posterior)
    cached=D.load_inputs(rel,post[0],emb[0])
    with np.load(D.DATA/rel/'lidar.npz') as z:xyz=z['xyz'].astype(np.float32)
    wave,rate=sf.read(D.DATA/rel/'audio.wav',dtype='float32')
    cal=json.loads((D.DATA/'metadata/microphone_array.json').read_text())
    raw=observation_inputs(xyz,wave,rate,cal,'utonia' if args.geometry else 'sonata',args.projection,spatial=args.spatial)
    models=load_models(args.masks,args.selector,args.apex)
    a=propose_and_score(cached,models);b=propose_and_score(raw,models)
    gate=None;gate2=None;ranker=None
    if args.gate:
        from .audio_gate import AudioSeedGate
        ck=torch.load(args.gate,map_location='cpu',weights_only=False)
        gate=AudioSeedGate().cuda().eval();gate.load_state_dict(ck['model'])
    if args.gate2:
        ck2=torch.load(args.gate2,map_location='cpu',weights_only=False)
        gate2=AudioSeedGate().cuda().eval();gate2.load_state_dict(ck2['model'])
    if args.reranker:
        from .rerank import Reranker
        ck=torch.load(args.reranker,map_location='cpu',weights_only=False)
        ranker=Reranker(ck['config']['n_features']).cuda().eval()
        ranker.load_state_dict(ck['model'])
    def selected_mask(x,pred):
        features=pred['rank_features'];scores=pred['scores']['hybrid']
        if gate is not None:
            from .evaluate_audio_gate import candidate_gate_values
            values=candidate_gate_values(x,models,gate,pred,'cuda')
            if gate2 is not None:
                values=values+args.gate2_weight*candidate_gate_values(x,models,gate2,pred,'cuda')
            features=np.column_stack([features,values]).astype(np.float32)
            scores=scores+args.gate_weight*values
        if ranker is not None:
            with torch.inference_mode():
                ranked=ranker(torch.as_tensor(features,device='cuda'),
                              torch.as_tensor(scores,device='cuda',dtype=torch.float32))
            choice=int(ranked.argmax())
        else:choice=int(scores.argmax())
        return pred['masks'][choice][x['inverse']]
    gate_mask_agreement=float(np.mean(selected_mask(cached,a)==selected_mask(raw,b))) if gate else None
    result=dict(sample=rel,annotation_fields_opened=False,inputs=['xyz','four-channel waveform','fixed calibration'],
        features_max_abs_difference=float(np.max(np.abs(cached['features'].astype(float)-raw['features'].astype(float)))),
        audio_max_abs_difference=float(np.max(np.abs(cached['audio']-raw['audio']))),
        acoustic_max_abs_difference=float(np.max(np.abs(cached['acoustic']-raw['acoustic']))),
        same_point_mapping=bool(np.array_equal(cached['inverse'],raw['inverse'])),
        seed_agreement=bool(a['seeds']==b['seeds']),
        mask_agreement={k:float(np.mean(a['masks'][a['choices'][k]][cached['inverse']]==b['masks'][b['choices'][k]][raw['inverse']])) for k in a['choices']},
        gate_mask_agreement=gate_mask_agreement,
        checkpoints={k:D.sha(getattr(args,k)) for k in ('masks','selector','apex')+
                     (('gate',) if args.gate else ())+(('gate2',) if args.gate2 else ())+
                     (('reranker',) if args.reranker else ())})
    # Small encoder/CLAP floating-point differences can reorder nearly tied
    # seeds. Deployment equivalence concerns the returned original-point mask;
    # retain seed_agreement separately as a stricter diagnostic.
    result['pass']=bool(result['same_point_mapping'] and
                        min(result['mask_agreement'].values())==1 and
                        (gate_mask_agreement is None or gate_mask_agreement==1) and
                        result['features_max_abs_difference']<=.01 and
                        result['audio_max_abs_difference']<=.001 and
                        result['acoustic_max_abs_difference']<=1e-5)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if not result['pass']:raise AssertionError('Cached and raw inference differ')


if __name__=='__main__':main()
