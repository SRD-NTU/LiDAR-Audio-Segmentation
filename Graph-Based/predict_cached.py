"""CPU deployment smoke test using frozen, observation-only cached encoders.

The mask/semantic/instance targets in graph files are never loaded. For new raw
XYZ and WAV inputs, use predict.py inside a CUDA allocation instead.
"""
import argparse
import hashlib
import json

import numpy as np
import torch
from scipy.special import softmax,expit,logit
from prepare import ROOT,DATASET
from graph_audio.model import AcousticGraphWorld
from train_graph import OBSERVATIONS
from train_trees import inputs,refine


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--model',help='Optional completed full-data model name; default is the validation-selected best')
    args = parser.parse_args()
    torch.set_num_threads(2)
    board=json.loads((ROOT/'reports/leaderboard.json').read_text())
    selected=board['best']
    if args.model:
        matches=[r for r in board['full_runs'] if r['name']==args.model]
        if len(matches)!=1:
            raise ValueError('Requested model must have a completed full-training/full-validation result')
        selected=matches[0]
    if selected is None:
        raise ValueError('No completed full-validation model')
    path = ROOT/selected['checkpoint']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=selected['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after selection')
    with np.load(ROOT/'cache/graphs'/(args.sample+'.npz')) as graph:
        observation = {k:graph[k].copy() for k in (*OBSERVATIONS,'counts') if k in graph.files}
        inverse = graph['inverse'].copy()
        xyz_digest = str(graph['xyz_sha256'])
    point_base = None
    if selected.get('teacher_checkpoint_sha256'):
        teacher_path = ROOT/'cache/teacher'/(args.sample+'.npz')
        if teacher_path.exists():
            with np.load(teacher_path) as teacher:
                if selected.get('teacher_checkpoint_sha256') and str(teacher['teacher_sha256'])!=selected['teacher_checkpoint_sha256']:
                    raise ValueError('Teacher checkpoint provenance mismatch')
                observation['teacher_geometry'] = teacher['geometry'].astype(np.float32)
                observation['teacher_probability'] = teacher['probability'].copy()
                point_base = teacher['point_probability'].copy()
    lut = ROOT/'cache/phase_lut'/(args.sample+'.npy')
    if 'delay_tables' not in observation and lut.exists():
        observation['delay_tables'] = np.load(lut).astype(np.float32)
    if selected.get('audio_head_sha256'):
        audio_root = ROOT/('cache/audio_category_spectral' if selected.get('audio_head_variant')=='spectral' else 'cache/audio_category')
        with np.load(audio_root/(args.sample+'.npz')) as audio:
            if str(audio['model_sha256'])!=selected['audio_head_sha256']:
                raise ValueError('Audio head cache changed')
            observation['learned_audio_logits'] = audio['logits'].copy()
    if selected['type']=='calibrated_world':
        node=np.clip(observation['teacher_probability'],1e-5,1-1e-5)
        smoothed=np.clip(refine(node,observation,selected['graph_strength']),1e-5,1-1e-5)
        from graph_audio.spatial_audio import selected_scene_field
        field=selected_scene_field(observation,selected,ROOT)
        field=(field-field.mean())/max(float(field.std()),.1)
        delta=logit(smoothed)-logit(node)+selected['phase_alpha']*field
        probability=expit(logit(np.clip(point_base,1e-5,1-1e-5))+delta[inverse])
    elif selected['type'] in ('trees','teacher_trees'):
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(path))
        if selected['type']=='trees':
            probability = refine(model.predict(inputs(observation,selected['physics']),num_threads=2),
                                 observation,selected['graph_strength'])[inverse]
        else:
            from train_teacher_trees import teacher_inputs
            values = teacher_inputs(observation,dict(geometry=observation['teacher_geometry'],
                probability=observation['teacher_probability']),learned_audio_logits=observation.get('learned_audio_logits'))
            node = model.predict(values,num_threads=2)
            delta = logit(np.clip(node,1e-5,1-1e-5))-logit(np.clip(observation['teacher_probability'],1e-5,1-1e-5))
            probability = expit(logit(np.clip(point_base,1e-5,1-1e-5))+selected['residual_strength']*delta[inverse])
    else:
        checkpoint = torch.load(path,map_location='cpu',weights_only=False)
        model = AcousticGraphWorld(use_graph=not checkpoint['config']['no_graph']).eval()
        if selected['type']=='semantic':
            for prefix in ('geometry','semantic'):
                getattr(model,prefix).load_state_dict({k[len(prefix)+1:]:v for k,v in checkpoint['model'].items()
                                                      if k.startswith(prefix+'.')})
            posterior = model.semantic(model.geometry(torch.from_numpy(observation['features'].astype(np.float32)))).softmax(-1).numpy()
            prior = softmax(observation['learned_audio_logits']) if 'learned_audio_logits' in observation else softmax(observation['text_logits']/2)
            match = (posterior[:,1:]*prior[None]).sum(-1)
            spatial = observation['acoustic'].astype(np.float32).mean(1)
            spatial = (spatial-spatial.mean())/max(float(spatial.std()),.1)
            probability = np.clip(match*np.exp(np.clip(selected['alpha']*spatial,-5,5)),0,1)
            probability = refine(probability,observation,selected['graph_strength'])[inverse]
        else:
            model.load_state_dict(checkpoint['model'])
            tensors = {k:torch.as_tensor(v,dtype=torch.long if k=='neighbors' else torch.float32)
                       for k,v in observation.items() if k in OBSERVATIONS}
            output = model(**tensors)
            if point_base is None:
                probability = output['logits'].sigmoid().numpy()[inverse]
            else:
                probability = expit(logit(np.clip(point_base,1e-5,1-1e-5))+output['logit_delta'].numpy()[inverse])
    with np.load(DATASET/args.sample/'lidar.npz') as lidar:
        xyz = lidar['xyz'].copy()  # No target fields are accessed.
    if hashlib.sha256(xyz.astype(np.float32).tobytes()).hexdigest()!=xyz_digest:
        raise ValueError('Observed point cloud changed')
    if probability.shape!=(len(xyz),) or not np.isfinite(probability).all():
        raise ValueError('Invalid point predictions')
    from pathlib import Path
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,xyz=xyz,probability=probability.astype(np.float32),
        mask=(probability>=selected['threshold']).astype(np.uint8),threshold=selected['threshold'],
        model=selected['name'],checkpoint_sha256=selected['checkpoint_sha256'],
        point_order='Original LiDAR XYZ order',encoder_mode='Frozen observation cache')
    print(f'Saved {len(xyz)} finite point predictions: {output}',flush=True)


if __name__=='__main__':
    main()
