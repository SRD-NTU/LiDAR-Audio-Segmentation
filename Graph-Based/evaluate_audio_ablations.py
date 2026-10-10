"""Fixed-threshold matched/shuffled/missing-audio neural-world diagnostics.

Never reselect thresholds for an intervention. No teacher is supported: keeping
its matched-audio probabilities would invalidate a claimed shuffled-audio test.
"""
import argparse
import hashlib
import json
import io
from pathlib import Path

import numpy as np
import torch

from graph_audio.metrics import score, summarize
from graph_audio.model import AcousticGraphWorld
from train_graph import load
from prepare import ROOT


AUDIO_KEYS = ('embedding', 'text_logits', 'audio_context', 'delay_tables', 'learned_audio_logits')


def missing_audio(observations):
    result = dict(observations)
    for key in (*AUDIO_KEYS, 'acoustic'):
        if key in result:
            result[key] = torch.zeros_like(result[key])
    return result


def swapped_audio(observations, donor):
    result = dict(observations)
    for key in AUDIO_KEYS:
        if key in donor:
            result[key] = donor[key]
        else:
            result.pop(key, None)
    # Forward recomputes the phase field on the recipient's predicted geometry
    # using the donor's audio-only lookup. Never copy a donor-shaped node field.
    if 'delay_tables' not in result:
        raise ValueError('Swapping spatial audio requires an audio-only phase lookup')
    result['acoustic'] = torch.zeros_like(observations['acoustic'])
    return result


def donor_audio(name, device):
    with np.load(ROOT/'cache/graphs'/(name+'.npz')) as graph:
        data = {k:torch.as_tensor(graph[k].copy(), device=device, dtype=torch.float32)
                for k in AUDIO_KEYS if k in graph.files}
    data['delay_tables'] = torch.as_tensor(np.load(ROOT/'cache/phase_lut'/(name+'.npy')).astype(np.float32), device=device)
    with np.load(ROOT/'cache/audio_category'/(name+'.npz')) as audio:
        data['learned_audio_logits'] = torch.as_tensor(audio['logits'].copy(), device=device)
    return data


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=128, help='Zero evaluates the full validation split')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint_bytes=args.checkpoint.read_bytes()
    checkpoint_digest=hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=args.device, weights_only=False)
    config = checkpoint['config']
    if not config.get('without_teacher', False):
        raise ValueError('Use an independent world: a matched-audio teacher would confound this test')
    model = AcousticGraphWorld(use_graph=not config['no_graph']).to(args.device).eval()
    model.load_state_dict(checkpoint['model'])
    names = sorted(json.loads((args.checkpoint.parent/'val_manifest.json').read_text()))
    rng = np.random.default_rng(20261007)
    # Deterministic random cycle: no labels determine donor assignment.
    order = rng.permutation(len(names))
    donor_index = {int(order[i]):int(order[(i+1)%len(order)]) for i in range(len(order))}
    selected = sorted(rng.choice(len(names), min(args.limit, len(names)), replace=False)) if args.limit else list(range(len(names)))
    threshold = float(checkpoint['validation']['threshold'])
    rows = {k:[] for k in ('matched', 'shuffled', 'missing')}
    for i,index in enumerate(selected, 1):
        name, donor = names[index], names[donor_index[index]]
        observations, targets = load(name, args.device, use_teacher=False)
        world=model.construct_world(*(observations[k] for k in ('features','centers','neighbors','edge_valid')))
        interventions = dict(matched=observations,
                             shuffled=swapped_audio(observations, donor_audio(donor,args.device)),
                             missing=missing_audio(observations))
        for condition,inputs in interventions.items():
            output = model(**inputs,world_state=world)
            probability = output['logits'].sigmoid().cpu().numpy()
            if not np.isfinite(probability).all():
                raise FloatingPointError(f'Nonfinite {condition} prediction: {name}')
            row = score(probability, targets['counts'].cpu().numpy(), targets['positives'].cpu().numpy(), threshold)
            rows[condition].append(dict(sample=name, donor=donor if condition=='shuffled' else None, **row))
        if i%16 == 0 or i==len(selected):
            print(f'Audio intervention diagnostics {i}/{len(selected)}', flush=True)
    result = dict(split='val', samples=len(selected), full_split=len(selected)==len(names),
                  diagnostic=True, threshold_fixed_from_checkpoint=threshold,
                  checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_digest,
                  note='Shuffled audio is paired with the original target only to test prompt sensitivity; this is not a new segmentation leaderboard entry.',
                  conditions={key:summarize(value) for key,value in rows.items()},
                  samples_detail=rows)
    destination = args.output or ROOT/'reports'/f'{args.checkpoint.parent.name}-audio-interventions.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['conditions'], indent=2), flush=True)


if __name__ == '__main__':
    main()
