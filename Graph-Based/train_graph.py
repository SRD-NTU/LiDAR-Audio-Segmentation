"""Joint graph/audio training with validation selection and point-weighted losses."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from graph_audio.model import AcousticGraphWorld, world_loss
from graph_audio.metrics import THRESHOLDS, score, summarize
from prepare import DATASET, ROOT
from train_trees import manifests, wait_for_cache

# Capture provenance before potentially long cache/GPU waits. Hashing current
# files after the wait could describe later edits rather than the loaded code.
SOURCE_TEXT = {str(p.relative_to(ROOT)):p.read_text() for p in
               [ROOT/'train_graph.py',*sorted((ROOT/'graph_audio').glob('*.py'))]}


OBSERVATIONS = ('features', 'centers', 'neighbors', 'edge_valid', 'acoustic',
                'embedding', 'text_logits', 'audio_context', 'delay_tables', 'teacher_geometry', 'teacher_probability',
                'learned_audio_logits')


def load(name, device, use_teacher=True):
    with np.load(ROOT / 'cache/graphs' / (name + '.npz')) as a:
        data = {key: torch.as_tensor(a[key].copy(), device=device,
                 dtype=torch.long if key in ('neighbors', 'semantic', 'instance') else torch.float32)
                for key in (*OBSERVATIONS, 'counts', 'positives', 'semantic', 'instance') if key in a.files}
    lut = ROOT / 'cache/phase_lut' / (name + '.npy')
    if 'delay_tables' not in data and lut.exists():
        data['delay_tables'] = torch.as_tensor(np.load(lut).astype(np.float32), device=device)
    teacher = ROOT / 'cache/teacher' / (name + '.npz')
    if use_teacher and teacher.exists():
        with np.load(teacher) as a:
            if not np.isfinite(a['geometry']).all() or not np.isfinite(a['probability']).all():
                raise FloatingPointError(f'Nonfinite teacher cache: {name}')
            data['teacher_geometry'] = torch.as_tensor(a['geometry'].astype(np.float32), device=device)
            data['teacher_probability'] = torch.as_tensor(a['probability'], device=device)
    audio = ROOT / 'cache/audio_category' / (name + '.npz')
    if (ROOT / 'cache/audio_category/ready.json').exists():
        with np.load(audio) as a:
            data['learned_audio_logits'] = torch.as_tensor(a['logits'].copy(),device=device)
    return {key: data[key] for key in OBSERVATIONS if key in data}, data


def prefetched_cases(names, use_teacher=True):
    """One CPU-only case ahead; never invoke CUDA from an I/O worker."""
    iterator=iter(names)
    name=next(iterator,None)
    if name is None:
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(load,name,'cpu',use_teacher)
        while name is not None:
            observations,targets=future.result()
            next_name=next(iterator,None)
            if next_name is not None:
                future=executor.submit(load,next_name,'cpu',use_teacher)
            yield name,observations,targets
            name=next_name


def device_tensors(values, device):
    return {key:value.to(device) for key,value in values.items()}


def training_subgraph(observations, targets, limit, rng):
    """Keep positive training anchors plus their observation-only neighbors.

    This is a loss sampler, not an inference proposal rule. Validation/prediction
    always use every observed node, without labels defining the graph.
    """
    n = len(targets['counts'])
    if not limit or n <= limit:
        return observations, targets
    device = observations['features'].device
    positive = np.flatnonzero(targets['positives'].cpu().numpy() > 0)
    positive = rng.choice(positive, min(128, len(positive)), replace=False)
    anchors = np.unique(np.concatenate((positive, rng.choice(n, max(1, limit // 16), replace=False))))
    neighbors = observations['neighbors'].cpu().numpy()
    candidates = np.unique(np.concatenate((anchors, neighbors[anchors].ravel())))
    candidates = np.setdiff1d(candidates, positive)
    rest = rng.choice(candidates, min(limit - len(positive), len(candidates)), replace=False)
    ids = np.unique(np.concatenate((positive, rest)))
    ids = torch.as_tensor(ids, device=device, dtype=torch.long)
    mapping = torch.full((n,), -1, device=device, dtype=torch.long)
    mapping[ids] = torch.arange(len(ids), device=device)
    remapped = mapping[observations['neighbors'][ids]]
    valid = observations['edge_valid'][ids] * (remapped >= 0)
    remapped = torch.where(remapped >= 0, remapped, torch.arange(len(ids), device=device)[:, None])
    subset = {key: value for key, value in observations.items()}
    for key in ('features', 'centers', 'acoustic', 'teacher_geometry', 'teacher_probability'):
        if key in observations:
            subset[key] = observations[key][ids]
    subset.update(neighbors=remapped, edge_valid=valid)
    # Full-graph ground-truth object centroids remain training targets only.
    with torch.no_grad():
        _, inverse = torch.unique(targets['instance'], return_inverse=True)
        groups = int(inverse.max()) + 1
        mass = targets['counts'].float()
        totals = mass.new_zeros(groups).index_add(0, inverse, mass)
        centers = observations['centers'].new_zeros(groups, 3).index_add(
            0, inverse, observations['centers'] * mass[:, None]) / totals[:, None].clamp_min(1)
    selected_targets = {key: targets[key][ids] for key in ('counts', 'positives', 'semantic', 'instance')}
    selected_targets.update(embedding=targets['embedding'], centers=subset['centers'], instance_centers=centers[inverse[ids]])
    return subset, selected_targets


def corrupt_teacher_prior(observations, scale, rng):
    """Training-only geometry-correlated perturbations prevent blind copying.

    The existing teacher was trained on this training split, so its in-sample
    probabilities are much easier than held-out ones. No labels determine these
    perturbations. Validation and deployment retain the actual teacher output.
    """
    if 'teacher_probability' not in observations or scale<=0:
        return observations
    result = dict(observations)
    base = observations['teacher_probability'].clamp(1e-5,1-1e-5)
    centers = observations['centers']
    direction = centers.new_tensor(rng.normal(size=3))
    phase = float(rng.uniform(-np.pi,np.pi))
    coherent = torch.sin(centers @ direction + phase)
    bias = float(rng.normal())
    temperature = float(rng.uniform(.65,1.5))
    perturbed = torch.logit(base)/temperature + scale*(bias+coherent)
    if rng.random()<.15:
        perturbed = centers.new_full((len(base),),float(np.log(.05/.95)))+scale*.5*coherent
    result['teacher_probability'] = perturbed.sigmoid()
    return result


@torch.inference_mode()
def evaluate(model, names, device, threshold=None):
    model.eval()
    thresholds = [threshold] if threshold is not None else THRESHOLDS
    rows = {float(t): [] for t in thresholds}
    for i, name in enumerate(names, 1):
        observations, targets = load(name, device,use_teacher=getattr(model,'use_teacher',True))
        output = model(**observations)
        probability = output['logits'].float().sigmoid().cpu().numpy()
        if not np.isfinite(probability).all():
            raise FloatingPointError(f'Nonfinite validation prediction: {name}')
        counts, positive = targets['counts'].cpu().numpy(), targets['positives'].cpu().numpy()
        teacher = ROOT / 'cache/teacher' / (name + '.npz')
        if 'teacher_probability' in observations:
            # Apply one learned graph residual per voxel to the original fine
            # point probabilities, preserving the teacher's boundary detail.
            with np.load(teacher) as a, np.load(ROOT / 'cache/graphs' / (name + '.npz')) as graph:
                base = torch.from_numpy(a['point_probability']).clamp(1e-5, 1-1e-5)
                delta = torch.from_numpy(output['logit_delta'].cpu().numpy()[graph['inverse']])
                probability = (torch.logit(base) + delta).sigmoid().numpy()
            with np.load(DATASET / name / 'lidar.npz') as a:
                positive = a['sound_mask'].astype(np.int32)
            counts = np.ones_like(positive)
        for t in thresholds:
            rows[float(t)].append(dict(sample=name, scene=name.split('/')[1], **score(probability, counts, positive, t)))
        if i % 128 == 0 or i == len(names):
            print(f'Graph validation {i}/{len(names)}', flush=True)
    selected = max(rows, key=lambda t: np.mean([r['iou'] for r in rows[t]]))
    return dict(threshold=float(selected), **summarize(rows[selected])), rows[selected]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='graph-world')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--train-limit', type=int)
    p.add_argument('--val-limit', type=int)
    p.add_argument('--available', action='store_true')
    p.add_argument('--teacher-only', action='store_true', help='Subset diagnostic: only cases with cached teacher observations')
    p.add_argument('--no-graph', action='store_true')
    p.add_argument('--without-teacher',action='store_true',help='Independent world: no inherited world context or probabilities')
    p.add_argument('--wait-cache', action='store_true')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--resume', type=Path)
    p.add_argument('--max-train-nodes', type=int, default=8192)
    p.add_argument('--teacher-prior-jitter',type=float,default=2.,help='Train-only correlated logit corruption; zero is the ablation')
    p.add_argument('--audio-prior-dropout',type=float,default=.35,help='Train-only mixing with zero-shot audio priors')
    p.add_argument('--no-spatial-warm-start',action='store_true',help='Ablation: initialize phase reliability uniformly')
    p.add_argument('--no-prefetch',action='store_true',help='Disable one-case-ahead CPU loading for CUDA training')
    args = p.parse_args()
    torch.set_num_threads(3)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA training requires a Slurm GPU allocation')
    resumed = torch.load(args.resume,map_location=device,weights_only=False) if args.resume else None
    if resumed is not None and resumed['epoch']+1>=args.epochs:
        raise ValueError('--epochs is the total target epoch count; resume already reached it')
    if resumed is not None:
        for key in ('no_graph','without_teacher'):
            if bool(getattr(args,key)) != bool(resumed['config'].get(key,False)):
                raise ValueError(f'Resume must preserve {key}; changing the inference method requires a separate experiment')
    train = manifests('train', None if args.teacher_only else args.train_limit, args.available)
    val = manifests('val', None if args.teacher_only else args.val_limit, args.available)
    if args.teacher_only:
        train = [n for n in train if (ROOT / 'cache/teacher' / (n + '.npz')).exists()]
        val = [n for n in val if (ROOT / 'cache/teacher' / (n + '.npz')).exists()]
        train = train[:args.train_limit] if args.train_limit else train
        val = val[:args.val_limit] if args.val_limit else val
    if not train or not val:
        raise ValueError('Both training and validation require available observations')
    if args.wait_cache:
        wait_for_cache(train + val)
        while not (ROOT / 'cache/graphs/center_lut.ready').exists():
            print('Waiting for center acoustic lookup tables', flush=True)
            time.sleep(20)
        if not args.without_teacher and (ROOT / 'cache/teacher/enabled').exists() and not (ROOT / 'cache/teacher/teacher.ready').exists():
            if device.type == 'cuda':
                import subprocess, sys
                subprocess.run([sys.executable, str(ROOT / 'prepare_teacher.py'), '--device', 'cuda'], check=True)
            else:
                while not (ROOT / 'cache/teacher/teacher.ready').exists():
                    print('Waiting for pretrained world context cache', flush=True)
                    time.sleep(20)
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), dataset=str(DATASET), training_samples=len(train),
                  validation_samples=len(val), foundation_frozen=True,
                  encoder_overlap='Sonata pretraining HM3D overlap unresolved')
    config['resume'] = str(args.resume) if args.resume else None
    if resumed is not None:
        config['resume_checkpoint_sha256']=hashlib.sha256(args.resume.read_bytes()).hexdigest()
        config['resume_original_config']=resumed['config']
    teacher_ready = ROOT / 'cache/teacher/teacher.ready'
    if teacher_ready.exists() and not args.without_teacher:
        config['teacher_checkpoint_sha256'] = json.loads(teacher_ready.read_text())['checkpoint_sha256']
    elif args.teacher_only and not args.without_teacher:
        with np.load(ROOT / 'cache/teacher' / (train[0] + '.npz')) as teacher:
            config['teacher_checkpoint_sha256'] = str(teacher['teacher_sha256'])
    audio_ready = ROOT / 'cache/audio_category/ready.json'
    if audio_ready.exists():
        config['audio_head_sha256'] = json.loads(audio_ready.read_text())['model_sha256']
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    model = AcousticGraphWorld(use_graph=not args.no_graph,audio_prior_dropout=args.audio_prior_dropout).to(device)
    model.use_teacher=not args.without_teacher
    warm = ROOT / 'runs/semantic-head/geometry.pt'
    if warm.exists() and not args.resume:
        initial = torch.load(warm, map_location=device, weights_only=False)
        model.geometry.load_state_dict(initial['geometry'])
        model.semantic.load_state_dict(initial['semantic'])
        nn = torch.nn
        nn.init.zeros_(model.context.weight); nn.init.zeros_(model.context.bias)
        nn.init.zeros_(model.mask[-1].weight); nn.init.zeros_(model.mask[-1].bias)
        for block in model.graph:
            nn.init.zeros_(block.update[-1].weight); nn.init.zeros_(block.update[-1].bias)
        model.semantic_scale.data.fill_(1.)
        model.render_scale.data.zero_()
        config['geometry_initialization'] = str(warm)
        (run / 'config.json').write_text(json.dumps(config, indent=2))
        print(f'Loaded full-data geometry warm start: {warm}', flush=True)
    spatial_report=ROOT/'reports/spatial-audio-reliability.json'
    if spatial_report.exists() and not args.resume and not args.no_spatial_warm_start:
        from graph_audio.spatial_audio import SpatialReliability,initialize_world_reliability
        spatial_path=ROOT/'runs/spatial-audio-reliability/best.pt'
        spatial_digest=hashlib.sha256(spatial_path.read_bytes()).hexdigest()
        if spatial_digest!=json.loads(spatial_report.read_text())['checkpoint_sha256']:
            raise ValueError('Completed spatial head changed after validation')
        spatial=SpatialReliability().to(device)
        spatial.load_state_dict(torch.load(spatial_path,map_location=device,weights_only=False)['model'])
        initialize_world_reliability(model,spatial)
        config['spatial_reliability_initialization_sha256']=spatial_digest
        (run/'config.json').write_text(json.dumps(config,indent=2))
        print('Initialized world phase reliability from supervised spatial audio',flush=True)
    (run / 'train_manifest.json').write_text(json.dumps(train))
    (run / 'val_manifest.json').write_text(json.dumps(val))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    start_epoch = 0
    best_iou = -1
    if args.resume:
        checkpoint = resumed
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['validation']['mean_iou']
        previous_report=run/'validation.json'
        if previous_report.exists():
            best_iou=max(best_iou,json.loads(previous_report.read_text())['metrics']['mean_iou'])
        previous_sources=run/'source_snapshot.json'
        if previous_sources.exists():
            history=run/'source_history'
            history.mkdir(exist_ok=True)
            (history/f'before-resume-{start_epoch}-{time.time_ns()}.json').write_text(previous_sources.read_text())
    source_hashes = {name:hashlib.sha256(source.encode()).hexdigest() for name,source in SOURCE_TEXT.items()}
    (run / 'source_hashes.json').write_text(json.dumps(source_hashes, indent=2))
    (run / 'source_snapshot.json').write_text(json.dumps(SOURCE_TEXT))
    for epoch in range(start_epoch, args.epochs):
        model.train()
        random.shuffle(train)
        totals = {}
        start = time.monotonic()
        prefetch=device.type=='cuda' and not args.no_prefetch
        cases=prefetched_cases(train,not args.without_teacher) if prefetch else (
            (name,*load(name,device,use_teacher=not args.without_teacher)) for name in train)
        for i,(name,observations,targets) in enumerate(cases,1):
            observations, targets = training_subgraph(observations, targets, args.max_train_nodes, rng)
            if prefetch:
                observations=device_tensors(observations,device)
                targets=device_tensors(targets,device)
            observations = corrupt_teacher_prior(observations,args.teacher_prior_jitter,rng)
            optimizer.zero_grad(set_to_none=True)
            output = model(**observations)
            losses = world_loss(output, targets, observations['neighbors'], observations['edge_valid'])
            if not torch.isfinite(losses['total']):
                raise FloatingPointError(f'Nonfinite loss: {name}')
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.) + float(value.detach())
            if i % 128 == 0 or i == len(train):
                print(f'{args.run} epoch {epoch}: {i}/{len(train)}; loss {totals["total"]/i:.4f}; '
                      f'{time.monotonic()-start:.1f}s', flush=True)
        validation, rows = evaluate(model, val, device)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                     validation=validation, config=config,source_hashes=source_hashes)
        temporary = run / 'last.tmp.pt'
        torch.save(state, temporary); os.replace(temporary, run / 'last.pt')
        if validation['mean_iou'] > best_iou:
            best_iou = validation['mean_iou']
            torch.save(state, run / 'best.tmp.pt'); os.replace(run/'best.tmp.pt',run/'best.pt')
            (run / 'validation.json').write_text(json.dumps(dict(split='val', full_split=len(val) == 768,
                  metrics=validation, samples_detail=rows), indent=2))
        result = dict(epoch=epoch, training={k: v / len(train) for k, v in totals.items()},
                      validation=validation, best_iou=best_iou, elapsed=time.monotonic()-start)
        with (run / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)
        from summarize import main as refresh_reports
        refresh_reports()
    print(f'Finished {run}; best validation mean IoU {best_iou:.5f}', flush=True)
    from summarize import main as refresh_reports
    refresh_reports()


if __name__ == '__main__':
    main()
