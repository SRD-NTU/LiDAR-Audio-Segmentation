"""Resumable physical scene generation with immutable sources and split plans."""
from __future__ import annotations
from pathlib import Path
import json
import shutil
import time
import traceback
import numpy as np
import soundfile as sf
from audiolidar.utils.io import read_json,write_json,sha256
from audiolidar.habitat.scenes import discover,annotations,SemanticMesh
from audiolidar.habitat.backend import Backend
from audiolidar.lidar.rays import scan
from audiolidar.geometry.frames import quaternion_xyzw,relative
from audiolidar.audio.spatialize import dry_wave,convolve
from audiolidar.benchmark.protocol import validate_audio,validate_reviewed_audio,validate_source_plan,split_plan,difficulty
from audiolidar.benchmark.validate import validate_sample,validate_dataset
from audiolidar.visualization.sample import preview


def rejection(root: Path, scene: str, source, listener, reason: str):
    # Per-scene logs allow independent worker processes without interleaved writes.
    p=root/'logs'/f'rejected-{scene}.jsonl';p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('a') as f:f.write(json.dumps(dict(scene=scene,source_instance=source,
        listener_pose=listener,reason=reason),allow_nan=False)+'\n')


def scene_worker(args):
    scene,split,cfg,categories,clips,resume=args
    root=Path(cfg['output']); seed=(cfg['seed']+int(sha256(Path(scene['labels']))[:8],16))%(2**32)
    rng=np.random.default_rng(seed)
    mesh=SemanticMesh(scene,categories,np.asarray(cfg['mesh_to_world_rotation']))
    backend=Backend(scene,dict(cfg,seed=seed),mesh=mesh)
    try:
        scene_report=backend.summary()
        # Confirm every actual annotation ID/category against Habitat's own ontology.
        exposed={int(o['semantic_id']):o['category'] for o in scene_report['objects']}
        if not set(mesh.objects)<=set(exposed):raise RuntimeError('Missing Habitat instance identities')
        for ident,o in mesh.objects.items():
            if ident in exposed and exposed[ident]!=o['category']:
                raise RuntimeError('Habitat/triangle category mismatch')
        capable=set(cfg['selected_categories'])
        candidates=[o for o in mesh.objects.values() if o['category'] in capable
                    and any(c['semantic_category']==o['category'] and c['split']==split for c in clips)]
        rng.shuffle(candidates)
        planned={row['instance_id']:row for row in cfg.get('planned_sources',[])
                 if row['scene_id']==scene['scene_id'] and row['split']==split}
        source_limit=len(planned) if cfg.get('source_plan') else cfg['sources_per_scene']
        if cfg.get('source_plan'):
            if not planned:raise ValueError('No explicit sources for this scene')
            candidates=[mesh.objects[ident] for ident in sorted(planned)]
            for obj in candidates:
                if obj['category']!=planned[obj['instance_id']]['category']:
                    raise ValueError('Planned instance/category disagrees with scene')
        accepted_sources=0; total=0; source_results=[]
        for obj in candidates:
            if accepted_sources>=source_limit:break
            ident=obj['instance_id'];center=np.asarray(obj['center'])
            source_root=root/split/scene['scene_id']/str(ident)
            rejected_root=root/'rejected_sources'/split/scene['scene_id']/str(ident)
            if rejected_root.exists():continue
            source_record=source_root/'source.json'
            source=read_json(source_record) if source_record.exists() else None
            if ident in planned:
                emitter=np.asarray(planned[ident]['position_world'])
                if not mesh.clearance(emitter,cfg['source_surface_offset_m']*.45):
                    raise ValueError('Planned emitter fails surface clearance')
                if source is not None and not np.allclose(source['position_world'],emitter):
                    raise ValueError('Existing emitter disagrees with explicit plan')
                source=dict(obj,scene_id=scene['scene_id'],position_world=emitter.tolist(),
                            placement='visible-instance-surface-offset',surface_offset_m=cfg['source_surface_offset_m'])
                write_json(source_record,source)
            completed=sorted(source_root.glob('*/COMPLETE.json'))
            if resume and len(completed)==cfg['listeners_per_source']:
                for marker in completed:validate_sample(marker.parent)
                accepted_sources+=1;total+=len(completed)
                source_results.append(dict(instance_id=ident,samples=len(completed),resumed=True))
                continue
            achieved=0; bins=set(); used_positions=[]
            # Advance the navmesh RNG identically on resume by persisting candidate poses.
            pose_path=root/'metadata/poses'/f"{scene['scene_id']}-{ident}.json"
            if ident in planned:
                poses=planned[ident]['listeners']
                if pose_path.exists() and read_json(pose_path)!=poses:raise ValueError('Explicit listener plan changed')
                write_json(pose_path,poses)
            elif pose_path.exists():poses=read_json(pose_path)
            else:
                poses=[]
                for _ in range(cfg['max_listener_attempts']):
                    floor=np.asarray(backend.sim.pathfinder.get_random_navigable_point())
                    if not np.isfinite(floor).all():continue
                    pos=floor+np.array([0.,cfg['lidar']['sensor_height_m'],0.])
                    d=np.linalg.norm(pos-center)
                    if cfg['listener_distance_m'][0]<=d<=cfg['listener_distance_m'][1]:
                        poses.append({'floor':floor.tolist(),'position':pos.tolist(),'yaw':float(rng.uniform(-np.pi,np.pi))})
                # Interleave distance/angle strata instead of exhausting only near positions.
                groups={}
                for pose in poses:
                    d=np.linalg.norm(np.asarray(pose['position'])-center)
                    delta=np.asarray(pose['position'])-center
                    key=(int(np.digitize(d,[2.,4.])),int((np.arctan2(delta[2],delta[0])+np.pi)/(np.pi/2)))
                    groups.setdefault(key,[]).append(pose)
                poses=[]
                while any(groups.values()):
                    for group in groups.values():
                        if group:poses.append(group.pop())
                write_json(pose_path,poses)
            for attempt,pose in enumerate(poses):
                if achieved>=cfg['listeners_per_source']:break
                pos=np.asarray(pose['position']);yaw=pose['yaw']
                sample=source_root/f'{attempt:05d}'
                if (sample/'COMPLETE.json').exists():
                    if not resume:raise FileExistsError(f'{sample}: use --resume')
                    validate_sample(sample);achieved+=1;total+=1;used_positions.append(pos);continue
                if any(np.linalg.norm(pos-old)<cfg['min_listener_separation_m'] for old in used_positions):continue
                if not backend.sim.pathfinder.is_navigable(np.asarray(pose['floor'])) or not mesh.clearance(pos,.15):
                    rejection(root,scene['scene_id'],ident,pose,'listener off navmesh or collision');continue
                points=scan(mesh,pos,yaw,cfg['lidar'])
                mask=(points['instance_id']==ident).astype('uint8')
                if mask.sum()<cfg['min_source_points']:
                    rejection(root,scene['scene_id'],ident,pose,'too few source LiDAR points: '+str(mask.sum()));continue
                if source is None:
                    try:emitter=mesh.source_position(ident,pos,cfg['source_surface_offset_m'])
                    except ValueError as error:
                        rejection(root,scene['scene_id'],ident,pose,str(error));continue
                    source=dict(obj,scene_id=scene['scene_id'],position_world=emitter.tolist(),
                                placement='visible-instance-surface-offset',surface_offset_m=cfg['source_surface_offset_m'])
                    write_json(source_record,source)
                emitter=np.asarray(source['position_world'])
                candidates_audio=[c for c in clips if c['semantic_category']==obj['category'] and c['split']==split]
                clip=candidates_audio[achieved%len(candidates_audio)]
                try:
                    rir=backend.render(emitter,pos,yaw)
                    dry=dry_wave(clip['file_path'],cfg['audio']['sample_rate'],cfg['audio']['duration_sec'])
                    wave,gain=convolve(dry,rir,len(dry))
                except Exception as error:
                    rejection(root,scene['scene_id'],ident,pose,'audio simulation failed: '+repr(error))
                    raise  # System/audio errors are never suppressed as ordinary sampling rejection.
                points['sound_mask']=mask
                sample.mkdir(exist_ok=True,parents=True)
                if any(sample.iterdir()):
                    # Incomplete sample only; validated completions above are never overwritten.
                    if not resume:raise FileExistsError(sample)
                    shutil.rmtree(sample);sample.mkdir()
                np.savez_compressed(sample/'lidar.npz',**points)
                np.save(sample/'rir.npy',rir)
                sf.write(sample/'audio.wav',wave,cfg['audio']['sample_rate'],subtype='FLOAT')
                soundspaces=dict(cfg['soundspaces']['acoustics'],enableMaterials=cfg['soundspaces']['enableMaterials'],
                                 materials_json=cfg['soundspaces']['materials_json'],
                                 materials_sha256=sha256(Path(cfg['config_path']).parent.parent/cfg['soundspaces']['materials_json']),
                                 seed=None,seed_note='RLR binding exposes no acoustic RNG seed; RIR is retained',
                                 acoustic_material_geometry='original triangles, centroid texture categories; LiDAR uses original per-hit textures',
                                 habitat_seed=seed,channel_type=cfg['audio']['mode'],sample_rate=cfg['audio']['sample_rate'])
                metadata=dict(benchmark=cfg['benchmark'],version='0.1',split=split,scene_id=scene['scene_id'],
                    config_sha256=cfg['config_sha256'],source=source,
                    listener=dict(position_world=pos.tolist(),floor_position_world=pose['floor'],
                                  rotation_world=quaternion_xyzw(yaw),yaw_rad=yaw,yaw_pitch_roll=[yaw,0.,0.]),
                    relative_source=relative(emitter,pos,yaw),
                    lidar=dict(cfg['lidar'],num_points=len(mask),num_source_points=int(mask.sum()),source_point_fraction=float(mask.mean())),
                    audio=dict(cfg['audio'],dry_audio_id=clip['clip_id'],dry_audio_file=clip['file_path'],dry_audio_sha256=clip['sha256'],
                               dry_selection='maximum-energy duration window; zero pad if short',rir_file='rir.npy',joint_gain=gain),
                    soundspaces=soundspaces,difficulty=difficulty(points['instance_id'],ident,mesh.objects,capable,cfg['min_source_points']),
                    validity=dict(min_source_points=cfg['min_source_points']),
                    scene_assets={k:sha256(Path(scene[k])) for k in ['labels','semantic','visual','navmesh']},
                    active_sources=[ident],sequence_id=None,timestamp_sec=0.,source_motion=None,
                    provenance=cfg['provenance'])
                write_json(sample/'metadata.json',metadata);preview(sample)
                validate_sample(sample,checksums=False)
                write_json(sample/'COMPLETE.json',{'sha256':{p.name:sha256(p) for p in sample.iterdir()}})
                achieved+=1;total+=1;used_positions.append(pos)
                print(json.dumps({'sample':str(sample),'points':len(mask),'source_points':int(mask.sum())}),flush=True)
            source_results.append(dict(instance_id=ident,samples=achieved))
            if achieved>=cfg['listeners_per_source']:accepted_sources+=1
            if achieved<cfg['listeners_per_source']:
                rejection(root,scene['scene_id'],ident,None,f'insufficient receivers: {achieved}/{cfg["listeners_per_source"]}')
                # Partial source groups are diagnostic artifacts, not stage samples.
                total-=achieved
                if source_root.exists():
                    rejected_root.parent.mkdir(parents=True,exist_ok=True)
                    shutil.move(str(source_root),str(rejected_root))
        return {'scene_id':scene['scene_id'],'sources':accepted_sources,'samples':total,'source_results':source_results}
    finally:backend.close()


def prepare_generation(cfg: dict, overwrite: bool = False) -> tuple[list, dict, list, dict]:
    """Create immutable shared metadata before isolated scene workers start.

    The returned scene list, category IDs, clips, and split plan are stable for
    the life of a benchmark output directory.  This is intentionally separate
    from ``scene_worker`` so a Slurm array can process scenes in parallel
    without concurrently writing split-level metadata.
    """
    import yaml
    root=Path(cfg['output'])
    # Resuming an existing output must retain the provenance recorded for its
    # already-complete samples.  Scheduling-only tools do not change it.
    existing_benchmark=root/'metadata'/'benchmark.json'
    completed=sorted(root.glob('*/*/*/*/metadata.json'))
    if completed:
        # The data file is authoritative: it cannot have been produced by a
        # scheduler-only preparation step.  All future resumed samples must
        # use this exact provenance.
        cfg['provenance']=read_json(completed[0]).get('provenance')
    elif existing_benchmark.is_file():
        previous=read_json(existing_benchmark).get('config',{})
        if previous.get('config_sha256')==cfg.get('config_sha256') and previous.get('provenance'):
            cfg['provenance']=previous['provenance']
    if not cfg.get('provenance'):
        raise ValueError('Generation provenance must be supplied before preparing a benchmark')
    scenes=discover(Path(cfg['hm3d_root']))
    if not cfg.get('source_plan') and len(scenes)<sum(cfg['scene_counts'].values()):
        raise RuntimeError(f"HM3DSem access blocker: found {len(scenes)} complete scenes; need {sum(cfg['scene_counts'].values())}")
    if cfg.get('prerequisite_validation'):
        prerequisite=Path(cfg['config_path']).parent.parent/cfg['prerequisite_validation']
        if not prerequisite.exists() or read_json(prerequisite).get('status')!='PASS':
            raise RuntimeError('Required previous-stage validation missing: '+str(prerequisite))
    selected=yaml.safe_load(Path(cfg['category_config']).read_text())['sound_source_categories']
    cfg['selected_categories']=selected
    if not selected:raise ValueError('Run category analysis and curate observed categories first')
    manifest=Path(cfg['audio_manifest'])
    clips=validate_audio(read_json(manifest),manifest.parent)
    cats=sorted({o['category'] for s in scenes for o in __import__('audiolidar.habitat.scenes',fromlist=['annotations']).annotations(Path(s['labels'])).values()})
    category_ids={c:i for i,c in enumerate(cats)}
    if not set(selected)<=set(cats):raise ValueError('Curated categories absent from observed HM3DSem ontology')
    # Selection metadata is allowed to use simulator annotations, but never
    # enters an observation or inference input.  Requiring enough approved
    # instances up front avoids spending GPU hours on scenes that cannot meet
    # their configured source-group coverage.
    required_annotations=cfg.get('minimum_annotated_sources_per_scene',cfg.get('minimum_sources_per_scene',cfg['sources_per_scene']))
    eligibility=[];eligible_scenes=[]
    for scene in scenes:
        instance_ids=sorted(ident for ident,obj in annotations(Path(scene['labels'])).items()
                            if obj['category'] in selected)
        row=dict(scene_id=scene['scene_id'],eligible_instance_ids=instance_ids,
                 eligible_instance_count=len(instance_ids),eligible=len(instance_ids)>=required_annotations)
        eligibility.append(row)
        if row['eligible']:eligible_scenes.append(scene)
    if not cfg.get('source_plan') and len(eligible_scenes)<sum(cfg['scene_counts'].values()):
        raise RuntimeError(f'Only {len(eligible_scenes)} scenes have at least {required_annotations} approved source instances; need {sum(cfg["scene_counts"].values())}')
    if overwrite and root.exists():
        if root==Path(cfg['hm3d_root']) or Path(cfg['hm3d_root']).is_relative_to(root):
            raise ValueError('Refusing to overwrite HM3D input')
        if not (root/'metadata/benchmark.json').exists():raise ValueError('Refusing to overwrite unrecognized directory')
        shutil.rmtree(root)
    root.mkdir(parents=True,exist_ok=True)
    fingerprints={'category_config':sha256(Path(cfg['category_config'])),
                  'dataset_config':sha256(Path(cfg['dataset_config'])),
                  'materials':sha256(Path(cfg['config_path']).parent.parent/cfg['soundspaces']['materials_json']),
                  'scenes':{s['scene_id']:{k:sha256(Path(s[k])) for k in ['labels','semantic','visual','navmesh']} for s in scenes}}
    fingerprint_path=root/'metadata/input-fingerprints.json'
    if fingerprint_path.exists() and read_json(fingerprint_path)!=fingerprints:
        raise ValueError('Input assets/category configuration changed; use a new output')
    write_json(fingerprint_path,fingerprints)
    plan=split_plan(eligible_scenes,clips,cfg,root/'metadata/splits.json')
    if cfg.get('source_plan'):cfg['planned_sources']=plan['sources']
    write_json(root/'metadata/benchmark.json',{'name':cfg['benchmark'],'stage':cfg['stage'],'config':cfg})
    write_json(root/'metadata/categories.json',category_ids)
    write_json(root/'metadata/scenes.json',scenes)
    write_json(root/'metadata/source-eligibility.json',dict(
        required_annotated_sources=required_annotations,
        source_categories=selected,
        eligible_scene_count=len(eligible_scenes),scenes=eligibility))
    return scenes,category_ids,clips,plan


def load_generation(cfg: dict) -> tuple[list, dict, list, dict]:
    """Read and verify an already prepared output directory for a scene job."""
    import yaml
    root=Path(cfg['output'])
    required=['input-fingerprints.json','splits.json','categories.json','scenes.json','source-eligibility.json']
    missing=[name for name in required if not (root/'metadata'/name).is_file()]
    if missing:raise RuntimeError('Benchmark metadata is not prepared: '+', '.join(missing))
    scenes=discover(Path(cfg['hm3d_root']))
    fingerprints={'category_config':sha256(Path(cfg['category_config'])),
                  'dataset_config':sha256(Path(cfg['dataset_config'])),
                  'materials':sha256(Path(cfg['config_path']).parent.parent/cfg['soundspaces']['materials_json']),
                  'scenes':{s['scene_id']:{k:sha256(Path(s[k])) for k in ['labels','semantic','visual','navmesh']} for s in scenes}}
    if read_json(root/'metadata/input-fingerprints.json')!=fingerprints:
        raise RuntimeError('Prepared benchmark inputs no longer match the configured assets')
    plan=read_json(root/'metadata/splits.json')
    if plan.get('config_sha256')!=cfg['config_sha256']:
        raise RuntimeError('Prepared split plan belongs to a different configuration')
    if cfg.get('source_plan'):
        if plan.get('source_plan_sha256')!=sha256(Path(cfg['source_plan'])):
            raise RuntimeError('Explicit source plan changed')
        for path,digest in read_json(Path(cfg['source_plan']))['preflight_sha256'].items():
            if sha256(Path(path))!=digest:raise RuntimeError('Explicit source preflight evidence changed')
        cfg['planned_sources']=plan['sources']
    categories=read_json(root/'metadata/categories.json')
    expected=sorted({o['category'] for s in scenes for o in __import__('audiolidar.habitat.scenes',fromlist=['annotations']).annotations(Path(s['labels'])).values()})
    if categories!={c:i for i,c in enumerate(expected)}:
        raise RuntimeError('Prepared category IDs no longer match the input ontology')
    selected=yaml.safe_load(Path(cfg['category_config']).read_text())['sound_source_categories']
    if not selected or not set(selected)<=set(expected):
        raise RuntimeError('Configured source categories no longer match the input ontology')
    cfg['selected_categories']=selected
    eligibility=read_json(root/'metadata/source-eligibility.json')
    if cfg.get('source_plan'):validate_source_plan(read_json(Path(cfg['source_plan'])),cfg)
    required_annotations=cfg.get('minimum_annotated_sources_per_scene',cfg.get('minimum_sources_per_scene',cfg['sources_per_scene']))
    if eligibility.get('required_annotated_sources')!=required_annotations or eligibility.get('source_categories')!=selected:
        raise RuntimeError('Prepared source-eligibility plan does not match this configuration')
    allowed={row['scene_id'] for row in eligibility['scenes'] if row['eligible']}
    planned={scene_id for ids in plan['scenes'].values() for scene_id in ids}
    if not planned<=allowed:raise RuntimeError('Split plan contains scene(s) that fail source eligibility')
    clips=validate_audio(read_json(Path(cfg['audio_manifest'])),Path(cfg['audio_manifest']).parent)
    if cfg.get('source_plan'):
        validate_reviewed_audio(clips,selected)
        if {c['clip_id']:c['sha256'] for c in clips}!=plan['audio_hashes']:
            raise RuntimeError('Audio inventory changed after plan preparation')
        if any(c['clip_id'] not in plan['audio'][c['split']] for c in clips):
            raise RuntimeError('Audio split changed after plan preparation')
    return scenes,categories,clips,plan


def finalize_generation(cfg: dict) -> dict:
    """Validate a fully generated parallel stage and write its authoritative status."""
    root=Path(cfg['output'])
    _,_,_,plan=load_generation(cfg)
    expected=[scene for split in cfg['scene_counts'] for scene in plan['scenes'][split]]
    results=[]
    for scene_id in expected:
        report=root/'metadata'/'scene-results'/f'{scene_id}.json'
        if not report.is_file():raise RuntimeError('Missing scene completion report: '+scene_id)
        result=read_json(report)
        if result.get('scene_id')!=scene_id or result.get('status')!='PASS':
            raise RuntimeError('Scene did not pass isolated generation: '+scene_id)
        results.append(result['generation'])
    write_json(root/'metadata/generation.json',results)
    report=validate_dataset(root)
    for split in cfg['scene_counts']:
        paths=sorted(str(p.parent.relative_to(root)) for p in root.glob(f'{split}/*/*/*/COMPLETE.json'))
        write_json(root/f'metadata/{split}.json',paths)
    minimum_sources=cfg.get('minimum_sources_per_scene',cfg['sources_per_scene'])
    enough=all(r['sources']>=minimum_sources and r['samples']>=minimum_sources*cfg['listeners_per_source'] for r in results)
    if cfg.get('source_plan'):
        from collections import Counter
        required=Counter(row['scene_id'] for row in plan['sources'])
        enough=all(r['sources']==required[r['scene_id']] and r['samples']==required[r['scene_id']]*cfg['listeners_per_source'] for r in results)
        planned={(row['split'],row['scene_id'],row['instance_id']):row for row in plan['sources']}
        actual=Counter()
        for split in plan['scenes']:
            for marker in root.glob(f'{split}/*/*/*/COMPLETE.json'):
                m=read_json(marker.parent/'metadata.json')
                key=(split,m['scene_id'],m['source']['instance_id'])
                if key not in planned or m['source']['category']!=planned[key]['category']:
                    raise RuntimeError('Unplanned source in output')
                index=int(marker.parent.name)
                row=planned[key]
                if index>=len(row['listeners']):raise RuntimeError('Unplanned listener index')
                pose=row['listeners'][index]
                if (not np.allclose(m['source']['position_world'],row['position_world']) or
                    not np.allclose(m['listener']['position_world'],pose['position']) or
                    not np.isclose(m['listener']['yaw_rad'],pose['yaw'])):
                    raise RuntimeError('Output pose disagrees with immutable source plan')
                actual[key]+=1
        enough=enough and set(actual)==set(planned) and all(n==cfg['listeners_per_source'] for n in actual.values())
        report['class_samples']=dict(Counter({category:sum(n for key,n in actual.items() if planned[key]['category']==category) for category in cfg['selected_categories']}))
    report.update(stage=cfg['stage'],scene_results=results,status='PASS' if enough else 'INCOMPLETE')
    write_json(root/'metadata/stage_validation.json',report)
    if not enough:raise RuntimeError('Dataset integrity passed but stage coverage incomplete; see stage_validation.json')
    return report


def generate(cfg: dict, resume=False, overwrite=False, workers=1):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    if workers<1:raise ValueError('workers must be positive')
    scenes,category_ids,clips,plan=prepare_generation(cfg,overwrite)
    jobs=[(s,split,cfg,category_ids,clips,resume) for split,ids in plan['scenes'].items() for s in scenes if s['scene_id'] in ids]
    if workers==1:results=[scene_worker(a) for a in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            results=list(pool.map(scene_worker,jobs))
    root=Path(cfg['output'])
    scene_results=root/'metadata'/'scene-results'
    for result in results:
        write_json(scene_results/f"{result['scene_id']}.json",dict(scene_id=result['scene_id'],status='PASS',generation=result))
    return finalize_generation(cfg)
