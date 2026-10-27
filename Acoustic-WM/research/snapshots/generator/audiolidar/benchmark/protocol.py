"""Scene and recording-disjoint immutable split plans."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from audiolidar.utils.io import read_json,write_json,sha256


def validate_source_plan(plan: dict, cfg: dict) -> dict:
    """Validate evaluator-only explicit source identities and preflight poses."""
    from collections import Counter
    if 'geometry' in plan and any(cfg.get(key)!=value for key,value in plan['geometry'].items()):
        raise ValueError('Source plan geometry differs from generation configuration')
    for name,digest in plan.get('implementation_sha256',{}).items():
        if sha256(Path(__file__).resolve().parents[2]/name)!=digest:
            raise ValueError('Source plan geometry implementation changed')
    rows=plan['sources']; seen=set(); scene_splits={}; counts=Counter()
    for row in rows:
        key=(row['scene_id'],row['instance_id'])
        if key in seen:raise ValueError('Duplicate physical source group')
        seen.add(key)
        split=row['split']; category=row['category']
        if split not in ('train','val','test'):raise ValueError('Invalid source split')
        if scene_splits.setdefault(row['scene_id'],split)!=split:raise ValueError('Source plan scene leakage')
        if category not in cfg['selected_categories']:raise ValueError('Unapproved planned source category')
        counts[category,split]+=1
        listeners=row['listeners']
        if len(listeners)!=cfg['listeners_per_source']:raise ValueError('Wrong listener quota')
        emitter=np.asarray(row['position_world'],dtype=float)
        if emitter.shape!=(3,) or not np.isfinite(emitter).all():raise ValueError('Invalid emitter')
        positions=[]
        for pose in listeners:
            pos=np.asarray(pose['position'],dtype=float);floor=np.asarray(pose['floor'],dtype=float)
            if pos.shape!=(3,) or floor.shape!=(3,) or not np.isfinite(np.r_[pos,floor,pose['yaw']]).all():
                raise ValueError('Invalid listener pose')
            if any(np.linalg.norm(pos-old)<cfg['min_listener_separation_m'] for old in positions):
                raise ValueError('Listener separation failed')
            positions.append(pos)
    quotas=cfg['source_groups_per_class_split']
    for category in cfg['selected_categories']:
        for split in ('train','val','test'):
            if counts[category,split]!=quotas[split]:raise ValueError('Class/source split quota failed: '+category+'/'+split)
    return {split:sorted(scene for scene,s in scene_splits.items() if s==split) for split in ('train','val','test')}


def validate_reviewed_audio(clips: list, categories: list) -> None:
    """Automatic download sanity checks do not establish physical audio pairing."""
    for clip in clips:
        if clip['semantic_category'] not in categories:continue
        review=clip.get('review',{})
        if any(review.get(key)!='approved' for key in ('physical_match','license','source')):
            raise ValueError('Audio needs physical/source/license review: '+clip['clip_id'])
        if not review.get('reviewer') or not review.get('evidence'):
            raise ValueError('Audio review lacks reviewer/evidence: '+clip['clip_id'])
        if not clip.get('original_recording_id'):
            raise ValueError('Audio lacks original recording identity: '+clip['clip_id'])


def validate_audio(clips: list, base: Path) -> list:
    if not clips:raise ValueError('No licensed audio clips supplied')
    ids,hash_splits={},{};recording_splits={}
    for c in clips:
        for key in ['clip_id','semantic_category','file_path','duration','sample_rate','license','source','split']:
            if key not in c or c[key] is None:raise ValueError('Missing audio field '+key)
        if c['split'] not in ['train','val','test']:raise ValueError('Invalid audio split')
        if c['clip_id'] in ids:raise ValueError('Duplicate audio ID')
        p=Path(c['file_path']); p=p if p.is_absolute() else base/p
        digest=sha256(p)
        # A recording may not reappear in another split under a different filename.
        if digest in hash_splits and hash_splits[digest]!=c['split']:raise ValueError('Audio content leakage')
        recording=c.get('original_recording_id',digest)
        if recording in recording_splits and recording_splits[recording]!=c['split']:raise ValueError('Original recording leakage')
        recording_splits[recording]=c['split']
        if c.get('sha256') and c['sha256']!=digest:raise ValueError('Dry audio checksum changed')
        ids[c['clip_id']]=c['split']; hash_splits[digest]=c['split']
        c.update(file_path=str(p.resolve()),sha256=digest)
    return clips


def split_plan(scenes: list, clips: list, cfg: dict, path: Path) -> dict:
    counts=cfg['scene_counts']; need=sum(counts.values())
    if not cfg.get('source_plan') and len(scenes)<need:raise ValueError(f'Need {need} complete HM3DSem scenes, found {len(scenes)}')
    ids=sorted(s['scene_id'] for s in scenes)
    rng=np.random.default_rng(cfg['seed']); rng.shuffle(ids)
    scene_splits={};start=0
    for split,n in counts.items():scene_splits[split]=ids[start:start+n];start+=n
    explicit=None
    if cfg.get('source_plan'):
        explicit=read_json(Path(cfg['source_plan']))
        if not explicit.get('geometry') or not explicit.get('preflight_sha256'):
            raise ValueError('Explicit source plan lacks geometry/preflight evidence')
        for file,digest in explicit.get('preflight_sha256',{}).items():
            if sha256(Path(file))!=digest:raise ValueError('Source preflight evidence changed')
        scene_splits=validate_source_plan(explicit,cfg)
        if not {s for group in scene_splits.values() for s in group}<=set(ids):
            raise ValueError('Explicit plan contains unavailable/ineligible scene')
    result={'seed':cfg['seed'],'scenes':scene_splits,
            'audio':{s:sorted(c['clip_id'] for c in clips if c['split']==s) for s in counts},
            'audio_hashes':{c['clip_id']:c['sha256'] for c in clips},
            'config_sha256':cfg['config_sha256']}
    if explicit is not None:
        validate_reviewed_audio(clips,cfg['selected_categories'])
        result['sources']=explicit['sources']
        result['source_plan_sha256']=sha256(Path(cfg['source_plan']))
        for category in cfg['selected_categories']:
            for split in scene_splits:
                recordings={c.get('original_recording_id',c['sha256']) for c in clips
                            if c['semantic_category']==category and c['split']==split}
                if len(recordings)<cfg.get('minimum_recordings_per_class_split',2):
                    raise ValueError('Insufficient independent recordings: '+category+'/'+split)
    for s,n in counts.items():
        if n and not result['audio'][s]:raise ValueError('No dry audio assigned to '+s)
    if path.exists():
        old=read_json(path)
        if old!=result:raise ValueError('Existing split plan differs; choose a new output directory')
        return old
    write_json(path,result);return result


def difficulty(instance_ids, source_id: int, objects: dict, capable_categories: set, min_points: int) -> dict:
    ids,counts=np.unique(instance_ids,return_counts=True)
    visible=[int(i) for i,n in zip(ids,counts) if n>=min_points and int(i) in objects]
    source=objects[source_id]['category']
    same=[i for i in visible if i!=source_id and objects[i]['category']==source]
    other=[i for i in visible if i!=source_id and objects[i]['category']!=source and objects[i]['category'] in capable_categories]
    return {'setting':'S3' if same else ('S2' if other else 'S1'),
            'same_class_distractors':len(same),'different_sound_capable_distractors':len(other),
            'visible_instance_ids':visible,'same_class_instance_ids':same}
