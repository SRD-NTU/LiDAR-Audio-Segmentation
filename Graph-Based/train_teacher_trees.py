"""Context-rich CPU challenger with observation-only world/graph features."""
import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import numpy as np
from scipy.special import expit, logit, softmax
from graph_audio.metrics import THRESHOLDS, summarize
from evaluate_teacher_refinement import threshold_scores
from train_trees import manifests, inputs
from prepare import ROOT, DATASET


def teacher_inputs(graph, teacher, ids=None, learned_audio_logits=None):
    n = len(graph['counts'])
    ids = np.arange(n) if ids is None else ids
    probability = np.clip(teacher['probability'],1e-5,1-1e-5)
    geometry = teacher['geometry'].astype(np.float32)
    if not np.isfinite(geometry).all() or not np.isfinite(probability).all():
        raise FloatingPointError('Invalid world observations')
    posterior = softmax(geometry[:,24:],axis=-1)
    prior = softmax(graph['text_logits'].astype(np.float32)/2) if learned_audio_logits is None else softmax(learned_audio_logits)
    match = (posterior[:,1:]*prior[None]).sum(-1)
    local = np.column_stack((geometry,logit(probability),match))
    neighbors = graph['neighbors'][ids]
    valid = graph['edge_valid'][ids]
    context = (local[neighbors]*valid[...,None]).sum(1)/np.maximum(valid.sum(1,keepdims=True),1)
    # Same train/inference schema. No category/source/instance labels appear here.
    features = graph['features'][ids].astype(np.float32)
    field = graph['acoustic'].astype(np.float32)
    normalized = (field[ids]-field.mean(0))/np.maximum(field.std(0),.1)
    audio = np.concatenate((prior,graph['audio_context']))
    return np.concatenate((features,np.broadcast_to(audio,(len(ids),21)),field[ids],
                           normalized,local[ids],context),1)


def training_case(task):
    name,index = task
    rng = np.random.default_rng(1337+index)
    with np.load(ROOT/'cache/graphs'/(name+'.npz')) as graph, np.load(ROOT/'cache/teacher'/(name+'.npz')) as teacher:
        label = graph['positives']/np.maximum(graph['counts'],1) >= .5
        positive,negative = np.flatnonzero(label),np.flatnonzero(~label)
        ids = np.concatenate((rng.choice(positive,min(64,len(positive)),replace=False),
                              rng.choice(negative,min(256,len(negative)),replace=False)))
        audio = np.load(ROOT/'cache/audio_category'/(name+'.npz'))['logits']
        return teacher_inputs(graph,teacher,ids,audio),label[ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-cache',action='store_true')
    parser.add_argument('--teacher-only',action='store_true')
    parser.add_argument('--train-limit',type=int)
    parser.add_argument('--val-limit',type=int)
    parser.add_argument('--rounds',type=int,default=600)
    parser.add_argument('--run',default='world-context-trees-full')
    args = parser.parse_args()
    train,val = manifests('train'),manifests('val')
    if args.teacher_only:
        train = [n for n in train if (ROOT/'cache/teacher'/(n+'.npz')).exists()]
        val = [n for n in val if (ROOT/'cache/teacher'/(n+'.npz')).exists()]
    train = train[:args.train_limit] if args.train_limit else train
    val = val[:args.val_limit] if args.val_limit else val
    if not train or not val:
        raise ValueError('Empty training/validation manifest')
    if args.wait_cache:
        while not (ROOT/'cache/audio_category/ready.json').exists() or not all((ROOT/'cache/teacher'/(n+'.npz')).exists() for n in train+val):
            print('Waiting for complete world context caches',flush=True)
            time.sleep(20)
    run = ROOT/'runs'/args.run
    audio_digest = json.loads((ROOT/'cache/audio_category/ready.json').read_text())['model_sha256']
    run.mkdir(parents=True,exist_ok=True)
    (run/'train_manifest.json').write_text(json.dumps(train))
    (run/'val_manifest.json').write_text(json.dumps(val))
    with np.load(ROOT/'cache/teacher'/(train[0]+'.npz')) as teacher:
        digest = str(teacher['teacher_sha256'])
    rows = len(train)*320
    x = np.memmap(run/'training_features.mmap',mode='w+',dtype=np.float32,shape=(rows,483))
    y = np.zeros(rows,np.uint8)
    position = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,(values,labels) in enumerate(pool.map(training_case,enumerate_tasks(train)),1):
            x[position:position+len(values)] = values; y[position:position+len(values)] = labels
            position += len(values)
            if i%128==0 or i==len(train):
                print(f'Context tree observations {i}/{len(train)}; nodes {position}',flush=True)
    x.flush()
    model = lgb.LGBMClassifier(n_estimators=args.rounds,learning_rate=.04,num_leaves=47,max_depth=12,
                min_child_samples=128,colsample_bytree=.8,reg_lambda=3.,n_jobs=3,random_state=1337,
                verbosity=-1,force_col_wise=True)
    model.fit(x[:position],y[:position])
    model.booster_.save_model(str(run/'model.txt'))
    with (DATASET/'metadata/samples.csv').open() as f:
        metadata = {r['sample']:r for r in csv.DictReader(f)}
    scores = {(strength,float(t)):[] for strength in (0.,.25,.5,1.) for t in THRESHOLDS}
    for i,name in enumerate(val,1):
        with np.load(ROOT/'cache/graphs'/(name+'.npz')) as graph, np.load(ROOT/'cache/teacher'/(name+'.npz')) as teacher:
            audio = np.load(ROOT/'cache/audio_category'/(name+'.npz'))['logits']
            probability = model.predict_proba(teacher_inputs(graph,teacher,learned_audio_logits=audio))[:,1]
            base = np.clip(teacher['point_probability'],1e-5,1-1e-5)
            node = np.clip(teacher['probability'],1e-5,1-1e-5)
            delta = logit(np.clip(probability,1e-5,1-1e-5))-logit(node)
            inverse = graph['inverse']
            with np.load(DATASET/name/'lidar.npz') as lidar:
                target = lidar['sound_mask'].astype(np.int64)
            for strength in (0.,.25,.5,1.):
                prediction = expit(logit(base)+strength*delta[inverse])
                for threshold,score in zip(THRESHOLDS,threshold_scores(prediction,target)):
                    scores[strength,float(threshold)].append(dict(sample=name,scene=metadata[name]['scene'],
                                          category=metadata[name]['category'],**score))
        if i%128==0 or i==len(val):
            print(f'Context tree validation {i}/{len(val)}',flush=True)
    selected = max(scores,key=lambda k: np.mean([r['iou'] for r in scores[k]]))
    report = dict(type='teacher_trees',split='val',full_split=len(val)==768,training_samples=len(train),
        teacher_checkpoint_sha256=digest,audio_head_sha256=audio_digest,threshold=selected[1],residual_strength=selected[0],
        metrics=summarize(scores[selected]),samples_detail=scores[selected],training_nodes=position,
        model_sha256=hashlib.sha256((run/'model.txt').read_bytes()).hexdigest(),
        calibration_trials=[dict(residual_strength=s,threshold=t,mean_iou=summarize(v)['mean_iou'])
                            for (s,t),v in scores.items()],
        test_used_for_selection=False)
    (run/'validation.json').write_text(json.dumps(report,indent=2))
    (run/'config.json').write_text(json.dumps({**vars(args),'teacher_checkpoint_sha256':digest,'audio_head_sha256':audio_digest,
                                'training_samples':len(train),'validation_samples':len(val)},indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples_detail','calibration_trials')},indent=2),flush=True)
    from summarize import main as refresh_reports
    refresh_reports()


def enumerate_tasks(names):
    return ((name,index) for index,name in enumerate(names))


if __name__ == '__main__':
    main()
