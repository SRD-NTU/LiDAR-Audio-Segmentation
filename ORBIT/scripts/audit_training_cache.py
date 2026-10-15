"""Count raw scan relationships, cached relationships, and seed conflicts.

Raw XYZ fingerprints are read from the full encoder archives; the report labels
that provenance explicitly. This is not an independent rehash of every raw NPZ.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from apex import data as D
from orbit.cache_contracts import geometry_uid
from orbit.audio_gate import sample_seeds


def summarize(groups):
    valid=[group for group in groups.values() if len({item[0] for item in group})>1]
    return dict(groups=len(valid),distinct_source_pairs=sum(sum(a[0]!=b[0]
                for i,a in enumerate(group) for b in group[i+1:]) for group in valid))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    source=D.ROOT/'.redun/20261015/source-before-audit-fixes/ORBIT/orbit/audio_gate.py'
    spec=importlib.util.spec_from_file_location('original_seed_sampler',source)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    def read(path):
        with np.load(path,allow_pickle=False) as z:
            row={key:z[key] for key in ('rel','xyz','node_object','positive','count','source_object')}
        rel=str(row['rel'])
        with np.load(args.geometry/(rel+'.npz'),allow_pickle=False) as z:digest=str(z['xyz_sha256'])
        meta=json.loads((D.DATA/rel/'metadata.json').read_text())
        uid=geometry_uid(digest,rel.split('/')[1],meta['listener'],meta['lidar'])
        sampled=(uid,row['xyz'].shape,hashlib.sha256(row['xyz'].tobytes()).hexdigest())
        positive=row['positive']>0
        stats=Counter(rows=1,positive_voxels=int(positive.sum()),
            support_overlap_voxels=int((positive&(row['node_object']!=int(row['source_object']))).sum()))
        if positive.any():
            rng=np.random.default_rng(1337)
            before=original.sample_seeds(row,rng)
            after=sample_seeds(row,np.random.default_rng(1337))
            stats['legacy_duplicate_seed_rows']=int(len(before)!=len(set(before)))
            stats['legacy_positive_negative_seed_conflicts']=int((row['positive'][before[1:]]>0).sum())
            stats['corrected_duplicate_seed_rows']=int(len(after)!=len(set(after)))
            stats['corrected_positive_negative_seed_conflicts']=int((row['positive'][after[1:]]>0).sum())
        return uid,sampled,int(meta['source']['instance_id']),stats
    report=dict(seed_sampling_trials_per_observation=1,
                raw_fingerprint_provenance='full encoder xyz_sha256 plus scene/listener/sensor configuration',splits={})
    for split in ('train','val'):
        raw=defaultdict(list);sampled=defaultdict(list);stats=Counter()
        with ThreadPoolExecutor(8) as pool:
            for i,(uid,key,source,counts) in enumerate(pool.map(read,sorted((args.cache/split).glob('*.npz'))),1):
                raw[uid].append((source,));sampled[key].append((source,));stats.update(counts)
                if i%800==0:print(split,i,flush=True)
        report['splits'][split]=dict(raw_identity=summarize(raw),existing_sampled_geometry=summarize(sampled),
            corrected_identity_grouping=summarize(raw),corrected_grouping_is_prospective_until_cache_rebuild=True,
            sampling=dict(stats))
        print(json.dumps(report['splits'][split],indent=2),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
