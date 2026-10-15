"""Validate every corrected supervision row and measure usable source pairs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from orbit.cache_contracts import require_consistent_cache, sampling_statistics
from orbit.train_audio_gate_pairs import same_scan_groups, pairs_from_group, usable_pairs, query_seeds


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result={}
    omitted={'features','audio','acoustic','knn'}
    for split in ('train','val'):
        rows=[]
        for path in sorted((args.cache/split).glob('*.npz')):
            with np.load(path,allow_pickle=False) as z:
                row={key:z[key] for key in z.files if key not in omitted}
            require_consistent_cache(row)
            rows.append(row)
        manifest=json.loads((args.cache/f'{split}_manifest.json').read_text())
        if [str(row['rel']) for row in rows]!=manifest:raise ValueError('Cache manifest order mismatch')
        groups=same_scan_groups(rows)
        pairs=[pair for group in groups for pair in usable_pairs(group)]
        rng=np.random.default_rng(1337)
        conflicts=0
        for a,b in pairs:
            seeds=query_seeds(a,b,rng)
            conflicts+=int(len(seeds)!=len(set(seeds)))
        if conflicts:raise ValueError('Corrected pair sampler produced duplicate seeds')
        stats=sampling_statistics(rows)
        result[split]=dict(rows=len(rows),shared_geometry_groups=len(groups),
            distinct_source_pairs=sum(len(pairs_from_group(g)) for g in groups),
            usable_exclusive_source_pairs=len(pairs),
            usable_groups=sum(bool(usable_pairs(g)) for g in groups),
            no_retained_source_rows=sum(not (row['positive']>0).any() for row in rows),
            no_dominant_source_rows=sum(not (row['node_object']==row['source_object']).any() for row in rows),
            paired_duplicate_seed_rows=conflicts,random_trials_per_pair=1,
            sampling_statistics=stats,
            raw_xyz_identity='verified against actual raw NPZ bytes by orbit.prepare')
        print(json.dumps({split:result[split]}),flush=True)
        del rows,groups,pairs
    args.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
