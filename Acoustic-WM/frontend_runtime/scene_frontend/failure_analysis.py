"""Report candidate coverage and selection failures from a saved validation run."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def summarize(rows, key):
    def mean(values):return float(np.mean(values)) if values else None
    d={
        'n':len(rows),
        'selected_mean_iou':mean([r[key] for r in rows]),
        'candidate_oracle_mean_iou':mean([r['oracle_candidate_iou'] for r in rows]),
        'selection_regret':mean([r['oracle_candidate_iou']-r[key] for r in rows]),
        'candidate_oracle_below_0.25':sum(r['oracle_candidate_iou']<.25 for r in rows),
        'candidate_oracle_at_least_0.5_but_selected_below_0.1':sum(
            r['oracle_candidate_iou']>=.5 and r[key]<.1 for r in rows),
        'source_seed_missed':sum(not r['source_seed_hit'] for r in rows),
    }
    return d


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rule',default='hybrid')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=json.loads(args.input.read_text())
    if report['split']!='val' or not report['full_validation']:
        raise ValueError('Requires complete validation report')
    rows=report['samples']
    for row in rows:
        if args.rule not in row:raise ValueError('Requested rule absent')
    by_tier=defaultdict(list);by_category=defaultdict(list);by_scene=defaultdict(list)
    for row in rows:
        by_tier[row['tier']].append(row)
        by_category[row['category']].append(row)
        by_scene[row['scene']].append(row)
    result=dict(input=str(args.input),rule=args.rule,test_used=False,
                overall=summarize(rows,args.rule),
                by_difficulty={k:summarize(v,args.rule) for k,v in sorted(by_tier.items())},
                by_category={k:summarize(v,args.rule) for k,v in sorted(by_category.items())},
                by_scene={k:summarize(v,args.rule) for k,v in sorted(by_scene.items())},
                worst_selection_regret=[dict(rel=r['rel'],category=r['category'],tier=r['tier'],
                    selected_iou=r[args.rule],oracle_candidate_iou=r['oracle_candidate_iou'],
                    source_seed_hit=r['source_seed_hit']) for r in sorted(rows,
                    key=lambda r:r['oracle_candidate_iou']-r[args.rule],reverse=True)[:20]])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result['overall'],indent=2))


if __name__=='__main__':main()
