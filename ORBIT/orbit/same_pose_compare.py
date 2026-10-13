"""Assess whether changing sound changes selection for exactly shared XYZ scans."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--rule', default='hybrid')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    groups = json.loads(args.groups.read_text())
    evaluation = json.loads(args.evaluation.read_text())
    if groups['split'] != 'val' or evaluation['split'] != 'val' or not evaluation['full_validation']:
        raise ValueError('Requires full VAL evaluation and VAL XYZ groups')
    lookup = {r['rel']: r for r in evaluation['samples']}
    if len(lookup) != 768:
        raise ValueError('Expected 768 unique validation observations')
    pairs = []
    for group in groups['groups']:
        observations = group['observations']
        for i, a in enumerate(observations):
            for b in observations[i+1:]:
                if a['source_instance_id'] == b['source_instance_id']:
                    continue
                ra, rb = lookup[a['rel']], lookup[b['rel']]
                ca, cb = ra['choices'][args.rule], rb['choices'][args.rule]
                pairs.append(dict(a=a['rel'], b=b['rel'], same_category=ra['category']==rb['category'],
                                  same_selected_origin=ca == cb,
                                  same_geometry_prompt=ca == cb and ca.startswith('prompt:'),
                                  both_apex=ca == cb == 'apex',
                                  iou_a=float(ra[args.rule]), iou_b=float(rb[args.rule])))
    if not pairs:
        raise ValueError('No distinct-source shared-XYZ pairs')
    def summarize(rows):
        return dict(n=len(rows), same_selected_origin_fraction=float(np.mean([r['same_selected_origin'] for r in rows])),
                    same_geometry_prompt_fraction=float(np.mean([r['same_geometry_prompt'] for r in rows])),
                    both_apex_fraction=float(np.mean([r['both_apex'] for r in rows])),
                    both_iou_ge_half_fraction=float(np.mean([r['iou_a'] >= .5 and r['iou_b'] >= .5 for r in rows])),
                    mean_pair_iou=float(np.mean([(r['iou_a'] + r['iou_b']) / 2 for r in rows]))) if rows else dict(n=0)
    output = dict(split='val', test_used=False, rule=args.rule,
                  pair_count=len(pairs), same_selected_origin_fraction=float(np.mean([r['same_selected_origin'] for r in pairs])),
                  both_iou_ge_half_fraction=float(np.mean([r['iou_a'] >= .5 and r['iou_b'] >= .5 for r in pairs])),
                  mean_pair_iou=float(np.mean([(r['iou_a'] + r['iou_b']) / 2 for r in pairs])),
                  by_category_relation={'same':summarize([r for r in pairs if r['same_category']]),
                                        'different':summarize([r for r in pairs if not r['same_category']])},
                  note='Descriptive within-scan audio intervention. Identical geometry-prompt origins imply identical masks; an apex origin may still yield different audio-conditioned masks.',
                  pairs=pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!='pairs'}, indent=2))


if __name__ == '__main__':
    main()
