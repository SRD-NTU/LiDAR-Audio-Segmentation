"""Paired comparisons at equal recording budgets; never select a TEST checkpoint."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.evaluation import scene_summary
import numpy as np
import argparse

p = argparse.ArgumentParser()
p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
p.add_argument('--protocol',default='config/frozen_multiview_neo1.json')
args = p.parse_args()
protocol = json.loads((ROOT/args.protocol).read_text())
if protocol.get('memory_update') != 'trained_gate_on_frozen_singleview_gru_at_acquired_pose_v1':
    raise RuntimeError('Neo report requires a frozen recurrent-memory protocol')
provenance={name:protocol['provenance'][name] for name in
            ['source_manifest_sha256','config_manifest_sha256','dependency_manifest_sha256']}
rows = samples(args.split)
values = {}
counts = {}
for method in protocol['methods']:
    values[method] = {}
    counts[method] = {}
    for views in protocol['report_views']:
        found = {}
        for sample in rows:
            path = ROOT/protocol['results_root']/sample/method/('view-%d.json' % views)
            if path.exists():
                record = json.loads(path.read_text())
                if (record['protocol_sha256'] != protocol['protocol_sha256'] or
                    record.get('provenance')!=provenance or record['recordings'] != views):
                    raise RuntimeError('Stale or invalid result: '+str(path))
                if not np.isfinite(record['iou']) or not 0 <= record['iou'] <= 1:
                    raise RuntimeError('Invalid IoU')
                found[sample] = record
        values[method][views] = found
        counts[method][str(views)] = len(found)
complete = all(n == len(rows) for method in counts.values() for n in method.values())
report = dict(split=args.split, complete=complete, expected_references=len(rows), completed=counts,
              protocol_sha256=protocol['protocol_sha256'],provenance=provenance,
              limitations=protocol['limitations'])
if complete:
    budgets = {}
    for views in protocol['report_views']:
        iou = {m: np.array([values[m][views][r]['iou'] for r in rows]) for m in protocol['methods']}
        random = np.stack([iou['random-%d' % seed] for seed in range(3)])
        baseline = np.array([values['world'][views][r]['singleview_iou'] for r in rows])
        entry = dict(singleview=scene_summary(baseline, rows),
                     methods={m: scene_summary(v, rows) for m, v in iou.items()},
                     random_mean=scene_summary(random.mean(0), rows),
                     random_seed_means=random.mean(1).tolist(),
                     learned_minus_random=scene_summary(iou['world']-random.mean(0), rows),
                     learned_minus_frozen=scene_summary(iou['world']-iou['frozen'], rows),
                     mean_travel_m={m: float(np.mean([values[m][views][r]['cumulative_travel_m'] for r in rows]))
                                    for m in protocol['methods']})
        budgets[str(views)] = entry
    report['budgets'] = budgets
    lines = ['# Neo recurrent-memory sequential evaluation', '',
             'Views include the initial recording. All masks segment the original LiDAR cloud.', '',
             '| Total views | Learned IoU (%) | Frozen-world IoU (%) | Random IoU (%) | Learned − random (pp; 95% scene CI) |',
             '|---:|---:|---:|---:|---:|']
    for views, entry in budgets.items():
        delta = entry['learned_minus_random']
        lines.append('| %s | %.3f | %.3f | %.3f | %.3f [%.3f, %.3f] |' % (
            views, 100*entry['methods']['world']['mean'], 100*entry['methods']['frozen']['mean'],
            100*entry['random_mean']['mean'], 100*delta['mean'], 100*delta['ci95'][0], 100*delta['ci95'][1]))
    lines += ['', 'Random is averaged across three seeds per reference. Confidence intervals resample scenes.', '',
              '## Limits', ''] + ['- '+s for s in protocol['limitations']]
    output_path('reports/'+protocol['revision']+'/multiview-%s.md' % args.split).write_text('\n'.join(lines)+'\n')
write_json('reports/'+protocol['revision']+'/multiview-%s.json' % args.split, report)
print(json.dumps(report, indent=2), flush=True)
