"""Plot saved LiDARFusion test scores on the original reference point set."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / 'Acoustic-WM-LidarFusion'
DATASET = HERE.parent.parent / 'dataset/v2-tetra4-4channel-full'


def main():
    with (DATASET / 'metadata/samples.csv').open(newline='') as stream:
        categories = {r['sample']: r['category'] for r in csv.DictReader(stream) if r['split'] == 'test'}
    rows = [json.loads(p.read_text()) for p in sorted((MODEL / 'outputs/samples/test').rglob('*.json'))]
    assert len(rows) == 768 and len({r['sample'] for r in rows}) == 768
    values = [r['iou'] for r in rows]
    reported = json.loads((MODEL / 'outputs/metrics.json').read_text())['test_iou']
    assert math.isclose(statistics.fmean(values), reported, abs_tol=1e-10)
    grouped = defaultdict(list)
    for row in rows:
        grouped[categories[row['sample']]].append(row['iou'])
    ranked = sorted(((name, statistics.fmean(scores)) for name, scores in grouped.items()), key=lambda x: x[1], reverse=True)
    aggregate = [statistics.fmean(values), statistics.median(values),
                 sum(v >= .5 for v in values) / len(values), sum(v > 0 for v in values) / len(values)]
    counts = [sum(v == 0 for v in values), sum(0 < v < .5 for v in values), sum(v >= .5 for v in values)]
    assert sum(counts) == 768
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlepad': 12})
    fig, axes = plt.subplots(1, 3, figsize=(24, 9), dpi=100, gridspec_kw={'wspace': .29})
    fig.subplots_adjust(left=.045, right=.985, bottom=.17, top=.85)
    fig.suptitle('Acoustic-WM LiDARFusion: test performance (768 samples)', fontsize=22, fontweight='bold', y=.96)
    ax = axes[0]
    bars = ax.bar(['mIoU', 'Median IoU', 'IoU ≥ 0.5', 'IoU > 0'], aggregate, color='#337fb6', width=.78)
    ax.set_title('Aggregate metrics', fontsize=16)
    ax.set_ylabel('Score (fraction of samples or mean IoU)')
    ax.set_ylim(0, max(.8, math.ceil((max(aggregate) + .03) * 10) / 10))
    ax.yaxis.grid(True, color='#dddddd'); ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f'{v:.1%}' for v in aggregate], padding=4)
    ax = axes[1]
    bars = ax.barh([r[0] for r in ranked], [r[1] for r in ranked], color='#49a6cc', height=.78)
    ax.invert_yaxis(); ax.set_title('mIoU by source category', fontsize=16)
    ax.set_xlabel('Mean IoU'); ax.set_xlim(0, max(.8, math.ceil((max(r[1] for r in ranked) + .06) * 10) / 10))
    ax.xaxis.grid(True, color='#dddddd'); ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f'{r[1]:.1%}' for r in ranked], padding=4, fontsize=9)
    ax = axes[2]
    colors = ['#e66100', '#edaf00', '#1b9e77']
    wedges, _ = ax.pie(counts, colors=colors, startangle=270, counterclock=False,
                       wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax.set_title('Per-sample IoU outcomes', fontsize=16)
    labels = [f'{label}: {count} ({count / len(values):.1%})' for label, count in zip(['IoU = 0', '0 < IoU < 0.5', 'IoU ≥ 0.5'], counts)]
    ax.legend(wedges, labels, loc='upper center', bbox_to_anchor=(.5, -.025), frameon=True, fontsize=10)
    fig.text(.045, .035, 'IoU evaluated on original reference points; two acoustic observations and two measured LiDAR scans.', fontsize=11, color='#444444')
    output = HERE / 'acoustic_wm_lidarfusion_test_performance.png'
    fig.savefig(output, dpi=100, facecolor='white'); plt.close(fig)
    print(json.dumps(dict(output=str(output),samples=len(values),mean_iou=aggregate[0],median_iou=aggregate[1],
                          iou_at_least_half=counts[2],nonzero_iou=sum(v > 0 for v in values),zero_iou=counts[0],categories=len(ranked)), indent=2))


if __name__ == '__main__':
    main()
