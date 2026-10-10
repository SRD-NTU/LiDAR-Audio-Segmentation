import numpy as np

THRESHOLDS = np.array([.0001, .0003, .001, .003, .005, .01, .03, .05, .1, .15, .2, .25, .3, .4, .5, .6, .7,
                       .8, .85, .9, .95, .98], np.float32)


def score(probability, counts, positives, threshold):
    pred = probability >= threshold
    tp = int(positives[pred].sum())
    fp = int((counts[pred] - positives[pred]).sum())
    fn = int(positives[~pred].sum())
    return dict(tp=tp, fp=fp, fn=fn, iou=tp / max(tp + fp + fn, 1),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                f1=2 * tp / max(2 * tp + fp + fn, 1))


def summarize(rows):
    pooled = {k: sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn')}
    tp, fp, fn = (pooled[k] for k in ('tp', 'fp', 'fn'))
    return dict(samples=len(rows), mean_iou=float(np.mean([r['iou'] for r in rows])),
                macro={k: float(np.mean([r[k] for r in rows])) for k in ('iou', 'precision', 'recall', 'f1')},
                micro=dict(**pooled, iou=tp / max(tp + fp + fn, 1), precision=tp / max(tp + fp, 1),
                           recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1)))


def threshold_weighted_scores(probability, counts, positive, thresholds=THRESHOLDS):
    """Exact voxel-to-point confusion counts without repeated Boolean scans."""
    bins = np.searchsorted(thresholds, probability, side='right')
    total = np.bincount(bins,weights=counts,minlength=len(thresholds)+1).astype(np.int64)
    foreground = np.bincount(bins,weights=positive,minlength=len(thresholds)+1).astype(np.int64)
    tp = np.cumsum(foreground[::-1])[::-1][1:]
    selected = np.cumsum(total[::-1])[::-1][1:]
    fp = selected-tp; fn = int(positive.sum())-tp
    return [dict(tp=int(t),fp=int(f),fn=int(n),iou=float(t/max(t+f+n,1)),
                 precision=float(t/max(t+f,1)),recall=float(t/max(t+n,1)),
                 f1=float(2*t/max(2*t+f+n,1))) for t,f,n in zip(tp,fp,fn)]
