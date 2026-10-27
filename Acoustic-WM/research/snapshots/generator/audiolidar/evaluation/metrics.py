"""Binary point metrics. Oracle analyses are evaluator-only debugging."""
from __future__ import annotations
import numpy as np


def metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    p,g=np.asarray(pred_mask),np.asarray(gt_mask)
    if p.ndim!=1 or p.shape!=g.shape or not p.size:raise ValueError('Masks must be matching nonempty N arrays')
    if not np.isin(p,[0,1]).all() or not np.isin(g,[0,1]).all():raise ValueError('Masks must be binary')
    p,g=p.astype(bool),g.astype(bool)
    tp=int(np.count_nonzero(p&g)); fp=int(np.count_nonzero(p&~g)); fn=int(np.count_nonzero(~p&g))
    return {'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,
            'precision':tp/(tp+fp) if tp+fp else float(not g.any()),
            'recall':tp/(tp+fn) if tp+fn else 1.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.}
