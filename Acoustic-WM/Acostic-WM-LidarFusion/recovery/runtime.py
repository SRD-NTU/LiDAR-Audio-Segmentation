"""Runtime helpers for the selected two-scan acoustic world model."""
import os


def setup(output, rank):
    for key, folder in [('XDG_CACHE_HOME', 'xdg'), ('TORCH_HOME', 'torch'),
                        ('MPLCONFIGDIR', 'matplotlib'), ('NUMBA_CACHE_DIR', 'numba')]:
        path = output/'runtime_cache'/str(rank)/folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    (output/'runtime_cache'/str(rank)/'xdg/torch/kernels').mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      USE_TF='0', CUBLAS_WORKSPACE_CONFIG=':4096:8')


def bank_input(encoder,model,xyz,wave,rate,action,observed,reliability,future_observed,future_reliability,*,row=None,plan=None):
    import numpy as np
    import torch
    from acoustic_wm.models.pointbind_adapter import proposal_points
    from acoustic_wm.audio.predictive_features import geometric_tokens
    if row is None:row=encoder.encode_arrays(xyz,wave,rate)
    body=[]
    clouds=np.stack([proposal_points(row['xyz'],m) for m in row['masks']])
    for start in range(0,len(clouds),16):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            body.append(model.pointbind.points(model.tensor(clouds[start:start+16])).half().float())
    body=torch.cat(body);row['pointbind_feature']=body.cpu().numpy()
    # This action was selected by the original blind policy before acquisition.
    if plan is None:plan=model.plan(row,observed,reliability,action[None])
    future=torch.view_as_complex(model.tensor(future_observed).contiguous())
    energy=model.future.evidence(plan['prediction'],plan['log_variance'],future)
    geometry=geometric_tokens(row['xyz'],row['full_point_features'])
    value=dict(points=model.tensor(geometry[0]),features=model.tensor(geometry[1]),valid=model.tensor(geometry[2]),
        hypotheses=model.tensor(row['hypotheses']),observed=plan['observed'],
        reliability=model.tensor(reliability),future_reliability=model.tensor(future_reliability),
        action=model.tensor(action),baseline=model.tensor(row['baseline']),
        semantic=-model.emission.log_probability(body,model.tensor(row['audio'])),
        future_energy=energy,rank_features=model.tensor(row['rank_features']))
    return row,value
