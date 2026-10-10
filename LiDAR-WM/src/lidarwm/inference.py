"""Calibrated square-array symmetry ensembling using observations only."""
import torch


def rotate_observations(lidar,audio,valid,turns):
    lidar=lidar.clone();audio=audio.clone();valid=valid.clone()
    for _ in range(turns%4):
        lidar=lidar.roll(lidar.shape[-1]//4,-1)
        for a,b in ((0,1),(4,5)):
            old=lidar[:,a].clone();lidar[:,a]=-lidar[:,b];lidar[:,b]=old
        audio=audio[:,[1,3,0,2]]
        valid=valid.roll(valid.shape[-1]//4,-1)
    return lidar,audio,valid


@torch.inference_mode()
def predict_mask_logits(model,lidar,audio,valid,extra=None,tta=False):
    extra={} if extra is None else extra
    if not tta:return model(lidar,audio,valid,**extra)['mask_logits']
    total=None
    for turns in range(4):
        x,a,v=rotate_observations(lidar,audio,valid,turns)
        p=model(x,a,v,**extra)['mask_logits'].float().sigmoid()
        p=p.roll(-turns*(lidar.shape[-1]//4),-1)
        total=p if total is None else total+p
    return torch.logit((total/4).clamp(1e-6,1-1e-6))
