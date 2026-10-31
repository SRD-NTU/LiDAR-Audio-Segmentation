"""Label-free features from frozen acoustic forecasts and mask hypotheses."""
import math
import torch
from acoustic_wm.models.future_grounding import initial_belief


@torch.no_grad()
def acoustic_features(model, row):
    """All row inputs here are measured data, candidate geometry or predictions."""
    k,h,_=row['hypotheses'].shape
    q,lv,_=model.world(row['points'][None],row['features'][None],row['valid'][None],
                       row['hypotheses'].reshape(1,-1,3))
    current=((q[0]-row['observed']).abs().square().sum((-1,-2))*(-lv[0]).exp()+lv[0]).reshape(k,h,48)
    energy=torch.cat([current*row['reliability'][None,None]*48,
                       row['semantic'][:,None,None].expand(-1,h,1)],-1)
    base,log_site=initial_belief(model.selector,energy,row['baseline'])
    score=model.head(base,log_site,row['future_energy'],row['future_reliability'])
    sites=log_site.exp()
    next_energy=row['future_energy'].float()
    def band_summary(values,reliability):
        return torch.cat([(values*sites[...,None]).sum(1)*reliability[None]*48,
                          values.amin(1)*reliability[None]*48],-1)
    centered=score-score.max()
    mean=score.mean(); std=score.std(unbiased=False).clamp_min(.1)
    scalar=torch.stack([row['baseline'],row['semantic'],base,score,score-base,
                         centered,(score-mean)/std,score.softmax(0),
                         -(sites*log_site).sum(1),log_site.max(1).values],-1)
    position=row['hypotheses'][:,0]
    spread=(row['hypotheses']-position[:,None]).square().mean(1).sqrt()
    geometry=torch.cat([position/10,position.norm(dim=-1,keepdim=True)/10,spread/5,
                        row['action'][:3].norm().expand(k,1)/5],-1)
    features=torch.cat([row['rank_features'].float(),scalar,
                        band_summary(current,row['reliability']),
                        band_summary(next_energy,row['future_reliability']),geometry],-1)
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite recovery features')
    return features,score


def pad_rows(rows, device):
    count=len(rows); width=max(len(r['base']) for r in rows); dimension=rows[0]['x'].shape[1]
    x=torch.zeros(count,width,dimension,device=device)
    score=torch.full((count,width),-1e4,device=device)
    quality=torch.zeros(count,width,device=device)
    valid=torch.zeros(count,width,dtype=torch.bool,device=device)
    for i,row in enumerate(rows):
        n=len(row['base']);x[i,:n]=row['x'].float().to(device)
        score[i,:n]=row['base'].float().to(device);quality[i,:n]=row['iou'].float().to(device)
        valid[i,:n]=True
    return x,score,quality,valid


class ConservativeRanker(torch.nn.Module):
    def __init__(self,mean,std,kind='linear',bound=2.):
        super().__init__()
        self.register_buffer('mean',mean.float())
        self.register_buffer('std',std.float().clamp_min(.01))
        self.kind=kind;self.bound=bound
        dimension=len(mean)
        if kind=='linear':
            self.network=torch.nn.Linear(dimension*2,1)
            torch.nn.init.zeros_(self.network.weight);torch.nn.init.zeros_(self.network.bias)
        else:
            self.network=torch.nn.Sequential(torch.nn.Linear(dimension*2,64),torch.nn.GELU(),
                torch.nn.Dropout(.1),torch.nn.Linear(64,1))
            torch.nn.init.zeros_(self.network[-1].weight);torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self,x,base,valid):
        normalized=((x-self.mean)/self.std).clamp(-10,10)
        probability=base.masked_fill(~valid,-1e4).softmax(-1)
        context=(normalized*probability[...,None]).sum(1,keepdim=True).expand_as(normalized)
        delta=self.bound*torch.tanh(self.network(torch.cat([normalized,context],-1)).squeeze(-1))
        return (base+delta).masked_fill(~valid,-1e4),delta


def objective(score,base,quality,valid,delta):
    """TRAIN labels only: original-point regret, significant pair ordering, restraint."""
    probability=score.softmax(-1)
    regret=(quality.max(-1).values-(probability*quality).sum(-1)).mean()
    # Weight mistakes involving plausible predictions; avoid density-based targets.
    difference=quality[:,:,None]-quality[:,None,:]
    pairs=(difference>.05)&valid[:,:,None]&valid[:,None,:]
    emphasis=(base.softmax(-1)[:,:,None]+base.softmax(-1)[:,None,:]).detach()
    weights=difference.clamp_min(0)*pairs*emphasis
    pair=torch.nn.functional.softplus(score[:,None,:]-score[:,:,None]+.05)
    ranking=(pair*weights).sum((1,2))/weights.sum((1,2)).clamp_min(1e-6)
    regularization=(delta.square()*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    return regret+ranking.mean()+.02*regularization.mean()
