"""Multi-band physical SRP and a small supervised reliability adapter.

Waveforms and global microphone calibration build features. TRAIN source
directions supervise reliability; source positions never enter inference.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from apex import data as D

BANDS=[(150,600),(600,1200),(1200,2000),(2000,3000),(3000,6000)]


class BandSRP:
    def __init__(self,positions,rate=44100,device='cuda'):
        self.device=device;self.rate=rate
        f=torch.fft.rfftfreq(2048,1/rate,device=device)
        self.keep=(f>=150)&(f<=6000);self.f=f[self.keep]
        self.bands=[(self.f>=lo)&(self.f<hi) for lo,hi in BANDS]
        dirs=torch.tensor(D.acoustic.DIRS,dtype=torch.float32,device=device)
        positions=torch.as_tensor(positions,dtype=torch.float32,device=device)
        self.pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
        delay=torch.stack([-(dirs@(positions[i]-positions[j]))/343 for i,j in self.pairs])
        self.steering=torch.exp(2j*torch.pi*delay[:,:,None]*self.f[None,None])
        self.window=torch.hann_window(2048,device=device)

    @torch.inference_mode()
    def __call__(self,waves):
        x=torch.as_tensor(waves,dtype=torch.float32,device=self.device).transpose(1,2)
        b,c,t=x.shape
        s=torch.stft(x.reshape(b*c,t),n_fft=2048,hop_length=1024,window=self.window,
                     center=True,pad_mode='constant',return_complex=True).reshape(b,c,1025,-1)[:,:,self.keep]
        energy=s.abs().square().sum((1,2))
        active=energy>energy.amax(1,keepdim=True)*.05
        crosses=[]
        for i,j in self.pairs:
            cross=s[:,i]*s[:,j].conj()
            cross=cross/cross.abs().clamp_min(1e-12)
            crosses.append((cross*active[:,None]).sum(-1)/active.sum(-1)[:,None].clamp_min(1))
        cross=torch.stack(crosses,1)
        scores=[];coherence=[]
        for band in self.bands:
            scores.append(torch.einsum('pdf,bpf->bd',self.steering[:,:,band],cross[:,:,band]).real/(6*band.sum()))
            coherence.append(cross[:,:,band].abs().mean((1,2)))
        return torch.stack(scores,1).cpu().numpy(),torch.stack(coherence,1).cpu().numpy()


class BandReliability(nn.Module):
    def __init__(self,adaptive=False):
        super().__init__();self.adaptive=adaptive
        self.logits=nn.Parameter(torch.zeros(len(BANDS)))
        self.adapter=nn.Sequential(nn.Linear(15,32),nn.Tanh(),nn.Linear(32,5)) if adaptive else None
        if adaptive:nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)

    def forward(self,scores,coherence):
        logits=self.logits[None].expand(len(scores),-1)
        if self.adapter is not None:
            stats=torch.cat([coherence,scores.amax(-1),scores.std(-1)],-1)
            logits=logits+self.adapter(stats)
        weights=logits.softmax(-1)
        return (scores*weights[:,:,None]).sum(1)/.05


def target_directions(rs):
    # Supervision/evaluation access, physically separate from waveform extractor.
    directions=[]
    for r in rs:
        meta=json.loads((D.DATA/r/'metadata.json').read_text())
        p=np.array(meta['relative_source']['position_lidar'],np.float32)
        directions.append(p/np.linalg.norm(p))
    return np.asarray(directions)


@torch.inference_mode()
def metrics(logits,truth):
    idx=logits.argmax(1).cpu().numpy()
    angle=np.degrees(np.arccos(np.clip((D.acoustic.DIRS[idx]*truth).sum(1),-1,1)))
    return dict(mean_degrees=float(angle.mean()),median_degrees=float(np.median(angle)),
                p90_degrees=float(np.percentile(angle,90)),fraction_within_10_degrees=float((angle<10).mean()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'COMPLETE.json').exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.manual_seed(1337)
    cal=json.loads((D.DATA/'metadata/microphone_array.json').read_text())
    extractor=BandSRP(cal['positions_lidar_m'],device=args.device)
    cache={};start=time.time()
    for split in ('train','val'):
        rs=D.rels(split);scores=[];coherences=[]
        def read(r):
            wave,rate=sf.read(D.DATA/r/'audio.wav',dtype='float32')
            if rate!=44100 or wave.shape[1]!=4:raise ValueError('Invalid physical microphone waveform')
            return wave
        with ThreadPoolExecutor(4) as pool:
            for i in range(0,len(rs),16):
                waves=np.stack(list(pool.map(read,rs[i:i+16])))
                score,coherence=extractor(waves);scores.append(score);coherences.append(coherence)
                if i%400==0:print(f'multiband {split} {i}/{len(rs)} {time.time()-start:.1f}s',flush=True)
        scores=np.concatenate(scores).astype(np.float32);coherence=np.concatenate(coherences).astype(np.float32)
        np.savez(args.output/f'{split}_observation_features.npz',rels=rs,scores=scores,coherence=coherence)
        truth=target_directions(rs)
        cache[split]=(torch.tensor(scores,device=args.device),torch.tensor(coherence,device=args.device),truth)
    train,tc,truth=cache['train'];val,vc,vtruth=cache['val']
    dirs=torch.tensor(D.acoustic.DIRS,dtype=torch.float32,device=args.device)
    # Smooth angular target, not a brittle one-bin direction label.
    target=(torch.tensor(truth,device=args.device)@dirs.T*80).softmax(-1)
    reports={}
    for adaptive in (False,True):
        key='adaptive' if adaptive else 'global'
        model=BandReliability(adaptive).to(args.device)
        opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.05)
        rng=np.random.default_rng(1337);best=-1;history=[]
        for epoch in range(1,101):
            model.train()
            for idx in np.array_split(rng.permutation(len(train)),13):
                ids=torch.tensor(idx,device=args.device)
                logits=model(train[ids],tc[ids])
                loss=-(target[ids]*logits.log_softmax(-1)).sum(1).mean()
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if epoch==1 or epoch%10==0:
                model.eval()
                result=metrics(model(val,vc),vtruth);train_result=metrics(model(train,tc),truth)
                row=dict(epoch=epoch,validation=result,train=train_result);history.append(row)
                print(key,json.dumps(row),flush=True)
                score=result['fraction_within_10_degrees']
                if score>best:
                    best=score
                    torch.save(dict(model=model.state_dict(),adaptive=adaptive,epoch=epoch),args.output/f'{key}.pt')
                    with torch.inference_mode():
                        for split,(s,c,_) in cache.items():
                            logits=model(s,c);post=(logits-logits.amax(1,keepdim=True)).exp().cpu().numpy()
                            np.save(args.output/f'{key}_{split}_posterior.npy',post)
        reports[key]=history
    base={}
    for split in ('train','val'):
        _,post,_,_=D.audio_bank(split)
        base[split]=metrics(torch.tensor(post),cache[split][2])
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(histories=reports,original_srp=base,
        test_used=False,bands=BANDS,calibration_sha256=D.sha(D.DATA/'metadata/microphone_array.json'),
        source_sha256=D.sha(Path(__file__))),indent=2))


if __name__=='__main__':main()
