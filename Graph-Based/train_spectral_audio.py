"""Supervised audio adaptation with raw four-channel spectral/temporal cues."""
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.nn import functional as F
from prepare import ROOT,DATASET,CLAP
from graph_audio.features import CATEGORIES,waveform
from graph_audio.spectral_semantics import spectral_features,SpectralAudioCategoryHead


def observation(name):
    path = ROOT/'cache/audio_spectrum'/(name+'.npz')
    if path.exists():
        with np.load(path) as a:
            values = a['features']
    else:
        audio_path = DATASET/name/'audio.wav'
        values = spectral_features(waveform(audio_path))
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,features=values,audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest())
    with np.load(CLAP/(name+'.npz')) as a:
        return np.concatenate((a['embedding'],values)).astype(np.float32)


def main():
    torch.set_num_threads(2); torch.manual_seed(1337)
    with (DATASET/'metadata/samples.csv').open() as f:
        metadata = {r['sample']:r for r in csv.DictReader(f)}
    names = {s:json.loads((DATASET/f'metadata/{s}.json').read_text()) for s in ('train','val')}
    features,targets = {},{}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for split in names:
            values = []
            for i,v in enumerate(pool.map(observation,names[split]),1):
                values.append(v)
                if i%256==0 or i==len(names[split]):
                    print(f'Four-channel spectral observations {split}: {i}/{len(names[split])}',flush=True)
            features[split] = torch.from_numpy(np.stack(values))
            targets[split] = torch.tensor([CATEGORIES.index(metadata[n]['category']) for n in names[split]])
    model = SpectralAudioCategoryHead()
    model.mean.copy_(features['train'].mean(0)); model.scale.copy_(features['train'].std(0).clamp_min(.01))
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.05)
    run = ROOT/'runs/audio-category-spectral'; run.mkdir(parents=True,exist_ok=True)
    best=(-1.,float('inf'))
    for epoch in range(100):
        model.train()
        for ids in torch.randperm(len(features['train'])).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features['train'][ids]),targets['train'][ids],label_smoothing=.05)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits=model(features['val'])
            accuracy=float((logits.argmax(-1)==targets['val']).float().mean())
            ce=float(F.cross_entropy(logits,targets['val']))
        if accuracy>best[0] or (accuracy==best[0] and ce<best[1]):
            best=(accuracy,ce)
            torch.save(dict(model=model.state_dict(),epoch=epoch,architecture='spectral',
                            validation_accuracy=accuracy,training_samples=len(names['train'])),run/'best.pt')
        if epoch%10==0 or epoch==99:
            print(json.dumps(dict(epoch=epoch,validation_accuracy=accuracy,best_accuracy=best[0])),flush=True)
    state=torch.load(run/'best.pt',weights_only=False); model.load_state_dict(state['model']); model.eval()
    digest=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
    cache=ROOT/'cache/audio_category_spectral'; cache.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for split in names:
            logits=model(features[split]).numpy()
            for name,value in zip(names[split],logits):
                path=cache/(name+'.npz'); path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix('.tmp.npz')
                np.savez(temporary,logits=value,model_sha256=digest)
                os.replace(temporary,path)
    (cache/'ready.json').write_text(json.dumps(dict(samples=4032,model_sha256=digest,architecture='spectral')))
    with torch.inference_mode():
        train_accuracy=float((model(features['train']).argmax(-1)==targets['train']).float().mean())
        prediction=model(features['val']).argmax(-1)
    report=dict(split='val',samples=768,training_samples=3264,accuracy=best[0],training_accuracy=train_accuracy,
        epoch=state['epoch'],model_sha256=digest,architecture='spectral',test_used=False,no_segmentation_metric=True,
        per_category={c:float((prediction[targets['val']==i]==i).float().mean()) for i,c in enumerate(CATEGORIES)})
    (ROOT/'reports/spectral_audio_semantics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
