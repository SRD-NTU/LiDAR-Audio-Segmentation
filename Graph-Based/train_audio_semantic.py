"""Train audio category prediction; cache predictions, never category labels."""
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from graph_audio.audio_semantic import AudioCategoryHead
from graph_audio.features import CATEGORIES
from prepare import ROOT,DATASET,CLAP


def main():
    torch.set_num_threads(2); torch.manual_seed(1337); np.random.seed(1337)
    with (DATASET/'metadata/samples.csv').open() as handle:
        metadata = {r['sample']:r for r in csv.DictReader(handle)}
    names = {s:json.loads((DATASET/f'metadata/{s}.json').read_text()) for s in ('train','val')}
    features,targets = {},{}
    for split in names:
        features[split] = torch.from_numpy(np.stack([np.load(CLAP/(n+'.npz'))['embedding'] for n in names[split]])).float()
        targets[split] = torch.tensor([CATEGORIES.index(metadata[n]['category']) for n in names[split]])
    model = AudioCategoryHead()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.03)
    run = ROOT/'runs/audio-category'
    run.mkdir(parents=True,exist_ok=True)
    best_accuracy = -1.; best_loss = float('inf')
    for epoch in range(100):
        model.train()
        order = torch.randperm(len(features['train']))
        for ids in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features['train'][ids]),targets['train'][ids],label_smoothing=.05)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(features['val'])
            accuracy = float((logits.argmax(-1)==targets['val']).float().mean())
            validation_loss = float(F.cross_entropy(logits,targets['val']))
        if accuracy>best_accuracy or (accuracy==best_accuracy and validation_loss<best_loss):
            best_accuracy,best_loss = accuracy,validation_loss
            torch.save(dict(model=model.state_dict(),epoch=epoch,validation_accuracy=accuracy,
                            training_samples=len(names['train']),selection_split='val'),run/'best.pt')
        if epoch%10==0 or epoch==99:
            print(json.dumps(dict(epoch=epoch,validation_accuracy=accuracy,best_accuracy=best_accuracy)),flush=True)
    checkpoint = torch.load(run/'best.pt',weights_only=False)
    model.load_state_dict(checkpoint['model']); model.eval()
    digest = hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
    cache = ROOT/'cache/audio_category'
    cache.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for split in names:
            logits = model(features[split]).numpy()
            for name,value in zip(names[split],logits):
                path = cache/(name+'.npz'); path.parent.mkdir(parents=True,exist_ok=True)
                temporary = path.with_suffix('.tmp.npz')
                np.savez(temporary,logits=value,model_sha256=digest)
                os.replace(temporary,path)
    (cache/'ready.json').write_text(json.dumps(dict(samples=sum(map(len,names.values())),model_sha256=digest)))
    with torch.inference_mode():
        prediction = model(features['val']).argmax(-1)
    report = dict(split='val',samples=len(names['val']),training_samples=len(names['train']),
        accuracy=best_accuracy,epoch=checkpoint['epoch'],model_sha256=digest,
        per_category={c:float((prediction[targets['val']==i]==i).float().mean()) for i,c in enumerate(CATEGORIES)},
        no_segmentation_metric=True,test_used_for_selection=False,
        note='Audio category accuracy is not segmentation IoU. Only training category labels optimize weights.')
    (ROOT/'reports/learned_audio_semantics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
