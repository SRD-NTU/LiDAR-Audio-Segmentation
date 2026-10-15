"""Fixed CLAP text priors derived from TRAIN event descriptions, never val/test audio."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from apex import data as D

# Train manifest documents knocking (door), static (TV), and feedback/hum
# (speaker). Do not inherit the old speech/music descriptions for these classes.
EVENTS = {
    'refrigerator':['a refrigerator compressor humming','a fridge motor running'],
    'fan':['an electric fan running','a ventilation fan blowing air'],
    'tv':['television static noise','digital static crackling from a television'],
    'door':['knocking on a wooden door','a person knocking on a door'],
    'toilet':['a toilet flushing','water rushing during a toilet flush'],
    'washing machine':['a washing machine running','a washing machine spinning clothes'],
    'vacuum cleaner':['a vacuum cleaner running','vacuum cleaning a floor'],
    'clock':['a clock ticking','a mechanical clock ticking rhythmically'],
    'microwave':['a microwave oven humming and beeping','a microwave oven running'],
    'dishwasher':['a dishwasher washing dishes','muffled water spraying inside a dishwasher'],
    'toaster':['a toaster popping up','a toaster clicking and popping'],
    'coffee machine':['a coffee machine brewing coffee','an espresso machine pumping water'],
    'speaker':['a loudspeaker buzzing with electrical feedback','computer speakers humming'],
    'printer':['a printer printing paper','a printer starting up with mechanical whirring'],
    'keyboard':['typing on a computer keyboard','computer keyboard keys clicking'],
    'kettle':['a kettle boiling water','water bubbling in an electric kettle'],
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4)
    from transformers import ClapModel,ClapProcessor
    cache=str(D.ROOT/'ORBIT/pretrained/clap-hub')
    processor=ClapProcessor.from_pretrained('laion/clap-htsat-unfused',cache_dir=cache,local_files_only=True)
    model=ClapModel.from_pretrained('laion/clap-htsat-unfused',cache_dir=cache,local_files_only=True).to(args.device).eval()
    text=[f'The sound of {event}.' for c in D.CATS for event in EVENTS[c]]
    with torch.inference_mode():
        inp=processor(text=text,padding=True,return_tensors='pt').to(args.device)
        feat=F.normalize(model.get_text_features(**inp),dim=-1).reshape(16,2,-1).mean(1)
        feat=F.normalize(feat,dim=-1)
        scale=float(model.logit_scale_a.exp())
    features=feat.cpu().numpy();scores={}
    for split in ('train','val'):
        rows=torch.load(args.cache/f'perfect_objects_{split}.pt',weights_only=False)
        emb=np.stack([r['audio'] for r in rows]);labels=np.array([r['source'] for r in rows])
        logits=emb@features.T*scale
        scores[split]=dict(n=len(rows),audio_category_accuracy=float(np.mean(logits.argmax(1)==labels)),
                          by_category={c:float(np.mean(logits.argmax(1)[labels==k]==k)) for k,c in enumerate(D.CATS)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,text_features=features,logit_scale=scale,categories=D.CATS)
    report=dict(prompts=text,prompt_design='Fixed from training manifest descriptions only',
                scores=scores,test_used=False,weights='existing local laion/clap-htsat-unfused',
                source_sha256=D.sha(Path(__file__)))
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
