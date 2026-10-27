"""Calibrate prompted mask thresholds; all results explicitly privileged."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from apex import data as D
from .masks import PromptMasks
from . import train_masks as trainer


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=PromptMasks().cuda().eval();model.load_state_dict(ck['model'])
    trainer.THRESHOLDS=np.array([.2,.35,.5,.65,.8,.9,.95,.975,.99],np.float32)
    rows=trainer.load_split(args.cache,'val')
    result=trainer.evaluate(model,rows,'cuda')
    result['checkpoint_sha256']=D.sha(args.checkpoint)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2),flush=True)


if __name__=='__main__':main()
