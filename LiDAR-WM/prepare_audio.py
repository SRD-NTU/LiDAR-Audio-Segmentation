"""Cache frozen CLAP features from observation waveforms, for train and val only."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import ClapModel, ClapProcessor
from lidarwm.data import read_manifest, _read_float_wav, CATEGORIES

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default=os.environ.get('AUDIO_LIDAR_DATA_ROOT'))
    p.add_argument('--output',default='cache/clap'); p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args(); root=Path(args.root); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    model_id='laion/clap-htsat-unfused'
    processor=ClapProcessor.from_pretrained(model_id,cache_dir='pretrained/hub')
    model=ClapModel.from_pretrained(model_id,cache_dir='pretrained/hub').cuda().eval()
    descriptions={
      'refrigerator':'a refrigerator humming', 'fan':'an electric fan running',
      'tv':'a television playing speech and music', 'door':'a door opening or closing',
      'toilet':'a toilet flushing','washing machine':'a washing machine running',
      'vacuum cleaner':'a vacuum cleaner cleaning','clock':'a clock ticking',
      'microwave':'a microwave oven beeping and running','dishwasher':'a dishwasher running',
      'toaster':'a toaster clicking and popping','coffee machine':'a coffee machine brewing coffee',
      'speaker':'a loudspeaker playing music','printer':'a printer printing paper',
      'keyboard':'typing on a computer keyboard','kettle':'an electric kettle boiling water'}
    texts=[f'The sound of {descriptions[c]}.' for c in CATEGORIES]
    with torch.inference_mode():
        ti=processor(text=texts,padding=True,return_tensors='pt').to('cuda')
        text_features=model.get_text_features(**ti); text_features=torch.nn.functional.normalize(text_features,dim=-1)
    totals={}
    for split in ('train','val'):
        records=read_manifest(root,split); correct=0
        for start in range(0,len(records),args.batch_size):
            rr=records[start:start+args.batch_size]; paths=[output/(r.relative_path+'.npz') for r in rr]
            if all(x.exists() for x in paths):
                for r,x in zip(rr,paths):
                    with np.load(x) as z: correct+=int(z['text_logits'].argmax()==CATEGORIES.index(r.category))
                continue
            waves=[]
            for r in rr:
                rate,a=_read_float_wav(root/r.relative_path/'audio.wav')
                mono=a.mean(1); mono=mono/max(float(np.sqrt(np.mean(mono**2))),1e-6)*0.1
                waves.append(resample_poly(mono,160,147).astype('float32'))
            ai=processor(audios=waves,sampling_rate=48000,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                features=model.get_audio_features(**ai); features=torch.nn.functional.normalize(features,dim=-1)
                scores=features@text_features.T*model.logit_scale_a.exp()
            for r,path,f,s in zip(rr,paths,features.cpu().numpy(),scores.cpu().numpy()):
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez(path,embedding=f.astype('float32'),text_logits=s.astype('float32'))
                correct+=int(s.argmax()==CATEGORIES.index(r.category))
            print(json.dumps({'split':split,'done':min(start+len(rr),len(records)),'total':len(records),'zero_shot_accuracy':correct/(start+len(rr))}),flush=True)
        totals[split]=correct/len(records)
    (output/'provenance.json').write_text(json.dumps({'model':model_id,'classes':list(CATEGORIES),'descriptions':texts,'accuracy':totals,'inputs':'mono average of observed four-channel waveform; no labels passed to model'},indent=2))

if __name__=='__main__': main()
