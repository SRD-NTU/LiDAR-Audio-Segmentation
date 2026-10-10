"""Frozen audio features shared by preprocessing and deployable inference."""
import numpy as np
import torch
from scipy.signal import resample_poly
from .data import CATEGORIES

DESCRIPTIONS={
 'refrigerator':'a refrigerator humming','fan':'an electric fan running',
 'tv':'a television playing speech and music','door':'a door opening or closing',
 'toilet':'a toilet flushing','washing machine':'a washing machine running',
 'vacuum cleaner':'a vacuum cleaner cleaning','clock':'a clock ticking',
 'microwave':'a microwave oven beeping and running','dishwasher':'a dishwasher running',
 'toaster':'a toaster clicking and popping','coffee machine':'a coffee machine brewing coffee',
 'speaker':'a loudspeaker playing music','printer':'a printer printing paper',
 'keyboard':'typing on a computer keyboard','kettle':'an electric kettle boiling water'}

class FrozenClap:
    def __init__(self,device='cuda'):
        from transformers import ClapModel,ClapProcessor
        self.device=device
        self.processor=ClapProcessor.from_pretrained('laion/clap-htsat-unfused',cache_dir='pretrained/hub')
        self.model=ClapModel.from_pretrained('laion/clap-htsat-unfused',cache_dir='pretrained/hub').to(device).eval()
        texts=[f'The sound of {DESCRIPTIONS[c]}.' for c in CATEGORIES]
        with torch.inference_mode():
            inp=self.processor(text=texts,padding=True,return_tensors='pt').to(device)
            self.text_features=torch.nn.functional.normalize(self.model.get_text_features(**inp),dim=-1)

    @torch.inference_mode()
    def __call__(self,waveforms,sample_rate=44100):
        waves=[]
        import math
        divisor=math.gcd(sample_rate,48000)
        for audio in waveforms:
            mono=np.asarray(audio,dtype='float32').mean(axis=1)
            mono=mono/max(float(np.sqrt(np.mean(mono**2))),1e-6)*.1
            waves.append(resample_poly(mono,48000//divisor,sample_rate//divisor).astype('float32'))
        inp=self.processor(audios=waves,sampling_rate=48000,return_tensors='pt').to(self.device)
        embedding=torch.nn.functional.normalize(self.model.get_audio_features(**inp),dim=-1)
        text_logits=embedding@self.text_features.T*self.model.logit_scale_a.exp()
        return embedding,text_logits
