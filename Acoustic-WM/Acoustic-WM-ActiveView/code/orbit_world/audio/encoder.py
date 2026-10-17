"""Local-only CLAP loading: all model files are read from existing snapshots."""
from pathlib import Path
import math
import numpy as np
from scipy.signal import resample_poly
import torch


class ReadOnlyClap:
    def __init__(self,snapshot,device='cuda'):
        from transformers import ClapModel,ClapProcessor
        self.device=device
        self.processor=ClapProcessor.from_pretrained(str(snapshot),local_files_only=True)
        self.model=ClapModel.from_pretrained(str(snapshot),local_files_only=True).to(device).eval()

    @torch.inference_mode()
    def __call__(self,waveforms,sample_rate=44100):
        divisor=math.gcd(sample_rate,48000);waves=[]
        for wave in waveforms:
            mono=np.asarray(wave,np.float32).mean(1)
            mono=mono/max(float(np.sqrt(np.mean(mono**2))),1e-6)*.1
            waves.append(resample_poly(mono,48000//divisor,sample_rate//divisor).astype(np.float32))
        x=self.processor(audios=waves,sampling_rate=48000,return_tensors='pt').to(self.device)
        emb=torch.nn.functional.normalize(self.model.get_audio_features(**x),dim=-1)
        return emb


def find_clap_snapshot(orbit):
    cache=Path(orbit)/'pretrained/clap-hub/models--laion--clap-htsat-unfused'
    ref=cache/'refs/main'
    if not ref.exists():raise FileNotFoundError('Existing local CLAP snapshot required')
    snapshot=cache/'snapshots'/ref.read_text().strip()
    if not (snapshot/'config.json').exists():raise FileNotFoundError(snapshot)
    return snapshot
