"""Four-channel spectral/temporal cues complement broad CLAP semantics."""
import numpy as np
import torch
from torch import nn
from scipy.signal import stft,resample_poly


def spectral_features(wave):
    if wave.ndim!=2 or wave.shape[1]!=4 or not np.isfinite(wave).all():
        raise ValueError('Expected finite four-channel audio')
    audio = resample_poly(wave,1,3,axis=0).astype(np.float32)
    channel_rms = np.sqrt(np.mean(audio**2,axis=0)+1e-10)
    rms = float(np.sqrt(np.mean(audio**2)+1e-10))
    frequency,_,spectrum = stft((audio/max(rms,1e-5)).T,fs=14700,nperseg=1024,noverlap=512)
    magnitude = np.abs(spectrum)
    edges = np.geomspace(20,7300,65)
    bands = []
    for left,right in zip(edges[:-1],edges[1:]):
        keep = (frequency>=left)&(frequency<right)
        # Empty low-frequency intervals share the nearest observable FFT bin.
        if not keep.any():
            keep[np.argmin(abs(frequency-np.sqrt(left*right)))] = True
        bands.append(np.log1p(100*magnitude[:,keep].mean(1)))
    bands = np.stack(bands,axis=1)
    level = np.log(channel_rms/max(float(channel_rms.mean()),1e-5))
    crest = np.log(np.max(np.abs(audio),axis=0)/np.maximum(channel_rms,1e-5)+1)
    result = np.concatenate((bands.mean(-1).ravel(),bands.std(-1).ravel(),level,
                             [np.log(max(rms,1e-5))],crest)).astype(np.float32)
    if result.shape!=(521,) or not np.isfinite(result).all():
        raise ValueError('Invalid spectral feature schema')
    return result


class SpectralAudioCategoryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.zeros(1033))
        self.register_buffer('scale',torch.ones(1033))
        self.net = nn.Sequential(nn.LayerNorm(1033),nn.Linear(1033,192),nn.GELU(),
                                  nn.Dropout(.3),nn.Linear(192,16))

    def forward(self,features):
        return self.net((features-self.mean)/self.scale.clamp_min(.01))
