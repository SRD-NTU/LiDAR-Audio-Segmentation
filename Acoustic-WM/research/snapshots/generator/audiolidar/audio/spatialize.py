"""Convolution preserving binaural inter-channel level differences."""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve, resample_poly
from math import gcd
import soundfile as sf


def dry_wave(path, sample_rate: int, duration: float) -> np.ndarray:
    x, rate = sf.read(path, dtype='float64', always_2d=True)
    if not np.isfinite(x).all() or not x.size: raise ValueError('Invalid dry audio')
    x = x.mean(axis=1)
    if rate != sample_rate:
        divisor = gcd(rate, sample_rate)
        x = resample_poly(x, sample_rate//divisor, rate//divisor)
    n = round(sample_rate*duration)
    if len(x)<n: x=np.pad(x,(0,n-len(x)))
    # Choose the strongest window for sparse events, deterministically.
    elif len(x)>n:
        energy=np.r_[0.,np.cumsum(x*x)]
        start=int(np.argmax(energy[n:]-energy[:-n])); x=x[start:start+n]
    if np.max(np.abs(x))<1e-8: raise ValueError('Silent dry waveform')
    return x


def convolve(dry: np.ndarray, rir: np.ndarray, length: int) -> tuple:
    if rir.ndim!=2 or rir.shape[0] not in (2,4) or not np.isfinite(rir).all():
        raise ValueError('Invalid spatial RIR')
    if np.max(np.abs(rir))<1e-12: raise ValueError('Silent RIR')
    wave=np.stack([fftconvolve(dry,h)[:length] for h in rir],axis=1)
    peak=float(np.max(np.abs(wave)))
    if peak<1e-12: raise ValueError('Silent spatialized waveform')
    gain=min(1.,.95/peak)
    return (wave*gain).astype('float32'), gain
