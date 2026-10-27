"""Source-power normalized spatial covariance; physical four-microphone input.

X_m(f,t)=H_m(f)U(f,t) implies C_X/tr(C_X)=HH*/||H||² in the
noise-free, single-source narrowband limit. Finite windows, noise, and
reverberation break exact invariance; the evaluation measures this gap.
"""
import numpy as np
from scipy.signal import stft

RATE = 44100
NFFT = 4096
BANDS = 12
FREQUENCIES = np.fft.rfftfreq(NFFT, 1 / RATE)
KEEP = (FREQUENCIES >= 150) & (FREQUENCIES <= 3000)
FREQUENCIES = FREQUENCIES[KEEP]
EDGES = np.linspace(150, 3000.01, BANDS + 1)
POOL = np.stack([((FREQUENCIES >= lo) & (FREQUENCIES < hi)).astype(np.float32)
                 for lo, hi in zip(EDGES[:-1], EDGES[1:])])
POOL /= POOL.sum(1, keepdims=True)


def pack(z):
    return np.stack([z.real, z.imag], -1).astype(np.float32)


def covariance_from_wave(wave, rate=RATE):
    """Return [band,4,4,real/imag] covariance and band reliability weights."""
    wave = np.asarray(wave)
    if rate != RATE or wave.ndim != 2 or wave.shape[1] != 4 or len(wave) < NFFT:
        raise ValueError('Expected >=4096 samples, 44100 Hz, four calibrated channels')
    if not np.isfinite(wave).all():
        raise ValueError('Nonfinite microphone input')
    _, _, x = stft(wave.T, fs=rate, nperseg=NFFT, noverlap=NFFT//2, boundary=None)
    x = x[:, KEEP]
    c = np.einsum('mft,nft->fmn', x, x.conj(), optimize=True) / x.shape[-1]
    power = np.trace(c, axis1=-2, axis2=-1).real
    c /= np.maximum(power[:, None, None], 1e-20)
    q = np.einsum('bf,fmn->bmn', POOL, c)
    energy = POOL @ power
    valid = energy > max(float(energy.max()) * 1e-4, 1e-20)
    weight = valid.astype(np.float32)
    weight /= max(weight.sum(), 1)
    return pack(q), weight


def covariance_from_rir(rir):
    """True transfer supervision, never an inference input.

    Zero-pad to a multiple of NFFT so sampled frequencies match observations.
    Do not truncate the 2-second response to a short STFT window.
    """
    rir = np.asarray(rir)
    if rir.ndim != 2 or rir.shape[0] != 4 or not np.isfinite(rir).all():
        raise ValueError('Expected finite RIR [4,time]')
    if (np.max(np.abs(rir), axis=1) <= 1e-12).any():
        raise ValueError('Empty RIR channel')
    multiple = int(np.ceil(rir.shape[1] / NFFT))
    h = np.fft.rfft(rir, n=multiple*NFFT, axis=-1)[:, ::multiple][:, KEEP].T
    q = h[:, :, None] * h[:, None, :].conj()
    q /= np.maximum((np.abs(h)**2).sum(-1)[:,None,None], 1e-20)
    return pack(np.einsum('bf,fmn->bmn', POOL, q))
