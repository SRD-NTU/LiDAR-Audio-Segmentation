"""Label-free four-microphone SRP-PHAT direction features.

The array geometry and frequency range match this workspace's calibrated
Point-SAM audio-LiDAR implementation. Only observed audio is read here.
"""
import numpy as np
from scipy.signal import stft

POSITIONS = np.array([[.04, .04, 0], [.04, -.04, 0], [-.04, .04, 0], [-.04, -.04, 0]], dtype=np.float32)
ANGLES = np.arange(360, dtype=np.float32) * (np.pi / 180)
FREQUENCY = np.fft.rfftfreq(2048, 1 / 44100)
VALID_FREQUENCY = (FREQUENCY >= 150) & (FREQUENCY <= 3000)
FREQUENCY = FREQUENCY[VALID_FREQUENCY].astype(np.float32)
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
_DIRECTIONS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES), np.zeros(len(ANGLES)))).astype(np.float32)
STEERING = []
for i, j in PAIRS:
    delay = -(_DIRECTIONS @ (POSITIONS[i] - POSITIONS[j])) / 343.0
    STEERING.append(np.exp(2j * np.pi * delay[:, None] * FREQUENCY).astype(np.complex64))


def srp_features(audio, sample_rate=44100):
    """Return 360 azimuth likelihoods and peak coherence from 4-channel audio."""
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != 44100 or audio.shape != (88200, 4):
        raise ValueError("Expected two seconds of four-channel 44.1 kHz audio")
    _, _, spectrum = stft(audio.T, fs=sample_rate, nperseg=2048, noverlap=1024)
    spectrum = spectrum[:, VALID_FREQUENCY]
    energy = (spectrum.real ** 2 + spectrum.imag ** 2).sum(axis=(0, 1))
    active = energy > max(float(energy.max()) * .05, 1e-20)
    if not active.any():
        return np.ones(360, dtype=np.float32), 0.0
    spectrum = spectrum[:, :, active]
    score = np.zeros(360, dtype=np.float32)
    for (i, j), steering in zip(PAIRS, STEERING):
        cross = spectrum[i] * spectrum[j].conj()
        cross /= np.maximum(np.abs(cross), 1e-12)
        score += np.real(steering @ cross.mean(axis=1)).astype(np.float32) / len(FREQUENCY)
    score /= len(PAIRS)
    peak = float(score.max())
    likelihood = np.exp((score - peak) / .05).astype(np.float32)
    return likelihood, peak


def expand_likelihood(likelihood):
    """Interpolate 360 one-degree bins to the 1,800 LiDAR azimuth steps."""
    return np.interp(np.arange(1800, dtype=np.float32) / 5,
                     np.arange(361, dtype=np.float32),
                     np.r_[likelihood, likelihood[0]]).astype(np.float32)
