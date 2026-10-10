"""AudioLiDAR observations with a label-free spatial audio map."""
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset

from .data import AudioLidarDataset, load_sample
from .spatial import expand_likelihood, srp_features


def load_sample_v2(root, sample, likelihood=None, coherence=None, labeled=True):
    item = load_sample(root, sample, labeled=labeled)
    if likelihood is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)
            rate, raw_audio = wavfile.read(Path(root) / sample / "audio.wav")
        likelihood, coherence = srp_features(raw_audio, rate)
    grid = item["grid"]
    valid = grid[4] > 0
    range_m = np.expm1(grid[3] * 4)
    previous_az = np.roll(range_m, 1, axis=1)
    previous_az_valid = np.roll(valid, 1, axis=1)
    previous_beam = np.roll(range_m, 1, axis=0)
    previous_beam_valid = np.roll(valid, 1, axis=0)
    previous_beam_valid[0] = False
    az_gradient = np.clip((range_m - previous_az) / 5, -2, 2) * (valid & previous_az_valid)
    beam_gradient = np.clip((range_m - previous_beam) / 5, -2, 2) * (valid & previous_beam_valid)
    direction = np.broadcast_to(expand_likelihood(likelihood)[None], (64, 1800))
    quality = np.full((64, 1800), float(coherence), dtype=np.float32)
    item["grid"] = np.ascontiguousarray(np.concatenate((grid, az_gradient[None], beam_gradient[None],
                                                          direction[None], quality[None]), axis=0), dtype=np.float32)
    return item


class AudioLidarV2Dataset(Dataset):
    def __init__(self, root, split, cache, augment=False):
        self.root = Path(root)
        self.samples = AudioLidarDataset(root, split).samples
        with np.load(Path(cache) / f"{split}.npz", allow_pickle=False) as archive:
            cached_names = archive["samples"].tolist()
            if cached_names != self.samples:
                raise ValueError(f"Spatial cache does not match {split} sample order")
            self.likelihood = archive["likelihood"].astype(np.float32)
            self.coherence = archive["coherence"].astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = load_sample_v2(self.root, self.samples[index], self.likelihood[index], self.coherence[index])
        grid, audio, target = item["grid"], item["audio"], item["target"]
        if self.augment:
            # Roll every spatial channel and the target together. The xyz and
            # audio coordinates stay physical; only the scan seam moves.
            shift = int(np.random.randint(0, 1800))
            grid = np.roll(grid, shift, axis=2).copy()
            target = np.roll(target, shift, axis=1).copy()
            audio = audio * np.float32(np.exp(np.random.uniform(-.5, .5)))
        return grid, np.ascontiguousarray(audio), target, item["sample"]
