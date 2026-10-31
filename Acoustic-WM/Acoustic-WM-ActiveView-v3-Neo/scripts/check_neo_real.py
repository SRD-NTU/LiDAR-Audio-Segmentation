"""Read-only CPU check of the frozen model's recurrent Neo path on TRAIN audio."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from activeview_v3.runtime import ROOT, ARCHIVE, samples, seed_all
from activeview_v3.inference import Adapter
from activeview_v3.multiview import SequentialSession
import numpy as np
import torch


def run():
    seed_all(7)
    sample = next(s for s in samples('train') if (ROOT/'data/features'/(s+'.npz')).exists())
    with np.load(ARCHIVE/'cache/reference'/(sample+'.npz')) as z:
        row = {k: z[k] for k in z.files}
    with np.load(ROOT/'data/features'/(sample+'.npz')) as z:
        actions, distances = z['actions'][:2], z['distance'][:2]
    with np.load(ROOT/'data/rendered'/(sample+'.npz')) as z:
        observed, weight = z['observed'][0], z['weight'][0]
    adapter = Adapter(ROOT/'checkpoints/world-s1337/world.pt', device='cpu')
    measured = SequentialSession(adapter, row)
    silent = SequentialSession(adapter, row)
    original, _ = adapter.prepare(row, actions[:1], distances[:1])
    first, _ = measured.prepare(None, actions[:1], distances[:1])
    for key in ('mu','var','latent','residual','sites'):
        torch.testing.assert_close(first[key], original[key], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(adapter.evidence.prior(first).softmax(-1),
                               adapter.evidence.prior(original).softmax(-1), atol=1e-5, rtol=1e-5)
    measured.observe(first, observed, weight, 'view-2')
    silent_first, _ = silent.prepare(None, actions[:1], distances[:1])
    silent.observe(silent_first, observed, np.zeros_like(weight), 'view-2')
    after = measured.prepare(None, actions[1:2], distances[1:2])[0]['mu']
    control = silent.prepare(None, actions[1:2], distances[1:2])[0]['mu']
    memory_delta = float((measured.memory-silent.memory).norm())
    forecast_delta = float((after-control).norm())
    if not np.isfinite(memory_delta+forecast_delta) or memory_delta <= 1e-5 or forecast_delta <= 1e-5:
        raise AssertionError('Real audio did not affect the next latent and forecast')
    print(dict(sample=sample, first_view_parity=True, memory_delta=memory_delta,
               next_forecast_delta=forecast_delta, recordings=measured.recordings))


if __name__ == '__main__':
    run()
