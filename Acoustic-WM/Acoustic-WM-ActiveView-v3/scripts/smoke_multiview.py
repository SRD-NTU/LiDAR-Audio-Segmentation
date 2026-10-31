"""TRAIN-only real-renderer sequential integration and deployment parity."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.inference import Adapter, plan
from activeview_v3.multiview import SequentialSession
from activeview_v3.multiview_simulator import RouteSimulator
import numpy as np
import torch
import time
import argparse

parser=argparse.ArgumentParser();parser.add_argument('--revision',default='auditfix1');args=parser.parse_args()
seed_all(1337)
start = time.time()
protocol = json.loads((ROOT/'config'/('frozen_evaluation_'+args.revision+'.json')).read_text())
adapter = Adapter(ROOT/protocol['world'],evidence_path=ROOT/protocol['evidence'])
sample = sorted(samples('train'))[0]
with np.load(ARCHIVE/'cache/reference'/(sample+'.npz')) as z:
    row = {k: z[k] for k in z.files}
with np.load(ROOT/'data/features'/(sample+'.npz')) as z:
    actions, distances = z['actions'][:2], z['distance'][:2]
original, prepared = adapter.prepare(row, actions, distances)
session = SequentialSession(adapter, row)
cached, _ = session.prepare(None, actions, distances)
for key in ['mu', 'var', 'latent', 'residual', 'support', 'actions', 'overlap']:
    torch.testing.assert_close(cached[key], original[key], atol=1e-3, rtol=1e-3)
torch.testing.assert_close(adapter.evidence.prior(cached).softmax(-1), adapter.evidence.prior(original).softmax(-1), atol=1e-5, rtol=1e-5)
simulator = RouteSimulator('logs/multiview-'+args.revision+'-smoke-renderer.log',safe=True)
results = {}
try:
    listener = simulator.reference(sample)
    for method in ['world', 'frozen', 'random']:
        session = SequentialSession(adapter, row)
        replay = SequentialSession(adapter, row)
        floor = listener['floor_position_world']
        visited = [listener['position_world']]
        trace = []
        for views in range(2, 6):
            simulator.route(floor, visited)
            choice = plan(session, None, listener, simulator, method, seed=1337+views, **protocol['search'])
            pose, batch = choice['pose'], choice['batch']
            prefix = 'examples/multiview-'+args.revision+'-smoke/%s-%d' % (method, views)
            write_json(prefix+'-commit.json', {k:v for k,v in choice.items() if k not in ['batch', 'prepared']})
            receipt = simulator.acquire(pose, prefix+'-audio.npz')
            with np.load(receipt['output']) as z:
                observed, weight = z['observed'], z['weight']
            if views == 2:
                original, prepared = adapter.prepare(row, batch['actions'].cpu().numpy(), [pose['path_distance_m']])
                old = adapter.update(original, prepared, 0, observed, weight)
            prediction = session.observe(batch, observed, weight, str(views))
            if views == 2:
                np.testing.assert_array_equal(old['mask'], prediction['mask'])
                np.testing.assert_allclose(torch.as_tensor(old['scores']).softmax(-1).numpy(),
                                           torch.as_tensor(prediction['scores']).softmax(-1).numpy(), atol=1e-4, rtol=1e-4)
            replay_batch, _ = replay.prepare(None, batch['actions'].cpu().numpy(), [pose['path_distance_m']])
            replay_prediction = replay.observe(replay_batch, observed, weight, str(views))
            np.testing.assert_array_equal(prediction['mask'], replay_prediction['mask'])
            np.testing.assert_allclose(prediction['joint_log_probability'], replay_prediction['joint_log_probability'], atol=1e-6)
            assert len(prediction['mask']) == len(row['inverse'])
            assert session.recordings == views
            assert .15 <= pose['path_distance_m'] <= 3.000001
            assert pose['anchor_distance_m'] <= 3.000001
            assert np.linalg.norm(np.asarray(visited)-pose['position'], axis=1).min() >= .15
            floor = pose['floor']
            visited.append(pose['position'])
            trace.append(dict(views=views, candidate=prediction['candidate'], pose=pose,
                              original_point_count=len(prediction['mask'])))
            print(json.dumps(dict(method=method, views=views, seconds=round(time.time()-start, 1))), flush=True)
        results[method] = trace
finally:
    simulator.close()
for name, digest in protocol['checkpoints'].items():
    assert sha(ROOT/name) == digest
assert fingerprints() == protocol['singleview_checkpoints']
report = dict(passed=True, sample=sample, seconds=time.time()-start, first_acquisition_parity=True,
              posterior_replay_parity=True, original_point_order=True, frozen_checkpoints_unchanged=True,
              results=results)
write_json('reports/'+args.revision+'/multiview-smoke.json', report)
print(json.dumps(dict(passed=True, seconds=report['seconds'])), flush=True)
