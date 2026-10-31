"""Persistent, scene-sharded 3/4/5-view evaluation on the original point cloud."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from activeview_v3.runtime import *
from activeview_v3.inference import Adapter, plan
from activeview_v3.multiview import SequentialSession
from activeview_v3.multiview_simulator import RouteSimulator
from activeview_v3.navigation import relative_pose
from activeview_v3.provenance import verify_manifest
import argparse
import numpy as np
import time
import fcntl
import subprocess


def protocol_check(protocol_path):
    protocol = json.loads((ROOT/protocol_path).read_text())
    if protocol.get('memory_update') != 'trained_gate_on_frozen_singleview_gru_at_acquired_pose_v1':
        raise RuntimeError('Neo evaluation requires a frozen recurrent-memory protocol')
    if 'provenance' not in protocol:
        raise RuntimeError('Legacy frozen protocol requires its archived evaluator')
    content = {k: v for k, v in protocol.items() if k != 'protocol_sha256'}
    if hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() != protocol['protocol_sha256']:
        raise RuntimeError('Protocol digest mismatch')
    verify_manifest(protocol,check_dependencies=True,learning_python=sys.executable,
                    renderer_python=UTILITY/'environments/audiolidar/bin/python')
    if sha(ROOT/protocol['renderer_server'])!=protocol['renderer_sha256']:
        raise RuntimeError('Frozen renderer changed')
    for name, digest in protocol['checkpoints'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Frozen input changed: '+name)
    if fingerprints() != protocol['singleview_checkpoints']:
        raise RuntimeError('SingleView checkpoints changed')
    return protocol


def read_record(path, protocol):
    value = json.loads(path.read_text())
    if value['protocol_sha256'] != protocol['protocol_sha256']:
        raise RuntimeError('Stale output: '+str(path))
    expected={name:protocol['provenance'][name] for name in
              ['source_manifest_sha256','config_manifest_sha256','dependency_manifest_sha256']}
    if value.get('provenance')!=expected:
        raise RuntimeError('Provenance mismatch: '+str(path))
    return value


def run(args):
    protocol = protocol_check(args.protocol)
    provenance = {name: protocol['provenance'][name] for name in
                  ['source_manifest_sha256','config_manifest_sha256','dependency_manifest_sha256']}
    methods = args.methods.split(',') if args.methods else protocol['methods']
    if any(m not in protocol['methods'] for m in methods):
        raise ValueError('Unregistered method')
    if args.split == 'test' and (args.limit or args.methods):
        raise ValueError('TEST uses all methods and all references in each shard')
    rows = samples(args.split)
    scenes = sorted({r.split('/')[1] for r in rows})[args.rank::args.workers]
    rows = sorted(r for r in rows if r.split('/')[1] in scenes)
    if args.limit:
        rows = rows[:args.limit]
    adapter = Adapter(ROOT/protocol['world'],evidence_path=ROOT/protocol['evidence'],
                      memory_filter_path=ROOT/protocol['memory_filter'])
    simulator = None
    scene = None
    start = time.time()
    completed = 0
    identity = dict(split=args.split, rank=args.rank, workers=args.workers,
                    protocol_sha256=protocol['protocol_sha256'], job_id=os.getenv('SLURM_JOB_ID'))
    def stop_requested():
        if args.stop_file and output_path(args.stop_file).exists():
            status = dict(identity, stopped_for_handoff=True, completed=completed, total=len(rows), seconds=round(time.time()-start, 1))
            write_json('reports/'+protocol['revision']+'/multiview-%s-%d-handoff.json' % (args.split, args.rank), status)
            print(json.dumps(status), flush=True)
            return True
        return False
    try:
        for sample in rows:
            if stop_requested():
                return
            directory = ROOT/protocol['results_root']/sample
            marker = directory/'complete.json'
            if marker.exists():
                previous = read_record(marker, protocol)
                if previous['methods'] == methods:
                    completed += 1
                    continue
            new_scene = sample.split('/')[1]
            if simulator is None or new_scene != scene:
                if simulator is not None:
                    simulator.close()
                simulator = RouteSimulator('logs/multiview-%s-renderer-%s-%d.log' % (protocol['revision'],args.split, args.rank),safe=True)
                scene = new_scene
            reference_path=ARCHIVE/'cache/reference'/(sample+'.npz')
            reference_sha256=sha(reference_path)
            with np.load(reference_path) as z:
                row = {k: z[k] for k in z.files}
            listener = simulator.reference(sample)
            seed_key = json.dumps(dict(listener=listener, scene=scene, seed=protocol['seed']), sort_keys=True)
            seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
            for method in methods:
                summary = directory/method/'complete.json'
                if summary.exists():
                    read_record(summary, protocol)
                    continue
                session = SequentialSession(adapter, row)
                current_floor = listener['floor_position_world']
                visited = [listener['position_world']]
                travel = 0.
                step_results = []
                for views in range(2, protocol['max_views']+1):
                    prefix = directory/method/('view-%d' % views)
                    commit_path = prefix.with_suffix('.commit.json')
                    receipt_path = prefix.with_suffix('.receipt.json')
                    result_path = prefix.with_suffix('.json')
                    observation_path = prefix.with_suffix('.audio.npz')
                    prediction_path = prefix.with_suffix('.prediction.npz')
                    simulator.route(current_floor, visited)
                    if commit_path.exists():
                        commit = read_record(commit_path, protocol)
                        if (commit['visited_positions'] != visited or commit['current_floor'] != current_floor or
                            commit['sample']!=sample or commit['reference_sha256']!=reference_sha256):
                            raise RuntimeError('Inconsistent committed history')
                        pose = commit['pose']
                        action = relative_pose(np.asarray(listener['position_world']), listener['yaw_rad'], pose['position'], pose['yaw'])
                        batch, _ = session.prepare(None, action[None], [pose['path_distance_m']])
                    else:
                        name = 'random' if method.startswith('random-') else method
                        trial = int(method.split('-')[1]) if name == 'random' else 0
                        step_seed = (seed + trial + (views-2)*1000003) % (2**32)
                        choice = plan(session, None, listener, simulator, name, seed=step_seed, **protocol['search'])
                        batch = choice['batch']
                        commit = {k: v for k, v in choice.items() if k not in ['batch', 'prepared']}
                        pose = choice['pose']
                        commit.update(method=method,sample=sample,reference_sha256=reference_sha256,
                                      seed=step_seed, views=views,
                                      current_floor=current_floor, visited_positions=list(visited),
                                      protocol_sha256=protocol['protocol_sha256'],provenance=provenance)
                        write_json(commit_path, commit)
                    if receipt_path.exists():
                        receipt = read_record(receipt_path, protocol)
                        if (receipt['observation_sha256'] != sha(observation_path) or
                            receipt['commit_sha256'] != sha(commit_path) or
                            receipt['renderer_sha256']!=protocol['renderer_sha256']):
                            raise RuntimeError('Acquisition receipt mismatch')
                    else:
                        receipt = simulator.acquire(pose, observation_path)
                        receipt.update(protocol_sha256=protocol['protocol_sha256'], observation_sha256=sha(observation_path),
                                       commit_sha256=sha(commit_path),provenance=provenance,
                                       renderer_server=protocol['renderer_server'],
                                       renderer_sha256=protocol['renderer_sha256'])
                        write_json(receipt_path, receipt)
                    with np.load(observation_path) as z:
                        prediction = session.observe(batch, z['observed'], z['weight'], '%s/%s/%d' % (sample, method, views))
                    travel += pose['path_distance_m']
                    if (pose['path_distance_m'] > protocol['max_leg_distance_m']+1e-6
                            or pose['anchor_distance_m'] > protocol['anchor_radius_m']+1e-6
                            or travel > (views-1)*protocol['max_leg_distance_m']+1e-6
                            or np.min(np.linalg.norm(np.asarray(visited)-pose['position'], axis=1)) < protocol['minimum_position_separation_m']):
                        raise RuntimeError('Acquisition violates frozen movement protocol')
                    if result_path.exists():
                        result = read_record(result_path, protocol)
                        with np.load(prediction_path) as z:
                            np.testing.assert_allclose(prediction['joint_log_probability'], z['joint_log_probability'], atol=1e-5, rtol=1e-5)
                            np.testing.assert_array_equal(prediction['mask'], z['mask'])
                        if sha(prediction_path) != result['prediction_sha256']:
                            raise RuntimeError('Saved prediction changed')
                    else:
                        # Privileged scoring occurs only after acquisition and segmentation.
                        with np.load(ARCHIVE/'cache/controlled'/(sample+'.npz')) as z:
                            iou, null_iou = z['iou'], float(z['null_iou'])
                        candidate = prediction['candidate']
                        quality = null_iou if candidate < 0 else float(iou[candidate])
                        baseline = null_iou if session.state['silent'] else float(iou[int(session.state['base'].argmax())])
                        temporary = output_path(prediction_path).with_suffix('.partial.npz')
                        np.savez_compressed(temporary, **prediction)
                        temporary.replace(prediction_path)
                        result = dict(sample=sample, method=method, recordings=views, lidar_scans=1, candidate=candidate,
                                      iou=quality, singleview_iou=baseline, pose=pose, cumulative_travel_m=travel,
                                      predicted_utility=commit['utility'], evaluated_poses=commit['evaluated_poses'],
                                      original_point_count=len(prediction['mask']), nonzero_channels=receipt['nonzero_channels'],
                                      observation_sha256=receipt['observation_sha256'], prediction_sha256=sha(prediction_path),
                                      protocol_sha256=protocol['protocol_sha256'],provenance=provenance,
                                      receipt_sha256=sha(receipt_path),commit_sha256=sha(commit_path),
                                      renderer_sha256=protocol['renderer_sha256'])
                        write_json(result_path, result)
                    step_results.append(result)
                    current_floor = pose['floor']
                    visited.append(pose['position'])
                    print(json.dumps(dict(event='view_complete', sample=sample, method=method, views=views,
                                          cumulative_travel_m=round(travel, 3))), flush=True)
                    write_json('reports/'+protocol['revision']+'/multiview-%s-%d-progress.json' % (args.split, args.rank),
                               dict(identity, completed=completed, total=len(rows), current_sample=sample,
                                    current_method=method, latest_recordings=views, seconds=round(time.time()-start, 1)))
                    if stop_requested():
                        return
                write_json(summary, dict(sample=sample, method=method, complete=True,
                                         views=[r['recordings'] for r in step_results], protocol_sha256=protocol['protocol_sha256'],provenance=provenance))
            write_json(marker, dict(sample=sample, methods=methods, complete=True, protocol_sha256=protocol['protocol_sha256'],provenance=provenance))
            completed += 1
            status = dict(identity, completed=completed, total=len(rows), seconds=round(time.time()-start, 1), last_sample=sample)
            print(json.dumps(status), flush=True)
            write_json('reports/'+protocol['revision']+'/multiview-%s-%d-progress.json' % (args.split, args.rank), status)
        write_json('reports/'+protocol['revision']+'/multiview-%s-%d-complete.json' % (args.split, args.rank),
                   dict(identity, complete=True, rows=len(rows), methods=methods,provenance=provenance))
    finally:
        if simulator is not None:
            simulator.close()
    if args.split == 'test':
        # Report incomplete status until every reference/method/budget exists.
        with open(output_path('runtime/'+protocol['revision']+'-multiview-report.lock'), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            subprocess.run([sys.executable, '-B', str(ROOT/'scripts/report_multiview.py'), '--split', args.split,
                            '--protocol',args.protocol], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--protocol',default='config/frozen_multiview_neo1.json')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--methods', default='')
    p.add_argument('--stop-file', default='', help='Graceful allocation handoff after a durable observation step')
    args = p.parse_args()
    if args.workers < 1 or not 0 <= args.rank < args.workers:
        p.error('Invalid shard')
    seed_all(1337)
    run(args)
