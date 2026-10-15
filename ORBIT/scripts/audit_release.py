"""Check the frozen ORBIT result without loading labels or running a model."""
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def within_root(path):
    path = Path(path)
    return ROOT / Path(*path.parts[1:]) if path.parts and path.parts[0] == 'ORBIT' else ROOT / path


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(evaluator_source=None):
    freeze = json.loads((ROOT / 'reports/final_method_frozen.json').read_text())
    validation_path = within_root(freeze['selected_validation_report'])
    check(digest(validation_path) == freeze['selected_validation_report_sha256'],
          'Frozen validation report hash differs')
    evaluator_source = ROOT / 'orbit/evaluate_audio_gate.py' if evaluator_source is None else Path(evaluator_source)
    check(digest(evaluator_source) == freeze['evaluator_source_sha256'],
          'Frozen evaluator source differs; use --evaluator-source pointing to the archived original source')
    check(digest(within_root(freeze['test_posterior'])) == freeze['test_posterior_sha256'],
          'Frozen TEST posterior hash differs')
    validation = json.loads(validation_path.read_text())
    test = json.loads((ROOT / 'reports/final_test.json').read_text())
    check(freeze['selected_rule'] == 'fused' and freeze['selection_split'] == 'val',
          'Unexpected frozen selection rule or split')
    check(not freeze['test_performance_seen_at_freeze'], 'Freeze records prior TEST access')
    for name, report in [('validation', validation), ('TEST', test)]:
        split = 'val' if name == 'validation' else 'test'
        check(report['split'] == split and report['full_split'] and report['control'] == 'matched',
              f'{name} is not a complete matched {split} report')
        check(report['test_used'] == (split == 'test'),
              f'{name} TEST access flag disagrees with its split')
        check(len(report['samples']) == 768, f'{name} does not contain 768 observations')
        check(len({row['rel'] for row in report['samples']}) == 768,
              f'{name} has duplicate observations')
        check(report['gate_weight'] == freeze['gate_weight'] and
              report['gate2_weight'] == freeze['gate2_weight'],
              f'{name} ensemble weights differ from freeze')
        check(report['source_sha256'] == freeze['evaluator_source_sha256'],
              f'{name} evaluator differs from freeze')
        for key, expected in freeze['checkpoints'].items():
            record = report['checkpoints'][key]
            check(record['sha256'] == expected and digest(within_root(record['path'])) == expected,
                  f'{name} {key} checkpoint differs from freeze')
        mean = statistics.fmean(row['fused'] for row in report['samples'])
        check(abs(mean - report['metrics']['fused']['mean_iou']) < 1e-12,
              f'{name} fused sample mean differs from summary')
    check(abs(validation['metrics']['fused']['mean_iou'] -
              freeze['selected_validation_mean_iou']) < 1e-12,
          'Frozen validation score differs')
    check(test['provenance']['replacement_posterior_sha256'] ==
          freeze['test_posterior_sha256'], 'TEST report used another posterior')
    check(test['geometry'] == str(within_root(freeze['test_geometry']).relative_to(ROOT)),
          'TEST report used another geometry cache')
    train_rels = json.loads((ROOT / 'cache/aligned-utonia-v1/train_manifest.json').read_text())
    val_rels = json.loads((ROOT / 'cache/aligned-utonia-v1/val_manifest.json').read_text())
    test_rels = json.loads((ROOT / 'cache/utonia-heldout-test-v1/rels.json').read_text())
    check([row['rel'] for row in validation['samples']] == val_rels,
          'Validation report order differs from the cached observation order')
    check([row['rel'] for row in test['samples']] == test_rels,
          'TEST report order differs from the frozen posterior observation order')
    train_scenes = {rel.split('/')[1] for rel in train_rels}
    val_scenes = {row['scene'] for row in validation['samples']}
    test_scenes = {row['scene'] for row in test['samples']}
    check(train_scenes.isdisjoint(val_scenes) and
          train_scenes.isdisjoint(test_scenes) and val_scenes.isdisjoint(test_scenes),
          'TRAIN, validation, and TEST scenes overlap')
    raw = json.loads((ROOT / 'reports/utonia_dualgate_raw_check.json').read_text())
    check(raw['pass'] and not raw['annotation_fields_opened'] and
          raw['gate_mask_agreement'] == 1.0,
          'Raw XYZ/WAV comparison did not pass')
    for key, expected in freeze['checkpoints'].items():
        check(raw['checkpoints'][key] == expected, f'Raw check {key} checkpoint differs')
    return {'validation_mean_iou': validation['metrics']['fused']['mean_iou'],
            'test_mean_iou': test['metrics']['fused']['mean_iou'],
            'train_scenes': len(train_scenes), 'validation_scenes': len(val_scenes),
            'test_scenes': len(test_scenes),
            'checked_observations': 1536, 'evaluator_source_checked':str(evaluator_source),
            'current_code_reproduced':False, 'status': 'pass'}


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluator-source',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.evaluator_source), indent=2))
