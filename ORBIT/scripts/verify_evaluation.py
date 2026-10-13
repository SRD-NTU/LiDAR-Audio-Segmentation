"""Verify the bundled frozen result records without checkpoints or dataset access."""
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / 'evaluation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    freeze = json.loads((EVALUATION / 'final_method_frozen.json').read_text())
    val_path = EVALUATION / 'utonia_dualgate_equal_val.json'
    require(sha(val_path) == freeze['selected_validation_report_sha256'],
            'Validation report differs from the frozen record')
    require(sha(ROOT / 'orbit/evaluate_audio_gate.py') == freeze['evaluator_source_sha256'],
            'Evaluator source differs from the frozen record')
    validation = json.loads(val_path.read_text())
    test = json.loads((EVALUATION / 'final_test.json').read_text())
    for split, report in (('val', validation), ('test', test)):
        samples = report['samples']
        require(report['split'] == split and report['full_split'] and report['control'] == 'matched',
                f'Invalid {split} split record')
        require(len(samples) == len({row['rel'] for row in samples}) == 768,
                f'{split} record must contain 768 unique observations')
        require(report['gate_weight'] == freeze['gate_weight'] and
                report['gate2_weight'] == freeze['gate2_weight'],
                f'{split} weights differ from frozen method')
        require(report['source_sha256'] == freeze['evaluator_source_sha256'],
                f'{split} evaluator hash differs')
        require(all(report['checkpoints'][key]['sha256'] == digest
                    for key, digest in freeze['checkpoints'].items()),
                f'{split} checkpoint identifiers differ')
        mean = statistics.fmean(row['fused'] for row in samples)
        require(abs(mean - report['metrics']['fused']['mean_iou']) < 1e-12,
                f'{split} sample mean differs from summary')
    require(abs(validation['metrics']['fused']['mean_iou'] -
                freeze['selected_validation_mean_iou']) < 1e-12,
            'Validation score differs from freeze')
    require({row['scene'] for row in validation['samples']}.isdisjoint(
            {row['scene'] for row in test['samples']}),
            'Validation and TEST scenes overlap')
    raw = json.loads((EVALUATION / 'utonia_dualgate_raw_check.json').read_text())
    require(raw['pass'] and not raw['annotation_fields_opened'] and
            raw['gate_mask_agreement'] == 1.0,
            'Saved raw XYZ/WAV comparison did not pass')
    print(json.dumps(dict(status='pass', validation_mean_iou=freeze['selected_validation_mean_iou'],
                          test_mean_iou=test['metrics']['fused']['mean_iou'],
                          observations_per_split=768), indent=2))


if __name__ == '__main__':
    main()
