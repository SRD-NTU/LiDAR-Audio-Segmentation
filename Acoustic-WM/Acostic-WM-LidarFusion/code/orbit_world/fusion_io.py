"""Pose-only query menus. Paths are resolved without opening future recordings.

Menu JSON:
  reference: {lidar, audio, receiver_to_world: 4x4, source_id?, activity_id?}
  observations: [{id, lidar, audio, receiver_to_world: 4x4, valid?, ...}]
  availability_basis: a description of how receiver availability was determined

All LiDAR XYZ coordinates must be local to the calibrated receiver frame and
use metres. Audio must describe one stationary source and unchanged activity.
IDs are audit metadata, never model inputs. A source-independent menu must be
established by dataset construction; a declaration here is not proof of that.
"""
import json
from pathlib import Path

import numpy as np

from .fusion import actions_from_poses, validate_cloud


def load_menu(path, max_translation_m=None):
    path = Path(path).resolve()
    menu = json.loads(path.read_text())
    if not isinstance(menu.get('availability_basis'), str) or not menu['availability_basis'].strip():
        raise ValueError('Describe source-independent receiver availability in availability_basis')
    reference = menu['reference']
    entries = menu['observations']
    if not entries:
        raise ValueError('The observation menu is empty')
    ids = [str(e['id']) for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError('Receiver observation IDs must be unique')
    actions = actions_from_poses(reference['receiver_to_world'],
                                 [e['receiver_to_world'] for e in entries])
    valid = []
    for entry in entries:
        if not isinstance(entry.get('valid', True), bool):
            raise ValueError('valid must be a JSON boolean determined independently of the target')
        valid.append(entry.get('valid', True))
    valid = np.asarray(valid, bool)
    distances = np.linalg.norm(actions[:, :3], axis=1)
    if max_translation_m is not None:
        if not np.isfinite(max_translation_m) or max_translation_m < 0:
            raise ValueError('max_translation_m must be finite and nonnegative')
        valid &= distances <= max_translation_m
    costs = np.asarray([e.get('movement_cost', float(d)) for e, d in zip(entries, distances)], np.float32)
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError('Invalid movement costs')
    if not valid.any():
        raise ValueError('No receiver locations satisfy validity and translation constraints')
    for entry in [reference, *entries]:
        for key in ('lidar', 'audio'):
            value = Path(entry[key]).expanduser()
            entry[key] = str(value.resolve() if value.is_absolute() else (path.parent/value).resolve())
    return menu, actions, valid, costs


def load_observation(entry):
    import soundfile as sf
    with np.load(entry['lidar'], allow_pickle=False) as record:
        xyz = validate_cloud(record['xyz'])
    wave, rate = sf.read(entry['audio'], dtype='float32', always_2d=True)
    if rate != 44100 or wave.ndim != 2 or wave.shape[1] != 4 or not np.isfinite(wave).all():
        raise ValueError('Audio must be finite 44100 Hz with four channels')
    return xyz, wave, rate


def check_source_consistency(reference, selected):
    """Check metadata only after query selection; never filter the action menu."""
    checked = []
    for key in ('source_id', 'activity_id'):
        if key in reference or key in selected:
            if key not in reference or key not in selected or reference[key] != selected[key]:
                raise ValueError(f'Selected observation does not match the reference {key}')
            checked.append(key)
    return checked
