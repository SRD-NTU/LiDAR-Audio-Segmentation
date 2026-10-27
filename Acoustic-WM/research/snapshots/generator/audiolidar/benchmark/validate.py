"""Dataset integrity checks, strict checksums and evaluator-only ground truth."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
from audiolidar.utils.io import read_json,sha256,write_json
from audiolidar.geometry.frames import to_world,relative
from audiolidar.evaluation.metrics import metrics
from audiolidar.audio.spatialize import dry_wave,convolve


def validate_sample(path: Path, checksums: bool = True) -> dict:
    path=Path(path); m=read_json(path/'metadata.json')
    with np.load(path/'lidar.npz',allow_pickle=False) as file: p=dict(file)
    n=len(p['xyz']); assert n>0
    for key in ['range','beam_id','ray_id','triangle_id','semantic_id','instance_id','sound_mask','azimuth','elevation']:
        assert p[key].shape==(n,), f'{key}: point count mismatch'
    assert p['xyz'].shape==p['xyz_world'].shape==(n,3)
    assert all(np.isfinite(a).all() for a in p.values()), 'Nonfinite point field'
    assert np.isin(p['sound_mask'],[0,1]).all()
    expected=p['instance_id']==m['source']['instance_id']
    assert np.array_equal(p['sound_mask'].astype(bool),expected),'Instance-mask disagreement'
    assert m['source']['scene_id']==m['scene_id'],'Cross-scene source'
    assert m['scene_id']==path.parents[1].name,'Scene-directory disagreement'
    assert m['split']==path.parents[2].name,'Split-directory disagreement'
    assert np.all(p['semantic_id'][expected]==m['source']['semantic_id'])
    assert expected.sum()>=m['validity']['min_source_points']
    assert int(expected.sum())==m['lidar']['num_source_points']
    assert n==m['lidar']['num_points']
    assert np.isclose(expected.mean(),m['lidar']['source_point_fraction'])
    assert len(np.unique(p['ray_id']))==n,'Multiple hits per ray'
    assert (np.diff(p['ray_id'])>0).all(),'Ray order not stable'
    assert ((p['beam_id']>=0)&(p['beam_id']<m['lidar']['num_beams'])).all()
    assert (p['range']>=m['lidar']['min_range_m']-1e-5).all()
    assert (p['range']<=m['lidar']['max_range_m']+1e-5).all()
    assert np.allclose(np.linalg.norm(p['xyz'],axis=1),p['range'],atol=2e-5)
    unit=p['xyz']/p['range'][:,None]
    expected_unit=np.column_stack([np.cos(p['elevation'])*np.cos(p['azimuth']),np.cos(p['elevation'])*np.sin(p['azimuth']),np.sin(p['elevation'])])
    assert np.allclose(unit,expected_unit,atol=2e-5),'Beam angle disagreement'
    assert np.array_equal(p['beam_id'],p['ray_id']%m['lidar']['num_beams'])
    assert path.parent.name==str(m['source']['instance_id'])
    offset=m['source']['surface_offset_m']+.02
    source_position=np.asarray(m['source']['position_world'])
    assert np.all(source_position>=np.asarray(m['source']['bbox_min'])-offset) and np.all(source_position<=np.asarray(m['source']['bbox_max'])+offset),'Source outside instance bounds'

    listener=np.asarray(m['listener']['position_world']);yaw=m['listener']['yaw_rad']
    assert np.allclose(to_world(p['xyz'],listener,yaw),p['xyz_world'],atol=2e-5)
    rel=relative(np.asarray(m['source']['position_world']),listener,yaw)
    assert np.isclose(rel['distance_m'],m['relative_source']['distance_m'])
    assert np.isclose(rel['azimuth_deg'],m['relative_source']['azimuth_deg'])
    wave,rate=sf.read(path/'audio.wav',always_2d=True)
    expected_channels={'binaural':2,'ambisonics':4,'microphone_array_4':4}
    channels=expected_channels.get(m['audio']['mode'])
    assert channels is not None, 'Unsupported audio mode'
    assert len(m['audio']['channel_order'])==channels, 'Channel-order mismatch'
    assert wave.shape==(round(m['audio']['duration_sec']*rate),channels)
    assert rate==m['audio']['sample_rate']==44100
    assert np.isfinite(wave).all() and np.max(np.abs(wave))>1e-10
    assert np.max(np.abs(wave))<=.951
    rir=np.load(path/'rir.npy',allow_pickle=False)
    assert rir.ndim==2 and rir.shape[0]==channels and np.isfinite(rir).all() and np.any(rir)
    dry_path=Path(m['audio']['dry_audio_file'])
    assert sha256(dry_path)==m['audio']['dry_audio_sha256'],'Dry audio changed'
    dry=dry_wave(dry_path,rate,m['audio']['duration_sec'])
    reconstructed,gain=convolve(dry,rir,len(dry))
    assert np.isclose(gain,m['audio']['joint_gain'])
    assert np.allclose(reconstructed,wave,atol=1e-7),'RIR/waveform mismatch'
    assert all(m['soundspaces'][k] for k in ['direct','indirect','diffraction','transmission'])
    assert metrics(expected,p['sound_mask'])==dict(iou=1.,precision=1.,recall=1.,f1=1.)
    if checksums:
        marker=read_json(path/'COMPLETE.json')
        for name,digest in marker['sha256'].items():assert sha256(path/name)==digest, 'Checksum mismatch: '+name
        required={'lidar.npz','audio.wav','rir.npy','metadata.json'}
        assert required.issubset(marker['sha256']), 'Missing core checksum'
        preview_files=sorted(path.joinpath('preview').glob('*.png'))
        assert preview_files and 'preview/rgb360_segmented.png' in {str(x.relative_to(path)) for x in preview_files}, 'Missing preview PNG set'
        assert all(str(x.relative_to(path)) in marker['sha256'] for x in preview_files), 'Preview checksum missing'
    return m


def samples(root: Path) -> list:
    return sorted(Path(root).glob('*/*/*/*/COMPLETE.json'))


def validate_dataset(root: Path) -> dict:
    root=Path(root); plan=read_json(root/'metadata/splits.json')
    scenes=[x for group in plan['scenes'].values() for x in group]
    audio=[x for group in plan['audio'].values() for x in group]
    assert len(scenes)==len(set(scenes)),'Scene leakage'
    assert len(audio)==len(set(audio)),'Audio ID leakage'
    hashes={}
    for split,ids in plan['audio'].items():
        for ident in ids:
            digest=plan['audio_hashes'][ident]
            assert digest not in hashes or hashes[digest]==split,'Audio content leakage'
            hashes[digest]=split
    paths=samples(root); assert paths,'No completed samples'
    for marker in paths:
        m=validate_sample(marker.parent)
        assert m['scene_id'] in plan['scenes'][m['split']]
        assert m['audio']['dry_audio_id'] in plan['audio'][m['split']]
        assert m['config_sha256']==plan['config_sha256']
    incomplete=[p for p in root.glob('*/*/*/*/metadata.json') if not (p.parent/'COMPLETE.json').exists()]
    assert not incomplete,'Incomplete sample directories'
    report={'status':'PASS','samples':len(paths),'scene_disjoint':True,'audio_disjoint':True,'oracle_iou':1.}
    write_json(root/'metadata/validation.json',report)
    return report
