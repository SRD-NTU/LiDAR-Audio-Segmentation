"""Run this release from raw XYZ/WAV or the supplied observation-only example."""
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE/'code'))
for key,folder in [('XDG_CACHE_HOME','xdg'),('TORCH_HOME','torch'),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba')]:
    os.environ[key]=str(HERE/'outputs/.cache'/folder)
    Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import argparse,json
import numpy as np
import soundfile as sf
import torch
from acoustic_wm.model import ReleasedModel
from acoustic_wm.audio.predictive_features import covariance48_from_wave

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',help='Observation-only reference NPZ, as in examples/')
    p.add_argument('--lidar');p.add_argument('--audio');p.add_argument('--actions')
    p.add_argument('--future-audios',nargs='+')
    p.add_argument('--future-observations',help='Prepared covariance observations; loaded only after planning')
    p.add_argument('--menu',help='Pose-only JSON menu for validity-aware, grouped one-query inference')
    p.add_argument('--policy',choices=['grouped','legacy','fixed'],default='grouped')
    p.add_argument('--max-translation-m',type=float,default=3.0)
    p.add_argument('--device',default='cuda');p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(1337);torch.backends.cuda.matmul.allow_tf32=False
    destination=Path(a.output);destination=(HERE/destination).resolve() if not destination.is_absolute() else destination.resolve()
    if not destination.is_relative_to(HERE/'outputs'):raise ValueError('Write predictions inside this release outputs/ directory')
    if destination.exists():raise FileExistsError(destination)
    if a.menu:
        if a.prepared or a.lidar or a.audio or a.actions or a.future_audios or a.future_observations:
            p.error('--menu supplies all observation paths and poses; do not combine it with legacy inputs')
        from acoustic_wm.deployment_menu import load_menu,load_observation,check_source_consistency
        from acoustic_wm.controlled import ControlledActiveView
        from acoustic_wm.frontend import RawObservationEncoder
        menu,actions,valid,costs=load_menu(a.menu)
        xyz,wave,rate=load_observation(menu['reference'])
        config=json.loads((HERE/'config.json').read_text())
        config['paths']['dataset']=str(HERE/'assets/sensor')
        encoder=RawObservationEncoder(config,a.device)
        row=encoder.encode_arrays(xyz,wave,rate)
        observed,reliability=covariance48_from_wave(wave)
        model=ControlledActiveView(HERE,a.device)
        def acquire(query):
            selected=menu['observations'][query]
            check_source_consistency(menu['reference'],selected)
            # The second LiDAR path is never opened by ActiveView.
            second,second_rate=sf.read(selected['audio'],dtype='float32',always_2d=True)
            if second_rate!=44100 or second.shape[1]!=4 or not np.isfinite(second).all():
                raise ValueError('Selected audio must be finite 44100 Hz with four channels')
            return covariance48_from_wave(second)
        result=model.run(row,observed,reliability,actions,acquire,valid=valid,costs=costs,
                         max_translation=a.max_translation_m,policy=a.policy)
        result['observation_menu_json']=np.str_(json.dumps(menu,sort_keys=True))
    elif a.prepared:
        with np.load(a.prepared) as z:
            row={k:z[k] for k in z.files if k not in ['observed','reliability']}
            observed,reliability=z['observed'],z['reliability']
    else:
        if not a.lidar or not a.audio:p.error('Provide --prepared, or both --lidar and --audio')
        from acoustic_wm.frontend import RawObservationEncoder
        config=json.loads((HERE/'config.json').read_text())
        config['paths']['dataset']=str(HERE/'assets/sensor')
        with np.load(a.lidar) as z:xyz=z['xyz'].astype(np.float32)
        wave,rate=sf.read(a.audio,dtype='float32',always_2d=True)
        if rate!=44100:raise ValueError('Expected 44100 Hz four-channel audio')
        encoder=RawObservationEncoder(config,a.device);row=encoder.encode_arrays(xyz,wave,rate)
        observed,reliability=covariance48_from_wave(wave)
    if not a.menu:
        model=ReleasedModel(HERE,a.device)
        if not model.future_enabled:result=model.segment(row,observed,reliability)
        else:
            if not a.actions:p.error('This version requires --actions and a future observation provider')
            actions=np.load(a.actions);plan=model.plan(row,observed,reliability,actions)
            if a.future_observations:
                with np.load(a.future_observations) as z:
                    second,weight=z['observed'][plan['query']],z['reliability'][plan['query']]
            else:
                if not a.future_audios or len(a.future_audios)!=len(actions):p.error('One --future-audios path per action is required')
                wave,rate=sf.read(a.future_audios[plan['query']],dtype='float32',always_2d=True)
                if rate!=44100:raise ValueError('Expected 44100 Hz at the selected receiver')
                second,weight=covariance48_from_wave(wave)
            result=model.complete(plan,second,weight)
    destination.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(destination,**result)
    print(json.dumps(dict(version=model.config['version'],selected_object=int(result['selected_object']),
                         foreground=int(result['sound_mask'].sum()),
                         observations=int(result.get('audio_observations',model.config['audio_observations'])))))
