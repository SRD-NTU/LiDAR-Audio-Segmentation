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
    p.add_argument('--device',default='cuda');p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(1337);torch.backends.cuda.matmul.allow_tf32=False
    destination=Path(a.output);destination=(HERE/destination).resolve() if not destination.is_absolute() else destination.resolve()
    if not destination.is_relative_to(HERE/'outputs'):raise ValueError('Write predictions inside this release outputs/ directory')
    if destination.exists():raise FileExistsError(destination)
    if a.prepared:
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
                         foreground=int(result['sound_mask'].sum()),observations=model.config['audio_observations'])))
