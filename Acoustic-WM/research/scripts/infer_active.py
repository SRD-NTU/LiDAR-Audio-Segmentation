"""Raw, auditable ActiveView inference; one acquisition after planning."""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import *
import numpy as np,torch
from controlled_model import ControlledActiveView
from acoustic_wm.frontend import RawObservationEncoder
from acoustic_wm.audio.predictive_features import covariance48_from_wave
import acoustic_wm
acoustic_wm.__path__.append(str(FUSION/'code/acoustic_wm'))
from deployment_menu import load_menu,load_observation,check_source_consistency
p=argparse.ArgumentParser();p.add_argument('--menu',required=True);p.add_argument('--output',required=True);p.add_argument('--policy',choices=['grouped','legacy','fixed'],default='grouped');p.add_argument('--max-translation-m',type=float,default=3.);a=p.parse_args()
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run GPU inference through scripts/gpu.sbatch')
out=Path(a.output).resolve()
if not out.is_relative_to(BASE):raise ValueError('Output must stay in Experiments_20261024')
if out.exists():raise FileExistsError(out)
seed_all();model=ControlledActiveView(ACTIVE);enc=RawObservationEncoder(json.loads((ACTIVE/'config.json').read_text()))
menu,actions,valid,cost=load_menu(a.menu);xyz,wave,rate=load_observation(menu['reference']);row=enc.encode_arrays(xyz,wave,rate);obs,weight=covariance48_from_wave(wave)
def acquire(query):
 selected=menu['observations'][query];check_source_consistency(menu['reference'],selected)
 # ActiveView consumes future audio only; never open the selected LiDAR file.
 import soundfile as sf
 audio,rate=sf.read(selected['audio'],dtype='float32',always_2d=True)
 if rate!=44100:raise ValueError('Expected four-channel 44100 Hz audio')
 return covariance48_from_wave(audio)
result=model.run(row,obs,weight,actions,acquire,valid=valid,costs=cost,max_translation=a.max_translation_m,policy=a.policy)
save_npz(out,**result);print(json.dumps(dict(output=str(out),chosen_action=int(result['chosen_action']),cost=float(result['chosen_movement_cost']),distance_m=float(result['distance_m']),budget_used=int(result['budget_used']))))
