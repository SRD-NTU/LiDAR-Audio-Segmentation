"""Run the corrected frozen ensemble directly from XYZ/WAV, with no caches.

Example (from project root):
  AUDIO_LIDAR_DATA_ROOT=/path/to/v2-tetra4-4channel-full \
    bash ORBIT/run.sh full ORBIT/scripts/run_sample.py --output /tmp/orbit-mask.npz
Output sound_mask is in the original lidar.npz point order. No labels are read.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from apex import data as D


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rel',help='Observation relative to the dataset; defaults to first validation or test observation')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    orbit=Path(__file__).resolve().parents[1]
    if not (D.DATA/'metadata/microphone_array.json').is_file():
        p.error(f'No V2 dataset at {D.DATA}; set AUDIO_LIDAR_DATA_ROOT to your dataset directory')
    if args.rel:
        rel=Path(args.rel)
        if rel.is_absolute() or '..' in rel.parts:p.error('--rel must stay inside the dataset')
    else:
        split='val' if (D.DATA/'metadata/val.json').is_file() else 'test'
        rows=D.rels(split)
        if not rows:p.error(f'Empty {split} manifest; provide --rel')
        rel=Path(rows[0])
    files={
        'masks':orbit/'runs/masks-consistent-v2-full-1337/best.pt',
        'selector':orbit/'runs/selector-utonia-spatial-1337/best.pt',
        'apex':orbit/'runs/apex-utonia-spatial-1337/best.pt',
        'gate':orbit/'runs/audio-gate-consistent-v2-full-1337/best.pt',
        'gate2':orbit/'runs/audio-gate-pairs-consistent-v2-full-1337/best.pt',
        'projection':orbit/'pretrained/utonia/projection.pt',
        'spatial':orbit/'pretrained/spatial/adaptive.pt',
        'lidar':D.DATA/rel/'lidar.npz',
        'audio':D.DATA/rel/'audio.wav',
        'calibration':D.DATA/'metadata/microphone_array.json'}
    missing=[str(path) for path in files.values() if not path.is_file()]
    if missing:p.error('Missing required inputs: '+', '.join(missing))
    command=[sys.executable,'-u','-m','orbit.predict','--geometry','utonia',
             '--rule','audio_gate','--gate2-weight','1','--device',args.device,
             '--output',str(args.output.resolve())]
    for key,path in files.items():command+=['--'+key,str(path)]
    print(json.dumps({'dataset':str(D.DATA),'observation':str(rel),'cache_required':False}),flush=True)
    subprocess.run(command,check=True)


if __name__=='__main__':main()
