"""Train and fully evaluate the audio-conditioned object query in one job."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path(__file__).resolve().parents[1]
    cache=root/'cache/aligned-utonia-v1'
    geometry=root/'cache/utonia-xyz-v1'
    spatial=root/'cache/spatial-v1'
    masks=root/'runs/masks-utonia-spatial-1337/best.pt'
    selector=root/'runs/selector-utonia-spatial-1337/best.pt'
    apex=root/'runs/apex-utonia-spatial-1337/best.pt'
    run=root/'runs/audio-gate-utonia-1337'
    output=root/'reports/utonia_audio_gate_val.json'
    journal=root/'reports/campaign-audio-gate.json'
    if journal.exists() or output.exists() or (run/'best.pt').exists():raise FileExistsError(journal)
    for path in (cache/'COMPLETE.json',geometry/'COMPLETE.json',spatial/'COMPLETE.json',masks,selector,apex):
        if not path.exists():raise FileNotFoundError(path)
    common=['--masks',str(masks),'--selector',str(selector),'--apex',str(apex)]
    commands=[['-m','scene_frontend.train_audio_gate','--cache',str(cache),*common,'--output',str(run),'--epochs','16'],
              ['-m','scene_frontend.evaluate_audio_gate',*common,'--gate',str(run/'best.pt'),
               '--geometry',str(geometry),'--spatial',str(spatial),'--output',str(output)]]
    state=dict(test_used=False,status='running',stages=[])
    for cmd in commands:
        stage=dict(command=[sys.executable,'-u',*cmd],status='running',start_time=time.time())
        state['stages'].append(stage);journal.write_text(json.dumps(state,indent=2))
        print('START',stage['command'],flush=True)
        try:subprocess.run(stage['command'],check=True)
        except BaseException:
            stage['status']='failed';state['status']='failed';raise
        else:stage['status']='complete'
        finally:
            stage['elapsed_seconds']=time.time()-stage['start_time']
            journal.write_text(json.dumps(state,indent=2))
    state['status']='complete';journal.write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
