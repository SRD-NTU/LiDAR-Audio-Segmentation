"""Bounded sequential GPU campaign; downstream evaluation cannot be forgotten.

Run in a Slurm allocation. Existing outputs are not overwritten or auto-resumed.
Each subprocess must finish successfully before its dependent stage starts.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('utonia','sonata-checks'),required=True)
    args=parser.parse_args()
    journal=ROOT/'reports'/f'campaign-{args.mode}.json'
    if journal.exists():raise FileExistsError(journal)
    state=dict(mode=args.mode,test_used=False,status='running',stages=[])

    def run(module,*arguments):
        command=[sys.executable,'-u','-m',module,*map(str,arguments)]
        stage=dict(command=command,status='running',started=time.time())
        state['stages'].append(stage);journal.write_text(json.dumps(state,indent=2))
        print('START',command,flush=True)
        try:
            subprocess.run(command,check=True)
        except BaseException:
            stage['status']='failed';state['status']='failed'
            raise
        else:stage['status']='complete'
        finally:
            stage['elapsed_seconds']=time.time()-stage['started']
            journal.write_text(json.dumps(state,indent=2))

    if args.mode=='utonia':
        cache=ROOT/'cache/aligned-utonia-v1'
        geometry=ROOT/'cache/utonia-xyz-v1'
        spatial=ROOT/'cache/spatial-v1'
        selector=ROOT/'runs/selector-utonia-spatial-1337'
        masks=ROOT/'runs/masks-utonia-spatial-1337'
        apex=ROOT/'runs/apex-utonia-spatial-1337'
        ranker=ROOT/'runs/reranker-utonia-spatial-1337'
        for path in (cache,geometry,spatial):
            if not (path/'COMPLETE.json').exists():raise FileNotFoundError(path/'COMPLETE.json')
        run('orbit.train_linear_selector','--cache',cache,'--priors',ROOT/'cache/clap_text_priors.npz','--output',selector)
        run('orbit.train_masks','--cache',cache,'--output',masks,'--epochs',24)
        run('orbit.apex_variant','--geometry',geometry,'--spatial',spatial,'--output',apex,'--epochs',24)
        common=['--masks',masks/'best.pt','--selector',selector/'best.pt','--apex',apex/'best.pt','--geometry',geometry]
        evaluation=[*common,'--posterior',spatial/'adaptive_val_posterior.npy']
        run('orbit.evaluate',*evaluation,'--output',ROOT/'reports/utonia_spatial_val.json')
        run('orbit.candidate_bank',*common,'--spatial',spatial,'--output',ROOT/'cache/candidates-utonia-spatial-v1',
            '--fit-reranker',ranker)
        for control in ('matched','shuffled','uniform-spatial'):
            run('orbit.evaluate',*evaluation,'--reranker',ranker/'best.pt','--control',control,
                '--output',ROOT/f'reports/utonia_reranker_{control}_val.json')
        run('orbit.raw_check',*common,'--projection',geometry/'projection.pt',
            '--spatial',spatial/'adaptive.pt','--posterior',spatial/'adaptive_val_posterior.npy',
            '--output',ROOT/'reports/utonia_raw_check.json')
    else:
        common=['--masks',ROOT/'runs/masks-1337/best.pt',
                '--selector',ROOT/'runs/linear-selector-v2-1337/best.pt',
                '--apex',ROOT/'runs/apex-tetra-aligned-1337/best.pt']
        for control in ('matched','shuffled','uniform-spatial'):
            run('orbit.evaluate',*common,'--reranker',ROOT/'runs/reranker-sonata-1337/best.pt',
                '--control',control,'--output',ROOT/f'reports/sonata_reranker_{control}_val.json')
        run('orbit.raw_check',*common,'--output',ROOT/'reports/sonata_raw_check.json')
    state['status']='complete';journal.write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
