"""Inference for Acoustic-WM LiDARFusion (56.84% validation reference IoU)."""
from pathlib import Path
import sys,argparse,json
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(HERE),str(HERE/'code')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--menu',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',default='checkpoints/fusion_ranker.pt')
    p.add_argument('--max-translation-m',type=float,default=3.0)
    args=p.parse_args();destination=(HERE/args.output).resolve()
    if not destination.is_relative_to(HERE/'outputs'):p.error('Outputs must remain inside this fusion version')
    if destination.exists():raise FileExistsError(destination)
    from recovery.runtime import setup
    setup(HERE/'outputs','raw-inference')
    import numpy as np,torch
    from acoustic_wm.fusion_model import FusionModel
    from acoustic_wm.fused_frontend import FusionObservationEncoder
    from acoustic_wm.models.pointbind_adapter import PointBindEncoder
    from acoustic_wm.fusion_io import load_menu,load_observation,check_source_consistency
    from recovery.features import ConservativeRanker
    from recovery.pipeline import RecoveryPipeline
    torch.set_num_threads(1);torch.manual_seed(1337);torch.backends.cuda.matmul.allow_tf32=False
    selected=torch.load(HERE/args.checkpoint,map_location='cpu',weights_only=False)
    config=selected['configuration']
    ranker=ConservativeRanker(selected['mean'],selected['std'],config['kind'],config['bound']).cuda().eval()
    ranker.load_state_dict(selected['model']);ranker.requires_grad_(False)
    configuration=json.loads((HERE/'config.json').read_text());configuration['paths']['dataset']=str(HERE/'assets/sensor')
    encoder=FusionObservationEncoder(configuration,'cuda');model=FusionModel(HERE,'cuda')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):model.pointbind=PointBindEncoder('cuda')
    torch.manual_seed(1337)
    menu,actions,valid,costs=load_menu(args.menu)
    xyz,wave,rate=load_observation(menu['reference'])
    def acquire(query):
        record=menu['observations'][query];check_source_consistency(menu['reference'],record)
        return load_observation(record)
    result=RecoveryPipeline(model,encoder,ranker,selected['kind']).run(
        xyz,wave,actions,acquire,rate=rate,valid=valid,
        movement_cost=costs,max_translation_m=args.max_translation_m)
    result['observation_menu_json']=np.str_(json.dumps(menu,sort_keys=True))
    result['checkpoint']=np.str_(str((HERE/args.checkpoint).resolve()))
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:np.savez_compressed(stream,**result)
    print(json.dumps(dict(output=str(destination),chosen_action=int(result['chosen_action']),
        selected_bank=str(result['selected_bank']),reference_points=len(xyz),second_points=len(result['second_xyz']),
        budget_used=int(result['budget_used']),distance_m=float(result['distance_m']))))


if __name__=='__main__':main()
