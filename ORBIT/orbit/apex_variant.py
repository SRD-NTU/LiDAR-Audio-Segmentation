"""Reuse the unchanged APEX trainer with explicitly recorded alternative features."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from apex import data as D


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--geometry',type=Path)
    p.add_argument('--spatial',type=Path)
    args,remaining=p.parse_known_args()
    if args.geometry is not None:D.GEOMETRY=args.geometry
    original=D.audio_bank
    def bank(split):
        rs,post,emb,provenance=original(split)
        provenance['geometry']=str(D.GEOMETRY)
        if (D.GEOMETRY/'COMPLETE.json').exists():
            provenance['geometry_manifest_sha256']=D.sha(D.GEOMETRY/'COMPLETE.json')
        if args.spatial:
            path=args.spatial/f'adaptive_{split}_posterior.npy'
            post=np.load(path)
            if post.shape!=(len(rs),D.acoustic.N_DIRS):raise ValueError('Wrong posterior dimensions')
            provenance['replacement_posterior_sha256']=D.sha(path)
        provenance['variant_wrapper_sha256']=D.sha(Path(__file__))
        return rs,post,emb,provenance
    D.audio_bank=bank
    sys.argv=[sys.argv[0]]+remaining
    from apex.run import main as train
    train()


if __name__=='__main__':main()
