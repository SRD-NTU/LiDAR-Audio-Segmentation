"""Explicitly attest historical arrays against recorded hashes and current WAVs.

This establishes a verifiable baseline, not a claim of independent inference
reproduction or proof of the original encoder weights. No array is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from apex import data as D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', required=True)
    args = parser.parse_args()
    full = D.ROOT/'dataset/v2-tetra4-4channel-full'
    root = D.ROOT/'ORBIT'
    recorded = json.loads((root/'cache/aligned-utonia-v1/COMPLETE.json').read_text())['provenance']
    test_record = json.loads((root/'reports/final_test.json').read_text())['provenance']
    report=[]
    for split in ('train','val','test'):
        rows=D.rels(split,full)
        hashes=D.waveform_hashes(rows,full)
        identity=hashlib.sha256((full/'metadata/microphone_array.json').read_bytes())
        identity.update(json.dumps({'n_dirs':D.acoustic.N_DIRS,'band':list(D.acoustic.BAND)}).encode())
        for rel,actual in zip(rows,hashes):
            completion=json.loads((full/rel/'COMPLETE.json').read_text())
            expected=completion['sha256']['audio.wav']
            if actual!=expected:raise ValueError(f'WAV differs from its completion record: {rel}')
            identity.update((full/rel/'COMPLETE.json').read_bytes())
        srp=root/f'cache/audio-legacy/posteriors_{split}_{len(rows)}_{identity.hexdigest()[:16]}.npy'
        clap=root/f'cache/audio-legacy/clap_tetra_{split}.npz'
        old=recorded[split] if split in recorded else test_record
        for name,path in (('srp',srp),('clap',clap)):
            if D.sha(path)!=old[name+'_sha256']:raise ValueError(f'Historical {name} differs from recorded provenance')
        with np.load(clap,allow_pickle=False) as z:
            if list(z['rels'])!=rows:raise ValueError('Historical CLAP rows differ')
        if old['calibration_sha256']!=D.sha(full/'metadata/microphone_array.json'):
            raise ValueError('Historical calibration differs')
        manifest=dict(schema=2,rels=rows,waveform_sha256=hashes,
                      calibration_sha256=old['calibration_sha256'],srp_sha256=old['srp_sha256'],
                      clap_sha256=old['clap_sha256'],dependencies={},
                      provenance='legacy-attested-against-recorded-artifact-hashes',
                      original_encoder_generation_identity_verified=False)
        target=clap.with_suffix('.npz.identity.json')
        if target.exists() and json.loads(target.read_text())!=manifest:
            raise ValueError(f'Existing legacy identity differs: {target}')
        target.write_text(json.dumps(manifest,indent=2)+'\n')
        if split=='test':
            path=root/'cache/utonia-heldout-test-v1/adaptive_test_posterior.npy'
            ordered=json.loads((path.parent/'rels.json').read_text())
            expected=test_record['replacement_posterior_sha256']
        else:
            path=root/f'cache/spatial-v1/adaptive_{split}_posterior.npy'
            with np.load(path.parent/f'{split}_observation_features.npz') as z:ordered=list(z['rels'])
            expected=recorded[split]['replacement_posterior_sha256']
        if ordered!=rows or D.sha(path)!=expected:raise ValueError('Historical spatial array or order differs')
        D.write_posterior_identity(path,rows,dataset=full,
            provenance='legacy-attested-against-recorded-artifact-hashes')
        report.append(dict(split=split,rows=len(rows),actual_waveforms_verified=True,
                           original_encoder_generation_identity_verified=False))
        print(json.dumps(report[-1]),flush=True)
    output=root/'reports/legacy-cache-identity-attestation-v2.json'
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
