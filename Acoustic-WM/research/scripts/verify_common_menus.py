"""Verify that the generated common menus contain no source-dependent actions."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

HERE=Path(__file__).resolve().parents[1]
PROJECT=HERE.parents[1]


def verify(data):
    scenes=[]
    for split in ('train','val','test'):
        for scene in sorted((data/split).iterdir()):
            if not scene.is_dir():continue
            menu_file=data/'scenes'/scene.name/'menu.json'
            raw=menu_file.read_bytes();menu=json.loads(raw)
            if menu['scene']!=scene.name or menu['source_inputs_used'] is not False:
                raise ValueError(f'Invalid geometry-only menu declaration: {scene}')
            expected=np.asarray([p['position'] for p in menu['poses']],np.float64)
            expected_yaw=np.asarray([p['yaw'] for p in menu['poses']],np.float64)
            menu_hash=hashlib.sha256(raw).hexdigest()
            sources=0;observations=0;validity_reference=None
            for source in sorted(scene.iterdir()):
                if not source.is_dir():continue
                receiver_files=sorted(source.glob('*/metadata.json'))
                if len(receiver_files)!=len(expected):
                    raise ValueError(f'{source}: {len(receiver_files)} receivers, expected {len(expected)}')
                seen=set();positions=np.empty_like(expected)
                for file in receiver_files:
                    index=int(file.parent.name)
                    if index in seen or not 0<=index<len(expected):
                        raise ValueError(f'Duplicate/invalid receiver index: {file}')
                    seen.add(index)
                    record=json.loads(file.read_text())
                    if record.get('menu_sha256')!=menu_hash:
                        raise ValueError(f'Source-dependent menu hash: {file}')
                    position=np.asarray(record['listener']['position_world'],np.float64)
                    if not np.allclose(position,expected[index],atol=1e-5,rtol=0):
                        raise ValueError(f'Source-dependent receiver position: {file}')
                    if not np.isclose(record['listener']['yaw_rad'],expected_yaw[index],atol=1e-5,rtol=0):
                        raise ValueError(f'Source-dependent receiver yaw: {file}')
                    positions[index]=position
                if seen!=set(range(len(expected))):
                    raise ValueError(f'Missing receiver indices: {source}')
                # Every starting receiver has the same <=3 m local action mask.
                validity=(np.linalg.norm(positions[:,None]-positions[None,:],axis=2)<=3.0+1e-6)
                if validity_reference is None:validity_reference=validity
                elif not np.array_equal(validity,validity_reference):
                    raise ValueError(f'Source-dependent local validity: {source}')
                sources+=1;observations+=len(receiver_files)
            scenes.append(dict(split=split,scene=scene.name,sources=sources,
                               receivers_per_source=len(expected),observations=observations,
                               menu_sha256=menu_hash,local_action_masks_identical=True))
    return dict(dataset=str(data.resolve()),rule='scene geometry and seed before acoustic source assignment',
                local_validity='Euclidean receiver displacement <= 3 m, same for every source in a scene',
                scenes=len(scenes),sources=sum(x['sources'] for x in scenes),
                observations=sum(x['observations'] for x in scenes),checks_passed=True,details=scenes)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=PROJECT/'.importants/Experiments_20261024/generated/common')
    parser.add_argument('--output',type=Path,default=HERE/'reports/common_menu_independence.json')
    args=parser.parse_args()
    result=verify(args.data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
