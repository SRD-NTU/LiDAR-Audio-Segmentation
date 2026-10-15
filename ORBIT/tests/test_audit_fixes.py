import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import torch
from apex import data as D
from orbit.cache_contracts import (geometry_uid, shared_indices, sparse_memberships,
                                    CACHE_SCHEMA, instance_target)
from orbit.audio_gate import sample_seeds
from orbit.train_audio_gate_pairs import same_scan_groups, query_seeds, usable_pairs
from orbit.train_masks import prompts


def cached_row(uid, selected, source):
    n=len(selected)
    return dict(cache_schema=CACHE_SCHEMA,geometry_uid=uid,raw_xyz_sha256='scan',
                full_voxel_count=80,selected_voxel_ids=selected,xyz=np.arange(n*3).reshape(n,3),
                count=np.ones(n)*2,node_object=np.arange(n)%2,
                instance_ids=np.array([14,48]),source_instance_id=[14,48][source],source_object=source,
                positive=(np.arange(n)%2==source).astype(float),
                instance_object=np.arange(n)%2,instance_voxel=np.arange(n),instance_count=np.ones(n),rel='train/scene/14/pose')


class AuditFixes(unittest.TestCase):
    def test_actual_preparation_preserves_same_scan_pair_after_budgeting(self):
        from orbit import prepare
        from orbit.train_masks import load_split
        n=80;ids=np.arange(n)%2
        x=dict(features=np.ones((n,256),np.float32),xyz=np.column_stack([np.arange(n)*.1,np.zeros(n),np.ones(n)]).astype(np.float32),
               count=np.ones(n,np.float32)*2,acoustic=np.ones((n,8),np.float32),audio=np.ones(512,np.float32),raw_xyz_sha256='same-scan')
        def labels(rel,inputs):
            source=int(rel.split('/')[2])
            return dict(memberships=[(np.flatnonzero(ids==j),np.ones(int((ids==j).sum()))*2) for j in (0,1)],
                        node_object=ids,node_category=ids,categories=np.array([0,1]),positive=(ids==source).astype(np.float32)*2,
                        source_object=source,source_class=source,instance_ids=np.array([10,20]),source_instance_id=[10,20][source])
        def audio(split):
            rows=[f'{split}/scene/{j}/pose' for j in (0,1)]
            return rows,np.ones((2,D.acoustic.N_DIRS)),np.ones((2,512)),{}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'cache'
            for split in ('train','val'):
                for source in (0,1):
                    path=root/f'{split}/scene/{source}/pose';path.mkdir(parents=True)
                    (path/'metadata.json').write_text(json.dumps({'listener':{'position_world':[0,0,0],'rotation_world':[0,0,0,1]},
                                                                  'lidar':{'profile':'same','num_source_points':source+1}}))
            with patch.object(D,'DATA',root),patch.object(D,'audio_bank',side_effect=audio),\
                 patch.object(D,'load_inputs',side_effect=lambda *args:{**x}),patch.object(prepare,'supervision',side_effect=labels),\
                 patch('sys.argv',['prepare','--output',str(out),'--workers','1','--train-nodes','20']):prepare.main()
            rows=load_split(out,'train')
            self.assertEqual(len(same_scan_groups(rows)),1)
            np.testing.assert_array_equal(rows[0]['xyz'],rows[1]['xyz'])
            self.assertEqual(len(rows[0]['xyz']),20)
            np.testing.assert_array_equal(instance_target(rows[0],0),instance_target(rows[1],0))

    def test_matched_single_scene_main_does_not_construct_shuffle_partners(self):
        from orbit import evaluate_audio_gate as evaluation
        import warnings
        rel='val/scene/1/pose'
        x=dict(xyz=np.zeros((2,3)),count=np.ones(2),inverse=np.arange(2))
        pred=dict(scores={'hybrid':np.array([0.]),'selector':np.array([0.])},quality=np.array([.5]),
                  choices={'hybrid':0},masks=np.array([[True,False]]),origins=['prompt:0:threshold:.5'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sample=root/rel;sample.mkdir(parents=True)
            (sample/'metadata.json').write_text(json.dumps({'relative_source':{'distance_m':1.}}))
            checkpoint=root/'weights.pt';checkpoint.write_bytes(b'dummy');output=root/'report.json'
            argv=['evaluate']
            for name in ('masks','selector','apex','gate'):argv+=['--'+name,str(checkpoint)]
            argv+=['--output',str(output),'--device','cpu']
            with patch('sys.argv',argv),patch.object(D,'DATA',root),\
                 patch.object(evaluation,'load_models',return_value=None),patch('torch.load',return_value={'model':{}}),\
                 patch.object(evaluation,'AudioSeedGate'),patch.object(D,'audio_bank',return_value=([rel],np.ones((1,D.acoustic.N_DIRS)),np.ones((1,512)),{})),\
                 patch.object(D,'load_inputs',return_value=x),patch.object(evaluation,'propose_and_score',return_value=pred),\
                 patch.object(evaluation,'candidate_gate_values',return_value=np.zeros(1)),\
                 patch.object(D,'load_targets',return_value={'positive':np.array([1.,0.]),'source':0}),warnings.catch_warnings():
                warnings.simplefilter('ignore');evaluation.main()
            self.assertEqual(json.loads(output.read_text())['metrics']['fused']['n'],1)

    def test_raw_identity_and_sampling_ignore_source_role(self):
        listener=dict(position_world=[0,0,0],rotation_world=[0,0,0,1])
        a=geometry_uid('same_xyz','scene',listener,{'profile':'sensor','num_source_points':4})
        b=geometry_uid('same_xyz','scene',listener,{'profile':'sensor','num_source_points':19})
        self.assertEqual(a,b)
        first=shared_indices(a,80,20);second=shared_indices(b,80,20)
        np.testing.assert_array_equal(first,second)
        group=same_scan_groups([cached_row(a,first,0),cached_row(b,second,1)])
        self.assertEqual(len(group),1)
        broken=cached_row(b,second[::-1],1)
        with self.assertRaisesRegex(ValueError,'inconsistent'):same_scan_groups([group[0][0],broken])

    def test_uniform_instance_targets_do_not_change_with_sound_role(self):
        row=dict(node_object=np.array([0,1]),count=np.array([2.,2.]),
                 instance_object=np.array([0,0,1,1]),instance_voxel=np.array([0,1,0,1]),
                 instance_count=np.ones(4),source_object=0,positive=np.array([1.,1.]))
        _,a=prompts(row,np.random.default_rng(1),2)
        _,b=prompts({**row,'source_object':1},np.random.default_rng(1),2)
        np.testing.assert_array_equal(a[0],b[1])
        np.testing.assert_array_equal(a[1],b[0])
        np.testing.assert_array_equal(a,np.full((2,2),.5))

    def test_mask_prompts_do_not_duplicate_mixed_voxel_seed_identities(self):
        row=dict(node_object=np.array([0,0,1]),count=np.array([2.,1.,1.]),
                 instance_object=np.array([0,0,1,1]),instance_voxel=np.array([0,1,0,2]),
                 instance_count=np.ones(4),source_object=0,positive=np.array([1.,1.,0.]))
        for seed in range(50):
            seeds,_=prompts(row,np.random.default_rng(seed),2)
            self.assertEqual(len(seeds),len(set(seeds)))

    def test_sparse_targets_reindex_selected_voxels(self):
        sparse=sparse_memberships([(np.array([0,2,4]),np.array([1,2,3]))],np.array([2,4]))
        row={**sparse,'count':np.array([4.,6.])}
        np.testing.assert_array_equal(instance_target(row,0),[.5,.5])

    def test_single_seeds_have_no_duplicates_or_positive_negatives(self):
        row=dict(node_object=np.array([-1,0,1]),positive=np.array([1.,0.,0.]),source_object=0)
        for seed in range(50):
            sampled=sample_seeds(row,np.random.default_rng(seed))
            self.assertEqual(len(sampled),len(set(sampled)))
            self.assertTrue(np.all(row['positive'][sampled[1:]]==0))

    def test_pairs_reject_ambiguous_prompts_and_use_exclusive_support(self):
        a=dict(node_object=np.array([0,1,-1]),positive=np.array([1.,1.,0.]),source_object=0)
        b=dict(node_object=np.array([0,1,-1]),positive=np.array([1.,0.,0.]),source_object=1)
        with self.assertRaisesRegex(ValueError,'Ambiguous'):query_seeds(a,b,np.random.default_rng(0))
        a['positive']=np.array([1.,0.,0.]);b['positive']=np.array([0.,1.,0.])
        seeds=query_seeds(a,b,np.random.default_rng(0))
        self.assertEqual(seeds[:2],[0,1]);self.assertEqual(len(seeds),len(set(seeds)))

    def test_uniform_and_silent_posteriors_are_direction_neutral(self):
        features=D.acoustic_features(D.acoustic.DIRS.astype(np.float32),np.ones(D.acoustic.N_DIRS))
        np.testing.assert_array_equal(features,np.repeat(features[:1],len(features),axis=0))
        positions=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])*.03266
        silent=D.acoustic.srp_phat_3d(np.zeros((88200,4)),44100,positions)
        self.assertTrue(np.isfinite(silent).all());self.assertTrue((silent==0).all())

    def test_zero_seed_budget_and_empty_association(self):
        from orbit.inference import diverse_seeds,propose_and_score
        self.assertEqual(diverse_seeds(np.zeros((2,3)),np.ones(2),0),[])
        result=propose_and_score({'count':np.ones(2),'silence':True},None,'cpu')
        self.assertFalse(result['masks'].any());self.assertTrue(result['origins'][0].startswith('null:'))

    def test_gate_centering_is_invariant_to_threshold_duplicates(self):
        from orbit.evaluate_audio_gate import candidate_gate_values
        pred=dict(seeds=[0,1],origins=['prompt:0:threshold:.5','prompt:1:threshold:.5','apex'])
        other=dict(seeds=[0,1],origins=['prompt:0:threshold:.5','prompt:0:threshold:.65','prompt:1:threshold:.5','apex'])
        with patch('orbit.evaluate_audio_gate.gate_per_seed',return_value={0:2.,1:-2.}):
            first=candidate_gate_values(None,None,None,pred,'cpu')
            second=candidate_gate_values(None,None,None,other,'cpu')
        np.testing.assert_array_equal(first,second[[0,2,3]])

    def test_posterior_rejects_reordered_rows_audio_calibration_and_array_changes(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'metadata').mkdir();cal=root/'metadata/microphone_array.json';cal.write_text('{}')
            rows=['val/scene/1/0','val/scene/2/0']
            for rel in rows:
                (root/rel).mkdir(parents=True);(root/rel/'audio.wav').write_bytes(rel.encode())
            posterior=root/'posterior.npy';np.save(posterior,np.ones((2,D.acoustic.N_DIRS),np.float32))
            D.write_posterior_identity(posterior,rows,dataset=root)
            D.load_posterior(posterior,rows,dataset=root)
            with self.assertRaisesRegex(ValueError,'order'):D.load_posterior(posterior,rows[::-1],dataset=root)
            wave=root/rows[0]/'audio.wav';before=wave.read_bytes();wave.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'waveform'):D.load_posterior(posterior,rows,dataset=root)
            wave.write_bytes(before);cal.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'calibration'):D.load_posterior(posterior,rows,dataset=root)
            cal.write_text('{}');np.save(posterior,np.zeros((2,D.acoustic.N_DIRS),np.float32))
            with self.assertRaisesRegex(ValueError,'array'):D.load_posterior(posterior,rows,dataset=root)

    def test_utonia_raw_extraction_never_constructs_sonata(self):
        from orbit.predict import ObservationEncoder
        from apex.predict import range_grid
        xyz=np.array([[1.,0,0],[0,1.,0]],np.float32)
        _,_,pixels=range_grid(xyz)
        backbone=Mock();backbone.encode.return_value=(torch.ones(2,256),pixels,np.arange(2))
        audio=Mock();audio.return_value=(torch.ones(1,512),None)
        cal={'positions_lidar_m':[[0,0,0]]*4}
        with patch('orbit.utonia_features.UtoniaFeatures',return_value=backbone),\
             patch('lidarwm.sonata_geometry.SonataGeometry',side_effect=AssertionError('Sonata must not run')),\
             patch('lidarwm.foundation.FrozenClap',return_value=audio),patch('torch.load',return_value={}):
            extractor=ObservationEncoder(cal,'utonia',Path('projection.pt'),'cpu')
            result=extractor(xyz,np.zeros((88200,4),np.float32),44100)
            self.assertTrue(result['silence']);self.assertEqual(len(result['inverse']),2)
            extractor(xyz,np.zeros((88200,4),np.float32),44100)
            self.assertEqual(backbone.encode.call_count,2)


if __name__=='__main__':unittest.main()
