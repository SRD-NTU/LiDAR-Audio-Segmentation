import unittest
import numpy as np
import torch
from scipy.signal import chirp
from graph_audio.features import MICROPHONES, acoustic_field
from graph_audio.model import AcousticGraphWorld, world_loss
from graph_audio.metrics import score, THRESHOLDS,threshold_weighted_scores
from evaluate_teacher_refinement import threshold_scores
from train_teacher_trees import teacher_inputs
from train_graph import corrupt_teacher_prior,prefetched_cases,device_tensors
from evaluate_audio_ablations import missing_audio, swapped_audio
from graph_audio.spatial_audio import SpatialReliability, direction_grid, grid_field,initialize_world_reliability,scene_field


class ObservationModelTests(unittest.TestCase):
    def test_pretrained_runtime_caches_are_confined_to_this_project(self):
        import os
        from unittest.mock import patch
        from graph_audio.local_assets import configure_runtime,ROOT
        from pathlib import Path
        with patch.dict(os.environ,clear=False):
            configure_runtime()
            for key in ('XDG_CACHE_HOME','HF_HOME','TORCH_EXTENSIONS_DIR','TORCH_HOME','NUMBA_CACHE_DIR'):
                self.assertTrue(Path(os.environ[key]).is_relative_to(ROOT/'cache'))
            self.assertEqual(os.environ['HF_HUB_OFFLINE'],'1')
    def test_prefetch_preserves_order_and_does_not_move_worker_data_to_cuda(self):
        from unittest.mock import patch
        names=['one','two','three']
        calls=[]
        def loader(name,device,use_teacher):
            calls.append((name,device,use_teacher))
            return dict(features=torch.tensor([names.index(name)])),dict(counts=torch.tensor([7]))
        with patch('train_graph.load',loader):
            rows=list(prefetched_cases(names,False))
        self.assertEqual([r[0] for r in rows],names)
        self.assertEqual(calls,[(name,'cpu',False) for name in names])
        self.assertEqual(list(prefetched_cases([],False)),[])
        for _,observation,_ in rows:
            torch.testing.assert_close(device_tensors(observation,'cpu')['features'],observation['features'])
    def test_spatial_head_is_physics_constrained_and_planar_ambiguity_is_explicit(self):
        grid=direction_grid()
        self.assertTrue((grid[:,2]>=0).all())
        np.testing.assert_allclose(np.linalg.norm(grid,axis=1),1,atol=1e-6)
        tables=np.random.default_rng(55).normal(size=(6,9,161)).astype(np.float32)
        field=grid_field(tables)
        self.assertEqual(field.shape,(len(grid),54))
        model=SpatialReliability()
        weights,temperature=model(torch.rand(3,54))
        torch.testing.assert_close(weights.sum(-1),torch.ones(3))
        self.assertTrue((weights>0).all())
        self.assertTrue(((temperature>=2)&(temperature<=50)).all())
        torch.testing.assert_close(weights,torch.ones_like(weights)/54)
        centers=grid[:10]*2
        mirrored=centers.copy(); mirrored[:,2]*=-1
        np.testing.assert_allclose(np.linalg.norm(centers[:,None]-MICROPHONES,axis=-1),
                                   np.linalg.norm(mirrored[:,None]-MICROPHONES,axis=-1),atol=1e-7)
        world=AcousticGraphWorld(dim=32,layers=1).eval()
        with torch.no_grad():
            model.head[-1].weight.normal_()
            model.head[-1].bias.normal_()
        initialize_world_reliability(world,model)
        coherence=torch.rand(3,54)
        prior_and_levels=torch.randn(3,21)
        expected,_=model(coherence)
        actual=world.reliability(torch.cat((prior_and_levels,coherence),-1)).softmax(-1)
        torch.testing.assert_close(actual,expected)
        uniform=SpatialReliability().eval()
        node_field=np.random.default_rng(66).normal(size=(15,54)).astype(np.float32)
        np.testing.assert_allclose(scene_field(node_field,tables,uniform),node_field.mean(1),atol=1e-6)
    def test_audio_interventions_preserve_geometry_and_do_not_copy_donor_nodes(self):
        observations=dict(features=torch.randn(10,268),acoustic=torch.randn(10,54),
            embedding=torch.randn(512),text_logits=torch.randn(16),audio_context=torch.randn(5),
            delay_tables=torch.randn(6,9,161),learned_audio_logits=torch.randn(16))
        donor=dict(embedding=torch.randn(512),text_logits=torch.randn(16),audio_context=torch.randn(5),
            delay_tables=torch.randn(6,9,161),learned_audio_logits=torch.randn(16),acoustic=torch.randn(27,54))
        shuffled=swapped_audio(observations,donor)
        missing=missing_audio(observations)
        self.assertIs(shuffled['features'],observations['features'])
        self.assertIs(missing['features'],observations['features'])
        self.assertEqual(shuffled['acoustic'].shape,(10,54))
        self.assertIs(shuffled['delay_tables'],donor['delay_tables'])
        self.assertFalse(torch.equal(missing['embedding'],observations['embedding']))
        self.assertGreater(float(observations['embedding'].abs().sum()),0)
    def test_counterfactual_world_is_geometry_only_and_audio_changes_selection(self):
        torch.set_num_threads(1); torch.manual_seed(45)
        n=12
        model=AcousticGraphWorld(dim=32,layers=1).eval()
        observed=dict(features=torch.randn(n,268),centers=torch.randn(n,3),
            neighbors=torch.roll(torch.arange(n),1)[:,None],edge_valid=torch.ones(n,1),
            acoustic=torch.randn(n,54),embedding=torch.randn(512),text_logits=torch.randn(16),
            audio_context=torch.randn(5),delay_tables=torch.randn(6,9,161),
            learned_audio_logits=torch.randn(16),teacher_geometry=torch.randn(n,41))
        other=dict(observed)
        for key in ('acoustic','embedding','text_logits','audio_context','delay_tables','learned_audio_logits'):
            other[key]=torch.randn_like(other[key])
        first,second=model(**observed),model(**other)
        world=model.construct_world(*(observed[k] for k in ('features','centers','neighbors','edge_valid')),observed['teacher_geometry'])
        cached=model(**observed,world_state=world)
        for key in ('logits','rendered','semantic','predicted_centers'):
            torch.testing.assert_close(first[key],cached[key],rtol=0,atol=0)
        for key in ('rendered','semantic','predicted_centers'):
            torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
        self.assertFalse(torch.allclose(first['logits'],second['logits']))
        model.train()
        with self.assertRaises(ValueError):
            model(**observed,world_state=world)
    def test_report_refresh_does_not_shadow_metric_aggregation(self):
        import evaluate_teacher_refinement,train_teacher_trees,train_graph
        for module in (evaluate_teacher_refinement,train_teacher_trees,train_graph):
            self.assertNotIn('summarize',module.main.__code__.co_varnames)
    def test_teacher_corruption_is_training_only_and_observation_derived(self):
        observations = dict(centers=torch.randn(20,3),teacher_probability=torch.linspace(.01,.99,20))
        untouched = observations['teacher_probability'].clone()
        modified = corrupt_teacher_prior(observations,2.,np.random.default_rng(1))
        torch.testing.assert_close(observations['teacher_probability'],untouched)
        self.assertFalse(torch.equal(modified['teacher_probability'],untouched))
        self.assertTrue(torch.isfinite(modified['teacher_probability']).all())
        self.assertIs(corrupt_teacher_prior(observations,0.,np.random.default_rng(1)),observations)
    def test_teacher_feature_builder_ignores_annotation_fields(self):
        rng = np.random.default_rng(44)
        n = 16
        graph = dict(counts=np.ones(n)*10,features=rng.normal(size=(n,268)).astype(np.float32),
            acoustic=rng.normal(size=(n,54)).astype(np.float32),text_logits=rng.normal(size=16),
            audio_context=rng.normal(size=5),neighbors=np.roll(np.arange(n),1)[:,None],edge_valid=np.ones((n,1)))
        teacher = dict(geometry=rng.normal(size=(n,41)),probability=rng.uniform(.01,.99,n))
        before = teacher_inputs(graph,teacher)
        graph.update(positives=np.ones(n)*10,semantic=np.ones(n)*16,instance=np.arange(n))
        np.testing.assert_array_equal(before,teacher_inputs(graph,teacher))
        self.assertEqual(before.shape,(n,483))
    def test_histogram_threshold_metrics_are_exact(self):
        probability = np.concatenate((np.random.default_rng(42).random(30), THRESHOLDS))
        target = np.random.default_rng(43).integers(0,2,len(probability))
        actual = threshold_scores(probability,target)
        expected = [score(probability,np.ones(len(target)),target,t) for t in THRESHOLDS]
        self.assertEqual(actual,expected)
        counts = np.random.default_rng(45).integers(1,10,len(probability))
        positive = target*counts
        self.assertEqual(threshold_weighted_scores(probability,counts,positive),
                         [score(probability,counts,positive,t) for t in THRESHOLDS])
    def test_phase_sign_and_channel_calibration(self):
        source = np.array([1.5, .8, .2])
        nodes = np.stack((source, -source))
        time = np.arange(44100) / 44100
        signal = chirp(time, f0=220, f1=3300, t1=1) * np.hanning(len(time))
        distance = np.linalg.norm(source[None] - MICROPHONES, axis=1)
        audio = np.column_stack([np.interp(time - d / 343, time, signal, left=0, right=0) for d in distance])
        field, _ = acoustic_field(nodes, audio.astype(np.float32))
        self.assertGreater(field[0].mean(), field[1].mean() + .1)

    def test_isolated_nodes_have_finite_predictions_and_gradients(self):
        torch.set_num_threads(1); torch.manual_seed(42)
        n, k = 16, 3
        observations = dict(features=torch.randn(n, 268), centers=torch.randn(n, 3),
             neighbors=torch.stack([torch.roll(torch.arange(n), i) for i in range(1, k + 1)], 1),
             edge_valid=torch.zeros(n, k), acoustic=torch.randn(n, 54),
             embedding=torch.randn(512), text_logits=torch.randn(16), audio_context=torch.randn(5))
        model = AcousticGraphWorld(dim=32, layers=1)
        output = model(**observations)
        targets = dict(counts=torch.ones(n) * 10, positives=torch.cat((torch.ones(4) * 10, torch.zeros(n - 4))),
                       semantic=torch.ones(n).long(), instance=torch.arange(n), embedding=observations['embedding'])
        losses = world_loss(output, targets, observations['neighbors'], observations['edge_valid'])
        losses['total'].backward()
        self.assertTrue(torch.isfinite(output['logits']).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_center_phase_rendering_is_differentiable(self):
        torch.set_num_threads(1); torch.manual_seed(43)
        n = 16
        model = AcousticGraphWorld(dim=32, layers=1)
        observations = dict(features=torch.randn(n, 268), centers=torch.randn(n, 3),
                neighbors=torch.stack((torch.roll(torch.arange(n), 1), torch.roll(torch.arange(n), 2)), 1),
                edge_valid=torch.ones(n, 2), acoustic=torch.randn(n, 54),
                embedding=torch.randn(512), text_logits=torch.randn(16), audio_context=torch.randn(5),
                delay_tables=torch.randn(6, 9, 161))
        output = model(**observations)
        output['logits'].square().mean().backward()
        gradient = model.center_offsets.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_zero_residual_preserves_teacher_and_gate_gets_gradient(self):
        torch.set_num_threads(1); torch.manual_seed(44)
        n = 12
        model = AcousticGraphWorld(dim=32, layers=1).eval()
        probability = torch.linspace(.01, .99, n)
        observations = dict(features=torch.randn(n, 268), centers=torch.randn(n, 3),
            neighbors=torch.roll(torch.arange(n), 1)[:, None], edge_valid=torch.ones(n, 1),
            acoustic=torch.randn(n, 54), embedding=torch.randn(512), text_logits=torch.randn(16),
            audio_context=torch.randn(5), teacher_geometry=torch.randn(n, 41), teacher_probability=probability)
        output = model(**observations)
        torch.testing.assert_close(output['logits'].sigmoid(), probability)
        torch.testing.assert_close(output['logit_delta'], torch.zeros(n))
        output['logits'].sum().backward()
        self.assertTrue(torch.isfinite(model.refinement_scale.grad))
        self.assertGreater(float(model.refinement_scale.grad.abs()), 0.)


if __name__ == '__main__':
    unittest.main()
