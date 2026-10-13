import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from orbit.objects import pool_masks, neighbors, GEOM_DIM, ACOUSTIC_DIM
from orbit.masks import PromptMasks, objective
from orbit.selector import ObjectSelector, batch, loss


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(1)

    def test_mask_permutation_and_gradients(self):
        n=29
        f=torch.randn(n,256);xyz=torch.randn(n,3);count=torch.ones(n)
        knn=torch.tensor(neighbors(xyz.numpy()),dtype=torch.long)
        m=PromptMasks().eval()
        enc=m.encode(f,xyz,knn)
        seed=torch.tensor([2,11])
        out=m.decode(enc,xyz,count,seed)
        targets=torch.zeros(2,n);targets[0,:5]=1;targets[1,10:15]=1
        category=torch.randint(0,17,(n,));ids=torch.arange(n)%4-1
        value=objective(enc,out,targets,count,category,ids)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters()))
        perm=torch.randperm(n);inverse=torch.argsort(perm)
        other=m.encode(f[perm],xyz[perm],inverse[knn[perm]])
        pred=m.decode(other,xyz[perm],count[perm],inverse[seed])
        torch.testing.assert_close(out['logits'][:,perm],pred['logits'],atol=2e-5,rtol=2e-5)

    def test_selector_padding_and_gradients(self):
        r=dict(geometry=np.ones((3,GEOM_DIM),np.float32),acoustic=np.ones((3,ACOUSTIC_DIM),np.float32),
               audio=np.ones(512,np.float32),categories=np.array([0,1,16]),target=1,source=1)
        inputs,labels=batch([r], 'cpu')
        m=ObjectSelector().eval()
        out=m(*inputs)
        value=loss(out,labels,torch.ones(17));value.backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters()))
        g,a,s,v=inputs
        other=m(torch.cat([g,torch.zeros(1,2,GEOM_DIM)],1),torch.cat([a,torch.zeros(1,2,ACOUSTIC_DIM)],1),s,
                torch.cat([v,torch.zeros(1,2,dtype=torch.bool)],1))
        torch.testing.assert_close(out['scores'],other['scores'][:,:3])
        self.assertTrue((other['scores'][:,3:] < -999).all())

    def test_neural_selector_exposes_audio_logits_to_query_gate(self):
        model=ObjectSelector().eval()
        audio=torch.randn(2,512)
        torch.testing.assert_close(model.sound_logits(audio),model.audio_class(model.sound(audio)))

    def test_physics_calibration_uses_acoustic_feature_after_geometry(self):
        from orbit.physics_calibrate import score
        features=np.zeros((2,35),np.float32)
        features[:,16]=[100,-100]  # Different acoustic slot, not the intended cue.
        features[:,28]=[.2,.5]    # Acoustic slot 16 follows 12 geometry values.
        features[:,10]=[.1,.3]
        row=dict(features=features,baseline=np.zeros(2,np.float32))
        np.testing.assert_allclose(score(row,1,0),[2.8,7.])
        np.testing.assert_allclose(score(row,0,1),[1.,3.])

    def test_portable_audio_bank_checks_observation_order(self):
        from apex import data as D
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'metadata').mkdir()
            rels=['train/scene/pose/00000','train/scene/pose/00001']
            (root/'metadata/train.json').write_text(json.dumps(rels))
            (root/'metadata/microphone_array.json').write_text('{}')
            bank=root/'audio-bank';bank.mkdir()
            arrays=dict(rels=np.asarray(rels),post=np.ones((2,D.acoustic.N_DIRS),np.float32),
                        emb=np.ones((2,512),np.float32))
            np.savez(bank/'train.npz',**arrays)
            with patch.object(D,'DATA',root),patch.object(D,'AUDIO_BANK',bank):
                loaded,post,emb,provenance=D.audio_bank('train')
                self.assertEqual(loaded,rels)
                self.assertEqual(post.shape,(2,D.acoustic.N_DIRS))
                self.assertEqual(emb.shape,(2,512))
                self.assertIn('audio_bank_sha256',provenance)
                np.savez(bank/'train.npz',**{**arrays,'rels':np.asarray(rels[::-1])})
                with self.assertRaisesRegex(ValueError,'observation order'):
                    D.audio_bank('train')

    def test_pooling_point_counts(self):
        x=dict(features=np.stack([np.ones(256),np.ones(256)*3]).astype(np.float32),
               xyz=np.array([[0,0,1],[0,0,3]],np.float32),acoustic=np.ones((2,8),np.float32))
        g,a=pool_masks(x,[(np.array([0,1]),np.array([3,1]))])
        self.assertEqual(g.shape,(1,GEOM_DIM));self.assertEqual(a.shape,(1,ACOUSTIC_DIM))
        self.assertAlmostEqual(float(g[0,0]),1.5)
        self.assertAlmostEqual(float(g[0,514]),.15,places=6)
        self.assertTrue(np.isfinite(g).all() and np.isfinite(a).all())

    def test_observation_only_inference(self):
        from orbit.inference import propose_and_score
        from apex.model import APEX
        rng=np.random.default_rng(13);n=51
        x=dict(features=rng.normal(size=(n,256)).astype(np.float32),
               xyz=rng.normal(size=(n,3)).astype(np.float32),count=np.ones(n,np.float32)*4,
               acoustic=rng.uniform(size=(n,8)).astype(np.float32),audio=rng.normal(size=512).astype(np.float32))
        models=(PromptMasks().eval(),ObjectSelector().eval(),APEX().eval(),dict(mask_threshold=.5,apex_threshold=.6))
        out=propose_and_score(x,models,device='cpu',n_seeds=6)
        self.assertEqual(out['masks'].shape[1],n)
        self.assertTrue(out['masks'].dtype==bool)
        self.assertTrue(all(0<=i<len(out['masks']) for i in out['choices'].values()))
        from orbit.audio_gate import AudioSeedGate
        from orbit.evaluate_audio_gate import gate_per_seed
        gate_scores=gate_per_seed(x,models,AudioSeedGate().eval(),out['seeds'][:3],'cpu')
        self.assertEqual(set(gate_scores),set(out['seeds'][:3]))
        self.assertTrue(np.isfinite(list(gate_scores.values())).all())

    def test_seed_order_stable_under_small_numerical_perturbation(self):
        from orbit.inference import diverse_seeds
        xyz=np.array([[0.,0,0],[1.,0,0],[2.,0,0],[3.,0,0]],np.float32)
        score=np.array([1.,1.00002,.7,.6],np.float32)
        perturb=np.array([.00002,-.00002,0,0],np.float32)
        self.assertEqual(diverse_seeds(xyz,score,3),diverse_seeds(xyz,score+perturb,3))
        self.assertEqual(diverse_seeds(xyz,score,2,excluded=[0]),[1,2])

    def test_audio_query_scores_objects_and_backpropagates(self):
        from orbit.audio_gate import AudioSeedGate
        n=23;k=4
        encoded=dict(h=torch.randn(n,96),semantic=torch.randn(n,17),objectness=torch.randn(n))
        decoded=dict(logits=torch.randn(k,n),quality=torch.randn(k))
        gate=AudioSeedGate()
        scores=gate(encoded,decoded,torch.randn(n,8),torch.randn(512),torch.ones(n),
                    torch.arange(k),torch.softmax(torch.randn(16),dim=0))
        self.assertEqual(tuple(scores.shape),(k,))
        torch.nn.functional.cross_entropy(scores[None],torch.tensor([0])).backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in gate.parameters()))

    def test_multiband_direction_sign(self):
        from orbit.spatial import BandSRP
        from apex.data import acoustic
        rate=44100;n=11025
        positions=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])*.03266
        direction=np.array([.8,.5,.3]);direction/=np.linalg.norm(direction)
        freq=np.fft.rfftfreq(n,1/rate)
        spectrum=np.fft.rfft(np.random.default_rng(31).normal(size=n))
        spectrum[(freq<150)|(freq>6000)]=0
        delays=-(positions@direction)/343
        wave=np.stack([np.fft.irfft(spectrum*np.exp(-2j*np.pi*freq*t),n=n) for t in delays],1)
        scores,coherence=BandSRP(positions,device='cpu')(wave[None])
        peak=acoustic.DIRS[scores[0].mean(0).argmax()]
        angle=np.degrees(np.arccos(np.clip(peak@direction,-1,1)))
        self.assertLess(angle,5)
        self.assertTrue(np.isfinite(scores).all() and np.isfinite(coherence).all())

    def test_ranking_loss_ignores_padding(self):
        logits=torch.tensor([[1.,2.,3.]],requires_grad=True)
        target=torch.tensor([1])
        a=torch.nn.functional.cross_entropy(logits,target)
        b=torch.nn.functional.cross_entropy(torch.cat([logits,torch.full((1,7),-1e4)],1),target)
        torch.testing.assert_close(a,b)

    def test_reranker_gradients(self):
        from orbit.rerank import Reranker
        model=Reranker();f=torch.randn(4,8,35);base=torch.randn(4,8)
        loss=torch.nn.functional.cross_entropy(model(f,base),torch.tensor([1,2,3,4]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__':unittest.main()
