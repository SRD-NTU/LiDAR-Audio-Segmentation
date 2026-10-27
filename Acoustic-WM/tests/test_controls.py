"""CPU regression checks for the collaborator's ablation and action concerns."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Acoustic-WM-ActiveView/code'))
from acoustic_wm.world_model.predictive_world import PredictiveAcousticWorld
from acoustic_wm.action_protocol import valid_actions,mask_groups,grouped_information


MIC=[[.03266,.03266,.03266],[.03266,-.03266,-.03266],
     [-.03266,.03266,-.03266],[-.03266,-.03266,.03266]]


class WorldAblationTests(unittest.TestCase):
    def test_modes_have_explicit_independent_effects(self):
        torch.manual_seed(7)
        reference=PredictiveAcousticWorld(MIC,'world').eval()
        models={name:PredictiveAcousticWorld(MIC,name).eval()
                for name in ('static','no_assimilation','no_transition')}
        for model in models.values():model.load_state_dict(reference.state_dict())
        state=torch.randn(1,1,128)
        source=torch.tensor([[[1.,0.,0.]]])
        action=torch.cat([torch.tensor([[[.2,0.,0.]]]),torch.eye(3).reshape(1,1,9)],-1)
        with torch.no_grad():
            q,_=reference.decode(state,source)
            observed=q+.01
            counts={name:[0,0] for name in models}
            handles=[]
            for name,model in models.items():
                handles.append(model.assimilation.register_forward_hook(
                    lambda _m,_i,_o,n=name:counts[n].__setitem__(0,counts[n][0]+1)))
                handles.append(model.transition.register_forward_hook(
                    lambda _m,_i,_o,n=name:counts[n].__setitem__(1,counts[n][1]+1)))
            for name,model in models.items():
                memory=model.assimilate(state,q,observed)
                moved_state,moved_source,_,_=model.imagine(memory,source,action)
                self.assertTrue(torch.allclose(moved_source,source-torch.tensor([[[.2,0.,0.]]])))
                if name=='static':self.assertTrue(torch.equal(memory,state))
            for handle in handles:handle.remove()
        self.assertEqual(counts['static'],[0,0])
        self.assertEqual(counts['no_assimilation'],[0,1])
        self.assertEqual(counts['no_transition'],[1,0])
        with self.assertRaises(ValueError):PredictiveAcousticWorld(MIC,'typo_mode')

    def test_pose_toggle_is_separate(self):
        model=PredictiveAcousticWorld(MIC,'no_transition',pose_transform=False)
        state=torch.zeros(1,1,128);source=torch.tensor([[[1.,0.,0.]]])
        action=torch.cat([torch.tensor([[[.2,0.,0.]]]),torch.eye(3).reshape(1,1,9)],-1)
        with torch.no_grad():_,new_source,_,_=model.imagine(state,source,action)
        self.assertTrue(torch.equal(new_source,source))


class ActionProtocolTests(unittest.TestCase):
    def test_action_validity_and_costs(self):
        rotations=np.tile(np.eye(3).reshape(1,9),(2,1))
        actions=np.concatenate([np.array([[1.,0,0],[4.,0,0]]),rotations],1)
        _,valid,costs,distance=valid_actions(actions,costs=[2.,5.],max_translation=3.)
        self.assertEqual(valid.tolist(),[True,False])
        self.assertEqual(costs.tolist(),[2.,5.])
        self.assertEqual(distance.tolist(),[1.,4.])

    def test_grouped_utility_ignores_exact_duplicate(self):
        masks=np.array([[1,0,0],[0,1,0]],dtype=bool)
        duplicate=np.concatenate([masks,masks[:1]])
        q=torch.zeros(2,2,1,48,4,4,dtype=torch.complex64)
        q[:,0]=1.;q[:,1]=2.
        q[1,0]=3.
        lv=torch.zeros(2,2,1,48)
        site=torch.zeros(2,1)
        base=torch.tensor([.3,.1])
        reliability=torch.full((48,),1/48)
        g,d,_=mask_groups(masks)
        expected=grouped_information(base,site,q,lv,reliability,g,d)
        gd,dd,_=mask_groups(duplicate)
        actual=grouped_information(torch.cat([base,base[:1]]),torch.cat([site,site[:1]]),
                                   torch.cat([q,q[:,:1]],1),torch.cat([lv,lv[:,:1]],1),
                                   reliability,gd,dd)
        self.assertTrue(torch.allclose(expected,actual,rtol=1e-5,atol=1e-5))


if __name__=='__main__':unittest.main()
