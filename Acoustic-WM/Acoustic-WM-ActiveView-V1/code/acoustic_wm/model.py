"""The two released models, using only their packaged selected checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
from .audio.predictive_features import geometric_tokens
from .world_model.predictive_world import PredictiveAcousticWorld
from .world_model.prototype_emission import load_emission
from .models.predictive_grounding import PredictiveReasoner
from .models.future_grounding import FutureAcousticReasoner, FutureEvidenceHead, initial_belief


class ReleasedModel:
    def __init__(self, directory, device='cuda'):
        self.directory=Path(directory).resolve();self.device=device
        self.config=json.loads((self.directory/'config.json').read_text())
        self.future_enabled=self.config['version'] in ('Acoustic-WM-ActiveView',
                                                       'Acoustic-WM-ActiveView-V1')
        checkpoint=self.directory/'checkpoints'
        microphones=json.loads((self.directory/'assets/sensor/metadata/microphone_array.json').read_text())['positions_lidar_m']
        self.world=PredictiveAcousticWorld(microphones).to(device).eval().requires_grad_(False)
        self.world.load_state_dict(torch.load(checkpoint/'acoustic_world.pt',map_location='cpu',weights_only=False)['model'])
        self.selector=PredictiveReasoner(49).to(device).eval().requires_grad_(False)
        self.selector.load_state_dict(torch.load(checkpoint/'source_selector.pt',map_location='cpu',weights_only=False)['model'])
        self.emission=load_emission(torch.load(checkpoint/'sound_library.pt',map_location='cpu',weights_only=True),device)
        self.pointbind=None
        if self.future_enabled:
            self.head=FutureEvidenceHead().to(device).eval().requires_grad_(False)
            self.head.load_state_dict(torch.load(checkpoint/'future_evidence.pt',map_location='cpu',weights_only=False)['model'])
            self.future=FutureAcousticReasoner(self.world)

    def tensor(self,value):return torch.from_numpy(np.asarray(value)).to(self.device)

    @torch.inference_mode()
    def current(self,row,observed,reliability):
        if all(k in row for k in ['world_points','world_features','world_valid']):
            geometry=[row[k] for k in ['world_points','world_features','world_valid']]
        else:geometry=geometric_tokens(np.asarray(row['xyz']),np.asarray(row['full_point_features']))
        points,features,valid=[self.tensor(x) for x in geometry]
        hypotheses=self.tensor(row['hypotheses']);k,h,_=hypotheses.shape
        obs=torch.view_as_complex(self.tensor(observed).contiguous())
        q,lv,_=self.world(points[None],features[None],valid[None],hypotheses.reshape(1,k*h,3))
        energy=((q[0]-obs).abs().square().sum((-1,-2))*(-lv[0]).exp()+lv[0]).reshape(k,h,48)
        if 'pointbind_feature' in row:body=self.tensor(np.asarray(row['pointbind_feature']).astype(np.float32))
        else:
            from .models.pointbind_adapter import PointBindEncoder,proposal_points
            if self.pointbind is None:self.pointbind=PointBindEncoder(self.device)
            cloud=np.stack([proposal_points(row['xyz'],m) for m in row['masks']]);values=[]
            for begin in range(0,k,16):
                with torch.autocast(device_type=torch.device(self.device).type,dtype=torch.bfloat16,
                                    enabled=torch.device(self.device).type=='cuda'):
                    values.append(self.pointbind.points(self.tensor(cloud[begin:begin+16])).half().float())
            body=torch.cat(values)
        semantic=-self.emission.log_probability(body,self.tensor(row['audio']))
        energy=torch.cat([energy*self.tensor(reliability)[None,None]*48,
                          semantic[:,None,None].expand(-1,h,1)],-1)
        base,sites=initial_belief(self.selector,energy,self.tensor(row['baseline']))
        return dict(base=base,log_site=sites,geometry=(points,features,valid),
                    hypotheses=hypotheses,observed=obs,reliability=self.tensor(reliability),
                    masks=np.asarray(row['masks'],bool),inverse=np.asarray(row.get('inverse',np.arange(len(row['xyz'])))),
                    silent=not bool(np.any(reliability)))

    def result(self,state,score):
        selected=int(score.argmax());mask=state['masks'][selected][state['inverse']].astype(np.uint8)
        probability=mask.astype(np.float32) if self.future_enabled else np.where(mask,1.,np.float32(2.0611537e-9)).astype(np.float32)
        if state['silent']:mask[:]=0;probability[:]=0
        return dict(sound_mask=mask,point_probability=probability,posterior=score.softmax(-1).cpu().numpy(),
                    selected_object=selected,original_selected_object=int(state['base'].argmax()))

    @torch.inference_mode()
    def segment(self,row,observed,reliability):
        if self.future_enabled:raise ValueError('This release requires plan() and complete() with a second observation')
        state=self.current(row,observed,reliability)
        return self.result(state,state['base'])

    @torch.inference_mode()
    def plan(self,row,observed,reliability,actions):
        if not self.future_enabled:raise ValueError('This release uses one observation; call segment()')
        state=self.current(row,observed,reliability)
        action=self.tensor(np.asarray(actions,np.float32))
        if action.ndim!=2 or action.shape[1]!=12 or not len(action) or not torch.isfinite(action).all():
            raise ValueError('Available relative receiver actions must have shape N by 12')
        q,lv=self.future.predict(*state['geometry'],state['hypotheses'],state['observed'],action)
        utility=self.future.information(state['base'],state['log_site'],q,lv,state['reliability'])
        query=int(utility.argmax());state.update(prediction=q[query],log_variance=lv[query],query=query,utilities=utility)
        return state

    @torch.inference_mode()
    def complete(self,state,observed,reliability):
        obs=torch.view_as_complex(self.tensor(observed).contiguous())
        energy=self.future.evidence(state['prediction'],state['log_variance'],obs)
        score=self.head(state['base'],state['log_site'],energy,self.tensor(reliability))
        result=self.result(state,score)
        result.update(chosen_action=state['query'],query_utilities=state['utilities'].cpu().numpy())
        return result
