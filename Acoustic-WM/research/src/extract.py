"""One persistent GPU worker. Forecasts are computed before opening future audio.

RIR/dry audio rendering is offline acquisition, never a planner input. Labels
are opened only after all predictions, actions, and observation features exist.
"""
from common import *
import argparse,fcntl,time,traceback,math
import numpy as np,torch,soundfile as sf
from scipy.signal import fftconvolve,resample_poly
from protocol import valid_actions,mask_groups,grouped_information
from acoustic_wm.model import ReleasedModel
from acoustic_wm.audio.predictive_features import covariance48_from_wave,geometric_tokens
from acoustic_wm.models.pointbind_adapter import PointBindEncoder,proposal_points
from acoustic_wm.world_model.predictive_world import move_source

VARIANTS=['repeat_current','analytic_direct','no_assimilation','no_transition','world']
POLICIES=['fixed','random','farthest','legacy_active','grouped_active','analytic_active']

def pose(metadata):
 centre=np.asarray(metadata['listener']['position_world'],np.float64)
 array=metadata['audio']['microphone_array'];local=np.asarray(array['positions_lidar_m']);world=np.asarray(array['positions_world_m'])
 rt,_,rank,_=np.linalg.lstsq(local,world-centre,rcond=None)
 if rank!=3 or not np.allclose(local@rt+centre,world,atol=1e-5):raise ValueError('Invalid calibrated receiver pose')
 return centre,rt.T

def actions_for(metadata,index):
 pos,rot=pose(metadata[index]);others=[i for i in range(len(metadata)) if i!=index]
 p,r=zip(*(pose(metadata[i]) for i in others));p=np.stack(p);r=np.stack(r)
 return np.concatenate([(p-pos)@rot,(rot.T[None]@r).reshape(-1,9)],1).astype(np.float32),others

class CacheGeometry:
 def __init__(self,original):self.original=original;self.path=None
 def encode(self,xyz):
  if self.path is None or not self.path.exists():return self.original.encode(xyz)
  with np.load(self.path) as z:
   return torch.as_tensor(z['features'].astype(np.float32),device='cuda'),z['pixels'],z['inverse']

def encoder():
 from acoustic_wm.frontend import RawObservationEncoder
 cfg=json.loads((ACTIVE/'config.json').read_text())
 cfg['paths']['dataset']=str(DATA)
 enc=RawObservationEncoder(cfg,'cuda');enc.geometry=CacheGeometry(enc.geometry)
 return enc

def reference(enc,model,sample):
 path=ROOT/'cache/reference'/(sample+'.npz')
 if path.exists():
  with np.load(path) as z:return {k:z[k] for k in z.files}
 with np.load(DATA/sample/'lidar.npz') as z:xyz=z['xyz'].astype(np.float32)
 wave,rate=sf.read(DATA/sample/'audio.wav',dtype='float32',always_2d=True)
 if rate!=44100:raise ValueError('Expected 44100 Hz')
 geo='utonia-heldout-test-v1' if sample.startswith('test/') else 'utonia-xyz-v1'
 enc.geometry.path=None  # Recompute geometry locally; archived cache remains optional.
 row=enc.encode_arrays(xyz,wave,rate)
 clouds=np.stack([proposal_points(row['xyz'],m) for m in row['masks']]);body=[]
 for start in range(0,len(clouds),16):
  with torch.autocast('cuda',dtype=torch.bfloat16):body.append(model.pointbind.points(model.tensor(clouds[start:start+16])).half().float())
 row['pointbind_feature']=torch.cat(body).cpu().numpy()
 row['world_points'],row['world_features'],row['world_valid']=geometric_tokens(row['xyz'],row['full_point_features'])
 row['observed'],row['reliability']=covariance48_from_wave(wave)
 # Store only observation-derived model inputs. No source IDs, labels or RIRs.
 save_npz(path,**row)
 return row

def acquire_episode(enc,metadata,episode,index):
 m=metadata[index];key=m['audio']['dry_audio_sha256']+'-'+str(m['audio']['duration_sec'])
 path=ROOT/'cache/acquisition'/episode/(key+'.npz')
 if path.exists():
  with np.load(path) as z:return {k:z[k] for k in z.files}
 dry,rate=sf.read(DATA/m['audio']['dry_audio_file'],dtype='float32',always_2d=True);dry=dry.mean(1)
 if rate!=44100:
  g=math.gcd(rate,44100);dry=resample_poly(dry,44100//g,rate//g).astype(np.float32)
 n=round(float(m['audio']['duration_sec'])*44100)
 if len(dry)>n:
  cs=np.r_[0.,np.cumsum(dry.astype(np.float64)**2)];start=int((cs[n:]-cs[:-n]).argmax());dry=dry[start:start+n]
 else:dry=np.pad(dry,(0,n-len(dry)))
 observations=[];weights=[];embeddings=[];transfers=[]
 for j,other in enumerate(metadata):
  # Offline sensor provider reads supervision assets; only wave features escape.
  rir=np.load(DATA/episode/f'{j:05d}'/other['audio']['rir_file'])
  wave=np.stack([fftconvolve(dry,h)[:n] for h in rir],1).astype(np.float32)
  obs,w=covariance48_from_wave(wave)
  observations.append(obs);weights.append(w);embeddings.append(enc.clap([wave],44100)[0].cpu().numpy())
 value=dict(observed=np.stack(observations),weight=np.stack(weights),audio=np.stack(embeddings))
 save_npz(path,**value);return value

@torch.inference_mode()
def forecasts(model,state,actions):
 p,f,v=state['geometry'];hyp=state['hypotheses'];k,h,_=hyp.shape;a=model.tensor(actions)
 src=hyp.reshape(1,-1,3);q0,lv0,prior=model.world(p[None],f[None],v[None],src)
 memory=model.world.assimilate(prior,q0,state['observed'][None,None].expand_as(q0))
 act=a[:,None].expand(-1,k*h,-1);s=src.expand(len(a),-1,-1);m=memory.expand(len(a),-1,-1);prior=prior.expand_as(m)
 _,moved,qw,lw=model.world.imagine(m,s,act)
 _,_,qa,la=model.world.imagine(prior,s,act)
 qn,ln=model.world.decode(m,moved)
 direct=model.world.direct(moved)
 repeat=state['observed'][None,None].expand_as(qw)
 q=torch.stack([repeat,direct,qa,qn,qw]).reshape(5,len(a),k,h,48,4,4)
 lv=torch.stack([torch.full_like(lw,-2),torch.full_like(lw,-2),la,ln,lw]).reshape(5,len(a),k,h,48)
 return q,lv

@torch.inference_mode()
def extract_one(enc,model,sample,metadata,index):
 row=reference(enc,model,sample);state=model.current(row,row['observed'],row['reliability'])
 actions,others=actions_for(metadata,index);actions,valid,cost,distance=valid_actions(actions,max_translation=3.)
 q,lv=forecasts(model,state,actions)
 legacy=model.future.information(state['base'],state['log_site'],q[-1],lv[-1],state['reliability'])
 group_values={};group_stats={}
 for threshold in [.8,.9,.95,1.]:
  ids,duplicates,stats=mask_groups(row['masks'],threshold)
  group_values[str(threshold)]=grouped_information(state['base'],state['log_site'],q[-1],lv[-1],state['reliability'],ids,duplicates)
  group_stats[str(threshold)]=stats
  if threshold==.9:groups=ids;dups=duplicates
 direct=model.future.information(state['base'],state['log_site'],q[1],lv[1],state['reliability'])
 available=np.flatnonzero(valid);queries=np.full(len(POLICIES),-1,np.int64)
 if len(available):
  def best(u):return int(u.masked_fill(~model.tensor(valid),-torch.inf).argmax())
  queries=np.array([available[0],available[stable_int(sample)%len(available)],available[np.argmax(distance[available])],best(legacy),best(group_values['0.9']),best(direct)])
 budget_utilities=[]
 for budget in [32,64,128,len(row['masks'])]:
  select=state['base'].argsort(descending=True)[:min(budget,len(row['masks']))]
  gg,dd,_=mask_groups(row['masks'][select.cpu().numpy()],.9)
  budget_utilities.append(grouped_information(state['base'][select],state['log_site'][select],q[-1].index_select(1,select),lv[-1].index_select(1,select),state['reliability'],gg,dd).cpu().numpy())
 # Explicit planning/acquisition boundary: q, utilities, and queries exist first.
 prediction_done=time.time()
 acquired=acquire_episode(enc,metadata,str(Path(sample).parent),index)
 obs=model.tensor(acquired['observed'][others]);target=torch.view_as_complex(obs.contiguous())
 weight=model.tensor(acquired['weight'][others]);energy=model.future.evidence(q,lv,target[None,:,None,None])
 error=(q-target[None,:,None,None]).abs().square().sum((-1,-2))
 # Future semantic evidence is computed for the SAME reference candidate bank.
 body=model.tensor(row['pointbind_feature']);audio=model.tensor(acquired['audio'][others]);sem=torch.stack([-model.emission.log_probability(body,x) for x in audio])
 # Equal-information fusion receives both full covariances, semantic compatibility,
 # candidate supports, rank descriptors and the relative receiver pose.
 current_sem=-model.emission.log_probability(body,model.tensor(row['audio']))
 standard=np.concatenate([np.broadcast_to(row['rank_features'][None],(len(actions),*row['rank_features'].shape)),
  np.broadcast_to(row['hypotheses'].reshape(1,len(body),-1)/10,(len(actions),len(body),12)),
  np.broadcast_to(state['base'].cpu().numpy()[None,:,None],(len(actions),len(body),1)),
  np.broadcast_to(current_sem.cpu().numpy()[None,:,None],(len(actions),len(body),1)),sem.cpu().numpy()[...,None],
  energy[1].float().mean(2).cpu().numpy()],-1).astype(np.float32)
 # Duplication stress test uses only pre-action quantities and duplicates top prior.
 top=int(state['base'].argmax());ix=torch.cat([torch.arange(len(body),device='cuda'),torch.full((4,),top,device='cuda')])
 # Avoid advanced-index axis permutation by explicit index_select.
 dq=q[-1].index_select(1,ix);dlv=lv[-1].index_select(1,ix)
 dup_legacy=model.future.information(state['base'][ix],state['log_site'][ix],dq,dlv,state['reliability'])
 dg,dd,_=mask_groups(row['masks'][ix.cpu().numpy()],.9)
 dup_group=grouped_information(state['base'][ix],state['log_site'][ix],dq,dlv,state['reliability'],dg,dd)
 # Labels first enter here, strictly after forecasts and selection.
 with np.load(DATA/sample/'lidar.npz') as z:truth=z['sound_mask'].astype(bool)
 mapping=row['inverse'];masks=row['masks'];positive=np.bincount(mapping[truth],minlength=masks.shape[1]);negative=np.bincount(mapping[~truth],minlength=masks.shape[1])
 inter=masks@positive;union=truth.sum()+masks@negative;iou=np.divide(inter,union,out=np.ones(len(masks)),where=union>0).astype(np.float32)
 sources=np.asarray(metadata[index]['relative_source']['position_lidar'],np.float32)
 # Separate true-source diagnostic; only source-supervised world evaluation.
 # Candidate-based future metrics below do not use these source coordinates.
 best_candidate=int(iou.argmax());site=state['log_site'].exp()
 candidate_mse=(error*weight[None,:,None,None,:]*site[None,None,:,:,None]).sum((-1,-2))
 predicted_var=(lv.exp()*weight[None,:,None,None,:]*site[None,None,:,:,None]).sum((-1,-2))
 result=dict(schema=np.int64(2),null_iou=np.float32(not truth.any()),reference_foreground_points=np.int64(truth.sum()),reference_point_count=np.int64(len(truth)),base=state['base'].cpu().numpy(),sites=state['log_site'].cpu().numpy(),energy=energy.cpu().numpy(),weight=weight.cpu().numpy(),
  iou=iou,actions=actions,valid=valid,cost=cost,queries=queries,legacy_utilities=legacy.cpu().numpy(),group_utilities=np.stack([u.cpu().numpy() for u in group_values.values()]),
  candidate_budget_utilities=np.stack(budget_utilities),direct_utilities=direct.cpu().numpy(),duplicate_legacy_utilities=dup_legacy.cpu().numpy(),duplicate_group_utilities=dup_group.cpu().numpy(),groups=groups,duplicates=dups,
  observed=row['observed'],future_observed=acquired['observed'][others],current_audio=row['audio'],future_audio=acquired['audio'][others],standard=standard,
  candidate_mse=candidate_mse.cpu().numpy(),predicted_variance=predicted_var.cpu().numpy(),group_counts=np.array([s['groups'] for s in group_stats.values()]),
  prior_candidate_forecast=torch.view_as_real(q[-1,queries[4] if queries[4]>=0 else 0,best_candidate]).cpu().numpy(),
  prediction_completed_at=np.array(prediction_done),acquisition_completed_at=np.array(time.time()),oracle_candidate=np.array(best_candidate),
  source_for_supervised_world_training=sources,reference_world_points=row['world_points'],reference_world_features=row['world_features'],reference_world_valid=row['world_valid'],reference_reliability=row['reliability'],
  receiver=np.asarray(pose(metadata[index])[0],np.float32),rotation=np.asarray(pose(metadata[index])[1],np.float32))
 save_npz(ROOT/'cache/controlled'/(sample+'.npz'),**result)
 return dict(sample=sample,candidates=len(masks),groups=len(set(groups)),valid_actions=int(valid.sum()),queries=queries.tolist())

def cache_complete(path):
 if not path.exists():return False
 with np.load(path) as z:return 'schema' in z and int(z['schema'])==2

def worker(rank=0,limit=0):
 seed_all();enc=encoder();model=ReleasedModel(ACTIVE,'cuda');model.pointbind=PointBindEncoder('cuda')
 episodes={}
 for split in ['train','val','test']:
  for sample in json.loads((ROOT/'manifests'/f'{split}.json').read_text()):episodes.setdefault(str(Path(sample).parent),[]).append(sample)
 tasks=sorted(episodes,key=lambda x:({'train':0,'val':1,'test':2}[x.split('/')[0]],x));tasks=tasks[rank%len(tasks):]+tasks[:rank%len(tasks)];done=0
 for episode in tasks:
  samples=sorted(episodes[episode]);paths=[ROOT/'cache/controlled'/(r+'.npz') for r in samples]
  if all(cache_complete(p) for p in paths):continue
  lock=ROOT/'cache/locks'/(episode+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
  with lock.open('a') as stream:
   try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   metadata=[json.loads((DATA/r/'metadata.json').read_text()) for r in samples]
   for index,sample in enumerate(samples):
    if cache_complete(paths[index]):continue
    seed_all()
    try:
     result=extract_one(enc,model,sample,metadata,index);done+=1
     write_json(ROOT/'reports/workers'/f'{rank}.json',dict(**result,done=done,updated=time.time()));print(json.dumps(result),flush=True)
    except Exception:
     write_json(ROOT/'reports/errors'/f'extract-{rank}.json',dict(sample=sample,traceback=traceback.format_exc()));raise
    if limit and done>=limit:return
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,default=0);p.add_argument('--limit',type=int,default=0);a=p.parse_args();worker(a.rank,a.limit)
