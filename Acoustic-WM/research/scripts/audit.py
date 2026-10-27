from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import *
import numpy as np,datetime

def main():
 manifests={s:json.loads((DATA/'metadata'/f'{s}.json').read_text()) for s in ['train','val','test']}
 scenes={s:sorted({r.split('/')[1] for r in rows}) for s,rows in manifests.items()}
 assert not any(set(scenes[a])&set(scenes[b]) for a,b in [('train','val'),('train','test'),('val','test')])
 groups={};identities={};audio_sets={s:set() for s in manifests}
 for split,rows in manifests.items():
  for rel in rows:
   m=json.loads((DATA/rel/'metadata.json').read_text());e=str(Path(rel).parent)
   pose=[*m['listener']['position_world'],*m['listener']['rotation_world']]
   groups.setdefault(e,[]).append(pose)
   identities[rel]={'scene':m['scene_id'],'source_instance':m['source']['instance_id'],'dry_audio_sha256':m['audio']['dry_audio_sha256'],'metadata_sha256':sha(DATA/rel/'metadata.json')}
   audio_sets[split].add(m['audio']['dry_audio_sha256'])
 signatures={e:hashlib.sha256(json.dumps(sorted(p),sort_keys=True).encode()).hexdigest() for e,p in groups.items()}
 per_scene={}
 for e,sig in signatures.items():per_scene.setdefault('/'.join(e.split('/')[:2]),set()).add(sig)
 gen=DATA/'provenance/generator/audiolidar/benchmark/generate.py'
 lines=gen.read_text().splitlines();evidence=[{'line':i+1,'text':s.strip()} for i,s in enumerate(lines) if "listener_distance_m" in s or "mask.sum()<cfg['min_source_points']" in s or 'key=(int(np.digitize' in s]
 result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'split_counts':{s:len(r) for s,r in manifests.items()},'scene_counts':{s:len(v) for s,v in scenes.items()},'scene_disjoint':True,'dry_audio_overlap':{a+'_'+b:len(audio_sets[a]&audio_sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]},'menus':len(groups),'scenes_with_different_menus_by_source':sum(len(v)>1 for v in per_scene.values()),'generator_sha256':sha(gen),'source_conditioning_evidence':evidence,'conclusion':'Historical receiver sampling is source-conditioned through distance strata and target-visible-point filtering. These data support conditional-menu controlled experiments, not source-independent deployment claims.','historical_test_previously_seen':True}
 write_json(ROOT/'reports/data_audit.json',result);write_json(ROOT/'manifests/sample_identities.json',identities)
 for s,rows in manifests.items():write_json(ROOT/'manifests'/f'{s}.json',rows)
 write_json(ROOT/'manifests/menu_signatures.json',signatures)
 protocol={'version':1,'frozen_utc':result['created_utc'],'main_method':'ActiveView','extension':'LiDARFusion','architecture':'Selected release weights frozen; adaptation confined to declared control heads/rankers','budget':{'initial_audio':1,'additional_audio_max':1,'main_point_clouds':1,'extension_point_clouds':2,'max_translation_m':3.0,'objective':'fixed observation count; no travel optimality claim','travel_cost':'record Euclidean receiver displacement and supplied nonnegative costs; no shortest path claim','no_valid_action':'return single-observation result with zero query cost; include in all-sample metrics'},'deployment_inputs':['initial measured XYZ','initial four-channel audio','known microphone array','known relative receiver poses','source-independent validity in new generated benchmark'],'offline_only':['true source positions','instance labels','RIRs','dry recordings','candidate IoU'],'historical_scope':result['conclusion'],'splits':{'train':'head/ranker fitting only','val':'epoch selection only, fixed hyperparameters','test':'previously exposed supplemental evaluation; never model selection','fresh_confirmation':'requires separately generated common-menu evidence; no pristine unseen-scene claim'},'weights':'equal starting-observation weight; paired scene-cluster bootstrap confidence intervals; report scene macro mean','seeds':[1337,2027,3407],'planning':'between-group predictive disagreement heuristic, not exact expected information gain','grouping':{'method':'deterministic complete-link mask IoU, exact-duplicate normalization','iou_threshold':.9,'meaning':'geometric hypothesis families; not proven physical instances','sensitivity':[.8,.9,.95,1.0]},'prediction_variants':['repeat_current','analytic_direct','no_assimilation','no_transition','world'],'policies':['fixed','random','farthest','legacy_active','grouped_active','analytic_active'],'updates':['single','released_predictive','trained_predictive','standard_pose_covariance_semantic_fusion'],'standard_control':'same candidate support, two audio semantic representations, full complex covariances, pose, spatial direct residuals and same TRAIN supervision','fusion_controls':['reference_only_ranker','reference_expansion_ranker','union_ranker','zero_correction'],'fusion_metrics':['reference','second','two_scan','second_only_voxel'],'interpretation':{'new_surface':'second-only measured voxels; no verified occlusion completion','pose':'provided rigid receiver poses; no estimated registration','uncertainty':'log_variance error association and calibration diagnostics; no calibrated uncertainty claim','ranker':'bounded correction can degrade; no safety guarantee'},'epochs':40,'fusion_epochs':30,'fusion_patience':6,'split_hashes':{s:sha(ROOT/'manifests'/f'{s}.json') for s in manifests},'protected_originals':sha(ROOT/'manifests/protected_source.json')}
 path=ROOT/'configs/protocol.json'
 if path.exists():raise FileExistsError('Protocol already frozen')
 write_json(path,protocol);write_json(ROOT/'manifests/protocol_freeze.json',{'sha256':sha(path),'created_utc':result['created_utc']})
 print(json.dumps(result,indent=2))
main()
