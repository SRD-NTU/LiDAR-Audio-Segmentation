"""Index completed archived results without relabeling them as new-code results."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parents[1]
PROJECT=HERE.parents[1]
ARCHIVED=PROJECT/'.importants/Experiments_20261024'
PARENT=PROJECT/'dataset/v2-tetra4-4channel-full'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    campaign=ARCHIVED/'manifests/campaign_source.json'
    snapshot=ARCHIVED/'manifests/snapshot.json'
    protocol=HERE/'configs/protocol.json'
    audio_manifest=PARENT/'metadata/audio_manifest.json'
    recordings=[r for r in json.loads(audio_manifest.read_text()) if r['split']=='test']
    recording_identity=dict(manifest_path=str(audio_manifest),manifest_sha256=sha(audio_manifest),
                            split='test',clips=len(recordings),
                            original_recording_ids=sorted({r['original_recording_id'] for r in recordings}),
                            dry_audio_sha256=sorted(r['sha256'] for r in recordings))
    source=dict(campaign_source_path=str(campaign),campaign_source_sha256=sha(campaign),
                source_snapshot_path=str(snapshot),source_snapshot_sha256=sha(snapshot),
                protocol_path=str(protocol),protocol_sha256=sha(protocol),
                released_acoustic_world_checkpoint_sha256='5a972cac224cf7267160791d7d6631ff2b23747bc30f52b794406a81c77b9506',
                git_commit=None,
                identity_note='The workspace has no Git metadata; pinned per-file SHA-256 snapshots provide the immutable source identity.')
    historical_menus=json.loads((ARCHIVED/'manifests/menu_signatures.json').read_text())
    common_menus={p.parent.name:json.loads(p.read_text())['menu_sha256']
                  for p in (ARCHIVED/'generated/common/scenes').glob('*/COMPLETE.json')}
    records=[]
    for scenario,folder in [('historical_source_conditioned',ARCHIVED/'runs/controls'),
                            ('source_independent_common_menu',ARCHIVED/'scopes/common/runs/controls')]:
        for result in sorted(folder.glob('*/test.json')):
            run=json.loads(result.read_text())
            checkpoint_file=result.parent/'complete.json'
            checkpoint=json.loads(checkpoint_file.read_text()) if checkpoint_file.exists() else {}
            if 'samples' not in run:continue
            seed=run.get('seed')
            methods=run.get('policies',[])
            sample_paths=[r['sample'] for r in run['samples']]
            scenes=sorted({r.split('/')[1] for r in sample_paths})
            menu_description=('one scene-level geometry-only pose set per scene, identical across sources; '
                              'common-menu generator stores menu_sha256 in each scene and sample'
                              if scenario.startswith('source_independent') else
                              'twelve source-conditioned recorded poses per physical source group')
            menu_hashes=(sorted({common_menus[s] for s in scenes}) if scenario.startswith('source_independent')
                         else sorted({historical_menus['/'.join(s.split('/')[:3])] for s in sample_paths}))
            records.append(dict(result_type='controlled_segmentation',scenario=scenario,
                                split='test',episodes=len(sample_paths),scenes=scenes,
                                original_recordings=recording_identity,seed=seed,
                                candidate_bank=dict(description='frozen reference proposals, 48 seeds and four source sites per mask',
                                                    per_sample_cache='.importants/Experiments_20261024/{scope}/cache/reference/{sample}.npz',
                                                    extraction_source_sha256=json.loads(campaign.read_text()).get('src/extract.py'),
                                                    limitation='Per-sample candidate-bank cache bytes are retained in the original experiment but are not individually hashed in this compact index.'),
                                world_checkpoint_sha256=source['released_acoustic_world_checkpoint_sha256'],
                                update_checkpoint_sha256=checkpoint.get('checkpoint_sha256'),
                                update_checkpoint_file=str(checkpoint_file),
                                menu_identity=dict(description=menu_description,menu_sha256=menu_hashes),
                                policies=methods,
                                statistical_method='equal weight per starting observation; paired scene-cluster bootstrap, 2000 draws, RNG seed 20261024',
                                source_snapshot=source,
                                result_path=str(result),result_sha256=sha(result)))
    for scenario,report in [('historical_source_conditioned',ARCHIVED/'reports/prediction.json'),
                            ('source_independent_common_menu',ARCHIVED/'scopes/common/reports/prediction.json')]:
        if report.exists():
            data=json.loads(report.read_text());n=data['summary']['test']['world']['oracle_candidate_covariance_error']['n']
            records.append(dict(result_type='candidate_forecast_diagnostic',scenario=scenario,
                                split='test',episodes=n,original_recordings=recording_identity,
                                seed='released world checkpoint seed 1337; no forecast retraining',
                                candidate_bank='ground truth selects highest-IoU saved reference mask only after predictions are computed',
                                world_checkpoint_sha256=source['released_acoustic_world_checkpoint_sha256'],
                                menu_identity='saved valid action poses, <=3 m receiver translation',
                                statistical_method='future-band-reliability and source-site-posterior weighted complex covariance error; equal valid-action and starting-observation average; scene-cluster bootstrap',
                                source_snapshot=source,result_path=str(report),result_sha256=sha(report)))
    for name in ['multiview','multiview_random','fusion_multiview']:
        folder=PROJECT/'.importants/Experiments_20261025/results'/name
        report=folder/'summary.json'
        if not report.exists():continue
        inputs=folder/'inputs.json'
        data=json.loads(report.read_text())
        records.append(dict(result_type='supplemental_multilocation',scenario='full historical source-conditioned twelve-pose menu',
                            split='test',episodes=data.get('episodes',768),
                            original_recordings=recording_identity,
                            seed='see saved inputs.json and summary.json; random choices have three recorded seeds',
                            candidate_bank='fixed initial reference candidate bank; see saved inputs.json',
                            world_checkpoint_sha256=source['released_acoustic_world_checkpoint_sha256'],
                            menu_identity='original source-conditioned twelve-pose group',
                            statistical_method='scene-cluster bootstrap; see summary.json for each contrast',
                            inputs_path=str(inputs),inputs_sha256=sha(inputs) if inputs.exists() else None,
                            result_path=str(report),result_sha256=sha(report),
                            note='Three and four locations extrapolate the frozen two-location evidence head.'))
    output=HERE/'EVIDENCE_INDEX.json'
    output.write_text(json.dumps(dict(version=1,created_from='completed saved artifacts; no experiment rerun',
                                      records=records),indent=2)+'\n')
    print(f'Indexed {len(records)} result families in {output}')


if __name__=='__main__':main()
