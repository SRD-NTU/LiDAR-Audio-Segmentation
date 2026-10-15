"""Quota-aware continuation of this campaign, without reserving an idle GPU.

Polls Slurm every 30 seconds. Submits each follow-up exactly once, never retries
failed training, and records failure logs. Does not cancel unrelated user jobs.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'ORBIT/reports/consistent-v2-full-20261014'
STATE=STUDY/'scheduler-state.json'
INITIAL={'cache':204502,'masks':204509,'frozen-standard':204510,
         'frozen-paired':204511,'baseline':204512}
FAILURES={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}


def output(command):
    return subprocess.check_output(command,text=True,cwd=ROOT).strip()


def save(state):
    temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2)+'\n')
    temporary.replace(STATE)


def active_dependencies(dependencies):
    """Completed jobs can age out of slurmctld before their descendants submit.

    Confirm success with accounting, but only send unfinished job IDs to sbatch.
    Never omit a failed or unknown prerequisite merely to bypass a dependency.
    """
    if not dependencies:return []
    record=output(['sacct','-n','-P','-X','-j',','.join(map(str,dependencies)),
                   '--format=JobIDRaw,State,ExitCode'])
    observed={}
    for line in record.splitlines():
        fields=line.split('|')
        if fields[0].isdigit():observed[int(fields[0])]=(fields[1].split()[0],fields[2])
    active=[]
    for job in dependencies:
        state,exit_code=observed.get(job,('UNKNOWN',''))
        if state=='COMPLETED' and exit_code=='0:0':continue
        if state not in ('RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED'):
            raise RuntimeError(f'Prerequisite {job} is not successful or active: {state}, {exit_code}')
        active.append(job)
    return active


def verify_completed_artifact(stage):
    if not (STUDY/f'{stage}-COMPLETE.json').is_file():
        raise RuntimeError(f'Missing completed-stage artifact: {stage}')
    runs={'masks':'masks-consistent-v2-full-1337',
          'new-standard':'audio-gate-consistent-v2-full-1337',
          'new-paired':'audio-gate-pairs-consistent-v2-full-1337',
          'frozen-standard':'audio-gate-consistent-v2-frozen-masks-1337',
          'frozen-paired':'audio-gate-pairs-consistent-v2-frozen-masks-1337'}
    if stage in runs:
        import hashlib
        run=ROOT/'ORBIT/runs'/runs[stage]
        completed=json.loads((run/'COMPLETE.json').read_text())
        actual=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
        if completed['checkpoint_sha256']!=actual:
            raise RuntimeError(f'Completed checkpoint identity changed: {stage}')


def submit(stage, dependencies):
    # Live association, QOS and partition records are authoritative every time.
    output(['sacctmgr','-n','-P','show','assoc','where',f'user={os.environ["USER"]}',
            'format=Cluster,Account,User,QOS,DefaultQOS'])
    qos=output(['sacctmgr','-n','-P','show','qos','where','name=wanglin_2026_08',
                'format=Name,MaxSubmitJobsPU,GrpTRES'])
    if not any(line.startswith('wanglin_2026_08|5|') for line in qos.splitlines()):
        raise RuntimeError('Live QOS limits changed; review before submitting')
    partition=output(['scontrol','show','partition','cluster02'])
    if 'State=UP' not in partition:raise RuntimeError('Partition is not UP')
    dependencies=active_dependencies(dependencies)
    command=['sbatch','--parsable',f'--job-name=orbit-v2-{stage}']
    if dependencies:command+=['--dependency=afterok:'+':'.join(map(str,dependencies))]
    command+=['ORBIT/scripts/consistency_gpu.sbatch',stage]
    response=output(command)
    job=int(response.splitlines()[-1].split(';')[0])
    print(json.dumps({'submitted_stage':stage,'job_id':job,'dependencies':dependencies}),flush=True)
    return job


def summary():
    result={'status':'complete','scope':'fixed-Utonia-selector-APEX-spatial consistency study',
            'each_fix_causally_isolated':False,'research_extensions_complete':False}
    for path in sorted(STUDY.glob('*.json')):
        if path.name in ('SUMMARY.json','scheduler-state.json'):continue
        data=json.loads(path.read_text())
        if 'metrics' in data:
            result[path.stem]={name:metric['mean_iou'] for name,metric in data['metrics'].items()}
        elif 'raw-cached' in path.name:result[path.stem]=data
    (STUDY/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    lock=(STUDY/'scheduler.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(STATE.read_text()) if STATE.exists() else {'jobs':INITIAL.copy(),'status':'running'}
    state['status']='running'
    state['monitor_pid']=os.getpid()
    start=time.time()
    while time.time()-start<7*86400:
        jobs=state['jobs']
        record=output(['sacct','-n','-P','-X','-j',','.join(map(str,jobs.values())),
                       '--format=JobIDRaw,State,ExitCode'])
        observed={}
        for line in record.splitlines():
            fields=line.split('|')
            if fields[0].isdigit():observed[int(fields[0])]={'state':fields[1].split()[0],'exit_code':fields[2]}
        state['observed']={name:observed.get(job,{'state':'UNKNOWN'}) for name,job in jobs.items()}
        state['updated_at_unix']=time.time()
        failed=[name for name,info in state['observed'].items() if info['state'] in FAILURES]
        if failed:
            state['status']='failed';state['failed_stages']=failed
            state['failure_log_tails']={}
            for name in failed:
                paths=list((ROOT/'ORBIT/logs').glob(f'*{jobs[name]}.*'))
                state['failure_log_tails'][name]={str(p):'\n'.join(p.read_text(errors='replace').splitlines()[-50:]) for p in paths}
            save(state);print(json.dumps(state),flush=True);return
        queued=output(['squeue','-h','-u',os.environ['USER'],'-o','%i']).splitlines()
        available=max(0,5-len(queued))
        # After cache preparation leaves the queue, admit downstream jobs as
        # slots open. No GPU is reserved merely waiting for a prerequisite.
        for stage in ('new-standard','new-paired','evaluate'):
            if stage in jobs or not available:continue
            if stage=='evaluate':
                required=('new-standard','new-paired','frozen-standard','frozen-paired','baseline')
                if not all(name in jobs for name in required):continue
                dependencies=[jobs[name] for name in required]
            else:
                required=('masks',)
                dependencies=[jobs['masks']]
            for name in required:
                if state['observed'].get(name,{}).get('state')=='COMPLETED':
                    verify_completed_artifact(name)
            try:
                jobs[stage]=submit(stage,dependencies)
            except Exception as error:
                state['status']='submission-failed'
                state['submission_failure']={'stage':stage,'error':str(error),'time':time.time()}
                save(state)
                raise
            save(state)
            available-=1
        if 'evaluate' in jobs and all(info['state']=='COMPLETED' for info in state['observed'].values()) and len(state['observed'])==len(jobs):
            state['status']='complete';save(state);summary();print('CAMPAIGN COMPLETE',flush=True);return
        save(state)
        time.sleep(30)
    state['status']='monitor-deadline';save(state)


if __name__=='__main__':main()
