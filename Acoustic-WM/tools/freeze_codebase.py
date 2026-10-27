"""Write a content-addressed identity for the revised Acoustic-WM codebase."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
PACKAGES=['Acoustic-WM-ActiveView','Acoustic-WM-SingleView','Acostic-WM-LidarFusion']


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def include():
    files=[ROOT/'README.md',ROOT/'results.md',ROOT/'ENVIRONMENT.json',ROOT/'.venv/pyvenv.cfg']
    for package in PACKAGES:
        folder=ROOT/package
        files.extend(p for p in folder.rglob('*.py') if 'outputs' not in p.parts and 'assets' not in p.parts)
        files.extend([folder/'config.json',folder/'requirements.txt'])
        files.extend((folder/'checkpoints').glob('*.pt'))
        pointbind=folder/'assets/pointbind/point-audio.pt'
        if pointbind.exists():files.append(pointbind)
        files.extend((folder/'assets/sensor/metadata').glob('*.json'))
        provenance=folder/'assets/pointbind/PROVENANCE.json'
        if provenance.exists():files.append(provenance)
    frontend=ROOT/'frontend_runtime'
    for folder in ['scene_frontend','runtime','third_party/Utonia','third_party/sonata']:
        files.extend((frontend/folder).rglob('*.py'))
    for folder in ['cache','runs','pretrained/utonia']:
        files.extend(p for p in (frontend/folder).rglob('*') if p.is_file())
    clap=frontend/'pretrained/clap-hub/models--laion--clap-htsat-unfused'
    if clap.exists():
        files.extend(p for p in (clap/'snapshots').rglob('*') if p.is_file())
        files.extend(p for p in (clap/'refs').rglob('*') if p.is_file())
    files.append(frontend/'README.md')
    for folder in ['research/src','research/scripts','research/tests','research/configs',
                   'research/paper','research/docs','tests','tools']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
                     and p.suffix in ('.py','.json','.md','.txt','.tex','.sbatch','.yaml')
                     and p.name!='READY' and '__pycache__' not in p.parts)
    files.extend([ROOT/'research/README.md',ROOT/'research/EVIDENCE_INDEX.json'])
    files.extend((ROOT/'research/manifests').glob('*.json'))
    files.extend((ROOT/'research/reports').glob('*.json'))
    return sorted(set(p for p in files if p.is_file()))


def main():
    files=include()
    entries={str(p.relative_to(ROOT)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in files}
    canonical=json.dumps(entries,sort_keys=True,separators=(',',':')).encode()
    source_id=hashlib.sha256(canonical).hexdigest()
    external={}
    for name in ['train','val','test']:
        path=PROJECT/'dataset/v2-tetra4-4channel-full/metadata'/f'{name}.json'
        if path.exists():external[str(path)]=digest(path)
    comments=PROJECT/'message.txt'
    if comments.exists():external[str(comments)]=digest(comments)
    original=PROJECT/'.importants/20261026/Acoustic-WM/MANIFEST.json'
    if original.exists():external[str(original)]=digest(original)
    manifest=dict(version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  codebase_sha256=source_id,
                  provenance='Copied from the untouched .importants/20261026/Acoustic-WM archive; original release results are not measurements of the revised defaults.',
                  git_commit=None,git_note='No Git metadata at the workspace root; SHA-256 over the enumerated files is the source identity.',
                  files=entries,external_inputs_sha256=external)
    output=ROOT/'MANIFEST.json'
    output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),codebase_sha256=source_id,files=len(entries)),indent=2))


if __name__=='__main__':main()
