"""Content identity for the untouched release source outside this codebase."""
from pathlib import Path
import hashlib
import json

PACKAGES=('Acoustic-WM-ActiveView','Acoustic-WM-SingleView','Acostic-WM-LidarFusion')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def identity(root):
    root=Path(root)
    files=[]
    for name in PACKAGES:
        package=root/name
        files.extend(p for p in package.rglob('*.py') if 'outputs' not in p.parts)
        files.extend([package/'config.json',package/'requirements.txt'])
        files.extend((package/'checkpoints').glob('*.pt'))
        pointbind=package/'assets/pointbind/point-audio.pt'
        if pointbind.exists():files.append(pointbind)
    manifest=root/'MANIFEST.json'
    if manifest.exists():files.append(manifest)
    entries={str(p.relative_to(root)):digest(p) for p in sorted(set(files)) if p.is_file()}
    source_hash=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(root=str(root.resolve()),sha256=source_hash,files=len(entries))
