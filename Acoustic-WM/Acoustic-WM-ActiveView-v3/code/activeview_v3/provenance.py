"""Immutable experiment inputs for revised evaluations."""
from .runtime import ROOT,BASE,DATA,ARCHIVE,UTILITY,sha
import hashlib,json,subprocess
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def _files(paths):
    return {str(p):sha(p) for p in sorted(set(paths)) if p.is_file()}

def source_files():
    local=(list((ROOT/'code').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+
           list((ROOT/'scripts').glob('*.sh'))+list((ROOT/'scripts').glob('*.sbatch'))+[ROOT/'infer.py'])
    backbone=list((BASE/'code/acoustic_wm').rglob('*.py'))
    renderer=list((ARCHIVE/'snapshots/generator').rglob('*.py'))
    utility=list((UTILITY/'audiolidar').rglob('*.py'))
    return _files(local+backbone+renderer+utility)

def config_files():
    local=[p for p in (ROOT/'config').glob('*') if not p.name.startswith('frozen_')]
    external=[BASE/'config.json',ROOT/'data/feature_contract.json',
              DATA/'metadata/microphone_array.json',DATA/'metadata/categories.json',
              DATA/'metadata/audio_manifest.json']
    external += [DATA/'metadata'/(split+'.json') for split in ['train','val','test']]
    external += [
              DATA/'assets/mp3d_material_config.json',ARCHIVE/'snapshots/generator/base_config.yaml',
              UTILITY/'data/hm3d-v02/hm3d_annotated_basis.scene_dataset_config.json']
    return _files(local+external)

def dependency_versions(interpreter):
    code=("import json,sys,importlib.metadata as m;"
          "print(json.dumps({'python':sys.version,'packages':sorted((d.metadata['Name'].lower(),d.version) "
          "for d in m.distributions() if d.metadata['Name'])},sort_keys=True))")
    result=subprocess.run([str(interpreter),'-B','-c',code],capture_output=True,text=True,
                          timeout=120,check=True)
    return json.loads(result.stdout)

def make_manifest(learning_python,renderer_python):
    sources=source_files();configs=config_files()
    dependencies={'learning':dependency_versions(learning_python),
                  'renderer':dependency_versions(renderer_python)}
    return dict(source_sha256=sources,config_sha256=configs,dependency_versions=dependencies,
                source_manifest_sha256=digest(sources),config_manifest_sha256=digest(configs),
                dependency_manifest_sha256=digest(dependencies))

def verify_manifest(protocol,check_dependencies=False,learning_python=None,renderer_python=None):
    content={k:v for k,v in protocol.items() if k!='protocol_sha256'}
    expected=hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
    if expected!=protocol['protocol_sha256']:
        raise RuntimeError('Frozen protocol digest mismatch')
    manifest=protocol['provenance']
    for name,expected in {**manifest['source_sha256'],**manifest['config_sha256']}.items():
        if sha(Path(name))!=expected:raise RuntimeError('Frozen source/configuration changed: '+name)
    if check_dependencies:
        actual={'learning':dependency_versions(learning_python),
                'renderer':dependency_versions(renderer_python)}
        if digest(actual)!=manifest['dependency_manifest_sha256']:
            raise RuntimeError('Frozen dependency versions changed')
    return {name:manifest[name] for name in
            ['source_manifest_sha256','config_manifest_sha256','dependency_manifest_sha256']}
