"""Read-only pretrained assets, with runtime caches confined to this project."""
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WM=Path(os.environ.get('LIDAR_WM_ROOT',ROOT.parent/'LiDAR-WM')).expanduser().resolve()


def configure_runtime():
    sys.dont_write_bytecode=True
    for name,folder in (('XDG_CACHE_HOME','runtime'),('HF_HOME','hf'),
                        ('TORCH_EXTENSIONS_DIR','torch_extensions'),('TORCH_HOME','torch'),
                        ('NUMBA_CACHE_DIR','numba')):
        os.environ[name]=str(ROOT/'cache'/folder)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['USE_TF']='0'
    os.environ['USE_FLAX']='0'


def clap_snapshot():
    repository=WM/'pretrained/hub/models--laion--clap-htsat-unfused'
    revision=(repository/'refs/main').read_text().strip()
    if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Invalid local CLAP revision')
    snapshot=repository/'snapshots'/revision
    for name in ('config.json','preprocessor_config.json','tokenizer_config.json','pytorch_model.bin'):
        if not (snapshot/name).is_file():
            raise FileNotFoundError(f'Missing read-only CLAP asset: {snapshot/name}')
    return snapshot


def load_clap(device):
    """Same encoder/prompts/preprocessing as the training cache, offline only."""
    configure_runtime()
    import torch
    from transformers import ClapModel,ClapProcessor
    from lidarwm.foundation import FrozenClap,DESCRIPTIONS
    from .features import CATEGORIES
    snapshot=clap_snapshot()
    provenance=json.loads((WM/'cache/clap/provenance.json').read_text())
    texts=[f'The sound of {DESCRIPTIONS[c]}.' for c in CATEGORIES]
    if provenance['classes']!=list(CATEGORIES) or provenance['descriptions']!=texts:
        raise ValueError('Local fixed text bank differs from the training cache')
    if torch.device(device).type=='cpu':
        torch.backends.mkldnn.enabled=False
    # Bypass its network-aware constructor, retaining the exact observed-wave
    # normalization, mono averaging and 48 kHz resampling in __call__.
    encoder=FrozenClap.__new__(FrozenClap)
    encoder.device=device
    encoder.processor=ClapProcessor.from_pretrained(snapshot,local_files_only=True)
    encoder.model=ClapModel.from_pretrained(snapshot,local_files_only=True,use_safetensors=False).to(device).eval()
    with torch.inference_mode():
        tokens=encoder.processor(text=texts,padding=True,return_tensors='pt').to(device)
        encoder.text_features=torch.nn.functional.normalize(encoder.model.get_text_features(**tokens),dim=-1)
    return encoder
