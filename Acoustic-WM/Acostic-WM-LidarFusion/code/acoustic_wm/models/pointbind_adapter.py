"""Frozen official Point-Bind point/audio branches, without unused modalities.

Weights and source stay under assets/pointbind. This is an external baseline,
not a contribution of the world model. FPS/KNN use ordinary PyTorch operations
because the upstream CUDA extensions predate the cluster's GPUs.
"""
import sys
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from ..utils import ROOT

path_manager = SimpleNamespace(open=open)


def fps(x, number):
    b, n, _ = x.shape
    distance = torch.full((b, n), float('inf'), device=x.device)
    farthest = torch.zeros(b, dtype=torch.long, device=x.device)
    indices = torch.empty((b, number), dtype=torch.long, device=x.device)
    batch = torch.arange(b, device=x.device)
    for j in range(number):
        indices[:, j] = farthest
        distance = torch.minimum(distance, (x - x[batch, farthest, None]).square().sum(-1))
        farthest = distance.argmax(-1)
    return x.gather(1, indices[..., None].expand(-1, -1, 3))


class KNN:
    def __init__(self, k, transpose_mode=True):
        if not transpose_mode:
            raise ValueError('Only point-major KNN is used by Point-Bind')
        self.k = k

    def __call__(self, reference, query):
        return torch.cdist(query.float(), reference.float()).topk(self.k, largest=False)


class PointBindEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        source = ROOT / 'assets/pointbind/source'
        sys.path.insert(0, str(source))
        from models.i2p_mae.I2P_MAE import I2P_MAE_BIND
        from imagebind.helpers import EinOpsRearrange, Normalize, SelectElement
        from imagebind.multimodal_preprocessors import AudioPreprocessor, PatchEmbedGeneric, SpatioTemporalPosEmbeddingHelper
        from imagebind.transformer import MultiheadAttention, SimpleTransformer
        from functools import partial
        self.point_encoder = I2P_MAE_BIND()
        self.pc_projection = nn.Parameter(torch.empty(384, 512))
        self.point_head = nn.Sequential(nn.LayerNorm(512, eps=1e-6), nn.Dropout(.5), nn.Linear(512, 1024, bias=False))
        stem = PatchEmbedGeneric([nn.Conv2d(1, 768, 16, stride=10, bias=False)], nn.LayerNorm(768))
        self.audio_preprocessor = AudioPreprocessor(img_size=[1, 128, 204], num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True), audio_stem=stem)
        self.audio_trunk = SimpleTransformer(embed_dim=768, num_blocks=12, ffn_dropout_rate=0., drop_path_rate=.1,
            attn_target=partial(MultiheadAttention, embed_dim=768, num_heads=12, bias=True, add_bias_kv=True),
            pre_transformer_layer=nn.Sequential(nn.Identity(), EinOpsRearrange('b l d -> l b d')),
            post_transformer_layer=EinOpsRearrange('l b d -> b l d'))
        self.audio_head = nn.Sequential(nn.LayerNorm(768, eps=1e-6), SelectElement(0), nn.Linear(768, 1024, bias=False))
        checkpoint=ROOT/'assets/pointbind/point-audio.pt'
        if not checkpoint.exists():checkpoint=ROOT/'assets/pointbind/pointbind_i2pmae.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        mapping = [('point_encoder.', self.point_encoder), ('bind.modality_head_point.', self.point_head),
                   ('bind.modality_preprocessors.audio.', self.audio_preprocessor),
                   ('bind.modality_trunks.audio.', self.audio_trunk), ('bind.modality_heads.audio.', self.audio_head)]
        for prefix, module in mapping:
            module.load_state_dict({k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        self.pc_projection.data.copy_(state['pc_projection'])
        self.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def points(self, points):
        return torch.nn.functional.normalize(self.point_head(self.point_encoder(points) @ self.pc_projection), dim=-1)

    @torch.inference_mode()
    def audio(self, mel):
        tokens = self.audio_preprocessor(audio=mel)
        return torch.nn.functional.normalize(self.audio_head(self.audio_trunk(**tokens['trunk'])), dim=-1)


def audio_mel(wave, rate):
    """Official ImageBind Kaldi front end; observed channels are averaged."""
    import torchaudio
    wave = torch.as_tensor(np.asarray(wave).mean(-1), dtype=torch.float32)[None]
    if rate != 16000:
        wave = torchaudio.functional.resample(wave, rate, 16000)
    duration = 32000
    starts = np.linspace(0, max(0, wave.shape[1]-duration), 3).astype(int)
    out = []
    for start in starts:
        clip = wave[:, start:start+duration].clone()
        clip -= clip.mean()
        mel = torchaudio.compliance.kaldi.fbank(clip, htk_compat=True, sample_frequency=16000,
            use_energy=False, window_type='hanning', num_mel_bins=128, dither=0., frame_length=25, frame_shift=10).T
        mel = torch.nn.functional.pad(mel[:, :204], (0, max(0, 204-mel.shape[1])))
        out.append((mel[None]+4.268)/9.138)
    return torch.stack(out)


def proposal_points(xyz, mask, points=1024):
    """Deterministic measured points, centered/scaled; LiDAR Z-up to ModelNet Y-up."""
    xyz = np.asarray(xyz[mask], dtype=np.float32)
    if len(xyz) == 0:
        return np.zeros((points, 3), np.float32)
    xyz = xyz - xyz.mean(0)
    xyz = xyz[:, [0, 2, 1]] * np.array([1, 1, -1], np.float32)
    xyz /= max(float(np.linalg.norm(xyz, axis=-1).max()), 1e-6)
    indices = np.linspace(0, len(xyz)-1, points).astype(int)
    return xyz[indices]


@lru_cache(maxsize=2)
def shared_pointbind_encoder(device='cuda'):
    return PointBindEncoder(device)
