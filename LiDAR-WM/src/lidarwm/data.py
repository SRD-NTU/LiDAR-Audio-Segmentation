from __future__ import annotations

import csv
import hashlib
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import Dataset


CATEGORIES = (
    "refrigerator", "fan", "tv", "door", "toilet", "washing machine",
    "vacuum cleaner", "clock", "microwave", "dishwasher", "toaster",
    "coffee machine", "speaker", "printer", "keyboard", "kettle",
)
CATEGORY_TO_ID = {name: index for index, name in enumerate(CATEGORIES)}


@dataclass(frozen=True)
class SampleRecord:
    relative_path: str
    split: str
    scene: str
    source: str
    category: str


def read_manifest(root: str | Path, split: str) -> list[SampleRecord]:
    root = Path(root)
    manifest = root / "metadata" / "samples.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = [
        SampleRecord(
            relative_path=row["sample"], split=row["split"], scene=row["scene"],
            source=row["source"], category=row["category"],
        )
        for row in rows if row["split"] == split
    ]
    if not records:
        raise ValueError(f"No samples for split {split!r} in {manifest}")
    return sorted(records, key=lambda item: item.relative_path)


def _read_float_wav(path: Path) -> tuple[int, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate, audio = wavfile.read(path)
    if audio.ndim != 2 or audio.shape[1] != 4:
        raise ValueError(f"Expected [time,4] audio at {path}, got {audio.shape}")
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32, copy=False)
    return rate, audio


class AudioLidarDataset(Dataset):
    """Loads only deployable inputs (xyz and audio); annotations are targets only."""

    height = 64
    width = 1800

    def __init__(
        self,
        root: str | Path,
        split: str,
        audio_stride: int = 3,
        augment: bool = False,
        max_samples: int | None = None,
        feature_root: str | Path | None = None,
        equivariant_augmentation: bool = False,
        antialias: bool = False,
        instance_centers: bool = False,
        instance_masks: bool = False,
        geometry_feature_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.audio_stride = int(audio_stride)
        self.augment = bool(augment)
        self.feature_root = Path(feature_root) if feature_root else None
        self.equivariant_augmentation = equivariant_augmentation
        self.antialias = antialias
        self.instance_centers = instance_centers
        self.instance_masks = instance_masks
        self.geometry_feature_root=Path(geometry_feature_root) if geometry_feature_root else None
        self.records = read_manifest(self.root, split)
        import json
        taxonomy = json.loads((self.root / "metadata" / "categories.json").read_text())
        max_taxonomy_id = max(int(value) for value in taxonomy.values())
        self.semantic_lookup = np.zeros(max_taxonomy_id + 1, dtype=np.int64)
        for class_index, name in enumerate(CATEGORIES, start=1):
            self.semantic_lookup[int(taxonomy[name])] = class_index
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        sample_dir = self.root / record.relative_path
        with np.load(sample_dir / "lidar.npz", allow_pickle=False) as archive:
            xyz = archive["xyz"].astype(np.float32, copy=False)
            ray_id = archive["ray_id"].astype(np.int64, copy=False)
            target = archive["sound_mask"].astype(np.float32, copy=False)
            semantic_id = archive["semantic_id"].astype(np.int64, copy=False)
            if self.instance_centers or self.instance_masks:
                instance_id=archive['instance_id'].astype(np.int64,copy=False)

        if xyz.shape != (len(ray_id), 3) or len(target) != len(ray_id):
            raise ValueError(f"Inconsistent LiDAR arrays in {sample_dir}")
        if np.any(ray_id < 0) or np.any(ray_id >= self.height * self.width):
            raise ValueError(f"Out-of-range ray_id in {sample_dir}")

        grid_xyz = np.zeros((self.height * self.width, 3), dtype=np.float32)
        grid_target = np.zeros(self.height * self.width, dtype=np.float32)
        grid_valid = np.zeros(self.height * self.width, dtype=np.float32)
        grid_semantic = np.full(self.height * self.width, 255, dtype=np.int64)
        grid_xyz[ray_id] = xyz
        grid_target[ray_id] = target
        grid_valid[ray_id] = 1.0
        # 0 is non-sound-capable background, 1..16 are the benchmark classes.
        point_semantic = np.zeros_like(semantic_id)
        known = (semantic_id >= 0) & (semantic_id < len(self.semantic_lookup))
        point_semantic[known] = self.semantic_lookup[semantic_id[known]]
        grid_semantic[ray_id] = point_semantic
        center_grid=None
        instance_grid=None
        if self.instance_masks:
            # Annotation-only auxiliary targets. Never exposed to model.forward.
            foreground=(point_semantic>0)&(instance_id>=0)
            compact=np.zeros(self.height*self.width,dtype=np.int64)
            if foreground.any():
                _,inverse=np.unique(instance_id[foreground],return_inverse=True)
                compact[ray_id[foreground]]=inverse+1
            instance_grid=compact.reshape(self.width,self.height).T
        if self.instance_centers:
            foreground=(point_semantic>0)&(instance_id>=0)
            center_flat=np.zeros((self.height*self.width,3),dtype='float32')
            weight_flat=np.zeros(self.height*self.width,dtype='float32')
            if foreground.any():
                _,inverse=np.unique(instance_id[foreground],return_inverse=True)
                count=np.bincount(inverse)
                centers=np.stack([np.bincount(inverse,weights=xyz[foreground,d])/count for d in range(3)],axis=-1)
                center_flat[ray_id[foreground]]=centers[inverse]/20
                weight_flat[ray_id[foreground]]=1/count[inverse]
            center_grid=center_flat.reshape(self.width,self.height,3).transpose(2,1,0)
            center_weight=weight_flat.reshape(self.width,self.height).T[None]

        grid_xyz = grid_xyz.reshape(self.width, self.height, 3).transpose(2, 1, 0)
        ranges = np.linalg.norm(grid_xyz, axis=0, keepdims=True)
        directions = grid_xyz / np.maximum(ranges, 1e-5)
        # Metres are kept in a bounded numeric range while preserving absolute range.
        lidar = np.concatenate((grid_xyz / 20.0, ranges / 20.0, directions), axis=0)
        mask = grid_target.reshape(self.width, self.height).T[None]
        valid = grid_valid.reshape(self.width, self.height).T[None]
        semantic = grid_semantic.reshape(self.width, self.height).T
        geometry_features=None
        if self.geometry_feature_root:
            with np.load(self.geometry_feature_root/(record.relative_path+'.npz'),allow_pickle=False) as feature:
                if str(feature['xyz_sha256'])!=hashlib.sha256(xyz.tobytes()).hexdigest():
                    raise ValueError(f'Geometry cache does not match observed XYZ: {record.relative_path}')
                values=feature['features'];inverse=feature['inverse'];pixels=feature['pixels']
                if values.ndim!=2 or values.shape[1]!=256 or len(inverse)!=len(pixels):
                    raise ValueError(f'Invalid XYZ-only geometry cache: {record.relative_path}')
                # Preserve sparse voxels until GPU decoding. A dense cached
                # [256,64,1800] array would waste ~59 MB per sample/augmentation.
                geometry_features=dict(features=values.copy(),inverse=inverse.copy(),pixels=pixels.copy())

        sample_rate, audio = _read_float_wav(sample_dir / "audio.wav")
        if sample_rate != 44100:
            raise ValueError(f"Expected 44100 Hz at {sample_dir}, got {sample_rate}")
        audio = (resample_poly(audio, 1, self.audio_stride, axis=0) if self.antialias else audio[:: self.audio_stride]).T.copy()
        if self.augment:
            lidar, audio = self._augment(lidar, audio, valid)
            if self.equivariant_augmentation:
                turns = random.randrange(4)
                for _ in range(turns):
                    lidar = np.roll(lidar, self.width // 4, axis=-1)
                    # A +90-degree sensor-frame rotation permutes the square array.
                    for a,b in ((0,1),(4,5)):
                        old = lidar[a].copy()
                        lidar[a] = -lidar[b]
                        lidar[b] = old
                    audio = audio[[1,3,0,2]]
                    mask = np.roll(mask, self.width // 4, axis=-1)
                    valid = np.roll(valid, self.width // 4, axis=-1)
                    semantic = np.roll(semantic, self.width // 4, axis=-1)
                    if geometry_features is not None:
                        pixels=geometry_features['pixels']
                        geometry_features['pixels']=(pixels//self.width)*self.width+(pixels%self.width+self.width//4)%self.width
                    if instance_grid is not None:
                        instance_grid=np.roll(instance_grid,self.width//4,axis=-1)
                    if center_grid is not None:
                        center_grid=np.roll(center_grid,self.width//4,axis=-1)
                        cx=center_grid[0].copy();center_grid[0]=-center_grid[1];center_grid[1]=cx
                        center_weight=np.roll(center_weight,self.width//4,axis=-1)

        output = {
            "lidar": torch.from_numpy(lidar.copy()),
            "valid": torch.from_numpy(valid.copy()),
            "audio": torch.from_numpy(audio.copy()),
            "mask": torch.from_numpy(mask.copy()),
            "category": torch.tensor(CATEGORY_TO_ID[record.category], dtype=torch.long),
            "semantic": torch.from_numpy(semantic.copy()),
            "ray_id": torch.from_numpy(ray_id.copy()),
            "path": record.relative_path,
        }
        if self.feature_root:
            with np.load(self.feature_root / (record.relative_path + '.npz'), allow_pickle=False) as feature:
                output['audio_embedding'] = torch.from_numpy(feature['embedding'].copy())
                output['text_logits'] = torch.from_numpy(feature['text_logits'].copy())
        if center_grid is not None:
            output['instance_center']=torch.from_numpy(center_grid.copy())
            output['instance_weight']=torch.from_numpy(center_weight.copy())
        if instance_grid is not None:
            output['instance_map']=torch.from_numpy(instance_grid.copy())
        if geometry_features is not None:
            output['geometry_features']={k:torch.from_numpy(v.copy()) for k,v in geometry_features.items()}
        return output

    @staticmethod
    def _augment(
        lidar: np.ndarray, audio: np.ndarray, valid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Shared gain/noise/jitter only here. The caller separately handles square-
        # array 90-degree rotations with the matching channel permutation.
        gain = 10.0 ** random.uniform(-0.15, 0.15)
        audio = audio * gain
        if random.random() < 0.5:
            snr_db = random.uniform(25.0, 45.0)
            noise_scale = float(np.sqrt(np.mean(audio * audio))) * 10.0 ** (-snr_db / 20.0)
            audio = audio + np.random.randn(*audio.shape).astype(np.float32) * noise_scale
        if random.random() < 0.5:
            jitter = np.random.normal(0.0, 0.002 / 20.0, size=lidar[:3].shape).astype(np.float32)
            lidar[:3] += jitter * valid
        return lidar.astype(np.float32), audio.astype(np.float32)


def collate_samples(batch: list[dict]) -> dict:
    """Keep variable-length ray IDs and paths as lists; stack model tensors."""
    tensor_keys = ("lidar", "valid", "audio", "mask", "category", "semantic")
    output = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    for key in ('audio_embedding','text_logits','instance_center','instance_weight','instance_map'):
        if key in batch[0]: output[key] = torch.stack([item[key] for item in batch])
    if 'geometry_features' in batch[0]:output['geometry_features']=[item['geometry_features'] for item in batch]
    output["ray_id"] = [item["ray_id"] for item in batch]
    output["path"] = [item["path"] for item in batch]
    return output
