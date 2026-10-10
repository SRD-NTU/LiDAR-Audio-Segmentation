"""Deploy the selected trained graph/tree model on XYZ and four-channel WAV."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from graph_audio.local_assets import configure_runtime
configure_runtime()

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WM = Path(os.environ.get('LIDAR_WM_ROOT', ROOT.parent / 'LiDAR-WM')).expanduser().resolve()


def raw_geometry(xyz, device, encoder=None):
    # Read-only reuse of tested pretrained assets and sparse-encoder dependencies.
    from lidarwm.sonata_geometry import SonataGeometry
    radius = np.linalg.norm(xyz, axis=1)
    angle = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2 * np.pi)
    elevation = np.arcsin(np.clip(xyz[:, 2] / np.maximum(radius, 1e-6), -1, 1))
    col = np.rint(angle / (2 * np.pi) * 1800).astype('int64') % 1800
    row = np.rint((elevation + np.pi / 8) / (np.pi / 4) * 63).astype('int64')
    if np.any((row < 0) | (row >= 64)) or len(np.unique(row * 1800 + col)) != len(xyz):
        raise ValueError('XYZ must match the calibrated 64-beam LiDAR without pixel collisions')
    grid = np.zeros((3, 64, 1800), np.float32)
    valid = np.zeros((1, 64, 1800), np.float32)
    grid[:, row, col] = xyz.T; valid[:, row, col] = 1
    distance = np.linalg.norm(grid, axis=0, keepdims=True)
    lidar = np.concatenate((grid / 20, distance / 20, grid / np.maximum(distance, 1e-5)))
    if encoder is None:
        encoder = SonataGeometry(str(WM / 'pretrained/sonata/sonata.pth'), grid_size=.05).to(device).eval()
    lidar_tensor=torch.from_numpy(lidar)[None].to(device)
    valid_tensor=torch.from_numpy(valid)[None].to(device)
    # Use the encoder's exact observed voxel membership. Independently flooring
    # XYZ in NumPy can change cells at FP32 boundaries after raster scaling.
    sparse=encoder._encode(lidar_tensor,valid_tensor,sparse=True)[0]
    pixel_to_voxel=np.full(64*1800,-1,np.int64)
    pixel_to_voxel[sparse['pixels'].cpu().numpy()]=sparse['inverse'].cpu().numpy()
    inverse=pixel_to_voxel[row*1800+col]
    if (inverse<0).any():
        raise ValueError('Encoder lost an observed point')
    # Match the frozen-cache precision used by all training/evaluation paths.
    sparse['features']=sparse['features'].half()
    foundation=sparse['features'].float().cpu().numpy()
    raster=dict(lidar=lidar_tensor,valid=valid_tensor,geometry_features=[sparse],row=row,col=col)
    return inverse,foundation,raster


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lidar', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path, default=ROOT / 'reports/leaderboard.json')
    parser.add_argument('--device', default='cuda', choices=['cuda'],help='Raw sparse encoding requires CUDA; use predict_cached.py for CPU')
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())['best']
    if selection is None:
        raise ValueError('No completed full-training/full-validation model is available yet')
    checkpoint_path = ROOT / selection['checkpoint']
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != selection['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed; refresh the leaderboard')
    with np.load(args.lidar, allow_pickle=False) as archive:
        xyz = archive['xyz'].astype(np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Expected finite XYZ [points,3]')
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError('Raw XYZ encoding requires a Slurm CUDA allocation; cached CPU inference is available in predict_cached.py')
    inverse, foundation, raster = raw_geometry(xyz, device)
    from graph_audio.features import build_graph, waveform, acoustic_field
    from graph_audio.local_assets import load_clap
    observation = build_graph(xyz, inverse, foundation)
    observation['features']=observation['features'].astype(np.float16).astype(np.float32)
    wave = waveform(args.audio)
    observation['acoustic'], observation['audio_context'], observation['delay_tables'] = acoustic_field(observation['centers'], wave, True)
    observation['acoustic']=observation['acoustic'].astype(np.float16).astype(np.float32)
    observation['delay_tables']=observation['delay_tables'].astype(np.float16).astype(np.float32)
    embedding, logits = load_clap(device)([wave])
    observation['embedding'] = embedding[0].float().cpu().numpy()
    observation['text_logits'] = logits[0].float().cpu().numpy()
    if selection.get('audio_head_sha256'):
        spectral = selection.get('audio_head_variant')=='spectral'
        audio_path = ROOT/('runs/audio-category-spectral/best.pt' if spectral else 'runs/audio-category/best.pt')
        if hashlib.sha256(audio_path.read_bytes()).hexdigest() != selection['audio_head_sha256']:
            raise ValueError('Learned audio head provenance mismatch')
        if spectral:
            from graph_audio.spectral_semantics import SpectralAudioCategoryHead,spectral_features
            audio_head = SpectralAudioCategoryHead().to(device).eval()
            audio_features = torch.from_numpy(np.concatenate((observation['embedding'],spectral_features(wave)))).to(device)
        else:
            from graph_audio.audio_semantic import AudioCategoryHead
            audio_head = AudioCategoryHead().to(device).eval()
            audio_features = embedding[0]
        audio_head.load_state_dict(torch.load(audio_path,map_location=device,weights_only=False)['model'])
        observation['learned_audio_logits'] = audio_head(audio_features).float().cpu().numpy()
    point_base = None
    checkpoint = None
    if selection['type'] in ('graph','teacher_trees','calibrated_world'):
        if selection['type'] == 'graph':
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            context_config = checkpoint['config']
        else:
            context_config = selection
        if context_config.get('teacher_checkpoint_sha256'):
            from prepare_teacher import load_teacher
            from scipy.signal import resample_poly
            teacher, teacher_digest = load_teacher(device)
            if teacher_digest != context_config['teacher_checkpoint_sha256']:
                raise ValueError('Pretrained world context checkpoint changed')
            captured = {}
            handle = teacher.object_features.register_forward_pre_hook(
                lambda module, incoming: captured.update(geometry=incoming[0]))
            audio = torch.from_numpy(resample_poly(wave, 1, 3, axis=0).T.copy())[None].to(device)
            base = teacher(raster['lidar'], audio, raster['valid'], embedding, logits,
                           geometry_features=raster['geometry_features'])
            handle.remove()
            row, col = raster['row'], raster['col']
            point_base = base['mask_logits'][0, 0].sigmoid().float().cpu().numpy()[row, col]
            point_geometry = torch.cat((captured['geometry'], base['semantic_logits']), 1)[0]
            point_geometry = point_geometry.float().cpu().numpy()[:, row, col].T
            counts = observation['counts']
            observation['teacher_geometry'] = np.column_stack([
                np.bincount(inverse, weights=point_geometry[:, d], minlength=len(counts)) / np.maximum(counts, 1)
                for d in range(41)]).astype(np.float16).astype(np.float32)
            observation['teacher_probability'] = (np.bincount(inverse, weights=point_base,
                    minlength=len(counts)) / np.maximum(counts, 1)).astype(np.float32)
    if selection['type']=='calibrated_world':
        from train_trees import refine
        from scipy.special import expit,logit
        node=np.clip(observation['teacher_probability'],1e-5,1-1e-5)
        smoothed=np.clip(refine(node,observation,selection['graph_strength']),1e-5,1-1e-5)
        from graph_audio.spatial_audio import selected_scene_field
        field=selected_scene_field(observation,selection,ROOT)
        field=(field-field.mean())/max(float(field.std()),.1)
        delta=logit(smoothed)-logit(node)+selection['phase_alpha']*field
        probability=expit(logit(np.clip(point_base,1e-5,1-1e-5))+delta[inverse])
    elif selection['type'] == 'teacher_trees':
        import lightgbm as lgb
        from train_teacher_trees import teacher_inputs
        from scipy.special import expit, logit
        model = lgb.Booster(model_file=str(checkpoint_path))
        node = model.predict(teacher_inputs(observation,dict(geometry=observation['teacher_geometry'],
                                                           probability=observation['teacher_probability']),
                                           learned_audio_logits=observation.get('learned_audio_logits')))
        delta = logit(np.clip(node,1e-5,1-1e-5))-logit(np.clip(observation['teacher_probability'],1e-5,1-1e-5))
        probability = expit(logit(np.clip(point_base,1e-5,1-1e-5))
                            + selection['residual_strength']*delta[inverse])
    elif selection['type'] == 'trees':
        import lightgbm as lgb
        from train_trees import inputs, refine
        model = lgb.Booster(model_file=str(checkpoint_path))
        probability = model.predict(inputs(observation, selection['physics']))
        probability = refine(probability, observation, selection['graph_strength'])
    else:
        from graph_audio.model import AcousticGraphWorld
        from train_graph import OBSERVATIONS
        if checkpoint is None:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = AcousticGraphWorld(use_graph=not checkpoint['config']['no_graph']).to(device).eval()
        if selection['type'] == 'semantic':
            model.geometry.load_state_dict({key[len('geometry.'):]: value for key, value in checkpoint['model'].items()
                                            if key.startswith('geometry.')})
            model.semantic.load_state_dict({key[len('semantic.'):]: value for key, value in checkpoint['model'].items()
                                            if key.startswith('semantic.')})
        else:
            model.load_state_dict(checkpoint['model'])
        tensors = {key: torch.as_tensor(observation[key], device=device,
                   dtype=torch.long if key == 'neighbors' else torch.float32) for key in OBSERVATIONS if key in observation}
        if selection['type'] == 'semantic':
            from scipy.special import softmax
            from train_trees import refine
            semantic = model.semantic(model.geometry(tensors['features'])).softmax(-1).cpu().numpy()
            prior = softmax(observation['text_logits'] / 2) if 'learned_audio_logits' not in observation else softmax(observation['learned_audio_logits'])
            match = (semantic[:, 1:] * prior[None]).sum(-1)
            spatial = observation['acoustic'].mean(1)
            spatial = (spatial - spatial.mean()) / max(float(spatial.std()), .1)
            probability = np.clip(match * np.exp(np.clip(selection['alpha'] * spatial, -5, 5)), 0, 1)
            probability = refine(probability, observation, selection['graph_strength'])
        else:
            result = model(**tensors)
            probability = result['logits'].sigmoid().cpu().numpy()
            if point_base is not None:
                from scipy.special import expit, logit
                probability = expit(logit(np.clip(point_base, 1e-5, 1-1e-5))
                                     + result['logit_delta'].cpu().numpy()[inverse])
    if point_base is None:
        probability = probability[inverse]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, xyz=xyz, probability=probability.astype(np.float32),
                        mask=(probability >= selection['threshold']).astype(np.uint8),
                        threshold=selection['threshold'], model=selection['name'],
                        checkpoint_sha256=selection['checkpoint_sha256'],point_order='Original LiDAR XYZ order')
    print(f'Saved {len(xyz)} point predictions to {args.output}')


if __name__ == '__main__':
    main()
