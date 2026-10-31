"""Frozen Utonia features for a registered, unstructured two-scan cloud.

Unlike the calibrated single-scan adapter, this path imposes no reference-ray
field-of-view or one-return-per-ray restriction. It retains the frozen backbone,
5 cm physical feature voxels, coordinate scale and TRAIN-fitted projection.
Normals are estimated from measured fused geometry rather than a range image.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree

from .frontend import RawObservationEncoder
from .audio.spatial_features import covariance_from_wave
from .geometry.encoder import scene_tokens
from .geometry.object_proposals import source_hypotheses
from .fusion import validate_cloud


def measured_normals(points, viewpoints, neighbours=16):
    """Local PCA normals, oriented toward the receiver that measured each point."""
    count = min(neighbours, len(points))
    indices = cKDTree(points).query(points, k=count, workers=4)[1]
    indices = np.asarray(indices).reshape(len(points), count)
    local = points[indices].astype(np.float64)
    local -= local.mean(1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', local, local)/max(count, 1)
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, :, 0]
    toward = viewpoints-points
    normal *= np.where((normal*toward).sum(1) < 0, -1., 1.)[:, None]
    return normal.astype(np.float32)


@torch.inference_mode()
def encode_unstructured(geometry, xyz, second_receiver_position, view_membership):
    xyz = validate_cloud(xyz)
    membership = np.asarray(view_membership)
    position = np.asarray(second_receiver_position, dtype=np.float32)
    if membership.shape != (len(xyz),) or not np.isin(membership, [1, 2, 3]).all():
        raise ValueError('Each fused point needs its measured-view membership')
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Expected a finite second receiver position in the reference frame')
    points = torch.as_tensor(xyz, device=geometry.device)
    lower, upper = points.amin(0), points.amax(0)
    shift = torch.stack(((lower[0]+upper[0])/2, (lower[1]+upper[1])/2, lower[2]))
    centered = points-shift
    cells = torch.floor((centered-centered.amin(0))/.05).long()
    _, inverse = torch.unique(cells, dim=0, return_inverse=True)
    number = int(inverse.max())+1
    first = torch.full((number,), len(points), device=geometry.device, dtype=torch.long)
    first.scatter_reduce_(0, inverse, torch.arange(len(points), device=geometry.device), reduce='amin')
    representatives = first.cpu().numpy()
    measured = xyz[representatives]
    viewpoints = np.zeros_like(measured)
    # Both-view cells retain a reference representative by construction.
    viewpoints[membership[representatives] == 2] = position
    normals = torch.as_tensor(measured_normals(measured, viewpoints), device=geometry.device)
    coordinate = centered[first]*.5
    point = geometry.encoder(dict(coord=coordinate, grid_coord=cells[first].int(),
                                  feat=torch.cat([coordinate, torch.zeros_like(coordinate), normals], -1),
                                  batch=torch.zeros(number, dtype=torch.long, device=geometry.device)))
    while 'pooling_parent' in point:
        parent = point.pop('pooling_parent')
        parent.feat = torch.cat([parent.feat, point.feat[point.pop('pooling_inverse')]], -1)
        point = parent
    features = point.feat
    if geometry.projection is not None:
        projection = geometry.projection
        normalized = (features-projection['mean'])/projection['std']
        tail = normalized[:, 54:]@projection['components']
        features = torch.cat([normalized[:, :54], tail/projection['scale']], -1)
    if not torch.isfinite(features).all():
        raise ValueError('Fused backbone features are nonfinite')
    return features, inverse.cpu().numpy()


class FusionObservationEncoder(RawObservationEncoder):
    @torch.inference_mode()
    def encode_fused_arrays(self, xyz, wave, rate, *, second_receiver_position, view_membership):
        D = self.proposals.D
        observed, weight = covariance_from_wave(wave, rate)
        feat, inverse = encode_unstructured(self.geometry, xyz, second_receiver_position, view_membership)
        features = feat.half().cpu().numpy()
        centre, count = D.geometry_inputs(xyz, features, inverse)
        embedding = self.clap([wave], rate)[0].cpu().numpy()
        score, coherence = self.srp(wave[None])
        logits = self.reliability(torch.as_tensor(score, device=self.device),
                                  torch.as_tensor(coherence, device=self.device))
        posterior = (logits-logits.amax(1, keepdim=True)).exp()[0].cpu().numpy()
        silence = not bool(weight.any())
        if silence:
            posterior[:] = 0
        x = dict(features=features, xyz=centre, count=count, inverse=inverse, audio=embedding,
                 acoustic=D.acoustic_features(centre, posterior), silence=silence)
        pred = self.proposals(x)
        points, valid = scene_tokens(centre)
        return dict(scene_points=points, scene_valid=valid,
                    hypotheses=source_hypotheses(centre, count, pred['masks']),
                    rank_features=pred['rank_features'], audio=embedding, baseline=pred['baseline'],
                    observed=observed, reliability=weight, xyz=centre, count=count,
                    point_features=features[:, :16].astype(np.float32),
                    full_point_features=features, inverse=inverse, masks=pred['masks'])
