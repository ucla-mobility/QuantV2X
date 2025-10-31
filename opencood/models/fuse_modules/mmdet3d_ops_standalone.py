"""
Standalone implementations of mmdet3d ops that don't require compiled CUDA extensions.
These are pure PyTorch/Python implementations for L40S compatibility with CUDA 11.6.
"""

import torch
import torch_scatter
import numpy as np
from scipy.sparse.csgraph import connected_components


def furthest_point_sample(xyz, npoint):
    """
    Pure PyTorch implementation of Furthest Point Sampling (FPS).

    Args:
        xyz: (B, N, 3) tensor of 3D points
        npoint: number of samples

    Returns:
        idx: (B, npoint) tensor of sampled point indices
    """
    device = xyz.device
    B, N, C = xyz.shape

    idx = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return idx


def scatter_v2(feat, coors, mode, return_inv=True, min_points=0, unq_inv=None, new_coors=None):
    """
    Scatter features according to coordinates.
    Uses torch_scatter which is a standard PyTorch extension.

    Args:
        feat: (N, C) features
        coors: (N, D) coordinates
        mode: 'avg'/'mean', 'sum', or 'max'
        return_inv: whether to return inverse indices
        min_points: minimum points per voxel
        unq_inv: precomputed inverse indices (optional)
        new_coors: precomputed unique coordinates (optional)

    Returns:
        new_feat: (M, C) scattered features
        new_coors: (M, D) unique coordinates
        unq_inv: (N,) inverse indices (if return_inv=True)
    """
    assert feat.size(0) == coors.size(0)
    if mode == 'avg':
        mode = 'mean'

    if unq_inv is None and min_points > 0:
        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
    elif unq_inv is None:
        new_coors, unq_inv = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
    else:
        assert new_coors is not None

    if min_points > 0:
        cnt_per_point = unq_cnt[unq_inv]
        valid_mask = cnt_per_point >= min_points
        feat = feat[valid_mask]
        coors = coors[valid_mask]
        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)

    if mode == 'max':
        new_feat, argmax = torch_scatter.scatter_max(feat, unq_inv, dim=0)
    elif mode in ('mean', 'sum'):
        new_feat = torch_scatter.scatter(feat, unq_inv, dim=0, reduce=mode)
    else:
        raise NotImplementedError

    if not return_inv:
        return new_feat, new_coors
    else:
        return new_feat, new_coors, unq_inv


def find_connected_componets(points, batch_idx, dist):
    """
    Find connected components based on distance threshold.
    Uses scipy's connected_components for graph connectivity.

    Args:
        points: (N, 3) 3D points
        batch_idx: (N,) batch indices
        dist: distance threshold

    Returns:
        components_inds: (N,) component indices
    """
    device = points.device
    bsz = batch_idx.max().item() + 1
    base = 0
    components_inds = torch.zeros_like(batch_idx) - 1

    for i in range(bsz):
        batch_mask = batch_idx == i
        if batch_mask.any():
            this_points = points[batch_mask]
            dist_mat = this_points[:, None, :2] - this_points[None, :, :2]  # only care about xy
            dist_mat = (dist_mat ** 2).sum(2) ** 0.5
            adj_mat = dist_mat < dist
            adj_mat = adj_mat.cpu().numpy()
            c_inds = connected_components(adj_mat, directed=False)[1]
            c_inds = torch.from_numpy(c_inds).to(device).long() + base
            base = c_inds.max().item() + 1
            components_inds[batch_mask] = c_inds

    assert len(torch.unique(components_inds)) == components_inds.max().item() + 1

    return components_inds


# Placeholder implementations for ops not actually used in ClusterFusion
def get_inner_win_inds(*args, **kwargs):
    """Not used in ClusterFusion - placeholder"""
    raise NotImplementedError("get_inner_win_inds not implemented in standalone version")


def ball_query(*args, **kwargs):
    """Not used in ClusterFusion - placeholder"""
    raise NotImplementedError("ball_query not implemented in standalone version")


class Voxelization:
    """Not used in ClusterFusion - placeholder"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Voxelization not implemented in standalone version")
