import numpy as np
import torch
import torch.nn.functional as F

def all_pair_l2(A, B):
    """ Calculate all pair l2 distance"""
    A = A.unsqueeze(1)  # [N, 1, 3]
    B = B.unsqueeze(0)  # [1, M, 3]
    C = torch.sqrt(torch.sum((A - B) ** 2, dim=2))  # [N, M]
    return C

def add_distance_for_cluster_points(
        fore_cluster_inds,
        fore_points, 
        fore_points_cluster_inds,
        pred_dx,
    ):
    """ Add distance for cluster points
        Args:
        - fore_cluster_inds [N, 3] [cls, bs, cluster_idx]
        - fore_points [M, 3]
        - fore_points_cluster_inds [M, 3]   [cls, bs, cluster_idx]
        - pred_dx [N, 3]
    """
    # TODO: maybe some bug here
    # breakpoint()
    ids = (fore_points_cluster_inds[:, 2] - fore_points_cluster_inds[:, 2].min()).long()
    pred_dx_points_level = pred_dx[ids]    # [M, 3]
    fore_points[:, 0:3] += pred_dx_points_level
    return fore_points  # [M, 3]

def cluster_latency_align(
        pred_clusters_list,
        fore_cluster_inds,
        fore_points,
        fore_points_cluster_inds,
        up_thres = 1.5,
        down_thres = 0.5,
        dt = 0.1,       # delta t between two frames
        latency_time = 0.1,   # latency time
    ):
    # get cluster with two frame 
    cluster_1 = pred_clusters_list[0]   # [N, 3]
    cluster_2 = pred_clusters_list[1]   # [M, 3]

    # get distance between two cluster
    distance = all_pair_l2(cluster_1, cluster_2)        # [N, M]

    # get min distance and min distance index
    min_distance, ids = torch.min(distance, dim=1)      # [N], [N]
    # get cluster_1 matched cluster_2 
    cluster_1_min_2 = cluster_2[ids]    # [N, 3]
    # calculate velocity
    dx = cluster_1 - cluster_1_min_2    # [N, 3]
    v = dx / dt if dt > 1e-6 else 0     # [N, 3]

    pred_dx = v * latency_time

    # reset unmatched points dx
    cluster_mask = min_distance < up_thres
    pred_dx[cluster_mask == 0] = 0.0

    cluster_mask = min_distance > down_thres
    pred_dx[cluster_mask == 0] = 0.0
    
    fore_points = add_distance_for_cluster_points(
        fore_cluster_inds,
        fore_points,
        fore_points_cluster_inds,
        pred_dx
    )

    cluster_1_refine = cluster_1 + pred_dx

    return fore_points, cluster_1_refine


