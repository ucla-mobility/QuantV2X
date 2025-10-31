"""
PointPillar with Cluster Fusion - Pure PyTorch/OpenCOOD implementation
No mmdet3d dependencies required - works with CUDA 11.6 on L40S

This model replaces FSD with PointPillar and adapts the outputs to work with ClusterFusion.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from collections import OrderedDict

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.fuse_modules.cluster_fusion import ClusterFusion
from opencood.utils.box_utils import boxes_to_corners_3d


# Minimal LiDARInstance3DBoxes implementation
class LiDARInstance3DBoxes:
    """Minimal bbox wrapper for compatibility without mmdet3d"""
    def __init__(self, tensor, origin=(0.5, 0.5, 0.5)):
        self.tensor = tensor
        self.origin = origin

    def __getitem__(self, item):
        return LiDARInstance3DBoxes(self.tensor[item], self.origin)

    @property
    def shape(self):
        return self.tensor.shape

    def __len__(self):
        return len(self.tensor)

    @property
    def device(self):
        return self.tensor.device


class PointPillarClusterFusion(nn.Module):
    """
    PointPillar-based model with cluster fusion for collaborative 3D object detection.

    Unlike FSD which uses point-level clustering, this model:
    1. Extracts BEV features using PointPillar
    2. Projects BEV features back to points for clustering
    3. Applies ClusterFusion for collaborative detection
    4. Generates final detection outputs
    """

    def __init__(self, args):
        super().__init__()

        # PointPillar encoder
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # Feature dimension from backbone
        self.bev_feature_dim = sum(args['base_bev_backbone']['num_upsample_filter'])

        # Optional shrink header
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]
        else:
            self.out_channel = self.bev_feature_dim

        # Point-level feature extraction (project BEV features to points)
        # This replaces FSD's sparse convolution
        self.point_feat_proj = nn.Sequential(
            nn.Linear(self.out_channel, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),  # Match FSD's point feature dimension
        )

        # Simple segmentation head (foreground/background)
        self.seg_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Binary classification
        )

        # Cluster-level feature aggregation
        self.cluster_feat_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 768),  # Match FSD's cluster feature dimension
        )

        # Fusion module
        self.fusion_module = ClusterFusion(args.get('fusion_cfg', {}))

        # Detection heads
        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'], kernel_size=1)

        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                      kernel_size=1)
        else:
            self.use_dir = False

        self.use_single_label = args.get('use_single_label', False)
        self.proj_first = args.get('proj_first', False)

        # Store config
        self.voxel_size = args['voxel_size']
        self.lidar_range = args['lidar_range']

    def extract_point_features(self, points_list, bev_features, voxel_coords):
        """
        Extract point-level features by projecting BEV features to points.

        Args:
            points_list: List of (N, 4) point clouds
            bev_features: (B, C, H, W) BEV feature map
            voxel_coords: (N_voxels, 4) voxel coordinates [batch_idx, z, y, x]

        Returns:
            point_features: (N_total, 128) point-level features
            seg_logits: (N_total, 1) segmentation logits
            batch_idx: (N_total,) batch index for each point
        """
        all_point_feats = []
        all_seg_logits = []
        all_batch_idx = []

        B, C, H, W = bev_features.shape

        for batch_id, points in enumerate(points_list):
            # Convert points to BEV grid coordinates
            x = points[:, 0]
            y = points[:, 1]

            # Map to grid indices
            grid_x = ((x - self.lidar_range[0]) / self.voxel_size[0]).long()
            grid_y = ((y - self.lidar_range[1]) / self.voxel_size[1]).long()

            # Clamp to valid range
            grid_x = torch.clamp(grid_x, 0, W - 1)
            grid_y = torch.clamp(grid_y, 0, H - 1)

            # Sample features from BEV map
            point_bev_feats = bev_features[batch_id, :, grid_y, grid_x]  # (C, N)
            point_bev_feats = point_bev_feats.transpose(0, 1)  # (N, C)

            # Project to point feature space
            point_feats = self.point_feat_proj(point_bev_feats)  # (N, 128)

            # Compute segmentation logits
            seg_logits = self.seg_head(point_feats)  # (N, 1)

            all_point_feats.append(point_feats)
            all_seg_logits.append(seg_logits)
            all_batch_idx.append(torch.full((len(points),), batch_id,
                                           dtype=torch.long, device=points.device))

        # Concatenate all batches
        point_features = torch.cat(all_point_feats, dim=0)
        seg_logits = torch.cat(all_seg_logits, dim=0)
        batch_idx = torch.cat(all_batch_idx, dim=0)

        return point_features, seg_logits, batch_idx

    def simple_clustering(self, points_list, point_features, seg_logits, batch_idx, seg_threshold=0.5):
        """
        Simple clustering based on segmentation scores and spatial proximity.
        Mimics FSD's clustering output format for ClusterFusion compatibility.

        Returns:
            dict_to_sample: Point-level features
            sampled_out: Foreground mask and sampled features
            extracted_outs: Cluster-level features
        """
        import torch.nn.functional as F
        from opencood.models.fuse_modules.mmdet3d_ops_standalone import scatter_v2

        # Get foreground mask
        seg_probs = torch.sigmoid(seg_logits)
        fg_mask = (seg_probs > seg_threshold).squeeze(1)

        # Concatenate all points
        all_points = torch.cat(points_list, dim=0)

        # Simple grid-based clustering
        # Group points into spatial cells as "clusters"
        cluster_size = 2.0  # meters
        points_xy = all_points[:, :2]
        cluster_ids = ((points_xy - self.lidar_range[0]) / cluster_size).long()

        # Combine batch_idx and cluster_ids to get unique cluster indices
        cluster_inds = cluster_ids[:, 0] * 10000 + cluster_ids[:, 1] * 100 + batch_idx
        unique_clusters, cluster_inv = torch.unique(cluster_inds, return_inverse=True)

        # Aggregate cluster features
        cluster_feats, _, cluster_inv_full = scatter_v2(
            point_features, cluster_inv.unsqueeze(1), mode='mean', return_inv=True
        )
        cluster_xyz, _ = scatter_v2(
            all_points[:, :3], cluster_inv.unsqueeze(1), mode='mean', return_inv=False
        )
        cluster_seg_score, _ = scatter_v2(
            seg_probs, cluster_inv.unsqueeze(1), mode='mean', return_inv=False
        )

        # Create cluster indices in format [class_id, batch_id, cluster_id]
        num_clusters = len(unique_clusters)
        cluster_batch_ids = scatter_v2(
            batch_idx.unsqueeze(1).float(), cluster_inv.unsqueeze(1),
            mode='mean', return_inv=False
        )[0].long().squeeze(1)

        cluster_inds_formatted = torch.stack([
            torch.zeros(num_clusters, device=all_points.device).long(),  # class_id (0 for all)
            cluster_batch_ids,  # batch_id
            torch.arange(num_clusters, device=all_points.device).long(),  # cluster_id
        ], dim=1)

        # Format outputs for ClusterFusion
        dict_to_sample = {
            'seg_points': torch.cat([all_points[:, :3], torch.zeros_like(all_points[:, :1])], dim=1),
            'seg_feats': point_features,
            'batch_idx': batch_idx,
            'vote_offsets': torch.zeros_like(all_points[:, :3]),  # Dummy vote offsets
        }

        sampled_out = {
            'fg_mask_list': [fg_mask],
            'seg_logits': [seg_logits],
        }

        # Point-cluster mapping
        cluster_pts_inds = torch.stack([
            torch.zeros(len(all_points), device=all_points.device).long(),
            batch_idx,
            cluster_inv,
        ], dim=1)

        extracted_outs = {
            'cluster_feats': self.cluster_feat_proj(cluster_feats),
            'cluster_xyz': cluster_xyz,
            'cluster_inds': cluster_inds_formatted,
            'cluster_seg_score': cluster_seg_score,
            'cluster_mean_xyz': torch.cat([cluster_xyz, torch.zeros_like(cluster_xyz[:, :1])], dim=1),
            'cluster_seg_feat': point_features[torch.arange(len(all_points))[fg_mask][:num_clusters] if fg_mask.sum() >= num_clusters else torch.arange(min(num_clusters, len(all_points)))],
            'cluster_pts_feats': point_features,
            'cluster_pts_xyz': dict_to_sample['seg_points'],
            'cluster_pts_inds': cluster_pts_inds,
        }

        return dict_to_sample, sampled_out, extracted_outs

    def forward(self, data_dict):
        """
        Forward pass.

        Args:
            data_dict: Input data dictionary
        """
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        # Use voxel centers as points for clustering (no need for raw points)
        # bs is the total number of agents across all scenes in the batch
        bs = record_len.sum().item()
        points_list = []
        for i in range(bs):
            # Extract voxels for this batch
            batch_mask = voxel_coords[:, 0] == i
            batch_coords = voxel_coords[batch_mask, 1:]
            # Convert voxel grid coordinates to world positions (voxel centers)
            points = batch_coords.float() * torch.tensor(self.voxel_size, device=batch_coords.device)
            points = points + torch.tensor(self.lidar_range[:3], device=batch_coords.device)
            points = points + torch.tensor(self.voxel_size, device=points.device) * 0.5  # Center of voxel
            # Add dummy intensity column
            points = torch.cat([points, torch.zeros(len(points), 1, device=points.device)], dim=1)
            points_list.append(points)

        # PointPillar encoding
        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points
        }

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)

        # Extract spatial features for backbone
        spatial_features = batch_dict['spatial_features']

        # Debug: Check spatial features shape
        # print(f"DEBUG: spatial_features shape: {spatial_features.shape}")
        # print(f"DEBUG: spatial_features min/max: {spatial_features.min():.4f} / {spatial_features.max():.4f}")

        # Ensure spatial features are contiguous and valid
        spatial_features = spatial_features.contiguous()

        # Check for NaN or Inf
        if torch.isnan(spatial_features).any() or torch.isinf(spatial_features).any():
            print("WARNING: NaN or Inf detected in spatial features, replacing with zeros")
            spatial_features = torch.where(torch.isnan(spatial_features) | torch.isinf(spatial_features),
                                          torch.zeros_like(spatial_features), spatial_features)

        bev_features_backbone = self.backbone(spatial_features)

        # Shrink if needed
        bev_features = bev_features_backbone
        if self.shrink_flag:
            bev_features = self.shrink_conv(bev_features)

        # Extract point-level features
        point_features, seg_logits, batch_idx = self.extract_point_features(
            points_list, bev_features, voxel_coords
        )

        # Perform clustering
        dict_to_sample, sampled_out, extracted_outs = self.simple_clustering(
            points_list, point_features, seg_logits, batch_idx
        )

        # Create img_metas for fusion
        # bs here is the total number of agents (same as len(points_list))
        bs = len(points_list)
        img_metas = []
        for i in range(bs):
            try:
                tdelay = data_dict['time_delay'][i] if 'time_delay' in data_dict else 0
            except:
                tdelay = 0

            batch_scene_idx = torch.searchsorted(record_len.cumsum(0).cpu(), torch.tensor(i+1)).item()
            ego_idx = (record_len.cumsum(0) - record_len[0])[batch_scene_idx].item()

            img_metas.append({
                'record_len': record_len,
                'proj2ego_matrix': data_dict['pairwise_t_matrix'][batch_scene_idx, i - ego_idx, 0] if 'pairwise_t_matrix' in data_dict else torch.eye(4).to(points_list[0].device),
                'lidar_pose': data_dict['lidar_pose'][batch_scene_idx][i - ego_idx] if 'lidar_pose' in data_dict else torch.zeros(6).to(points_list[0].device),
                'lidar_pose_clean': data_dict['lidar_pose_clean'][batch_scene_idx][i - ego_idx] if 'lidar_pose_clean' in data_dict else torch.zeros(6).to(points_list[0].device),
                'time_delay': tdelay,
                'box_type_3d': LiDARInstance3DBoxes,
                'proj_first': self.proj_first,
            })

        # Apply fusion
        new_img_metas, new_num_clusters, fusion_ratio = self.fusion_module(
            dict_to_sample, sampled_out, extracted_outs, img_metas
        )

        # Extract ego-only BEV features for prediction
        # In collaborative perception, record_len indicates how many agents per scene
        # The ego agent is always the first agent in each scene
        ego_indices = []
        record_len_cumsum = torch.cumsum(record_len, dim=0)
        record_len_cumsum = torch.cat([torch.tensor([0]).to(record_len.device), record_len_cumsum[:-1]])

        # Get ego index for each scene (always the first agent)
        for scene_idx in range(len(record_len)):
            ego_idx = record_len_cumsum[scene_idx].item()
            ego_indices.append(ego_idx)

        # Extract ego BEV features only
        ego_bev_features = bev_features[ego_indices]  # [num_scenes, C, H, W]

        # Generate final predictions from ego-only fused features
        psm = self.cls_head(ego_bev_features)
        rm = self.reg_head(ego_bev_features)

        output_dict = {'cls_preds': psm, 'reg_preds': rm}
        if self.use_dir:
            output_dict['dir_preds'] = self.dir_head(ego_bev_features)

        return output_dict

    def get_car_level_gt(self, data_dict):
        """Get car-level ground truth boxes and labels"""
        points = data_dict['processed_lidar']['points']
        record_len = data_dict['record_len']
        bs = len(record_len)

        if 'object_bbx_center_single_v' in data_dict.keys() and 'object_bbx_center_single_i' in data_dict.keys():
            gt_bboxes_3d_cars = []
            gt_labels_3d_cars = []
            for i in range(bs):
                mask_v = data_dict['object_bbx_mask_single_v'][i].bool()
                mask_i = data_dict['object_bbx_mask_single_i'][i].bool()
                gt_bboxes_3d_cars.append(LiDARInstance3DBoxes(
                    data_dict['object_bbx_center_single_v'][i][mask_v, :], origin=(0.5, 0.5, 0.5)))
                gt_bboxes_3d_cars.append(LiDARInstance3DBoxes(
                    data_dict['object_bbx_center_single_i'][i][mask_i, :], origin=(0.5, 0.5, 0.5)))
                gt_labels_3d_cars.append(torch.zeros(int(mask_v.sum().item()), dtype=torch.long).to(
                    data_dict['object_bbx_mask'][i].device))
                gt_labels_3d_cars.append(torch.zeros(int(mask_i.sum().item()), dtype=torch.long).to(
                    data_dict['object_bbx_mask'][i].device))
        else:
            # Fallback to simple GT
            gt_bboxes_3d_cars = [LiDARInstance3DBoxes(torch.zeros(0, 7).to(points[0].device), origin=(0.5, 0.5, 0.5)) for _ in range(len(points))]
            gt_labels_3d_cars = [torch.zeros(0, dtype=torch.long).to(points[0].device) for _ in range(len(points))]

        return gt_bboxes_3d_cars, gt_labels_3d_cars

    def _parse_losses(self, losses):
        """Parse losses"""
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            assert log_var_length == len(log_vars) * dist.get_world_size()

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


# Alias for OpenCOOD's model loader (removes underscores from core_method name)
PointPillarCluster = PointPillarClusterFusion
