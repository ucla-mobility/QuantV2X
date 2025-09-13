# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck, BasicBlock
# from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple # causes the affine grid error
from opencood.visualization.debug_plot import plot_feature


def weighted_fuse(x, score, record_len, affine_matrix, align_corners, proj_first=False):
    """
    ONNX-compatible weighted fusion with proper padding.
    """
    total_agents, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    
    # Convert record_len to tensor if it's a list
    if isinstance(record_len, list):
        record_len = torch.tensor(record_len, device=x.device, dtype=torch.long)
    
    if torch.onnx.is_in_onnx_export():
        # During ONNX export, pad the input to B*L shape
        # This avoids dynamic indexing while ensuring correct shape
        padded_x = torch.zeros(B * L, C, H, W, device=x.device, dtype=x.dtype)
        padded_score = torch.zeros(B * L, 1, H, W, device=x.device, dtype=score.dtype)
        
        # Copy the existing agents to the beginning
        # This assumes agents are already in the correct order
        padded_x[:total_agents] = x
        padded_score[:total_agents] = score
        
        # Reshape to batch format
        padded_x = padded_x.reshape(B, L, C, H, W)
        padded_score = padded_score.reshape(B, L, 1, H, W)
    else:
        # Normal training/inference with variable agent counts
        padded_x = torch.zeros(B, L, C, H, W, device=x.device, dtype=x.dtype)
        padded_score = torch.zeros(B, L, 1, H, W, device=x.device, dtype=score.dtype)
        
        # Fill padded tensors using loops (only during training, not ONNX export)
        idx = 0
        for b in range(B):
            num_agents = int(record_len[b].item()) if torch.is_tensor(record_len[b]) else int(record_len[b])
            for i in range(num_agents):
                if idx < total_agents:
                    padded_x[b, i] = x[idx]
                    padded_score[b, i] = score[idx]
                    idx += 1
    
    # Create valid mask using simple comparison
    agent_indices = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)  # (B, L)
    record_len_expanded = record_len.unsqueeze(1)  # (B, 1)
    valid_mask = (agent_indices < record_len_expanded).float()  # (B, L)
    valid_mask = valid_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (B, L, 1, 1, 1)
    
    if proj_first or torch.onnx.is_in_onnx_export():
        # Features are already in ego frame, no warping needed
        feature_in_ego = padded_x
        scores_in_ego = padded_score
    else:
        # Apply warping using batch operations
        ego_matrices = affine_matrix[:, 0, :, :, :]  # (B, L, 2, 3)
        
        # Reshape for batch processing
        B_L = B * L
        ego_matrices_flat = ego_matrices.reshape(B_L, 2, 3)
        padded_x_flat = padded_x.reshape(B_L, C, H, W)
        padded_score_flat = padded_score.reshape(B_L, 1, H, W)
        
        # Apply warping
        feature_in_ego_flat = warp_affine_simple(
            padded_x_flat, ego_matrices_flat, (H, W), align_corners=align_corners
        )
        scores_in_ego_flat = warp_affine_simple(
            padded_score_flat, ego_matrices_flat, (H, W), align_corners=align_corners
        )
        
        # Reshape back
        feature_in_ego = feature_in_ego_flat.reshape(B, L, C, H, W)
        scores_in_ego = scores_in_ego_flat.reshape(B, L, 1, H, W)
    
    # Apply valid mask
    feature_in_ego = feature_in_ego * valid_mask
    scores_in_ego = scores_in_ego * valid_mask
    
    # Process scores - Replace zeros with -inf for invalid positions
    scores_masked = scores_in_ego.masked_fill(scores_in_ego == 0, -float('inf'))
    
    # Apply softmax along agent dimension
    scores_softmax = torch.softmax(scores_masked.squeeze(2), dim=1).unsqueeze(2)  # (B, L, 1, H, W)
    
    # Handle NaN values
    scores_softmax = torch.where(
        torch.isnan(scores_softmax),
        torch.zeros_like(scores_softmax),
        scores_softmax
    )
    
    # Weighted sum across agents
    fused = torch.sum(feature_in_ego * scores_softmax, dim=1)  # (B, C, H, W)
    
    return fused

class PyramidFusion(ResNetBEVBackbone):
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
        super().__init__(model_cfg, input_channels)

        self.stage = model_cfg["stage"]
        self.proj_first = model_cfg.get("proj_first", False) #hardcoded to false unless specified true
        print("Projection first: ", self.proj_first)
        
        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(
                Bottleneck, 
                self.model_cfg['layer_nums'],
                self.model_cfg['layer_strides'],
                self.model_cfg['num_filters'],
                inplanes=model_cfg.get('inplanes', 64),
                groups=32,
                width_per_group=4
            )
        
        self.align_corners = model_cfg.get('align_corners', False)
        print('Align corners: ', self.align_corners)
        
        # add single supervision head
        for i in range(self.num_levels):
            setattr(
                self,
                f"single_head_{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], 1, kernel_size=1),
            )

    def forward_single(self, spatial_features):
        """
        This is used for single agent pass.
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        
        # Create occ_map_list without list comprehension
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = getattr(self, f"single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
        
        final_feature = self.decode_multiscale_feature(feature_list)
        return final_feature, tuple(occ_map_list)
    
    def forward_collab(self, spatial_features, record_len, affine_matrix,
                       agent_modality_list=None, cam_crop_info=None):
        """
        Fusion forward pass.
        """
        # Extract multiscale features
        feature_list = self.get_multiscale_feature(spatial_features)
        
        # Process each level sequentially to avoid list comprehensions
        occ_map_list = []
        fused_feature_list = []
        
        for i in range(self.num_levels):
            # Generate occupancy map
            occ_map = getattr(self, f"single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
            
            # Calculate score
            score = torch.sigmoid(occ_map) + 1e-4
            
            # Apply camera crop mask if needed (simplified for ONNX)
            if cam_crop_info is not None and not self.training and not self.proj_first:
                # Skip complex camera cropping for ONNX export
                # This would need to be reimplemented without dynamic operations
                pass
            
            # Fuse features at this level
            fused = weighted_fuse(
                feature_list[i], 
                score, 
                record_len,
                affine_matrix, 
                self.align_corners,
                proj_first=self.proj_first
            )
            fused_feature_list.append(fused)
        
        # Decode fused features
        fused_feature = self.decode_multiscale_feature(tuple(fused_feature_list))
        
        return fused_feature, tuple(occ_map_list)
    
    def forward(self, spatial_features, record_len=None, affine_matrix=None, 
                agent_modality_list=None, cam_crop_info=None):
        """
        Unified forward method.
        """
        if self.stage == "single":
            return self.forward_single(spatial_features)
        elif self.stage == "collab":
            if record_len is None or affine_matrix is None:
                raise ValueError("record_len and affine_matrix are required for forward_collab()")
            return self.forward_collab(
                spatial_features, record_len, affine_matrix, 
                agent_modality_list, cam_crop_info
            )