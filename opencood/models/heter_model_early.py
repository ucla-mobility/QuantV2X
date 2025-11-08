# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# Heter functionality added by Aiden Wong <aidenwong@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv

class HeterModelEarly(nn.Module):
    """
    Heterogeneous early-fusion detector. All valid agents first share their
    raw point clouds, which are voxelized once and fed through a PointPillar
    style backbone.
    """

    def __init__(self, args):
        super().__init__()
        self.anchor_number = args['anchor_number']
        self.use_dir = 'dir_args' in args and args['dir_args'] is not None
        self.dir_bins = args['dir_args']['num_bins'] if self.use_dir else 0

        modality_name_list = [x for x in args.keys() if x.startswith("m")]
        if len(modality_name_list) == 0:
            raise ValueError("heter_model_early requires at least one modality configuration.")

        # currently we only support lidar-based early fusion
        primary_modality = modality_name_list[0]
        modality_cfg = args[primary_modality]
        if modality_cfg['sensor_type'] != 'lidar':
            raise ValueError("heter_model_early currently only supports lidar modality.")

        encoder_cfg = modality_cfg['encoder_args']
        voxel_size = np.array(encoder_cfg['voxel_size'])
        lidar_range = np.array(encoder_cfg.get('lidar_range', args['lidar_range']))

        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) / voxel_size).astype(np.int64)
        encoder_cfg['point_pillar_scatter']['grid_size'] = grid_size

        self.pillar_vfe = PillarVFE(encoder_cfg['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=encoder_cfg['voxel_size'],
                                    point_cloud_range=encoder_cfg.get('lidar_range', args['lidar_range']))
        self.scatter = PointPillarScatter(encoder_cfg['point_pillar_scatter'])

        self.backbone = ResNetBEVBackbone(modality_cfg['backbone_args'])
        base_channels = 64
        target_channels = modality_cfg['head_args']['in_head']
        if target_channels != base_channels:
            self.channel_align = nn.Conv2d(base_channels, target_channels, kernel_size=1, bias=False)
        else:
            self.channel_align = nn.Identity()
        self.out_channels = target_channels
        self.shrink_flag = False
        self.shrink_conv = nn.Identity()

        self.cls_head = nn.Conv2d(self.out_channels, self.anchor_number, kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channels, self.anchor_number * 7, kernel_size=1)
        if self.use_dir:
            self.dir_head = nn.Conv2d(self.out_channels,
                                      self.anchor_number * self.dir_bins,
                                      kernel_size=1)

    def forward(self, data_dict):
        lidar_dict = data_dict['processed_lidar']
        voxel_features = lidar_dict['voxel_features']
        voxel_coords = lidar_dict['voxel_coords']
        voxel_num_points = lidar_dict['voxel_num_points']

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points
        }

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        spatial_features = batch_dict['spatial_features']

        feature = self.backbone(spatial_features)
        feature = self.shrink_conv(feature)
        feature = self.channel_align(feature)

        cls_preds = self.cls_head(feature)
        reg_preds = self.reg_head(feature)
        output = {
            'cls_preds': cls_preds,
            'reg_preds': reg_preds
        }
        if self.use_dir:
            output['dir_preds'] = self.dir_head(feature)

        return output

    def get_memory_footprint(self):
        total_size = 0
        for param in self.parameters():
            total_size += param.nelement() * param.element_size()
        for buffer in self.buffers():
            total_size += buffer.nelement() * buffer.element_size()
        total_size_MB = total_size / (1024 ** 2)
        return f"Model Memory Footprint: {total_size_MB:.2f} MB"
