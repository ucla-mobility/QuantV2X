""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse_onnx import PyramidFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision

class HeterPyramidCollabOnnx(nn.Module):
    def __init__(self, args):
        super(HeterPyramidCollabOnnx, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        self.cam_crop_info = {} 

        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))

            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", self.cav_range[3] / camera_mask_args['grid_conf']['xbound'][1])
                setattr(self, f"crop_ratio_H_{modality_name}", self.cav_range[4] / camera_mask_args['grid_conf']['ybound'][1])
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }

        self.H = self.cav_range[4] - self.cav_range[1]
        self.W = self.cav_range[3] - self.cav_range[0]
        self.fake_voxel_size = 1

        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])

        self.shrink_flag = 'shrink_header' in args
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'], kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1)

        self.compress = 'compressor' in args
        if self.compress:
            self.compressor = NaiveCompressor(args['compressor']['input_dim'], args['compressor']['compress_ratio'])

        self.model_train_init()
        check_trainable_module(self)

    def model_train_init(self):
        if self.compress:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}
        raw_mod_list = data_dict['agent_modality_list']

        if isinstance(raw_mod_list, torch.Tensor):
            idxs = raw_mod_list.tolist()
            agent_modality_list = [self.modality_name_list[i - 1] for i in idxs]
        else:
            agent_modality_list = raw_mod_list

        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len']
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")(feature)
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict and self.sensor_type_dict[modality_name] == "camera":
                feature = modality_feature_dict[modality_name]
                _, _, H, W = feature.shape
                target_H = int(H * eval(f"self.crop_ratio_H_{modality_name}"))
                target_W = int(W * eval(f"self.crop_ratio_W_{modality_name}"))
                crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                modality_feature_dict[modality_name] = crop_func(feature)
                if eval(f"self.depth_supervision_{modality_name}"):
                    output_dict[f"depth_items_{modality_name}"] = eval(f"self.encoder_{modality_name}").depth_items

        counting_dict = {modality_name: 0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []

        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        fused_feature, occ_outputs = self.pyramid_backbone(
            heter_feature_2d, record_len, affine_matrix, agent_modality_list, self.cam_crop_info
        )

        if self.shrink_flag:
            shrinked_feature = self.shrink_conv(fused_feature)
            cls_preds = self.cls_head(shrinked_feature)
            reg_preds = self.reg_head(shrinked_feature)
            dir_preds = self.dir_head(shrinked_feature)
            output_dict['feature_map_after_shrink'] = shrinked_feature
        else:
            cls_preds = self.cls_head(fused_feature)
            reg_preds = self.reg_head(fused_feature)
            dir_preds = self.dir_head(fused_feature)

        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'dir_preds': dir_preds,
            'preds_tensor': torch.cat([cls_preds, reg_preds, dir_preds], dim=1),
            'feature_map_after_fusion': fused_feature,
            'occ_single_list': occ_outputs
        })

        return output_dict

    def forward_onnx_export(self,
                            voxel_features,
                            voxel_coords,
                            voxel_num_points,
                            pairwise_t_matrix,
                            record_len,
                            agent_modality_list):
        # Dummy ops to ensure inclusion in ONNX computation graph
        pairwise_t_matrix = pairwise_t_matrix + 0 * pairwise_t_matrix
        record_len = record_len + 0 * record_len
        agent_modality_list = agent_modality_list + 0 * agent_modality_list

        data_dict = {
            'pairwise_t_matrix': pairwise_t_matrix,
            'record_len': record_len,
            'agent_modality_list': agent_modality_list,
            'inputs_m1': {
                'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points
            }
        }
        return self.forward(data_dict)
