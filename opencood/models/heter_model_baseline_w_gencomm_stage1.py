# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# A unified framework for LiDAR-only / Camera-only / Heterogeneous collaboration.
# Support multiple fusion strategies.


import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, CoBEVT, Where2commFusion, Who2comFusion
from opencood.models.gencomm_modules.cond_diff import GenComm
from opencood.models.gencomm_modules.enhancer import Enhancer
from opencood.models.gencomm_modules.message_extractor_v2 import MessageExtractorv2
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision
from opencood.visualization.vis_bevfeat import vis_bev
from opencood.models.fuse_modules.fusion_in_one import regroup

class HeterModelBaselineWGenCommStage1(nn.Module):
    def __init__(self, args):
        super(HeterModelBaselineWGenCommStage1, self).__init__()
        self.args = args
        self.gencomm = GenComm(args['gencomm'])
        self.missing_message = args.get('missing_message', False)
        
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.ego_modality = args['ego_modality']

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        self.cam_crop_info = {}

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building 
            """
            if model_setting['backbone_args'] == 'identity':
                setattr(self, f"backbone_{modality_name}", nn.Identity())
            else:
                setattr(self, f"backbone_{modality_name}", BaseBEVBackbone(model_setting['backbone_args'], 
                                                                       model_setting['backbone_args'].get('inplanes',64)))

            """
            shrink conv building
            """
            setattr(self, f"shrinker_{modality_name}", DownsampleConv(model_setting['shrink_header']))
            setattr(self, f"message_extractor_{modality_name}", MessageExtractorv2(args['message_extractor']['in_ch'], args['message_extractor']['out_ch']))
            print('message_extractor: message_extractorv2')

            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1
        
        self.gmatch = False
        if 'gmatch' in args and args['gmatch']:
            self.gmatch = True
            
        self.num_class = args['num_class'] if "num_class" in args else 1
        self.supervise_single = False
        if args.get("supervise_single", False):
            self.supervise_single = True
            in_head_single = args['in_head_single']
            setattr(self, f'cls_head_single', nn.Conv2d(in_head_single, args['anchor_number'] * self.num_class * self.num_class, kernel_size=1))
            setattr(self, f'reg_head_single', nn.Conv2d(in_head_single, args['anchor_number'] * 7 * self.num_class, kernel_size=1))
            setattr(
                self,
                f'dir_head_single',
                nn.Conv2d(
                    in_head_single,
                    args['anchor_number'] * args['dir_args']['num_bins'] * self.num_class,
                    kernel_size=1,
                ),
            )

        if args['fusion_method'] == "max":
            self.fusion_net = MaxFusion()
        if args['fusion_method'] == "att":
            self.fusion_net = AttFusion(args['att']['feat_dim'])
        if args['fusion_method'] == "disconet":
            self.fusion_net = DiscoFusion(args['disconet']['feat_dim'])
        if args['fusion_method'] == "v2vnet":
            self.fusion_net = V2VNetFusion(args['v2vnet'])
        if args['fusion_method'] == 'v2xvit':
            self.fusion_net = V2XViTFusion(args['v2xvit'])
        if args['fusion_method'] == 'cobevt':
            self.fusion_net = CoBEVT(args['cobevt'])
        if args['fusion_method'] == 'where2comm':
            self.fusion_net = Where2commFusion(args['where2comm'])
        if args['fusion_method'] == 'who2com':
            self.fusion_net = Who2comFusion(args['who2com'])
        if args['fusion_method'] == 'pyramid':
            fusion_args = args['fusion_backbone']
            if fusion_args.get("proj_first", False):
                from opencood.models.fuse_modules.pyramid_fuse_onnx import PyramidFusion
            else:
                from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
            self.fusion_net = PyramidFusion(fusion_args)
        self.pyramid_fusion = args['fusion_method'] == 'pyramid'
        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'] * self.num_class * self.num_class,
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'] * self.num_class,
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(
            args['in_head'],
            args['dir_args']['num_bins'] * args['anchor_number'] * self.num_class,
            kernel_size=1,
        ) # BIN_NUM = 2

        if 'enhancer' in args:
            self.enhancer = Enhancer(self.args['enhancer']['in_ch'], [8, 8], 4)
            print("use enhancer")
        
        # compressor will be only trainable
        self.compress = False 
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])
            self.model_train_init()


        # check again which module is not fixed.
        check_trainable_module(self)

    def model_train_init(self):
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict):
        output_dict = {}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        # print(agent_modality_list)

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}
        modality_message_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            if not isinstance(eval(f"self.backbone_{modality_name}"), nn.Identity):
                feature = eval(f"self.backbone_{modality_name}")(feature)
                if isinstance(feature, dict):
                    feature = feature.get('spatial_features_2d', feature.get('spatial_features', feature))
            feature = eval(f"self.shrinker_{modality_name}")(feature)
            message = eval(f"self.message_extractor_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature
            modality_message_dict[modality_name] = message

        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        """
        Assemble heter features and messages
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        heter_message_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            heter_message_list.append(modality_message_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)
        heter_message = torch.stack(heter_message_list)
        
        if not self.training and self.missing_message:  # for missing_massage inference
            # 对heter_message应用mask，保持ego不变，其余40%置0
            for i in range(1, heter_message.shape[0]):
                mask = torch.rand(heter_message.shape[1], heter_message.shape[2], heter_message.shape[3], device=heter_message.device) > 0.4
                heter_message[i] = heter_message[i] * mask
            
            
        
        
        conditions = heter_message
        """
        Single supervision
        """
        if self.supervise_single:
            cls_preds_before_fusion = self.cls_head_single(heter_feature_2d)
            reg_preds_before_fusion = self.reg_head_single(heter_feature_2d)
            dir_preds_before_fusion = self.dir_head_single(heter_feature_2d)
            output_dict.update({'cls_preds_single': cls_preds_before_fusion,
                                'reg_preds_single': reg_preds_before_fusion,
                                'dir_preds_single': dir_preds_before_fusion})

        """
        Feature Fusion (multiscale).

        we omit self.backbone's first layer.
        """
        
        gt_feature = heter_feature_2d
        gen_data_dict = self.gencomm(heter_feature_2d, conditions, record_len)
        pred_feature = gen_data_dict['pred_feature']
        output_dict.update({'gt_feature': gt_feature,
                            'pred_feature': pred_feature})
        
        
        heter_feature_2d = pred_feature
        
        # ###
        # ###replace ego feat ure with gt_feature
        # split_gt_feature = regroup(gt_feature, record_len)
        # split_pred_feature = regroup(heter_feature_2d, record_len)
        # ego_index = 0
        # for index in range(len(split_gt_feature)):
        #     heter_feature_2d[ego_index] = split_gt_feature[index][0]
        #     ego_index = ego_index + split_gt_feature[index].shape[0]
    
        
 
        if len(heter_feature_2d.shape) == 3:
            heter_feature_2d = heter_feature_2d.unsqueeze(0) ## for the case of bs=1 and only ego
        if hasattr(self, 'enhancer'):
            heter_feature_2d = self.enhancer(heter_feature_2d, affine_matrix, record_len)
            
        if self.pyramid_fusion:
            fused_feature, occ_outputs = self.fusion_net(
                heter_feature_2d,
                record_len,
                affine_matrix,
                agent_modality_list,
                self.cam_crop_info,
            )
            output_dict.update({'pyramid': 'collab', 'occ_single_list': occ_outputs})
        else:
            fused_feature = self.fusion_net(heter_feature_2d, record_len, affine_matrix)

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)


        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds,
                            'message': conditions})

        return output_dict

    def get_memory_footprint(self):
        """Calculate the total memory footprint of the model's parameters and buffers."""
        total_size = 0
        for param in self.parameters():
            total_size += param.nelement() * param.element_size()
        for buffer in self.buffers():
            total_size += buffer.nelement() * buffer.element_size()

        total_size_MB = total_size / (1024 ** 2)  # Convert to MB
        return f"Model Memory Footprint: {total_size_MB:.2f} MB"
