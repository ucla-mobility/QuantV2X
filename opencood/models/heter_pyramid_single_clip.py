
""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
import torchvision
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
import importlib
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn

class HeterPyramidSingleClip(nn.Module):
    def __init__(self, args):
        super(HeterPyramidSingleClip, self).__init__()
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        print("Active modalities:", self.modality_name_list) # For debugging purposes
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        self.fix_modules = ['pyramid_backbone', 'cls_head', 'reg_head', 'dir_head']
        self.fix_modality = []
        
        
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

            # build encoder
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            # depth supervision for camera
            if model_setting['encoder_args'].get("depth_supervision", False) :
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            # setup backbone (very light-weight)
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))

            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))

            if args.get("fix_encoder", False):
                self.fix_modules += [f"encoder_{modality_name}", f"backbone_{modality_name}"]

            # make sure the old encoder is not trainable
            if model_setting["is_trainable"] == False:
                self.fix_modules += [f"encoder_{modality_name}", f"backbone_{modality_name}"]
                self.fix_modality += [modality_name] # keep track of the old modality (for example 'm1')

        """
        Would load from pretrain base.
        """
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])
        """
        Shrink header, Would load from pretrain base.
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.fix_modules.append('shrink_conv')

        """
        Shared Heads, Would load from pretrain base.
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        
        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)

    def model_train_init(self):
        for module in self.fix_modules:
            for p in eval(f"self.{module}").parameters():
                p.requires_grad_(False)
            eval(f"self.{module}").apply(fix_bn)

    # added for encoder forward
    def forward(self, data_dict):
        modality_names = [x for x in list(data_dict.keys()) if x.startswith("inputs_")]
        assert len(modality_names) == 2, f"Expected 2 modalities, got {len(modality_names)}" # we need 2 modalities for computing clip loss
        mod_b, mod_a = (modality_names[0], modality_names[1]) if modality_names[0] in self.fix_modality else (modality_names[1], modality_names[0])
                
        modality_name_a = mod_a.lstrip('inputs_') # this is the stage 1 modality, treated as teacher for feature space
        modality_name_b = mod_b.lstrip('inputs_') # this is the stage 2 modality, treated as student for feature space
        # print(f"Modality A: {modality_name_a}, Modality B: {modality_name_b}") # For debugging purposes
        feature_a = eval(f"self.encoder_{modality_name_a}")(data_dict, modality_name_a)
        feature_a = eval(f"self.backbone_{modality_name_a}")({"spatial_features": feature_a})['spatial_features_2d']
        feature_b = eval(f"self.encoder_{modality_name_b}")(data_dict, modality_name_b)
        feature_b = eval(f"self.backbone_{modality_name_b}")({"spatial_features": feature_b})['spatial_features_2d']
        feature_b = eval(f"self.aligner_{modality_name_b}")(feature_b)
        # ic(feature_a.shape, feature_b.shape)

        # Take the mean over the channel dimension (dim=1)
        output_feat_a = torch.mean(feature_a, dim=1)  # shape: [batch_size, 120, 240]
        output_feat_b = torch.mean(feature_b, dim=1)  # shape: [batch_size, 120, 240]

        # Flatten the spatial dimensions
        output_feat_a = feature_a.contiguous().view(output_feat_a.shape[0], -1)  # shape: [batch_size, 120*240]
        output_feat_b = feature_b.contiguous().view(output_feat_b.shape[0], -1)  # shape: [batch_size, 120*240]

        output_dict = {'pyramid': 'single'}

        if self.sensor_type_dict[modality_name_b] == "camera":
            # should be padding. Instead of masking
            _, _, H, W = feature_b.shape
            feature_b = torchvision.transforms.CenterCrop(
                    (int(H*eval(f"self.crop_ratio_H_{modality_name_b}")), int(W*eval(f"self.crop_ratio_W_{modality_name_b}")))
                )(feature_b)

            if eval(f"self.depth_supervision_{modality_name_b}"):
                output_dict.update({
                    f"depth_items_{modality_name_b}": eval(f"self.encoder_{modality_name_b}").depth_items
                })
        
        # multiscale fusion. 
        feature_b, occ_map_list = self.pyramid_backbone.forward_single(feature_b)

        if self.shrink_flag:
            feature_b = self.shrink_conv(feature_b)

        cls_preds = self.cls_head(feature_b)
        reg_preds = self.reg_head(feature_b)
        dir_preds = self.dir_head(feature_b)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        output_dict.update({'occ_single_list': occ_map_list})
        output_dict.update({'feature_1': output_feat_a, 'feature_2': output_feat_b})
        return output_dict
