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
from opencood.models.heter_pyramid_collab_mc import HeterPyramidCollabMC
from opencood.models.sub_modules.codebook import UMGMQuantizer #import codebook module
from opencood.utils.transformation_utils import normalize_pairwise_tfm
import importlib
import torchvision

class HeterPyramidCollabCodebookMC(HeterPyramidCollabMC):
    def __init__(self, args):
        super(HeterPyramidCollabCodebookMC, self).__init__(args)
        self.channel = 64
        
        if 'codebook' in args:
            self.seg_num = args['codebook']['seg_num']
            self.dict_size = [args['codebook']['dict_size']] * 3
        else:
            self.seg_num = 2
            self.dict_size = [256] * 3  # default to 256 for all stages
     
        self.p_rate = 0.0  # typically 0.0 - don't inject noise

        self.codebook = UMGMQuantizer(
            self.channel,
            self.seg_num,
            self.dict_size,
            self.p_rate,
            {
                "latentStageEncoder": lambda: nn.Linear(self.channel, self.channel),
                "quantizationHead": lambda: nn.Linear(self.channel, self.channel),
                "latentHead": lambda: nn.Linear(self.channel, self.channel),
                "restoreHead": lambda: nn.Linear(self.channel, self.channel),
                "dequantizationHead": lambda: nn.Linear(self.channel, self.channel),
                "sideHead": lambda: nn.Linear(self.channel, self.channel),
            }
        )

    def model_train_init(self):
        # if compress, only make compressor trainable
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
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        # print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")(feature)
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

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
        Assemble heter features
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        # === Codebook logic ===
        N, C, H, W = heter_feature_2d.shape
        """
        N = number of agents

        C = channels

        H, W = spatial size (feature map dimensions)
        """
        flattened = heter_feature_2d.permute(0, 2, 3, 1).contiguous().view(-1, C)
        # Flatten to [N*H*W, C] for quantization
        quantized, _, _, codebook_loss = self.codebook(flattened)
        quantized = quantized.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        heter_feature_2d = quantized
        output_dict.update({'codebook_loss': codebook_loss})
        # ======================

        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        
        fused_feature, occ_outputs = self.pyramid_backbone(
                                                heter_feature_2d,
                                                record_len, 
                                                affine_matrix, 
                                                agent_modality_list, 
                                                self.cam_crop_info
                                            )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})
        
        output_dict.update({'preds_tensor': torch.cat([cls_preds, reg_preds, dir_preds], dim=1)}) #for calibration

        return output_dict

