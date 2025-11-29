""" Author: Seth Z. Zhao <sethzhao506@g.ucla.edu>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception

Modified version with explicit encode/decode for inference
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
from opencood.models.heter_pyramid_collab_codebook_mc import HeterPyramidCollabCodebookMC
from opencood.models.sub_modules.codebook import UMGMQuantizer #import codebook module
from opencood.utils.transformation_utils import normalize_pairwise_tfm
import importlib
import torchvision

class HeterPyramidCollabCodebookMCEncDec(HeterPyramidCollabCodebookMC):
    """
    Extended version with explicit encode/decode methods for inference.
    This version separates the encoding and decoding stages to better simulate
    the actual compression/decompression process.
    """

    def encode_features(self, data_dict):
        """
        Encode the input features into compressed codes.

        Parameters
        ----------
        data_dict : dict
            Input data dictionary containing sensor data

        Returns
        -------
        codes : list of torch.Tensor
            Encoded codes from the codebook
        agent_modality_list : list
            List of modality names for each agent
        other_info : dict
            Additional information needed for decoding (affine matrix, record_len, etc.)
        """
        agent_modality_list = data_dict['agent_modality_list']
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len']

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        # Extract features from each modality
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")(feature)
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        # Crop/Pad camera feature map
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)

        # Assemble heterogeneous features
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        # Free memory of intermediate tensors
        del modality_feature_dict, heter_feature_2d_list

        # === Codebook encoding ===
        N, C, H, W = heter_feature_2d.shape
        # Flatten to [N*H*W, C] for encoding
        flattened = heter_feature_2d.permute(0, 2, 3, 1).contiguous().view(-1, C)

        # Free the original tensor after flattening
        del heter_feature_2d

        # Use codebook.encode() to get codes
        # Process with explicit no_grad to ensure no computation graph is built
        with torch.no_grad():
            codes = self.codebook.encode(flattened)

        # Free flattened tensor after encoding
        del flattened

        # Store metadata for decoding
        other_info = {
            'affine_matrix': affine_matrix,
            'record_len': record_len,
            'agent_modality_list': agent_modality_list,
            'feature_shape': (N, C, H, W)
        }

        # Force GPU memory cleanup after encoding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return codes, agent_modality_list, other_info

    def decode_features(self, codes, other_info):
        """
        Decode the compressed codes back to features and perform detection.

        Parameters
        ----------
        codes : list of torch.Tensor
            Encoded codes from the codebook
        other_info : dict
            Additional information from encoding (affine matrix, record_len, etc.)

        Returns
        -------
        output_dict : dict
            Detection results including cls_preds, reg_preds, dir_preds
        """
        output_dict = {'pyramid': 'collab'}

        # Retrieve metadata
        affine_matrix = other_info['affine_matrix']
        record_len = other_info['record_len']
        agent_modality_list = other_info['agent_modality_list']
        N, C, H, W = other_info['feature_shape']

        # === Codebook decoding ===
        # Use codebook.decode() to reconstruct features
        quantized = self.codebook.decode(codes)

        # Reshape back to [N, C, H, W]
        quantized = quantized.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        heter_feature_2d = quantized

        # Free quantized tensor after reshaping
        del quantized

        # Continue with fusion and detection
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

        output_dict.update({'occ_single_list': occ_outputs})

        output_dict.update({'preds_tensor': torch.cat([cls_preds, reg_preds, dir_preds], dim=1)})

        return output_dict

    def forward_with_encdec(self, data_dict):
        """
        Forward pass using encode/decode explicitly.
        This mimics the actual compression/decompression pipeline.

        Parameters
        ----------
        data_dict : dict
            Input data dictionary

        Returns
        -------
        output_dict : dict
            Detection results
        """
        # Encode stage
        codes, agent_modality_list, other_info = self.encode_features(data_dict)

        # Clear any remaining cached memory after encoding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Decode stage
        output_dict = self.decode_features(codes, other_info)

        return output_dict
