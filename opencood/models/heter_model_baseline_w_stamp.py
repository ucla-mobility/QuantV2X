# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

import torch
import cv2
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.stamp_modules.adapter import Adapter, Reverter
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.fuse_modules.fusion_in_one import (
    MaxFusion,
    AttFusion,
    DiscoFusion,
    V2VNetFusion,
    V2XViTFusion,
    CoBEVT,
    Where2commFusion,
    Who2comFusion,
)
try:
    from opencood.models.sub_modules.naive_decoder import NaiveDecoder
except ImportError:
    class NaiveDecoder(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x
try:
    from opencood.models.sub_modules.bev_seg_head import BevSegHead
except ImportError:
    class BevSegHead(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn

import importlib
import torchvision


class HeterModelBaselineWStamp(nn.Module):

    def __init__(self, args):
        super(HeterModelBaselineWStamp, self).__init__()
        self.args = args
        self.stage = args["stage"]
        self.crop_to_visible = args.get("crop_to_visible", False)
        ignored_modality = set(args.get("ignored_modality", []))
        modality_name_list = list(args.keys() - ignored_modality)
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()]
        self.modality_name_list = modality_name_list
        self.sensor_type_dict = OrderedDict()
        self.cam_crop_info = {}

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            setattr(self, f"cav_range_{modality_name}", model_setting["lidar_range"])
            setattr(
                self, f"visible_range_{modality_name}", model_setting.get("visible_range", model_setting["lidar_range"])
            )

            self.bulid_encoder(modality_name, model_setting)
            self.build_backbone(modality_name, model_setting)
            self.build_shrinker(modality_name, model_setting)
            self.build_aligner(modality_name, model_setting)

            """For feature transformation"""
            setattr(
                self,
                f"H_{modality_name}",
                (eval(f"self.cav_range_{modality_name}")[4] - eval(f"self.cav_range_{modality_name}")[1]),
            )
            setattr(
                self,
                f"W_{modality_name}",
                (eval(f"self.cav_range_{modality_name}")[3] - eval(f"self.cav_range_{modality_name}")[0]),
            )
            self.fake_voxel_size = 1

            # we ignore compressor because adapter and reverter is used
            # instead as a more general solution
            # self.build_compressor(modality_name, model_setting)
            if self.stage != "collab_train":
                self.build_adapter_and_reverter(modality_name, model_setting)
            if self.stage != "train_adapter":
                self.build_fusion(modality_name, model_setting)
                # self.build_shrink_header(modality_name, model_setting)
                self.build_head(modality_name, model_setting)

        self.model_train_init()
        # check again which module is not fixed.
        # check_trainable_module(self)

    def model_train_init(self):
        # if train adapter, then all modules are fixed except adapter and reverter
        if self.stage in ["train_adapter", "train_adapter_w_output", "train_adapter_output_only"]:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            for modality_name in self.modality_name_list:
                if modality_name in ["m0"]:
                    continue
                eval(f"self.adapter_{modality_name}").train()
                eval(f"self.reverter_{modality_name}").train()
                for p in eval(f"self.adapter_{modality_name}").parameters():
                    p.requires_grad_(True)
                for p in eval(f"self.reverter_{modality_name}").parameters():
                    p.requires_grad_(True)
        elif self.stage == "collab_train":
            self.train()
            for p in self.parameters():
                p.requires_grad_(True)
        elif self.stage == "infer":
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            raise NotImplementedError(f"Stage {self.stage} not implemented.")

    def forward(self, data_dict, show_bev=False):
        agent_modality_list = data_dict["agent_modality_list"]
        record_len = data_dict["record_len"]
        
        
        # Filter out the modality that is not ready for inference
        pairwise_t_matrix = data_dict["pairwise_t_matrix"]
        pairwise_t_matrix_new = torch.zeros_like(pairwise_t_matrix)
        agent_modality_list_filtered = []
        record_len_filtered = []
        cur = 0
        count = 0
        ptr = 0
        indices = []
        for m in agent_modality_list:
            if m in self.modality_name_list:
                agent_modality_list_filtered.append(m)
                count += 1
                indices.append(cur)
            cur += 1
            if record_len[ptr] == cur:
                record_len_filtered.append(count)
                if len(indices) > 0:
                    for i in range(len(indices)):
                        for j in range(len(indices)):
                            pairwise_t_matrix_new[ptr][i][j] = pairwise_t_matrix[ptr][indices[i]][indices[j]]
                cur = 0
                count = 0
                ptr += 1
                indices = []
        
        pairwise_t_matrix = pairwise_t_matrix_new
        record_len = torch.tensor(record_len_filtered, device=record_len.device)
        agent_modality_list = agent_modality_list_filtered
                
        output_dict = {m: {"pyramid": "collab"} for m in self.modality_name_list}
        
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        # setup each modality model
        for modality_name in self.modality_name_list:
            if self.stage not in ["train_adapter", "train_adapter_w_output", "train_adapter_output_only"] and modality_name not in modality_count_dict:
                continue

            feature = self.forward_encoder(data_dict, modality_name, output_dict)
            # import pdb; pdb.set_trace()
            feature = self.forward_backbone(feature, modality_name)
            feature = self.forward_shrinker(feature, modality_name)
            # max_val = torch.max(feature)
            # min_val = torch.min(feature)
            # cv2.imwrite("debug/feature.png", ((feature[0].abs().max(0)[0].cpu().numpy() - feature[0].abs().max(0)[0].cpu().numpy().max()) / (feature[0].abs().max(0)[0].cpu().numpy().max() - feature[0].abs().max(0)[0].cpu().numpy().min()) * 255).astype(np.uint8))
            feature = self.forward_aligner(feature, modality_name)

            modality_feature_dict[modality_name] = feature

            if not eval(f"self.multi_sensor_{modality_name}"):
                """
                Crop/Padd camera feature map.
                """
                if "camera" in self.sensor_type_dict[modality_name]:
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H * eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W * eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict[modality_name].update(
                            {f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items}
                        )

        
        """
        Assemble heter features
        """
        protocol_features = dict()
        cur_feature_dict = dict()
        self.forward_adapter_and_reverter(
            modality_count_dict, modality_feature_dict, protocol_features, cur_feature_dict
        )

        if self.stage in ["train_adapter", "train_adapter_w_output"]:
            FM, FP2M, FM2P2M, FP, FM2P = self.postprocess_feature(modality_feature_dict, protocol_features, cur_feature_dict)
            
            if self.stage == "train_adapter":
                return None, (FM, FP2M, FM2P2M, FP, FM2P)

        heter_feature_2d_list_dict = {m: [] for m in self.modality_name_list}
        
        fused_feature_dict = dict()
        
        for transform_idx, modality_name in enumerate(self.modality_name_list):
            if modality_name not in modality_count_dict:
                continue

            counting_dict = {m: 0 for m in self.modality_name_list}
            for m in agent_modality_list:
                feat_idx = counting_dict[m]
                heter_feature_2d_list_dict[modality_name].append(cur_feature_dict[modality_name][m][feat_idx])
                counting_dict[m] += 1
            heter_feature_2d_list_dict[modality_name] = torch.stack(heter_feature_2d_list_dict[modality_name])
            # heter_feature_2d_list_dict[modality_name] = self.forward_compress(heter_feature_2d_list_dict[modality_name], modality_name)

            # heter_feature_2d_list_dict[modality_name] is downsampled 2x
            # add croping information to collaboration module
            fused_feature = self.forward_fusion(
                heter_feature_2d_list_dict[modality_name],
                pairwise_t_matrix,
                modality_name,
                record_len,
                agent_modality_list,
                output_dict[modality_name],
            )

            # fused_feature = self.forward_shrink(fused_feature, modality_name)
            
            fused_feature_dict[modality_name] = fused_feature

            self.forward_head(fused_feature, modality_name, output_dict[modality_name])

        if self.stage in ["train_adapter_w_output", "train_adapter_output_only"] or show_bev:
            protocol_feat = []
            counting_dict = {m: 0 for m in self.modality_name_list}
            for m in agent_modality_list:
                feat_idx = counting_dict[m]
                protocol_feat.append(protocol_features[f"e2p_feature_{m}"][feat_idx])
                counting_dict[m] += 1

            protocol_feat = torch.stack(protocol_feat)
            protocol_fused_feature = self.forward_fusion(
                protocol_feat,
                pairwise_t_matrix,
                "m0",
                record_len,
                agent_modality_list,
                output_dict["m0"],
            )

            # protocol_fused_feature = self.forward_shrink(protocol_fused_feature, "m0")

            self.forward_head(protocol_fused_feature, "m0", output_dict["m0"])

        if show_bev:

            # feature_single = []
            ori_feat_dict = {m: dict() for m in self.modality_name_list}
            counting_dict = {m: 0 for m in self.modality_name_list}
            for i, modality_name in enumerate(agent_modality_list):
                ori_feat_dict[modality_name].update(
                    {i: modality_feature_dict[modality_name][counting_dict[modality_name]]}
                )
            #     feature = eval(f"self.pyramid_backbone_{modality_name}").forward_single(
            #         ori_feat_dict[modality_name][i].unsqueeze(0)
            #     )[0]
            #     if self.shrink_flag:
            #         feature = eval(f"self.shrink_conv_{modality_name}")(feature)[0]
            #     feature_single.append(feature)
                counting_dict[modality_name] += 1

            return (
                output_dict,
                ori_feat_dict,
                heter_feature_2d_list_dict,
                fused_feature_dict,
                None, # feature_single,
                protocol_feat, 
                protocol_fused_feature, 
            )
            
        if self.stage == "train_adapter_w_output":
            return output_dict, (FM, FP2M, FM2P2M, FP, FM2P)
        elif self.stage == "train_adapter_output_only":
            return output_dict, None

        if self.stage == "collab_train" and len(self.modality_name_list) == 1:
            return output_dict[self.modality_name_list[0]]

        return output_dict

    def bulid_encoder(self, modality_name, model_setting):
        """
        Builds the encoder for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        The function dynamically imports the encoder module, determines the type of encoder
        (single sensor or multi-sensor), and sets appropriate attributes for the encoder.
        """

        encoder_filename = "opencood.models.heter_encoders"
        encoder_lib = importlib.import_module(encoder_filename)
        setattr(self, f"multi_sensor_{modality_name}", False)
        if isinstance(model_setting["core_method"], str):
            setattr(self, f"multi_sensor_{modality_name}", False)
            target_model_name = model_setting["core_method"].replace("_", "")
            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            assert model_setting.get("encoder_args", None), "encoder_args should be provided"
            setattr(
                self,
                f"encoder_{modality_name}",
                encoder_class(model_setting["encoder_args"]),
            )
            if model_setting["encoder_args"].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

        elif isinstance(model_setting["core_method"], dict):
            setattr(self, f"multi_sensor_{modality_name}", True)
            target_model_name_camera = model_setting["core_method"]["camera"].replace("_", "")
            target_model_name_lidar = model_setting["core_method"]["lidar"].replace("_", "")
            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name_camera.lower():
                    encoder_class_camera = cls
                if name.lower() == target_model_name_lidar.lower():
                    encoder_class_lidar = cls

            assert model_setting.get("encoder_args_camera", None) and model_setting.get(
                "encoder_args_lidar", None
            ), "for multi_sensor, encoder_args_camera and encoder_args_lidar should be provided"
            setattr(
                self,
                f"encoder_{modality_name}_camera",
                encoder_class_camera(model_setting["encoder_args_camera"]),
            )
            setattr(
                self,
                f"encoder_{modality_name}_lidar",
                encoder_class_lidar(model_setting["encoder_args_lidar"]),
            )
            if model_setting["encoder_args_camera"].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

    def build_backbone(self, modality_name, model_setting):
        """
        Builds the backbone for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the backbone network if the necessary backbone arguments are provided.
        """

        setattr(self, f"backbone_flag_{modality_name}", False)
        if model_setting.get("backbone_args", None) and model_setting["backbone_args"] != 'identity':
            setattr(self, f"backbone_flag_{modality_name}", True)
            setattr(
                self,
                f"backbone_{modality_name}",
                BaseBEVBackbone(model_setting["backbone_args"],
                                model_setting['backbone_args'].get('inplanes',64)),
            )

    def build_shrinker(self, modality_name, model_setting):
        """
        Builds the shrinker for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the backbone network if the necessary backbone arguments are provided.
        """

        self.shrinker_flag = False
        if model_setting.get("shrink_header", None):
            self.shrinker_flag = True
            setattr(
                self,
                f"shrinker_{modality_name}",
                DownsampleConv(model_setting['shrink_header'])
            )

    def build_aligner(self, modality_name, model_setting):
        """
        Builds the aligner for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the aligner network and computes cropping ratios if the sensor type is a camera.
        """

        sensor_name = model_setting["sensor_type"]
        self.sensor_type_dict[modality_name] = sensor_name
        setattr(self, f"aligner_{modality_name}", AlignNet(model_setting["aligner_args"]))

        if "camera" in sensor_name:
            camera_mask_args = model_setting["camera_mask_args"]
            setattr(
                self,
                f"crop_ratio_W_{modality_name}",
                (eval(f"self.cav_range_{modality_name}")[3]) / (camera_mask_args["grid_conf"]["xbound"][1]),
            )
            setattr(
                self,
                f"crop_ratio_H_{modality_name}",
                (eval(f"self.cav_range_{modality_name}")[4]) / (camera_mask_args["grid_conf"]["ybound"][1]),
            )
            setattr(
                self,
                f"xdist_{modality_name}",
                (camera_mask_args["grid_conf"]["xbound"][1] - camera_mask_args["grid_conf"]["xbound"][0]),
            )
            setattr(
                self,
                f"ydist_{modality_name}",
                (camera_mask_args["grid_conf"]["ybound"][1] - camera_mask_args["grid_conf"]["ybound"][0]),
            )
            self.cam_crop_info[modality_name] = {
                f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
            }

    def build_fusion(self, modality_name, model_setting):
        """
        Builds the fusion module for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the fusion method based on the specified fusion method in the model settings.
        """

        """
        Fusion, by default multiscale fusion:
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """

        if model_setting["fusion_method"] == "max":
            setattr(self, f"fusion_net_{modality_name}", MaxFusion())
        elif model_setting["fusion_method"] == "att":
            setattr(
                self,
                f"fusion_net_{modality_name}",
                AttFusion(model_setting["att"]['feat_dim']),
            )
        elif model_setting["fusion_method"] == "disconet":
            setattr(
                self,
                f"fusion_net_{modality_name}",
                DiscoFusion(model_setting["disconet"]['feat_dim']),
            )
        elif model_setting["fusion_method"] == "v2vnet":
            setattr(
                self,
                f"fusion_net_{modality_name}",
                V2VNetFusion(model_setting["v2vnet"]),
            )
        elif model_setting["fusion_method"] == "v2xvit":
            setattr(
                self,
                f"fusion_net_{modality_name}",
                V2XViTFusion(model_setting["v2xvit"]),
            )
        elif model_setting["fusion_method"] == "cobevt":
            setattr(
                self,
                f"fusion_net_{modality_name}",
                CoBEVT(model_setting["cobevt"]),
            )
        elif model_setting["fusion_method"] == "where2comm":
            setattr(
                self,
                f"fusion_net_{modality_name}",
                Where2commFusion(model_setting["where2comm"]),
            )
        elif model_setting["fusion_method"] == "who2com":
            setattr(
                self,
                f"fusion_net_{modality_name}",
                Who2comFusion(model_setting["who2comm"]),
            )
        elif model_setting["fusion_method"] == "pyramid":
            setattr(
                self,
                f"fusion_net_{modality_name}",
                PyramidFusion(model_setting["fusion_backbone"]),
            )
        else:
            raise NotImplementedError(f"Method {model_setting['fusion_method']} not implemented.")
        setattr(self, f"fusion_method_{modality_name}", model_setting["fusion_method"])

        # if model_setting["fusion_method"] != "pyramid":
        #     # other method does not have agent_modality_list and cam_crop_info, neither returning occ_single_list
        #     pyramid_backbone = getattr(self, f"pyramid_backbone_{modality_name}")
        #     pyramid_backbone.forward_collab = lambda *args: (
        #         pyramid_backbone.forward(*args[:3]),
        #         [],
        #     )

    def build_shrink_header(self, modality_name, model_setting):
        """
        Builds the shrink header for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up a downsample convolutional layer if the shrink header is specified.
        """

        setattr(self, f"shrink_flag_{modality_name}", False)
        if "shrink_header" in model_setting:
            setattr(self, f"shrink_flag_{modality_name}", True)
            setattr(
                self,
                f"shrink_conv_{modality_name}",
                DownsampleConv(model_setting["shrink_header"]),
            )

    def build_head(self, modality_name, model_setting):
        """
        Builds the head for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the head network for various head methods such as object detection, segmentation, etc.
        """


        # By default, point pillar pyramid object detection head
        head_method = model_setting.get("head_method", "point_pillar_pyramid_object_detection_head")
        downsample_rate = model_setting.get("downsample_rate", 1)
        setattr(self, f"head_method_{modality_name}", head_method)
        setattr(self, f"downsample_rate_{modality_name}", downsample_rate)
        # self.head_method = model_setting.get("head_method", "point_pillar_pyramid_object_detection_head")
        # self.downsample_rate = model_setting.get("downsample_rate", 1)
        if head_method == "point_pillar_pyramid_object_detection_head":

            setattr(
                self,
                f"cls_head_{modality_name}",
                nn.Conv2d(
                    model_setting["in_head"],
                    model_setting["anchor_number"],
                    kernel_size=1,
                ),
            )
            setattr(
                self,
                f"reg_head_{modality_name}",
                nn.Conv2d(
                    model_setting["in_head"],
                    7 * model_setting["anchor_number"],
                    kernel_size=1,
                ),
            )
            if model_setting.get("dir_args", None):
                setattr(
                    self,
                    f"dir_head_{modality_name}",
                    nn.Conv2d(
                        model_setting["in_head"],
                        model_setting["dir_args"]["num_bins"] * model_setting["anchor_number"],
                        kernel_size=1,
                    ),
                )

        elif head_method == "point_pillar_object_detection_head":
            setattr(
                self,
                f"cls_head_{modality_name}",
                nn.Conv2d(model_setting["in_head"], 1, kernel_size=1),
            )
            setattr(
                self,
                f"reg_head_{modality_name}",
                nn.Conv2d(model_setting["in_head"], 7, kernel_size=1),
            )
            if model_setting.get("dir_args", None):
                setattr(
                    self,
                    f"dir_head_{modality_name}",
                    nn.Conv2d(
                        model_setting["in_head"],
                        model_setting["dir_args"]["num_bins"],
                        kernel_size=1,
                    ),
                )

        elif head_method == "bev_seg_head":
            setattr(
                self,
                f"head_{modality_name}",
                nn.Sequential(
                NaiveDecoder(model_setting["decoder_args"]),
                BevSegHead(
                    model_setting["target"],
                    model_setting["seg_head_dim"],
                    model_setting["output_class_dynamic"],
                    model_setting["output_class_static"],
                    ),
                ),
            )
            
        elif head_method == "seg_head":
            setattr(
                self,
                f"head_{modality_name}",
                nn.Sequential(
                BevSegHead(
                    model_setting["target"],
                    model_setting["seg_head_dim"],
                    model_setting["output_class_dynamic"],
                    model_setting["output_class_static"],
                    ),
                ),
            )

        elif head_method == "pixor_head":

            setattr(
                self,
                f"cls_head_{modality_name}",
                nn.Conv2d(model_setting["in_head"], 1, kernel_size=1),
            )
            setattr(
                self,
                f"reg_head_{modality_name}",
                nn.Conv2d(model_setting["in_head"], 6, kernel_size=1),
            )

        else:
            raise NotImplementedError(f"Head method {head_method} not implemented.")

    def build_compressor(self, modality_name, model_setting):
        """
        Builds the compressor for a given modality.
        # compressor will be only trainable

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the compressor module if the compressor settings are provided.
        """

        setattr(self, f"compress_{modality_name}", False)
        if "compressor" in model_setting:
            setattr(self, f"compress_{modality_name}", True)
            setattr(
                self,
                f"compressor_{modality_name}",
                NaiveCompressor(
                    model_setting["compressor"]["input_dim"],
                    model_setting["compressor"]["compress_ratio"],
                ),
            )

    def get_memory_footprint(self):
            """Calculate the total memory footprint of the model's parameters and buffers."""
            total_size = 0
            for param in self.parameters():
                total_size += param.nelement() * param.element_size()
            for buffer in self.buffers():
                total_size += buffer.nelement() * buffer.element_size()

            total_size_MB = total_size / (1024 ** 2)  # Convert to MB
            return f"Model Memory Footprint: {total_size_MB:.2f} MB"

    def build_adapter_and_reverter(self, modality_name, model_setting):
        """
        Builds the adapter and reverter for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the adapter and reverter modules for modalities other than 'm0'.
        """

        if modality_name != "m0":  # Never equip adapter and reverter for m0
            setattr(self, f"adapter_{modality_name}", Adapter(model_setting["adapter"]))
            setattr(self, f"reverter_{modality_name}", Reverter(model_setting["reverter"]))

    def forward_encoder(self, data_dict, modality_name, output_dict):
        """
        Forwards the input data through the encoder.
        """

        if eval(f"self.multi_sensor_{modality_name}"):
            feature_camera = eval(f"self.encoder_{modality_name}_camera")(
                data_dict, modality_name, eval(f"self.multi_sensor_{modality_name}")
            )

            """
            Crop/Padd camera feature map.
            
            Parameters:
            - data_dict (dict): Input data dictionary.
            - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
            - output_dict (dict): Output data dictionary.

            Returns:
            - feature (Tensor): Encoded features.
            """

            if "camera" in self.sensor_type_dict[modality_name]:
                # should be padding. Instead of masking
                _, _, H, W = feature_camera.shape
                target_H = int(H * eval(f"self.crop_ratio_H_{modality_name}"))
                target_W = int(W * eval(f"self.crop_ratio_W_{modality_name}"))

                crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                feature_camera = crop_func(feature_camera)
                if eval(f"self.depth_supervision_{modality_name}"):
                    output_dict.update(
                        {f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}_camera").depth_items}
                    )

            feature_lidar = eval(f"self.encoder_{modality_name}_lidar")(
                data_dict, modality_name, eval(f"self.multi_sensor_{modality_name}")
            )

            feature = feature_camera + feature_lidar
        else:
            # feature = eval(f"self.encoder_{modality_name}")(
            #     data_dict, modality_name, eval(f"self.multi_sensor_{modality_name}")
            # )
            feature = eval(f"self.encoder_{modality_name}")(
                data_dict, modality_name,)    ## 2025_5_9 changed
        return feature

    def forward_backbone(self, feature, modality_name):
        """
        Forwards the encoded feature through the backbone.

        Parameters:
        - feature (Tensor): Encoded features.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').

        Returns:
        - feature (Tensor): Backbone features.
        """

        if eval(f"self.backbone_flag_{modality_name}"):
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})["spatial_features_2d"]
        return feature

    def forward_aligner(self, feature, modality_name):
        """
        Forwards the feature through the aligner.

        Parameters:
        - feature (Tensor): Backbone features.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').

        Returns:
        - feature (Tensor): Aligned features.
        """

        feature = eval(f"self.aligner_{modality_name}")(feature)
        return feature

    def forward_shrinker(self, feature, modality_name):
        """
        Forwards the feature through the shrinker if available.

        Parameters:
        - feature (Tensor): Backbone features.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').

        Returns:
        - feature (Tensor): Shrunk features.
        """
        feature = eval(f"self.shrinker_{modality_name}")(feature)
        return feature
    
    
    def forward_shrink(self, feature, modality_name):
        """
        Forwards the feature through the shrink header if available.

        Parameters:
        - feature (Tensor): Aligned features.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').

        Returns:
        - feature (Tensor): Shrunken features.
        """

        if getattr(self, f"shrink_flag_{modality_name}"):
            feature = eval(f"self.shrink_conv_{modality_name}")(feature)
        return feature

    def forward_compress(self, feature, modality_name):
        """
        Forwards the feature through the compressor if available.

        Parameters:
        - feature (Tensor): Shrunken features.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').

        Returns:
        - feature (Tensor): Compressed features.
        """

        if getattr(self, f"compress_{modality_name}"):
            feature = eval(f"self.compressor_{modality_name}")(feature)
        return feature

    def forward_adapter_and_reverter(
        self, modality_count_dict, modality_feature_dict, protocol_features, cur_feature_dict
    ):
        """
        Forwards the features through the adapter and reverter.

        Parameters:
        - modality_count_dict (dict): Dictionary of modality counts.
        - modality_feature_dict (dict): Dictionary of modality features.
        - protocol_features (dict): Dictionary of protocol features.
        - cur_feature_dict (dict): Dictionary of current features.

        This function handles the adaptation and reversion of features for different modalities.
        """
        if self.stage == "collab_train":
            for modality_name in self.modality_name_list:
                if modality_name in modality_count_dict:
                    protocol_features[f"e2p_feature_{modality_name}"] = modality_feature_dict[modality_name]

            for cur in self.modality_name_list:
                if cur not in modality_count_dict:
                    continue
                cur_feature_dict[cur] = {}
                for src in self.modality_name_list:
                    if src not in modality_count_dict:
                        continue
                    # No protocol mapping during collab_train; use raw modality features.
                    cur_feature_dict[cur][src] = modality_feature_dict[src]
            return


        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                protocol_features[f"e2p_feature_{modality_name}"] = eval(f"self.adapter_{modality_name}")(modality_feature_dict[modality_name])

        for cur in self.modality_name_list:
            if cur not in modality_count_dict:
                continue
            cur_feature_dict[cur] = dict()
            cur_feature_dict["m0"] = dict()
            for src in self.modality_name_list:
                if src not in modality_count_dict:
                    continue
                if cur == src:  # For training adapter we need to calculate "->Ego" for learning reverter.
                    cur_feature_dict[cur][src] = (
                        modality_feature_dict[cur]
                        if self.stage not in ["train_adapter", "train_adapter_w_output", "train_adapter_output_only"]
                        else eval(f"self.reverter_{cur}")(protocol_features[f"e2p_feature_{src}"].detach())
                    )
                else:
                    cur_feature_dict[cur][src] = eval(f"self.reverter_{cur}")(protocol_features[f"e2p_feature_{src}"])

    def forward_fusion(
        self,
        feature,
        pairwise_t_matrix,
        modality_name,
        record_len,
        agent_modality_list,
        output_dict,
    ):
        """
        Forwards the feature through the fusion module.

        Parameters:
        - feature (Tensor): Compressed features.
        - data_dict (dict): Input data dictionary.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - record_len (int): Length of the record.
        - agent_modality_list (list): List of agent modalities.
        - output_dict (dict): Output data dictionary.

        Returns:
        - fused_feature (Tensor): Fused features.
        """

        affine_matrix = normalize_pairwise_tfm(
            pairwise_t_matrix,
            eval(f"self.H_{modality_name}"),
            eval(f"self.W_{modality_name}"),
            self.fake_voxel_size,
        )

        fusion_method = getattr(self, f"fusion_method_{modality_name}", None)
        if fusion_method == "pyramid":
            fused_out = eval(f"self.fusion_net_{modality_name}")(
                feature,
                record_len,
                affine_matrix,
                agent_modality_list,
                self.cam_crop_info,
            )
            if isinstance(fused_out, tuple):
                fused_feature, occ_outputs = fused_out
                output_dict.update({"occ_single_list": occ_outputs})
            else:
                fused_feature = fused_out
        else:
            fused_feature = eval(f"self.fusion_net_{modality_name}")(feature, record_len, affine_matrix)

        return fused_feature

    def forward_head(self, feature, modality_name, output_dict):
        """
        Forwards the feature through the head network.

        Parameters:
        - feature (Tensor): Fused features.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - output_dict (dict): Output data dictionary.

        This function handles the forward pass for different head methods such as object detection, segmentation, etc.
        """

        if eval(f"self.head_method_{modality_name}") in ["bev_seg_head", "seg_head"]:
            output_dict.update(eval(f"self.head_{modality_name}")(feature))
        else:
            cls_preds = eval(f"self.cls_head_{modality_name}")(feature)
            reg_preds = eval(f"self.reg_head_{modality_name}")(feature)
            if hasattr(self, f"dir_head_{modality_name}"):
                dir_preds = eval(f"self.dir_head_{modality_name}")(feature)
            else:
                dir_preds = None

            output_dict.update({"cls_preds": cls_preds, "reg_preds": reg_preds, "dir_preds": dir_preds})

    def postprocess_feature(self, modality_feature_dict, protocol_features, cur_feature_dict):
        """
        Post-processes the features for training the adapter.

        Parameters:
        - modality_feature_dict (dict): Dictionary of modality features.
        - protocol_features (dict): Dictionary of protocol features.
        - cur_feature_dict (dict): Dictionary of current features.
        - agent_modality_list (list): List of agent modalities.
        - output_dict (dict): Output data dictionary.

        This function handles the post-processing of features for training the adapter.
        """

        modality_list = self.modality_name_list.copy()
        modality_list.remove("m0")
        assert len(modality_list) == 1, "Only one modality is allowed for adapter training"
        ego_modality = modality_list[0]
        
        FM = modality_feature_dict[ego_modality]  # F^{Mi}
        FP2M = eval(f"self.reverter_{ego_modality}")(modality_feature_dict["m0"])  # F^{P->Mi}
        FM2P2M = cur_feature_dict[ego_modality][ego_modality]  # F^{Mi->P->Mi}
        FP = modality_feature_dict["m0"]  # F^P
        FM2P = protocol_features[f"e2p_feature_{ego_modality}"]  # F^{Mi->P}
        
        if self.crop_to_visible:
            # calculate the minimum cav_range to crop the feature
            min_cav_range = np.array([-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf])
            for modality_name in self.modality_name_list:
                cav_range = eval(f"self.visible_range_{modality_name}")
                min_cav_range = np.concatenate(
                    [np.maximum(min_cav_range[:3], cav_range[:3]), np.minimum(min_cav_range[3:], cav_range[3:])]
                )

            # Crop the feature in ego domain
            B, C_FM, H_FM, W_FM = FM.shape
            # feature-lidar range ratio
            X = eval(f"self.cav_range_{ego_modality}")[3] - eval(f"self.cav_range_{ego_modality}")[0]
            Y = eval(f"self.cav_range_{ego_modality}")[4] - eval(f"self.cav_range_{ego_modality}")[1]
            fl_ratio = np.array([X / W_FM, Y / H_FM])

            left_diff = (eval(f"self.cav_range_{ego_modality}")[0] - min_cav_range[0]) / fl_ratio[0]
            right_diff = (min_cav_range[3] - eval(f"self.cav_range_{ego_modality}")[3]) / fl_ratio[0]
            top_diff = (eval(f"self.cav_range_{ego_modality}")[1] - min_cav_range[1]) / fl_ratio[1]
            bottom_diff = (min_cav_range[4] - eval(f"self.cav_range_{ego_modality}")[4]) / fl_ratio[1]

            pad_ego = nn.ZeroPad2d((round(left_diff), round(right_diff), round(top_diff), round(bottom_diff)))
            FM = pad_ego(FM)
            FM2P2M = pad_ego(FM2P2M)
            FP2M = pad_ego(FP2M)

            # Crop the feature in protocol domain
            protocol_modality = "m0"
            B, C_FP, H_FP, W_FP = FP.shape
            # feature-lidar range ratio
            X = eval(f"self.cav_range_{protocol_modality}")[3] - eval(f"self.cav_range_{protocol_modality}")[0]
            Y = eval(f"self.cav_range_{protocol_modality}")[4] - eval(f"self.cav_range_{protocol_modality}")[1]
            fl_ratio = np.array([X / W_FP, Y / H_FP])
            left_diff = (eval(f"self.cav_range_{protocol_modality}")[0] - min_cav_range[0]) / fl_ratio[0]
            right_diff = (min_cav_range[3] - eval(f"self.cav_range_{protocol_modality}")[3]) / fl_ratio[0]
            top_diff = (eval(f"self.cav_range_{protocol_modality}")[1] - min_cav_range[1]) / fl_ratio[1]
            bottom_diff = (min_cav_range[4] - eval(f"self.cav_range_{protocol_modality}")[4]) / fl_ratio[1]
            pad_protocol = nn.ZeroPad2d((round(left_diff), round(right_diff), round(top_diff), round(bottom_diff)))
            FP = pad_protocol(FP)
            FM2P = pad_protocol(FM2P)
            

        return FM, FP2M, FM2P2M, FP, FM2P
