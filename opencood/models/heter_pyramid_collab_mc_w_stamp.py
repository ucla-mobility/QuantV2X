# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License
#
# STAMP-enabled pyramid collaboration model with multi-class heads.

import torch.nn as nn

from opencood.models.heter_model_baseline_w_stamp import HeterModelBaselineWStamp
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone


class HeterPyramidCollabMcWStamp(HeterModelBaselineWStamp):
    """STAMP adapter model using pyramid fusion and multi-class detection heads."""

    def build_backbone(self, modality_name, model_setting):
        """
        Build a ResNetBEVBackbone per modality (pyramid-style backbone).
        """
        setattr(self, f"backbone_flag_{modality_name}", False)
        if model_setting.get("backbone_args", None) and model_setting["backbone_args"] != "identity":
            setattr(self, f"backbone_flag_{modality_name}", True)
            setattr(
                self,
                f"backbone_{modality_name}",
                ResNetBEVBackbone(model_setting["backbone_args"]),
            )

    def forward_backbone(self, feature, modality_name):
        """Forward encoded features through the ResNet BEV backbone."""
        if eval(f"self.backbone_flag_{modality_name}"):
            backbone_out = eval(f"self.backbone_{modality_name}")(feature)
            if isinstance(backbone_out, dict):
                feature = backbone_out.get("spatial_features_2d", backbone_out.get("spatial_features", backbone_out))
            else:
                feature = backbone_out
        return feature

    def forward_shrinker(self, feature, modality_name):
        """Shrink features if a per-modality shrinker is configured."""
        shrinker = getattr(self, f"shrinker_{modality_name}", None)
        if shrinker is None:
            return feature
        return shrinker(feature)

    def build_head(self, modality_name, model_setting):
        """Build multi-class detection heads for pyramid fusion."""
        head_method = model_setting.get("head_method", "point_pillar_pyramid_object_detection_head")
        downsample_rate = model_setting.get("downsample_rate", 1)
        setattr(self, f"head_method_{modality_name}", head_method)
        setattr(self, f"downsample_rate_{modality_name}", downsample_rate)

        if head_method == "point_pillar_pyramid_object_detection_head":
            num_class = model_setting.get("num_class")
            if num_class is None:
                raise ValueError("num_class must be provided in the modality settings for MC heads.")
            anchor_number = model_setting["anchor_number"]

            setattr(
                self,
                f"cls_head_{modality_name}",
                nn.Conv2d(
                    model_setting["in_head"],
                    anchor_number * num_class * num_class,
                    kernel_size=1,
                ),
            )
            setattr(
                self,
                f"reg_head_{modality_name}",
                nn.Conv2d(
                    model_setting["in_head"],
                    7 * anchor_number * num_class,
                    kernel_size=1,
                ),
            )
            if model_setting.get("dir_args", None):
                setattr(
                    self,
                    f"dir_head_{modality_name}",
                    nn.Conv2d(
                        model_setting["in_head"],
                        model_setting["dir_args"]["num_bins"] * anchor_number * num_class,
                        kernel_size=1,
                    ),
                )
        else:
            super().build_head(modality_name, model_setting)
