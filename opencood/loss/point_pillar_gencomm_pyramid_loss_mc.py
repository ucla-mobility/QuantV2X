# -*- coding: utf-8 -*-

import torch.nn.functional as F

from opencood.loss.point_pillar_depth_loss_mc import PointPillarDepthLossMC
from opencood.loss.point_pillar_pyramid_loss_mc import PointPillarPyramidLossMC


class PointPillarGencommPyramidLossMC(PointPillarPyramidLossMC):
    """
    Multi-class GenComm loss with pyramid occupancy supervision.

    This keeps the standard detection/depth losses, adds pyramid occupancy
    supervision for pyramid fusion outputs, and adds the GenComm feature
    reconstruction loss between ground-truth and generated features.
    """

    def __init__(self, args):
        super().__init__(args)
        self.generate_weight = args.get("generate_weight", 1.0)

    def forward(self, output_dict, target_dict, suffix=""):
        total_loss = PointPillarDepthLossMC.forward(self, output_dict, target_dict, suffix)

        if "occ_single_list" in output_dict:
            occ_batch = output_dict["occ_single_list"][0].shape[0]
            target_batch = target_dict["pos_equal_one"].shape[0]
        else:
            occ_batch = None
            target_batch = None

        # Pyramid fusion returns occupancy maps for every agent. The main target
        # dict is per-scene, while label_dict_single is per-agent. Only apply
        # occupancy supervision when those batch dimensions match.
        if occ_batch is not None and occ_batch == target_batch:
            batch_size = target_dict["pos_equal_one"].shape[0]
            occ_loss = self.calc_occ_loss(
                output_dict["occ_single_list"],
                target_dict["pos_equal_one"],
                target_dict["neg_equal_one"],
                batch_size,
            )
            total_loss = total_loss + occ_loss
            self.loss_dict["pyramid_loss"] = occ_loss.item()

        if suffix == "" and "gt_feature" in output_dict and "pred_feature" in output_dict:
            generate_loss = F.mse_loss(output_dict["gt_feature"], output_dict["pred_feature"])
            total_loss = total_loss + self.generate_weight * generate_loss
            self.loss_dict["generate_loss"] = generate_loss.item()

        self.loss_dict["total_loss"] = total_loss.item() if hasattr(total_loss, "item") else total_loss
        return total_loss
