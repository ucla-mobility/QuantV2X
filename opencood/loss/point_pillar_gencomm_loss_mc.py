# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
from opencood.loss.point_pillar_depth_loss_mc import PointPillarDepthLossMC


class PointPillarGencommLossMC(PointPillarDepthLossMC):
    """
    Multi-class PointPillar loss with GenComm feature reconstruction loss.
    """
    def __init__(self, args):
        super().__init__(args)
        self.generate_weight = args.get('generate_weight', 1.0)

    def forward(self, output_dict, target_dict, suffix=""):
        total_loss = super().forward(output_dict, target_dict, suffix)

        gt_feature = output_dict['gt_feature']
        pred_feature = output_dict['pred_feature']
        generate_loss = nn.MSELoss()(gt_feature, pred_feature)

        total_loss = total_loss + self.generate_weight * generate_loss
        self.loss_dict.update({
            'total_loss': total_loss.item() if hasattr(total_loss, 'item') else total_loss,
            'generate_loss': generate_loss.item(),
        })
        return total_loss
