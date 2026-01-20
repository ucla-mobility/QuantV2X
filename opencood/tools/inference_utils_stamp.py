# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License



import os
from collections import OrderedDict

import numpy as np
import torch
import copy

from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.utils.transformation_utils import get_relative_transformation
from opencood.utils.box_utils import create_bbx, project_box3d, nms_rotated
from opencood.utils.camera_utils import indices_to_depth
from sklearn.metrics import mean_squared_error


def is_multiple_agents(output_dict):
    """
    Check if the output_dict contains multiple agents' prediction.
    
    This function deal with the case of old and new output format
    new format is {m1: pred_m1, m2: pred_m2, ...}
    old format is {pred}
    """
    return [(key[0] == 'm' or key[1:].isdigit()) for key in output_dict.keys()].count(True) != 0

def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

    return_dict = {"pred_box_tensor": pred_box_tensor, "pred_score": pred_score, "gt_box_tensor": gt_box_tensor}
    return return_dict

def inference_heter_late(batch_data, model, dataset, show_bev=False, infer_all=False):
    if infer_all:
        return inference_heter_all_late_fusion(batch_data, model, dataset)
    else:
        return inference_heter_late_fusion(batch_data, model, dataset, show_bev=show_bev)


def inference_heter_late_fusion(batch_data, model, dataset, show_bev=False):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data["ego"]
    ego_modality = cav_content["agent_modality_list"][0]
    if show_bev:
        output_dict, ori_feat_dict, feat_tensor_dict, fused_feat_dict, feat_single, protocol_feat, protocol_fused_feature = (
            model(cav_content, show_bev=show_bev)
        )
        # ori_feat = ori_feat_dict
        fused_feat = fused_feat_dict[ego_modality]
        feat_tensor = feat_tensor_dict[ego_modality]
    else:
        output_dict = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict, fusion='late')

    return_dict = {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "gt_box_tensor": gt_box_tensor,
        "ori_feat_dict": ori_feat_dict if show_bev else None,
        "feat_tensor": feat_tensor if show_bev else None,
        "fused_feat": fused_feat if show_bev else None,
        "feat_single": feat_single if show_bev else None,
        "protocol_feat": protocol_feat if show_bev else None,
        "protocol_fused_feature": protocol_fused_feature if show_bev else None,
        "ego_modality": ego_modality,
    }
    return return_dict


def inference_heter_all_late_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data["ego"]
    ego_modality = cav_content["agent_modality_list"][0]
    output_dict = model(cav_content)
    ret_list = []
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict, fusion='late', agent_idx=0)

    return_dict = {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "gt_box_tensor": gt_box_tensor,
        "ori_feat_dict": None,
        "feat_tensor":  None,
        "fused_feat": None,
        "feat_single": None,
        "protocol_feat": None,
        "protocol_fused_feature": None,
        "ego_modality": ego_modality,
    }
    for agent_idx, agent_modality in enumerate(cav_content["agent_modality_list"]):
        ret_list.append(return_dict)
        
    return ret_list


def inference_no_fusion(batch_data, model, dataset, single_gt=False):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    single_gt : bool
        if True, only use ego agent's label.
        else, use all agent's merged labels.
    """
    output_dict_ego = OrderedDict()
    if single_gt:
        batch_data = {"ego": batch_data["ego"]}

    output_dict_ego["ego"] = model(batch_data["ego"])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process_no_fusion(
        batch_data, output_dict_ego  # only for late fusion dataset
    )

    return_dict = {"pred_box_tensor": pred_box_tensor, "pred_score": pred_score, "gt_box_tensor": gt_box_tensor}
    return return_dict


def inference_no_fusion_w_uncertainty(batch_data, model, dataset):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict_ego = OrderedDict()

    output_dict_ego["ego"] = model(batch_data["ego"])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.
    
    pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor = dataset.post_process_no_fusion_uncertainty(
        batch_data, output_dict_ego  # only for late fusion dataset
    )

    return_dict = {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "gt_box_tensor": gt_box_tensor,
        "uncertainty_tensor": uncertainty_tensor,
    }

    return return_dict


def inference_early_fusion(batch_data, model, dataset, show_bev=False, protocol_result=False):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data["ego"]
    ego_modality = cav_content["agent_modality_list"][0]
    if show_bev:
        o_dict, ori_feat_dict, feat_tensor_dict, fused_feat_dict, feat_single, protocol_feat, protocol_fused_feature = model(
            cav_content, show_bev=show_bev
        )
        # ori_feat = ori_feat_dict
        fused_feat = fused_feat_dict[ego_modality] if fused_feat_dict is not None else None
        if feat_tensor_dict is not None:
            feat_tensor = feat_tensor_dict[ego_modality] if isinstance(feat_tensor_dict, dict) else feat_tensor_dict
    else:
        o_dict = model(cav_content)

    #  TODO: Not only ego but also all agents need to be inferred. This should be boardcasting not p2p.
    # In this version, we take `'agent_modality_list'][0]` to get the output of ego only
    if is_multiple_agents(o_dict):
        if protocol_result:
            o_dict = o_dict["m0"]
            batch_data['ego']['anchor_box'] = batch_data['ego']['anchor_box_dict']['m0'] 
        else:
            o_dict = o_dict[ego_modality]

    output_dict["ego"] = o_dict
    
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

    return_dict = {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "gt_box_tensor": gt_box_tensor,
        "ori_feat_dict": ori_feat_dict if show_bev else None,
        "feat_tensor": feat_tensor if show_bev else None,
        "fused_feat": fused_feat if show_bev else None,
        "feat_single": feat_single if show_bev else None,
        "protocol_feat": protocol_feat if show_bev else None,
        "protocol_fused_feature": protocol_fused_feature if show_bev else None,
        "ego_modality": ego_modality,
    }
    if "depth_items" in output_dict["ego"]:
        return_dict.update({"depth_items": output_dict["ego"]["depth_items"]})
    return return_dict


def inference_all_agents_early_fusion(batch_data, model, dataset, show_bev=False):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    cav_content = batch_data["ego"]
    ego_modality = cav_content["agent_modality_list"][0]
    if show_bev:
        o_dict, ori_feat_dict, feat_tensor_dict, fused_feat_dict, feat_single, protocol_feat, protocol_fused_feature = model(cav_content, show_bev=show_bev)
    else:
        o_dict = model(cav_content)
    ret_list = []
    if is_multiple_agents(o_dict):
        res = dataset.post_process(batch_data, o_dict)
        for pred_box_tensor, pred_score, gt_box_tensor in res:
            return_dict = {
                "pred_box_tensor": pred_box_tensor,
                "pred_score": pred_score,
                "gt_box_tensor": gt_box_tensor,
                "ori_feat_dict": ori_feat_dict if show_bev else None,
                "feat_tensor": feat_tensor_dict if show_bev else None,
                "fused_feat": fused_feat_dict if show_bev else None,
                "feat_single": feat_single if show_bev else None,
                "protocol_feat": protocol_feat if show_bev else None,
                "protocol_fused_feature": protocol_fused_feature if show_bev else None,
                "ego_modality": ego_modality,
            }
            ret_list.append(copy.deepcopy(return_dict))
    else:
        pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, {'ego':o_dict})
        return_dict = {
            "pred_box_tensor": pred_box_tensor,
            "pred_score": pred_score,
            "gt_box_tensor": gt_box_tensor,
            "ori_feat_dict": ori_feat_dict if show_bev else None,
            "feat_tensor": feat_tensor_dict if show_bev else None,
            "fused_feat": fused_feat_dict if show_bev else None,
            "feat_single": feat_single if show_bev else None,
            "protocol_feat": protocol_feat if show_bev else None,
            "protocol_fused_feature": protocol_fused_feature if show_bev else None,
            "ego_modality": ego_modality,
        }
        for modality in cav_content["agent_modality_list"]:
            ret_list.append(return_dict)
    
    return ret_list



def inference_intermediate_fusion(batch_data, model, dataset, infer_all=False, show_bev=False, protocol_result=False):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    if infer_all:
        return_dict = inference_all_agents_early_fusion(
            batch_data, model, dataset, show_bev=show_bev
        )
    else:
        return_dict = inference_early_fusion(
            batch_data, model, dataset, show_bev=show_bev, protocol_result=protocol_result
        )
    return return_dict


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, "%04d_pcd.npy" % timestamp), pcd_np)
    np.save(os.path.join(save_path, "%04d_pred.npy" % timestamp), pred_np)
    np.save(os.path.join(save_path, "%04d_gt.npy" % timestamp), gt_np)


def depth_metric(depth_items, grid_conf):
    # depth logdit: [N, D, H, W]
    # depth gt indices: [N, H, W]
    depth_logit, depth_gt_indices = depth_items
    depth_pred_indices = torch.argmax(depth_logit, 1)
    depth_pred = indices_to_depth(depth_pred_indices, *grid_conf["ddiscr"], mode=grid_conf["mode"]).flatten()
    depth_gt = indices_to_depth(depth_gt_indices, *grid_conf["ddiscr"], mode=grid_conf["mode"]).flatten()
    rmse = mean_squared_error(depth_gt.cpu(), depth_pred.cpu(), squared=False)
    return rmse


def fix_cavs_box(pred_box_tensor, gt_box_tensor, pred_score, batch_data):
    """
    Fix the missing pred_box and gt_box for ego and cav(s).
    Args:
        pred_box_tensor : tensor
            shape (N1, 8, 3), may or may not include ego agent prediction, but it should include
        gt_box_tensor : tensor
            shape (N2, 8, 3), not include ego agent in camera cases, but it should include
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    Returns:
        pred_box_tensor : tensor
            shape (N1+?, 8, 3)
        gt_box_tensor : tensor
            shape (N2+1, 8, 3)
    """
    if pred_box_tensor is None or gt_box_tensor is None:
        return pred_box_tensor, gt_box_tensor, pred_score, 0
    # prepare cav's boxes

    # if key only contains "ego", like intermediate fusion
    if "record_len" in batch_data["ego"]:
        lidar_pose = batch_data["ego"]["lidar_pose"].cpu().numpy()
        N = batch_data["ego"]["record_len"]
        relative_t = get_relative_transformation(lidar_pose)  # [N, 4, 4], cav_to_ego, T_ego_cav
    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data["transformation_matrix"])
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()

    extent = [2.45, 1.06, 0.75]
    ego_box = create_bbx(extent).reshape(1, 8, 3)  # [8, 3]
    ego_box[..., 2] -= 1.2  # hard coded now

    box_list = [ego_box]

    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_tensor = torch.tensor(np.concatenate(box_list, axis=0), device=pred_box_tensor.device)

    pred_box_tensor_ = torch.cat((cav_box_tensor, pred_box_tensor), dim=0)
    gt_box_tensor_ = torch.cat((cav_box_tensor, gt_box_tensor), dim=0)

    pred_score_ = torch.cat((torch.ones(N, device=pred_score.device), pred_score))

    gt_score_ = torch.ones(gt_box_tensor_.shape[0], device=pred_box_tensor.device)
    gt_score_[N:] = 0.5

    keep_index = nms_rotated(pred_box_tensor_, pred_score_, 0.01)
    pred_box_tensor = pred_box_tensor_[keep_index]
    pred_score = pred_score_[keep_index]

    keep_index = nms_rotated(gt_box_tensor_, gt_score_, 0.01)
    gt_box_tensor = gt_box_tensor_[keep_index]

    return pred_box_tensor, gt_box_tensor, pred_score, N


def get_cav_box(batch_data):
    """
    Args:
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    """

    # if key only contains "ego", like intermediate fusion
    if "record_len" in batch_data["ego"]:
        lidar_pose = batch_data["ego"]["lidar_pose"].cpu().numpy()
        N = batch_data["ego"]["record_len"]
        relative_t = get_relative_transformation(lidar_pose)  # [N, 4, 4], cav_to_ego, T_ego_cav
        agent_modality_list = batch_data["ego"]["agent_modality_list"]

    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        agent_modality_list = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data["transformation_matrix"])
            agent_modality_list.append(cav_data["modality_name"])
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()

    extent = [0.2, 0.2, 0.2]
    ego_box = create_bbx(extent).reshape(1, 8, 3)  # [8, 3]
    ego_box[..., 2] -= 1.2  # hard coded now

    box_list = [ego_box]

    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_np = np.concatenate(box_list, axis=0)

    return cav_box_np, agent_modality_list
