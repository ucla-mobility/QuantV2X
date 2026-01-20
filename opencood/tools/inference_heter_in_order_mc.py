"""
-*- coding: utf-8 -*-
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
Multi class functionality added by Aiden Wong <aidenwong@ucla.edu>
License: TDG-Attribution-NonCommercial-NoDistrib

Incrementally increase heterogeneous agents in order.

Actual collaborator m1 -> m1+m2 -> m1+m2+m3 -> m1+m2+m3+m4

Ego is always m1

commrange is 180 (large enough)

For Intermediate Fusion, we will switch to IntermediateHeter3classinferFusionDataset
"""

import argparse
import os
import time
import importlib

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

import opencood.data_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils_mc, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils_mc
from opencood.visualization import simple_vis, vis_bevfeat
from opencood.utils.common_utils import update_dict

torch.multiprocessing.set_sharing_strategy('file_system')


def _safe_num_workers(requested):
    if requested <= 0:
        return 0
    try:
        import multiprocessing as mp
        _ = mp.get_context().Lock()
    except Exception:
        return 0
    return requested


def _normalize_bev_map(bev_map):
    denom = bev_map.max() - bev_map.min()
    if denom <= 0:
        return np.zeros_like(bev_map, dtype=np.float32)
    return (bev_map - bev_map.min()) / denom


def _bev_map_to_heatmap(bev_map):
    bev_norm = _normalize_bev_map(bev_map)
    bev_u8 = (bev_norm * 255).astype(np.uint8)
    return cv2.applyColorMap(bev_u8, cv2.COLORMAP_VIRIDIS)


def _overlay_heatmap(base_path, heatmap, out_path, alpha=0.35):
    base = cv2.imread(base_path)
    if base is None:
        return False
    heatmap_resized = cv2.resize(heatmap, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(base, 1.0 - alpha, heatmap_resized, alpha, 0)
    cv2.imwrite(out_path, overlay)
    return True


def _unwrap_fused_output(output):
    if isinstance(output, (list, tuple)) and output:
        return output[0]
    return output


def build_arg_parser():
    parser = argparse.ArgumentParser(description="heterogeneous inference (multi-class)")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to checkpoint directory.')
    parser.add_argument('--fusion_method', type=str, default='intermediate',
                        help='late, early, intermediate, no, no_w_uncertainty, single')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='save prediction and GT results to npy files')
    parser.add_argument('--range', type=str, default="204.8,102.4",
                        help="detection range, e.g., 204.8,102.4")
    parser.add_argument('--no_score', action='store_true',
                        help="disable prediction score text in visualization")
    parser.add_argument('--use_cav', type=str, default="[1,2,3,4]",
                        help="evaluate with different collaborator counts")
    parser.add_argument('--lidar_degrade', action='store_true',
                        help="degrade lidar channels according to predefined settings")
    parser.add_argument('--note', default="", type=str,
                        help="extra note appended to the result folder name")
    parser.add_argument('--save_bevfeat', action='store_true',
                        help='save fused BEV feature heatmap and overlay')
    parser.add_argument('--bevfeat_scale', type=int, default=1,
                        help='upsample scale for fused BEV heatmap')
    parser.add_argument('--bevfeat_dpi', type=int, default=500,
                        help='dpi for fused BEV heatmap output')
    return parser


def main():
    opt = build_arg_parser().parse_args()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single']

    hypes = yaml_utils.load_yaml(None, opt)

    if 'heter' in hypes:
        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2],
                         x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        hypes = update_dict(hypes, {
            "mapping_dict": {
                "m1": "m1",
                "m2": "m2",
                "m3": "m3",
                "m4": "m4"
            }
        })

        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                hypes = func(hypes)
                break

    hypes['validate_dir'] = hypes['test_dir']
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False
    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes:
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    resume_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    feat_cache = None
    feat_handle = None
    if opt.save_bevfeat:
        feat_cache = {"fused": None}
        hook_target = None
        hook_name = None
        target_model = model.module if hasattr(model, "module") else model
        ego_modality = (
            hypes.get("ego_modality")
            or hypes.get("model", {}).get("args", {}).get("ego_modality")
            or getattr(target_model, "ego_modality", None)
        )
        if hasattr(target_model, "shrink_flag") and target_model.shrink_flag and hasattr(target_model, "shrink_conv"):
            hook_target = target_model.shrink_conv
            hook_name = "shrink_conv"
        elif hasattr(target_model, "fusion_net"):
            hook_target = target_model.fusion_net
            hook_name = "fusion_net"
        elif ego_modality and hasattr(target_model, f"fusion_net_{ego_modality}"):
            hook_target = getattr(target_model, f"fusion_net_{ego_modality}")
            hook_name = f"fusion_net_{ego_modality}"
        else:
            fusion_candidates = sorted(
                name for name in dir(target_model) if name.startswith("fusion_net_")
            )
            if fusion_candidates:
                if ego_modality:
                    for name in fusion_candidates:
                        if name == f"fusion_net_{ego_modality}":
                            hook_target = getattr(target_model, name)
                            hook_name = name
                            break
                if hook_target is None:
                    hook_name = fusion_candidates[0]
                    hook_target = getattr(target_model, hook_name)
        if hook_target is None:
            print("save_bevfeat disabled: fusion module not found on model.")
            opt.save_bevfeat = False
        if opt.save_bevfeat and hook_target is not None:
            def _fused_hook(_module, _input, output):
                feat_cache["fused"] = _unwrap_fused_output(output)
            feat_handle = hook_target.register_forward_hook(_fused_hook)
            print(f"Saving fused BEV features from {hook_name}.")

    np.random.seed(303)

    if opt.fusion_method == 'intermediate':
        fusion_core = hypes['fusion']['core_method']
        if fusion_core in ['intermediate', 'intermediateheter', 'intermediateheter3class']:
            hypes['fusion']['core_method'] = f"{fusion_core}infer"
    hypes['comm_range'] = 180
    hypes['heter']['assignment_path'] = hypes['heter']['assignment_path'].replace(".json", "_in_order.json")
    hypes = update_dict(hypes, {"ego_modality": 'm1'})
    gt_range = hypes['postprocess']['gt_range']

    if opt.lidar_degrade:
        lidar_dict1 = {"m1": 32, "m3": 16}
        lidar_dict2 = {"m1": 16, "m3": 16}
        opt.use_cav = "[4]"
        use_cav_and_lidar_config_pair = [(4, lidar_dict1), (4, lidar_dict2)]
    else:
        base_lidar = {'m3': 32}
        use_cav_and_lidar_config_pair = [(x, base_lidar) for x in eval(opt.use_cav)]

    class_names = list(opencood.data_utils.SUPER_CLASS_MAP.keys())

    for (use_cav, lidar_config) in use_cav_and_lidar_config_pair:
        hypes['use_cav'] = use_cav
        if lidar_config is not None:
            hypes['heter']['lidar_channels_dict'] = lidar_config
            print(hypes['heter']['lidar_channels_dict'])

        print('Dataset Building')
        opencood_dataset = build_dataset(hypes, visualize=True, train=False)
        data_loader = DataLoader(opencood_dataset,
                                 batch_size=1,
                                 num_workers=_safe_num_workers(4),
                                 collate_fn=opencood_dataset.collate_batch_test,
                                 shuffle=False,
                                 pin_memory=False,
                                 drop_last=False)

        result_stat = {
            class_name: {
                iou: {'tp': [], 'fp': [], 'gt': 0}
                for iou in [0.3, 0.5, 0.7]
            } for class_name in class_names
        }

        infer_info = opt.fusion_method + opt.note + f"_use_cav{use_cav}"
        if opt.lidar_degrade:
            infer_info += f"_m1_{lidar_config['m1']}_m3_{lidar_config['m3']}"

        for i, batch_data in enumerate(data_loader):
            print(f"{infer_info}_{i}")
            if batch_data is None:
                continue
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)
                if opt.save_bevfeat and feat_cache is not None:
                    feat_cache["fused"] = None

                if opt.fusion_method == 'late':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils_mc.inference_late_fusion(batch_data, model, opencood_dataset)
                elif opt.fusion_method == 'intermediate':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils_mc.inference_intermediate_fusion(batch_data, model, opencood_dataset)
                elif opt.fusion_method == 'early':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils_mc.inference_early_fusion(batch_data, model, opencood_dataset)
                elif opt.fusion_method in ['no', 'no_w_uncertainty', 'single']:
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils_mc.inference_nofusion(batch_data, model, opencood_dataset)
                else:
                    raise NotImplementedError('Unsupported fusion method for MC heter inference.')

                if pred_score is None:
                    score_column = None
                    class_column = None
                elif isinstance(pred_score, torch.Tensor):
                    if pred_score.dim() > 1:
                        score_column = pred_score[:, 0]
                        class_column = pred_score[:, -1]
                    else:
                        score_column = pred_score
                        class_column = torch.ones_like(score_column)
                else:
                    pred_score_np = np.asarray(pred_score)
                    if pred_score_np.ndim > 1:
                        score_column = torch.from_numpy(pred_score_np[:, 0]).to(device)
                        class_column = torch.from_numpy(pred_score_np[:, -1]).to(device)
                    else:
                        score_column = torch.from_numpy(pred_score_np).to(device)
                        class_column = torch.ones_like(score_column)

                for class_offset, class_name in enumerate(class_names, start=1):
                    keep_pred = (class_column == class_offset) if (pred_box_tensor is not None and class_column is not None) else None
                    keep_gt = (gt_label_tensor == class_offset)
                    for iou in result_stat[class_name]:
                        det_boxes = pred_box_tensor[keep_pred] if (pred_box_tensor is not None and keep_pred is not None) else pred_box_tensor
                        det_scores = score_column[keep_pred] if (pred_box_tensor is not None and keep_pred is not None) else score_column
                        gt_boxes = gt_box_tensor[keep_gt]
                        eval_utils_mc.caluclate_tp_fp(det_boxes,
                                                      det_scores,
                                                      gt_boxes,
                                                      result_stat[class_name],
                                                      iou)

                if opt.save_npy and pred_box_tensor is not None:
                    npy_save_path = os.path.join(opt.model_dir, 'npy_mc')
                    os.makedirs(npy_save_path, exist_ok=True)
                    inference_utils_mc.save_prediction_gt(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'][0],
                        i,
                        npy_save_path)

                if getattr(opencood_dataset, "heterogeneous", False):
                    cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                else:
                    cav_box_np, agent_modality_list = None, None

                infer_result = {
                    'pred_box_tensor': pred_box_tensor,
                    'gt_box_tensor': gt_box_tensor,
                    'score_tensor': score_column if score_column is not None else None,
                    'cav_box_np': cav_box_np,
                    'agent_modality_list': agent_modality_list
                }

                if not opt.no_score and pred_box_tensor is not None:
                    if (i % opt.save_vis_interval == 0):
                        vis_root = os.path.join(opt.model_dir, f'vis_mc_{infer_info}')
                        os.makedirs(vis_root, exist_ok=True)
                        vis_path = os.path.join(vis_root, 'bev_%05d.png' % i)
                        simple_vis.visualize(infer_result,
                                             batch_data['ego']['origin_lidar'][0],
                                             hypes['postprocess']['gt_range'],
                                             vis_path,
                                             method='bev',
                                             left_hand=left_hand)
                        if opt.save_bevfeat and feat_cache is not None:
                            fused_feature = feat_cache.get("fused")
                            if fused_feature is not None:
                                bev_map = vis_bevfeat.bev_feature_to_map(fused_feature, normalize=True)
                                scale = max(1, int(opt.bevfeat_scale))
                                target_shape = (
                                    int(round((gt_range[4] - gt_range[1]) * 10 * scale)),
                                    int(round((gt_range[3] - gt_range[0]) * 10 * scale)),
                                )
                                bevfeat_path = os.path.join(vis_root, "bevfeat_%05d.png" % i)
                                vis_bevfeat.vis_bev(
                                    bev_map,
                                    type=f"fused_{i}",
                                    normalize=False,
                                    save_path=bevfeat_path,
                                    target_shape=target_shape,
                                    dpi=max(1, int(opt.bevfeat_dpi)),
                                )
                                heatmap = _bev_map_to_heatmap(bev_map)
                                overlay_path = os.path.join(vis_root, "bevfeat_overlay_%05d.png" % i)
                                _overlay_heatmap(vis_path, heatmap, overlay_path)

        save_dir = os.path.join(opt.model_dir, f'eval_mc_{infer_info}')
        os.makedirs(save_dir, exist_ok=True)
        eval_utils_mc.eval_final_results(result_stat, save_dir)
    if feat_handle is not None:
        feat_handle.remove()


if __name__ == "__main__":
    main()
