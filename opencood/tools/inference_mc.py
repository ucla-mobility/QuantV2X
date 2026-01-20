# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import random
import time
import numpy as np
from tqdm import tqdm

import torch
import open3d as o3d
import cv2
from torch.utils.data import DataLoader

import opencood.data_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import inference_utils_mc, train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools.inference import Subset
from opencood.utils import eval_utils_mc
from opencood.visualization import vis_utils_mc, simple_vis, vis_bevfeat
import matplotlib.pyplot as plt


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result' 
                             'in npy_test file')
    parser.add_argument('--dataset_mode', type=str, default="")
    parser.add_argument('--epoch', default=None,
                        help="epoch number to load model")
    parser.add_argument('--save_vis_interval', type=int, default=10,
                        help='interval for saving visualization')
    parser.add_argument('--save_bevfeat', action='store_true',
                        help='save fused BEV feature heatmap and overlay')
    parser.add_argument('--bevfeat_scale', type=int, default=1,
                        help='upsample scale for fused BEV heatmap')
    parser.add_argument('--bevfeat_dpi', type=int, default=500,
                        help='dpi for fused BEV heatmap output')
    parser.add_argument('--debug_pred', action='store_true',
                        help='log prediction stats for debugging')
    parser.add_argument('--debug_max_frames', type=int, default=5,
                        help='max frames to log prediction stats')
    opt = parser.parse_args()
    return opt


def set_random_seed(seed=42): #IMPORTANT
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _extract_ego_output(output_dict, ego_modality):
    inner = output_dict.get('ego') if isinstance(output_dict, dict) else None
    if inner is None:
        inner = output_dict
    if isinstance(inner, tuple):
        inner = inner[0]
    if isinstance(inner, dict) and 'cls_preds' not in inner:
        if ego_modality and ego_modality in inner:
            inner = inner[ego_modality]
        elif len(inner) == 1:
            inner = next(iter(inner.values()))
    return inner


def _format_pred_stats(output_dict, batch_data, dataset):
    ego_modality = getattr(dataset, 'ego_modality', None)
    ego_output = _extract_ego_output(output_dict, ego_modality)
    if not isinstance(ego_output, dict):
        return "debug_pred: missing ego output dict"
    cls_preds = ego_output.get('cls_preds')
    reg_preds = ego_output.get('reg_preds')
    if cls_preds is None:
        return "debug_pred: missing cls_preds"

    with torch.no_grad():
        cls_nan = torch.isnan(cls_preds).any().item()
        cls_min = float(cls_preds.min().item())
        cls_max = float(cls_preds.max().item())
        cls_mean = float(cls_preds.mean().item())

        reg_info = "reg=N/A"
        if reg_preds is not None:
            reg_nan = torch.isnan(reg_preds).any().item()
            reg_min = float(reg_preds.min().item())
            reg_max = float(reg_preds.max().item())
            reg_mean = float(reg_preds.mean().item())
            reg_info = (
                f"reg[min={reg_min:.4f} max={reg_max:.4f} mean={reg_mean:.4f} nan={reg_nan}]"
            )

        score_thresh = None
        try:
            score_thresh = dataset.post_processor.params['target_args']['score_threshold']
        except Exception:
            score_thresh = None

        above_info = "above_thr=N/A"
        cav_content = batch_data.get('ego', {})
        all_anchors = cav_content.get('all_anchors')
        if all_anchors is not None:
            anchors = all_anchors.permute(1, 2, 0, 3, 4).contiguous()
            num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
            prob = torch.sigmoid(cls_preds.permute(0, 2, 3, 1))
            prob = prob.reshape(cls_preds.shape[0], num_anchors, -1)
            cls_pred, _ = torch.max(prob, dim=-1)
            thresh = score_thresh if score_thresh is not None else 0.2
            above = (cls_pred > thresh).sum().item()
            total = cls_pred.numel()
            above_info = f"above_thr={above}/{total} thr={thresh}"

    return (
        f"debug_pred: cls[min={cls_min:.4f} max={cls_max:.4f} mean={cls_mean:.4f} nan={cls_nan}] "
        f"{reg_info} {above_info}"
    )


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


def _score_for_vis(pred_score):
    if pred_score is None:
        return None
    if isinstance(pred_score, torch.Tensor):
        if pred_score.dim() > 1:
            return pred_score[:, 0]
        return pred_score
    pred_np = np.asarray(pred_score)
    if pred_np.ndim > 1:
        return torch.from_numpy(pred_np[:, 0])
    return torch.from_numpy(pred_np)
    

def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', "nofusion"]
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'
    inference_utils_mc.seed_all()
    
    hypes = yaml_utils.load_yaml(None, opt)
    if opt.dataset_mode:
        hypes['dataset_mode'] = opt.dataset_mode

    print(hypes['dataset_mode'])

    hypes['validate_dir'] = hypes['test_dir'] # change the validate_dir to test_dir for inference

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False, calibrate=False)
    print(f"{len(opencood_dataset)} samples found.")
    # opencood_subset = Subset(opencood_dataset, range(650,651))
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    print(model.get_memory_footprint())

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
            if isinstance(hook_target, torch.nn.ModuleList) and len(hook_target) > 0:
                hook_target = hook_target[-1]
                hook_name = "fusion_net[-1]"
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

    # Create the dictionary for evaluation
    result_stat = {}
    for class_name in opencood.data_utils.SUPER_CLASS_MAP.keys():
        result_stat[class_name] = {}
        for iou_threshold in [0.3, 0.5, 0.7]:
            result_stat[class_name][iou_threshold] = \
                {'tp': [], 'fp': [], 'gt': 0}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(100):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    total_time = 0
    num_batches = 0

    debug_handle = None
    if opt.debug_pred:
        debug_path = os.path.join(opt.model_dir, 'debug_pred_stats.txt')
        debug_handle = open(debug_path, 'w', encoding='utf-8')

    with tqdm(total=len(data_loader)) as pbar:
        for i, batch_data in enumerate(data_loader):
            # print(i)
            with torch.no_grad():
                # time starts here
                batch_data = train_utils.to_device(batch_data, device)
                if opt.save_bevfeat and feat_cache is not None:
                    feat_cache["fused"] = None
                start_time = time.time()
                if opt.fusion_method == 'late':
                    if opt.debug_pred:
                        output_dict, pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                            inference_utils_mc.inference_late_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                return_output=True)
                    else:
                        pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                            inference_utils_mc.inference_late_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                elif opt.fusion_method == 'nofusion':
                    if opt.debug_pred:
                        output_dict, pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                            inference_utils_mc.inference_nofusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                return_output=True)
                    else:
                        pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                            inference_utils_mc.inference_nofusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                elif opt.fusion_method == 'early':
                    if opt.debug_pred:
                        output_dict, pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                            inference_utils_mc.inference_early_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                return_output=True)
                    else:
                        pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                            inference_utils_mc.inference_early_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                elif opt.fusion_method == 'intermediate':
                    if opt.debug_pred:
                        output_dict, pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                            inference_utils_mc.inference_intermediate_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset,
                                                                        return_output=True)
                    else:
                        pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                            inference_utils_mc.inference_intermediate_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                else:
                    raise NotImplementedError('Only early, late and intermediate'
                                            'fusion is supported.')
                
                end_time = time.time()
                batch_time = end_time - start_time
                total_time += batch_time
                num_batches += 1

                if opt.debug_pred and (i < opt.debug_max_frames or pred_score is None):
                    debug_line = _format_pred_stats(output_dict, batch_data, opencood_dataset)
                    print(debug_line)
                    if debug_handle is not None:
                        debug_handle.write(debug_line + '\n')

                for class_id, class_name in enumerate(result_stat.keys()):
                    class_id += 1
                    for iou_threshold in result_stat[class_name].keys():
                        if pred_score is None or pred_box_tensor is None:
                            eval_utils_mc.caluclate_tp_fp(None,
                                                    None,
                                                    gt_box_tensor,
                                                    result_stat[class_name],
                                                    iou_threshold)
                            continue
                        keep_index_pred = pred_score[:, -1] == class_id
                        keep_index_gt = gt_label_tensor == class_id
                        eval_utils_mc.caluclate_tp_fp(pred_box_tensor[keep_index_pred, ...],
                                                pred_score[keep_index_pred, 0],
                                                gt_box_tensor[keep_index_gt, ...],
                                                result_stat[class_name],
                                                iou_threshold)

                if opt.save_npy:
                    npy_save_path = os.path.join(opt.model_dir, 'npy')
                    if not os.path.exists(npy_save_path):
                        os.makedirs(npy_save_path)
                    inference_utils_mc.save_prediction_gt(pred_box_tensor,
                                                    gt_box_tensor,
                                                    batch_data['ego'][
                                                        'origin_lidar'][0],
                                                    i,
                                                    npy_save_path)

                if opt.show_vis or opt.save_vis:
                    vis_save_path = ''
                    if opt.save_vis and (i % opt.save_vis_interval == 0):
                        vis_save_path = os.path.join(opt.model_dir, 'vis')
                        if not os.path.exists(vis_save_path):
                            os.makedirs(vis_save_path)
                        vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)
                        # Use BEV visualization as in inference.py
                        simple_vis.visualize(
                            {
                                'pred_box_tensor': pred_box_tensor,
                                'gt_box_tensor': gt_box_tensor,
                                'score_tensor': _score_for_vis(pred_score)
                            },
                            batch_data['ego']['origin_lidar'][0],
                            hypes['postprocess']['gt_range'],
                            vis_save_path,
                            method='bev',
                            left_hand=True  # Set to True or False as appropriate
                        )
                        if opt.save_bevfeat and feat_cache is not None:
                            fused_feature = feat_cache.get("fused")
                            if fused_feature is not None:
                                bev_map = vis_bevfeat.bev_feature_to_map(fused_feature, normalize=True)
                                scale = max(1, int(opt.bevfeat_scale))
                                gt_range = hypes['postprocess']['gt_range']
                                target_shape = (
                                    int(round((gt_range[4] - gt_range[1]) * 10 * scale)),
                                    int(round((gt_range[3] - gt_range[0]) * 10 * scale)),
                                )
                                bevfeat_path = os.path.join(os.path.dirname(vis_save_path), "bevfeat_%05d.png" % i)
                                vis_bevfeat.vis_bev(
                                    bev_map,
                                    type=f"fused_{i}",
                                    normalize=False,
                                    save_path=bevfeat_path,
                                    target_shape=target_shape,
                                    dpi=max(1, int(opt.bevfeat_dpi)),
                                )
                                heatmap = _bev_map_to_heatmap(bev_map)
                                overlay_path = os.path.join(os.path.dirname(vis_save_path), "bevfeat_overlay_%05d.png" % i)
                                _overlay_heatmap(vis_save_path, heatmap, overlay_path)
                    elif opt.show_vis:
                        opencood_dataset.visualize_result(pred_box_tensor,
                                                        gt_box_tensor,
                                                        batch_data['ego'][
                                                            'origin_lidar'],
                                                        None,
                                                        opt.show_vis,
                                                        vis_save_path,
                                                        dataset=opencood_dataset)

                if opt.show_sequence:
                    pcd, pred_o3d_box, gt_o3d_box = \
                        vis_utils_mc.visualize_inference_sample_dataloader_with_map(
                            pred_box_tensor,
                            gt_box_tensor,
                            batch_data['ego']['origin_lidar'],
                            None,
                            vis_pcd,
                            mode='constant'
                            )
                    if i == 0:
                        vis.add_geometry(pcd)
                        vis_utils_mc.linset_assign_list(vis,
                                                    vis_aabbs_pred,
                                                    pred_o3d_box,
                                                    update_mode='add')

                        vis_utils_mc.linset_assign_list(vis,
                                                    vis_aabbs_gt,
                                                    gt_o3d_box,
                                                    update_mode='add')

                    vis_utils_mc.linset_assign_list(vis,
                                                vis_aabbs_pred,
                                                pred_o3d_box)
                    vis_utils_mc.linset_assign_list(vis,
                                                vis_aabbs_gt,
                                                gt_o3d_box)
                    vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.001)
            pbar.update(1)

    if debug_handle is not None:
        debug_handle.close()
        # Access the elapsed time using `format_dict["elapsed"]`
        elapsed_time = pbar.format_dict["elapsed"]

        if elapsed_time > 0:
            average_speed = pbar.n / elapsed_time  # pbar.n is the total number of iterations
            print(f"Average iterations per second: {average_speed:.2f} it/s")
        else:
            print("Elapsed time too short to calculate iteration speed.")
    
    eval_utils_mc.eval_final_results(result_stat,
                                  opt.model_dir)
    if opt.show_sequence:
        vis.destroy_window()

    if feat_handle is not None:
        feat_handle.remove()

    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"Total number of batches: {num_batches}")
    print(f"Average inference time per batch: {total_time / num_batches:.4f} seconds")


if __name__ == '__main__':
    main()
