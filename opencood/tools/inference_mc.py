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
from torch.utils.data import DataLoader

import opencood.data_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import inference_utils_mc, train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools.inference import Subset
from opencood.utils import eval_utils_mc
from opencood.visualization import vis_utils_mc, simple_vis
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
    opt = parser.parse_args()
    return opt


def set_random_seed(seed=42): #IMPORTANT
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

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

    with tqdm(total=len(data_loader)) as pbar:
        for i, batch_data in enumerate(data_loader):
            # print(i)
            with torch.no_grad():
                # time starts here
                batch_data = train_utils.to_device(batch_data, device)
                start_time = time.time()
                if opt.fusion_method == 'late':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils_mc.inference_late_fusion(batch_data,
                                                            model,
                                                            opencood_dataset)
                elif opt.fusion_method == 'nofusion':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils_mc.inference_nofusion(batch_data,
                                                            model,
                                                            opencood_dataset)
                elif opt.fusion_method == 'early':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils_mc.inference_early_fusion(batch_data,
                                                            model,
                                                            opencood_dataset)
                elif opt.fusion_method == 'intermediate':
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

                for class_id, class_name in enumerate(result_stat.keys()):
                    class_id += 1
                    for iou_threshold in result_stat[class_name].keys():
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
                                'score_tensor': pred_score
                            },
                            batch_data['ego']['origin_lidar'][0],
                            hypes['postprocess']['gt_range'],
                            vis_save_path,
                            method='bev',
                            left_hand=True  # Set to True or False as appropriate
                        )
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

    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"Total number of batches: {num_batches}")
    print(f"Average inference time per batch: {total_time / num_batches:.4f} seconds")


if __name__ == '__main__':
    main()