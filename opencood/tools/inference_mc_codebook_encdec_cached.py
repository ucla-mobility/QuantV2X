# -*- coding: utf-8 -*-
# Author: Seth Z. Zhao <sethzhao506@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Inference script with disk caching to avoid OOM.
This version saves encoded codes to disk and loads them for decoding,
preventing memory buildup from keeping both encoding and decoding in memory.
"""

import argparse
import os
import random
import time
import numpy as np
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import DataLoader

import opencood.data_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import inference_utils_mc, train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools.inference import Subset
from opencood.utils import eval_utils_mc
from opencood.visualization import simple_vis


def test_parser():
    parser = argparse.ArgumentParser(description="Codebook encode/decode inference with caching")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='intermediate',
                        help='late, early or intermediate')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result')
    parser.add_argument('--dataset_mode', type=str, default="")
    parser.add_argument('--epoch', default=None,
                        help="epoch number to load model")
    parser.add_argument('--save_vis_interval', type=int, default=10,
                        help='interval for saving visualization')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='directory to cache encoded features')
    parser.add_argument('--encode_only', action='store_true',
                        help='only run encoding and save codes')
    parser.add_argument('--decode_only', action='store_true',
                        help='only run decoding from saved codes')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method == 'intermediate', 'Only intermediate fusion supported for encode/decode'

    inference_utils_mc.seed_all()

    hypes = yaml_utils.load_yaml(None, opt)
    if opt.dataset_mode:
        hypes['dataset_mode'] = opt.dataset_mode

    print(hypes['dataset_mode'])

    hypes['validate_dir'] = hypes['test_dir']

    # Set up cache directory
    if opt.cache_dir is None:
        opt.cache_dir = os.path.join(opt.model_dir, 'encoded_cache')
    os.makedirs(opt.cache_dir, exist_ok=True)
    print(f"Using cache directory: {opt.cache_dir}")

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False, calibrate=False)
    print(f"{len(opencood_dataset)} samples found.")

    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not opt.decode_only:
        print('Creating Model for Encoding')
        # Override to use encode/decode version
        hypes['model']['core_method'] = 'heter_pyramid_collab_codebook_mc_encdec'
        model = train_utils.create_model(hypes)

        if torch.cuda.is_available():
            model.cuda()

        print('Loading Model from checkpoint')
        _, model = train_utils.load_saved_model(opt.model_dir, model)
        model.eval()

        print("=" * 50)
        print("ENCODING PHASE: Extracting and encoding features")
        print("=" * 50)

        with tqdm(total=len(data_loader), desc="Encoding") as pbar:
            for i, batch_data in enumerate(data_loader):
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)
                    cav_content = batch_data['ego']

                    # Only encode, don't decode yet
                    codes, agent_modality_list, other_info = model.encode_features(cav_content)

                    # Save to disk
                    cache_file = os.path.join(opt.cache_dir, f'sample_{i:06d}.pkl')
                    # Move codes to CPU and convert to numpy for smaller file size
                    codes_cpu = [code.cpu().numpy() for code in codes]

                    cache_data = {
                        'codes': codes_cpu,
                        'other_info': {
                            'affine_matrix': other_info['affine_matrix'].cpu().numpy(),
                            'record_len': other_info['record_len'].cpu().numpy(),
                            'agent_modality_list': other_info['agent_modality_list'],
                            'feature_shape': other_info['feature_shape']
                        },
                        'batch_data': batch_data  # Keep for post-processing
                    }

                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)

                    # Clear GPU memory
                    del codes, other_info, batch_data, cav_content
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                pbar.update(1)

        print(f"\nEncoding complete! Cached {len(data_loader)} samples to {opt.cache_dir}")

        if opt.encode_only:
            print("Encode-only mode: exiting now")
            return

        # Free model memory before decoding
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # DECODING PHASE
    print("\n" + "=" * 50)
    print("DECODING PHASE: Loading codes and running detection")
    print("=" * 50)

    # Reload model for decoding
    print('Creating Model for Decoding')
    hypes['model']['core_method'] = 'heter_pyramid_collab_codebook_mc_encdec'
    model = train_utils.create_model(hypes)

    if torch.cuda.is_available():
        model.cuda()

    print('Loading Model from checkpoint')
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    # Create evaluation dict
    result_stat = {}
    for class_name in opencood.data_utils.SUPER_CLASS_MAP.keys():
        result_stat[class_name] = {}
        for iou_threshold in [0.3, 0.5, 0.7]:
            result_stat[class_name][iou_threshold] = {'tp': [], 'fp': [], 'gt': 0}

    total_time = 0
    num_batches = 0

    # Get list of cached files
    cache_files = sorted([f for f in os.listdir(opt.cache_dir) if f.endswith('.pkl')])

    with tqdm(total=len(cache_files), desc="Decoding") as pbar:
        for cache_file in cache_files:
            with torch.no_grad():
                # Load cached data
                with open(os.path.join(opt.cache_dir, cache_file), 'rb') as f:
                    cache_data = pickle.load(f)

                # Move codes back to GPU
                codes = [torch.from_numpy(code).to(device) for code in cache_data['codes']]
                other_info = {
                    'affine_matrix': torch.from_numpy(cache_data['other_info']['affine_matrix']).to(device),
                    'record_len': torch.from_numpy(cache_data['other_info']['record_len']).to(device),
                    'agent_modality_list': cache_data['other_info']['agent_modality_list'],
                    'feature_shape': cache_data['other_info']['feature_shape']
                }
                batch_data = cache_data['batch_data']

                start_time = time.time()

                # Decode and run detection
                from collections import OrderedDict
                output_dict = OrderedDict()
                output_dict['ego'] = model.decode_features(codes, other_info)

                # Post-process
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                    opencood_dataset.post_process(batch_data, output_dict)

                end_time = time.time()
                total_time += (end_time - start_time)
                num_batches += 1

                # Evaluate
                for class_id, class_name in enumerate(result_stat.keys()):
                    class_id += 1
                    for iou_threshold in result_stat[class_name].keys():
                        keep_index_pred = pred_score[:, -1] == class_id
                        keep_index_gt = gt_label_tensor == class_id
                        eval_utils_mc.caluclate_tp_fp(
                            pred_box_tensor[keep_index_pred, ...],
                            pred_score[keep_index_pred, 0],
                            gt_box_tensor[keep_index_gt, ...],
                            result_stat[class_name],
                            iou_threshold
                        )

                # Save visualization if needed
                i = int(cache_file.split('_')[1].split('.')[0])
                if opt.save_vis and (i % opt.save_vis_interval == 0):
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    os.makedirs(vis_save_path, exist_ok=True)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)
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
                        left_hand=True
                    )

                # Clear memory
                del codes, other_info, batch_data, output_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pbar.update(1)

    eval_utils_mc.eval_final_results(result_stat, opt.model_dir)

    print(f"\nTotal decoding time: {total_time:.4f} seconds")
    print(f"Total number of batches: {num_batches}")
    print(f"Average decoding time per batch: {total_time / num_batches:.4f} seconds")


if __name__ == '__main__':
    main()
