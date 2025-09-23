import argparse
import os
import time
import random
import importlib
import torch
import copy
from icecream import ic
from torch import nn
from opencood.quant.quant_block import QuantBaseBEVBackbone, \
                                            QuantDownsampleConv, \
                                            QuantPFNLayer, \
                                            QuantPillarVFE, \
                                            QuantPointPillar, \
                                            QuantCamEncode_Resnet101, \
                                            QuantPyramidFusion, \
                                            QuantResNetBEVBackbone, \
                                            QuantVoxelBackBone8x, \
                                            QuantV2XViTFusion, \
                                            QuantNaiveCompressor
import open3d as o3d
from torch.utils.data import DataLoader
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools.inference import Subset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
torch.multiprocessing.set_sharing_strategy('file_system')
from opencood.quant import (
    block_reconstruction,
    layer_reconstruction,
    pyramid_reconstruction,
    encoder_reconstruction,
    lss_reconstruction,
    second_reconstruction,
    v2xvit_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
)



def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_train_samples(train_loader, num_batches=128):
    train_data = []
    for batch in train_loader:
        # IMPORTANT: if agent number is inconsistent among frames in list, caching will have error later.
        if len(batch['ego']['agent_modality_list']) != 2:
            continue
        train_data.append(batch['ego'])
        if len(train_data) >= num_batches:
            break
    return train_data

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="102.4,102.4",
                        help="detection range is [-102.4, +102.4, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--seed', default=42, type=int, 
                        help='random seed for results reproduction')

    # quantization parameters
    parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', default=True, help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=8, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_cali_batches', default=128, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--keep_cpu', action='store_true', help='keep the calibration data on cpu')

    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')

    # activation calibration parameters
    parser.add_argument('--lr', default=4e-5, type=float, help='learning rate for LSQ')

    parser.add_argument('--init_wmode', default='minmax', type=str, choices=['minmax', 'mse', 'entropy', 'log'],
                        help='init opt mode for weight')
    parser.add_argument('--init_amode', default='minmax', type=str, choices=['minmax', 'mse', 'entropy', 'log'],
                        help='init opt mode for activation')

    parser.add_argument('--prob', default=0.5, type=float)
    parser.add_argument('--input_prob', default=0.5, type=float)
    parser.add_argument('--lamb_r', default=0.1, type=float, help='hyper-parameter for regularization')
    parser.add_argument('--T', default=4.0, type=float, help='temperature coefficient for KL divergence')
    parser.add_argument('--bn_lr', default=1e-3, type=float, help='learning rate for DC')
    parser.add_argument('--lamb_c', default=0.02, type=float, help='hyper-parameter for DC')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    seed_all(opt.seed)

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    if 'heter' in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir'] or "V2XREAL" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    # build dataset for each noise setting
    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True, calibrate=True) # if calibrate, ignore comm range
    opencood_test_dataset = build_dataset(hypes, visualize=True, train=False, calibrate=False)
    # opencood_test_subset = Subset(opencood_test_dataset, range(700,800))
    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=1,
                              num_workers=16,
                              collate_fn=opencood_train_dataset.collate_batch_test,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    data_loader = DataLoader(opencood_test_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_test_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    print('Creating Model')
    trained_model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, trained_model = train_utils.load_saved_model(saved_path, trained_model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        trained_model.cuda()
    trained_model.eval()

    fp_model = copy.deepcopy(trained_model)
    fp_model.cuda()
    fp_model.eval()

    # Build quantization parameters
    wq_params = {'n_bits': opt.n_bits_w
                ,'channel_wise': opt.channel_wise
                ,'scale_method': opt.init_wmode }
    aq_params = {'n_bits': opt.n_bits_a
                ,'channel_wise': False
                ,'scale_method': opt.init_amode
                ,'leaf_param': True
                ,'prob': opt.prob }
    
    cali_time_start = time.time()
    
    fp_model = QuantModel(model=fp_model, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    fp_model.cuda()
    fp_model.eval()
    fp_model.set_quant_state(False, False)
    
    qt_model = QuantModel(model=trained_model, weight_quant_params=wq_params, act_quant_params=aq_params)
    qt_model.cuda()
    qt_model.eval()

    print(f"FP model device: {next(fp_model.parameters()).device}")
    print(f"QT model device: {next(qt_model.parameters()).device}")

    # if not opt.disable_8bit_head_stem:
    #     print('Setting the first and the last layer to 8-bit')
    #     qt_model.set_first_last_layer_to_8bit()

    # qt_model.disable_network_output_quantization() # disable quantization for 3 detection head at the end

    print('fp_model is below')
    ic(fp_model)

    print('the quantized model is below!')
    ic(qt_model)

    cali_data = get_train_samples(train_loader, num_batches = opt.num_cali_batches)
    device = next(qt_model.parameters()).device

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=opt.iters_w, weight=opt.weight,
                b_range=(opt.b_start, opt.b_end), warmup=opt.warmup, opt_mode='mse',
                lr=opt.lr, input_prob=opt.input_prob, keep_gpu=not opt.keep_cpu, 
                lamb_r=opt.lamb_r, T=opt.T, bn_lr=opt.bn_lr, lamb_c=opt.lamb_c)


    '''init weight quantizer'''
    set_weight_quantize_params(qt_model)
        
    def recon_model(qt: nn.Module, fp: nn.Module):

        for (name, module), (_, fp_module) in zip(qt.named_children(), fp.named_children()):
            if isinstance(module, QuantModule):
                print('Reconstruction for layer {}'.format(name))
                layer_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)

            elif isinstance(module, QuantPyramidFusion): # since QuantPyramidFusion inherits from QuantResNetBEVBackbone, put it first here.
                print('Reconstruction for pyramid fusion block {}'.format(name))
                pyramid_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)

            elif isinstance(module, (QuantResNetBEVBackbone, QuantDownsampleConv, QuantBaseBEVBackbone, QuantNaiveCompressor)):
                print('Reconstruction for block {}'.format(name))
                block_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)

            elif isinstance(module, QuantPFNLayer):
                print('Reconstruction for PointPillar PFN {}'.format(name))
                encoder_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)

            elif isinstance(module, QuantCamEncode_Resnet101):
                print('Reconstruction for LSS ResNet {}'.format(name))
                lss_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)

            elif isinstance(module, QuantVoxelBackBone8x):
                print('Reconstruction for SECOND {}'.format(name))
                second_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)

            # elif isinstance(module, QuantV2XViTFusion):
            #     print('Reconstruction for V2X-ViT {}'.format(name))
            #     v2xvit_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)
            else:
                recon_model(module, fp_module)
    
    # Start calibration
    recon_model(qt_model, fp_model)
    qt_model.set_quant_state(weight_quant=True, act_quant=True)
    cali_time_end = time.time()
    print('Quantization is done!')
    qt_model.eval()
    print(f"Calibration time: {cali_time_end - cali_time_start:.2f}s")
    

    '''----------------------Inference with quantized model----------------------'''
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat_short = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat_middle = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat_long = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    
    infer_info = opt.fusion_method + opt.note


    for i, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        qt_model,
                                                        opencood_test_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        qt_model,
                                                        opencood_test_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                qt_model,
                                                                opencood_test_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                qt_model,
                                                                opencood_test_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                qt_model,
                                                                opencood_test_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                qt_model,
                                                                opencood_test_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']

            for iou_threshold in [0.3, 0.5, 0.7]:
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        iou_threshold)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat_short,
                                        iou_threshold, 
                                        left_range=0,
                                        right_range=30)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat_middle,
                                        iou_threshold, 
                                        left_range=30,
                                        right_range=50)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat_long,
                                        iou_threshold,
                                        left_range=50,
                                        right_range=100)
            
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_test_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
                
                # vis_feat_save_path = os.path.join(opt.model_dir, f'feat_vis_{infer_info}')
                # vis_utils.visualize_feature_distribution(infer_result, vis_feat_save_path, i)

        torch.cuda.empty_cache()

    eval_utils.eval_final_results(result_stat_short,
                                  opt.model_dir, infer_info="short")
    eval_utils.eval_final_results(result_stat_middle,
                                  opt.model_dir, infer_info="middle")
    eval_utils.eval_final_results(result_stat_long,
                                  opt.model_dir, infer_info="long")
    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)
    
    print(f"Calibration time: {cali_time_end - cali_time_start:.2f}s")

if __name__ == '__main__':
    main()
