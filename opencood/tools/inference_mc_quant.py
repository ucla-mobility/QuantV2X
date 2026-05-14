import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import random
import time
import copy

from tqdm import tqdm
import opencood
from opencood.quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
)

from torch.utils.data import DataLoader
import open3d as o3d
import opencood.hypes_yaml.yaml_utils as yaml_utils
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
from opencood.tools import inference_utils_mc, train_utils

from opencood.data_utils.datasets import build_dataset
from icecream import ic

from opencood.tools.inference import Subset
from opencood.utils import eval_utils_mc
from opencood.visualization import vis_utils_mc, simple_vis


def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _is_agent_modality(name):
    return isinstance(name, str) and name.startswith("m") and name[1:].isdigit() and name != "m0"


def _modality_sort_key(name):
    return int(name[1:]) if _is_agent_modality(name) else 10**9


def _infer_calibration_modalities(hypes):
    """
    Infer which real CAV modality branches are active from the config.
    Model args are the safest source because they describe branches that will
    actually be instantiated (for STAMP final: m1/m2/m3/m4; for pair models:
    m1/m2, m1/m3, etc.).
    """
    model_args = hypes.get("model", {}).get("args", {})
    modalities = sorted(
        [name for name in model_args.keys() if _is_agent_modality(name)],
        key=_modality_sort_key,
    )
    if modalities:
        return modalities

    mapping_dict = hypes.get("heter", {}).get("mapping_dict", {})
    modalities = sorted(
        [name for name in set(mapping_dict.values()) if _is_agent_modality(name)],
        key=_modality_sort_key,
    )
    return modalities


def _infer_calibration_cav_count(hypes, modalities):
    use_cav = hypes.get("use_cav", None)
    if isinstance(use_cav, int):
        return use_cav
    if isinstance(use_cav, str):
        try:
            return int(use_cav)
        except ValueError:
            pass
    return len(modalities) if modalities else None


def _build_calibration_hypes(hypes):
    cali_hypes = copy.deepcopy(hypes)
    quant_cfg = cali_hypes.get("quant", {})
    cali_cfg = quant_cfg.get("calibration", {})

    fusion_cfg = cali_hypes.get("fusion", {})
    fusion_core = fusion_cfg.get("core_method")
    if cali_cfg.get("fusion_core_method"):
        fusion_cfg["core_method"] = cali_cfg["fusion_core_method"]
    elif cali_cfg.get("use_non_infer_dataset", False) and isinstance(fusion_core, str) and fusion_core.endswith("infer"):
        fusion_cfg["core_method"] = fusion_core[:-5]

    if "dataset_mode" in cali_cfg:
        cali_hypes["dataset_mode"] = cali_cfg["dataset_mode"]
    if "comm_range" in cali_cfg:
        cali_hypes["comm_range"] = cali_cfg["comm_range"]
    if "use_cav" in cali_cfg:
        cali_hypes["use_cav"] = cali_cfg["use_cav"]

    heter_cfg = cali_hypes.get("heter", {})
    if "assignment_path" in cali_cfg:
        heter_cfg["assignment_path"] = cali_cfg["assignment_path"]
    if "mapping_dict" in cali_cfg:
        heter_cfg["mapping_dict"] = cali_cfg["mapping_dict"]

    return cali_hypes


def _default_skip_quant_module_names(model):
    skip_names = []
    if hasattr(model, "gencomm"):
        skip_names.append("gencomm")
    if hasattr(model, "enhancer"):
        skip_names.append("enhancer")
    for modality_name in getattr(model, "modality_name_list", []):
        extractor_name = f"message_extractor_{modality_name}"
        if hasattr(model, extractor_name):
            skip_names.append(extractor_name)
    return skip_names


def _merge_unique_names(*name_lists):
    merged = []
    for names in name_lists:
        for name in names or []:
            if not isinstance(name, str) or not name or name in merged:
                continue
            merged.append(name)
    return merged


def _get_agent_modalities(batch):
    if batch is None or "ego" not in batch:
        return []
    modalities = batch["ego"].get("agent_modality_list", [])
    if isinstance(modalities, tuple):
        modalities = list(modalities)
    return modalities


def _summarize_modalities(samples):
    summary = {}
    for sample in samples:
        key = tuple(sample.get("agent_modality_list", []))
        summary[key] = summary.get(key, 0) + 1
    return ", ".join([f"{list(k)} x{v}" for k, v in sorted(summary.items())])


def _prepare_runtime_activation_quantizers(model):
    reset_count = 0
    for module in model.modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)) and hasattr(module, "act_quantizer"):
            module.act_quantizer.set_inited(False)
            reset_count += 1
    return reset_count


def get_train_samples(train_loader, num_batches=128, hypes=None):
    if num_batches <= 0:
        print("Calibration target disabled: num_cali_batches <= 0")
        return []

    train_data = []
    same_count_fallback = []
    hypes = hypes or {}

    target_modalities = _infer_calibration_modalities(hypes)
    target_cav_count = _infer_calibration_cav_count(hypes, target_modalities)
    required_modalities = set(target_modalities)

    print(
        "Calibration target: "
        f"cav_count={target_cav_count}, modalities={target_modalities or 'any'}"
    )

    for batch in train_loader:
        modalities = _get_agent_modalities(batch)
        if len(modalities) == 0:
            continue

        # Keep CAV count consistent; quant caches can have shape mismatches
        # when samples with different agent counts are mixed.
        if target_cav_count is not None and len(modalities) != target_cav_count:
            continue

        if required_modalities and not required_modalities.issubset(set(modalities)):
            same_count_fallback.append(batch["ego"])
            continue

        train_data.append(batch["ego"])
        if len(train_data) >= num_batches:
            break

    if len(train_data) == 0 and len(same_count_fallback) > 0:
        print(
            "[WARNING] No calibration sample contained all required modalities; "
            "falling back to same-CAV-count samples."
        )
        train_data = same_count_fallback[:num_batches]

    if len(train_data) < num_batches:
        print(
            f"[WARNING] Requested {num_batches} calibration batches but found "
            f"{len(train_data)} matching batches."
        )

    print(f"Selected {len(train_data)} calibration batches.")
    if train_data:
        print("Calibration modality summary:", _summarize_modalities(train_data))
    return train_data

def test_parser():
    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # opencood argument
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
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result' 
                             'in npy_test file')
    parser.add_argument('--dataset_mode', type=str, default="")
    parser.add_argument('--epoch', default=None,
                        help="epoch number to load model")
    parser.add_argument('--seed', default=42, type=int, 
                        help='random seed for results reproduction')

    # quantization parameters
    parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', default=True, help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=8, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_cali_batches', default=128, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=5000, type=int, help='number of calibration steps')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--keep_cpu', action='store_true', help='keep the calibration data on cpu')

    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')

    # activation calibration parameters
    parser.add_argument('--lr', default=4e-5, type=float, help='learning rate for LSQ')

    parser.add_argument('--init_wmode', default='minmax', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for weight')
    parser.add_argument('--init_amode', default='minmax', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for activation')

    parser.add_argument('--prob', default=0.5, type=float)
    parser.add_argument('--input_prob', default=0.5, type=float)
    parser.add_argument('--lamb_r', default=0.1, type=float, help='hyper-parameter for regularization')
    parser.add_argument('--T', default=4.0, type=float, help='temperature coefficient for KL divergence')
    parser.add_argument('--bn_lr', default=1e-3, type=float, help='learning rate for DC')
    parser.add_argument('--lamb_c', default=0.02, type=float, help='hyper-parameter for DC')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    # Loading yaml
    opt = test_parser()
    seed_all(opt.seed) # IMPORTANT SEED!
    assert opt.fusion_method in ['late', 'early', 'intermediate', "nofusion"]
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'
    hypes = yaml_utils.load_yaml(None, opt)
    hypes['validate_dir'] = hypes['test_dir'] # for inference pipeline

    # Specify dataset mode for v2xreal
    if opt.dataset_mode:
        hypes['dataset_mode'] = opt.dataset_mode
    print(hypes['dataset_mode'])

    cali_hypes = _build_calibration_hypes(hypes)

    config_modalities = _infer_calibration_modalities(hypes)
    config_cav_count = _infer_calibration_cav_count(hypes, config_modalities)
    if config_cav_count is not None and 'use_cav' not in hypes:
        # Heter infer datasets use this to decide how many CAVs to keep.
        # Non-infer datasets ignore it, but calibration filtering below uses
        # the same inferred target.
        hypes['use_cav'] = config_cav_count

    cali_modalities = _infer_calibration_modalities(cali_hypes)
    cali_cav_count = _infer_calibration_cav_count(cali_hypes, cali_modalities)
    if cali_cav_count is not None and 'use_cav' not in cali_hypes:
        cali_hypes['use_cav'] = cali_cav_count

    # Building data loaders
    print('Dataset Building')
    print('Calibration fusion core:', cali_hypes['fusion']['core_method'])
    opencood_train_dataset = build_dataset(cali_hypes, visualize=False, train=True, calibrate=True) # if calibrate, ignore comm range
    opencood_test_dataset = build_dataset(hypes, visualize=True, train=False, calibrate=False)
    # opencood_test_subset = Subset(opencood_test_dataset, range(650,651))
    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=1,
                              num_workers=16,
                              collate_fn=opencood_train_dataset.collate_batch_test,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    test_loader = DataLoader(opencood_test_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_test_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Loading model
    print('Creating Model')
    trained_model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, trained_model = train_utils.load_saved_model(saved_path, trained_model)
    print(f"resume from {resume_epoch} epoch.")
    
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

    quant_cfg = hypes.get("quant", {})
    use_default_skip_quant_modules = quant_cfg.get("use_default_skip_quant_modules", True)
    default_skip_quant_module_names = (
        _default_skip_quant_module_names(trained_model)
        if use_default_skip_quant_modules
        else []
    )
    skip_quant_module_names = _merge_unique_names(
        default_skip_quant_module_names,
        quant_cfg.get("skip_quant_module_names", []),
    )
    if skip_quant_module_names:
        print("Keeping modules in FP during PTQ:", skip_quant_module_names)
    
    fp_model = QuantModel(model=fp_model,
                          weight_quant_params=wq_params,
                          act_quant_params=aq_params,
                          is_fusing=False,
                          skip_quant_module_names=skip_quant_module_names)
    fp_model.cuda()
    fp_model.eval()
    fp_model.set_quant_state(False, False)
    
    qt_model = QuantModel(model=trained_model,
                          weight_quant_params=wq_params,
                          act_quant_params=aq_params,
                          skip_quant_module_names=skip_quant_module_names)
    qt_model.cuda()
    qt_model.eval()

    print(f"FP model device: {next(fp_model.parameters()).device}")
    print(f"QT model device: {next(qt_model.parameters()).device}")

    # if not opt.disable_8bit_head_stem:
    #     print('Setting the first and the last layer to 8-bit')
    #     qt_model.set_first_last_layer_to_8bit()
    print('the fp model is below!')
    ic(fp_model)


    if quant_cfg.get("disable_output_head_quantization", True):
        qt_model.disable_network_output_quantization()
    print('the quantized model is below!')
    ic(qt_model)

    device = next(qt_model.parameters()).device

    '''init weight quantizer'''
    set_weight_quantize_params(qt_model)

    run_calibration = opt.num_cali_batches > 0
    kwargs = None
    if run_calibration:
        cali_data = get_train_samples(train_loader, num_batches=opt.num_cali_batches, hypes=cali_hypes)
        if len(cali_data) == 0:
            raise RuntimeError(
                "No calibration data matched the configured CAV/modalities. "
                "Check heter.mapping_dict/model.args or lower the requested CAV count."
            )
        # Kwargs for weight rounding calibration
        kwargs = dict(cali_data=cali_data, iters=opt.iters_w, weight=opt.weight,
                    b_range=(opt.b_start, opt.b_end), warmup=opt.warmup, opt_mode='mse',
                    lr=opt.lr, input_prob=opt.input_prob, keep_gpu=not opt.keep_cpu,
                    lamb_r=opt.lamb_r, T=opt.T, bn_lr=opt.bn_lr, lamb_c=opt.lamb_c)
    else:
        reset_count = _prepare_runtime_activation_quantizers(qt_model)
        print(
            "Skipping PTQ calibration/reconstruction because num_cali_batches <= 0. "
            f"Weights are still quantized, and {reset_count} activation quantizers "
            "will initialize on the first inference batches."
        )
        
    def recon_model(qt: nn.Module, fp: nn.Module):

        for (name, module), (_, fp_module) in zip(qt.named_children(), fp.named_children()):
            if isinstance(module, QuantModule):
                print('Reconstruction for layer {}'.format(name))
                try:
                    layer_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)
                except RuntimeError as exc:
                    if "No valid calibration" in str(exc):
                        print(f"Skip layer {name}: {exc}")
                    else:
                        raise

            elif isinstance(module, QuantPyramidFusion): # since QuantPyramidFusion inherits from QuantResNetBEVBackbone, put it first here.
                print('Reconstruction for pyramid fusion block {}'.format(name))
                try:
                    pyramid_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)
                except RuntimeError as exc:
                    if "No valid calibration" in str(exc):
                        print(f"Skip pyramid block {name}: {exc}")
                    else:
                        raise

            elif isinstance(module, (QuantResNetBEVBackbone, QuantDownsampleConv, QuantBaseBEVBackbone, QuantNaiveCompressor)):
                print('Reconstruction for block {}'.format(name))
                try:
                    block_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)
                except RuntimeError as exc:
                    if "No valid calibration" in str(exc):
                        print(f"Skip block {name}: {exc}")
                    else:
                        raise

            elif isinstance(module, QuantPFNLayer):
                print('Reconstruction for PointPillar PFN {}'.format(name))
                try:
                    encoder_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)
                except RuntimeError as exc:
                    if "No valid calibration" in str(exc):
                        print(f"Skip PFN {name}: {exc}")
                    else:
                        raise

            elif isinstance(module, QuantCamEncode_Resnet101):
                print('Reconstruction for LSS ResNet {}'.format(name))
                try:
                    lss_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)
                except RuntimeError as exc:
                    if "No valid calibration" in str(exc):
                        print(f"Skip LSS ResNet {name}: {exc}")
                    else:
                        raise

            elif isinstance(module, QuantVoxelBackBone8x):
                print('Reconstruction for SECOND {}'.format(name))
                try:
                    second_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)
                except RuntimeError as exc:
                    if "No valid calibration" in str(exc):
                        print(f"Skip SECOND {name}: {exc}")
                    else:
                        raise

            elif isinstance(module, QuantV2XViTFusion):
                print('Reconstruction for V2X-ViT {}'.format(name))
                try:
                    v2xvit_reconstruction(qt_model, fp_model, module, fp_module, **kwargs)
                except RuntimeError as exc:
                    if "No valid calibration" in str(exc):
                        print(f"Skip V2X-ViT {name}: {exc}")
                    else:
                        raise
            
            else:
                recon_model(module, fp_module)
    
    # Start calibration
    if run_calibration:
        recon_model(qt_model, fp_model)
    qt_model.set_quant_state(weight_quant=True, act_quant=True)
    cali_time_end = time.time()
    print('Quantization is done!')

    '''----------------------Inference with quantized model----------------------'''
    # ic(qt_model)
    qt_model.eval()

    print(qt_model.get_memory_footprint())

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

    with tqdm(total=len(test_loader)) as pbar:
        for i, batch_data in enumerate(test_loader):
            # print(i)
            with torch.no_grad():
                # time starts here
                batch_data = train_utils.to_device(batch_data, device)
                start_time = time.time()
                if opt.fusion_method == 'late':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils_mc.inference_late_fusion(batch_data,
                                                            qt_model,
                                                            opencood_test_dataset)
                elif opt.fusion_method == 'nofusion':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils_mc.inference_nofusion(batch_data,
                                                            qt_model,
                                                            opencood_test_dataset)
                elif opt.fusion_method == 'early':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils_mc.inference_early_fusion(batch_data,
                                                            qt_model,
                                                            opencood_test_dataset)
                elif opt.fusion_method == 'intermediate':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils_mc.inference_intermediate_fusion(batch_data,
                                                                    qt_model,
                                                                    opencood_test_dataset)
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
                        opencood_test_dataset.visualize_result(pred_box_tensor,
                                                        gt_box_tensor,
                                                        batch_data['ego'][
                                                            'origin_lidar'],
                                                        None,
                                                        opt.show_vis,
                                                        vis_save_path,
                                                        dataset=opencood_test_dataset)

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
    print(f"Calibration minutes: {(cali_time_end - cali_time_start) / 60:.2f}")
