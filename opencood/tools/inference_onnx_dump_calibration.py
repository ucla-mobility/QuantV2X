# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
from datetime import datetime
import os
import random
import time
from typing import OrderedDict
import importlib
import torch
import torch.onnx
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, onnx_export_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
import json
import pickle
from opencood.models import onnx_wrapper
import copy
import onnxruntime as ort
import onnx
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as _np
import torch as _torch

# ==== TensorRT ====
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
# ==================

FIXED_N = 20343  # P99 clamp for voxel rows

# ------------------------ TRT Engine Runner ------------------------
class TRTInference:
    """TensorRT inference wrapper for .plan files (TensorRT 8.5+)."""

    def __init__(self, engine_path, verbose=False):
        self.logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        print(f"Loading TensorRT engine from: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_names, self.output_names = [], []
        self.input_shapes, self.output_shapes = {}, {}
        self.tensor_dtypes = {}
        self.host_buffers, self.device_buffers, self.buffer_sizes = {}, {}, {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)
            self.tensor_dtypes[name] = dtype
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes[name] = shape
                print(f"  Input: {name}, shape={shape}, dtype={dtype}")
            else:
                self.output_names.append(name)
                self.output_shapes[name] = shape
                print(f"  Output: {name}, shape={shape}, dtype={dtype}")
                if all(d > 0 for d in shape):
                    size = int(np.prod(shape))
                    host_buffer = np.empty(size, dtype=self._trt_dtype_to_np(dtype))
                    nbytes = host_buffer.nbytes
                    self.host_buffers[name] = host_buffer
                    self.device_buffers[name] = cuda.mem_alloc(nbytes)
                    self.buffer_sizes[name] = nbytes

        print(f"Engine loaded successfully with {len(self.input_names)} inputs and {len(self.output_names)} outputs")

    def _trt_dtype_to_np(self, trt_dtype):
        return {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF:  np.float16,
            trt.DataType.INT8:  np.int8,
            trt.DataType.INT32: np.int32,
            getattr(trt.DataType, "INT64", None): np.int64 if hasattr(trt.DataType, "INT64") else np.int32,
            trt.DataType.BOOL:  np.bool_,
        }.get(trt_dtype, np.float32)

    def infer(self, input_dict):
        # Set inputs
        for name in self.input_names:
            if name not in input_dict:
                if name == 'agent_modality_list':
                    continue
                raise ValueError(f"Missing input '{name}'. Provided: {list(input_dict.keys())}")
            arr = input_dict[name]
            want = self._trt_dtype_to_np(self.tensor_dtypes[name])
            if arr.dtype != want:
                arr = arr.astype(want, copy=False)
            arr = np.ascontiguousarray(arr)
            if any(d == -1 for d in self.input_shapes[name]):
                self.context.set_input_shape(name, arr.shape)
            nbytes = arr.nbytes
            if name not in self.device_buffers or nbytes > self.buffer_sizes.get(name, 0):
                self.device_buffers[name] = cuda.mem_alloc(nbytes)
                self.buffer_sizes[name] = nbytes
            cuda.memcpy_htod_async(self.device_buffers[name], arr, self.stream)
            self.context.set_tensor_address(name, int(self.device_buffers[name]))

        # Prepare outputs
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                continue
            size = int(np.prod(shape))
            np_dtype = self._trt_dtype_to_np(self.tensor_dtypes[name])
            nbytes = size * np.dtype(np_dtype).itemsize
            if name not in self.host_buffers or size > self.host_buffers[name].size:
                self.host_buffers[name] = np.empty(size, dtype=np_dtype)
            if name not in self.device_buffers or nbytes > self.buffer_sizes.get(name, 0):
                self.device_buffers[name] = cuda.mem_alloc(nbytes)
                self.buffer_sizes[name] = nbytes
            self.context.set_tensor_address(name, int(self.device_buffers[name]))

        # Execute
        if hasattr(self.context, "execute_async_v3"):
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=None, stream_handle=self.stream.handle)
        for name in self.output_names:
            if name in self.host_buffers and name in self.device_buffers:
                cuda.memcpy_dtoh_async(self.host_buffers[name], self.device_buffers[name], self.stream)
        self.stream.synchronize()

        # Collect outputs (reshape by runtime shapes)
        outs = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                continue
            host = self.host_buffers[name][:int(np.prod(shape))].reshape(shape).copy()
            outs.append(host)
        return outs

    def benchmark(self, input_dict, warmup=10, iterations=100):
        for _ in range(warmup):
            _ = self.infer(input_dict)
        start = time.time()
        for _ in range(iterations):
            _ = self.infer(input_dict)
        dt = (start := time.time()) - start  # dummy to keep style linters happy
        total = time.time() - start
        avg = total / iterations
        print(f"TensorRT Benchmark Results:\n  Average inference time: {avg:.4f} s\n  FPS: {1.0/avg:.2f}")
        return avg
# -------------------------------------------------------------------

def _pad_or_truncate(vf, vc, vnp, N=FIXED_N):
    Ncur = vf.shape[0]
    if Ncur == N:
        return vf, vc, vnp
    if Ncur < N:
        pad = N - Ncur
        vf_pad  = _np.zeros((pad,) + vf.shape[1:], dtype=vf.dtype)
        vc_pad  = -_np.ones((pad,) + vc.shape[1:], dtype=vc.dtype)
        vnp_pad = _np.zeros((pad,), dtype=vnp.dtype)
        return _np.concatenate([vf, vf_pad], axis=0), \
               _np.concatenate([vc, vc_pad], axis=0), \
               _np.concatenate([vnp, vnp_pad], axis=0)
    idx = _np.argpartition(-vnp, N-1)[:N]
    return vf[idx], vc[idx], vnp[idx]

def _to_numpy(x):
    if isinstance(x, _torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, _np.ndarray):
        return x
    return _np.asarray(x)

def _enforce_fixed_N_on_flat_inputs(names, values, N=FIXED_N):
    name_to_arr = {k: (v.detach().cpu().numpy() if isinstance(v, _torch.Tensor) else _np.asarray(v))
                   for k, v in zip(names, values)}
    if "voxel_coords" in name_to_arr:
        name_to_arr["voxel_coords"] = name_to_arr["voxel_coords"].astype(_np.int32,  copy=False)
    if "voxel_num_points" in name_to_arr:
        name_to_arr["voxel_num_points"] = name_to_arr["voxel_num_points"].astype(_np.int32, copy=False)
    if "record_len" in name_to_arr:
        name_to_arr["record_len"] = name_to_arr["record_len"].astype(_np.int64,  copy=False)
    if all(k in name_to_arr for k in ["voxel_features","voxel_coords","voxel_num_points"]):
        vf, vc, vnp = name_to_arr["voxel_features"], name_to_arr["voxel_coords"], name_to_arr["voxel_num_points"]
        vf2, vc2, vnp2 = _pad_or_truncate(vf, vc, vnp, N=N)
        name_to_arr["voxel_features"]   = vf2
        name_to_arr["voxel_coords"]     = vc2
        name_to_arr["voxel_num_points"] = vnp2
    out_vals = []
    for k in names:
        arr = name_to_arr[k]
        if arr.dtype == _np.float32 or arr.dtype == _np.float64:
            out_vals.append(_torch.from_numpy(arr.astype(_np.float32, copy=False)))
        elif arr.dtype == _np.int64:
            out_vals.append(_torch.from_numpy(arr))
        elif arr.dtype in (_np.int32, _np.int16, _np.int8):
            out_vals.append(_torch.from_numpy(arr.astype(_np.int32, copy=False)))
        else:
            out_vals.append(_torch.from_numpy(arr))
    return out_vals, name_to_arr

# ---- Fixed: Pick correct heads from TRT outputs using concatenated tensor ----
def _select_trt_head_tensors_fixed(outputs, output_names):
    """
    Extract (cls_np, reg_np, dir_np) from TRT outputs.
    
    Based on your engine outputs:
    - Use tensor '1884' (shape 1,20,256,256) which contains all heads concatenated
    - Slice it properly: cls=[0:2], reg=[2:16], dir=[16:20]
    
    Returns NCHW format tensors.
    """
    name_to_out = {n: o for n, o in zip(output_names, outputs)}
    
    # Find the concatenated output tensor (20 channels)
    concat_tensor = None
    for name, tensor in name_to_out.items():
        if tensor.ndim == 4:
            # Check for 20-channel tensor at 256x256
            if tensor.shape == (1, 20, 256, 256):
                concat_tensor = tensor
                break
            # Handle NHWC format if needed
            elif tensor.shape == (1, 256, 256, 20):
                concat_tensor = np.transpose(tensor, (0, 3, 1, 2))  # NHWC -> NCHW
                break
    
    if concat_tensor is None:
        # Fallback: try to find by name
        if '1884' in name_to_out:
            concat_tensor = name_to_out['1884']
            if concat_tensor.shape == (1, 256, 256, 20):
                concat_tensor = np.transpose(concat_tensor, (0, 3, 1, 2))
    
    if concat_tensor is None:
        raise RuntimeError(f"Could not find concatenated output tensor (1,20,256,256). Available shapes: {[(n, o.shape) for n, o in name_to_out.items()]}")
    
    # Slice the concatenated tensor
    cls_np = concat_tensor[:, 0:2, :, :]    # channels 0-1 (2 channels for classification)
    reg_np = concat_tensor[:, 2:16, :, :]   # channels 2-15 (14 channels for regression)
    dir_np = concat_tensor[:, 16:20, :, :]  # channels 16-19 (4 channels for direction)
    
    # Verify shapes
    assert cls_np.shape == (1, 2, 256, 256), f"cls shape mismatch: {cls_np.shape}"
    assert reg_np.shape == (1, 14, 256, 256), f"reg shape mismatch: {reg_np.shape}"
    assert dir_np.shape == (1, 4, 256, 256), f"dir shape mismatch: {dir_np.shape}"
    
    return cls_np, reg_np, dir_np

def inference_intermediate_fusion_trt_fixed(batch_dict, output_dict, dataset):
    """
    Fixed version that properly calls post_process with the correct structure.
    
    The post_processor expects:
    - data_dict with CAV keys (e.g., {'ego': {...}})
    - output_dict with matching CAV keys containing the predictions
    """
    import torch
    
    # Ensure outputs are on GPU
    for k in ['cls_preds', 'reg_preds', 'dir_preds']:
        if k in output_dict['ego'] and not output_dict['ego'][k].is_cuda:
            output_dict['ego'][k] = output_dict['ego'][k].cuda()
    
    # Call post_process with the FULL dict structure (not just 'ego' portion)
    # This is the critical fix - pass the full dicts with CAV structure
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
        batch_dict,  # Full batch dict with {'ego': {...}}
        output_dict  # Full output dict with {'ego': {'cls_preds': ..., 'reg_preds': ..., 'dir_preds': ...}}
    )
    
    return {
        'pred_box_tensor': pred_box_tensor,
        'pred_score': pred_score,
        'gt_box_tensor': gt_box_tensor
    }

# ---- Local TRT adapter using dataset wrapper ----
def inference_intermediate_fusion_trt(batch_dict, output_dict, dataset):
    """
    Use the same dataset wrapper as the PyTorch path:
      dataset.post_process(batch_dict['ego'], output_dict['ego'])
    """
    import torch
    for k in ['cls_preds', 'reg_preds', 'dir_preds']:
        if k in output_dict['ego'] and not output_dict['ego'][k].is_cuda:
            output_dict['ego'][k] = output_dict['ego'][k].cuda()

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
        batch_dict['ego'], output_dict['ego']
    )
    return {
        'pred_box_tensor': pred_box_tensor,
        'pred_score': pred_score,
        'gt_box_tensor': gt_box_tensor
    }

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def test_parser():
    parser = argparse.ArgumentParser(description="ONNX/TRT export + inference")
    parser.add_argument('--model_dir', type=str, required=True, help='Continued training path')
    parser.add_argument('--fusion_method', type=str, default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40, help='vis interval')
    parser.add_argument('--save_npy', action='store_true', help='save pred/gt as npy')
    parser.add_argument('--range', type=str, default="102.4,102.4",
                        help="detection range [-X,+X,-Y,+Y]")
    parser.add_argument('--no_score', action='store_true', help="suppress score print")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--note', default="", type=str)
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--mode', choices=['infer', 'create'], default='create',
                        help='create: export ONNX; infer: run ONNX/TRT')
    parser.add_argument('--dump_npz', action='store_true',
                        help='dump up to 500 ONNX input snapshots to <model_dir>/calibration_npz')
    # TensorRT opts
    parser.add_argument('--trt_engine', type=str, default=None, help='Path to .plan engine for inference')
    parser.add_argument('--use_trt', action='store_true', help='Use TensorRT for inference')
    return parser.parse_args()

def main():
    opt = test_parser()
    seed_all(opt.seed)

    dump_npz = opt.dump_npz
    calib_npz_dir = os.path.join(opt.model_dir, "calibration_npz")
    dumped_npz = 0
    MAX_NPZ = 500

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single']

    hypes = yaml_utils.load_yaml(None, opt)

    if 'heter' in hypes:
        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"
        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2],
                         x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]
        hypes = update_dict(hypes, {"cav_lidar_range": new_cav_range,
                                    "lidar_range": new_cav_range,
                                    "gt_range": new_cav_range})
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']

    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir'] or "V2XREAL" in hypes['test_dir']) else False
    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes:
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"

    model = onnx_wrapper.OnnxWrapper(model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=opt.visualize, train=False, calibrate=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

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
    data_loader_len = len(data_loader)
    print(f"Length of data loader: {data_loader_len}")
    
    # Init TRT engine if requested
    trt_engine = None
    if opt.use_trt and opt.trt_engine:
        print(f"Initializing TensorRT engine from: {opt.trt_engine}")
        trt_engine = TRTInference(opt.trt_engine, verbose=True)
    exported_onnx_once = False
    for i, batch_data in enumerate(data_loader):
        if(i>100):
            break
        print(f"handling {infer_info}_{i}: " +
            ("generating onnx model" if opt.mode == 'create' else "inference onnx model"))
        if batch_data is None:
            print(f"batch_data is None at index {i}")
            continue

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == 'late':
                output_dict = OrderedDict()
                cav_content = onnx_export_utils.model_input_clean_late(batch_data['ego'])
                cav_content_onnx = copy.deepcopy(cav_content)
                cav_content['onnx_export'] = True

                onnx_input_key = onnx_export_utils.get_flat_key_list_from_dict(cav_content_onnx)
                onnx_input = onnx_export_utils.get_flat_value_list_from_dict(cav_content_onnx)

                if dump_npz and dumped_npz < MAX_NPZ:
                    os.makedirs(calib_npz_dir, exist_ok=True)
                    payload = {}
                    for k, v in zip(onnx_input_key, onnx_input):
                        payload[k] = _to_numpy(v)
                    np.savez(os.path.join(calib_npz_dir, f"calib_{dumped_npz:05d}.npz"), **payload)
                    dumped_npz += 1

                onnx_output_name = ['cls_preds', 'reg_preds', 'dir_preds']
                save_onnx_path = os.path.join(opt.model_dir, "onnx_model_export")
                if not os.path.exists(save_onnx_path):
                    os.makedirs(save_onnx_path)

                if opt.mode == 'create':
                    if not exported_onnx_once:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        onnx_file_name = os.path.join(save_onnx_path, f"model_late_val_{i}_ego_{timestamp}.onnx")
                        output_dict['ego'] = model.forward_normal(cav_content)
                        torch.onnx.export(
                            model, onnx_input, onnx_file_name,
                            input_names=onnx_input_key, output_names=onnx_output_name,
                            opset_version=15, do_constant_folding=True
                        )
                        exported_onnx_once = True
                    continue
                else:
                    # fall back to original late-fusion path (PyTorch)
                    from opencood.tools import inference_utils
                    infer_result = inference_utils.inference_late_fusion(batch_data, model, opencood_dataset)

            elif opt.fusion_method == 'early':
                from opencood.tools import inference_utils
                infer_result = inference_utils.inference_early_fusion(batch_data, model, opencood_dataset)

            elif opt.fusion_method == 'intermediate':
                output_dict = OrderedDict()
                cav_content = onnx_export_utils.model_input_clean(batch_data['ego'])
                cav_content_onnx = copy.deepcopy(cav_content)
                cav_content['onnx_export'] = True

                onnx_input_key = onnx_export_utils.get_flat_key_list_from_dict(cav_content_onnx)
                onnx_input = onnx_export_utils.get_flat_value_list_from_dict(cav_content_onnx)
                #onnx_input, name_to_arr_np = _enforce_fixed_N_on_flat_inputs(onnx_input_key, onnx_input, N=FIXED_N)
                #print(f"[FIXED_N] vf={name_to_arr_np['voxel_features'].shape}, vc={name_to_arr_np['voxel_coords'].shape}, vnp={name_to_arr_np['voxel_num_points'].shape}")
                name_to_arr_np = {k: _to_numpy(v) for k, v in zip(onnx_input_key, onnx_input)}

                if dump_npz and dumped_npz < MAX_NPZ:
                    os.makedirs(calib_npz_dir, exist_ok=True)
                    payload = {k: name_to_arr_np[k] for k in onnx_input_key}
                    np.savez(os.path.join(calib_npz_dir, f"calib_{dumped_npz:05d}.npz"), **payload)
                    dumped_npz += 1

                onnx_output_name = ['cls_preds', 'reg_preds', 'dir_preds']
                save_onnx_path = os.path.join(opt.model_dir, "onnx_model_export")
                if not os.path.exists(save_onnx_path):
                    os.makedirs(save_onnx_path)

                use_trt_inference = (opt.use_trt and (trt_engine is not None) and (opt.mode != 'create'))

                if opt.mode == 'create':
                    if not exported_onnx_once:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        onnx_file_name = os.path.join(
                            save_onnx_path,
                            f"model_intermediate_val_{i}_ego_{timestamp}.onnx"
                        )
                        output_dict['ego'] = model.forward_normal(cav_content)

                        dyn_axes = {}
                        def add(name, axes):
                            if name in onnx_input_key:
                                dyn_axes[name] = axes
                        add('record_len', {0: 'B'})
                        add('pairwise_t_matrix', {0: 'B', 1: 'A', 2: 'A'})
                        add('agent_modality_list', {0: 'A'})
                        out_dyn = {"cls_preds": {0: "B"}, "reg_preds": {0: "B"}, "dir_preds": {0: "B"}}
                        dyn_axes = {**dyn_axes, **out_dyn}

                        dev = next(model.parameters()).device
                        onnx_input_t = [t.to(dev) for t in onnx_input]
                        torch.onnx.export(
                            model, tuple(onnx_input_t), onnx_file_name,
                            input_names=list(onnx_input_key), output_names=list(onnx_output_name),
                            opset_version=20, do_constant_folding=True, dynamic_axes=dyn_axes
                        )
                        exported_onnx_once = True
                    # Still in 'create' mode: we only export once but keep iterating to dump NPZs
                    continue


                elif use_trt_inference:
                    print("Using TensorRT for inference")
                    input_dict = {}
                    for name in onnx_input_key:
                        if name not in name_to_arr_np:
                            continue
                        arr = name_to_arr_np[name]
                        if name == 'voxel_features':
                            arr = arr.astype(np.float32, copy=False)
                        elif name == 'voxel_coords':
                            arr = arr.astype(np.int32, copy=False)
                        elif name == 'voxel_num_points':
                            arr = arr.astype(np.int32, copy=False)
                        elif name == 'record_len':
                            arr = arr.astype(np.int64, copy=False)
                            if arr.ndim == 0:
                                arr = arr.reshape(1)
                        elif name == 'pairwise_t_matrix':
                            arr = arr.astype(np.float32, copy=False)
                        elif name == 'agent_modality_list':
                            arr = arr.astype(np.int64, copy=False)
                        input_dict[name] = arr

                    if i == 0:
                        print("[DEBUG] TRT Input shapes:")
                        for n, a in input_dict.items():
                            print(f"  {n}: {a.shape} dtype={a.dtype}")

                    try:
                        raw_outputs = trt_engine.infer(input_dict)
                        
                        # Use the FIXED head selection function
                        cls_np, reg_np, dir_np = _select_trt_head_tensors_fixed(raw_outputs, trt_engine.output_names)

                        if i == 0:
                            print("[DEBUG] TRT Output shapes (fixed selection):")
                            print(f"  cls_preds: {cls_np.shape}")
                            print(f"  reg_preds: {reg_np.shape}")
                            print(f"  dir_preds: {dir_np.shape}")
                            print(f"  Total raw outputs: {len(raw_outputs)}")
                            
                            # Debug: show all output shapes
                            for idx, (name, out) in enumerate(zip(trt_engine.output_names, raw_outputs)):
                                print(f"  Output[{idx}] {name}: {out.shape}")

                        # Convert to torch for post-processing
                        cls_preds = torch.from_numpy(cls_np).cuda()
                        reg_preds = torch.from_numpy(reg_np).cuda()
                        dir_preds = torch.from_numpy(dir_np).cuda()

                        # Create output_dict with proper structure
                        output_dict = {
                            'ego': {
                                'cls_preds': cls_preds,
                                'reg_preds': reg_preds,
                                'dir_preds': dir_preds,
                            }
                        }
                        
                        # Use the FIXED inference function
                        infer_result = inference_intermediate_fusion_trt_fixed(
                            batch_data,
                            output_dict,
                            opencood_dataset
                        )

                    except Exception as e:
                        print(f"[ERROR] TRT inference failed: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                    if i == 0:
                        print("\n[BENCHMARK] Running TensorRT benchmark...")
                        trt_engine.benchmark(input_dict, warmup=10, iterations=100)

                else:
                    # Fallback to original PyTorch path
                    from opencood.tools import inference_utils
                    infer_result = inference_utils.inference_intermediate_fusion(batch_data, model, opencood_dataset)

            elif opt.fusion_method == 'no':
                from opencood.tools import inference_utils
                infer_result = inference_utils.inference_no_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                from opencood.tools import inference_utils
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'single':
                from opencood.tools import inference_utils
                infer_result = inference_utils.inference_no_fusion(batch_data, model, opencood_dataset, single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate fusion is supported.')

            # ---- Evaluation ----
            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']

            for thr in [0.3, 0.5, 0.7]:
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, thr)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat_short, thr,
                                           left_range=0, right_range=30)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat_middle, thr,
                                           left_range=30, right_range=50)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat_long, thr,
                                           left_range=50, right_range=100)

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                os.makedirs(npy_save_path, exist_ok=True)
                from opencood.tools import inference_utils
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego']['origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                from opencood.tools import inference_utils
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np,
                                     "agent_modality_list": agent_modality_list})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                vis_tag = "trt" if (opt.use_trt and trt_engine is not None and opt.mode != 'create' and opt.fusion_method == 'intermediate') else "onnx"
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{vis_tag}_{infer_info}')
                os.makedirs(vis_save_path_root, exist_ok=True)
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                     batch_data['ego']['origin_lidar'][0],
                                     hypes['postprocess']['gt_range'],
                                     vis_save_path,
                                     method='bev',
                                     left_hand=left_hand)

        torch.cuda.empty_cache()

    _, ap50, ap70 = eval_utils.eval_final_results(result_stat, opt.model_dir, infer_info)
    print(f"[RESULT] AP@0.5: {ap50:.4f} | AP@0.7: {ap70:.4f}")
    eval_utils.eval_final_results(result_stat_short,  opt.model_dir, infer_info="short")
    eval_utils.eval_final_results(result_stat_middle, opt.model_dir, infer_info="middle")
    eval_utils.eval_final_results(result_stat_long,   opt.model_dir, infer_info="long")

if __name__ == '__main__':
    main()
