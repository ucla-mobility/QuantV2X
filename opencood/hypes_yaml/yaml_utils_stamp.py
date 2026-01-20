# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License



import re
import yaml
import os
import math

import numpy as np


def matrix_to_pose(matrix):
    # Ensure the matrix is a numpy array
    matrix = np.array(matrix)
    
    # Extract translation
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    
    # Extract rotation matrix
    R = matrix[:3, :3]
    
    # Calculate Euler angles from rotation matrix
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    
    return np.array([x, y, z, roll, pitch, yaw])

def load_yaml(file, opt=None, config=None):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    opt : argparser
         Argparser.
    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    if config:
        file = config
    else:
        if opt and opt.model_dir:
            file = os.path.join(opt.model_dir, 'config.yaml')

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    
    param = yaml.load(stream, Loader=loader)
    if 'lidar_pose' in param:
        if param['lidar_pose'] is not None and isinstance(param['lidar_pose'], np.ndarray) and param['lidar_pose'].shape == (4, 4):
            param['lidar_pose'] = matrix_to_pose(param['lidar_pose'])
            
    if "yaml_parser" in param:
        if isinstance(param["yaml_parser"], str):
            param = eval(param["yaml_parser"])(param)
        else:
            for yaml_parser in param["yaml_parser"]:
                param = eval(yaml_parser)(param)
    elif "yaml_parsers" in param:
        for modality_name in param["heter"]["modality_setting"]:
            if isinstance(param["yaml_parsers"][modality_name], str):
                yaml_parser = param["yaml_parsers"][modality_name]
                param["heter"]["modality_setting"][modality_name] = eval(yaml_parser)(param["heter"]["modality_setting"][modality_name])
                if yaml_parser == 'load_bev_params':
                    param['model']['args'][modality_name]['encoder_args']['geometry_param'] = \
                        param["heter"]["modality_setting"][modality_name]["preprocess"]["geometry_param"]                
            else:
                for yaml_parser in param["yaml_parsers"][modality_name]:
                    param["heter"]["modality_setting"][modality_name] = eval(yaml_parser)(param["heter"]["modality_setting"][modality_name])
                    if yaml_parser == 'load_bev_params':
                        param['model']['args'][modality_name]['encoder_args']['geometry_param'] = \
                            param["heter"]["modality_setting"][modality_name]["preprocess"]["geometry_param"]                


    # Ensure top-level anchor_args has W/H/D when present (STAMP stage2 uses yaml_parsers only).
    if isinstance(param, dict) and "postprocess" in param and isinstance(param["postprocess"], dict):
        anchor_args = param["postprocess"].get("anchor_args")
        if isinstance(anchor_args, dict) and "W" not in anchor_args:
            try:
                param = load_general_params(param)
            except Exception:
                pass

    return param

def update_yaml(param, opt=None):
    if "yaml_parser" in param:
        if isinstance(param["yaml_parser"], str):
            param = eval(param["yaml_parser"])(param)
        else:
            for yaml_parser in param["yaml_parser"]:
                param = eval(yaml_parser)(param)
    elif "yaml_parsers" in param:
        for modality_name in param["heter"]["modality_setting"]:
            if isinstance(param["yaml_parsers"][modality_name], str):
                yaml_parser = param["yaml_parsers"][modality_name]
                param["heter"]["modality_setting"][modality_name] = eval(yaml_parser)(param["heter"]["modality_setting"][modality_name])
                if yaml_parser == 'load_bev_params':
                    param['model']['args'][modality_name]['encoder_args']['geometry_param'] = \
                        param["heter"]["modality_setting"][modality_name]["preprocess"]["geometry_param"]                
            else:
                for yaml_parser in param["yaml_parsers"][modality_name]:
                    param["heter"]["modality_setting"][modality_name] = eval(yaml_parser)(param["heter"]["modality_setting"][modality_name])
                    if yaml_parser == 'load_bev_params':
                        param['model']['args'][modality_name]['encoder_args']['geometry_param'] = \
                            param["heter"]["modality_setting"][modality_name]["preprocess"]["geometry_param"]                
                

    # Ensure top-level anchor_args has W/H/D when present (STAMP stage2 uses yaml_parsers only).
    if isinstance(param, dict) and "postprocess" in param and isinstance(param["postprocess"], dict):
        anchor_args = param["postprocess"].get("anchor_args")
        if isinstance(anchor_args, dict) and "W" not in anchor_args:
            try:
                param = load_general_params(param)
            except Exception:
                pass

    return param
    


def load_rgb_params(param):
    return param


def load_voxel_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute `anchor_args[W][H][L]`
    """
    anchor_args = param['postprocess']['anchor_args']
    cav_lidar_range = anchor_args['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['postprocess'].update({'anchor_args': anchor_args})

    # sometimes we just want to visualize the data without implementing model
    if 'model' in param:
        param['model']['args']['W'] = anchor_args['W']
        param['model']['args']['H'] = anchor_args['H']
        param['model']['args']['D'] = anchor_args['D']
    
    if 'box_align_pre_calc' in param:
        param['box_align_pre_calc']['stage1_postprocessor_config'].update({'anchor_args': anchor_args})

    return param


def load_point_pillar_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param['model']['args']['point_pillar_scatter']['grid_size'] = grid_size

    anchor_args = param['postprocess']['anchor_args']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw) # W is image width, but along with x axis in lidar coordinate
    anchor_args['H'] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh) # H is image height
    anchor_args['D'] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['postprocess'].update({'anchor_args': anchor_args})

    return param


def load_second_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param['model']['args']['grid_size'] = grid_size

    anchor_args = param['postprocess']['anchor_args']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['postprocess'].update({'anchor_args': anchor_args})

    return param


def load_bev_params(param):
    """
    Load bev related geometry parameters s.t. boundary, resolutions, input
    shape, target shape etc.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute `geometry_param`.

    """
    res = param["preprocess"]["args"]["res"]
    L1, W1, H1, L2, W2, H2 = param["preprocess"]["cav_lidar_range"]
    downsample_rate = param["preprocess"]["args"]["downsample_rate"]

    def f(low, high, r):
        return int((high - low) / r)
    input_shape = (
        int((f(L1, L2, res))),
        int((f(W1, W2, res))),
        int((f(H1, H2, res)) + 1)
    )
    label_shape = (
        int(input_shape[0] / downsample_rate),
        int(input_shape[1] / downsample_rate),
        7
    )
    geometry_param = {
        'L1': L1,
        'L2': L2,
        'W1': W1,
        'W2': W2,
        'H1': H1,
        'H2': H2,
        "downsample_rate": downsample_rate,
        "input_shape": input_shape,
        "label_shape": label_shape,
        "res": res
    }
    param["preprocess"]["geometry_param"] = geometry_param
    param["postprocess"]["geometry_param"] = geometry_param
    if param.get("model", None): # For hetergenous setting, param refers to param[heter][modality_setting][modality_name], therefore model is not in param
        param["model"]["args"]["geometry_param"] = geometry_param
    return param



def save_yaml(data, save_name):
    """
    Save the dictionary into a yaml file.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """

    with open(save_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)



def load_point_pillar_params_stage1(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param['box_align_pre_calc']['stage1_model_config']['point_pillar_scatter']['grid_size'] = grid_size

    anchor_args = param['box_align_pre_calc']['stage1_postprocessor_config']['anchor_args']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw) # W is image width, but along with x axis in lidar coordinate
    anchor_args['H'] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh) # H is image height
    anchor_args['D'] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['box_align_pre_calc']['stage1_postprocessor_config'].update({'anchor_args': anchor_args})

    return param


def load_lift_splat_shoot_params(param):
    """
    Based on the detection range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    
    anchor_args = param['postprocess']['anchor_args']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw) # W is image width, but along with x axis in lidar coordinate
    anchor_args['H'] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh) # H is image height
    anchor_args['D'] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['postprocess'].update({'anchor_args': anchor_args})

    return param


def load_general_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    
    
    voxel_size = None
    try:
        # new version (July 2024)
        voxel_size = param['postprocess']['voxel_size']
    except:
        # old version
        voxel_size = param['preprocess']['args']['voxel_size']
        
    assert voxel_size is not None, "voxel_size is required for general_params"
    
    cav_lidar_range = param['postprocess']['gt_range']
    anchor_args = param['postprocess']['anchor_args']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw) # W is image width, but along with x axis in lidar coordinate
    anchor_args['H'] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh) # H is image height
    anchor_args['D'] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['postprocess'].update({'anchor_args': anchor_args})

    return param


def load_general_params_heter_task(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    postprocesser_param = param['postprocess']
    
    modality_names = param['postprocess'].keys()
    for modality_name in modality_names:
        assert modality_name[0] == 'm' and modality_name[1:].isdigit()
    postprocesser_param = param['postprocess'].values()
        
    for p in postprocesser_param:
    
        cav_lidar_range = p['gt_range']
        voxel_size = p['voxel_size']
        anchor_args = p['anchor_args']

        vw = voxel_size[0]
        vh = voxel_size[1]
        vd = voxel_size[2]

        anchor_args['vw'] = vw
        anchor_args['vh'] = vh
        anchor_args['vd'] = vd

        anchor_args['W'] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw) # W is image width, but along with x axis in lidar coordinate
        anchor_args['H'] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh) # H is image height
        anchor_args['D'] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

        p.update({'anchor_args': anchor_args})

    return param
