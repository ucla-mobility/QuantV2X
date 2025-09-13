# -*- coding: utf-8 -*-
# Author: Zhaowei Li

import numpy as np
import torch


def model_input_clean(data):
    '''
    Clean input data for model input.
    Only include required fields for inference.
    '''
    data_new = {}
    model_input_fields = get_model_input_fields()
    for key in model_input_fields.keys():
        if key == 'inputs_m1' or key == 'inputs_m2':
            if key in data:
                temp = {}
                for sub_key in model_input_fields[key]:
                    temp[sub_key] = data[key][sub_key]
                data_new[key] = temp
        elif key == 'agent_modality_list':
            # Extract numbers from modality strings (e.g., 'm1' -> 1)
            # and create a tensor
            modality_numbers = [int(modality[1:]) for modality in data[key]]
            data_new[key] = torch.tensor(modality_numbers)
        else:
            data_new[key] = data[key]
    return data_new


def get_model_input_fields():
    '''
    Return required fields for model input.
    '''
    input_list = {}
    input_list['inputs_m1'] = ['voxel_features', 'voxel_coords', 
                              'voxel_num_points']
    input_list['inputs_m2'] = ['img', 'rots', 'trans', 'intrins', 
                              'post_rots', 'post_trans']
    input_list['pairwise_t_matrix'] = None
    input_list['record_len'] = None
    input_list['agent_modality_list'] = None
    return input_list


def model_input_clean_late(data):
    '''
    Clean input data for model input, for late fusion.
    Only include required fields for inference.
    '''
    data_new = {}
    model_input_fields = get_model_input_fields_late()
    for key in model_input_fields.keys():
        if key == 'inputs_m1' or key == 'inputs_m2':
            if key in data:
                temp = {}
                for sub_key in model_input_fields[key]:
                    temp[sub_key] = data[key][sub_key]
                data_new[key] = temp
        else:
            data_new[key] = data[key]
    return data_new


def get_model_input_fields_late():
    '''
    Return required fields for model input, for late fusion.
    '''
    input_list = {}
    input_list['inputs_m1'] = ['voxel_features', 'voxel_coords', 
                              'voxel_num_points']
    return input_list


def get_empty_dict_with_keys(input):
    '''
    Get nested structure from dictionary that preserves structure.
    Input structure should only be list, dict, or tensor.
    '''
    if isinstance(input, dict):
        output = {}
        for key, value in input.items():
            if key == 'onnx_export':
                continue
            # If the value is a dictionary, create a nested structure
            output[key] = get_empty_dict_with_keys(value)
    elif isinstance(input, list):
        output = []
        # If the value is a list, create a nested structure
        for item in input:
            output.append(get_empty_dict_with_keys(item))
    elif isinstance(input, torch.Tensor):
        # If the value is a tensor, return None
        return None
    else:
        raise(TypeError(f'Input type is not supported: {type(input)}.'))
    return output


def get_flat_key_list_from_dict(input):
    '''
    Get flat key list from nested dictionary and list recursively.
    Return a flattened list of keys according to the order of keys in input_dict.
    '''
    flat_key_list = []
    if isinstance(input, dict):
        for key, value in input.items():
            if isinstance(value, torch.Tensor):
                flat_key_list.append(key)
            flat_key_list.extend(get_flat_key_list_from_dict(value))
    elif isinstance(input, list):
        for item in input:
            flat_key_list.extend(get_flat_key_list_from_dict(item))
    elif isinstance(input, torch.Tensor):
        return flat_key_list
    else:
        raise(TypeError(f'Input type is not supported: {type(input)}.'))
    return tuple(flat_key_list)


def get_flat_value_list_from_dict(input):
    '''
    Get flat value list from nested dictionary and list recursively.
    Return a flattened list of values according to the order of keys in input_dict.
    '''
    flat_value_list = []
    if isinstance(input, dict):
        for key, value in input.items():
            flat_value_list.extend(get_flat_value_list_from_dict(value))
    elif isinstance(input, list):
        for item in input:
            flat_value_list.extend(get_flat_value_list_from_dict(item))
    elif isinstance(input, torch.Tensor):
        flat_value_list.append(input)
    else:
        raise(TypeError(f'Input type is not supported: {type(input)}.'))
    return tuple(flat_value_list)


def reconstruct_dict_from_flat_value_list(input, values):
    '''
    Reconstruct structure from values (a list of values) to be filled into input.
    Input is the structure (of list or dict structure) to be filled with values.
    '''
    if isinstance(input, dict):
        for key in input.keys():
            if input[key] is None:
                input[key] = values.pop(0)
            else:
                reconstruct_dict_from_flat_value_list(input[key], values)
    elif isinstance(input, list):
        for i in range(len(input)):
            if input[i] is None:
                input[i] = values.pop(0)
            else:
                reconstruct_dict_from_flat_value_list(input[i], values)
    else:
        # If the value is not a dictionary or list, just add the key
        raise(TypeError(f'Input type is not supported: {type(input)}.'))