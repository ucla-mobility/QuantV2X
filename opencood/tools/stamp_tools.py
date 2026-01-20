# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import torch
import os
import sys
from collections import OrderedDict
import glob
import re

def get_model_path_from_dir(model_dir):
    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            raise "No checkpoint!"
        
        return os.path.join(save_dir, f'net_epoch{initial_epoch_}.pth')

    file_list = glob.glob(os.path.join(model_dir, 'net_epoch_bestval_at*.pth'))

    if len(file_list):
        assert len(file_list) == 1
        model_path = file_list[0]
    else:
        model_path = findLastCheckpoint(model_dir)

    print(f"find {model_path}.")
    
    return model_path


def rename_to_new_version(checkpoint_path):
    # stage1 model to new vesrion
    # 加载 checkpoint
    old_state_dict = torch.load(checkpoint_path)

    # 创建一个新的字典，用于保存重命名后的键值对
    new_state_dict = OrderedDict()

    # 遍历旧的 state_dict，将所有的键进行重命名，然后保存到新的字典中
    for key in old_state_dict:
        # 将 'model.model' 替换为 'channel_align.model'
        new_key = key.replace('model.model', 'channel_align.model')
        new_key = new_key.replace('model.warpnet', 'warpnet')
        new_state_dict[new_key] = old_state_dict[key]


    # 保存新的 checkpoint
    torch.save(new_state_dict, checkpoint_path)
    torch.save(old_state_dict, checkpoint_path.replace(".pth", ".pth.oldversion"))

def remove_m4_trunk(checkpoint_path):
    # 加载 checkpoint
    old_state_dict = torch.load(checkpoint_path)

    # 创建一个新的字典，用于保存重命名后的键值对
    new_state_dict = OrderedDict()

    # 遍历旧的 state_dict，将所有的键进行重命名，然后保存到新的字典中
    for key in old_state_dict:
        if key.startswith("encoder_m4.camencode.trunk") or \
            key.startswith('encoder_m4.camencode.final_conv') or \
            key.startswith("encoder_m4.camencode.layer3"):
            continue

        new_state_dict[key] = old_state_dict[key]

    # 保存新的 checkpoint
    torch.save(new_state_dict, checkpoint_path)
    torch.save(old_state_dict, checkpoint_path.replace(".pth", ".pth.oldversion"))

def merge_dict(single_model_dict, stage1_model_dict):
    merged_dict = OrderedDict()
    single_keys = set(single_model_dict.keys())
    if "*fusion_net*" in single_keys:
        print('fusion_net in single_model_dict')
    stage1_keys = set(stage1_model_dict.keys())
    symm_diff_set = single_keys & stage1_keys
    overlap_module = set([key.split(".")[0] for key in symm_diff_set])
    print("=======Overlap modules in two checkpoints=======")
    print(*overlap_module, sep="\n")
    for param in symm_diff_set:
        if not torch.equal(single_model_dict[param], stage1_model_dict[param]):
            print(f"[WARNING]: Different param in {param}")
    print("================================================")

    for key in single_model_dict:
        # remove keys like 'layers_m4.resnet.layer2.0.bn1.bias' / 'cls_head_m4.weight' / 'shrink_conv_m4.weight'
        # from single_model_dict
        # if 'layers_m' in key or 'head_m' in key or 'shrink_conv_m' in key: 
        #     print(f"Pass {key}")
        #     continue
        merged_dict[key] = single_model_dict[key]

    for key in stage1_keys:
        merged_dict[key] = stage1_model_dict[key]

    return merged_dict

def merge_dict_diffcomm(single_model_dict, stage1_model_dict):
    '''
    diffcomm means keeps all modality's backbone and encoder, 
    fusion_net, and head, message extractor, shrink_head
    '''
    
    merged_dict = OrderedDict()
    
    single_keys = set(single_model_dict.keys())
    stage1_keys = set(stage1_model_dict.keys())
    # single_modality = next((k for k in single_keys if "backbone" in k), None).split(".")[0].split("_")[1]
    # stage1_modality = next((k for k in stage1_keys if "backbone" in k), None).split(".")[0].split("_")[1]
    # single_model_dict['message_extractor_' + single_modality + '.weight'] \
    #     = single_model_dict['cls_head.weight']
    # single_model_dict['message_extractor_' + single_modality + '.bias'] \
    #     = single_model_dict['cls_head.bias']
    # stage1_model_dict['message_extractor_' + stage1_modality + '.weight'] \
    #     = stage1_model_dict['cls_head.weight']
    # stage1_model_dict['message_extractor_' + stage1_modality + '.bias'] \
    #     = stage1_model_dict['cls_head.bias']

    single_keys = set(single_model_dict.keys())
    stage1_keys = set(stage1_model_dict.keys())
    symm_diff_set = single_keys & stage1_keys
    overlap_module = set([key.split(".")[0] for key in symm_diff_set])
    print("=======Overlap modules in two checkpoints=======")
    print(*overlap_module, sep="\n")
    for param in symm_diff_set:
        if not torch.equal(single_model_dict[param], stage1_model_dict[param]):
            print(f"[WARNING]: Different param in {param}")
    print("================================================")

    for key in single_model_dict:
        # remove keys like 'layers_m4.resnet.layer2.0.bn1.bias' / 'cls_head_m4.weight' / 'shrink_conv_m4.weight'
        # from single_model_dict
        # if 'layers_m' in key or 'head_m' in key or 'shrink_conv_m' in key: 
        #     print(f"Pass {key}")
        #     continue
        merged_dict[key] = single_model_dict[key]

    for key in stage1_keys:
        merged_dict[key] = stage1_model_dict[key]

    return merged_dict
    
    
def merge_and_save(single_model_dir, stage1_model_dir, output_model_dir):
    single_model_path = get_model_path_from_dir(single_model_dir)
    stage1_model_path = get_model_path_from_dir(stage1_model_dir)
    single_model_dict = torch.load(single_model_path, map_location='cpu')
    stage1_model_dict = torch.load(stage1_model_path, map_location='cpu')
    merged_dict = merge_dict(single_model_dict, stage1_model_dict)
    
    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(merged_dict, output_model_path)
    
def merge_and_save_diffcomm(single_model_dir, stage1_model_dir, output_model_dir, dair_flag=False):
    if dair_flag:
        single_model_dict = change_modality_key_name(single_model_dir)
    else:
        single_model_path = get_model_path_from_dir(single_model_dir)
        single_model_dict = torch.load(single_model_path, map_location='cpu')
    stage1_model_path = get_model_path_from_dir(stage1_model_dir)
    stage1_model_dict = torch.load(stage1_model_path, map_location='cpu')
    
    for key in list(stage1_model_dict.keys()):
        if key.startswith('encoder_m3.'):
            new_key = key.replace('encoder_m3.', 'encoder_m0.')
            stage1_model_dict[new_key] = stage1_model_dict.pop(key)
    for key in list(stage1_model_dict.keys()):
        if key.startswith('backbone_m3.'):
            new_key = key.replace('backbone_m3.', 'backbone_m0.')
            stage1_model_dict[new_key] = stage1_model_dict.pop(key)
    for key in list(stage1_model_dict.keys()):
        if key.startswith('shrinker_m3.'):
            new_key = key.replace('shrinker_m3.', 'shrinker_m0.')
            stage1_model_dict[new_key] = stage1_model_dict.pop(key)
    
    
    # 对 stage1_model_dict 中的 message_extractor 参数加后缀 _m0
    for key in list(stage1_model_dict.keys()):
        if key.startswith('fusion_net.'):
            new_key = key.replace('fusion_net.', 'fusion_net_m0.')
            stage1_model_dict[new_key] = stage1_model_dict.pop(key)

    # 对 single_model_dict 中的 message_extractor 参数加后缀 _mx
    for key in list(single_model_dict.keys()):
        if key.startswith('fusion_net.'):
            new_key = key.replace('fusion_net.', 'fusion_net_m3.')
            single_model_dict[new_key] = single_model_dict.pop(key)
            
        # 对 stage1_model_dict 中的 message_extractor 参数加后缀 _m0
    for key in list(stage1_model_dict.keys()):
        if key.startswith('cls_head.'):
            new_key = key.replace('cls_head.', 'cls_head_m0.')
            stage1_model_dict[new_key] = stage1_model_dict.pop(key)

    # 对 single_model_dict 中的 message_extractor 参数加后缀 _mx
    for key in list(single_model_dict.keys()):
        if key.startswith('cls_head.'):
            new_key = key.replace('cls_head.', 'cls_head_m3.')
            single_model_dict[new_key] = single_model_dict.pop(key)
            
    for key in list(stage1_model_dict.keys()):
        if key.startswith('reg_head.'):
            new_key = key.replace('reg_head.', 'reg_head_m0.')
            stage1_model_dict[new_key] = stage1_model_dict.pop(key)

    # 对 single_model_dict 中的 message_extractor 参数加后缀 _mx
    for key in list(single_model_dict.keys()):
        if key.startswith('reg_head.'):
            new_key = key.replace('reg_head.', 'reg_head_m3.')
            single_model_dict[new_key] = single_model_dict.pop(key)

    for key in list(stage1_model_dict.keys()):
        if key.startswith('dir_head.'):
            new_key = key.replace('dir_head.', 'dir_head_m0.')
            stage1_model_dict[new_key] = stage1_model_dict.pop(key)

    # 对 single_model_dict 中的 message_extractor 参数加后缀 _mx
    for key in list(single_model_dict.keys()):
        if key.startswith('dir_head.'):
            new_key = key.replace('dir_head.', 'dir_head_m3.')
            single_model_dict[new_key] = single_model_dict.pop(key)
            
    
    merged_dict = merge_dict_diffcomm(single_model_dict, stage1_model_dict)
    
    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(merged_dict, output_model_path)

def merge_dict_mpda(path1, path2):
    dict1 = torch.load(get_model_path_from_dir(path1), map_location='cpu')   #ego
    dict2 = torch.load(get_model_path_from_dir(path2), map_location='cpu')
    
    merged_dict = OrderedDict()
    
    dict1_keys = set(dict1.keys())
    dict2_keys = set(dict2.keys())
    
    symm_diff_set = dict1_keys & dict2_keys
    overlap_module = set([key.split(".")[0] for key in symm_diff_set])
    print("=======Overlap modules in two checkpoints=======")
    print(*overlap_module, sep="\n")
    for param in symm_diff_set:
        if not torch.equal(dict2[param], dict1[param]):
            print(f"[WARNING]: Different param in {param}")
    print("================================================")

    for key in dict2:
        # remove keys like 'layers_m4.resnet.layer2.0.bn1.bias' / 'cls_head_m4.weight' / 'shrink_conv_m4.weight'
        # from single_model_dict
        if 'layers_m' in key or 'head_m' in key or 'shrink_conv_m' in key: 
            print(f"Pass {key}")
            continue
        merged_dict[key] = dict2[key]

    for key in dict1_keys:
        merged_dict[key] = dict1[key]

    return merged_dict

def merge_and_save_final(aligned_model_dir_list, output_model_dir):
    """
    aligned_model_dir_list:
        e.g. [m2_ALIGNTO_m1_model_dir, m3_ALIGNTO_m1_model_dir, m4_ALIGNTO_m1_model_dir, m1_collaboration_base_dir]

    output_model_dir:
        model_dir.
    """
    final_dict = OrderedDict()
    for aligned_model_dir in aligned_model_dir_list:
        aligned_model_path = get_model_path_from_dir(aligned_model_dir)
        model_dict = torch.load(aligned_model_path, map_location='cpu')
        final_dict = merge_dict(final_dict, model_dict)

    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(final_dict, output_model_path)


_MODALITY_PREFIXES = (
    "encoder",
    "backbone",
    "shrinker",
    "aligner",
    "fusion_net",
    "cls_head",
    "reg_head",
    "dir_head",
    "adapter",
    "reverter",
    "compressor",
    "shrink_conv",
    "pyramid_backbone",
    "head",
)


def _remap_modality_key(key, src_modality, dst_modality, map_unscoped=False):
    for prefix in _MODALITY_PREFIXES:
        src_prefix = f"{prefix}_{src_modality}."
        if key.startswith(src_prefix):
            return f"{prefix}_{dst_modality}." + key[len(src_prefix) :]
        if map_unscoped:
            unscoped_prefix = f"{prefix}."
            if key.startswith(unscoped_prefix):
                return f"{prefix}_{dst_modality}." + key[len(unscoped_prefix) :]
    return None


def _extract_modality_state(state_dict, src_modality, dst_modality, map_unscoped=False):
    remapped = OrderedDict()
    for key, value in state_dict.items():
        new_key = _remap_modality_key(key, src_modality, dst_modality, map_unscoped)
        if new_key is not None:
            remapped[new_key] = value
    return remapped


def merge_stage2_init(protocol_dir, ego_dir, output_model_dir,
                      protocol_src="m0", protocol_dst="m0",
                      ego_src="m1", ego_dst="m1", map_unscoped=False):
    """
    Merge protocol (m0 base) + ego (stage1) checkpoints into a stage2 init.
    """
    protocol_path = get_model_path_from_dir(protocol_dir)
    ego_path = get_model_path_from_dir(ego_dir)
    protocol_state = torch.load(protocol_path, map_location='cpu')
    ego_state = torch.load(ego_path, map_location='cpu')

    merged = OrderedDict()
    ego_remapped = _extract_modality_state(ego_state, ego_src, ego_dst, map_unscoped)
    protocol_remapped = _extract_modality_state(protocol_state, protocol_src, protocol_dst, map_unscoped)

    if not ego_remapped:
        raise RuntimeError(f"No keys matched ego modality {ego_src} in {ego_path}")
    if not protocol_remapped:
        raise RuntimeError(f"No keys matched protocol modality {protocol_src} in {protocol_path}")

    merged.update(ego_remapped)
    merged.update(protocol_remapped)

    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(merged, output_model_path)
    print(f"Saved stage2 init checkpoint: {output_model_path}")

def add_suffix_to_keys(model_dict, suffix):
    """
    Add suffix to keys in model_dict.
    """
    for key in model_dict.keys():
        if key.startswith('message_extractor.'):
            new_key = key.replace('message_extractor.', f'message_extractor_{suffix}.')
            model_dict[new_key] = model_dict[key]
    return model_dict

def add_suffix_to_keys_save(log_path, suffix, save_path):
    """
    Add suffix to keys in model_dict.
    """
    model_path = get_model_path_from_dir(log_path)

    model_dict = torch.load(model_path, map_location='cpu')
    for key in list(model_dict.keys()):
        if key.startswith('message_extractor.'):
            new_key = key.replace('message_extractor.', f'message_extractor_{suffix}.')
            model_dict[new_key] = model_dict.pop(key)
    torch.save(model_dict, os.path.join(save_path,'net_epoch1.pth'))

def change_modality_key_name(log_path,):
    """
    Change 'm1' to 'm3' in model_dict.
    """
    model_path = get_model_path_from_dir(log_path)
    model_dict = torch.load(model_path, map_location='cpu')
    for key in list(model_dict.keys()):
        if 'm1' in key:
            new_key = key.replace('m1', 'm3')
            model_dict[new_key] = model_dict.pop(key)
    return model_dict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: stamp_tools.py <rename_to_new_version|remove_m4_trunk|merge|merge_final|"
            "merge_diffcomm|add_suffix|merge_stage2_init> [args...]"
        )

    func = sys.argv[1]
    if func == "rename_to_new_version":
        checkpoint_path = sys.argv[2]
        rename_to_new_version(checkpoint_path)
    elif func == "remove_m4_trunk":
        checkpoint_path = sys.argv[2]
        remove_m4_trunk(checkpoint_path)
    elif func == "merge":
        single_model_dir = sys.argv[2]
        stage1_model_dir = sys.argv[3]
        output_model_dir = sys.argv[4]
        merge_and_save(single_model_dir, stage1_model_dir, output_model_dir)
    elif func == "merge_final":
        if len(sys.argv) < 4:
            raise SystemExit("Usage: stamp_tools.py merge_final <dir1> <dir2> ... <output_dir>")
        merge_and_save_final(sys.argv[2:-1], sys.argv[-1])
    elif func == "merge_diffcomm":
        single_model_dir = sys.argv[2]
        stage1_model_dir = sys.argv[3]
        output_model_dir = sys.argv[4]
        dair_flag = "--dair" in sys.argv[5:]
        merge_and_save_diffcomm(single_model_dir, stage1_model_dir, output_model_dir, dair_flag=dair_flag)
    elif func == "add_suffix":
        log_path = sys.argv[2]
        suffix = sys.argv[3]
        save_path = sys.argv[4]
        add_suffix_to_keys_save(log_path, suffix, save_path)
    elif func == "merge_stage2_init":
        parser = argparse.ArgumentParser(
            description="Merge protocol (m0) + ego (mX) checkpoints for stage2 adapter training."
        )
        parser.add_argument("protocol_dir", help="Checkpoint dir for protocol base (m0).")
        parser.add_argument("ego_dir", help="Checkpoint dir for ego base (e.g., stage1 m1).")
        parser.add_argument("output_dir", help="Output dir to write net_epoch1.pth.")
        parser.add_argument("--protocol-src", default="m0", help="Source modality in protocol checkpoint.")
        parser.add_argument("--protocol-dst", default="m0", help="Target modality name for protocol weights.")
        parser.add_argument("--ego-src", default="m1", help="Source modality in ego checkpoint.")
        parser.add_argument("--ego-dst", default="m1", help="Target modality name for ego weights.")
        parser.add_argument("--map-unscoped", action="store_true",
                            help="Map unscoped heads (cls_head.*, etc.) to target modality.")
        args = parser.parse_args(sys.argv[2:])
        merge_stage2_init(
            args.protocol_dir,
            args.ego_dir,
            args.output_dir,
            protocol_src=args.protocol_src,
            protocol_dst=args.protocol_dst,
            ego_src=args.ego_src,
            ego_dst=args.ego_dst,
            map_unscoped=args.map_unscoped,
        )
    else:
        raise SystemExit(f"Unknown function: {func}")
