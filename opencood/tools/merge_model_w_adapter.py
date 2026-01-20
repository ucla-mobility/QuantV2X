#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
import sys
import types

import torch
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _ensure_stubbed_imports():
    """Stub optional deps used by PyramidFusion import paths."""
    if "icecream" not in sys.modules:
        icecream = types.ModuleType("icecream")
        icecream.ic = lambda *args, **kwargs: None
        sys.modules["icecream"] = icecream

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")
        pyplot.figure = lambda *args, **kwargs: None
        pyplot.imshow = lambda *args, **kwargs: None
        pyplot.savefig = lambda *args, **kwargs: None
        pyplot.close = lambda *args, **kwargs: None
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.pyplot"] = pyplot


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _find_latest_ckpt(model_dir):
    ckpts = glob.glob(os.path.join(model_dir, "net_epoch*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    epochs = []
    for ckpt in ckpts:
        match = re.findall(r".*net_epoch(\d+)\.pth", ckpt)
        if match:
            epochs.append((int(match[0]), ckpt))
    if not epochs:
        raise FileNotFoundError(f"No epoch checkpoints found in {model_dir}")
    return max(epochs, key=lambda x: x[0])[1]


def _find_best_or_latest(model_dir):
    best = glob.glob(os.path.join(model_dir, "net_epoch_bestval_at*.pth"))
    if best:
        if len(best) > 1:
            raise RuntimeError(f"Multiple bestval checkpoints in {model_dir}")
        return best[0]
    return _find_latest_ckpt(model_dir)


def _init_conv_weights(in_channels, out_channels):
    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
    return conv.weight.detach().clone(), conv.bias.detach().clone()


def _build_fusion_state(fusion_cfg, stage1_state, modality):
    _ensure_stubbed_imports()
    from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion

    fusion_net = PyramidFusion(fusion_cfg)
    fusion_state = fusion_net.state_dict()
    for key, value in stage1_state.items():
        if key.startswith(f"pyramid_backbone_{modality}."):
            target_key = key[len(f"pyramid_backbone_{modality}.") :]
        elif key.startswith("pyramid_backbone."):
            target_key = key[len("pyramid_backbone.") :]
        else:
            continue
        if target_key in fusion_state and fusion_state[target_key].shape == value.shape:
            fusion_state[target_key] = value
    return fusion_state


def _add_head_weights(target_state, modality, model_setting, stage1_state, num_class):
    in_head = model_setting["in_head"]
    anchor_num = model_setting["anchor_number"]

    head_specs = {
        "cls_head": anchor_num * num_class * num_class,
        "reg_head": 7 * anchor_num * num_class,
    }
    if model_setting.get("dir_args"):
        head_specs["dir_head"] = model_setting["dir_args"]["num_bins"] * anchor_num * num_class

    for head_name, out_channels in head_specs.items():
        src_w = stage1_state.get(f"{head_name}_{modality}.weight")
        src_b = stage1_state.get(f"{head_name}_{modality}.bias")
        if src_w is None or src_b is None:
            src_w = stage1_state.get(f"{head_name}.weight")
            src_b = stage1_state.get(f"{head_name}.bias")
        expected_w_shape = (out_channels, in_head, 1, 1)
        expected_b_shape = (out_channels,)
        if src_w is not None and src_b is not None and src_w.shape == expected_w_shape and src_b.shape == expected_b_shape:
            weight, bias = src_w, src_b
        else:
            weight, bias = _init_conv_weights(in_head, out_channels)
        target_state[f"{head_name}_{modality}.weight"] = weight
        target_state[f"{head_name}_{modality}.bias"] = bias


def _collect_modalities(model_args):
    return [k for k in model_args.keys() if k.startswith("m") and k[1:].isdigit()]


def main():
    parser = argparse.ArgumentParser(
        description="Merge stage1 base weights + stage2 adapters into final STAMP inference checkpoint."
    )
    parser.add_argument("--infer_dir", required=True, help="Inference dir with config.yaml (m1m2m3m4_infer).")
    parser.add_argument("--stage1_root", required=True, help="Stage1 root dir (contains V2XREAL_mX).")
    parser.add_argument("--stage2_root", required=True, help="Stage2 root dir (contains V2XREAL_m0mX).")
    parser.add_argument("--modalities", default="", help="Comma list of modalities (default: from config).")
    parser.add_argument("--output_name", default="net_epoch1.pth", help="Output checkpoint name.")
    args = parser.parse_args()

    config_path = os.path.join(args.infer_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.yaml in {args.infer_dir}")

    config = _load_yaml(config_path)
    model_args = config["model"]["args"]
    modalities = _collect_modalities(model_args)
    if args.modalities:
        modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]

    num_class = None
    if "preprocess" in config and "num_class" in config["preprocess"]:
        num_class = config["preprocess"]["num_class"]
    if num_class is None:
        num_class = model_args.get("num_class")
    if num_class is None:
        raise RuntimeError("num_class not found in config.")

    merged_state = {}

    for modality in modalities:
        stage1_dir = os.path.join(args.stage1_root, f"V2XREAL_{modality}")
        stage2_dir = os.path.join(args.stage2_root, f"V2XREAL_m0{modality}")
        if not os.path.exists(stage1_dir):
            raise FileNotFoundError(f"Missing stage1 dir: {stage1_dir}")
        if not os.path.exists(stage2_dir):
            raise FileNotFoundError(f"Missing stage2 dir: {stage2_dir}")

        stage1_path = _find_best_or_latest(stage1_dir)
        stage2_path = _find_best_or_latest(stage2_dir)
        stage1_state = torch.load(stage1_path, map_location="cpu")
        stage2_state = torch.load(stage2_path, map_location="cpu")

        for key, value in stage1_state.items():
            if key.startswith(f"encoder_{modality}.") or key.startswith(f"backbone_{modality}."):
                merged_state[key] = value
            elif key.startswith(f"aligner_{modality}."):
                merged_state[key] = value
            elif key.startswith(f"shrinker_{modality}."):
                merged_state[key] = value
            elif key.startswith("shrink_conv."):
                merged_state[f"shrinker_{modality}." + key[len("shrink_conv.") :]] = value

        fusion_copied = False
        for key, value in stage1_state.items():
            if key.startswith(f"fusion_net_{modality}."):
                merged_state[key] = value
                fusion_copied = True
        fusion_cfg = model_args[modality].get("fusion_backbone")
        if fusion_cfg and not fusion_copied:
            fusion_state = _build_fusion_state(fusion_cfg, stage1_state, modality)
            for key, value in fusion_state.items():
                merged_state[f"fusion_net_{modality}.{key}"] = value

        _add_head_weights(merged_state, modality, model_args[modality], stage1_state, num_class)

        for key, value in stage2_state.items():
            if key.startswith(f"adapter_{modality}.") or key.startswith(f"reverter_{modality}."):
                merged_state[key] = value
            elif key.startswith(f"shrinker_{modality}."):
                merged_state[key] = value

    output_path = os.path.join(args.infer_dir, args.output_name)
    torch.save(merged_state, output_path)
    print(f"Saved merged checkpoint: {output_path}")


if __name__ == "__main__":
    main()
