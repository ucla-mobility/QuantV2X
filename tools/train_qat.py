
# -*- coding: utf-8 -*-
"""
train_qat.py — QAT smoke-test entry point for QuantV2X.

Phases:
  1. Config & model init  : load hypes yaml, build dataset/dataloader, build
                             the FP32 base model via train_utils.create_model.
  2. Graph surgery         : QATQuantModel swaps eligible conv/linear/spconv
                             leaves for ternary/2-bit QAT wrappers.
  3. Optimizer fix         : AdamW is built AFTER surgery from
                             qat_model.qat_param_groups() — building it
                             earlier would bind to the old FP32 leaf modules
                             instead of the wrappers' shadow weights/scales/
                             levels, silently orphaning the quant params.
  4. Smoke-test loop        : overfit a single batch for a few iterations,
                             confirming loss decreases and QATTelemetry
                             reports live (non-zero, non-thrashing) flip rates.
"""

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.qat import QATQuantModel, QATTelemetry


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="QAT smoke test")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help="training config yaml (with optional `qat:` section)")
    parser.add_argument("--model_dir", default="",
                        help="optional path to FP32 checkpoint to load before surgery")
    parser.add_argument("--iters", type=int, default=50,
                        help="number of overfit steps on the single smoke-test batch")
    parser.add_argument("--telemetry_every", type=int, default=5,
                        help="print QATTelemetry metrics every N iterations")
    parser.add_argument("--lr_weight", type=float, default=None,
                        help="override qat.lr_weight (FP32 shadow weights)")
    parser.add_argument("--lr_quant", type=float, default=None,
                        help="override qat.lr_quant (scales/levels)")
    return parser.parse_args()


def main():
    seed_all()
    opt = parse_args()

    # ---------------------------------------------------------------- #
    # Phase 1 — config & model initialization
    # ---------------------------------------------------------------- #
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print("[1/4] Building dataset & base model...")
    train_dataset = build_dataset(hypes, visualize=False, train=True, calibrate=False)
    train_loader = DataLoader(train_dataset,
                              batch_size=hypes["train_params"]["batch_size"],
                              num_workers=0,
                              collate_fn=train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

    model = train_utils.create_model(hypes)

    if opt.model_dir:
        _, model = train_utils.load_saved_model(opt.model_dir, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------------------------------------------------------------- #
    # Phase 2 — graph surgery
    # ---------------------------------------------------------------- #
    print("[2/4] Running QAT graph surgery...")
    qat_model = QATQuantModel(model, qat_cfg=hypes)
    qat_model.to(device)
    qat_model.coverage_report(printer=print)

    # ---------------------------------------------------------------- #
    # Phase 3 — optimizer fix: build AFTER surgery
    # ---------------------------------------------------------------- #
    print("[3/4] Building AdamW over post-surgery param groups...")
    param_groups = qat_model.qat_param_groups(lr_weight=opt.lr_weight,
                                              lr_quant=opt.lr_quant)
    optimizer = torch.optim.AdamW(param_groups)

    criterion = train_utils.create_loss(hypes)
    telemetry = QATTelemetry(qat_model)

    # ---------------------------------------------------------------- #
    # Phase 4 — smoke test: overfit a single batch
    # ---------------------------------------------------------------- #
    print("[4/4] Smoke-test overfit loop...")
    batch_data = None
    for candidate in train_loader:
        if candidate is not None and candidate["ego"]["object_bbx_mask"].sum() > 0:
            batch_data = candidate
            break
    if batch_data is None:
        raise RuntimeError("Could not find a non-empty batch in train_loader.")

    batch_data = train_utils.to_device(batch_data, device)
    batch_data["ego"]["epoch"] = 0

    qat_model.train()
    for it in range(opt.iters):
        optimizer.zero_grad()

        output_dict = qat_model(batch_data["ego"])
        loss = criterion(output_dict, batch_data["ego"]["label_dict"])

        loss.backward()
        optimizer.step()

        if it % opt.telemetry_every == 0 or it == opt.iters - 1:
            metrics = telemetry.step(it)
            alarms = telemetry.alarms(metrics)

            flip_rates = [v for k, v in metrics.items() if k.endswith("/flip_rate")]
            avg_flip = sum(flip_rates) / len(flip_rates) if flip_rates else 0.0

            print(f"iter {it:4d} | loss {loss.item():.6f} | avg_flip_rate {avg_flip:.4%}")
            for name, warning in alarms.items():
                print(f"  [ALARM] {name}: {warning}")

    print("Smoke test complete.")


if __name__ == "__main__":
    main()
