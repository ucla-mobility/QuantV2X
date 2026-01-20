# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License

# This is used for STAMP

import argparse
import os
import cv2
import numpy as np
import statistics

# from fvcore.nn import FlopCountAnalysis
# from fvcore.nn import flop_count_table

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils_stamp as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic
from collections import OrderedDict
from opencood.tools import multi_gpu_utils


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True, help="data generation yaml file needed ")
    parser.add_argument("--model_dir", default="", help="Continued training path")
    parser.add_argument("--fusion_method", "-f", default="intermediate", help="passed to inference.")
    parser.add_argument('--flop_count', action='store_true')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)

    print("Dataset Building")
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes["train_params"]["batch_size"], drop_last=True
        )

        train_loader = DataLoader(
            opencood_train_dataset,
            batch_sampler=batch_sampler_train,
            num_workers=32,
            collate_fn=opencood_train_dataset.collate_batch_train,
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            sampler=sampler_val,
            num_workers=32,
            collate_fn=opencood_train_dataset.collate_batch_train,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            opencood_train_dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=32,
            collate_fn=opencood_train_dataset.collate_batch_train,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=32,
            collate_fn=opencood_train_dataset.collate_batch_train,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    print("Creating Model")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion_dict = train_utils.create_losses_heter(hypes)
    criterion_adapter = train_utils.create_adapter_loss(hypes)
    # criterion = nn.MSELoss()

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        if hasattr(model, "model_train_init"):
            model.model_train_init()
        # model = train_utils.load_original_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")

    else:
        raise NotImplementedError("model_dir must be provided for training adapter")

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # ddp setting
    model_without_ddp = model

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[opt.gpu], find_unused_parameters=True
        )  # True
        model_without_ddp = model.module

    # record training
    writer = SummaryWriter(saved_path)

    print("Training start")
    epoches = hypes["train_params"]["epoches"]
    supervise_single_flag = (
        False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    )
    # used to help schedule learning rate
    
    for epoch in [0] if opt.flop_count else range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print("learning rate %f" % param_group["lr"])
        # the model will be evaluation mode during validation
        model.train()
        try:  # heter_model stage2
            model.model_train_init()
        except:
            print("No model_train_init function")
        total_flops_sum = 0
        grad_flops_sum = 0
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data["ego"]["object_bbx_mask"].sum() == 0:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data["ego"]["epoch"] = epoch
            
            
            # if opt.flop_count:
            #     flops = FlopCountAnalysis(model, batch_data['ego'])
            #     flops_counter = flops.by_module()
            #     for key, value in flops_counter.items():
                    
            #         if getattr(model, key, None) is not None:
            #             requires_grad = False
            #             for param in getattr(model, key).parameters():
            #                 if param.requires_grad == True:
            #                     requires_grad = True
            #             if requires_grad:
            #                 grad_flops_sum += value
            #             total_flops_sum += value
            #     print(i, "/", len(train_loader), "Grad FLOPS: ", grad_flops_sum)
            #     print(i, "/", len(train_loader), "Total FLOPS: ", total_flops_sum)
            #     del flops
            #     torch.cuda.empty_cache()
            #     continue
            
            
            
            output_dict, output_feat = model(batch_data["ego"])
            loss_adapter = 0
            if output_feat is not None:
                FM, FP2M, FM2P2M, FP, FM2P = output_feat
                loss_adapter = criterion_adapter(FM, FP2M, FM2P2M, FP, FM2P)
                criterion_adapter.logging(epoch, i, len(train_loader), writer)
                
            final_loss_dict = dict()
            if output_dict is not None:
                for modality_name in criterion_dict.keys():
                    if modality_name == "m0":  # protocol modality
                        final_loss_dict[modality_name] = criterion_dict[modality_name](
                            output_dict[modality_name], batch_data["ego"]["label_dict_protocol"]
                        )
                        criterion_dict[modality_name].logging(epoch, i, len(train_loader), writer)
                        pass
                    else:
                        final_loss_dict[modality_name] = criterion_dict[modality_name](
                            output_dict[modality_name], batch_data["ego"]["label_dict"]
                        )
                        criterion_dict[modality_name].logging(epoch, i, len(train_loader), writer)
            final_loss = sum(final_loss_dict.values()) + loss_adapter
            
            # d1 = output_dict[modality_name]['dynamic_seg'][0][0].cpu().detach().numpy()
            # d2 = output_dict[modality_name]['dynamic_seg'][0][1].cpu().detach().numpy()
            # dmax = max(d1.max(), d2.max())
            # dmin = min(d1.min(), d2.min())
            # s1 = output_dict[modality_name]['static_seg'][0][0].cpu().detach().numpy()
            # s2 = output_dict[modality_name]['static_seg'][0][1].cpu().detach().numpy()
            # s3 = output_dict[modality_name]['static_seg'][0][2].cpu().detach().numpy()
            # smax = max(s1.max(), s2.max(), s3.max())
            # smin = min(s1.min(), s2.min(), s3.min())
            # cv2.imwrite('debug/dynamic_seg.png', (((d1 - dmin) / (dmax - dmin) * 255).astype(np.uint8)))
            # cv2.imwrite('debug/dynamic_seg2.png', (((d2 - dmin) / (dmax - dmin) * 255).astype(np.uint8)))
            # cv2.imwrite('debug/static_seg.png', (((s1 - smin) / (smax - smin) * 255).astype(np.uint8)))
            # cv2.imwrite('debug/static_seg2.png', (((s2 - smin) / (smax - smin) * 255).astype(np.uint8)))
            # cv2.imwrite('debug/static_seg3.png', (((s3 - smin) / (smax - smin) * 255).astype(np.uint8)))
            
            # min_val_p = min(FP[0].mean(0).min(), FM2P[0].mean(0).min())
            # min_val_m = min(FP2M[0].mean(0).min(), FM[0].mean(0).min(), FM2P2M[0].mean(0).min())
            # max_val_p = max(FP[0].mean(0).max(), FM2P[0].mean(0).max())
            # max_val_m = max(FP2M[0].mean(0).max(), FM[0].mean(0).max(), FM2P2M[0].mean(0).max())
            # cv2.imwrite('debug/FP2M.png', (((FP2M[0].mean(0) - min_val_m) / (max_val_m - min_val_m)).cpu().detach().numpy() * 255).astype(np.uint8))
            # cv2.imwrite('debug/FM.png', (((FM[0].mean(0) - min_val_m) / (max_val_m - min_val_m)).cpu().detach().numpy() * 255).astype(np.uint8))
            # cv2.imwrite('debug/FM2P2M.png', (((FM2P2M[0].mean(0) - min_val_m) / (max_val_m- min_val_m)).cpu().detach().numpy() * 255).astype(np.uint8))
            # cv2.imwrite('debug/FP.png', (((FP[0].mean(0) - min_val_p) / (max_val_p - min_val_p)).cpu().detach().numpy() * 255).astype(np.uint8))
            # cv2.imwrite('debug/FM2P.png', (((FM2P[0].mean(0) - min_val_p) / (max_val_p - min_val_p)).cpu().detach().numpy() * 255).astype(np.uint8))
            # if i % 10 == 0:
            # print(f'epoch {epoch}, iter {i}, loss_p2e {loss_p2e:.3f}, loss_e2p2e {loss_e2p2e:.3f}, loss_e2p {loss_e2p:.3f}, final_loss {final_loss:.3f}')

            # back-propagation
            final_loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Module: {name}, Grad: {param.grad.mean()}")
            optimizer.step()

            # torch.cuda.empty_cache()  # it will destroy memory buffer

        if epoch % hypes["train_params"]["save_freq"] == 0:
            torch.save(model.state_dict(), os.path.join(saved_path, "net_epoch%d.pth" % (epoch + 1)))

        if epoch % hypes["train_params"]["eval_freq"] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data["ego"]["epoch"] = epoch
                    output_dict, output_feat = model(batch_data["ego"])
                    loss_adapter = 0
                    if output_feat is not None:
                        FM, FP2M, FM2P2M, FP, FM2P = output_feat
                        loss_adapter = criterion_adapter(FM, FP2M, FM2P2M, FP, FM2P)
                        criterion_adapter.logging(epoch, i, len(train_loader), writer)
                        
                    final_loss_dict = dict()
                    if output_dict is not None:
                        for modality_name in criterion_dict.keys():
                            if modality_name == "m0":  # protocol modality
                                final_loss_dict[modality_name] = criterion_dict[modality_name](
                                    output_dict[modality_name], batch_data["ego"]["label_dict_protocol"]
                                )
                            else:
                                final_loss_dict[modality_name] = criterion_dict[modality_name](
                                    output_dict[modality_name], batch_data["ego"]["label_dict"]
                                )
                            criterion_dict[modality_name].logging(epoch, i, len(train_loader), writer)
                    final_loss = sum(final_loss_dict.values()) + loss_adapter
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print("At epoch %d, the validation loss is %f" % (epoch, valid_ave_loss))
            writer.add_scalar("Validate_Loss", valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(), os.path.join(saved_path, "net_epoch_bestval_at%d.pth" % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(
                    os.path.join(saved_path, "net_epoch_bestval_at%d.pth" % (lowest_val_epoch))
                ):
                    os.remove(os.path.join(saved_path, "net_epoch_bestval_at%d.pth" % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        scheduler.step(epoch)

        opencood_train_dataset.reinitialize()

    print("Training Finished, checkpoints saved to %s" % saved_path)
    # if opt.flop_count:
    #     print("Total FLOPS: ", total_flops)
    #     return

    # run_test = True
    # if run_test:
    #     fusion_method = opt.fusion_method
    #     cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
    #     print(f"Running command: {cmd}")
    #     os.system(cmd)


if __name__ == "__main__":
    main()
