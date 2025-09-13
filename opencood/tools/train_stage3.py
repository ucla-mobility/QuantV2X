#need to save config.yaml manually right now, add that

import shutil
from datetime import datetime
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.heter_pyramid_collab_codebook import HeterPyramidCollabCodebook


def train_parser():
    parser = argparse.ArgumentParser(description="Stage 3: End-to-End Fine-tuning")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='YAML file for training configuration')
    parser.add_argument('--stage2_model', type=str, required=True,
                        help='Path to trained stage 2 model')
    parser.add_argument('--model_dir', default='',
                        help='Directory to save or resume fine-tuned models')
    return parser.parse_args()


def find_latest_checkpoint(model_dir):
    ckpts = [f for f in os.listdir(model_dir) if f.startswith("finetune_epoch") and f.endswith(".pth")]
    if not ckpts:
        return None, 0
    ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest = ckpts[-1]
    epoch = int(latest.split('_')[-1].split('.')[0])
    return os.path.join(model_dir, latest), epoch


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    val_dataset = build_dataset(hypes, visualize=False, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hypes['train_params']['batch_size'],
        num_workers=8,
        shuffle=True,
        collate_fn=train_dataset.collate_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        collate_fn=val_dataset.collate_batch
    )

    print('Model')
    model = train_utils.create_model(hypes)

    # Set up model directory
    if not opt.model_dir:
        model_dir = train_utils.setup_train(hypes)
        print(f"New training run, saving to: {model_dir}")
        # Load stage 2 model
        print(f'Loading stage 2 model from {opt.stage2_model}')
        checkpoint = torch.load(opt.stage2_model, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        init_epoch = 0
    else:
        model_dir = opt.model_dir
        os.makedirs(model_dir, exist_ok=True)
        latest_ckpt, init_epoch = find_latest_checkpoint(model_dir)
        if latest_ckpt:
            print(f"Resuming from checkpoint {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
        else:
            print(f"No checkpoint found in {model_dir}, loading stage 2 model fresh.")
            checkpoint = torch.load(opt.stage2_model, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            init_epoch = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model = model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    lowest_val_loss = 1e5
    lowest_val_epoch = -1
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer)
    codebook_weight = hypes['train_params'].get('stage3_codebook_weight', 0.5)




    writer = SummaryWriter(model_dir)
    num_epochs = hypes['train_params'].get('epoches', 20)

    print('Starting Stage 3 Fine-tuning...')
    for epoch in range(init_epoch, num_epochs):
        model.train()
        total_loss_epoch = 0

        for i, batch_data in enumerate(train_loader):
            batch_data = train_utils.to_device(batch_data, device)
            output_dict = model(batch_data['ego'])
            detection_loss = criterion(output_dict, batch_data['ego']['label_dict'])
            codebook_loss = output_dict.get('codebook_loss', 0)
            total_loss = detection_loss + codebook_weight * codebook_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_epoch += total_loss.item()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                      f"Detection Loss: {detection_loss.item():.4f}, Codebook Loss: {codebook_loss.item():.4f}")
                global_step = epoch * len(train_loader) + i
                writer.add_scalar('train/detection_loss', detection_loss.item(), global_step)
                writer.add_scalar('train/codebook_loss', codebook_loss.item(), global_step)
                writer.add_scalar('train/total_loss', total_loss.item(), global_step)

        # Validation
        if epoch % hypes['train_params']['eval_freq'] == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    batch_data = train_utils.to_device(batch_data, device)
                    output_dict = model(batch_data['ego'])
                    det_loss = criterion(output_dict, batch_data['ego']['label_dict'])
                    cb_loss = output_dict.get('codebook_loss', 0)
                    val_loss += det_loss.item() #(det_loss + codebook_weight * cb_loss).item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")
            writer.add_scalar('val/total_loss', avg_val_loss, epoch)
            writer.add_scalar('val/det_loss', det_loss, epoch)

            if avg_val_loss < lowest_val_loss:
                lowest_val_loss = avg_val_loss
                torch.save(model.state_dict(),
                       os.path.join(model_dir,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(model_dir,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(model_dir,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        # Save checkpoint
        if (epoch + 1) % hypes['train_params']['save_freq'] == 0:
            ckpt_path = os.path.join(model_dir, f'net_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        scheduler.step()
        print(f"LR after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}")

    print('Training completed.')
    writer.close()


if __name__ == '__main__':
    main()
