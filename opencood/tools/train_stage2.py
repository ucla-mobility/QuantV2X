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
    parser = argparse.ArgumentParser(description="Stage 2: Codebook Training")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed')
    parser.add_argument('--stage1_model', type=str, required=True,
                        help='Path to trained stage 1 model')
    parser.add_argument('--model_dir', default='',
                        help='Directory to save codebook models')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for codebook training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    # Create datasets
    print('Dataset Building')
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hypes['train_params']['batch_size'],
        num_workers=8,
        shuffle=True,
        collate_fn=train_dataset.collate_batch
    )
    
    val_dataset = build_dataset(hypes, visualize=False, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        collate_fn=val_dataset.collate_batch
    )

    # Create model
    print('Creating Model with Codebook...')
    model = train_utils.create_model(hypes)

    # Load pretrained stage 1 model
    print(f'Loading stage 1 model from {opt.stage1_model}')
    checkpoint = torch.load(opt.stage1_model, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    print('Stage 1 model loaded successfully!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model = model.to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    model.codebook.train()
    for p in model.codebook.parameters():
        p.requires_grad_(True)

    # Print trainable parameters
    print('\n----------- Trainable Parameters -----------')
    total_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data.shape}")
            total_trainable += param.numel()
    print(f'Total trainable parameters: {total_trainable:,}')
    print('-------------------------------------------\n')

    # Setup loss
    criterion = train_utils.create_loss(hypes)
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # Optimizer (only for codebook)
    codebook_params = [p for n, p in model.named_parameters() 
                       if p.requires_grad] # and 'codebook' in n
    optimizer = torch.optim.Adam(codebook_params, lr=opt.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Model dir
    if not opt.model_dir:
        model_dir = train_utils.setup_train(hypes)
    else:
        model_dir = opt.model_dir
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory is set to: {model_dir}")

    writer = SummaryWriter(model_dir)

    print('Starting Stage 2 Codebook Training...')
    num_epochs = hypes['train_params'].get('epoches', 20)

    for epoch in range(num_epochs):
        model.train()
        train_codebook_loss = 0

        for i, batch_data in enumerate(train_loader):
            batch_data = train_utils.to_device(batch_data, device)
            output_dict = model(batch_data['ego'])
            codebook_loss = output_dict.get('codebook_loss', 0)

            optimizer.zero_grad()
            codebook_loss.backward()
            optimizer.step()
            train_codebook_loss += codebook_loss.item()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Step [{i}/{len(train_loader)}], "
                      f"Codebook Loss: {codebook_loss.item():.4f}")
                global_step = epoch * len(train_loader) + i
                writer.add_scalar('train/codebook_loss', codebook_loss.item(), global_step)
                if 'codebook_entropy' in output_dict:
                    for level_idx, entropy in enumerate(output_dict['codebook_entropy']):
                        writer.add_scalar(f'codebook/entropy_level_{level_idx}', entropy.item(), global_step)

        if epoch % hypes['train_params']['eval_freq'] == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    batch_data = train_utils.to_device(batch_data, device)
                    output_dict = model(batch_data['ego'])
                    codebook_loss = output_dict.get('codebook_loss', 0)
                    val_loss += codebook_loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
            writer.add_scalar('val/total_loss', avg_val_loss, epoch)
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

        if (epoch + 1) % hypes['train_params']['save_freq'] == 0:
            checkpoint_path = os.path.join(model_dir, f'net_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

    print(f'Training finished! Models saved to {model_dir}')
    writer.close()


if __name__ == '__main__':
    main()
