"""Training script for the DINO/Deformable DETR architecture.

This script handles the instantiation of the multi-scale model, 
the focal loss criterion, and the main training loops.
"""

import os
import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# Import the custom DINO modules we just built
from models.dino import build_dino
from models.loss import HungarianMatcher, SetCriterion
from data_utils.dataset import DigitDetectionDataset
from data_utils.transforms import build_transforms

def collate_fn(batch):
    """Batches images and targets into a format the model can process."""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets
    
def get_args_parser():
    """Parses training arguments specific to DINO."""
    parser = argparse.ArgumentParser('DINO training script', add_help=False)
    
    # Training Parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int) 
    
    # NEW: Gradient Accumulation parameter
    parser.add_argument('--accum_steps', default=1, type=int,
                        help='Number of batches to accumulate gradients before stepping.')
    
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=24, type=int) 
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    # Model Parameters
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    
    # Loss & Matcher Parameters
    parser.add_argument('--aux_loss', action='store_true')
    parser.add_argument('--set_cost_class', default=2.0, type=float)
    parser.add_argument('--set_cost_bbox', default=5.0, type=float)
    parser.add_argument('--set_cost_giou', default=2.0, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # Directories
    parser.add_argument('--train_img_dir', default='dataset/train', type=str)
    parser.add_argument('--train_ann_file', default='dataset/train.json', type=str)
    parser.add_argument('--val_img_dir', default='dataset/valid', type=str)
    parser.add_argument('--val_ann_file', default='dataset/valid.json', type=str)
    parser.add_argument('--output_dir', default='checkpoints/run_dino', type=str)

    return parser

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    accum_steps: int = 1):
    """Trains the model for one single epoch with gradient accumulation."""
    model.train()
    criterion.train()
    
    # Tracking metrics
    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_bbox = 0.0
    total_loss_giou = 0.0
    
    # Progress bar setup
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    
    # NEW: Ensure gradients are clean before starting the epoch
    optimizer.zero_grad() 
    
    # NEW: Added enumerate to track batch index for accumulation
    for batch_idx, (samples, targets) in enumerate(pbar):
        # Move inputs to device
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # Forward pass
        outputs = model(samples)
        
        # Calculate loss mapping predictions to targets
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # Sum all losses (including auxiliary layers) multiplied by their respective weights
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # NEW: Scale the loss down by the accumulation steps
        losses = losses / accum_steps

        # Catch exploding gradients early
        import math
        import sys
        if not math.isfinite(losses.item()):
            print(f"Loss is {losses.item() * accum_steps}, stopping training")
            print(loss_dict)
            sys.exit(1)

        # Backward pass (Accumulates gradients)
        losses.backward()
        
        # NEW: Step the optimizer ONLY when we hit the accumulation threshold or the end of the epoch
        if ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(data_loader)):
            # Gradient clipping - crucial for Deformable DETR stability
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
            optimizer.step()
            optimizer.zero_grad() # Clear gradients for the next accumulation cycle

        # Update metrics (multiplying back by accum_steps for accurate logging display)
        true_loss = losses.item() * accum_steps
        total_loss += true_loss
        total_loss_ce += loss_dict.get('loss_ce', torch.tensor(0.0)).item()
        total_loss_bbox += loss_dict.get('loss_bbox', torch.tensor(0.0)).item()
        total_loss_giou += loss_dict.get('loss_giou', torch.tensor(0.0)).item()
        
        pbar.set_postfix({
            'loss': f"{true_loss:.4f}",
            'ce': f"{loss_dict.get('loss_ce', torch.tensor(0.0)).item():.4f}",
            'bbox': f"{loss_dict.get('loss_bbox', torch.tensor(0.0)).item():.4f}"
        })

    # Calculate epoch averages
    avg_loss = total_loss / len(data_loader)
    avg_ce = total_loss_ce / len(data_loader)
    avg_bbox = total_loss_bbox / len(data_loader)
    avg_giou = total_loss_giou / len(data_loader)
    
    return {
        'train_loss': avg_loss,
        'train_ce': avg_ce,
        'train_bbox': avg_bbox,
        'train_giou': avg_giou
    }

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    """Evaluates the model on the validation set."""
    model.eval()
    criterion.eval()
    
    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_bbox = 0.0
    total_loss_giou = 0.0
    
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        total_loss += losses.item()
        total_loss_ce += loss_dict.get('loss_ce', torch.tensor(0.0)).item()
        total_loss_bbox += loss_dict.get('loss_bbox', torch.tensor(0.0)).item()
        total_loss_giou += loss_dict.get('loss_giou', torch.tensor(0.0)).item()

    return {
        'val_loss': total_loss / len(data_loader),
        'val_ce': total_loss_ce / len(data_loader),
        'val_bbox': total_loss_bbox / len(data_loader),
        'val_giou': total_loss_giou / len(data_loader)
    }

def main(args):
    """Main training execution function."""
    # 1. Device and Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Build Model and Criterion
    print("Building DINO model...")
    model = build_dino(args)
    model.to(device)

    # Setup the Hungarian Matcher
    matcher = HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou
    )

    # Define the base weights for our three losses
    weight_dict = {
        'loss_ce': args.set_cost_class,
        'loss_bbox': args.set_cost_bbox,
        'loss_giou': args.set_cost_giou
    }

    # If using auxiliary losses (which DINO always should), copy the weights for every layer
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # Build the Loss Criterion
    losses = ['labels', 'boxes']
    criterion = SetCriterion(
        args.num_classes, 
        matcher, 
        weight_dict, 
        args.focal_alpha, 
        losses
    )
    criterion.to(device)

    # 3. Datasets and DataLoaders
    print("Setting up data loaders...")
    
    # Build transforms (Make sure your transforms.py is ready for this)
    train_transforms = build_transforms(is_train=True)
    val_transforms = build_transforms(is_train=False)

    # Initialize SVHN datasets
    dataset_train = DigitDetectionDataset(args.train_img_dir, args.train_ann_file, transforms=train_transforms)
    dataset_val = DigitDetectionDataset(args.val_img_dir, args.val_ann_file, transforms=val_transforms)

    # Create PyTorch dataloaders
    # Using num_workers=4 is usually safe for an A5000, adjust if you hit CPU bottlenecks
    data_loader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=collate_fn,
        drop_last=True # Drops the last incomplete batch to stabilize BatchNorm/gradients
    )
    
    data_loader_val = DataLoader(
        dataset_val, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    # 4. Optimizer and Learning Rate Scheduler
    print("Setting up optimizer...")
    
    # Separate backbone parameters from transformer parameters for different LRs
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    # AdamW is the standard optimizer for DETR variants
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # StepLR will multiply the LR by 0.1 when it hits args.lr_drop (e.g., epoch 20)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # 5. The Epoch Loop and Logging
    print("Start training")
    
    # Define paths
    log_path = os.path.join(args.output_dir, "log.txt")
    best_val_loss = float('inf')
    start_epoch = 0

    # RESUME LOGIC: Check for checkpoint.pth to pick up where we left off
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        
        # Recover best_val_loss from logs so we don't overwrite best_model incorrectly
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    for line in f.readlines():
                        data = json.loads(line)
                        if data.get('val_loss', float('inf')) < best_val_loss:
                            best_val_loss = data['val_loss']
            except Exception as e:
                print(f"Could not parse best_val_loss from logs: {e}")

    # Iterate through remaining epochs
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch (Passing accum_steps!)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args.accum_steps
        )
        
        # Step the learning rate scheduler
        lr_scheduler.step()
        
        # Evaluate on the validation set
        val_stats = evaluate(
            model, criterion, data_loader_val, device
        )
        
        # Combine statistics for logging
        log_stats = {**{f'{k}': v for k, v in train_stats.items()},
                     **{f'{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}

        # LOG DUMP: Write to log.txt
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_stats) + "\n")

        # Checkpoint Saving Logic
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, checkpoint_path)
        
        # Save the "best" model specifically
        if val_stats['val_loss'] < best_val_loss:
            best_val_loss = val_stats['val_loss']
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, best_model_path)
            print(f"--> Saved new best model at epoch {epoch} with val_loss: {best_val_loss:.4f}")

    print("Training complete!")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

