"""Training script for the DINO/Deformable DETR architecture.

This script handles the instantiation of the multi-scale model, 
the focal loss criterion, and the main training loops.
"""

import os
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
    
    # Stack images into a single tensor [Batch, Channel, Height, Width]
    # NOTE: This assumes all images in your batch are resized to the same dimensions
    images = torch.stack(images, dim=0)
    
    return images, targets
    
def get_args_parser():
    """Parses training arguments specific to DINO."""
    parser = argparse.ArgumentParser('DINO training script', add_help=False)
    
    # Training Parameters (Optimized for DINO)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    # Batch size 16 is safer for 24GB VRAM with multi-scale features
    parser.add_argument('--batch_size', default=16, type=int) 
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    # DINO converges much faster than vanilla DETR; 24 epochs is usually plenty
    parser.add_argument('--epochs', default=24, type=int) 
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='Gradient clipping max norm to prevent explosion')

    # Model Parameters
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    
    # Loss & Matcher Parameters (Using Focal Loss)
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
                    device: torch.device, epoch: int, max_norm: float = 0):
    """Trains the model for one single epoch."""
    model.train()
    criterion.train()
    
    # Tracking metrics
    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_bbox = 0.0
    total_loss_giou = 0.0
    
    # Progress bar setup
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    
    for samples, targets in pbar:
        # Move inputs to device
        samples = samples.to(device)
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # Forward pass
        outputs = model(samples)
        
        # Calculate loss mapping predictions to targets
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # Sum all losses (including auxiliary layers) multiplied by their respective weights
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Catch exploding gradients early
        import math
        if not math.isfinite(losses.item()):
            print(f"Loss is {losses.item()}, stopping training")
            print(loss_dict)
            sys.exit(1)

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        
        # Gradient clipping - crucial for Deformable DETR stability
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
        optimizer.step()

        # Update metrics (extracting only the final layer stats for logging clarity)
        total_loss += losses.item()
        total_loss_ce += loss_dict.get('loss_ce', torch.tensor(0.0)).item()
        total_loss_bbox += loss_dict.get('loss_bbox', torch.tensor(0.0)).item()
        total_loss_giou += loss_dict.get('loss_giou', torch.tensor(0.0)).item()
        
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
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
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
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

    # Iterate through epochs
    for epoch in range(args.epochs):
        # Train for one epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
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
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
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