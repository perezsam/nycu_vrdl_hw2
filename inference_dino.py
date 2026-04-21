"""Inference script for DINO on SVHN using exact training transforms and NMS."""

import os
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torchvision.ops import nms

# Ensure local imports work cleanly
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.dino import build_dino
from data_utils.transforms import build_transforms

def get_args_parser():
    parser = argparse.ArgumentParser('DINO inference script', add_help=False)
    parser.add_argument('--test_dir', default='dataset/test', type=str)
    # Defaulting to the new Version C checkpoint directory
    parser.add_argument('--checkpoint', default='checkpoints/run_dino/best_model.pth', type=str)
    parser.add_argument('--output', default='pred.json', type=str)
    
    # ---------------------------------------------------------
    # CRITICAL FIX 1: Lower threshold to 0.01 to catch small digits
    # ---------------------------------------------------------
    parser.add_argument('--conf_threshold', default=0.01, type=float)
    
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--nheads', default=8, type=int)
    
    # ---------------------------------------------------------
    # CRITICAL FIX 2: Must match training queries (100 -> 300)
    # ---------------------------------------------------------
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--aux_loss', action='store_true')
    return parser

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on device: {device}")

    args.lr_backbone = 0.0
    args.masks = False
    if not hasattr(args, 'position_embedding'): args.position_embedding = 'sine'
    
    model = build_dino(args)
    print(f"Loading weights from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    val_transforms = build_transforms(is_train=False, image_size=(448, 448))

    image_paths = glob.glob(os.path.join(args.test_dir, "*.png"))
    results = []
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Detecting"):
            img_name = os.path.basename(img_path)
            
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
            img_h, img_w = image_np.shape[:2]
            
            transformed = val_transforms(image=image_np, bboxes=[], class_labels=[])
            img_tensor = transformed['image'].unsqueeze(0).to(device)
            
            outputs = model(img_tensor)
            
            logits = outputs['pred_logits'][0][:, :10]
            boxes = outputs['pred_boxes'][0]
            
            probs = logits.sigmoid()
            scores, labels = probs.max(-1)
            
            keep = scores > args.conf_threshold
            scores = scores[keep].cpu()
            labels = labels[keep].cpu()
            boxes = boxes[keep].cpu()
            
            if len(boxes) > 0:                

                # 1. Reverse Albumentations A.LongestMaxSize logic
                scale = 448.0 / max(img_h, img_w)
                if img_h > img_w:
                    new_h, new_w = 448, int(round(img_w * scale))
                else:
                    new_h, new_w = int(round(img_h * scale)), 448
                    
                # 2. Reverse Albumentations A.PadIfNeeded (center padding) logic
                pad_top = (448 - new_h) // 2
                pad_left = (448 - new_w) // 2
                
                # 3. Unnormalize DETR boxes to 448x448 canvas
                cx, cy, bw, bh = boxes.unbind(-1)
                cx, cy = cx * 448.0, cy * 448.0
                bw, bh = bw * 448.0, bh * 448.0
                
                x_min = cx - (bw / 2.0)
                y_min = cy - (bh / 2.0)
                
                x_min -= pad_left
                y_min -= pad_top
                
                x_min /= scale
                y_min /= scale
                bw /= scale
                bh /= scale
                
                x2 = x_min + bw
                y2 = y_min + bh
                
                x_min = x_min.clamp(min=0, max=img_w)
                y_min = y_min.clamp(min=0, max=img_h)
                x2 = x2.clamp(min=0, max=img_w)
                y2 = y2.clamp(min=0, max=img_h)
                
                final_w = x2 - x_min
                final_h = y2 - y_min
                
                valid_mask = (final_w > 0) & (final_h > 0)
                
                x_min = x_min[valid_mask]
                y_min = y_min[valid_mask]
                x2 = x2[valid_mask]
                y2 = y2[valid_mask]
                scores = scores[valid_mask]
                labels = labels[valid_mask] 
                

                # ---------------------------------------------------------
                # CRITICAL FIX 3: Non-Maximum Suppression (NMS)
                # ---------------------------------------------------------
                if len(scores) > 0:
                    nms_boxes = torch.stack([x_min, y_min, x2, y2], dim=1)
                    keep_idx = nms(nms_boxes, scores, iou_threshold=0.6)
                    
                    # 1. Do the math while x_min and y_min are still Tensors
                    final_w = (x2[keep_idx] - x_min[keep_idx]).tolist()
                    final_h = (y2[keep_idx] - y_min[keep_idx]).tolist()
                    
                    # 2. Now it is safe to overwrite them as Python lists
                    x_min = x_min[keep_idx].tolist()
                    y_min = y_min[keep_idx].tolist()
                    
                    scores = scores[keep_idx].tolist()
                    labels = labels[keep_idx].tolist()
                else:
                    x_min, y_min, final_w, final_h, scores, labels = [], [], [], [], [], []
                
                try:
                    image_id = int(os.path.splitext(img_name)[0])
                except ValueError:
                    image_id = img_name
                
                for x, y, w, h, score, label in zip(x_min, y_min, final_w, final_h, scores, labels):
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label) + 1,
                        "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                        "score": round(score, 4)
                    })

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Done! {args.output} is ready.")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)