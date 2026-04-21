import sys
import os
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_utils.dataset import DigitDetectionDataset
from data_utils.transforms import build_transforms
from models.dino import build_dino

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    val_transforms = build_transforms(is_train=False, image_size=(448, 448))
    
    val_dataset = DigitDetectionDataset(
        'dataset/valid', 
        'dataset/valid.json', 
        transforms=val_transforms
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    import argparse
    args = argparse.Namespace(
        hidden_dim=256, 
        nheads=8, 
        enc_layers=6,     
        dec_layers=6,     
        num_queries=300,  # CRITICAL: Match Version C training
        aux_loss=True,
        lr_backbone=0.0,
        backbone='resnet50',
        dilation=True,
        position_embedding='sine',
        masks=False,
        num_classes=10
    )
    model = build_dino(args)
    
    # CRITICAL: Point to Version C directory
    checkpoint_path = 'checkpoints/run_dino/best_model.pth'
    #checkpoint_path = 'checkpoints/run_dino/checkpoint_ep35.pth'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'checkpoints/run_dino/checkpoint.pth'
        if not os.path.exists(checkpoint_path):
            print("Error: No checkpoint found in checkpoints/run_dino/")
            return
            
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

    print("Running inference...")
    printed_debug = False
    
    with torch.no_grad():
        for samples, targets in tqdm(val_loader):
            samples = samples.to(device)
            outputs = model(samples)

            pred_logits = outputs['pred_logits'][:, :, :10]
            prob = pred_logits.sigmoid()
            scores, labels = prob.max(dim=-1)
            boxes = box_cxcywh_to_xyxy(outputs['pred_boxes']) * 448.0

            preds = []
            target_list = []
            
            for i in range(len(targets)):
                b = boxes[i]
                s = scores[i]
                l = labels[i]
                
                # CRITICAL: Match inference threshold and NMS
                keep_conf = s > 0.01
                b = b[keep_conf]
                s = s[keep_conf]
                l = l[keep_conf]
                
                if len(b) > 0:
                    keep_nms = nms(b, s, iou_threshold=0.6)
                    b = b[keep_nms]
                    s = s[keep_nms]
                    l = l[keep_nms]

                if not printed_debug:
                    print("\n--- DEBUG: FIRST IMAGE PREDICTIONS ---")
                    print(f"Top 3 Scores : {s[:3].cpu().numpy()}")
                    print(f"Top 3 Labels : {l[:3].cpu().numpy()}")
                    print(f"Top 3 Boxes  : \n{b[:3].cpu().numpy()}")
                    print("--------------------------------------\n")
                    printed_debug = True

                preds.append({
                    "boxes": b.cpu(),
                    "scores": s.cpu(),
                    "labels": l.cpu()
                })
                
                gt_boxes = box_cxcywh_to_xyxy(targets[i]['boxes']) * 448.0
                
                target_list.append({
                    "boxes": gt_boxes.cpu(),
                    "labels": targets[i]['labels'].cpu()
                })
                
            metric.update(preds, target_list)

    print("Computing mAP...")
    result = metric.compute()
    
    print("\n" + "="*40)
    print("DINO EVALUATION RESULTS")
    print("="*40)
    print(f"mAP @ IoU=0.50:0.95 : {result['map']:.4f}")
    print(f"mAP @ IoU=0.50      : {result['map_50']:.4f}")
    print(f"mAP @ IoU=0.75      : {result['map_75']:.4f}")
    print(f"mAP (Small Objects) : {result['map_small']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()