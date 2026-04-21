import json
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from collections import defaultdict

print("Loading predictions for WBF...")
# Ensure these match your CodaBench test predictions
with open('pred_bright.json', 'r') as f:
    preds1 = json.load(f)
with open('pred_040.json', 'r') as f: # Original 448 predictions (0.40 on codabench)
    preds2 = json.load(f)
with open('pred_contrast.json', 'r') as f:
    preds3 = json.load(f)

# Group predictions by image_id
# We don't group by category yet, because WBF handles multiple classes automatically
grouped1 = defaultdict(list)
for p in preds1: grouped1[p['image_id']].append(p)

grouped2 = defaultdict(list)
for p in preds2: grouped2[p['image_id']].append(p)

grouped3 = defaultdict(list)
for p in preds3: grouped3[p['image_id']].append(p)

all_image_ids = set(grouped1.keys()).union(set(grouped2.keys())).union(set(grouped3.keys()))

final_results = []

# WBF Hyperparameters
# 0.60 is the standard WBF starting point (it acts differently than NMS)
iou_thr = 0.60 
skip_box_thr = 0.0001
weights = [1, 1, 1] # Treat all three scales equally

# SVHN images are varying sizes, but if your inference script outputted
# coordinates relative to the original image size, we need a rough normalizer.
# WBF requires coordinates between 0 and 1. 
# We use a large arbitrary constant to normalize, then multiply it back.
NORM_CONST = 1000.0 

print("Applying Weighted Boxes Fusion...")

for img_id in all_image_ids:
    boxes_list = []
    scores_list = []
    labels_list = []
    
    for epoch_preds in [grouped1[img_id], grouped2[img_id], grouped3[img_id]]:
        ep_boxes, ep_scores, ep_labels = [], [], []
        for p in epoch_preds:
            x, y, w, h = p['bbox']
            # Convert to [x1, y1, x2, y2] and normalize to [0, 1]
            x1, y1 = x / NORM_CONST, y / NORM_CONST
            x2, y2 = (x + w) / NORM_CONST, (y + h) / NORM_CONST
            
            ep_boxes.append([x1, y1, x2, y2])
            ep_scores.append(p['score'])
            ep_labels.append(p['category_id'])
            
        boxes_list.append(ep_boxes)
        scores_list.append(ep_scores)
        labels_list.append(ep_labels)

    # Run WBF
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, 
        weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    
    # Denormalize and convert back to JSON format
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        x1, y1 = x1 * NORM_CONST, y1 * NORM_CONST
        x2, y2 = x2 * NORM_CONST, y2 * NORM_CONST
        w, h = x2 - x1, y2 - y1
        
        final_results.append({
            "image_id": img_id,
            "category_id": int(label),
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "score": round(float(score), 4)
        })

with open('pred_wbf.json', 'w') as f:
    json.dump(final_results, f, indent=4)

print("WBF complete! Rename 'pred_wbf.json' to 'pred.json' and submit.")