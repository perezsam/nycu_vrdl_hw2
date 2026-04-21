"""Dataset module for loading digit images and bounding box annotations."""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class DigitDetectionDataset(Dataset):
    """Custom dataset for digit detection."""

    def __init__(self, image_dir: str, annotation_file: str, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            
        self.images = {img['id']: img for img in data['images']}
        
        self.annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
            
        self.image_ids = list(self.images.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        file_name = img_info['file_name'] 
        
        img_path = os.path.join(self.image_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        
        img_h, img_w = image_np.shape[:2]
        
        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        
        for ann in anns:
            x_min, y_min, w, h = ann['bbox']
            
            # Ensure coordinates do not exceed actual image dimensions
            x_min = max(0.0, float(x_min))
            y_min = max(0.0, float(y_min))
            w = min(float(w), img_w - x_min - 1e-4)
            h = min(float(h), img_h - y_min - 1e-4)
            
            if w > 0 and h > 0:
                boxes.append([x_min, y_min, w, h])
                labels.append(ann['category_id'])
            
        if self.transforms is not None:
            transformed = self.transforms(image=image_np, bboxes=boxes, class_labels=labels)
            image_np = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
            
        if not isinstance(image_np, torch.Tensor):
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        else:
            image_tensor = image_np

        # Convert to DETR format: normalized [cx, cy, w, h]
        _, tr_h, tr_w = image_tensor.shape
        target_boxes = []
        
        for box in boxes:
            x_min, y_min, w, h = box
            cx = (x_min + (w / 2.0)) / tr_w
            cy = (y_min + (h / 2.0)) / tr_h
            bw = w / tr_w
            bh = h / tr_h
            target_boxes.append([cx, cy, bw, bh])

        # THE FIX: Shift category_id (1-10) to DETR labels (0-9)
        labels_shifted = [int(l) - 1 for l in labels]
            
        targets = {
            "boxes": torch.tensor(target_boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels_shifted, dtype=torch.int64), # CRITICAL: Applied shift
            "image_id": torch.tensor([img_id]),
            "file_name": file_name 
        }
        
        return image_tensor, targets