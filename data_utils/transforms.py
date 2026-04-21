"""Data augmentation module using Albumentations - Optimized for DINO on SVHN."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(is_train: bool, image_size: tuple = (448, 448)):
# def build_transforms(is_train: bool, image_size: tuple = (384, 384)):
# def build_transforms(is_train: bool, image_size: tuple = (544, 544)):
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_train:
        return A.Compose([
            # -----------------------------------------------------------------
            # 1. MULTI-SCALE TRAINING (MST) & STATIC CANVAS
            # -----------------------------------------------------------------
            # Randomly select a base scale for the digits to force scale invariance
            A.OneOf([
                A.LongestMaxSize(max_size=384),
                A.LongestMaxSize(max_size=416),
                A.LongestMaxSize(max_size=448),
                A.LongestMaxSize(max_size=480),
                A.LongestMaxSize(max_size=512),
            ], p=1.0),
            
            # Pad the remaining space to a static 544x544 canvas. 
            # This keeps the torch.stack() in train_dino.py collate_fn from crashing.
            A.PadIfNeeded(min_height=544, min_width=544, 
                          border_mode=0, fill=(0,0,0)),

            # -----------------------------------------------------------------
            # 2. LARGE SCALE JITTER (LSJ)
            # -----------------------------------------------------------------
            A.Affine(
                scale=(0.6, 1.2), 
                translate_percent=0.1, 
                rotate=(-15, 15), 
                cval=(0,0,0),
                p=0.7
            ),

            # -----------------------------------------------------------------
            # 3. EXTREME PHOTOMETRIC DISTORTIONS (SOTA Color Invariance)
            # -----------------------------------------------------------------
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.8),
            
            # --- THE SOTA SVHN FIXES ---
            # Random inversion forces the model to learn geometric edges rather than color
            A.InvertImg(p=0.5), 
            # Reduces color gradient depth, mimicking low-quality camera sensors
            A.Posterize(num_bits=4, p=0.2), 

            # -----------------------------------------------------------------
            # 4. CAMERA ARTIFACTS
            # -----------------------------------------------------------------
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            ], p=0.4),

            # -----------------------------------------------------------------
            # 5. COARSE DROPOUT
            # -----------------------------------------------------------------
            A.CoarseDropout(
                num_holes_range=(1, 4), 
                hole_height_range=(1, 12), 
                hole_width_range=(1, 12),
                fill_value=0,
                p=0.4
            ),

            normalize,
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['class_labels'],
            min_visibility=0.3, 
            clip=True           
        ))
        
    return A.Compose([
        # -----------------------------------------------------------------
        # VALIDATION / INFERENCE PIPELINE
        # -----------------------------------------------------------------
        # This MUST strictly remain at 448x448 to ensure the un-normalization math 
        # in inference_dino.py and val_map.py remains 100% accurate.
        A.LongestMaxSize(max_size=image_size[0]),
        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1], 
                      border_mode=0, fill=(0,0,0)),
        
        # A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=0, p=1.0), # Add this line
        # A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.2, 0.2), p=1.0),

        normalize,
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='coco', 
        label_fields=['class_labels'], 
        clip=True
    ))