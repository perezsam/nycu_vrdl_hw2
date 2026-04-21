# NYCU Computer Vision 2026 HW2
**Student ID:** 314540033

**Name:** Samuel Perez 培雷斯

## Introduction
This repository contains the training, evaluation, and inference pipeline for the VRDL Homework 2 Digit Detection task on the Street View House Numbers (SVHN) dataset. 

The core architecture utilizes **DINO (DETR with Improved DeNoising Anchor Boxes)** constrained to a pre-trained **ResNet-50 backbone**. To maximize performance and address extreme scale variance, the pipeline features multi-scale padding, severe photometric augmentations, intermediate auxiliary decoding losses, and a temporal ensemble strategy utilizing **Weighted Boxes Fusion (WBF)** across peak and converged training checkpoints.

## Environment Setup
It is recommended to use a virtual environment (e.g., Conda) with Python 3.10+.

```bash
# Create and activate environment
conda create -n vrdl_hw2 python=3.10 -y
conda activate vrdl_hw2

# Install required dependencies
pip install -r requirements.txt
```


## Repository Structure
* `models/`: Contains the DINO architecture modules (`dino.py`, `backbone.py`, `deformable_transformer.py`, `loss.py`).
* `data_utils/`: Contains the SVHN dataset loader (`dataset.py`) and Albumentations pipeline (`transforms.py`).
* `train_dino.py` / `train_dino_accum.py`: Native and gradient-accumulated training scripts.
* `val_map.py`: Local validation evaluation using `torchmetrics`.
* `inference_dino.py`: Generates the raw prediction JSON files for the hidden test set.
* `ensemble_wbf.py`: Fuses multiple checkpoint predictions to maximize bounding box localization.

## Usage

### 1. Training
The model is trained for 36 epochs with a step learning rate drop at Epoch 30. Depending on your GPU VRAM availability, you can use either the standard script or the gradient accumulation script to maintain the target **effective batch size of 12**.

**For High-VRAM GPUs (e.g., RTX A5000):**
```bash
python train_dino.py \
    --output_dir checkpoints/run_dino \
    --batch_size 12 \
    --epochs 36 \
    --lr_drop 30 \
    --num_queries 300 \
    --dilation \
    --aux_loss
```

**For Constrained-VRAM GPUs (e.g., RTX A4000):**
```bash
python train_dino_accum.py \
    --output_dir checkpoints/run_dino \
    --batch_size 4 \
    --accum_steps 3 \
    --epochs 36 \
    --lr_drop 30 \
    --num_queries 300 \
    --dilation \
    --aux_loss
```

### 2. Local Validation
To evaluate the local mAP on the validation set, point the evaluation script to your generated checkpoint. The script utilizes `torchmetrics` for strict COCO-style mAP calculation.

```bash
python val_map.py --checkpoint checkpoints/run_dino/best_model.pth
```

### 3. Inference & Ensembling (WBF)
To reproduce the final CodaBench submission, we utilize a temporal ensembling strategy, fusing the predictions of the Peak Validation model (Epoch 30) and the Deep Convergence model (Epoch 36).

**Generate predictions for both checkpoints:**
```bash
# Generate predictions for Epoch 30
python inference_dino.py --checkpoint checkpoints/run_dino/best_model.pth --output preds_ep30.json

# Generate predictions for Epoch 36
python inference_dino.py --checkpoint checkpoints/run_dino/checkpoint.pth --output preds_ep36.json
```

**Fuse the predictions using Weighted Boxes Fusion:**
```bash
python ensemble_wbf.py \
    --input1 preds_ep30.json \
    --input2 preds_ep36.json \
    --output final_submission.json
```

## Performance
The final temporal WBF ensemble of the DINO pipeline achieved a private leaderboard score of **0.40 mAP** on CodaBench.

![CodaBench Final Score](snapshot.png)
