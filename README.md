# Face Recognition Pipeline

This project contains a self-contained training, transfer-learning, and inference pipeline for face recognition classes A-J.

## Dataset Requirements (Important)

You must prepare/import your own dataset database/folders before running the full pipeline.

- Classes A-H: LFW dataset, 100 images per class.
- Class I: LFW dataset, 20 images.
- Class J: your own dataset, 20 images.

Required for setup:
- Import classes A-H into your dataset database/folders.
- Import class I and class J as well, each with 20 images.

Why this split is used:
- A-H provide a stronger base representation (more images per class).
- I and J are intentionally lower-data classes to evaluate behavior on less-seen data.
- Class J is custom data, so it helps test generalization to unseen/non-LFW images.

## Project Files
- `train.py`: Trains a base model for classes A-H from random initialization.
- `transfer.py`: Loads the base model and transfers to A-J.
- `inference.py`: Runs live camera or image inference with top-3 predictions.

## Requirements

Install dependencies with:

```bash
pip install torch torchvision numpy opencv-python pillow scikit-learn
```

## Dataset Layout

Expected structure:

```text
dataset/
  A/
  B/
  C/
  D/
  E/
  F/
  G/
  H/
  I/
  J/
```

Suggested minimum counts:
- `A`-`H`: 100 images each
- `I`: 20 images
- `J`: 20 images

The pipeline assumes these classes are already imported into your local dataset structure.

## How The Pipeline Works

1. `train.py` (base training):
  - Trains on A-H only.
  - Produces `base_model.pth` with model weights and metadata.

2. `transfer.py` (transfer learning):
  - Loads `base_model.pth`.
  - Expands/adapts the classifier to A-J using classes I and J (and optional replay from A-H).
  - Produces `final_model.pth`.

3. `inference.py` (evaluation/inference):
  - Loads `final_model.pth`.
  - Runs predictions on live camera or images.
  - Useful for checking performance on I and J, including unseen/custom-style data.

## Hyperparameters Summary

Base training (`train.py`) defaults:
- Model: `compact_facenet_v2` (optional: `resnet18`)
- Image size: `160`
- Embedding dimension: `128`
- Epochs: `80`
- Batch size: `16`
- Optimizer: `AdamW`
- Learning rate: `8e-4`
- Weight decay: `1e-3`
- Scheduler: `CosineAnnealingLR` (`eta_min = lr * 0.05`)
- Loss: `CrossEntropyLoss(label_smoothing=0.05)`
- Gradient clipping: `2.0`
- Early stopping patience: `18`
- Balanced sampler target: `96` samples per class per epoch

Transfer learning (`transfer.py`) defaults:
- Stage 1 (head warmup): `25` epochs, `lr_head=6e-4`
- Stage 2 (last-stage fine-tune): `30` epochs, `lr_finetune=2e-4`
- Batch size: `8`
- Optimizer: `AdamW`
- Weight decay: `1e-3`
- Loss: `CrossEntropyLoss(label_smoothing=0.05)`
- Gradient clipping: `2.0`
- Stage scheduler: `CosineAnnealingLR` per stage
- Balanced sampler target: `120` samples per class per epoch

Inference (`inference.py`) defaults:
- Confidence threshold: checkpoint value (typically `0.28`)
- TTA: `2` views (original + horizontal flip)
- Prototype fusion: weight `0.45`, temperature `12.0`
- Temporal smoothing (largest face): `4` frames

## Augmentations And Preprocessing

Common preprocessing:
- EXIF orientation correction
- RGB conversion
- Face crop with Haar cascade (enabled by default)
- CLAHE contrast normalization (enabled by default)
- Resize to square input size
- Tensor normalization with mean/std `[0.5, 0.5, 0.5]`

Base training augmentations (`train.py`):
- Random resized crop
- Horizontal flip
- Small random rotation
- Color jitter (brightness/contrast/saturation)
- Occasional Gaussian blur
- Occasional grayscale conversion
- Random erasing

Transfer augmentations (`transfer.py`):
- Two augmentation strengths:
  - Weaker path for replay/base classes (A-H)
  - Stronger path for new classes (I/J)
- Includes the same augmentation families as base training with class-aware strength

Inference-time preprocessing (`inference.py`):
- Face detection + margin crop
- Optional CLAHE
- Test-time augmentation averaging
- Optional prototype-assisted scoring

## Model Architectures

Two self-contained model options are implemented in code:

1. `compact_facenet_v2`
- Lightweight CNN with residual blocks
- Embedding head + classifier
- Designed for smaller datasets and faster experimentation

2. `resnet18`
- Standard 4-stage residual architecture (`[2,2,2,2]` blocks)
- Larger capacity than the compact model

Both architectures are implemented directly inside this repository (no external pretrained model API required).

## No Pretrained Weights: Why

This pipeline intentionally starts from random initialization for base training.

Reasons:
- Fairness and reproducibility for coursework evaluation (same starting point for all teams)
- Avoids dependency on external pretrained sources and version mismatch issues
- Keeps the training/inference stack fully self-contained
- Makes transfer-learning improvements attributable to your own dataset design and pipeline choices

Implication:
- Training may require more epochs than an ImageNet-pretrained setup, but behavior is easier to audit and reproduce end-to-end.


## 1) Train Base Model (A-H)

Example:

```bash
python train.py --data_dir dataset --out base_model.pth
```

Useful options:
- `--model compact_facenet_v2|resnet18`
- `--img_size 160`
- `--epochs 80`
- `--batch_size 16`
- `--no_face_crop`
- `--no_clahe`

## 2) Transfer to A-J

Example:

```bash
python transfer.py --base_model base_model.pth --data_dir dataset --out final_model.pth
```

Useful options:
- `--head_epochs 25`
- `--finetune_epochs 30`
- `--no_replay` (train with I/J only)
- `--no_download_missing` (skip fetching missing classes)

## 3) Run Inference

Live camera:

```bash
python inference.py --model final_model.pth --camera 0
```

Image mode:

```bash
python inference.py --model final_model.pth --image path/to/image.jpg --out_image pred.jpg
```

## Checkpoint Compatibility

The scripts support:
- Project checkpoints with metadata (`state_dict`, `class_names`, preprocessing details).
- Legacy/plain `state_dict` checkpoints.

