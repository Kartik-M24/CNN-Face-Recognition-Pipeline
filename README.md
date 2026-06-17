# CNN Face Recognition Pipeline

## Brief

PyTorch-based face recognition pipeline that trains a base classifier on classes A-H, performs transfer learning to extend the model to classes A-J, and runs live or image-based inference with top-3 predictions.

## Files

- `train.py` - base training script for classes A-H; saves `base_model.pth`
- `transfer.py` - transfer-learning script that adapts the base model to classes A-J; saves `final_model.pth`
- `inference.py` - webcam and still-image inference script using the final checkpoint
- `dataset/` - local dataset folder expected by the scripts
- `dataset/J/` - example/custom class folder currently present in the repository

Please note that the full dataset and generated checkpoint files are not included in the repository. They must be prepared locally or will be generated when you run the pipeline.

## Requirements

- Python 3.9 or above
- PyTorch
- NumPy
- OpenCV
- Pillow
- scikit-learn

Install dependencies with:

```bash
pip install torch numpy opencv-python pillow scikit-learn
```

## Quick Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy opencv-python pillow scikit-learn
```

## Dataset Setup

Expected dataset structure:

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

Recommended class counts:

- Classes A-H: 100 images per class
- Class I: 20 images
- Class J: 20 images

Dataset assumptions used by this project:

- Classes A-H come from the LFW dataset
- Class I also comes from LFW, but with fewer samples
- Class J is your own custom dataset

If classes A-H are not already available locally, `train.py` can fetch them from the configured `--base_url` when `--data_dir` is omitted. `transfer.py` can also try to fetch missing A-I classes unless `--no_download_missing` is used. Class J still needs to exist locally in your dataset folder. Please note if you want to run yourself you might want to import the dataset yourself prior

## Using The Pipeline

Run the scripts in this order.

### 1. Train the base model on A-H

```bash
python train.py --data_dir dataset --out base_model.pth
```

Useful options:

- `--model compact_facenet_v2` or `--model resnet18`
- `--img_size 160`
- `--epochs 80`
- `--batch_size 16`

### 2. Transfer from A-H to A-J

```bash
python transfer.py --base_model base_model.pth --data_dir dataset --out final_model.pth
```

Useful options:

- `--head_epochs 25`
- `--finetune_epochs 30`
- `--no_replay`

### 3. Run inference

Live camera mode:

```bash
python inference.py --model final_model.pth --camera 0
```

Still-image mode:

```bash
python inference.py --model final_model.pth --image path/to/image.jpg --out_image pred.jpg
```

Useful inference options:

- `--threshold 0.0` to always force a label
- `--tta 2` for test-time augmentation averaging
- `--max_faces 1 for number of predictions to be set to 1

## Notes And Suggestions

- The scripts expect the dataset folders to exist locally before transfer and inference.
- `base_model.pth` and `final_model.pth` are generated during training and transfer; they are not committed to the repository.
- If OpenCV cannot open your default camera, try `--camera 1` or `--camera auto`.
- If you want faster experimentation, start with the default compact model before trying `resnet18`.
- If you are missing the LFW class folders locally, check the configured `--base_url` before running download-enabled modes.

## Detailed Description

This project implements a complete face-recognition pipeline in three stages:

1. A base classifier is trained on classes A-H.
2. Transfer learning expands the classifier to include classes I and J.
3. Inference runs on webcam frames or still images and returns top predictions.

The goal is to build a self-contained workflow that starts with stronger base classes, then tests adaptation to low-data and custom classes.

## What The Code Does

### `train.py`

- Loads classes A-H from the dataset folder or downloads them from `--base_url`
- Applies preprocessing such as EXIF correction, RGB conversion, optional face cropping, optional CLAHE, resizing, and normalization
- Applies training augmentations including random resized crops, flips, rotation, color jitter, blur, grayscale, and random erasing
- Splits data by class into train, validation, and test subsets
- Trains a CNN classifier from random initialization
- Prints accuracy, F1, classification report, and confusion matrix
- Saves the trained checkpoint as `base_model.pth`

### `transfer.py`

- Loads `base_model.pth`
- Expands the classifier head from A-H to A-J when needed
- Uses class-aware augmentation, with stronger augmentation for new classes I and J
- Supports replay from base classes to reduce forgetting
- Runs head warmup followed by selective fine-tuning
- Prints validation metrics, classification report, and confusion matrix
- Saves the transferred checkpoint as `final_model.pth`

### `inference.py`

- Loads `final_model.pth`
- Detects faces with an OpenCV Haar cascade
- Applies optional CLAHE and margin-based face cropping
- Uses test-time augmentation for more stable predictions
- Returns top-3 predictions per detected face
- Supports confidence thresholding, prototype-assisted scoring, and temporal smoothing
- Can annotate and save still-image predictions or run live webcam inference

## How It Works

### Preprocessing

- EXIF orientation correction is applied to input images
- Images are converted to RGB
- Face crops are extracted with a Haar cascade when enabled
- CLAHE is used to normalize contrast when enabled
- Images are resized to the configured square input size
- Pixel values are normalized using mean/std `[0.5, 0.5, 0.5]`

### Base Training

- Default model: `compact_facenet_v2`
- Alternative model: `resnet18`
- Default image size: `160`
- Default embedding dimension: `128`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- Loss: cross-entropy with label smoothing
- Early stopping and balanced sampling are used to stabilize training on small datasets

### Transfer Learning

- The saved base checkpoint is loaded and reused
- The classifier layer is expanded to include classes I and J
- Training is split into head warmup and fine-tuning stages
- Replay from A-H can be kept or disabled with `--no_replay`
- Missing A-I classes can optionally be downloaded from the configured `--base_url`

### Inference

- Faces are detected from webcam frames or still images
- Predictions are averaged across test-time augmentation views
- Prototype fusion can refine class scoring when prototypes are available in the checkpoint
- Temporal smoothing is applied to the largest face during live inference
- The script displays or saves annotated predictions with top-3 labels and confidences

## Augmentation, Hyperparameters, And Why They Were Chosen

### Base training defaults in `train.py`

- Model: `compact_facenet_v2`
- Optional backbone: `resnet18`
- Image size: `160`
- Embedding dimension: `128`
- Epochs: `80`
- Batch size: `16`
- Learning rate: `8e-4`
- Weight decay: `1e-3`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- Loss: `CrossEntropyLoss(label_smoothing=0.05)`
- Early stopping patience: `18`
- Balanced sampler target: `96` samples per class per epoch
- Gradient clipping: `2.0`

Base augmentations:

- Random resized crop
- Horizontal flip
- Small random rotation
- Color jitter
- Occasional Gaussian blur
- Occasional grayscale conversion
- Random erasing

Why these choices were made:

- The dataset is small, so stronger augmentation helps reduce overfitting and improves robustness to pose, lighting, and framing changes.
- `compact_facenet_v2` is the default because it is lighter and more practical for smaller datasets than a larger backbone.
- `AdamW` and moderate weight decay help stabilize optimization while discouraging the model from memorizing small class-specific details.
- Label smoothing reduces overconfident predictions, which is useful when classes have limited images.
- Balanced sampling keeps larger and smaller classes from dominating the training signal.
- Early stopping limits wasted epochs once validation performance plateaus.

### Transfer-learning defaults in `transfer.py`

- Total epoch budget: `55`
- Head warmup epochs: `25`
- Fine-tune epochs: `30`
- Batch size: `8`
- Head learning rate: `6e-4`
- Fine-tune learning rate: `2e-4`
- Weight decay: `1e-3`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- Loss: `CrossEntropyLoss(label_smoothing=0.05)`
- Balanced sampler target: `120` samples per class per epoch
- Gradient clipping: `2.0`

Transfer augmentations:

- Weaker augmentation path for replay/base classes A-H
- Stronger augmentation path for the new classes I and J
- Random resized crop
- Horizontal flip
- Rotation
- Color jitter
- Occasional blur
- Random erasing

Why these choices were made:

- Transfer learning is split into head warmup and fine-tuning so the new classifier layer can adapt first before deeper weights are updated.
- The lower fine-tuning learning rate reduces the risk of destroying features learned during base training.
- Stronger augmentation for I and J compensates for the fact that these classes have much less data.
- Replay from A-H is included to reduce catastrophic forgetting when the model is expanded to A-J.
- The smaller batch size is a practical choice because transfer learning uses a deeper checkpointed model and more varied augmentation.

### Inference defaults in `inference.py`

- Confidence threshold: loaded from checkpoint by default, typically around `0.28`
- Test-time augmentation views: `2`
- Prototype fusion weight: `0.45`
- Prototype temperature: `12.0`
- Temporal smoothing window: `4` frames
- Face crop margin: `0.28`

Why these choices were made:

- A relatively low confidence threshold is useful because the project includes low-data classes that may otherwise be rejected too often.
- Test-time augmentation improves stability without making inference too slow.
- Prototype fusion helps class scoring when a class has limited examples and embedding similarity is informative.
- Temporal smoothing reduces frame-to-frame flicker during live webcam inference.

## Context And Assumptions

- The pipeline is designed around ten classes labeled A-J
- Classes A-H are the stronger base classes
- Classes I and J are intentionally lower-data classes used to test adaptation
- Class J is assumed to be custom data rather than part of the original LFW source
- The project uses random initialization for base training rather than external pretrained weights

## Expected Outputs

After a full run, you should expect:

- `base_model.pth` - trained base checkpoint for classes A-H
- `final_model.pth` - transferred checkpoint for classes A-J
- printed accuracy, F1, classification reports, and confusion matrices during training and transfer
- annotated output images when using `inference.py --image`
- live top-3 predictions when using webcam inference
