"""
train.py

Purpose:
- Train the base classifier for classes A-H.

Dataset assumptions for this stage:
- Classes A-H are from the LFW dataset, 100 images per class.
- These images are not bundled by this script for local/offline use; you must import
    A-H yourself into your own dataset database/folders before training.
- The transfer stage expects class I and class J to also exist in your database,
    each with 20 images.

Pipeline role:
- This is stage 1 of the pipeline (base training).
- It outputs a base checkpoint used by transfer.py to extend to A-J.
"""

import argparse
import copy
import math
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

BASE_CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H"]
CLASS_NAMES = BASE_CLASS_NAMES
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover for older Pillow
    RESAMPLE_BILINEAR = Image.BILINEAR

_face_cascade = None


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch for reproducible data splits and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def safe_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def get_face_cascade() -> cv2.CascadeClassifier:
    """Lazily construct and cache the Haar cascade detector."""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            raise RuntimeError("Could not load OpenCV Haar face cascade.")
    return _face_cascade


def center_square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = max(0, (w - side) // 2)
    top = max(0, (h - side) // 2)
    return img.crop((left, top, left + side, top + side))


def face_crop_pil(img: Image.Image, margin_ratio: float = 0.33) -> Image.Image:
    """Crop the largest detected face. Fall back to a center square crop."""
    img = ImageOps.exif_transpose(img.convert("RGB"))
    w, h = img.size
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = get_face_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.06, minNeighbors=4, minSize=(35, 35))
    if len(faces) == 0:
        return center_square_crop(img)
    x, y, fw, fh = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
    margin = int(max(fw, fh) * margin_ratio)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + fw + margin)
    y2 = min(h, y + fh + margin)
    return img.crop((x1, y1, x2, y2))


def apply_clahe_pil(img: Image.Image) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    merged = cv2.merge((l_chan, a_chan, b_chan))
    out_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)


def resize_square(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), RESAMPLE_BILINEAR)


def random_resized_crop(img: Image.Image, scale=(0.76, 1.0), ratio=(0.90, 1.10)) -> Image.Image:
    """Torchvision-free random resized crop with safe fallback."""
    w, h = img.size
    area = w * h
    for _ in range(10):
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect = random.uniform(ratio[0], ratio[1])
        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))
        if 0 < crop_w <= w and 0 < crop_h <= h:
            left = random.randint(0, w - crop_w)
            top = random.randint(0, h - crop_h)
            return img.crop((left, top, left + crop_w, top + crop_h))
    return center_square_crop(img)


def color_jitter(img: Image.Image, brightness=0.22, contrast=0.22, saturation=0.12) -> Image.Image:
    if brightness > 0:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(1.0 - brightness, 1.0 + brightness))
    if contrast > 0:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.0 - contrast, 1.0 + contrast))
    if saturation > 0:
        img = ImageEnhance.Color(img).enhance(random.uniform(1.0 - saturation, 1.0 + saturation))
    return img


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def normalize_tensor(x: torch.Tensor, mean=NORM_MEAN, std=NORM_STD) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    return (x - mean_t) / std_t


def random_erasing(x: torch.Tensor, p=0.15, scale=(0.02, 0.09)) -> torch.Tensor:
    if random.random() > p:
        return x
    c, h, w = x.shape
    area = h * w
    for _ in range(10):
        erase_area = random.uniform(scale[0], scale[1]) * area
        aspect = random.uniform(0.35, 2.8)
        eh = int(round(math.sqrt(erase_area * aspect)))
        ew = int(round(math.sqrt(erase_area / aspect)))
        if 0 < eh < h and 0 < ew < w:
            y = random.randint(0, h - eh)
            x0 = random.randint(0, w - ew)
            x[:, y:y + eh, x0:x0 + ew] = 0.0
            return x
    return x


class FaceTransform:
    def __init__(self, img_size: int, train: bool, mean=NORM_MEAN, std=NORM_STD,
                 face_crop: bool = True, crop_margin: float = 0.33, clahe: bool = True):
        self.img_size = img_size
        self.train = train
        self.mean = mean
        self.std = std
        self.face_crop = face_crop
        self.crop_margin = crop_margin
        self.clahe = clahe

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = ImageOps.exif_transpose(img.convert("RGB"))
        if self.face_crop:
            img = face_crop_pil(img, self.crop_margin)

        if self.train:
            # Augment heavily during training to improve robustness on small datasets.
            img = random_resized_crop(img, scale=(0.74, 1.0), ratio=(0.88, 1.12))
            if random.random() < 0.5:
                img = ImageOps.mirror(img)
            angle = random.uniform(-9.0, 9.0)
            img = img.rotate(angle, resample=RESAMPLE_BILINEAR, fillcolor=(128, 128, 128))
            img = color_jitter(img, brightness=0.25, contrast=0.25, saturation=0.14)
            if random.random() < 0.10:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
            if random.random() < 0.05:
                img = ImageOps.grayscale(img).convert("RGB")

        if self.clahe:
            img = apply_clahe_pil(img)
        img = resize_square(img, self.img_size)
        tensor = pil_to_tensor(img)
        tensor = normalize_tensor(tensor, self.mean, self.std)
        if self.train:
            tensor = random_erasing(tensor, p=0.14, scale=(0.02, 0.08))
        return tensor


class FaceDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, class_name = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def retrieve_lfw_dataset(group: str, base_url: str, class_names, target_dir: Path) -> Path:
    print(f"Ensuring dataset exists at: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for cls in class_names:
        class_dir = target_dir / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        img_num = 1
        downloaded_or_found = 0
        while True:
            file_name = f"{cls}_{img_num:03d}.jpg"
            save_path = class_dir / file_name
            if save_path.exists():
                downloaded_or_found += 1
                img_num += 1
                continue
            url = f"{base_url.rstrip('/')}/{cls}/{file_name}"
            try:
                urllib.request.urlretrieve(url, save_path)
                downloaded_or_found += 1
                img_num += 1
            except HTTPError as err:
                if err.code == 404:
                    break
                raise
            except URLError as err:
                raise RuntimeError(f"Could not download {url}: {err}") from err
        print(f"  {cls}: {downloaded_or_found} image(s)")
        if downloaded_or_found == 0:
            raise RuntimeError(f"Class {cls} has no images. Check --base_url or --data_dir.")
    return target_dir


def list_image_files(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def list_samples(data_root: Path, class_names):
    samples = []
    counts = {}
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = data_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        files = list_image_files(class_dir)
        counts[class_name] = len(files)
        if len(files) == 0:
            raise RuntimeError(f"No images found in {class_dir}")
        for path in files:
            samples.append((str(path), class_to_idx[class_name], class_name))
    print("Class mapping:", class_to_idx)
    print("Per-class counts:", counts)
    return samples


def split_by_class(samples, val_fraction=0.15, test_fraction=0.15, seed=42):
    """Create per-class train/val/test splits with safeguards for tiny classes."""
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample[2]].append(sample)
    train, val, test = [], [], []
    for class_name, items in sorted(grouped.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        if n >= 6:
            n_test = max(1, int(round(n * test_fraction)))
            n_val = max(1, int(round(n * val_fraction)))
        elif n >= 3:
            n_test, n_val = 1, 1
        elif n == 2:
            n_test, n_val = 0, 1
        else:
            n_test, n_val = 0, 0
        n_test = min(n_test, max(0, n - 2))
        n_val = min(n_val, max(0, n - n_test - 1))
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])
    return train, val, test


def make_loader(dataset, batch_size, balanced=False, samples_per_class=96, num_workers=0, shuffle=True):
    pin = torch.cuda.is_available()
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin)
    counts = Counter(label for _, label, _ in dataset.samples)
    weights = [1.0 / counts[label] for _, label, _ in dataset.samples]
    num_samples = max(len(weights), len(counts) * samples_per_class)
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1):
        pad = kernel // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 3, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CompactFaceNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.model_name = "compact_facenet_v2"
        self.embedding_dim = embedding_dim
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, 3, 2),
            ConvBNAct(32, 48, 3, 1),
        )
        self.stage1 = nn.Sequential(ResidualBlock(48, 64, 2, 0.02), ResidualBlock(64, 64, 1, 0.02))
        self.stage2 = nn.Sequential(ResidualBlock(64, 128, 2, 0.03), ResidualBlock(128, 128, 1, 0.03))
        self.stage3 = nn.Sequential(ResidualBlock(128, 192, 2, 0.04), ResidualBlock(192, 192, 1, 0.04))
        self.stage4 = nn.Sequential(ResidualBlock(192, 256, 2, 0.05), ResidualBlock(256, 256, 1, 0.05))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(256, embedding_dim, bias=False)
        self.embed_bn = nn.BatchNorm1d(embedding_dim)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.embed(x)
        x = self.embed_bn(x)
        return x

    def forward(self, x, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.classifier(self.dropout(features))
        if return_features:
            return logits, F.normalize(features, dim=1)
        return logits


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=8):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.fc(features)
        if return_features:
            return logits, F.normalize(features, dim=1)
        return logits


def build_model(model_name: str, num_classes: int, embedding_dim: int = 128):
    name = (model_name or "compact_facenet_v2").lower()
    if name in {"compact", "compact_facenet", "compact_facenet_v2"}:
        return CompactFaceNet(num_classes=num_classes, embedding_dim=embedding_dim)
    if name == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    raise ValueError(f"Unknown model name: {model_name}")


def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=2.0):
    """Run one training or evaluation epoch and return summary metrics."""
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if train:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(yb.detach().cpu().tolist())
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(targets, preds) if targets else 0.0
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0) if targets else 0.0
    return avg_loss, acc, macro_f1, targets, preds


@torch.no_grad()
def compute_prototypes(model, loader, num_classes, device):
    """Compute class prototypes by averaging normalized feature embeddings."""
    model.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.long)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        _, feats = model(xb, return_features=True)
        feats = feats.detach().cpu()
        if sums is None:
            sums = torch.zeros(num_classes, feats.shape[1], dtype=torch.float32)
        for feat, label in zip(feats, yb):
            sums[int(label)] += feat
            counts[int(label)] += 1
    if sums is None:
        return None, None
    mask = counts > 0
    protos = torch.zeros_like(sums)
    protos[mask] = sums[mask] / counts[mask].float().unsqueeze(1)
    protos[mask] = F.normalize(protos[mask], dim=1)
    return protos, mask


def main():
    parser = argparse.ArgumentParser(description="Train the A-H base face model from random weights only.")
    parser.add_argument("--group", default="11")
    parser.add_argument("--data_dir", default=None, help="Folder containing A-H class subfolders. If omitted, downloads from --base_url.")
    parser.add_argument("--base_url", default="http://localhost/dataset/lfw_2026/")
    parser.add_argument("--out", default="base_model.pth")
    parser.add_argument("--model", default="compact_facenet_v2", choices=["compact_facenet_v2", "resnet18"])
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--samples_per_class", type=int, default=96, help="Balanced sampler draws about this many examples per class each epoch.")
    parser.add_argument("--no_face_crop", action="store_true")
    parser.add_argument("--no_clahe", action="store_true")
    parser.add_argument("--crop_margin", type=float, default=0.33)
    parser.add_argument("--patience", type=int, default=18)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Important: this script uses random initialization only. No pretrained weights are loaded.")

    if args.data_dir:
        data_root = Path(args.data_dir)
    else:
        data_root = Path.home() / "cs721" / f"group{args.group}" / "dataset"
        retrieve_lfw_dataset(args.group, args.base_url, CLASS_NAMES, data_root)

    # Build stable splits once so model selection remains fair and repeatable.
    samples = list_samples(data_root, CLASS_NAMES)
    train_samples, val_samples, test_samples = split_by_class(samples, seed=args.seed)
    print("Train counts:", dict(Counter(s[2] for s in train_samples)))
    print("Val counts:", dict(Counter(s[2] for s in val_samples)))
    print("Test counts:", dict(Counter(s[2] for s in test_samples)))

    train_tf = FaceTransform(args.img_size, train=True, face_crop=not args.no_face_crop,
                             crop_margin=args.crop_margin, clahe=not args.no_clahe)
    eval_tf = FaceTransform(args.img_size, train=False, face_crop=not args.no_face_crop,
                            crop_margin=args.crop_margin, clahe=not args.no_clahe)
    train_ds = FaceDataset(train_samples, train_tf)
    val_ds = FaceDataset(val_samples, eval_tf)
    test_ds = FaceDataset(test_samples, eval_tf)
    proto_ds = FaceDataset(samples, eval_tf)

    # Use balanced sampling to reduce class imbalance impact each epoch.
    train_loader = make_loader(train_ds, args.batch_size, balanced=True, samples_per_class=args.samples_per_class,
                               num_workers=args.num_workers, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, balanced=False, num_workers=args.num_workers, shuffle=False)
    test_loader = make_loader(test_ds, args.batch_size, balanced=False, num_workers=args.num_workers, shuffle=False)
    proto_loader = make_loader(proto_ds, args.batch_size, balanced=False, num_workers=args.num_workers, shuffle=False)

    model = build_model(args.model, num_classes=len(CLASS_NAMES), embedding_dim=args.embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)

    # Select the best checkpoint by validation macro-F1.
    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{args.epochs} | lr={lr_now:.2e} | train_loss={train_loss:.4f} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1, targets, preds = run_epoch(model, test_loader, criterion, device)
    print("\nFINAL TEST RESULTS")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test macro-F1: {test_f1:.4f}")
    print(classification_report(targets, preds, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, digits=4, zero_division=0))
    print("Confusion matrix rows=true, cols=pred:")
    print(confusion_matrix(targets, preds, labels=list(range(len(CLASS_NAMES)))))

    # Save prototypes for optional prototype-assisted scoring during inference.
    prototypes, proto_mask = compute_prototypes(model, proto_loader, len(CLASS_NAMES), device)
    out_path = Path(args.out)
    checkpoint = {
        "model_name": args.model,
        "state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "img_size": args.img_size,
        "embedding_dim": args.embedding_dim,
        "normalization": {"mean": NORM_MEAN, "std": NORM_STD},
        "preprocess": {"face_crop": not args.no_face_crop, "clahe": not args.no_clahe, "crop_margin": args.crop_margin},
        "prototypes": prototypes,
        "prototype_mask": proto_mask,
        "prototype_class_names": CLASS_NAMES,
        "confidence_threshold": 0.28,
        "score_mode": "softmax_plus_prototype",
        "best_val_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "best_config": vars(args),
    }
    safe_torch_save(checkpoint, out_path)
    print(f"Saved base checkpoint to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
