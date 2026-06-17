"""
transfer.py

Purpose:
- Perform transfer learning from the A-H base model to the final A-J model.

Dataset assumptions for this stage:
- Classes A-H: LFW dataset, 100 images per class.
- Class I: LFW dataset, 20 images.
- Class J: your own dataset, 20 images.
- You must import all required classes (A-H, I, and J) into your own dataset
    database/folders before running this stage.

Pipeline role:
- This is stage 2 of the pipeline (transfer).
- Head warmup and selective fine-tuning are used so new classes are learned while
    preserving old-class performance.
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

CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
BASE_CLASS_NAMES = CLASS_NAMES[:8]
NEW_CLASS_NAMES = ["I", "J"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover
    RESAMPLE_BILINEAR = Image.BILINEAR

_face_cascade = None


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch so transfer runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def safe_torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def safe_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


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


def random_resized_crop(img: Image.Image, scale=(0.78, 1.0), ratio=(0.90, 1.10)) -> Image.Image:
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


def color_jitter(img: Image.Image, brightness=0.20, contrast=0.20, saturation=0.12) -> Image.Image:
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


def random_erasing(x: torch.Tensor, p=0.12, scale=(0.02, 0.08)) -> torch.Tensor:
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
                 face_crop: bool = True, crop_margin: float = 0.33, clahe: bool = True,
                 strong: bool = False):
        self.img_size = img_size
        self.train = train
        self.mean = mean
        self.std = std
        self.face_crop = face_crop
        self.crop_margin = crop_margin
        self.clahe = clahe
        self.strong = strong

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = ImageOps.exif_transpose(img.convert("RGB"))
        if self.face_crop:
            img = face_crop_pil(img, self.crop_margin)
        if self.train:
            # Apply class-aware augmentation strength (stronger for I/J).
            if self.strong:
                img = random_resized_crop(img, scale=(0.72, 1.0), ratio=(0.86, 1.14))
                angle = random.uniform(-10.0, 10.0)
                b, c, s = 0.28, 0.28, 0.16
                erase_p = 0.16
            else:
                img = random_resized_crop(img, scale=(0.82, 1.0), ratio=(0.92, 1.08))
                angle = random.uniform(-6.0, 6.0)
                b, c, s = 0.16, 0.16, 0.08
                erase_p = 0.08
            if random.random() < 0.5:
                img = ImageOps.mirror(img)
            img = img.rotate(angle, resample=RESAMPLE_BILINEAR, fillcolor=(128, 128, 128))
            img = color_jitter(img, brightness=b, contrast=c, saturation=s)
            if random.random() < (0.12 if self.strong else 0.06):
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
            if random.random() < 0.04:
                img = ImageOps.grayscale(img).convert("RGB")
        if self.clahe:
            img = apply_clahe_pil(img)
        img = resize_square(img, self.img_size)
        tensor = pil_to_tensor(img)
        tensor = normalize_tensor(tensor, self.mean, self.std)
        if self.train:
            tensor = random_erasing(tensor, p=erase_p, scale=(0.02, 0.08))
        return tensor


class FaceDataset(Dataset):
    def __init__(self, samples, base_transform, new_transform=None, new_classes=None):
        self.samples = list(samples)
        self.base_transform = base_transform
        self.new_transform = new_transform or base_transform
        self.new_classes = set(new_classes or NEW_CLASS_NAMES)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, class_name = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tf = self.new_transform if class_name in self.new_classes else self.base_transform
        return tf(img), label


def retrieve_lfw_classes(class_names, base_url: str, target_dir: Path, required_classes=None) -> None:
    required_classes = set(required_classes or [])
    target_dir.mkdir(parents=True, exist_ok=True)
    for cls in class_names:
        class_dir = target_dir / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        img_num = 1
        found = 0
        while True:
            file_name = f"{cls}_{img_num:03d}.jpg"
            save_path = class_dir / file_name
            if save_path.exists():
                found += 1
                img_num += 1
                continue
            url = f"{base_url.rstrip('/')}/{cls}/{file_name}"
            try:
                urllib.request.urlretrieve(url, save_path)
                found += 1
                img_num += 1
            except HTTPError as err:
                if err.code == 404:
                    break
                raise
            except URLError as err:
                print(f"Warning: could not download {url}: {err}")
                break
        print(f"  {cls}: {found} image(s) in {class_dir}")
        if cls in required_classes and found == 0:
            raise RuntimeError(f"Required class {cls} is missing and could not be downloaded.")


def list_image_files(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def list_samples(data_root: Path, class_names, required_classes=None):
    samples = []
    counts = {}
    required_classes = set(required_classes or [])
    for class_name in class_names:
        class_dir = data_root / class_name
        files = list_image_files(class_dir) if class_dir.exists() else []
        counts[class_name] = len(files)
        if class_name in required_classes and len(files) == 0:
            raise RuntimeError(f"Required transfer class {class_name} has no images in {class_dir}")
        for path in files:
            samples.append((str(path), CLASS_TO_IDX[class_name], class_name))
    print("Class mapping is fixed:", CLASS_TO_IDX)
    print("Images found:", counts)
    if not samples:
        raise RuntimeError(f"No images found under {data_root.resolve()}")
    return samples


def split_by_class(samples, val_fraction=0.20, min_val_new=4, seed=42):
    """Split transfer data per class, reserving enough validation for new classes."""
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample[2]].append(sample)
    train, val = [], []
    for class_name, items in sorted(grouped.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        if n < 2:
            train.extend(items)
            continue
        if class_name in NEW_CLASS_NAMES:
            n_val = min(max(1, min_val_new), n - 1)
        else:
            n_val = max(1, int(round(n * val_fraction)))
            n_val = min(n_val, n - 1)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


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
        self.stem = nn.Sequential(ConvBNAct(3, 32, 3, 2), ConvBNAct(32, 48, 3, 1))
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


def get_classifier(model):
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "fc"):
        return model.fc
    raise AttributeError("Model has no classifier/fc layer")


def set_classifier(model, new_layer):
    if hasattr(model, "classifier"):
        model.classifier = new_layer
    elif hasattr(model, "fc"):
        model.fc = new_layer
    else:
        raise AttributeError("Model has no classifier/fc layer")


def infer_num_classes_from_state_dict(state_dict):
    for key in ("classifier.weight", "fc.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return len(BASE_CLASS_NAMES)


def load_base_model(base_path: Path, device):
    """Load base checkpoint and expand classifier to A-J when required."""
    ckpt = safe_torch_load(base_path, device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = strip_module_prefix(ckpt["state_dict"])
        model_name = ckpt.get("model_name", "resnet18")
        old_class_names = list(ckpt.get("class_names", BASE_CLASS_NAMES))
        img_size = int(ckpt.get("img_size", 224 if model_name == "resnet18" else 160))
        embedding_dim = int(ckpt.get("embedding_dim", 128))
        norm = ckpt.get("normalization", {"mean": NORM_MEAN, "std": NORM_STD})
        preprocess = ckpt.get("preprocess", {"face_crop": True, "clahe": True, "crop_margin": 0.33})
        base_prototypes = ckpt.get("prototypes", None)
        base_proto_mask = ckpt.get("prototype_mask", None)
        base_proto_names = list(ckpt.get("prototype_class_names", old_class_names))
    else:
        state_dict = strip_module_prefix(ckpt)
        model_name = "resnet18"
        old_class_names = BASE_CLASS_NAMES[:infer_num_classes_from_state_dict(state_dict)]
        img_size = 224
        embedding_dim = 128
        norm = {"mean": NORM_MEAN, "std": NORM_STD}
        preprocess = {"face_crop": True, "clahe": True, "crop_margin": 0.33}
        base_prototypes = None
        base_proto_mask = None
        base_proto_names = old_class_names

    old_num_classes = len(old_class_names)
    if old_num_classes <= 0:
        old_num_classes = infer_num_classes_from_state_dict(state_dict)
        old_class_names = BASE_CLASS_NAMES[:old_num_classes]

    model = build_model(model_name, old_num_classes, embedding_dim=embedding_dim)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Warning: missing checkpoint keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("Warning: unexpected checkpoint keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    if old_num_classes != len(CLASS_NAMES):
        old_layer = get_classifier(model)
        new_layer = nn.Linear(old_layer.in_features, len(CLASS_NAMES))
        with torch.no_grad():
            rows = min(old_layer.out_features, len(CLASS_NAMES))
            new_layer.weight[:rows].copy_(old_layer.weight[:rows])
            new_layer.bias[:rows].copy_(old_layer.bias[:rows])
            if rows < len(CLASS_NAMES):
                nn.init.xavier_uniform_(new_layer.weight[rows:])
                nn.init.zeros_(new_layer.bias[rows:])
        set_classifier(model, new_layer)
        print(f"Expanded classifier from {old_num_classes} to {len(CLASS_NAMES)} classes.")
    else:
        print("Base checkpoint already has A-J classifier size.")

    model.to(device)
    return {
        "model": model,
        "model_name": model_name,
        "old_class_names": old_class_names,
        "img_size": img_size,
        "embedding_dim": embedding_dim,
        "normalization": norm,
        "preprocess": preprocess,
        "base_prototypes": base_prototypes,
        "base_proto_mask": base_proto_mask,
        "base_proto_names": base_proto_names,
    }


def freeze_all_but_head(model):
    """Freeze backbone and leave only classifier trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in get_classifier(model).parameters():
        p.requires_grad = True


def freeze_old_classifier_rows(model, old_rows: int):
    layer = get_classifier(model)
    if old_rows <= 0:
        return

    def zero_old_rows(grad):
        grad = grad.clone()
        grad[:old_rows] = 0
        return grad

    layer.weight.register_hook(zero_old_rows)
    layer.bias.register_hook(zero_old_rows)
    print(f"Frozen gradients for old classifier rows 0..{old_rows - 1}; only new rows will move.")


def unfreeze_last_stage(model):
    # Fine-tune only the identity-rich high-level part, not the whole network.
    if hasattr(model, "stage4"):
        for module in [model.stage4, model.embed, model.embed_bn, model.classifier]:
            for p in module.parameters():
                p.requires_grad = True
        return [
            {"params": model.stage4.parameters(), "lr_scale": 0.15},
            {"params": list(model.embed.parameters()) + list(model.embed_bn.parameters()), "lr_scale": 0.25},
            {"params": model.classifier.parameters(), "lr_scale": 1.0},
        ]
    if hasattr(model, "layer4"):
        for module in [model.layer4, model.fc]:
            for p in module.parameters():
                p.requires_grad = True
        return [
            {"params": model.layer4.parameters(), "lr_scale": 0.10},
            {"params": model.fc.parameters(), "lr_scale": 1.0},
        ]
    for p in get_classifier(model).parameters():
        p.requires_grad = True
    return [{"params": get_classifier(model).parameters(), "lr_scale": 1.0}]


def make_optimizer_from_groups(groups, base_lr, weight_decay):
    optim_groups = []
    for group in groups:
        params = [p for p in group["params"] if p.requires_grad]
        if params:
            optim_groups.append({"params": params, "lr": base_lr * group.get("lr_scale", 1.0)})
    return optim.AdamW(optim_groups, lr=base_lr, weight_decay=weight_decay)


def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=2.0):
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
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(yb.detach().cpu().tolist())
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(targets, preds) if targets else 0.0
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0) if targets else 0.0
    return avg_loss, acc, macro_f1, targets, preds


def train_stage(model, train_loader, val_loader, device, epochs, lr, weight_decay, stage_name, optimizer=None):
    """Train one transfer stage with cosine LR and stage-level early stopping."""
    if epochs <= 0:
        return copy.deepcopy(model.state_dict()), -1.0
    if optimizer is None:
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    stale = 0
    patience = max(6, min(14, epochs // 2))
    print(f"\nStarting {stage_name} for {epochs} epoch(s).")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, criterion, device) if val_loader else (0, train_acc, train_f1, [], [])
        scheduler.step()
        print(f"{stage_name} {epoch:03d}/{epochs} | train_loss={train_loss:.4f} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping {stage_name}.")
                break
    return best_state, best_f1


@torch.no_grad()
def evaluate(model, loader, device, title="Validation"):
    if loader is None or len(loader.dataset) == 0:
        return 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    loss, acc, macro_f1, targets, preds = run_epoch(model, loader, criterion, device)
    labels = sorted(set(targets) | set(preds))
    names = [CLASS_NAMES[i] for i in labels]
    print(f"\n{title}: loss={loss:.4f}, accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")
    print(classification_report(targets, preds, labels=labels, target_names=names, digits=4, zero_division=0))
    print("Confusion matrix rows=true, cols=pred:")
    print(confusion_matrix(targets, preds, labels=labels))
    return acc, macro_f1


@torch.no_grad()
def compute_prototypes(model, loader, num_classes, device):
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


def merge_base_and_transfer_prototypes(base_prototypes, base_mask, base_names, transfer_prototypes, transfer_mask):
    """Fill missing transfer prototypes using compatible base prototypes."""
    if transfer_prototypes is None:
        return None, None
    protos = transfer_prototypes.clone()
    mask = transfer_mask.clone()
    if base_prototypes is not None:
        if not torch.is_tensor(base_prototypes):
            base_prototypes = torch.tensor(base_prototypes, dtype=torch.float32)
        if base_mask is None:
            base_mask = torch.ones(base_prototypes.shape[0], dtype=torch.bool)
        elif not torch.is_tensor(base_mask):
            base_mask = torch.tensor(base_mask, dtype=torch.bool)
        for old_i, name in enumerate(base_names):
            if name in CLASS_TO_IDX and old_i < base_prototypes.shape[0]:
                new_i = CLASS_TO_IDX[name]
                same_dim = base_prototypes.shape[1] == protos.shape[1]
                if same_dim and bool(base_mask[old_i]) and not bool(mask[new_i]):
                    protos[new_i] = F.normalize(base_prototypes[old_i].float(), dim=0)
                    mask[new_i] = True
    return protos, mask


def main():
    parser = argparse.ArgumentParser(description="Transfer learning to classes A-J.")
    parser.add_argument("--base_model", default="base_model.pth")
    parser.add_argument("--data_dir", default="dataset", help="Folder containing dataset/J plus optional A-I folders.")
    parser.add_argument("--base_url", default="http://localhost/dataset/lfw_2026/", help="Used to fetch missing A-I replay/I data if available.")
    parser.add_argument("--out", default="final_model.pth")
    parser.add_argument("--epochs", type=int, default=55, help="Total budget; split into head warmup and fine tune.")
    parser.add_argument("--head_epochs", type=int, default=25)
    parser.add_argument("--finetune_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_head", type=float, default=6e-4)
    parser.add_argument("--lr_finetune", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--samples_per_class", type=int, default=120)
    parser.add_argument("--no_download_missing", action="store_true", help="Do not try to download missing A-I images from --base_url.")
    parser.add_argument("--no_replay", action="store_true", help="Train with I/J only even if A-H replay exists.")
    parser.add_argument("--no_face_crop", action="store_true")
    parser.add_argument("--no_clahe", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    base_path = Path(args.base_model)
    data_root = Path(args.data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base model: {base_path.resolve()}")
    data_root.mkdir(parents=True, exist_ok=True)

    if not args.no_download_missing:
        print("Checking/downloading missing LFW A-I images for replay/calibration if the local server is available.")
        retrieve_lfw_classes(CLASS_NAMES[:-1], args.base_url, data_root, required_classes=["I"])

    info = load_base_model(base_path, device)
    model = info["model"]
    img_size = info["img_size"]
    norm = info["normalization"]
    preprocess = info["preprocess"]
    face_crop = (not args.no_face_crop) and bool(preprocess.get("face_crop", True))
    clahe = (not args.no_clahe) and bool(preprocess.get("clahe", True))
    crop_margin = float(preprocess.get("crop_margin", 0.33))

    required = ["I", "J"]
    # Require I/J locally; replay classes A-H are optional but beneficial.
    all_samples = list_samples(data_root, CLASS_NAMES, required_classes=required)
    if args.no_replay:
        all_samples = [s for s in all_samples if s[2] in NEW_CLASS_NAMES]
        print("--no_replay enabled; using only I/J samples.")

    counts = Counter(s[2] for s in all_samples)
    has_replay = any(name in counts and counts[name] > 0 for name in BASE_CLASS_NAMES)
    print("Transfer counts:", dict(counts))
    if not has_replay:
        print("No A-H replay images are available. The backbone will stay frozen to avoid forgetting old classes.")
    else:
        print("A-H replay is available. The script will fine-tune the last stage after head warmup.")

    train_samples, val_samples = split_by_class(all_samples, seed=args.seed)
    print("Train counts:", dict(Counter(s[2] for s in train_samples)))
    print("Val counts:", dict(Counter(s[2] for s in val_samples)))

    base_train_tf = FaceTransform(img_size, train=True, mean=norm["mean"], std=norm["std"],
                                  face_crop=face_crop, crop_margin=crop_margin, clahe=clahe, strong=False)
    new_train_tf = FaceTransform(img_size, train=True, mean=norm["mean"], std=norm["std"],
                                 face_crop=face_crop, crop_margin=crop_margin, clahe=clahe, strong=True)
    eval_tf = FaceTransform(img_size, train=False, mean=norm["mean"], std=norm["std"],
                            face_crop=face_crop, crop_margin=crop_margin, clahe=clahe, strong=False)

    train_ds = FaceDataset(train_samples, base_train_tf, new_train_tf, NEW_CLASS_NAMES)
    val_ds = FaceDataset(val_samples, eval_tf, eval_tf, NEW_CLASS_NAMES)
    proto_ds = FaceDataset(all_samples, eval_tf, eval_tf, NEW_CLASS_NAMES)
    # Balanced sampling helps new classes stay visible during transfer.
    train_loader = make_loader(train_ds, args.batch_size, balanced=True, samples_per_class=args.samples_per_class,
                               num_workers=args.num_workers, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, balanced=False, num_workers=args.num_workers, shuffle=False) if len(val_ds) else None
    proto_loader = make_loader(proto_ds, args.batch_size, balanced=False, num_workers=args.num_workers, shuffle=False)

    # Stage 1: train only the classifier. If no replay, freeze old rows so A-H is preserved.
    freeze_all_but_head(model)
    if not has_replay:
        freeze_old_classifier_rows(model, old_rows=len(BASE_CLASS_NAMES))
    head_state, head_f1 = train_stage(model, train_loader, val_loader, device, args.head_epochs,
                                      args.lr_head, args.weight_decay, "head-warmup")
    model.load_state_dict(head_state)

    # Stage 2: if A-H replay exists, fine-tune only the high-level visual features.
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = head_f1
    if has_replay and args.finetune_epochs > 0:
        groups = unfreeze_last_stage(model)
        opt = make_optimizer_from_groups(groups, args.lr_finetune, args.weight_decay)
        ft_state, ft_f1 = train_stage(model, train_loader, val_loader, device, args.finetune_epochs,
                                      args.lr_finetune, args.weight_decay, "last-stage-finetune", optimizer=opt)
        if ft_f1 >= best_f1:
            best_state = ft_state
            best_f1 = ft_f1
    model.load_state_dict(best_state)

    evaluate(model, val_loader, device, title="Final validation")

    transfer_protos, transfer_mask = compute_prototypes(model, proto_loader, len(CLASS_NAMES), device)
    final_protos, final_mask = merge_base_and_transfer_prototypes(
        info["base_prototypes"], info["base_proto_mask"], info["base_proto_names"], transfer_protos, transfer_mask
    )
    if final_mask is not None:
        print("Prototype coverage:", {CLASS_NAMES[i]: bool(final_mask[i]) for i in range(len(CLASS_NAMES))})

    checkpoint = {
        "model_name": info["model_name"],
        "state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "img_size": img_size,
        "embedding_dim": info["embedding_dim"],
        "normalization": norm,
        "preprocess": {"face_crop": face_crop, "clahe": clahe, "crop_margin": crop_margin},
        "prototypes": final_protos,
        "prototype_mask": final_mask,
        "prototype_class_names": CLASS_NAMES,
        "confidence_threshold": 0.28,
        "score_mode": "softmax_plus_prototype",
        "best_val_f1": best_f1,
        "transfer_counts": dict(counts),
        "best_config": vars(args),
    }
    out_path = Path(args.out)
    safe_torch_save(checkpoint, out_path)
    print(f"Saved transfer checkpoint to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
