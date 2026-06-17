"""
inference.py

Purpose:
- Run inference using the final A-J checkpoint on camera feed or still images.

Dataset context:
- Classes A-H are trained from LFW (100 images per class).
- Class I uses LFW (20 images).
- Class J uses your own dataset (20 images).
- I and J are used to evaluate how well the pipeline generalizes to less-data and
    unseen/custom data compared to the stronger A-H base classes.

Pipeline role:
- This is stage 3 of the pipeline (inference/evaluation).
- It consumes the checkpoint produced by transfer.py and displays top predictions.
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR

try:
    FLIP_LEFT_RIGHT_COMPAT = Image.Transpose.FLIP_LEFT_RIGHT
except AttributeError:
    FLIP_LEFT_RIGHT_COMPAT = Image.FLIP_LEFT_RIGHT

DEFAULT_CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]


def safe_torch_load(path: Path, device):
    """Load checkpoints across PyTorch versions with and without weights_only support."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


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
    def __init__(self, block, layers, num_classes=10):
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


def infer_num_classes_from_state_dict(state_dict):
    for key in ("classifier.weight", "fc.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return len(DEFAULT_CLASS_NAMES)


def load_checkpoint(model_path: Path, device):
    """Load model + inference metadata from a project or legacy checkpoint."""
    ckpt = safe_torch_load(model_path, device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = strip_module_prefix(ckpt["state_dict"])
        class_names = list(ckpt.get("class_names", DEFAULT_CLASS_NAMES))
        model_name = ckpt.get("model_name", "resnet18")
        img_size = int(ckpt.get("img_size", 224 if model_name == "resnet18" else 160))
        embedding_dim = int(ckpt.get("embedding_dim", 128))
        norm = ckpt.get("normalization", {"mean": NORM_MEAN, "std": NORM_STD})
        preprocess = ckpt.get("preprocess", {"clahe": True, "crop_margin": 0.33})
        threshold = float(ckpt.get("confidence_threshold", 0.28))
        prototypes = ckpt.get("prototypes", None)
        proto_mask = ckpt.get("prototype_mask", None)
    else:
        state_dict = strip_module_prefix(ckpt)
        class_names = DEFAULT_CLASS_NAMES[:infer_num_classes_from_state_dict(state_dict)]
        model_name = "resnet18"
        img_size = 224
        embedding_dim = 128
        norm = {"mean": NORM_MEAN, "std": NORM_STD}
        preprocess = {"clahe": True, "crop_margin": 0.25}
        threshold = 0.28
        prototypes = None
        proto_mask = None

    model = build_model(model_name, len(class_names), embedding_dim=embedding_dim)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Warning: missing checkpoint keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("Warning: unexpected checkpoint keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    model.to(device)
    model.eval()

    if prototypes is not None:
        if not torch.is_tensor(prototypes):
            prototypes = torch.tensor(prototypes, dtype=torch.float32)
        prototypes = prototypes.float().to(device)
        prototypes = F.normalize(prototypes, dim=1)
        if proto_mask is None:
            proto_mask = torch.ones(prototypes.shape[0], dtype=torch.bool, device=device)
        elif not torch.is_tensor(proto_mask):
            proto_mask = torch.tensor(proto_mask, dtype=torch.bool, device=device)
        else:
            proto_mask = proto_mask.bool().to(device)
        if prototypes.shape[0] != len(class_names):
            print("Warning: prototype rows do not match classes. Disabling prototypes.")
            prototypes, proto_mask = None, None

    return model, class_names, img_size, norm, preprocess, threshold, prototypes, proto_mask


def apply_clahe_bgr(bgr_image):
    """Apply CLAHE on luminance channel while preserving color information."""
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    merged = cv2.merge((l_chan, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def crop_with_margin(frame, x, y, w, h, margin_ratio=0.28):
    margin = int(max(w, h) * margin_ratio)
    y1 = max(0, y - margin)
    y2 = min(frame.shape[0], y + h + margin)
    x1 = max(0, x - margin)
    x2 = min(frame.shape[1], x + w + margin)
    return frame[y1:y2, x1:x2]


def bgr_to_tensor(bgr_image, img_size, norm, do_clahe=True, flip=False):
    if do_clahe:
        bgr_image = apply_clahe_bgr(bgr_image)
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).convert("RGB")
    if flip:
        pil = pil.transpose(FLIP_LEFT_RIGHT_COMPAT)
    pil = pil.resize((img_size, img_size), RESAMPLE_BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr)
    mean = torch.tensor(norm["mean"], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(norm["std"], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def make_tta_batch(face_bgr, img_size, norm, use_clahe=True, tta=2):
    # Keep TTA variants deterministic and ordered for stable averaging.
    variants = [bgr_to_tensor(face_bgr, img_size, norm, do_clahe=use_clahe, flip=False)]
    if tta >= 2:
        variants.append(bgr_to_tensor(face_bgr, img_size, norm, do_clahe=use_clahe, flip=True))
    if tta >= 3:
        variants.append(bgr_to_tensor(face_bgr, img_size, norm, do_clahe=False, flip=False))
    if tta >= 4:
        variants.append(bgr_to_tensor(face_bgr, img_size, norm, do_clahe=False, flip=True))
    return torch.stack(variants, dim=0)


@torch.no_grad()
def predict_face(model, face_bgr, device, class_names, img_size, norm, prototypes=None, proto_mask=None,
                 use_clahe=True, tta=2, proto_weight=0.45, proto_temp=12.0):
    """Predict top-3 labels using softmax and optional prototype fusion."""
    batch = make_tta_batch(face_bgr, img_size, norm, use_clahe=use_clahe, tta=tta).to(device)
    logits, feats = model(batch, return_features=True)
    soft_probs = F.softmax(logits, dim=1).mean(dim=0)
    scores = soft_probs
    if prototypes is not None and proto_mask is not None and prototypes.shape[1] == feats.shape[1]:
        sims = feats @ prototypes.t()
        sims[:, ~proto_mask] = -20.0
        proto_probs = F.softmax(sims * proto_temp, dim=1).mean(dim=0)
        scores = (1.0 - proto_weight) * soft_probs + proto_weight * proto_probs
    top_probs, top_idxs = torch.topk(scores, k=min(3, len(class_names)))
    return [(class_names[int(idx)], float(prob)) for prob, idx in zip(top_probs, top_idxs)], scores.detach().cpu()


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes, threshold=0.30):
    """Simple area-sorted NMS to reduce overlapping face boxes."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    keep = []
    for box in boxes:
        if all(iou(box, kept) < threshold for kept in keep):
            keep.append(box)
    return keep


def detect_faces(frame, face_cascade, min_face=55, min_neighbors=6):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray_eq, scaleFactor=1.07, minNeighbors=min_neighbors, minSize=(min_face, min_face)
    )
    boxes = [tuple(map(int, f)) for f in faces]
    return nms_boxes(boxes, threshold=0.30)


def draw_label_block(frame, x, y, w, h, top3, best_prob, threshold, show_unknown=True):
    ok = best_prob >= threshold or not show_unknown
    colour = (0, 220, 0) if ok else (0, 165, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    if ok:
        header = f"{top3[0][0]} ({top3[0][1] * 100:.1f}%)"
    else:
        header = f"Low confidence ({best_prob * 100:.1f}%)"
    cv2.putText(frame, header, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
    base_y = min(frame.shape[0] - 10, y + h + 24)
    for i, (label, prob) in enumerate(top3):
        text = f"#{i + 1}: {label} {prob * 100:.1f}%"
        yy = base_y + i * 22
        if yy < frame.shape[0] - 5:
            cv2.putText(frame, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2)


def process_frame(frame, model, device, class_names, img_size, norm, preprocess, threshold, face_cascade,
                  prototypes, proto_mask, args, history=None):
    """Detect faces, score each crop, smooth primary face, and annotate frame."""
    boxes = detect_faces(frame, face_cascade, min_face=args.min_face, min_neighbors=args.min_neighbors)
    if len(boxes) == 0:
        cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return []

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:args.max_faces]
    predictions = []
    for face_id, (x, y, w, h) in enumerate(boxes):
        crop = crop_with_margin(frame, x, y, w, h, margin_ratio=args.crop_margin)
        if crop.size == 0:
            continue
        top3, scores = predict_face(
            model, crop, device, class_names, img_size, norm,
            prototypes=prototypes, proto_mask=proto_mask,
            use_clahe=bool(preprocess.get("clahe", True)) and not args.no_clahe,
            tta=args.tta, proto_weight=args.prototype_weight, proto_temp=args.prototype_temp,
        )
        if history is not None and face_id == 0:
            history.append(scores)
            avg_scores = torch.stack(list(history), dim=0).mean(dim=0)
            top_probs, top_idxs = torch.topk(avg_scores, k=min(3, len(class_names)))
            top3 = [(class_names[int(idx)], float(prob)) for prob, idx in zip(top_probs, top_idxs)]
        best_prob = top3[0][1]
        predictions.append((x, y, w, h, top3))
        draw_label_block(frame, x, y, w, h, top3, best_prob, threshold, show_unknown=not args.no_unknown)
    return predictions


def run_image(args, model, device, class_names, img_size, norm, preprocess, threshold, face_cascade, prototypes, proto_mask):
    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    preds = process_frame(frame, model, device, class_names, img_size, norm, preprocess, threshold,
                          face_cascade, prototypes, proto_mask, args, history=None)
    for x, y, w, h, top3 in preds:
        print(f"Box ({x},{y},{w},{h}): " + ", ".join(f"{lab}={prob*100:.1f}%" for lab, prob in top3))
    out_path = Path(args.out_image or (image_path.stem + "_pred.jpg"))
    cv2.imwrite(str(out_path), frame)
    print(f"Saved annotated image to: {out_path.resolve()}")


def _camera_candidates(camera_arg):
    """Return robust camera candidates from a user hint (index/path/auto)."""
    raw = str(camera_arg).strip()
    if raw.lower() in {"auto", "scan"}:
        return list(range(0, 8)) + [f"/dev/video{i}" for i in range(0, 8)]

    candidates = []
    try:
        first = int(raw)
        candidates.append(first)
        candidates.extend(i for i in range(0, 8) if i != first)
        candidates.extend(f"/dev/video{i}" for i in range(0, 8))
    except ValueError:
        candidates.append(raw)
        candidates.extend(list(range(0, 8)))
        candidates.extend(f"/dev/video{i}" for i in range(0, 8))

    deduped = []
    seen = set()
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def _open_camera(args):
    """Try camera candidates until one provides an actual frame."""
    errors = []
    for candidate in _camera_candidates(args.camera):
        # V4L2 is the most reliable backend on Jetson for integer camera indexes.
        if isinstance(candidate, int):
            cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(candidate)

        if args.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        if not cap.isOpened():
            errors.append(f"{candidate}: not opened")
            cap.release()
            continue

        ok, first_frame = cap.read()
        if ok and first_frame is not None:
            print(f"Opened camera: {candidate}")
            return cap, candidate, first_frame

        errors.append(f"{candidate}: opened but no frame returned")
        cap.release()

    raise RuntimeError(
        "Could not open any camera. Tried: " + ", ".join(str(c) for c in _camera_candidates(args.camera)) +
        "\nCheck the camera with: ls /dev/video*\n" +
        "Then try: python inference.py --camera 1\n" +
        "Or try: python inference.py --camera auto"
    )


def run_live(args, model, device, class_names, img_size, norm, preprocess, threshold, face_cascade, prototypes, proto_mask):
    cap, opened_as, first_frame = _open_camera(args)
    _ = opened_as  # kept for future diagnostics; ensures camera source is preserved if needed

    history = deque(maxlen=max(1, args.smooth))
    last_time = time.time()
    fps = 0.0
    print("Starting live inference. Press q to quit.")

    pending_frame = first_frame
    while True:
        if pending_frame is not None:
            frame = pending_frame
            pending_frame = None
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera stopped returning frames.")
                break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        # In live mode we smooth only the largest face over a short history.
        process_frame(frame, model, device, class_names, img_size, norm, preprocess, threshold,
                  face_cascade, prototypes, proto_mask, args, history=history)

        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.90 * fps + 0.10 * (1.0 / dt) if fps > 0 else 1.0 / dt

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("COMPSYS 721 Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live face recognition inference for COMPSYS 721 Project 2.")
    parser.add_argument("--model", default="final_model.pth")
    parser.add_argument("--camera", default="0", help="Camera index/path. Use 0, 1, /dev/video0, or auto.")
    parser.add_argument("--image", default=None, help="Optional: run on a still image instead of camera.")
    parser.add_argument("--out_image", default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Override saved confidence threshold. Use 0 to force labels.")
    parser.add_argument("--min_face", type=int, default=55)
    parser.add_argument("--min_neighbors", type=int, default=6)
    parser.add_argument("--max_faces", type=int, default=4)
    parser.add_argument("--crop_margin", type=float, default=0.28)
    parser.add_argument("--tta", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--prototype_weight", type=float, default=0.45)
    parser.add_argument("--prototype_temp", type=float, default=12.0)
    parser.add_argument("--smooth", type=int, default=4, help="Temporal smoothing frames for the largest face.")
    parser.add_argument("--no_clahe", action="store_true")
    parser.add_argument("--no_unknown", action="store_true", help="Always display the top label even below threshold.")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, img_size, norm, preprocess, saved_threshold, prototypes, proto_mask = load_checkpoint(Path(args.model), device)
    threshold = saved_threshold if args.threshold is None else args.threshold
    print("Device:", device)
    print("Loaded classes:", class_names)
    print("Image size:", img_size)
    print("Threshold:", threshold)
    print("Prototype scoring:", prototypes is not None)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Could not load Haar cascade.")

    if args.image:
        run_image(args, model, device, class_names, img_size, norm, preprocess, threshold, face_cascade, prototypes, proto_mask)
    else:
        run_live(args, model, device, class_names, img_size, norm, preprocess, threshold, face_cascade, prototypes, proto_mask)


if __name__ == "__main__":
    main()
