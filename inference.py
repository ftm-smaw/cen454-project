import os
import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from train import (
    classical_preprocess,
    inference_transform,
    CLASS_NAMES,
    NUM_CLASSES,
    IMG_SIZE,
    DEVICE,
)


def load_model(checkpoint_path: str) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"  Model loaded from {checkpoint_path}")
    return model


def extract_gt_bbox(mask_path: str) -> list:
    MIN_AREA = 100

    mask_img = np.array(Image.open(mask_path))
    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]

    binary = (mask_img > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]

    if not valid_contours:
        return None

    best = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    return [x, y, x + w, y + h]


def classify(model: nn.Module, img_rgb: np.ndarray):
    tensor = inference_transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        pred_idx = logits.argmax(dim=1).item()

    return CLASS_NAMES[pred_idx], pred_idx, tensor


def localize(model: nn.Module, tensor: torch.Tensor,
             pred_idx: int, original_size: tuple) -> list:
    original_w, original_h = original_size

    target_layer = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layer)
    targets = [ClassifierOutputTarget(pred_idx)]
    heatmap = cam(input_tensor=tensor, targets=targets)[0]

    binary_mask = (heatmap >= 0.5).astype(np.uint8) * 255

    kernel = np.ones((15, 15), np.uint8)
    mask_clean = cv2.dilate(binary_mask, kernel, iterations=2)
    mask_clean = cv2.erode(mask_clean, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return [0, 0, original_w, original_h]

    def mean_heatmap_score(contour):
        x, y, w, h = cv2.boundingRect(contour)
        region = heatmap[y:y + h, x:x + w]
        return np.mean(region) if region.size > 0 else 0.0

    best_contour = max(contours, key=mean_heatmap_score)
    x, y, w, h = cv2.boundingRect(best_contour)

    scale_x = original_w / IMG_SIZE
    scale_y = original_h / IMG_SIZE

    return [
        int(x * scale_x),
        int(y * scale_y),
        int((x + w) * scale_x),
        int((y + h) * scale_y),
    ]


def calculate_iou(box_a: list, box_b: list) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = float(area_a + area_b - inter_area)

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def run_inference(model: nn.Module, test_dir: str,
                  annotation_dir: str = None) -> list:
    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    results = []

    image_files = sorted([
        f for f in os.listdir(test_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ])

    print(f"  Running inference on {len(image_files)} images...")

    for fname in image_files:
        fpath = os.path.join(test_dir, fname)
        stem = os.path.splitext(fname)[0]

        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            img_rgb = np.array(Image.open(fpath).convert("RGB"))
            original_size = (img_rgb.shape[1], img_rgb.shape[0])
        else:
            original_size = (img_bgr.shape[1], img_bgr.shape[0])
            img_rgb = classical_preprocess(img_bgr)

        label, pred_idx, tensor = classify(model, img_rgb)

        bbox = None
        if label != "safe":
            bbox = localize(model, tensor, pred_idx, original_size)

        gt_bbox = None
        iou = None
        if annotation_dir and bbox is not None:
            mask_path = os.path.join(annotation_dir, label, stem + ".png")
            if os.path.exists(mask_path):
                gt_bbox = extract_gt_bbox(mask_path)
                if gt_bbox:
                    iou = calculate_iou(bbox, gt_bbox)

        results.append({
            "image_name": fname,
            "label": label,
            "bbox": bbox,
            "gt_bbox": gt_bbox,
            "iou": iou
        })

        iou_str = f" | IoU: {iou:.4f}" if iou is not None else ""
        print(f"    {fname} → {label}" + (f" | bbox: {bbox}{iou_str}" if bbox else ""))

    return results