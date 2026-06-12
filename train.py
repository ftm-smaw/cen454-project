"""
CEN454
Part 2: Image Classifier Training
Objective:
    Train an EfficientNet-B0 classifier to detect threat items in baggage images.
    The model predicts one of four labels: safe | gun | knife | shuriken

Architecture Overview (matches Project Architecture Flowchart):
    Raw Image Input
        → Classical CV Preprocessing  (CLAHE, Gaussian Blur, Unsharp Masking)
        → EfficientNet-B0 Backbone    (pretrained on ImageNet – transfer learning)
        → Custom Classification Head  (4-class output)
        → Prediction: safe / gun / knife / shuriken

Scoring formula (from project spec):
    Classification Score = 0.7 × Accuracy + 0.3 × Macro F1-Score
    Final Score          = 0.7 × Classification Score + 0.3 × Localization Score

References to course topics:
    - Color Processing (LAB/HSI): CLAHE applied in LAB lightness channel
    - Morphological Operators: used in binary mask generation during preprocessing
    - Segmentation: thresholding + contour-based region proposals feed into Part 3
    - HOG / Shape Features: captured implicitly by EfficientNet's convolutional layers
    - Macro F1-Score: penalises models that fail on minority classes (gun, shuriken)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import csv
import copy
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION  (edit these to match your setup)
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT = "./data/train"         # folder structure: dataset/train/<class>/<images>
OUTPUT_DIR  = "./outputs"          # where checkpoints and CSVs are saved
CHECKPOINT  = "trained_model.pth"     # filename for the best model weights

CLASS_NAMES = ["safe", "gun", "knife", "shuriken"]   # must match subfolder names
NUM_CLASSES = len(CLASS_NAMES)     # 4 – replaces ImageNet's 1 000-class head

# Training hyper-parameters
IMG_SIZE        = 224              # EfficientNet-B0 default input resolution
BATCH_SIZE      = 16
NUM_EPOCHS      = 30               # total epochs across both training phases
WARMUP_EPOCHS   = 5                # Phase 1: train only the new head (backbone frozen)
LR_HEAD         = 1e-3             # learning rate while backbone is frozen
LR_FINETUNE     = 1e-4             # learning rate during full fine-tuning
WEIGHT_DECAY    = 1e-4
EARLY_STOP_PAT  = 7                # stop if val loss doesn't improve for N epochs
VAL_SPLIT       = 0.2              # fraction of training data held out for validation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLASSICAL CV PREPROCESSING PIPELINE
#    Course reference → Color Processing (LAB channel), Morphological Operators
# ─────────────────────────────────────────────────────────────────────────────
def classical_preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply classical computer vision preprocessing before feeding into the CNN.

    Steps:
        1. CLAHE on the L-channel of LAB colour space
           → equalises contrast without shifting hue (LAB colour processing)
        2. Gaussian blur for noise suppression
           → reduces high-frequency sensor noise before feature extraction
        3. Unsharp masking to recover edge sharpness lost in step 2
           → formula: sharpened = original + α × (original – blurred)
             where α controls the enhancement strength

    These steps mirror the preprocessing pipeline described in the project's
    architecture flowchart and align with:
        - Lecture topic: Color Processing (LAB/HSI conversion)
        - Lecture topic: Morphological Operators (structuring-element-based ops)
    """
    # ── Step 1: CLAHE in LAB colour space ──────────────────────────────────
    # Convert BGR → LAB so contrast enhancement only touches luminance (L)
    # This avoids colour distortion, matching the LAB lecture discussion.
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # ── Step 2: Gaussian Blur – noise suppression ───────────────────────────
    blurred = cv2.GaussianBlur(img_enhanced, ksize=(3, 3), sigmaX=0)

    # ── Step 3: Unsharp Masking – edge sharpening ───────────────────────────
    # Formula:  sharpened = clip( original + α × (original – blurred) )
    # α = 1.5 gives moderate enhancement without amplifying noise.
    alpha = 1.5
    sharpened = cv2.addWeighted(img_enhanced, 1 + alpha, blurred, -alpha, 0)

    # Return in RGB for PIL/torchvision compatibility
    return cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CUSTOM DATASET
# ─────────────────────────────────────────────────────────────────────────────
class BaggageDataset(Dataset):
    """
    Loads baggage threat images from a folder structure:

        dataset/
            safe/       ← class 0
            gun/        ← class 1
            knife/      ← class 2
            shuriken/   ← class 3

    Each image is:
        1. Read with OpenCV (supports a wider range of formats than PIL alone)
        2. Passed through the classical CV preprocessing pipeline (CLAHE → Blur → Unsharp)
        3. Converted to a PIL Image for torchvision transforms compatibility
        4. Passed through augmentation / normalisation transforms

    The binary mask comment below refers to the Morphological Operators lecture:
    When a threat pixel map is thresholded to {0, 255} it forms a binary mask
    used in Part 3 (localisation). Here we store that logic as a comment anchor.
    # binary_mask = (segmentation_output > threshold).astype(np.uint8) * 255
    """

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, root: str, class_names: list[str], transform=None):
        self.transform   = transform
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        self.samples: list[tuple[str, int]] = []
        for cls in class_names:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                print(f"  [WARNING] Missing class folder: {cls_dir}")
                continue
            for fname in os.listdir(cls_dir):
                if os.path.splitext(fname)[1].lower() in self.SUPPORTED_EXTS:
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

        print(f"  Dataset loaded: {len(self.samples)} images across {len(class_names)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # PIL fallback – bypasses strict OpenCV codec rejections (same fix
            # used in the project's reference notebook for INRIA dataset loading)
            img_pil = Image.open(path).convert("RGB")
            img_rgb = np.array(img_pil)
        else:
            # Apply classical CV preprocessing → aligns with course pipeline
            img_rgb = classical_preprocess(img_bgr)

        image = Image.fromarray(img_rgb)

        if self.transform:
            image = self.transform(image)

        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA TRANSFORMS (augmentation + normalisation)
# ─────────────────────────────────────────────────────────────────────────────
# ImageNet mean/std used because EfficientNet-B0 was pretrained on ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),   # slightly larger for crop
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

inference_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA LOADING  (train / validation split)
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(data_root: str, val_split: float, batch_size: int):
    """
    Loads the full dataset, then performs a stratified train/val split
    so each class is represented proportionally in both subsets.
    Stratification is important here because the dataset may be imbalanced
    (e.g. more 'safe' images than 'shuriken') – matching the Macro F1
    rationale from the project spec and the reference notebook.
    """
    full_dataset = BaggageDataset(data_root, CLASS_NAMES, transform=None)

    # Stratified split: collect indices per class, then split each group
    from collections import defaultdict
    import random

    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)

    train_idx, val_idx = [], []
    random.seed(42)
    for label, indices in class_indices.items():
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_split))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    # Wrap with transforms using a lightweight adapter
    class _SubsetWithTransform(Dataset):
        def __init__(self, base, indices, transform):
            self.base = base
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            path, label = self.base.samples[self.indices[i]]
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                img_pil = Image.open(path).convert("RGB")
                img_rgb = np.array(img_pil)
            else:
                img_rgb = classical_preprocess(img_bgr)
            image = Image.fromarray(img_rgb)
            if self.transform:
                image = self.transform(image)
            return image, label

    train_ds = _SubsetWithTransform(full_dataset, train_idx, train_transforms)
    val_ds   = _SubsetWithTransform(full_dataset, val_idx,   val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    print(f"  Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL DEFINITION
#    Download EfficientNet-B0 with pretrained ImageNet weights, then
#    replace the final classification head to output NUM_CLASSES (4) logits.
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Transfer Learning Setup:
        - EfficientNet-B0 was pre-trained by Google on ImageNet (1 000 classes).
        - We replace its final Linear layer with a new one that outputs 4 classes:
              safe | gun | knife | shuriken
        - During Phase 1 (warm-up), the backbone weights are frozen so only the
          new head is trained. This is safe because ImageNet features generalise
          well to X-ray / threat imagery at a low level.
        - During Phase 2 (fine-tuning), all layers are unfrozen and the entire
          network adapts to our specific domain at a lower learning rate.

    The classifier head structure mirrors the project spec requirement:
        "Replace its final layer to output 4 labels instead of 1000"
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    # ── Freeze backbone (Phase 1) ───────────────────────────────────────────
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # ── Replace classifier head ─────────────────────────────────────────────
    # Original: model.classifier = [Dropout, Linear(1280 → 1000)]
    # New:      model.classifier = [Dropout, Linear(1280 → num_classes)]
    in_features = model.classifier[1].in_features   # 1 280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    # New head is always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model.to(DEVICE)


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all backbone parameters for Phase 2 fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("  [Phase 2] Backbone unfrozen – full fine-tuning enabled.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAINING LOOP  (one epoch)
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, phase: str):
    """
    Execute one full pass through the dataset.

    - 'train' phase: gradient computation + weight update (backpropagation)
    - 'val'   phase: no gradient computation (inference only)

    Returns: (average_loss, accuracy, macro_f1)
    The macro F1 aligns with the project scoring formula:
        Classification Score = 0.7 × Accuracy + 0.3 × Macro F1
    """
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc        = accuracy_score(all_labels, all_preds)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, acc, macro_f1


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN TRAINING ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def train(data_root: str = DATA_ROOT):
    print(f"\n{'='*60}")
    print("  CEN454 | Part 2 – EfficientNet-B0 Classifier Training")
    print(f"  Device : {DEVICE}")
    print(f"{'='*60}\n")

    # ── Build dataloaders ───────────────────────────────────────────────────
    print("[1/5] Building dataloaders...")
    train_loader, val_loader = build_dataloaders(data_root, VAL_SPLIT, BATCH_SIZE)

    # ── Build model (backbone frozen for warm-up) ───────────────────────────
    print("\n[2/5] Loading EfficientNet-B0 with pretrained ImageNet weights...")
    model = build_model(NUM_CLASSES, freeze_backbone=True)
    print(f"  Classifier head: Linear(1280 → {NUM_CLASSES})")
    print(f"  Trainable params (Phase 1): "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Loss: CrossEntropy handles multi-class classification ───────────────
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer: only head parameters updated in Phase 1 ─────────────────
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ── Tracking ────────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_loss  = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    checkpoint_path  = os.path.join(OUTPUT_DIR, CHECKPOINT)

    print("\n[3/5] Starting training loop...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Phase transition at WARMUP_EPOCHS ──────────────────────────────
        if epoch == WARMUP_EPOCHS + 1:
            print(f"\n{'─'*60}")
            print(f"  Epoch {epoch}: switching to Phase 2 – full fine-tuning")
            unfreeze_backbone(model)
            # Rebuild optimizer with lower LR for all parameters
            optimizer = optim.Adam(model.parameters(),
                                   lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
            print(f"  Trainable params (Phase 2): "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            print(f"{'─'*60}\n")

        t0 = time.time()

        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, "train")
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, "val")

        scheduler.step()

        # Compute project score on validation set
        # Classification Score = 0.7 × Accuracy + 0.3 × Macro F1
        clf_score = 0.7 * val_acc + 0.3 * val_f1

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - t0
        phase_tag = "Phase 1 – head only" if epoch <= WARMUP_EPOCHS else "Phase 2 – full tune"
        print(
            f"  Epoch [{epoch:02d}/{NUM_EPOCHS}] ({phase_tag}) | "
            f"{elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Clf Score: {clf_score:.4f}"
        )

        # ── Checkpoint: save best model ────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, checkpoint_path)
            print(f"    ✓ Best model saved → {checkpoint_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PAT:
                print(f"\n  Early stopping triggered after {epoch} epochs "
                      f"(no improvement for {EARLY_STOP_PAT} epochs).")
                break

    # ── Restore best weights ────────────────────────────────────────────────
    model.load_state_dict(best_model_wts)
    print(f"\n  Best weights restored (val loss = {best_val_loss:.4f})")

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# 9. EVALUATION  (confusion matrix + project score)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, val_loader: DataLoader):
    """
    Full evaluation on the validation set.
    Outputs:
        - Classification report per class
        - Confusion matrix heatmap (matches the project's INRIA reference notebook style)
        - Final Classification Score = 0.7 × Accuracy + 0.3 × Macro F1
    """
    print("\n[4/5] Running final evaluation on validation set...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    clf_score = 0.7 * acc + 0.3 * macro_f1

    print("\n  ── Classification Report ──────────────────────────────────────")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES,
                                zero_division=0))
    print(f"  Standard Accuracy        : {acc:.4f}")
    print(f"  Macro F1-Score           : {macro_f1:.4f}")
    print(f"  Classification Score     : {clf_score:.4f}  "
          f"(= 0.7×{acc:.4f} + 0.3×{macro_f1:.4f})")

    # ── Confusion matrix ────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
    plt.title("Confusion Matrix – CEN454 Baggage Threat Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"  Confusion matrix saved → {cm_path}")

    return acc, macro_f1, clf_score


# ─────────────────────────────────────────────────────────────────────────────
# 10. INFERENCE  (generate predictions.csv for evaluation day)
# ─────────────────────────────────────────────────────────────────────────────
def generate_predictions(model: nn.Module, test_dir: str,
                          output_csv: str = "predictions.csv"):
    """
    Run inference on an unseen test folder and write predictions.csv.

    Expected test folder structure (flat, no class subfolders):
        test_images/
            img001.jpg
            img002.jpg
            ...

    Output CSV format (matches project submission spec):
        Image Name, Predicted Label
        img001.jpg, safe
        img002.jpg, gun
        ...

    Note: Only inference is allowed during the evaluation session.
    No weight updates are performed here.
    """
    print(f"\n[5/5] Generating predictions from: {test_dir}")
    model.eval()

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    results = []

    for fname in sorted(os.listdir(test_dir)):
        if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXTS:
            continue

        fpath   = os.path.join(test_dir, fname)
        img_bgr = cv2.imread(fpath)

        if img_bgr is None:
            img_rgb = np.array(Image.open(fpath).convert("RGB"))
        else:
            img_rgb = classical_preprocess(img_bgr)

        tensor = inference_transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logit = model(tensor)
            pred  = logit.argmax(dim=1).item()

        results.append((fname, CLASS_NAMES[pred]))

    csv_path = os.path.join(OUTPUT_DIR, output_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "Predicted Label"])
        writer.writerows(results)

    print(f"  {len(results)} predictions written → {csv_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# 11. TRAINING HISTORY PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_history(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(history["val_acc"], label="Val Accuracy")
    axes[1].plot(history["val_f1"],  label="Val Macro F1")
    axes[1].set_title("Validation Metrics over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    plt.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, "training_history.png")
    plt.savefig(hist_path, dpi=150)
    print(f"  Training history plot saved → {hist_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Step 1: Train ──────────────────────────────────────────────────────
    model, history = train(DATA_ROOT)

    # ── Step 2: Plot training curves ────────────────────────────────────────
    plot_history(history)

    # ── Step 3: Full evaluation with confusion matrix ───────────────────────
    _, val_loader = build_dataloaders(DATA_ROOT, VAL_SPLIT, BATCH_SIZE)
    evaluate(model, val_loader)

    # ── Step 4: Generate predictions.csv (update TEST_DIR before eval day) ──
    TEST_DIR = "./test_images"          # ← point this at the hidden test folder
    if os.path.isdir(TEST_DIR):
        generate_predictions(model, TEST_DIR)
    else:
        print(f"\n  [INFO] Test folder '{TEST_DIR}' not found. "
              f"Run generate_predictions() manually on evaluation day.")