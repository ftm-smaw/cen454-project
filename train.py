"""
Part 2: Training the Classifier

HOW THIS FILE IS ORGANISED
───────────────────────────
 0. Settings (paths, hyper-parameters)
 1. Classical CV preprocessing  ← Topics 5, 6, 7 techniques
 2. Dataset loader
 3. Data augmentation / transforms
 4. Build the neural-network model
 5. Training loop
 6. Evaluation (accuracy, F1, confusion matrix)
 7. Inference / prediction (for evaluation day)
 8. Main – ties everything together
"""

import os, copy, time, csv, random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#  SETTINGS  –  edit these if your folder names are different

DATA_ROOT      = Path("data")      # root folder that holds train/ val/ test/
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# The 4 classes we must predict (order matters – index 0=safe, 1=gun …)
CLASS_NAMES = ["gun", "knife", "safe", "shuriken"]
NUM_CLASSES = len(CLASS_NAMES)       # 4

# Training hyper-parameters
BATCH_SIZE   = 32     # images processed in one forward pass
NUM_EPOCHS   = 30     # how many times the model sees the whole training set
LR           = 1e-3   # learning rate = 0.001
WEIGHT_DECAY = 1e-4   # L2 regularisation (prevents overfitting)
IMG_SIZE     = 224    # EfficientNet-B0 expects 224 × 224 pixels
NUM_WORKERS  = 4      # parallel data-loading threads
SEED         = 42     # for reproducibility

# Use GPU if available, otherwise fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on: {DEVICE}")


#  1.  CLASSICAL CV PREPROCESSING PIPELINE
#      Techniques from Topics 5, 6, 7

def classical_preprocess(img_bgr: np.ndarray) -> np.ndarray:
    # Convert from BGR (OpenCV default) → LAB colour space
    # L = lightness (0-255), A = green↔red, B = blue↔yellow
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to the L (brightness) channel only
    clahe = cv2.createCLAHE(
        clipLimit=2.0,          # cap to avoid over-amplifying noise
        tileGridSize=(8, 8)     # image divided into 8×8 tiles
    )
    l_channel = clahe.apply(l_channel)

    # Merge the enhanced L back with the original A and B channels
    lab = cv2.merge([l_channel, a_channel, b_channel])
    img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)   # back to BGR

    """
    ┌─────────────────────────────────────────────────────────────┐
    │ STEP 2 – Gaussian Smoothing / Denoising  (Topic 5, Topic 6)│
    │                                                             │
    │ Topic 5 covered colour image smoothing by filtering each   │
    │ channel with a mask of all 1s (averaging).                 │
    │ Gaussian blur does the same but with a weighted mask       │
    │ (centre pixel gets more weight than edges).                │
    │ This removes random pixel noise from the scan.             │
    └─────────────────────────────────────────────────────────────┘
    """
    # (3,3) = tiny 3×3 kernel  |  sigmaX=0.5 = very light blur
    blurred = cv2.GaussianBlur(img_bgr, (3, 3), sigmaX=0.5)

    """
    ┌─────────────────────────────────────────────────────────────┐
    │ STEP 3 – Unsharp Mask (Edge Sharpening)  (Topic 5)         │
    │                                                             │
    │ Topic 5 showed colour image sharpening using spatial masks.│
    │ Unsharp masking = original − blurred version.              │
    │ This makes edges (gun outline, knife blade) sharper so the │
    │ model can detect them more easily.                         │
    │                                                             │
    │ Formula: result = 1.5 × original − 0.5 × blurred          │
    └─────────────────────────────────────────────────────────────┘
    """
    img_bgr = cv2.addWeighted(
        img_bgr, 1.5,    # original  × 1.5
        blurred, -0.5,   # blurred   × −0.5  (subtract)
        0                # no constant offset
    )

    return img_bgr

    """
    NOTE on binary masks (professor's guideline):
    The dataset annotation masks have pixel values in [0, 1].
    To VIEW them properly multiply by 255:
        mask_visible = mask_array * 255
    We do NOT need this for classification (Part 2), but Part 3
    (localisation) will need it.
    """


# ═══════════════════════════════════════════════════════════════════
#  2.  DATASET – loads images from folders and applies preprocessing
# ═══════════════════════════════════════════════════════════════════

class BaggageDataset(Dataset):
    """
    Reads images organised like this:

        dataset/
          train/
            safe/        ← images of safe bags
            gun/         ← images with guns
            knife/       ← images with knives
            shuriken/    ← images with shurikens
          val/
            safe/ gun/ knife/ shuriken/
          test/
            img1.jpg  img2.jpg  ...   ← no sub-folders on test day

    PyTorch's DataLoader calls __getitem__(index) repeatedly
    to feed batches of images into the model during training.
    """

    def __init__(self, root: Path, split: str,
                 transform=None, apply_classical: bool = True):
        self.samples         = []    # list of (image_path, label_number)
        self.transform       = transform
        self.apply_classical = apply_classical

        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Folder not found: {split_dir}")

        # Walk through each class folder and collect image paths
        for label_idx, class_name in enumerate(CLASS_NAMES):
            # label_idx: safe=0, gun=1, knife=2, shuriken=3
            class_folder = split_dir / class_name
            if not class_folder.exists():
                continue
            for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for img_path in class_folder.glob(extension):
                    self.samples.append((img_path, label_idx))

        random.shuffle(self.samples)
        print(f"[Dataset] {split:6s} → {len(self.samples)} images found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Read image with OpenCV (returns a numpy array in BGR format)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            # Fallback using Pillow if OpenCV fails
            img_bgr = np.array(Image.open(img_path).convert("RGB"))
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

        # Apply our classical CV preprocessing pipeline (Step 1-3 above)
        if self.apply_classical:
            img_bgr = classical_preprocess(img_bgr)

        # Convert BGR → RGB → PIL Image (PyTorch transforms expect PIL)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Apply torchvision transforms (resize, flip, normalise, etc.)
        if self.transform:
            img_pil = self.transform(img_pil)

        # Return the processed image tensor and its numeric label
        return img_pil, label


# ═══════════════════════════════════════════════════════════════════
#  3.  DATA TRANSFORMS  (what happens to each image before training)
# ═══════════════════════════════════════════════════════════════════

# These are the mean and std that ImageNet was normalised with.
# EfficientNet was pretrained on ImageNet so we must use the same values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Training transforms include random augmentations to make the model robust
train_transforms = transforms.Compose([
    # Resize slightly larger than 224 then crop randomly → simulates zoom
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),

    # Randomly flip left-right (a gun is still a gun when mirrored)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),

    # Randomly change brightness/contrast (like Topic 5 intensity manipulation)
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),

    # Rotate up to 15 degrees (handles tilted baggage scans)
    transforms.RandomRotation(degrees=15),

    # Convert PIL image → PyTorch tensor (pixel values become 0.0 to 1.0)
    transforms.ToTensor(),

    # Standardise pixel values using ImageNet mean/std
    # Formula: output = (pixel - mean) / std
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation transforms: NO random augmentation – we test on clean images
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ═══════════════════════════════════════════════════════════════════
#  4.  BUILD THE MODEL
#      EfficientNet-B0 pretrained → replace final layer for 4 classes
# ═══════════════════════════════════════════════════════════════════

def build_model(num_classes: int = NUM_CLASSES,
                freeze_backbone: bool = False) -> nn.Module:
    """
    Transfer Learning approach:
      1. Download EfficientNet-B0 already trained on 1.2M ImageNet images.
         It already knows how to detect shapes, edges, textures.
      2. Replace its output layer (originally 1000 classes) with our
         own layer for 4 classes (safe / gun / knife / shuriken).
      3. Optionally freeze the backbone so only our new layer trains
         (used in warm-up phase to avoid destroying pretrained weights).
    """
    # Load EfficientNet-B0 with pretrained ImageNet weights
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)

    # Freeze all layers if requested (backbone won't update its weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False   # requires_grad=False → no gradient → no update

    # Replace the classifier head
    # Original: Dropout → Linear(1280 → 1000)
    # Ours:     Dropout → Linear(1280 → 512) → ReLU → Dropout → Linear(512 → 4)
    in_features = model.classifier[1].in_features   # 1280 features from backbone

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),              # randomly zero 30% of neurons → less overfitting
        nn.Linear(in_features, 512),    # 1280 inputs → 512 outputs
        nn.ReLU(inplace=True),          # activation: keeps positive values, zeroes negatives
        nn.Dropout(p=0.2),              # another dropout layer
        nn.Linear(512, num_classes),    # 512 → 4 final class scores (called logits)
    )

    return model.to(DEVICE)   # move model to GPU (if available) or CPU


# ═══════════════════════════════════════════════════════════════════
#  5.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_model(model, dataloaders, criterion, optimizer,
                scheduler, num_epochs: int = NUM_EPOCHS):
    """
    The core learning loop. For each epoch:
      1. Show the model a batch of images
      2. Model makes a prediction
      3. Compute how wrong it was (loss)
      4. Backpropagate: figure out which weights caused the error
      5. Update weights using the optimizer

    Also includes:
      • Best-model saving (checkpointing)
      • Early stopping if accuracy stops improving
    """
    # Track loss and accuracy over epochs for plotting
    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    best_weights = copy.deepcopy(model.state_dict())  # copy of current weights
    best_acc     = 0.0
    patience     = 7    # stop if no improvement for 7 epochs in a row
    no_improve   = 0

    for epoch in range(num_epochs):
        t_start = time.time()
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 45)

        # Each epoch has a training phase and a validation phase
        for phase in ("train", "val"):

            # Switch model mode:
            # train() → dropout active, weights update
            # eval()  → dropout off,    weights frozen (just measuring)
            model.train() if phase == "train" else model.eval()

            running_loss    = 0.0
            running_correct = 0

            for images, labels in dataloaders[phase]:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()   # clear gradients from previous batch

                # Forward pass (with gradient tracking only during training)
                with torch.set_grad_enabled(phase == "train"):
                    # Model outputs raw scores (logits) for each of the 4 classes
                    logits = model(images)

                    # CrossEntropyLoss: measures how wrong the prediction was
                    # (combines Softmax + Negative Log Likelihood internally)
                    loss = criterion(logits, labels)

                    # Pick the class with the highest score as our prediction
                    predictions = logits.argmax(dim=1)

                    if phase == "train":
                        loss.backward()     # compute gradients (backpropagation)
                        optimizer.step()    # update weights using those gradients

                running_loss    += loss.item() * images.size(0)
                running_correct += (predictions == labels).sum().item()

            # Average loss and accuracy for this epoch/phase
            n           = len(dataloaders[phase].dataset)
            epoch_loss  = running_loss    / n
            epoch_acc   = running_correct / n

            print(f"  {phase.capitalize():6s} | "
                  f"Loss: {epoch_loss:.4f}  Acc: {epoch_acc*100:.2f}%")

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            # ── Save best model checkpoint ──────────────────────────
            if phase == "val" and epoch_acc > best_acc:
                best_acc     = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                no_improve   = 0
                ckpt_path    = CHECKPOINT_DIR / "best_model.pth"
                torch.save({
                    "epoch"      : epoch + 1,
                    "model_state": best_weights,
                    "val_acc"    : best_acc,
                    "class_names": CLASS_NAMES,
                }, ckpt_path)
                print(f"  ✔  Best val acc: {best_acc*100:.2f}% → saved {ckpt_path}")
            elif phase == "val":
                no_improve += 1

        # Step the learning-rate scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

        print(f"  Time: {time.time()-t_start:.1f}s")

        # ── Early stopping ──────────────────────────────────────────
        if no_improve >= patience:
            print(f"\n[Early Stop] No val improvement for {patience} epochs. Stopping.")
            break

    # Load the best weights back into the model before returning
    model.load_state_dict(best_weights)
    print(f"\n[INFO] Training done. Best val acc: {best_acc*100:.2f}%")
    return model, history


# ═══════════════════════════════════════════════════════════════════
#  6.  EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate(model, dataloader, split_name: str = "Validation"):
    """
    Measures how well the model performs using the rubric metrics:
      • Accuracy
      • Macro F1-Score  (important when class sizes differ – e.g. fewer
        shuriken images than gun images)
      • Classification Score = 0.7 × Accuracy + 0.3 × Macro F1  (from rubric)

    Also plots a Confusion Matrix – rows = true labels, cols = predicted.
    This is useful to see WHICH classes are being confused with each other.
    (e.g. if the model often confuses knife for shuriken)
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():    # no gradients needed during evaluation
        for images, labels in dataloader:
            images = images.to(DEVICE)
            preds  = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc      = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # This is the exact formula from the project rubric
    cls_score = 0.7 * acc + 0.3 * macro_f1

    print(f"\n{'='*50}")
    print(f"  {split_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy              : {acc*100:.2f}%")
    print(f"  Macro F1-Score        : {macro_f1:.4f}")
    print(f"  Classification Score  : {cls_score:.4f}  (0.7×acc + 0.3×F1)")
    print(f"\n{classification_report(all_labels, all_preds, target_names=CLASS_NAMES)}")

    # ── Confusion Matrix ────────────────────────────────────────────
    # Related to Topic 7 (segmentation evaluation concepts)
    # Each cell [i][j] = how many class-i images were predicted as class-j
    # Diagonal = correct predictions
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix — {split_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    cm_path = CHECKPOINT_DIR / f"confusion_matrix_{split_name.lower()}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {cm_path}")

    return acc, macro_f1, cls_score


def plot_training_curves(history: dict):
    """Save loss and accuracy curves so you can see how training went."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [a*100 for a in history["train_acc"]], label="Train")
    axes[1].plot(epochs, [a*100 for a in history["val_acc"]],   label="Val")
    axes[1].set_title("Accuracy per Epoch (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    plt.tight_layout()
    path = CHECKPOINT_DIR / "training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved → {path}")


# ═══════════════════════════════════════════════════════════════════
#  7.  INFERENCE  –  used on evaluation day
# ═══════════════════════════════════════════════════════════════════

def predict_folder(model, folder: Path,
                   output_csv: Path = Path("predictions.csv")):
    """
    On evaluation day:
      1. Put test images in dataset/test/
      2. This function runs the model on each image
      3. Writes predictions.csv in the exact format the rubric requires:
            Image Name | Predicted Label
    """
    model.eval()
    rows = []

    img_paths = sorted([
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for p in folder.glob(ext)
    ])

    if not img_paths:
        print(f"[WARN] No images found in {folder}")
        return

    for img_path in img_paths:
        # Apply the same preprocessing as training (important!)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [SKIP] Cannot read {img_path.name}")
            continue

        img_bgr = classical_preprocess(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Convert to tensor, add batch dimension (model expects [B, C, H, W])
        tensor = val_transforms(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits     = model(tensor)
            pred_idx   = logits.argmax(dim=1).item()   # index of highest score
            pred_label = CLASS_NAMES[pred_idx]         # e.g. "gun"

        rows.append({"Image Name": img_path.name,
                     "Predicted Label": pred_label})

    # Write CSV in the format required by the rubric
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image Name", "Predicted Label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Predictions written → {output_csv}  ({len(rows)} images)")


# ═══════════════════════════════════════════════════════════════════
#  8.  MAIN  –  runs everything in order
# ═══════════════════════════════════════════════════════════════════

def set_seed(seed: int = SEED):
    """Make results reproducible across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed()

    # ── Step 1: Load datasets ──────────────────────────────────────
    train_dataset = BaggageDataset(DATA_ROOT, "train", transform=train_transforms)
    val_dataset   = BaggageDataset(DATA_ROOT, "val",   transform=val_transforms)

    # DataLoader feeds batches of images to the model automatically
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    dataloaders = {"train": train_loader, "val": val_loader}

    # ── Step 2: Loss function ──────────────────────────────────────
    # CrossEntropyLoss = standard loss for multi-class classification
    # label_smoothing=0.1 → instead of "100% gun", use "90% gun, 3.3% others"
    # This prevents the model from becoming overconfident
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Step 3: PHASE 1 – Train only the new head (backbone frozen) ─
    # This is a warm-up: we let the new 4-class layer learn first
    # before we touch the pretrained EfficientNet weights.
    print("\n" + "="*50)
    print("PHASE 1: Warm-up (backbone frozen, head only)")
    print("="*50)
    model = build_model(freeze_backbone=True)

    # Adam optimizer – only optimises parameters that require gradients
    # (the backbone is frozen so only the head parameters are here)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    model, _ = train_model(model, dataloaders, criterion,
                           optimizer, scheduler=None, num_epochs=5)

    # ── Step 4: PHASE 2 – Fine-tune the whole network ──────────────
    # Now we unfreeze everything and train with a smaller learning rate
    # so we don't destroy the pretrained weights we warmed up
    print("\n" + "="*50)
    print("PHASE 2: Full fine-tuning (all layers unfrozen)")
    print("="*50)
    for param in model.parameters():
        param.requires_grad = True   # unfreeze backbone

    # AdamW = Adam with better weight decay handling
    # LR/10 = 0.0001 – smaller steps so we don't overwrite good weights
    optimizer = optim.AdamW(model.parameters(),
                            lr=LR / 10, weight_decay=WEIGHT_DECAY)

    # CosineAnnealingLR: gradually reduces learning rate in a cosine curve
    # so training slows down smoothly as we approach convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    model, history = train_model(model, dataloaders, criterion,
                                 optimizer, scheduler, num_epochs=NUM_EPOCHS)

    # ── Step 5: Plot and evaluate ──────────────────────────────────
    plot_training_curves(history)
    evaluate(model, val_loader, split_name="Validation")

    # ── Step 6: Predict test set (evaluation day) ──────────────────
    test_dir = DATA_ROOT / "test"
    if test_dir.exists():
        print("\n[INFO] Running inference on test folder ...")
        predict_folder(model, test_dir, output_csv=Path("predictions.csv"))
    else:
        print(f"\n[INFO] No test folder found at '{test_dir}'.")
        print("       On evaluation day: put test images in dataset/test/ and re-run.")

    print("\n[DONE] All output files:")
    print("  • checkpoints/best_model.pth          ← trained model")
    print("  • checkpoints/training_curves.png     ← loss & accuracy plots")
    print("  • checkpoints/confusion_matrix_*.png  ← which classes were confused")
    print("  • predictions.csv                     ← submit this on evaluation day")


if __name__ == "__main__":
    main()