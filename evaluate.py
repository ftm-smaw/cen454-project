import os
import csv
from inference import load_model, run_inference

CHECKPOINT = "./trained_model.pth"
TEST_DIR   = "./Test_data"

# Only use annotations if the folder actually exists on this machine
ANNOTATION_DIR = "./data/annotations" if os.path.isdir("./data/annotations") else None

# Load model
model = load_model(CHECKPOINT)

# Check test folder exists and has images
if not os.path.isdir(TEST_DIR):
    print(f"ERROR: Test folder '{TEST_DIR}' not found. Please create it and add test images.")
    exit(1)

image_files = [f for f in os.listdir(TEST_DIR)
               if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}]

if len(image_files) == 0:
    print(f"ERROR: No images found in '{TEST_DIR}'.")
    exit(1)

# Run inference
results = run_inference(model=model, test_dir=TEST_DIR, annotation_dir=ANNOTATION_DIR)

# predictions.csv — saved to project root
with open("predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ImageName", "PredictedLabel"])
    for r in results:
        writer.writerow([r["image_name"], r["label"]])
print("predictions.csv saved.")

# localization.csv — saved to project root
threat_results = [r for r in results if r["bbox"] is not None]
with open("localization.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ImageName", "Label", "X_min", "Y_min", "X_max", "Y_max"])
    for r in threat_results:
        writer.writerow([
            r["image_name"],
            r["label"],
            r["bbox"][0],
            r["bbox"][1],
            r["bbox"][2],
            r["bbox"][3]
        ])
print("localization.csv saved.")

# IoU summary — only printed if annotations were available
iou_scores = [r["iou"] for r in threat_results if r.get("iou") is not None]
if iou_scores:
    avg_iou = sum(iou_scores) / len(iou_scores)
    valid = sum(1 for s in iou_scores if s >= 0.5)
    print(f"Average IoU      : {avg_iou:.4f}")
    print(f"Valid detections : {valid}/{len(iou_scores)} (IoU >= 0.5)")
    print(f"Localization Score : {avg_iou:.4f}")