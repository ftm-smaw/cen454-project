import os
import csv
from train import CLASS_NAMES, DEVICE
from inference import load_model, run_inference

CHECKPOINT     = "./trained_model.pth"
TEST_DIR       = "./Test_data"
ANNOTATION_DIR = "./data/annotations"
OUTPUT_DIR     = "./outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#loads model from saved weights
model = load_model(CHECKPOINT)

# prediction and localization are in this function
results = run_inference(model=model, test_dir=TEST_DIR, annotation_dir=ANNOTATION_DIR if os.path.isdir(ANNOTATION_DIR) else None
)

predictions_path = os.path.join(OUTPUT_DIR, "predictions.csv")
with open(predictions_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ImageName", "PredictedLabel"])
    for r in results:
        writer.writerow([r["image_name"], r["label"]])
print(f"Predictions saved → {predictions_path}")

localization_path = os.path.join(OUTPUT_DIR, "localization.csv")
threat_results = [r for r in results if r["bbox"] is not None]
with open(localization_path, "w", newline="") as f:
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
print(f"Localization saved → {localization_path}")

iou_scores = [r["iou"] for r in threat_results if r["iou"] is not None]
if iou_scores:
    avg_iou = sum(iou_scores) / len(iou_scores)
    valid = sum(1 for s in iou_scores if s >= 0.5)
    print(f"Average IoU        : {avg_iou:.4f}")
    print(f"Valid detections   : {valid}/{len(iou_scores)} (IoU ≥ 0.5)")
    print(f"Localization Score : {avg_iou:.4f}")
else:
    print("No IoU scores computed (no annotations or no threats detected)")