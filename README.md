CEN454 – Computer Vision and Machine Learning
Baggage Threat Detection System
=====================================

PYTHON VERSION
--------------
Python 3.11 is required as PyTorch does not support Python 3.12 or above.

IMPORTANT: numpy must be pinned to 1.x (see requirements.txt, we used version 1.26.4).
Make sure not to upgrade numpy as it will conflict with PyTorch.

SETUP
-----
1. Create a virtual environment with Python 3.11:

    python3.11 -m venv .venv
    source .venv/bin/activate        (Mac/Linux)
    .venv\Scripts\activate           (Windows)

2. Install dependencies:

    pip install -r requirements.txt

HOW TO RUN
----------
1. Place test images flat inside the Test_data/ folder (no subfolders)
2. Make sure trained_model.pth is in the project root
3. Run:

    python evaluate.py

4. All outputs saved to outputs/:
    predictions.csv    → ImageName, PredictedLabel
    localization.csv   → ImageName, Label, X_min, Y_min, X_max, Y_max
--> this includes best_model.pth, which is copied into the root folder as trained_model.pth for use in evaluate.py

PROJECT STRUCTURE
-----------------
    evaluate.py       Main entry point
    train.py          EfficientNet-B0 training pipeline
    inference.py      GradCAM localization and IoU scoring
    dataset.py        Dataset validation utility (optional)
    trained_model.pth    Trained model weights
    requirements.txt  Package dependencies
    Test_data/        Place test images here

TRAINING RESULTS (validation set)
----------------------------------
    Accuracy          : 0.7036
    Macro F1-Score    : 0.8172
    Classification Score : 0.7377