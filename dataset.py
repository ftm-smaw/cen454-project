import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split


def verify_and_load_dataset(image_dir):
    """
    Scans the data directories, validates the images using OpenCV,
    and structures them into a data framework.
    """
    data_records = []
    # Match the project's explicit four target classes
    categories = ['safe', 'gun', 'knife', 'shuriken']

    for category in categories:
        category_path = os.path.join(image_dir, category)
        if not os.path.exists(category_path):
            print(f"Skipping: Directory '{category}' not found in {image_dir}")
            continue

        print(f"Processing category: {category}...")
        for img_name in os.listdir(category_path):
            # Ignore hidden operating system files (.DS_Store, etc.)
            if img_name.startswith('.'):
                continue

            img_path = os.path.join(category_path, img_name)

            # Verify if OpenCV can open the image without errors
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Warning: Skipping invalid image asset: {img_name}")
                continue

            h, w, c = img.shape
            data_records.append({
                'Image Name': img_name,  # Required submission header name
                'image_path': img_path,
                'Predicted Label': category,  # Required submission header name
                'height': h,
                'width': w
            })

    df = pd.DataFrame(data_records)
    print(f"\nTotal valid images found: {len(df)}")
    return df


def split_dataset(df):
    """
    Splits the data into 80% training and 20% validation sets.
    Uses stratification to ensure classes remain balanced.
    """
    # stratify ensures an equal mix of gun, safe, knife, shuriken in both splits
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['Predicted Label'],
        random_state=42
    )
    print(f"Data Split: {len(train_df)} training samples | {len(val_df)} validation samples.")
    return train_df, val_df


if __name__ == "__main__":
    # Point to the data directory you just created
    RAW_IMAGE_DIR = os.path.join("data")
    PROCESSED_DIR = os.path.join("data", "processed")

    # Run the validation and loading pipeline
    dataset_df = verify_and_load_dataset(RAW_IMAGE_DIR)

    if not dataset_df.empty:
        # Create structural train/validation splits
        train_data, val_data = split_dataset(dataset_df)

        # Save splits as metadata CSVs so your teammates can easily load them
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        train_data.to_csv(os.path.join(PROCESSED_DIR, "train_split.csv"), index=False)
        val_data.to_csv(os.path.join(PROCESSED_DIR, "val_split.csv"), index=False)
        print(f"\nSuccess! Split metadata sheets saved into: {PROCESSED_DIR}")
    else:
        print("\nError: No images were found. Please verify your data/ folder contents.")