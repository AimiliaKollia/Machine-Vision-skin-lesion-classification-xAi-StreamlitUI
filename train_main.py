import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from src import data, features, model, augmentation


def process_dataset_with_augmentation(df, is_training=False):
    """
    Loops through the dataframe.
    If is_training=True, it augments minority classes to balance the data.
    """
    X = []
    y = []

    # 1. Calculate Statistics for Balancing (Only needed for training)
    if is_training:
        class_counts = df['target'].value_counts().to_dict()
        max_count = max(class_counts.values())
        print(f"  [Augmentation] Balancing classes to match majority count: {max_count}")

    total = len(df)

    for idx, row in df.iterrows():
        if idx % 100 == 0: print(f"  Processing image {idx}/{total}...")

        # Load Original Image
        img = cv2.imread(row['path'])
        if img is None: continue

        # A. Extract Features for Original Image
        feats = features.extract_all_features_pipeline(img)
        if feats is not None:
            X.append(feats)
            y.append(row['label_idx'])

        # B. Augmentation Logic (Training Only)
        if is_training:
            # Check how many extra copies we need
            class_name = row['target']

            # Calculate factor. e.g., if factor is 5, we generate 4 NEW images
            # so total = 1 original + 4 augmented = 5
            factor = augmentation.get_augmentation_factor(class_name, class_counts, max_count)
            num_new_images = factor - 1

            if num_new_images > 0:
                # Generate variations
                aug_imgs = augmentation.generate_augmented_images(img, count=num_new_images)

                # Extract features for every augmented variation
                for aug_img in aug_imgs:
                    aug_feats = features.extract_all_features_pipeline(aug_img)
                    if aug_feats is not None:
                        X.append(aug_feats)
                        y.append(row['label_idx'])

    return np.array(X), np.array(y)


def main():
    # 1. Load Data (Metadata only)
    df, classes = data.load_metadata(limit=None)  # Adjust limit as needed

    print("-" * 50)
    print("STEP 1: Splitting Data (Train/Test) on File Paths")
    print("-" * 50)

    # Split DataFrame FIRST to avoid data leakage
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df['label_idx'], random_state=42
    )

    print(f"Training Samples (Files): {len(df_train)}")
    print(f"Test Samples (Files): {len(df_test)}")

    # 2. Process Test Data (No Augmentation, just feature extraction)
    print("\n" + "-" * 50)
    print("STEP 2: Extracting Test Features (Standard)")
    print("-" * 50)
    X_test, y_test = process_dataset_with_augmentation(df_test, is_training=False)

    # 3. Process Training Data (WITH Augmentation)
    print("\n" + "-" * 50)
    print("STEP 3: Extracting Training Features (With Keras Augmentation)")
    print("-" * 50)
    X_train, y_train = process_dataset_with_augmentation(df_train, is_training=True)

    print(f"\nFinal Feature Matrix Shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # 4. Train & Evaluate
    # We pass the pre-split arrays directly to a modified train function
    if len(X_train) > 0 and len(X_test) > 0:
        model.train_and_evaluate_split(X_train, y_train, X_test, y_test, classes)
    else:
        print("Error: Feature extraction failed.")


if __name__ == "__main__":
    main()