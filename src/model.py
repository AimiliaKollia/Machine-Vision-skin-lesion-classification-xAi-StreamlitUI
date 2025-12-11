import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import resample  # Needed for upsampling arrays
from . import config, plots, explainability, features


def balance_training_set(X_train, y_train):
    """
    Upsamples the TRAINING set only.
    Duplicates samples from minority classes to match the majority class count.
    """
    print("Balancing Training Set (Upsampling)...")

    # 1. Identify unique classes and the target count (Majority class size)
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    max_count = class_counts.max()

    X_balanced = []
    y_balanced = []

    # 2. Resample each class
    for cls in unique_classes:
        # Get indices of all samples belonging to this class
        indices = np.where(y_train == cls)[0]

        X_subset = X_train[indices]
        y_subset = y_train[indices]

        # Upsample (replace=True means duplicate)
        X_resampled, y_resampled = resample(
            X_subset, y_subset,
            replace=True,
            n_samples=max_count,
            random_state=42
        )

        X_balanced.append(X_resampled)
        y_balanced.append(y_resampled)

    # 3. Combine and Shuffle
    X_final = np.vstack(X_balanced)
    y_final = np.hstack(y_balanced)

    # Shuffle to ensure classes aren't grouped sequentially
    X_final, y_final = resample(X_final, y_final, replace=False, random_state=42)

    print(f"  Training set size increased from {len(X_train)} to {len(X_final)}")
    return X_final, y_final


def train_and_evaluate(X, y, classes):
    """
    Trains models, generates extensive performance plots (ROC, CM),
    creates Explainable AI plots (Feature Importance), and saves the best model.
    """

    # 1. Define Models
    # Note: We keep class_weight='balanced' as an extra safety measure
    techniques = {
        "RF": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "SVM": SVC(probability=True, class_weight='balanced', random_state=42)
    }

    best_score = 0
    best_model = None
    best_scaler = None
    best_name = ""

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 2. Split Data FIRST (Critical for preventing Data Leakage)
    print(f"Splitting data (Total samples: {len(y)})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 3. Upsample ONLY the Training Data
    # The Test set remains imbalanced to reflect real-world difficulty
    X_train_bal, y_train_bal = balance_training_set(X_train, y_train)

    # 4. Scaling
    # Fit scaler on BALANCED training data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_bal)
    # Transform Test data using the same scaler
    X_test_s = scaler.transform(X_test)

    # 5. Training Loop
    for name, model in techniques.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_s, y_train_bal)
        preds = model.predict(X_test_s)
        acc = accuracy_score(y_test, preds)

        print(f"--> {name} Accuracy on Test Set: {acc:.4f}")

        # --- PLOTTING METRICS ---
        print(f"Generating ROC and Confusion Matrix for {name}...")
        plots.plot_confusion_matrix(y_test, preds, classes, name)
        plots.plot_multiclass_roc(model, X_test_s, y_test, classes, name)
        plots.save_classification_report(y_test, preds, classes, name)

        # --- EXPLAINABLE AI (RF Only) ---
        if name == "RF":
            print("Generating Feature Importance Plot (XAI)...")
            feature_names = features.get_feature_names()
            if len(feature_names) != X_train.shape[1]:
                feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
            explainability.plot_rf_feature_importance(model, feature_names)

        # Track Best
        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name
            best_scaler = scaler

            # Save Artifacts
    print(f"\nSaving Best Model: {best_name}")
    joblib.dump(best_model, os.path.join(config.MODEL_DIR, 'skin_cancer_model.pkl'))
    joblib.dump(best_scaler, os.path.join(config.MODEL_DIR, 'scaler.pkl'))
    joblib.dump(classes, os.path.join(config.MODEL_DIR, 'classes.pkl'))