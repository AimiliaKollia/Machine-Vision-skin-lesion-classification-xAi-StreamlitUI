import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from . import config, plots, explainability, features


def train_and_evaluate_split(X_train, y_train, X_test, y_test, classes):
    """
    Accepts PRE-SPLIT and PRE-AUGMENTED data.
    Trains models, generates plots, and saves artifacts.
    """

    # 1. Define Models
    techniques = {
        "RF": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "SVM": SVC(probability=True, class_weight='balanced', random_state=42)
    }

    best_score = 0
    best_model = None
    best_scaler = None
    best_name = ""

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 2. Prepare Feature Names for XAI
    # We fetch these once so we can use them for LIME (all models) and RF Importance
    feature_names = features.get_feature_names()
    # Safety fallback if feature count mismatches name list
    if len(feature_names) != X_train.shape[1]:
        print(f"Warning: Feature names count ({len(feature_names)}) != Data columns ({X_train.shape[1]})")
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    # 3. Scaling
    # Important: Fit on Train, Transform Test
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- NEW: Save Training Sample for LIME in App ---
    # We save a subset (e.g., 500 samples) to keep the app lightweight and fast.
    # LIME needs this to understand the "background" distribution of features.
    print("Saving training sample for App LIME initialization...")
    if X_train_s.shape[0] > 500:
        indices = np.random.choice(X_train_s.shape[0], 500, replace=False)
        X_sample = X_train_s[indices]
    else:
        X_sample = X_train_s
    np.save(os.path.join(config.MODEL_DIR, 'X_train_sample.npy'), X_sample)
    # -------------------------------------------------

    # 4. Training Loop
    for name, model in techniques.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        acc = accuracy_score(y_test, preds)

        print(f"--> {name} Accuracy on Test Set: {acc:.4f}")

        # --- PLOTTING METRICS ---
        print(f"Generating ROC and Confusion Matrix for {name}...")
        plots.plot_confusion_matrix(y_test, preds, classes, name)
        plots.plot_multiclass_roc(model, X_test_s, y_test, classes, name)
        plots.save_classification_report(y_test, preds, classes, name)

        # --- EXPLAINABLE AI (Global: Feature Importance) ---
        if name == "RF":
            print("Generating Global Feature Importance Plot (RF)...")
            explainability.plot_rf_feature_importance(model, feature_names)

        # --- EXPLAINABLE AI (Local: LIME) ---
        # This works for BOTH RF and SVM
        print(f"Generating Local LIME Explanations for {name}...")
        explainability.generate_lime_explanations(
            model=model,
            X_train=X_train_s,  # LIME needs training distribution
            X_test=X_test_s,  # Instances to explain
            y_test=y_test,  # For labeling plots
            feature_names=feature_names,
            class_names=classes,
            model_name=name
        )

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