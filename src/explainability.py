import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import lime
import lime.lime_tabular
from . import config


def plot_rf_feature_importance(model, feature_names):
    """
    Plots and saves the Feature Importance for a Random Forest model.
    """
    if not hasattr(model, 'feature_importances_'):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]

    plt.figure(figsize=(14, 8))
    plt.title("Random Forest: Feature Importance (Global XAI)")
    plt.bar(range(len(importances)), importances[indices], align="center", color='teal')
    plt.xticks(range(len(importances)), sorted_names, rotation=90)
    plt.xlim([-1, len(importances)])
    plt.ylabel("Relative Importance")
    plt.tight_layout()

    save_path = os.path.join(config.MODEL_DIR, 'rf_feature_importance.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Global XAI Plot saved to {save_path}")


def generate_lime_explanations(model, X_train, X_test, y_test, feature_names, class_names, model_name, num_samples=3):
    """
    Generates LIME (Local Interpretable Model-agnostic Explanations) for specific test instances.
    This works for ANY model (RF, SVM, etc.).
    """
    print(f"  Initializing LIME Explainer for {model_name}...")

    # 1. Initialize Explainer
    # LIME needs the training data to learn the distribution of features (mean, std, etc.)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        verbose=False
    )

    # 2. Pick sample indices to explain
    # We pick evenly spaced samples from the test set to get a variety
    indices = np.linspace(0, len(X_test) - 1, num_samples, dtype=int)

    output_dir = os.path.join(config.MODEL_DIR, 'lime_explanations')
    os.makedirs(output_dir, exist_ok=True)

    for i in indices:
        # 3. Generate Explanation
        # LIME perturbs this specific instance and sees how the model's prediction changes
        exp = explainer.explain_instance(
            data_row=X_test[i],
            predict_fn=model.predict_proba,
            num_features=10,
            top_labels=1
        )

        # 4. Save Plot
        # We title it with the True Label for context
        true_label = class_names[y_test[i]]
        fig = exp.as_pyplot_figure()
        plt.title(f"LIME ({model_name}): Test Instance {i} | True Label: {true_label}")
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'{model_name}_inst_{i}_lime.png')
        plt.savefig(save_path)
        plt.close()

    print(f"  LIME explanations saved to {output_dir}")