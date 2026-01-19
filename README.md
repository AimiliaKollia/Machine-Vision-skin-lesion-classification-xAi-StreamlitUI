# By Kollia Aimilia, Kontoudakis Nikos, Skiada Kyriaki, Lampropoulou Nancy
The project aims to classify skin lesions.

# Project Structure
The project follows a modular structure to separate configuration, data processing, computer vision logic, and model training.

```text
ham10000_project/
│
├── data/
│   ├── images/              # Original dermoscopy images (e.g., ISIC_0024306.jpg)
│   └── GroundTruth.csv      # Metadata and labels
│
├── models/                  # Generated automatically during training
│   ├── skin_cancer_model.pkl # Trained Random Forest/SVM model
│   ├── scaler.pkl           # StandardScaler for feature normalization
│   ├── classes.pkl          # List of class names (MEL, NV, etc.)
│   └── comparison_results.png # Confusion matrix plot
│
├── src/                     # Core Logic Package
│   ├── __init__.py          # Makes this folder a Python package
│   ├── config.py            # Configuration, Constants, and Hyperparameters
│   ├── data.py              # Data loading and stratified splitting logic
│   ├── features.py          # Computer Vision pipeline (CLAHE, Otsu, Sobel, etc.)
│   └── model.py             # Model training, evaluation, and saving
│
├── train_main.py            # Script 1: Main entry point to train the model
├── app.py                   # Script 2: Streamlit Web Interface for inference
└── requirements.txt         # Project dependencies

```

# Interpretable Skin Lesion Classification

This project presents a classical computer vision pipeline combined with machine learning classifiers to classify skin lesions into 7 diagnostic categories using the HAM10000 dataset from Kaggle.
The focus is on interpretability and transparency through handcrafted feature extraction and explainable AI techniques.

## Features
- Preprocessing: cropping, resizing, grayscale conversion, CLAHE, Gaussian blur
- Lesion segmentation using Otsu thresholding and morphological operations
- Handcrafted feature extraction (color, shape, texture)
- Classification using: Random Forest and SVM
- Class imbalance handling using augmentation and class weighting
- Explainability via feature importance and LIME (xAI)

## Dataset
The project uses the HAM10000 dataset, containing 10,015 dermatoscopic images of seven lesion types.

## Results
Random Forest achieved higher accuracy than SVM. Explainability analysis showed that color histogram features were the most influential.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- scikit-learn
- LIME

 


