# src/config.py
import os

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_FOLDER = os.path.join(base_dir, 'data', 'images')
CSV_PATH = os.path.join(base_dir, 'data', 'GroundTruth.csv')
MODEL_DIR = os.path.join(base_dir, 'models')

# Constants
IMG_SIZE = 224
CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# Hyperparameters
BLUR_KERNEL = (9, 9)
MORPH_OPEN_KERNEL = (5, 5)   # Increase to remove more noise but risk losing information
MORPH_DILATE_KERNEL = (5, 5)

# CLAHE Settings (Smart Equalization)
CLAHE_CLIP = 2.0             # Threshold for contrast limiting
CLAHE_GRID = (8, 8)          # Grid size for local equalization

# Histogram Config
HIST_BINS = 8

LEGEND_DATA = {
    "Abbreviation": ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
    "Full Diagnosis": [
        "Melanoma",
        "Melanocytic nevus",
        "Basal cell carcinoma",
        "Actinic keratoses",
        "Benign keratosis-like lesions",
        "Dermatofibroma",
        "Vascular lesions"
    ],
    "Description": [
        "Malignant skin tumor (Cancerous).",
        "Benign melanocytic proliferations (Moles).",
        "Common variant of skin cancer (Cancerous).",
        "Pre-cancerous skin lesions.",
        "Non-cancerous skin growths (e.g., solar lentigines).",
        "Benign skin lesion (nodules).",
        "Benign blood vessel lesions."
    ],
    "More Info": [
        "https://en.wikipedia.org/wiki/Melanoma",
        "https://en.wikipedia.org/wiki/Melanocytic_nevus",
        "https://en.wikipedia.org/wiki/Basal-cell_carcinoma",
        "https://en.wikipedia.org/wiki/Actinic_keratosis",
        "https://en.wikipedia.org/wiki/Seborrheic_keratosis",
        "https://en.wikipedia.org/wiki/Dermatofibroma",
        "https://en.wikipedia.org/wiki/Cherry_angioma"
    ]
}