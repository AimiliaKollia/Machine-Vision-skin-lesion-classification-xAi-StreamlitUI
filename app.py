# ==========================================
# FILE 2: app.py (Web GUI)
# ==========================================
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew


def extract_features_deployment(image_array):
    IMG_SIZE = 224
    img = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))

    features = []

    # Color Stats
    for i in range(3):
        channel = img[:, :, i]
        features.append(np.mean(channel))
        features.append(np.std(channel))
        features.append(skew(channel.flatten()))

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Increased Blur to remove texture
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Tuned Adaptive Threshold
    # Block Size: 99 (Look at large area), C: 15 (Be strict about what is dark)
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 99, 15)

    # Morphological Operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Shape & Largest Component
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area, perimeter, compactness = 0, 0, 0

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area > 50:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)

    features.extend([area, perimeter, compactness])

    # Texture (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    features.append(np.mean(mag))

    return np.array(features).reshape(1, -1)


# --- App Layout ---
st.set_page_config(page_title="DermAI Classification", layout="centered")

st.title("🔬 Skin Lesion Classification")
st.markdown("""
This system uses **Classical Machine Vision** techniques (Adaptive Thresholding, Morphology, Sobel Edges) 
to classify skin lesions.
""")

# Load Model
try:
    model = joblib.load('models/skin_cancer_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    classes = joblib.load('models/classes.pkl')
    st.success("System Ready: Model Loaded Successfully")
except FileNotFoundError:
    st.error("Model files not found. Please run 'train_model.py' first.")
    st.stop()

# File Upload
uploaded_file = st.file_uploader("Choose a dermoscopy image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display Original
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)

    # Process for Prediction
    with st.spinner('Extracting Handcrafted Features...'):
        features = extract_features_deployment(image)
        features_scaled = scaler.transform(features)
        probs = model.predict_proba(features_scaled)
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]

    # Results
    with col2:
        st.subheader(f"Prediction: **{pred_label}**")
        st.metric("Confidence", f"{probs[0][pred_idx] * 100:.2f}%")

    # Detailed Chart
    st.subheader("Class Probabilities")
    chart_data = pd.DataFrame({
        "Class": classes,
        "Probability": probs[0] * 100
    })
    st.bar_chart(chart_data.set_index("Class"))

    # --- UPDATED Feature Visualization (Explainability) ---
    with st.expander("See Internal Logic (Computer Vision Pipeline Steps)", expanded=False):
        st.info("Below are the exact transformations the system performs to extract numeric features from the image.")

        IMG_SIZE = 224

        # --- Step 1 & 2 & 3 ---
        col_prep1, col_prep2, col_prep3 = st.columns(3)

        # 1. Resize
        img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        col_prep1.image(img_resized, channels="BGR", caption="1. Resized", use_container_width=True)

        # 2. Gray
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        col_prep2.image(img_gray, caption="2. Grayscale", use_container_width=True)

        # 3. Blur
        # Updated to (9, 9)
        img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)
        col_prep3.image(img_blur, caption="3. Strong Blur (9x9)", use_container_width=True)

        st.divider()

        # --- Step 4: Segmentation (Adaptive) ---
        st.markdown("### 4. Adaptive Thresholding (Tuned)")
        st.write("Using BlockSize=99 and C=15 to ignore skin texture.")
        mask_raw = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 99, 15)
        st.image(mask_raw, caption="Raw Adaptive Mask")

        st.divider()

        # --- Step 5: Morphology ---
        st.markdown("### 5. Morphological Operations (Cleaning)")
        st.write("Removing noise using Opening (Erosion followed by Dilation).")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel, iterations=2)
        st.image(mask_clean, caption="Cleaned Mask")

        st.divider()

        # --- Step 6: Largest Connected Component ---
        st.markdown("### 6. Keep Largest Connected Component")

        final_mask_vis = np.zeros_like(mask_clean)
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(final_mask_vis, [cnt], -1, 255, -1)

        st.image(final_mask_vis, caption="Final Shape Mask")

        st.divider()

        # --- Step 7: Texture ---
        st.markdown("### 7. Texture Analysis (Sobel)")

        sobelx_64f = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely_64f = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx_64f ** 2 + sobely_64f ** 2)
        mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        st.image(mag_vis, caption="Edge Magnitude (Texture)")