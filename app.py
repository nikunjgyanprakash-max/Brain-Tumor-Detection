import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

# --- DRIVE ID (CRITICAL: You must update this!) ---
# 1. Upload your NEW 'brain_tumor_model.h5' (the 3000-image one) to Google Drive.
# 2. Right-click > Share > Anyone with link > Copy Link.
# 3. Paste the NEW ID here.
file_id = '11WbPy5hYZQo-rPNXjmf2ewcWI8368M61' # <--- UPDATE THIS ID

@st.cache_resource
def load_model_from_drive():
    filename = 'brain_tumor_model.h5'
    if not os.path.exists(filename):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filename, quiet=False)
    return load_model(filename)

st.title("Brain Tumor Detection")
model = load_model_from_drive()

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Convert to Grayscale (L)
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='MRI Scan', width=300)

    if st.button("Run Diagnostic"):
        # 2. Resize to 224x224
        img = image.resize((224, 224))

        # 3. Normalize (0-1 range)
        img_array = np.asarray(img).astype("float32") / 255.0

        img_array = np.expand_dims(img_array, axis=-1) # Shape: (224, 224, 1)
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 1)

        st.write("Input shape:", img_array.shape) # Should say (1, 224, 224, 1)

        # 4. Predict
        prediction = model.predict(img_array, verbose=0)
        score = float(prediction[0][0])

        st.divider()
        # Logic: 0 = Tumor, 1 = Healthy (Based on your folder structure)
        if score < 0.5:
            confidence = (1.0 - score)
            st.error(f"🚨 Tumor Detected (Confidence: {confidence:.2%})")
        else:
            st.success(f"✅ Healthy (Confidence: {score:.2%})")
