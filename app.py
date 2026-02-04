import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import gdown
import os

# --- PASTE YOUR DRIVE ID HERE ---
# Based on your previous message, this is your ID:
file_id = '1o91lMmfr_rRxlwT8_ygoU78KBs1Q50ax' 
# --------------------------------

@st.cache_resource
def load_model_from_drive():
    # 1. NEW FILENAME: '_v5' forces a fresh download
    filename = 'brain_tumor_model_v5.h5'
    url = f'https://drive.google.com/uc?id={file_id}'

    # 2. DOWNLOAD 
    if not os.path.exists(filename):
        with st.spinner("Downloading Huge 500MB AI Brain... (This takes 1-2 mins)"):
            gdown.download(url, filename, quiet=False)
    
    # 3. VERIFY SIZE
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        st.write(f"🔍 System Check: Downloaded File Size: **{size_mb:.2f} MB**")
        
        # If it's small (like 10MB), it's the wrong file!
        if size_mb < 200:
            st.error("🚨 Error: File is too small! Check your Drive Link.")
            st.stop()
    
    model = load_model(filename)
    return model

# --- APP INTERFACE ---
st.set_page_config(page_title="NeuroScan Pro", page_icon="🧠")
st.title("🧠 NeuroScan: 500MB Expert Edition")
st.write("System Status: Loading High-Precision Model...")

try:
    model = load_model_from_drive()
    st.success("✅ 500MB Model Loaded Successfully")
except Exception as e:
    st.error(f"Failed to load model: {e}")

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Scan', width=300)
    
    if st.button("Analyze Scan"):
        # Preprocessing (Standard 224x224 for VGG models)
        img = image.resize((224, 224))
        img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        prediction = model.predict(img_array)
        score = prediction[0][0]
        
        st.divider()
        st.write(f"**AI Confidence Score:** {score:.5f}")
        st.caption("0.00 = Healthy | 1.00 = Tumor")
        
        # 0.5 Threshold
        if score > 0.5:
            st.error(f"🚨 TUMOR DETECTED (Confidence: {score:.1%})")
        else:
            st.success(f"✅ HEALTHY (Confidence: {(1-score):.1%})")
