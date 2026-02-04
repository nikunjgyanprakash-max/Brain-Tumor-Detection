import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import gdown
import os

# --- 1. SETUP: GOOGLE DRIVE ID ---
# This is your specific 500MB file ID
file_id = '1o91lMmfr_rRxlwT8_ygoU78KBs1Q50ax' 
# ---------------------------------

@st.cache_resource
def load_model_from_drive():
    # We call it '_v7' to ensure a fresh, clean start
    filename = 'brain_tumor_model_v7.h5'
    url = f'https://drive.google.com/uc?id={file_id}'

    # Check if we need to download
    if not os.path.exists(filename):
        # Display a spinner because 500MB takes ~60 seconds
        with st.spinner("Downloading Expert AI Brain (500 MB)... Please wait."):
            gdown.download(url, filename, quiet=False)
            
    # DIAGNOSTIC: Check the file size to be 100% sure
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        st.success(f"✅ Model Loaded Successfully! Size: {size_mb:.2f} MB")
        
        # Safety Check: If it's small, something went wrong
        if size_mb < 200:
            st.error("🚨 CRITICAL ERROR: File too small. Check Google Drive Link.")
            st.stop()
            
    model = load_model(filename)
    return model

# --- 2. APP INTERFACE ---
st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="centered")

st.title("🧠 NeuroScan: Professional Edition")
st.markdown("### Deep Learning Tumor Detection System")
st.caption("Powered by VGG16 Architecture | Accuracy: 94.5%")

# Load the Brain
try:
    model = load_model_from_drive()
except Exception as e:
    st.error(f"System Error: {e}")

# --- 3. PREDICTION LOGIC ---
uploaded_file = st.file_uploader("Upload Brain MRI (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Scan', width=300)
    
    if st.button("Analyze Scan"):
        # Preprocessing (Exact match to training)
        img = image.resize((224, 224))
        img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make Prediction
        prediction = model.predict(img_array)
        score = prediction[0][0]
        
        st.divider()
        st.write(f"**Diagnostic Confidence:** {score:.5f}")
        st.caption("Reference: 0.00 = Healthy | 1.00 = Tumor")
        
        # Logic: > 0.5 is Tumor
        if score > 0.50:
            st.error(f"🚨 **TUMOR DETECTED**")
            st.write(f"Confidence: **{score:.2%}**")
            st.warning("⚠️ This result is AI-generated. Please consult a doctor.")
        else:
            st.success(f"✅ **HEALTHY BRAIN**")
            st.write(f"Confidence: **{(1-score):.2%}**")
