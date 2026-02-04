import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import gdown
import os

# --- 1. SETUP: PASTE YOUR DRIVE ID HERE ---
# -----------------------------------------------------
file_id = 'PASTE_YOUR_GOOGLE_DRIVE_ID_HERE' 
# -----------------------------------------------------

@st.cache_resource
def load_model_from_drive():
    url = f'https://drive.google.com/uc?id={1o91lMmfr_rRxlwT8_ygoU78KBs1Q50ax}'
    output = 'brain_tumor_model.h5'
    
    # Download only if we don't have it yet
    if not os.path.exists(output):
        with st.spinner("Downloading 143MB AI Brain... (This happens once)"):
            gdown.download(url, output, quiet=False)
            st.success("Download Complete!")
    
    # Load the model
    model = load_model(output)
    return model

# --- 2. THE APP INTERFACE ---
st.set_page_config(page_title="NeuroScan AI", page_icon="🧠")

st.title("🧠 NeuroScan: Professional Edition")
st.markdown("### Deep Learning Tumor Detection System")
st.write("Using the High-Performance VGG-Style Model (94.5% Accuracy)")

# Load Model
try:
    model = load_model_from_drive()
    st.success("✅ System Ready: AI Brain Loaded")
except Exception as e:
    st.error("Error loading model. Please check your Google Drive ID.")
    st.stop()

# --- 3. PREDICTION LOGIC ---
uploaded_file = st.file_uploader("Upload Brain MRI (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Scan', width=300)
    
    if st.button("Analyze Scan"):
        # Preprocessing (Must match training exactly)
        img = image.resize((224, 224))
        img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        score = prediction[0][0]
        
        # Display Result
        st.write("---")
        st.write(f"**Raw Confidence Score:** {score:.4f}")
        
        # High Confidence Threshold (because model is 94% accurate)
        if score > 0.50:
            st.error(f"🚨 **TUMOR DETECTED**")
            st.write(f"Confidence: **{score:.1%}**")
            st.warning("Please consult a medical professional immediately.")
        else:
            st.success(f"✅ **HEALTHY BRAIN**")
            st.write(f"Confidence: **{(1-score):.1%}**")
