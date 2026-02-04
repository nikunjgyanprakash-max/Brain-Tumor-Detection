import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import gdown
import os

# --- PASTE YOUR DRIVE ID HERE ---
file_id = '1o91lMmfr_rRxlwT8_ygoU78KBs1Q50ax' 
# --------------------------------

@st.cache_resource
def load_model_from_drive():
    # TRICK: We changed the name to '_v2' to FORCE a new download
    output = 'brain_tumor_model_v2.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(output):
        with st.spinner("Downloading 94% AI Brain (Fresh Copy)..."):
            gdown.download(url, output, quiet=False)
            st.success("Download Complete!")
    
    model = load_model(output)
    return model

st.title("🧠 NeuroScan: Pro (94% Accuracy)")
st.write("Using Model Version: v2.0 (Expert)")

try:
    model = load_model_from_drive()
    st.success("✅ Expert AI Loaded")
except Exception as e:
    st.error("Error loading model. Check your Drive ID.")

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Scan', width=300)
    
    if st.button("Analyze"):
        # Image Processing
        img = image.resize((224, 224))
        img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        prediction = model.predict(img_array)
        score = prediction[0][0]
        
        # --- DEBUG INFO (This helps us verify) ---
        st.write("---")
        st.write(f"🔍 **AI Confidence Score:** {score:.5f}")
        st.caption("0.0 = 100% Healthy | 1.0 = 100% Tumor")
        
        # Logic: High score means Tumor
        if score > 0.5:
            st.error(f"🚨 TUMOR DETECTED")
            st.write(f"Confidence: {score:.1%}")
        else:
            st.success(f"✅ HEALTHY")
            st.write(f"Confidence: {(1-score):.1%}")
