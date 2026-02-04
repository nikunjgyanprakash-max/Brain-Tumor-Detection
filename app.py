import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np
import gdown
import os

# --- DRIVE ID ---
file_id = '1o91lMmfr_rRxlwT8_ygoU78KBs1Q50ax' 
# ----------------

@st.cache_resource
def load_model_from_drive():
    filename = 'brain_tumor_model_v7.h5' # v7: The Logic Fix
    url = f'https://drive.google.com/uc?id={file_id}'

    if not os.path.exists(filename):
        with st.spinner("Downloading AI Brain (500 MB)..."):
            gdown.download(url, filename, quiet=False)
            
    model = load_model(filename)
    return model

st.set_page_config(page_title="NeuroScan Pro", page_icon="🧠")
st.title("🧠 NeuroScan: Professional Edition")

try:
    model = load_model_from_drive()
    st.success("✅ Model System Active")
except Exception as e:
    st.error(f"System Error: {e}")

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Scan', width=300)
    
    if st.button("Analyze Scan"):
        # 1. Standard Preprocessing (No Sliders, Just Math)
        img = image.resize((224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 2. Prediction
        prediction = model.predict(img_array)
        score = prediction[0][0]
        
        # --- THE LOGIC FIX ---
        # If Class 0 is Tumor, then a LOW score means TUMOR.
        # We calculate "Tumor Probability" as (1 - score)
        
        tumor_probability = 1 - score
        
        st.divider()
        st.write(f"**Diagnostic Analysis:**")
        
        if tumor_probability > 0.50:
            st.error(f"🚨 **TUMOR DETECTED**")
            st.write(f"Confidence: **{tumor_probability:.2%}**")
            st.caption("The AI has identified patterns consistent with a brain tumor.")
        else:
            st.success(f"✅ **HEALTHY BRAIN**")
            st.write(f"Confidence: **{(1-tumor_probability):.2%}**")
            st.caption("No tumor patterns detected.")
