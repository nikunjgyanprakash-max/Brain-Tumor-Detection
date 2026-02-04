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
    filename = 'brain_tumor_model_v5.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(filename):
        with st.spinner("Loading Expert Brain..."):
            gdown.download(url, filename, quiet=False)
    return load_model(filename)

st.set_page_config(page_title="NeuroScan Pro", page_icon="🧠")
st.title("🧠 NeuroScan: Clinical Dashboard")

model = load_model_from_drive()

uploaded_file = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Target Scan', width=300)
    
    if st.button("Run Diagnostic"):
        # Preprocessing
        img = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        score = prediction[0][0]
        
        # Our Logic Fix: 1 - score = Tumor probability
        prob = (1 - score) * 100
        
        st.subheader("Analysis Result")
        
        # --- THE PROBABILITY GAUGE ---
        if prob > 50:
            st.error(f"Prediction: TUMOR DETECTED")
            st.progress(prob / 100) # Red-ish bar
        else:
            st.success(f"Prediction: HEALTHY")
            st.progress(prob / 100) # Green-ish bar
            
        st.write(f"**Tumor Probability Index:** {prob:.2f}%")
        
        # --- SMART ADVICE ---
        if 30 < prob < 70:
            st.warning("⚠️ **Low Confidence Alert:** The AI is seeing conflicting patterns. This often happens with necrotic (dark-centered) tumors or T1-weighted scans.")
