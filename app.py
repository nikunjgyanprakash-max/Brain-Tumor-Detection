import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import gdown
import os

# --- DRIVE ID ---
file_id = '1o91lMmfr_rRxlwT8_ygoU78KBs1Q50ax' 

@st.cache_resource
def load_model_from_drive():
    filename = 'brain_tumor_model_v5.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(filename):
        with st.spinner("Initializing AI Neural Network..."):
            gdown.download(url, filename, quiet=False)
    return load_model(filename)

st.set_page_config(page_title="NeuroScan Clinical", page_icon="🧠")
st.title("🧠 NeuroScan: Clinical Analysis")

try:
    model = load_model_from_drive()
    st.sidebar.success("✅ AI Engine Online")
except Exception as e:
    st.sidebar.error("❌ Engine Offline")
    st.stop()

uploaded_file = st.file_uploader("Upload MRI Scan for Analysis", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Source MRI Scan', width=300)
    
    if st.button("Generate Diagnostic Report"):
        # Preprocessing
        img = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        raw_score = float(prediction[0][0])
        
        # Logic: 1 - raw_score because of our "Alphabetical Trap" fix
        tumor_prob = 1.0 - raw_score
        
        # Clamp value between 0 and 1 to prevent the Progress Bar Error
        safe_prob = max(0.0, min(1.0, tumor_prob))
        
        st.subheader("Diagnostic Results")
        
        # Display Gauge
        if safe_prob > 0.5:
            st.error(f"CONSULTATION ADVISED: Tumor Detected Patterns")
            st.progress(safe_prob) 
        else:
            st.success(f"ANALYSIS COMPLETE: Healthy Patterns Detected")
            st.progress(safe_prob)

        st.write(f"**Confidence Level:** {safe_prob:.2%}")
        
        # Clinical Notes
        st.divider()
        st.info("**AI Clinical Note:** High confidence in detecting dense white masses. Note that T1-weighted necrotic centers may result in lower probability scores.")
