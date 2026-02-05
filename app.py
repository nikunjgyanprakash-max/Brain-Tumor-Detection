import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

# --- DRIVE ID ---
# Paste the ID of your NEW 85% model here
file_id = '11WbPy5hYZQo-rPNXjmf2ewcWI8368M61' 

@st.cache_resource
def load_model_from_drive():
    filename = 'brain_tumor_model.h5'
    if not os.path.exists(filename):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filename, quiet=False)
    return load_model(filename)

st.title("🧠 NeuroScan: Detection System")
st.write("MIRM Research Internship Project")

try:
    model = load_model_from_drive()
    st.success("System Ready: Model Loaded")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='MRI Scan', width=300)
    
    if st.button("Run Diagnostic"):
        img = image.resize((224, 224))
        img_array = np.asarray(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        score = float(prediction[0][0])
        
        st.divider()
        
        # --- THE CORRECTED LOGIC ---
        # 0 = Healthy (H comes before T)
        # 1 = Tumor
        
        if score > 0.5: 
            # Score is closer to 1 (Tumor)
            st.error(f"🚨 TUMOR DETECTED ({score:.2%})")
        else:
            # Score is closer to 0 (Healthy)
            confidence = 1.0 - score
            st.success(f"✅ HEALTHY BRAIN ({confidence:.2%})")























