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
    filename = 'brain_tumor_model_v7.h5' # v7 to force new file check
    url = f'https://drive.google.com/uc?id={file_id}'

    if not os.path.exists(filename):
        with st.spinner("Downloading 500MB Model..."):
            gdown.download(url, filename, quiet=False)
            
    model = load_model(filename)
    return model

st.title("🧠 NeuroScan: Final Debug Mode")

try:
    model = load_model_from_drive()
    st.success("✅ Model Loaded")
except Exception as e:
    st.error(f"Error: {e}")

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Open Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Scan', width=300)
    
    if st.button("Analyze Scan"):
        # --- NEW PREPROCESSING (STRICT) ---
        # 1. Ensure RGB (removes Alpha channel if PNG)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # 2. Resize to 224x224 (Standard for VGG)
        img = image.resize((224, 224))
        
        # 3. Convert to Array
        img_array = img_to_array(img)
        
        # 4. Normalize (Divide by 255.0) - CRITICAL STEP
        img_array = img_array / 255.0
        
        # 5. Expand Dimensions (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        # ----------------------------------
        
        # Prediction
        prediction = model.predict(img_array)
        score = prediction[0][0]
        
        st.write(f"**Raw Score:** {score:.5f}")
        
        if score > 0.5:
            st.error(f"🚨 TUMOR DETECTED ({score:.1%})")
        else:
            st.success(f"✅ HEALTHY ({(1-score):.1%})")
