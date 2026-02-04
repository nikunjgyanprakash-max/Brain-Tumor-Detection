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
        gdown.download(url, filename, quiet=False)
    return load_model(filename)

st.title("🧠 NeuroScan: Final Logic Fix")

model = load_model_from_drive()
uploaded_file = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)
    
    if st.button("Deep Analysis"):
        # 1. Resize and Prepare
        img = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 2. Get the Prediction
        prediction = model.predict(img_array)
        score = float(prediction[0][0])
        
        # 3. Calculate "Tumor Confidence" (Flipping the 0/1 logic)
        tumor_confidence = 1.0 - score 
        
        st.write(f"**AI Suspicion Level:** {tumor_confidence:.2%}")

        # --- THE FIX: HIGH SENSITIVITY THRESHOLD ---
        # If the AI is even slightly suspicious (>15%), we flag it.
        if tumor_confidence > 0.15:
            st.error(f"🚨 TUMOR DETECTED ({tumor_confidence:.1%})")
            st.warning("Clinical Alert: AI has identified suspicious patterns.")
        else:
            st.success(f"✅ HEALTHY ({1.0 - tumor_confidence:.1%})")
