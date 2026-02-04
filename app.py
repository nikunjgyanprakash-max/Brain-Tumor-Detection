import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import gdown
import os

# --- PASTE YOUR DRIVE FILE ID HERE ---
# Example: file_id = '12345ABCDE...'
file_id = 'YOUR_ACTUAL_ID_GOES_HERE'
# -------------------------------------

@st.cache_resource
def load_model_from_drive():
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'brain_tumor_model.h5'
    
    if not os.path.exists(output):
        st.info("Downloading 143MB AI Model... Please wait.")
        gdown.download(url, output, quiet=False)
        st.success("Download Complete!")
    
    model = load_model(output)
    return model

st.title("🧠 NeuroScan: Pro Edition")
st.write("System Status: Initializing Heavy Model...")

try:
    model = load_model_from_drive()
    st.success("✅ System Ready: 143MB Model Loaded")
except Exception as e:
    st.error("Error loading model. Check your Google Drive Link ID.")
    st.stop()

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Scan', use_container_width=True)
    
    if st.button("Analyze"):
        # Resize and Convert to RGB (The Fat Model needs RGB)
        img = image.resize((224, 224))
        img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        score = prediction[0][0]
        
        st.write(f"Confidence Score: {score:.4f}")
        
        if score > 0.5:
            st.error(f"🚨 TUMOR DETECTED ({score:.1%})")
        else:
            st.success(f"✅ HEALTHY ({(1-score):.1%})")
