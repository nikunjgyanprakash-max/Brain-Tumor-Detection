import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import gdown
import os

# --- DRIVE ID ---
file_id = '1bw5iMUCnJe0pP0AOSeXvv4U-V4HxPbyH' 

@st.cache_resource
def load_model_from_drive():
    filename = 'brain_tumor_model.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(filename):
        with st.spinner("Downloading ResNet50 Expert Brain..."):
            gdown.download(url, filename, quiet=False)
    return load_model(filename)

st.set_page_config(page_title="NeuroScan Expert", page_icon="🧠")
st.title("🧠 NeuroScan: Final Stable Edition")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("System Settings")
flip_logic = st.sidebar.checkbox("Flip Tumor/Healthy Logic", value=False)
st.sidebar.info("If the results are exactly backward, check the box above.")

try:
    model = load_model_from_drive()
    st.sidebar.success("✅ Engine Online")
except Exception as e:
    st.sidebar.error(f"❌ Engine Error: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Scan', width=300)
    
    if st.button("Run Final Diagnostic"):
        # 1. Image Prep
        img = image.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 2. ResNet Preprocessing
        img_ready = preprocess_input(img_array)
        
        # 3. Predict
        prediction = model.predict(img_ready)
        raw_score = float(prediction[0][0])
        
        # 4. Display Raw Score (This helps us debug!)
        st.write(f"**Raw Model Score:** {raw_score:.4f}")
        st.caption("Score near 0.0 usually means Class 0. Score near 1.0 usually means Class 1.")

        # 5. Logical Decision
        # If flip_logic is OFF: 0 is Tumor, 1 is Healthy
        # If flip_logic is ON: 1 is Tumor, 0 is Healthy
        if not flip_logic:
            tumor_prob = 1.0 - raw_score
        else:
            tumor_prob = raw_score

        st.divider()
        st.subheader(f"Detection Confidence: {tumor_prob:.2%}")

        if tumor_prob > 0.50:
            st.error("🚨 TUMOR DETECTED")
        else:
            st.success("✅ HEALTHY BRAIN")

        # Clinical Warning
        st.warning("⚠️ **Note:** AI results should only be used as a second opinion. Always consult a radiologist.")
