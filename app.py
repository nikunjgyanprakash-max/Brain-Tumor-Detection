import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import gdown
import os

# --- DRIVE ID ---
file_id = '1bw5iMUCnJe0pP0AOSeXvv4U-V4HxPbyH'

@st.cache_resource
def load_model_from_drive():
    filename = 'model.h5'
    if not os.path.exists(filename):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filename, quiet=False)
    return load_model(filename)

st.title("Brain Tumor Detection")
model = load_model_from_drive()

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)
    
    # 1. Prepare image
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0  # Basic normalization
    img_array = np.expand_dims(img_array, axis=0)
    
    # 2. Get prediction
    prediction = model.predict(img_array)
    score = prediction[0][0]
    
    # 3. Output result
    if score < 0.5:
        st.error(f"Tumor Detected (Score: {score:.4f})")
    else:
        st.success(f"Healthy (Score: {score:.4f})")
