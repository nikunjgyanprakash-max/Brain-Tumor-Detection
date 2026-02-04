import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import gdown
import os

# 1. Provide your Model ID
file_id = '1bw5iMUCnJe0pP0AOSeXvv4U-V4HxPbyH'

@st.cache_resource
def load_model_from_drive():
    filename = 'brain_tumor_model.h5'
    if not os.path.exists(filename):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filename, quiet=False)
    return load_model(filename)

# 2. Setup App
st.title("Brain Tumor Detector")
model = load_model_from_drive()

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)
    
    # 3. Process and Predict
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0  # Simple normalization
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    
    # 4. Show Result
    # (Assuming 0-0.5 is Tumor and 0.5-1.0 is Healthy)
    if prediction[0][0] < 0.5:
        st.error("Result: Tumor Detected")
    else:
        st.success("Result: Healthy")
