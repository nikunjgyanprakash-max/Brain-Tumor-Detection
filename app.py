import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

# --- DRIVE ID ---
file_id = '1bw5iMUCnJe0pP0AOSeXvv4U-V4HxPbyH'

@st.cache_resource
def load_model_from_drive():
    filename = 'brain_tumor_model.h5'
    if not os.path.exists(filename):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filename, quiet=False)
    return load_model(filename)

st.title("Brain Tumor Detection")
model = load_model_from_drive()

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")

    img = image.resize((224, 224))

    img_array = np.asarray(img).astype("float32") / 255.0

    img_array = np.stack([img_array]*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    st.write("Input shape:", img_array.shape)

    prediction = model.predict(img_array, verbose=0)

    score = float(prediction[0][0])

    if score < 0.5:
        st.error(f"Tumor Detected (Confidence: {score:.2%})")
    else:
        st.success(f"Healthy (Confidence: {(1-score):.2%})")
