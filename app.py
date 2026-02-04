import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import gdown
import os

file_id = '1bw5iMUCnJe0pP0AOSeXvv4U-V4HxPbyH' 

@st.cache_resource
def load_model_from_drive():
    filename = 'brain_tumor_model.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)
    return load_model(filename)

st.title("🧠 NeuroScan: Final Resolution")

try:
    model = load_model_from_drive()
    st.success("✅ Engine Ready")
except Exception as e:
    st.error(f"Engine Error: {e}")

uploaded_file = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)
    
    if st.button("Final Analysis"):
        # 1. Prepare Image
        img = image.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 2. ResNet Math (Crucial)
        img_ready = preprocess_input(img_array)
        
        # 3. Get Raw Answer
        prediction = model.predict(img_ready)
        raw_score = float(prediction[0][0])
        
        # 4. SHOW THE RAW DATA
        st.write(f"**Raw Model Output:** {raw_score:.4f}")
        
        # TRY FLIPPING THIS IF RESULTS ARE WRONG:
        # Option A: tumor_prob = 1.0 - raw_score
        # Option B: tumor_prob = raw_score
        tumor_prob = 1.0 - raw_score 

        if tumor_prob > 0.5:
            st.error(f"🚨 TUMOR DETECTED ({tumor_prob:.2%})")
        else:
            st.success(f"✅ HEALTHY BRAIN ({(1-tumor_prob):.2%})")
