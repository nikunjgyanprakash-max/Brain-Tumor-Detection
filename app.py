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

st.title("🧠 NeuroScan: Detection Mission")
model = load_model_from_drive()

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L") # 1. Convert to Grayscale
    st.image(image, caption='MRI Scan', width=300)
    
    if st.button("Run Diagnostic"):
        # 2. Resize to 224x224
        img = image.resize((224, 224))
        
        # 3. Normalize (Must match your 1/255 training)
        img_array = np.asarray(img).astype("float32") / 255.0
        
        # 4. Fix the Shape (Must be 1 channel to match model.py)
        img_array = np.expand_dims(img_array, axis=-1) # Becomes (224, 224, 1)
        img_array = np.expand_dims(img_array, axis=0)  # Becomes (1, 224, 224, 1)
        
        # 5. Predict
        prediction = model.predict(img_array, verbose=0)
        score = float(prediction[0][0])
        
        st.divider()
        
        # Logic: Alphabetical order usually makes 0=Tumor, 1=Healthy
        # Adjust this if your results are backward!
        if score < 0.5:
            confidence = (1.0 - score)
            st.error(f"🚨 TUMOR DETECTED ({confidence:.2%})")
        else:
            st.success(f"✅ HEALTHY BRAIN ({score:.2%})")
