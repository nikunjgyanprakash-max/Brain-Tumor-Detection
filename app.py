import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input # UPDATED TO RESNET
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import gdown
import os

# --- DRIVE ID (Update this after you upload your new .h5 file) ---
file_id = '1bw5iMUCnJe0pP0AOSeXvv4U-V4HxPbyH' 
# ----------------------------------------------------------------

@st.cache_resource
def load_model_from_drive():
    filename = 'brain_tumor_model.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(filename):
        with st.spinner("Downloading ResNet50 Expert Model..."):
            gdown.download(url, filename, quiet=False)
    return load_model(filename)

st.set_page_config(page_title="NeuroScan ResNet", page_icon="🧠")
st.title("🧠 NeuroScan: ResNet50 Expert Edition")

try:
    model = load_model_from_drive()
    st.sidebar.success("✅ ResNet50 Engine Online")
except Exception as e:
    st.sidebar.error("❌ Engine Offline")
    st.stop()

uploaded_file = st.file_uploader("Upload MRI for Expert Analysis", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='MRI Scan', width=300)
    
    if st.button("Run ResNet Analysis"):
        # 1. Resize to 224x224
        img = image.resize((224, 224))
        
        # 2. Convert to Array
        img_array = img_to_array(img)
        
        # 3. Add Batch Dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # 4. OFFICIAL RESNET PREPROCESSING (Essential!)
        # Do NOT divide by 255 manually; this function handles it.
        img_preprocessed = preprocess_input(img_array)
        
        # 5. # Use this block to see exactly what the model "thinks"
prediction = model.predict(img_preprocessed)
raw_score = float(prediction[0][0])

st.write(f"**Raw AI Output Score:** {raw_score:.4f}")

# Map the scores to classes based on alphabetical order
# Score < 0.5 usually means Class 0 (the first folder alphabetically)
if raw_score < 0.5:
    st.error("Model Result: Class 0 (Check if this is Tumor in your folders)")
else:
    st.success("Model Result: Class 1 (Check if this is Healthy in your folders)")
        
        # Logic Check: Did your training use 'Brain_Tumor' as the first folder?
        # If yes, 0 is Tumor. We calculate Tumor Probability:
        tumor_prob = 1.0 - score 
        
        st.divider()
        st.subheader(f"AI Confidence: {tumor_prob:.2%}")
        
        # We use a 0.5 threshold for the expert model
        if tumor_prob > 0.5:
            st.error("🚨 TUMOR DETECTED")
            st.info("The ResNet50 model identifies patterns consistent with a tumor.")
        else:
            st.success("✅ HEALTHY BRAIN")
            st.info("No tumorous patterns identified by the deep neural network.")
