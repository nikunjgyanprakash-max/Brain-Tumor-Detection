import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# 1. Load the Model ONCE (Caching makes it fast)
@st.cache_resource
def load_brain_model():
    model = load_model('brain_tumor_model.h5')
    return model

# 2. Page Title
st.title("🧠 NeuroScan: AI Brain Tumor Detector")
st.write("Upload an MRI scan to detect potential tumors.")

# Load model with error handling
try:
    model = load_brain_model()
    st.success("System Ready: AI Model Loaded")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. The Upload Button
uploaded_file = st.file_uploader("Choose a Brain MRI...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI Scan', use_container_width=True)
    
    # 4. Preprocess (Resize & Color Match)
    if st.button("Analyze Scan"):
        # Resize to 224x224
        img_resized = img.resize((224, 224))
        
        # FIX: Convert to RGB (3 channels) to match the new model
        img_rgb = img_resized.convert('RGB')
        
        # Convert to array and normalize
        img_array = np.array(img_rgb)
        img_array = img_array / 255.0  
        
        # Reshape for the model: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0) 

        # 5. Predict
        prediction = model.predict(img_array)
        confidence = prediction[0][0]
        
        st.write("---")
        st.subheader("Analysis Results:")
        
        # Logic: > 0.5 is Tumor, <= 0.5 is Healthy
        if confidence > 0.5:
            st.error(f"🚨 TUMOR DETECTED (Confidence: {confidence:.2%})")
            st.warning("Recommendation: Immediate Clinical Referral.")
        else:
            st.success(f"✅ HEALTHY (Confidence: {(1-confidence):.2%})")
            st.info("Recommendation: Routine Check-up.")
