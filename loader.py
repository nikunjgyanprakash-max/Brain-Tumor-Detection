import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_process_mri(image_path):
    """
    Loads an MRI scan and prepares it for analysis.
    """
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found at {image_path}")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    img_resized = cv2.resize(img, (224, 224))
    
    img_normalized = img_resized / 255.0
    
    print("✅ MRI Loaded Successfully")
    print(f"Original Shape: {img.shape}")
    print(f"New Shape: {img_normalized.shape}")
    
    return img_normalized
