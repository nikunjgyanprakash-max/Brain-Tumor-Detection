import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_process_mri(image_path):
    """
    Loads an MRI scan and prepares it for analysis.
    """
    # 1. Check if file exists (Crucial for research automation)
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found at {image_path}")
        return None

    # 2. Read the image as Grayscale
    # (Color doesn't matter for MRI intensity, and it saves processing power)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 3. Resize to a standard size (e.g., 224x224)
    # AI models require all inputs to be the exact same size.
    img_resized = cv2.resize(img, (224, 224))
    
    # 4. Normalize pixel values (0 to 1 instead of 0 to 255)
    # This helps the math in the neural network work faster.
    img_normalized = img_resized / 255.0
    
    print("✅ MRI Loaded Successfully")
    print(f"Original Shape: {img.shape}")
    print(f"New Shape: {img_normalized.shape}")
    
    return img_normalized

# --- Testing Section ---
# You would replace 'test_scan.jpg' with a real file name
# processed_mri = load_and_process_mri('dataset/test_scan.jpg')

# To see what the computer sees:
# if processed_mri is not None:
#     plt.imshow(processed_mri, cmap='gray')
#     plt.title("Processed MRI Input")
#     plt.show()