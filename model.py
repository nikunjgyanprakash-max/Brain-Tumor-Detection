import tensorflow as tf
from tensorflow.keras import layers, models

def build_tumor_detector():
    """
    Constructs the Neural Network architecture.
    """
    model = models.Sequential()

    # --- Layer 1: Feature Extraction (The "Magnifying Glass") ---
    # Conv2D: Scans the image looking for edges and curves (basic shapes).
    # 32 filters, size 3x3.
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
    # MaxPooling: Shrinks the image to focus on the most important features.
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Layer 2: Deeper Analysis (Finding Texture) ---
    # We increase filters to 64 to find more complex patterns (like tumor textures).
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Layer 3: The Decision Maker (The "Doctor") ---
    # Flatten: Turns the 2D image data into a long 1D list of numbers.
    model.add(layers.Flatten())
    
    # Dense: The actual neurons that "think" about the data.
    model.add(layers.Dense(64, activation='relu'))
    
    # Dropout: Randomly turns off 50% of neurons during training.
    # CRITICAL FOR RESEARCH: This prevents the model from "memorizing" the training images.
    model.add(layers.Dropout(0.5))

    # Output Layer: 1 single neuron. 
    # 0 = No Tumor, 1 = Tumor.
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# --- Testing the Architecture ---
model = build_tumor_detector()
# This prints a summary table of your model's structure
model.summary()