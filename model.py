import tensorflow as tf
from tensorflow.keras import layers, models

def build_tumor_detector():
    """
    Constructs the Neural Network architecture.
    """
    model = models.Sequential()

    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
    
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    
    model.add(layers.Dense(64, activation='relu'))
    
    model.add(layers.Dropout(0.5))

    # Output Layer: 1 single neuron. 
    # 0 = No Tumor, 1 = Tumor.
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

model = build_tumor_detector()

model.summary()
