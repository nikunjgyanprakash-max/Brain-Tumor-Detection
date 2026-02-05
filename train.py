import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_tumor_detector

import os



# 1. Setup Data Paths

# Ensure you have 'train' and 'val' folders inside your dataset folder

base_dir = 'dataset' 

train_dir = os.path.join(base_dir, 'train')

val_dir = os.path.join(base_dir, 'val')



# 2. The Data Pipeline (Automated Loading)

# rescale=1./255 does the normalization we discussed earlier

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=10,

    zoom_range=0.1,

    width_shift_range=0.05,

    height_shift_range=0.05

)



val_datagen = ImageDataGenerator(rescale=1./255)



# 3. Connect Pipeline to Folders

# This looks into your folders and finds the images automatically

print("--- Loading Data ---")

train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(224, 224), # Must match the Input Shape in model.py

    batch_size=32,

    class_mode='binary',    # Binary because we are detecting Tumor vs No Tumor

    color_mode='grayscale'  # Must match the '1' channel in model.py

)

print("Class mapping:", train_generator.class_indices)



validation_generator = val_datagen.flow_from_directory(

    val_dir,

    target_size=(224, 224),

    batch_size=32,

    class_mode='binary',

    color_mode='grayscale'

)



# 4. Initialize the Model

print("--- Building Model ---")

model = build_tumor_detector()



# 5. Compile the Model

# Optimizer: 'adam' is the standard "smart" learning algorithm

# Loss: 'binary_crossentropy' is the math formula for Yes/No errors

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



# 6. Start Training

print("--- Starting Training ---")

history = model.fit(

    train_generator,

    epochs=25,             # How many times to loop through the entire dataset

    validation_data=validation_generator

)



# 7. Save the "Brain"

# This creates a file you can use later without retraining

model.save('brain_tumor_model.h5')

print("✅ Model saved as brain_tumor_model.h5")
