import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# --- ADDED TO TOP ---
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from model import build_tumor_detector
import os

# 1. Setup Data Paths
base_dir = 'dataset' 
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# 2. The Data Pipeline
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 3. Connect Pipeline to Folders
print("--- Loading Data ---")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale'
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
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- ADDED JUST ABOVE STEP 6 ---
# These are your safety nets for the 35-epoch mission
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# 6. Start Training
print("--- Starting Mission Training (35 Epochs) ---")
history = model.fit(
    train_generator,
    epochs=35, # Updated to 35
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr] # Connecting the safety nets
)

# 7. Save the "Brain"
model.save('brain_tumor_model.h5')
print("✅ Model saved as brain_tumor_model.h5")
