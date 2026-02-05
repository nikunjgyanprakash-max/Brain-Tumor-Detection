import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_tumor_detector
import os
import matplotlib.pyplot as plt # --- CODE 3 (Part A): Graphing Tool ---
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # --- CODE 1 (Part A): Safety Nets ---

# 1. Setup Data Paths
base_dir = 'dataset' 
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# --- CODE 2: The Power Multiplier (Data Augmentation) ---
# Instead of just "rescale", we add rotation, zoom, and flips.
# This makes your 3,000 images feel like 30,000 to the AI.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,      # Tilt the head
    width_shift_range=0.1,  # Shift left/right
    height_shift_range=0.1, # Shift up/down
    zoom_range=0.1,         # Zoom in on tumor
    horizontal_flip=True,   # Mirror image
    fill_mode='nearest'
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

# --- CODE 1 (Part B): Defining the Safety Nets ---
# EarlyStopping: Stops if it stops learning (saves time).
# ReduceLROnPlateau: Slows down to study hard details.
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# 6. Start Training
print("--- Starting Mission Training (Up to 25 Epochs) ---")
history = model.fit(
    train_generator,
    epochs=25,  # Increased to 25 because Safety Nets protect us!
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr] # Activate Code 1
)

# 7. Save the "Brain"
model.save('brain_tumor_model.h5')
print("✅ Model saved as brain_tumor_model.h5")

# --- CODE 3 (Part B): The Result Map (Accuracy Graph) ---
# This generates a professional chart of your mission's success.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('NeuroScan Mission: Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('NeuroScan Mission: Loss')
plt.savefig('training_report.png') # Saves the graph as an image!
print("📊 Mission Report saved as 'training_report.png'")
