import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_tumor_detector
import os
import matplotlib.pyplot as plt # --- CODE 3 (Part A): Graphing Tool ---
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # --- CODE 1 (Part A): Safety Nets ---

base_dir = 'dataset' 
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

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

print("--- Building Model ---")
model = build_tumor_detector()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

print("--- Starting Mission Training (Up to 25 Epochs) ---")
history = model.fit(
    train_generator,
    epochs=25,  # Increased to 25 because Safety Nets protect us!
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr] # Activate Code 1
)

model.save('brain_tumor_model.h5')
print("Model saved as brain_tumor_model.h5")

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
print("Mission Report saved as 'training_report.png'")
