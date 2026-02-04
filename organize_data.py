import os
import shutil
import random

# 1. Define Paths
# Where the data is NOW
source_dir = 'raw_data' 
# Where we want the data to GO
base_dir = 'dataset'    

# 2. Define Split Ratio (80% Train, 20% Validation)
split_size = 0.8

def organize_data():
    # Create the folder structure if it doesn't exist
    for root in ['train', 'val']:
        for label in ['tumor', 'healthy']:
            os.makedirs(os.path.join(base_dir, root, label), exist_ok=True)
            
    # The categories in the raw folder (usually 'yes' and 'no')
    categories = {'yes': 'tumor', 'no': 'healthy'}

    print("--- Starting Data Organization ---")
    
    for raw_label, new_label in categories.items():
        source_path = os.path.join(source_dir, raw_label)
        
        # Get list of all images
        images = os.listdir(source_path)
        # Shuffle them randomly to ensure a fair test
        random.shuffle(images)
        
        # Calculate the split point
        split_point = int(len(images) * split_size)
        train_images = images[:split_point]
        val_images = images[split_point:]
        
        print(f"Processing '{raw_label}': {len(train_images)} for Training, {len(val_images)} for Validation.")
        
        # Copy files to Training folder
        for img in train_images:
            src = os.path.join(source_path, img)
            dst = os.path.join(base_dir, 'train', new_label, img)
            shutil.copyfile(src, dst)
            
        # Copy files to Validation folder
        for img in val_images:
            src = os.path.join(source_path, img)
            dst = os.path.join(base_dir, 'val', new_label, img)
            shutil.copyfile(src, dst)

    print("✅ Data Organization Complete. Ready for training!")

# Run the function
if __name__ == "__main__":
    organize_data()