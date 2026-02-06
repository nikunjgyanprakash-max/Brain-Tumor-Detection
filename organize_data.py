import os
import shutil
import random

source_dir = 'raw_data' 

base_dir = 'dataset'    

split_size = 0.8

def organize_data():
    
    for root in ['train', 'val']:
        for label in ['tumor', 'healthy']:
            os.makedirs(os.path.join(base_dir, root, label), exist_ok=True)
            
    categories = {'yes': 'tumor', 'no': 'healthy'}

    print("--- Starting Data Organization ---")
    
    for raw_label, new_label in categories.items():
        source_path = os.path.join(source_dir, raw_label)
        
        
        images = os.listdir(source_path)
        
        random.shuffle(images)
        
        
        split_point = int(len(images) * split_size)
        train_images = images[:split_point]
        val_images = images[split_point:]
        
        print(f"Processing '{raw_label}': {len(train_images)} for Training, {len(val_images)} for Validation.")
        
        
        for img in train_images:
            src = os.path.join(source_path, img)
            dst = os.path.join(base_dir, 'train', new_label, img)
            shutil.copyfile(src, dst)
            
        
        for img in val_images:
            src = os.path.join(source_path, img)
            dst = os.path.join(base_dir, 'val', new_label, img)
            shutil.copyfile(src, dst)

    print("✅ Data Organization Complete. Ready for training!")


if __name__ == "__main__":
    organize_data()
