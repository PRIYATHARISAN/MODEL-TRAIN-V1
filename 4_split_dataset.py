import os
import random
import shutil

# Paths
TRAIN_IMG = "mall_dataset/train/images"
TRAIN_LBL = "mall_dataset/train/labels"
VAL_IMG = "mall_dataset/val/images"
VAL_LBL = "mall_dataset/val/labels"

# Create validation directories
os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(VAL_LBL, exist_ok=True)

# Get all images currently in train
images = sorted([f for f in os.listdir(TRAIN_IMG) if f.endswith(".jpg")])

# Pick 20% randomly to move to validation
val_count = int(len(images) * 0.20)
val_images = random.sample(images, val_count)

print(f"Moving {val_count} images and labels to the Validation split...")

for img_name in val_images:
    lbl_name = img_name.replace(".jpg", ".txt")
    
    # Paths for images
    src_img = os.path.join(TRAIN_IMG, img_name)
    dst_img = os.path.join(VAL_IMG, img_name)
    
    # Paths for labels
    src_lbl = os.path.join(TRAIN_LBL, lbl_name)
    dst_lbl = os.path.join(VAL_LBL, lbl_name)
    
    # Move them safely
    if os.path.exists(src_img):
        shutil.move(src_img, dst_img)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

print(f"Split complete! Train: {len(os.listdir(TRAIN_IMG))}, Val: {len(os.listdir(VAL_IMG))}")
