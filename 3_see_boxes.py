import os
import shutil
import cv2

# Paths to your dataset files
IMAGE_DIR = "mall_dataset/train/images"
LABEL_DIR = "mall_dataset/train/labels"
OUTPUT_DIR = "annotated_visuals"

# CRITICAL FIX: Delete the old folder completely so old images disappear!
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get ALL the images in the directory
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])

print(f"Generating clean, STRICT person-only boxes for {len(image_files)} images...")

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    txt_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt"))
    
    if not os.path.exists(txt_path):
        continue
        
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if not parts:
                continue
                
            cls_id = int(parts[0])
            
            # CRITICAL FILTER: Skip drawing if it is NOT a person (Class 0)
            if cls_id != 0:
                continue
                
            x_center, y_center, box_w, box_h = map(float, parts[1:])
            
            # Convert math coordinates to image pixel locations
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)
            
            # Draw green box and write "Person"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "Person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    # Save into your visual folder
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"visual_{img_name}"), img)

print(f"Done! Check the '{OUTPUT_DIR}' folder. Handbags and backpacks are now completely hidden!")
