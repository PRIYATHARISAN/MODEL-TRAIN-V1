import os
import shutil
from ultralytics import YOLO

# 1. Setup clean workspace folders
IMAGE_OUT = "mall_dataset/train/images"
LABEL_OUT = "mall_dataset/train/labels"

# Clear old runs to avoid duplicate mixing bugs
if os.path.exists("mall_dataset"):
    shutil.rmtree("mall_dataset")

os.makedirs(IMAGE_OUT, exist_ok=True)
os.makedirs(LABEL_OUT, exist_ok=True)

# 2. Load the Large high-capacity YOLOv11 model
print("Loading YOLOv11 Large model...")
model = YOLO("yolo11l.pt")

# 3. Read raw extracted image directory frames
input_folder = "extracted_images"
images = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])

print(f"Auto-labeling exactly the 'person' class across {len(images)} images...")

for img_name in images:
    img_path = os.path.join(input_folder, img_name)
    
    # CRITICAL CHANGE: classes=[0] targets ONLY people (ignores bags, carts, seats)
    results = model(img_path, classes=[0], verbose=False)
    result = results[0] # Safely target the primary frame object
    
    # Verify if a person was found in this specific picture frame
    if len(result.boxes) > 0:
        # Save structural image path targets
        dst_img_path = os.path.join(IMAGE_OUT, img_name)
        
        # Safely copy/move image to dataset tracking folder
        shutil.copy(img_path, dst_img_path)
        
        # Format the text coordinate label matching output names
        txt_name = img_name.replace(".jpg", ".txt")
        with open(os.path.join(LABEL_OUT, txt_name), "w") as f:
            for box in result.boxes:
                cls = int(box.cls[0]) # Extract class array integer index
                xywh = box.xywhn[0].tolist() # Parse list layout cleanly
                
                # Write YOLO standardized syntax format structure
                f.write(f"{cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")

print("Finished! All strict single-class person labels saved inside 'mall_dataset/train/'")
