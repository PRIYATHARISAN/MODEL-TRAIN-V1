import os
from ultralytics import YOLO

# 1. Setup folders for the training data
IMAGE_OUT = "mall_dataset/train/images"
LABEL_OUT = "mall_dataset/train/labels"

os.makedirs(IMAGE_OUT, exist_ok=True)
os.makedirs(LABEL_OUT, exist_ok=True)

# 2. Load the Large YOLOv11 model
print("Loading YOLOv11 Large model...")
model = YOLO("yolo11l.pt")

# 3. Read extracted images
input_folder = "extracted_images"
images = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])

print(f"Auto-labeling {len(images)} images. Please wait...")

for img_name in images:
    img_path = os.path.join(input_folder, img_name)
    
    # Run model prediction
    results = model(img_path, verbose=False)[0]
    
    # Only keep the image if YOLO actually finds objects inside it
    if len(results.boxes) > 0:
        # Move image to final dataset folder
        os.rename(img_path, os.path.join(IMAGE_OUT, img_name))
        
        # Save labels to matching .txt file
        txt_name = img_name.replace(".jpg", ".txt")
        with open(os.path.join(LABEL_OUT, txt_name), "w") as f:
            for box in results.boxes:
                cls = int(box.cls)
                xywh = box.xywhn.tolist()[0] # Get normalized coordinates
                f.write(f"{cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")

print("Finished! All labels saved cleanly inside 'mall_dataset/train/'")
