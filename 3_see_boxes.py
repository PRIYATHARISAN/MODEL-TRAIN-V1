import os
import cv2

# Paths to your dataset files
IMAGE_DIR = "mall_dataset/train/images"
LABEL_DIR = "mall_dataset/train/labels"
OUTPUT_DIR = "annotated_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map class numbers to names
CLASS_MAP = {0: "Person", 24: "Backpack", 26: "Handbag", 28: "Suitcase"}

# Get ALL the images in the directory (removed the [:5] limit)
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])

print(f"Generating annotated boxes for all {len(image_files)} images...")

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
            x_center, y_center, box_w, box_h = map(float, parts[1:])
            
            # Convert math coordinates to image pixel locations
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)
            
            # Draw green box and text label on the frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = CLASS_MAP.get(cls_id, f"ID: {cls_id}")
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    # Save into your visual folder
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"visual_{img_name}"), img)

print(f"Done! Check the '{OUTPUT_DIR}' folder to see all your images.")
