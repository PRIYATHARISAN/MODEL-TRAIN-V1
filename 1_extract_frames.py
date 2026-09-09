import os
import cv2
from PIL import Image
import imagehash

# 1. Setup paths and folders
VIDEO_PATH = "singapore_mall.mp4"
OUTPUT_DIR = "extracted_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Open video and get properties
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))

frame_count = 0
saved_count = 0
last_hash = None

print("Extracting images and removing duplicates...")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Video finished

    frame_count += 1
    
    # Capture exactly 1 frame per second
    if frame_count % fps == 0:
        # Convert frame to image to check for duplicates
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        current_hash = imagehash.phash(pil_img)

        # Check if the image is a duplicate of the last one
        if last_hash is not None and (current_hash - last_hash) < 5:
            continue  # Skip duplicate frame

        last_hash = current_hash
        saved_count += 1
        
        # Save the clean unique image
        cv2.imwrite(f"{OUTPUT_DIR}/frame_{saved_count:05d}.jpg", frame)

cap.release()
print(f"Finished! Saved {saved_count} unique images into '{OUTPUT_DIR}'.")
