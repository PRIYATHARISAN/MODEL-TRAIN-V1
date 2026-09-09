import os
from ultralytics import YOLO

# 1. Load YOUR custom trained model brain
model = YOLO("runs/detect/train/weights/best.pt")

# 2. Automatically grab the first image found in the validation folder
VAL_DIR = "mall_dataset/val/images"

if os.path.exists(VAL_DIR) and len(os.listdir(VAL_DIR)) > 0:
    # Get a sorted list of all images available in validation
    available_images = sorted([f for f in os.listdir(VAL_DIR) if f.endswith(".jpg")])
    TEST_IMAGE = os.path.join(VAL_DIR, available_images[0])
    
    print(f"Found test image: {TEST_IMAGE}")
    print("Running custom model inference...")
    
    # Run the model and save the visual result box automatically
    model(TEST_IMAGE, save=True, project="test_results", name="predictions")
    
    print("\n🎉 Success! Look inside the new folder 'test_results/predictions/' to see your custom model working!")
else:
    print(f"Error: Could not find any images inside '{VAL_DIR}'. Check your folders!")
