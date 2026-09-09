from ultralytics import YOLO

print("🚀 Activating NVIDIA Blackwell RTX 5070 Pipeline...")
# We can now confidently train the higher-accuracy Small model!
model = YOLO("yolo11s.pt") 

print("🔥 Launching GPU-Accelerated Training...")
model.train(
    data="dataset.yaml", 
    epochs=50,          # Boosted to 50 epochs for outstanding professional accuracy
    imgsz=640,          # Full standard 640x640 resolution for crisp detail detection
    batch=16,           # Processes 16 images at once effortlessly using GPU memory
    device=0,           # CRITICAL: Forces YOLO to use CUDA on your Nvidia 5070 GPU
    workers=4           # Speeds up image loading using multi-threading
)

print("🎉 Training completed seamlessly on your RTX 5070!")
