from ultralytics import YOLO

print("Loading lightweight YOLOv11 Nano model...")
model = YOLO("yolo11n.pt")

print("Starting custom training pipeline...")
model.train(
    data="dataset.yaml", 
    epochs=25,         # 25 loops is perfect for a great school presentation
    imgsz=640,         # Standard input size
    device="cpu"       # Uses your computer processor
)

print("Training finished! Check 'runs/detect/train/' for your final results!")
