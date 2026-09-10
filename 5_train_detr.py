from ultralytics import RTDETR

print(" Activating NVIDIA RTX 5070 Transformer Pipeline...")
# Load the pre-trained RT-DETR Small or Medium model weights
model = RTDETR("rtdetr-l.pt")  # 'l' stands for Large. You can use 'rtdetr-x.pt' for Extra Large!

print(" Launching Transformer Training Loop...")
model.train(
    data="dataset.yaml", 
    epochs=50,          # Keep it at 50 loops to let the transformer layers adapt to humans
    imgsz=640,          # Standard crisp pixel layout 
    batch=16,           # Processes 16 images effortlessly using your 5070 GPU VRAM
    device=0,           # Forces CUDA processing on your graphics hardware
    workers=4           
)

print(" Transformer training completed successfully!")
