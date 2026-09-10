import cv2
from ultralytics import RTDETR

# 1. Load your newly minted Transformer Weights
# Change "train-detr" to "train-5"
MODEL_PATH = "runs/detect/train-5/weights/best.pt"

print(f"Loading custom RT-DETR from {MODEL_PATH}...")
model = RTDETR(MODEL_PATH)

# 2. Point it to your video file or webcam (0)
VIDEO_SOURCE = "singapore_mall.mp4" 

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error opening source: {VIDEO_SOURCE}")
    exit()

print("🚀 Displaying Live Transformer Inference Window...")
print("💡 Click inside the player window and press 'q' to shut down safely.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Transformers use global attention instead of local anchor boxes!
    results = model(frame, device=0, conf=0.33, verbose=False)
    
    annotated_frame = results[0].plot()
    cv2.imshow("Custom Real-Time Detection Transformer (RT-DETR)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Transformer demo shut down cleanly.")
