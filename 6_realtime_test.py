import cv2
from ultralytics import YOLO

# 1. Load your optimized train-4 model weights
MODEL_PATH = "runs/detect/train-4/weights/best.pt"
print(f"Loading custom model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# 2. Configure video source
# For a raw video file: Replace "your_new_video.mp4" with your actual file name
# For a live USB webcam stream: Change the string below to the integer 0 (e.g., VIDEO_SOURCE = 0)
VIDEO_SOURCE = "singapore_mall.mp4" 

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source '{VIDEO_SOURCE}'")
    exit()

print("🚀 Launching live RTX 5070 video detection pipeline...")
print("💡 Press the 'q' key on your keyboard inside the pop-up window to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or video file reached.")
        break

    # Run inference on the frame using your GPU
    # conf=0.33 matches the optimal F1-confidence score threshold found in your charts
    results = model(frame, device=0, conf=0.33, verbose=False)
    
    # Extract the image frame with the bounding boxes drawn on it
    annotated_frame = results[0].plot()

    # Display the processed frame live on your desktop screen
    cv2.imshow("Custom YOLOv11s Mall Crowd Tracker", annotated_frame)

    # Break the live loop instantly if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up window memory safely
cap.release()
cv2.destroyAllWindows()
print("Real-time pipeline shut down successfully.")
