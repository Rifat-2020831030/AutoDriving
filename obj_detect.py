import cv2
import torch
import time
from ultralytics import YOLO

# Load the trained model
model = YOLO("./model/best.pt")  

def detect_objects(image):
    """
    Detects traffic signs in the input image.
    
    Args:
        Input image
        
    Returns:
        results: YOLO model output
        detections: List of with detected obj as tuples (class_name, (x_min, y_min, x_max, y_max))
    """
    results = model(image, conf=0.5)  # Run inference
    detections = []

    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]  # Get class name
            confidence = float(box.conf[0])
            detections.append((class_name, (x_min, y_min, x_max, y_max), confidence))

    return detections


def process_webcam(resize_factor=0.5):
    """
    Process real-time webcam video and display detections.
    """
    cap = cv2.VideoCapture(0)  # Open default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_skip = 2  # Process every 2nd frame for efficiency
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip processing some frames

        # Resize frame to improve speed
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (int(width * resize_factor), int(height * resize_factor)))

        start_time = time.time()
        detections = detect_objects(frame)
        end_time = time.time()

        # Draw bounding boxes only if detections exist
        if detections:
            for detection in detections:
                # if len(detection) != 2:
                #     continue  # Skip invalid detections

                class_name, (x_min, y_min, x_max, y_max), confidence = detection
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # Draw bounding box
                cv2.putText(frame, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2) # Display class name
                cv2.putText(frame, f"{confidence:.2f}", (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2) # Display confidence
                

        # Display FPS
        fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 255), 2)

        cv2.imshow("Webcam Traffic Sign Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()