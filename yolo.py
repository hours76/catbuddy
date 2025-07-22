# cat_detector_picamera2.py - Detect cats using YOLOv8‑nano with Picamera2
# Usage:
#   python cat_detector_picamera2.py        # real‑time detection from Pi Camera
#
# Requirements (Raspberry Pi OS Bookworm or later):
#   sudo apt install python3-picamera2 libcamera-apps
#   pip install ultralytics opencv-python
#
# Expect about 2 FPS on Raspberry Pi 4/5 CPU with yolov8n.pt.

import cv2
from ultralytics import YOLO
import os
import sys
import time
from datetime import datetime
import argparse

# ============================= Settings =============================
MODEL_PATH   = "models/yolov8n.pt"   # Nano model (3 MB)
IMG_SIZE     = 640             # Inference resolution
CONF_THRESH  = 0.25
NMS_THRESH   = 0.45

save_dir = "yolo_captures"
os.makedirs(save_dir, exist_ok=True)

# Clear all existing pictures before starting
import glob
files_removed = 0
for file_path in glob.glob(os.path.join(save_dir, "*.jpg")):
    try:
        os.remove(file_path)
        files_removed += 1
    except Exception as e:
        print(f"[WARNING] Could not remove {file_path}: {e}")
if files_removed > 0:
    print(f"[INFO] Cleared {files_removed} existing images from {save_dir}")

last_saved_time = 0

# FPS calculation variables
fps_counter = 0
fps_start_time = time.time()
fps_display = 0.0

# Object tracking for movement arrows
object_positions = {}  # {class_name: [(center_x, center_y, timestamp), ...]}
ARROW_TIME_WINDOW = 1.0  # Show movement over 1 second

# ====================== Configure Arguments =====================
parser = argparse.ArgumentParser()
parser.add_argument('--preview', action='store_true', help='Enable OpenCV preview window')
parser.add_argument('--camera', type=int, default=0, help='Camera index for OpenCV (macOS/webcam)')
parser.add_argument('--force-pi', action='store_true', help='Force use of PiCamera2 (Raspberry Pi)')
parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold for detection')
parser.add_argument('--fps', type=float, default=0, help='Limit processing FPS (0 = unlimited)')
parser.add_argument('--cpu', action='store_true', help='Force CPU-only execution (disable Metal)')
args = parser.parse_args()

# ====================== Load YOLO model ====================
model = YOLO(MODEL_PATH)
print(f"[INFO] Loaded YOLO model: {MODEL_PATH}")

# Enable Metal (MPS) acceleration if available
import torch
if torch.backends.mps.is_available() and not args.cpu:
    try:
        device = torch.device('mps')
        model.model = model.model.to(device)
        print(f"[INFO] Using Metal Performance Shaders (MPS) acceleration")
        print(f"[INFO] Model device: {next(model.model.parameters()).device}")
    except Exception as e:
        print(f"[WARNING] Failed to move model to MPS: {e}")
        print(f"[INFO] Falling back to CPU")
else:
    if args.cpu:
        print(f"[INFO] CPU-only mode requested")
    else:
        print(f"[INFO] MPS not available, using CPU")

# Platform detection
def is_raspberry_pi():
    if args.force_pi:
        return True
    try:
        with open('/proc/device-tree/model') as f:
            return 'Raspberry Pi' in f.read()
    except Exception:
        return False

IS_PI = is_raspberry_pi()

# Initialize camera based on platform
if IS_PI:
    try:
        from picamera2 import Picamera2 # type: ignore
    except ImportError:
        print("Please install picamera2: pip install picamera2")
        sys.exit(1)
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1920, 1080)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.start()
    time.sleep(2)
else:
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("[INFO] Press q in the window or Ctrl+C in terminal to quit")

# ---------------------- Helper: draw cat boxes ----------------------

def draw_detection_boxes(frame, detections):
    current_time = time.time()
    
    for r in detections:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if conf > args.confidence:
                x1, y1, x2, y2 = map(int, box)
                class_name = model.model.names[int(cls)]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Track this object's position
                object_key = f"{class_name}_{int(cls)}"
                
                # Add current position to history
                if object_key not in object_positions:
                    object_positions[object_key] = []
                object_positions[object_key].append((center_x, center_y, current_time))
                
                # Color based on object type
                if class_name in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']:
                    color = (0, 255, 0)  # Green for vehicles/people
                elif class_name in ['cat', 'dog', 'bird', 'horse', 'sheep', 'cow']:
                    color = (255, 0, 0)  # Blue for animals
                else:
                    color = (0, 0, 255)  # Red for other objects
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{class_name} {conf:.2f}",
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
                
                # Draw movement arrow based on 1-second movement
                if object_key in object_positions and len(object_positions[object_key]) > 1:
                    # Find position from 1 second ago
                    positions = object_positions[object_key]
                    start_pos = None
                    for pos_x, pos_y, timestamp in positions:
                        if current_time - timestamp >= ARROW_TIME_WINDOW:
                            start_pos = (pos_x, pos_y)
                            break
                    
                    if start_pos:
                        start_x, start_y = start_pos
                        distance = ((center_x - start_x)**2 + (center_y - start_y)**2)**0.5
                        if distance > 10:  # Only show if moved >10 pixels in 1 second
                            cv2.arrowedLine(frame, (start_x, start_y), (center_x, center_y), 
                                          (255, 0, 255), 2, tipLength=0.3)
                            # Show movement distance over 1 second
                            cv2.putText(frame, f"1s: {distance:.0f}px", 
                                      (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (255, 0, 255), 1)
    
    # Clean up old positions (older than 2 seconds) for each object
    for object_key in list(object_positions.keys()):
        if object_key in object_positions:
            # Remove positions older than 2 seconds
            object_positions[object_key] = [
                (x, y, t) for x, y, t in object_positions[object_key]
                if current_time - t <= 2.0
            ]
            # Remove object entirely if no recent positions
            if not object_positions[object_key]:
                del object_positions[object_key]
    
    return frame

# ============================== Loop ================================
# FPS limiting variables
frame_time_target = 1.0 / args.fps if args.fps > 0 else 0
last_frame_time = time.time()

if args.fps > 0:
    print(f"[INFO] FPS limited to {args.fps} FPS")
else:
    print("[INFO] FPS unlimited")

try:
    while True:
        # FPS limiting
        if args.fps > 0:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_time_target:
                time.sleep(frame_time_target - elapsed)
            last_frame_time = time.time()
        # Get frame based on platform
        if IS_PI:
            frame = picam2.capture_array()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        else:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break
        
        # Time the inference
        inference_start = time.time()
        results = model(
            frame,
            imgsz=IMG_SIZE,
            conf=args.confidence,
            iou=NMS_THRESH,
            verbose=False,
        )
        inference_time = (time.time() - inference_start) * 1000  # Convert to milliseconds
        
        # Print timing info every 30 frames
        if fps_counter % 30 == 0:
            print(f"[DEBUG] Inference time: {inference_time:.1f}ms")
        frame = draw_detection_boxes(frame, results)

        # Check if any object is detected and save
        any_detection = any(
            conf > args.confidence
            for r in results for conf in r.boxes.conf
        )
        if any_detection:
            now = time.time()
            if now - last_saved_time >= 1.0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Get the highest confidence detection for filename
                best_detection = max(
                    [(model.model.names[int(cls)], conf) for r in results 
                     for cls, conf in zip(r.boxes.cls, r.boxes.conf) if conf > args.confidence],
                    key=lambda x: x[1], default=("object", 0)
                )
                filename = os.path.join(save_dir, f"{timestamp}_{best_detection[0]}.jpg")
                if IS_PI:
                    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(filename, frame)
                print(f"[INFO] Saved detection: {filename}")
                last_saved_time = now
        else:
            last_saved_time = 0  # Reset timer when no objects detected

        # Calculate and display FPS
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= 1.0:
            fps_display = fps_counter / (current_time - fps_start_time)
            fps_counter = 0
            fps_start_time = current_time
        
        # Display FPS on frame
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if args.preview:
            cv2.imshow("YOLO Object Detection - press q to quit", frame)
        if args.preview and cv2.waitKey(1) & 0xFF == ord("q"):
            break
except KeyboardInterrupt:
    pass
finally:
    if args.preview:
        cv2.destroyAllWindows()
    if IS_PI:
        picam2.close()
    else:
        cap.release()
