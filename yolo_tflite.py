# yolo_tflite.py - Fast YOLO detection using TensorFlow Lite
# Usage:
#   python yolo_tflite.py --preview        # real-time detection with preview
#
# Requirements:
#   pip install ultralytics opencv-python tensorflow
#
# Uses TensorFlow Lite models for faster inference on CPU

import argparse
import os
import sys
import time
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("Please install tensorflow: pip install tensorflow")
    sys.exit(1)

# ============================= Settings =============================
TFLITE_MODEL = "models/yolov8n_int8.tflite"  # TensorFlow Lite model
FALLBACK_MODEL = "models/yolov8n.pt"         # Fallback PyTorch model
IMG_SIZE = 640
CONF_THRESH = 0.25
NMS_THRESH = 0.45

parser = argparse.ArgumentParser()
parser.add_argument('--preview', action='store_true', help='Enable OpenCV preview window')
parser.add_argument('--camera', type=int, default=0, help='Camera index for OpenCV (macOS/webcam)')
parser.add_argument('--force-pi', action='store_true', help='Force use of PiCamera2 (Raspberry Pi)')
parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold for detection')
args = parser.parse_args()

save_dir = "yolo_captures"
os.makedirs(save_dir, exist_ok=True)

# Clear all existing pictures before starting
import glob
for file_path in glob.glob(os.path.join(save_dir, "*.jpg")):
    try:
        os.remove(file_path)
        print(f"[INFO] Removed: {file_path}")
    except Exception as e:
        print(f"[WARNING] Could not remove {file_path}: {e}")

last_saved_time = 0

# FPS calculation variables
fps_counter = 0
fps_start_time = time.time()
fps_display = 0.0

# Object tracking for movement arrows
object_positions = {}  # {class_name: [(center_x, center_y, timestamp), ...]}
ARROW_TIME_WINDOW = 1.0  # Show movement over 1 second

# Platform detection
def is_raspberry_pi():
    if args.force_pi:
        print('[INFO] Forcing Raspberry Pi mode via --force-pi')
        return True
    try:
        with open('/proc/device-tree/model') as f:
            return 'Raspberry Pi' in f.read()
    except Exception:
        return False

IS_PI = is_raspberry_pi()

# Initialize camera
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

# Try to load TensorFlow Lite model, fallback to PyTorch
use_tflite = False
interpreter = None
model = None
class_names = None

if os.path.exists(TFLITE_MODEL):
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        use_tflite = True
        print(f"[INFO] Loaded TensorFlow Lite model: {TFLITE_MODEL}")
        
        # Load class names from ultralytics YOLO
        temp_model = YOLO(FALLBACK_MODEL)
        class_names = temp_model.model.names
        
    except Exception as e:
        print(f"[WARNING] Failed to load TensorFlow Lite model: {e}")
        use_tflite = False

if not use_tflite:
    print(f"[INFO] Using fallback PyTorch model: {FALLBACK_MODEL}")
    model = YOLO(FALLBACK_MODEL)
    class_names = model.model.names


def preprocess_image(image):
    """Preprocess image for YOLO input"""
    # Resize to model input size
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension: (640, 640, 3) -> (1, 640, 640, 3)
    # TensorFlow Lite expects NHWC format, not NCHW
    img_input = np.expand_dims(img_normalized, axis=0)
    
    return img_input

def postprocess_tflite_output(output, original_shape, conf_threshold=0.25):
    """Post-process TensorFlow Lite YOLO output"""
    # TensorFlow Lite output shape is (1, 84, 8400) for YOLOv8
    # where 84 = 4 (bbox) + 80 (classes), and 8400 is the number of detections
    
    output = output[0]  # Remove batch dimension: (84, 8400)
    
    # Transpose to get (8400, 84) format
    output = output.T  # Now (8400, 84)
    
    # Extract bounding boxes and class probabilities
    boxes = output[:, :4]  # First 4 columns are bounding box coordinates
    class_probs = output[:, 4:]  # Remaining 80 columns are class probabilities
    
    # Find best class for each detection
    class_ids = np.argmax(class_probs, axis=1)
    confidences = np.max(class_probs, axis=1)
    
    
    # Filter by confidence threshold
    valid_indices = confidences > conf_threshold
    
    if not np.any(valid_indices):
        return [], [], []
    
    boxes = boxes[valid_indices]
    confidences = confidences[valid_indices]
    class_ids = class_ids[valid_indices]
    
    # YOLOv8 outputs normalized coordinates (0-1 range)
    # Scale boxes to original image size
    orig_h, orig_w = original_shape[:2]
    
    # Convert from normalized center format to corner format and scale
    x_center = boxes[:, 0] * orig_w    # normalized to pixel coordinates
    y_center = boxes[:, 1] * orig_h
    width = boxes[:, 2] * orig_w
    height = boxes[:, 3] * orig_h
    
    boxes[:, 0] = x_center - width / 2   # x1
    boxes[:, 1] = y_center - height / 2  # y1
    boxes[:, 2] = x_center + width / 2   # x2
    boxes[:, 3] = y_center + height / 2  # y2
    
    # Apply Non-Maximum Suppression to remove duplicate detections
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_threshold, NMS_THRESH)
    
    if len(indices) > 0:
        indices = indices.flatten()
        boxes = boxes[indices]
        confidences = confidences[indices]
        class_ids = class_ids[indices]
    else:
        boxes = []
        confidences = []
        class_ids = []
    
    return boxes, confidences, class_ids

def run_tflite_inference(image):
    """Run inference using TensorFlow Lite model"""
    start_time = time.time()
    
    # Preprocess image
    preprocess_start = time.time()
    input_data = preprocess_image(image)
    preprocess_time = time.time() - preprocess_start
    
    # Set input tensor and run inference
    inference_start = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    inference_time = time.time() - inference_start
    
    # Post-process
    postprocess_start = time.time()
    boxes, confidences, class_ids = postprocess_tflite_output(output_data, image.shape, args.confidence)
    postprocess_time = time.time() - postprocess_start
    
    total_time = time.time() - start_time
    
    return boxes, confidences, class_ids

def draw_detection_boxes(frame, boxes, confidences, class_ids):
    """Draw detection boxes with movement tracking"""
    current_time = time.time()
    
    
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        if conf > args.confidence:
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[int(cls_id)]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Track this object's position
            object_key = f"{class_name}_{int(cls_id)}"
            
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

# ============================== Main Loop ==============================
try:
    while True:
        # Get frame
        if IS_PI:
            frame = picam2.capture_array()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            rgb_frame = frame
        else:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        if use_tflite:
            boxes, confidences, class_ids = run_tflite_inference(rgb_frame)
        else:
            # Use PyTorch YOLO
            results = model(rgb_frame, imgsz=IMG_SIZE, conf=args.confidence, iou=NMS_THRESH, verbose=False)
            boxes, confidences, class_ids = [], [], []
            for r in results:
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    if conf >= args.confidence:
                        boxes.append(box.cpu().numpy())
                        confidences.append(conf.cpu().numpy())
                        class_ids.append(cls.cpu().numpy())
        
        # Draw detections on frame (same for both paths)
        if IS_PI:
            # On Pi, work directly with RGB frame to avoid color conversion
            frame = draw_detection_boxes(rgb_frame, boxes, confidences, class_ids)
        else:
            # On other platforms, draw on BGR frame
            frame = draw_detection_boxes(frame, boxes, confidences, class_ids)
        
        # Save detection if any objects found
        if len(boxes) > 0:
            now = time.time()
            if now - last_saved_time >= 1.0:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Get the highest confidence detection for filename
                if len(confidences) > 0:
                    best_idx = np.argmax(confidences)
                    best_class = class_names[int(class_ids[best_idx])]
                    filename = os.path.join(save_dir, f"{timestamp}_{best_class}.jpg")
                    if IS_PI:
                        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(filename, frame)
                    print(f"[INFO] Saved detection: {filename}")
                    last_saved_time = now

        # Calculate and display FPS
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= 1.0:
            fps_display = fps_counter / (current_time - fps_start_time)
            fps_counter = 0
            fps_start_time = current_time
        
        # Display FPS and model type
        model_type = "TFLite" if use_tflite else "PyTorch"
        cv2.putText(frame, f"FPS: {fps_display:.1f} ({model_type})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if args.preview:
            cv2.imshow("YOLO TensorFlow Lite Detection", frame)
        if args.preview and cv2.waitKey(1) & 0xFF == ord('q'):
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