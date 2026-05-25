import argparse
import os
import sys
import time
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument('--preview', action='store_true', help='Enable OpenCV preview window')
parser.add_argument('--camera', type=int, default=0, help='Camera index for OpenCV (macOS/webcam)')
parser.add_argument('--force-pi', action='store_true', help='Force use of PiCamera2 (Raspberry Pi)')
parser.add_argument('--fps', type=float, default=0, help='Limit processing FPS (0 = unlimited)')
args = parser.parse_args()

save_dir = "triple_captures"
os.makedirs(save_dir, exist_ok=True)

# Clear all existing pictures before starting
import glob
for file_path in glob.glob(os.path.join(save_dir, "*.jpg")):
    try:
        os.remove(file_path)
        print(f"[INFO] Removed: {file_path}")
    except Exception as e:
        print(f"[WARNING] Could not remove {file_path}: {e}")
print(f"[INFO] Cleared all existing images from {save_dir}")

last_saved_time = 0

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
print(f"[INFO] Detected Raspberry Pi: {IS_PI}")

if IS_PI:
    try:
        from picamera2 import Picamera2 # type: ignore
    except ImportError:
        print("Please install picamera2: pip install picamera2")
        sys.exit(1)
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (800, 600)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.start()
    time.sleep(2)
else:
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

model = YOLO("models/yolov8n.pt")
object_class_id = next(k for k, v in model.model.names.items() if v.lower() == "cat")

prev_gray = None

# Sensitivity for motion detection (minimum contour area)
MOTION_SENSITIVITY = 500  # Lower = more sensitive, higher = less sensitive

# FPS calculation variables
fps_counter = 0
fps_start_time = time.time()
fps_display = 0.0

# Use OpenCV tracker after object detection
class CVTracker:
    def __init__(self, tracker_type="CSRT"):
        self.tracker_type = tracker_type
        self.tracker = None
        self.active = False
        self.last_bbox = None
        self.last_seen = 0
        self.prev_center = None
        self.curr_center = None
        self.last_movement_time = time.time()
        self.use_centroid_tracking = False

    def start(self, frame, bbox):
        # bbox: (x1, y1, x2, y2)
        self.active = True
        self.last_bbox = bbox
        self.last_seen = time.time()
        self.prev_center = self.curr_center
        x1, y1, x2, y2 = bbox
        self.curr_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.last_movement_time = time.time()
        tracker_bbox = (x1, y1, x2 - x1, y2 - y1)
        # Try CSRT first (best performance), fallback to MIL
        self.tracker = None
        tracker_types = ["CSRT", "MIL"]
        
        for tracker_name in tracker_types:
            try:
                if tracker_name == "CSRT":
                    if hasattr(cv2, 'TrackerCSRT_create'):
                        self.tracker = cv2.TrackerCSRT_create()  # type: ignore
                    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                        self.tracker = cv2.legacy.TrackerCSRT_create()  # type: ignore
                elif tracker_name == "MIL":
                    if hasattr(cv2, 'TrackerMIL_create'):
                        self.tracker = cv2.TrackerMIL_create()  # type: ignore
                    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMIL_create'):
                        self.tracker = cv2.legacy.TrackerMIL_create()  # type: ignore
                
                if self.tracker is not None:
                    self.tracker.init(frame, tracker_bbox)
                    print(f"[INFO] Successfully initialized {tracker_name} tracker")
                    break
                    
            except Exception as e:
                print(f"[DEBUG] Failed to create {tracker_name} tracker: {e}")
                self.tracker = None
                continue
        
        # If no OpenCV tracker works, use simple centroid tracking as fallback
        if self.tracker is None:
            print("[INFO] No OpenCV tracker available, using simple centroid tracking")
            self.use_centroid_tracking = True
            self.tracker = "centroid"  # Use string to indicate centroid tracking
        else:
            self.use_centroid_tracking = False

    def update(self, frame):
        if self.active and self.tracker is not None:
            # Handle centroid tracking fallback
            if self.use_centroid_tracking:
                # Simple centroid tracking - just show last known position
                if self.last_bbox:
                    x1, y1, x2, y2 = self.last_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, 'Centroid Tracking', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Show movement arrow if we have previous center
                    if self.prev_center and self.curr_center:
                        cv2.arrowedLine(frame, self.prev_center, self.curr_center, (255, 0, 255), 2, tipLength=0.3)
                    
                    # Auto-stop centroid tracking after 5 seconds
                    if time.time() - self.last_seen > 5.0:
                        print("[INFO] Centroid tracking timeout, stopping tracker.")
                        self.active = False
                        self.tracker = None
                        self.last_bbox = None
                        self.prev_center = None
                        self.curr_center = None
            else:
                # Use OpenCV tracker
                if self.tracker is not None and hasattr(self.tracker, 'update'):
                    ok, bbox = self.tracker.update(frame)
                    if ok:
                        x, y, w, h = [int(v) for v in bbox]
                        self.last_bbox = (x, y, x + w, y + h)
                        self.prev_center = self.curr_center
                        self.curr_center = (x + w // 2, y + h // 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, 'Tracking', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        movement = 0
                        if self.prev_center and self.curr_center:
                            cv2.arrowedLine(frame, self.prev_center, self.curr_center, (255, 0, 255), 2, tipLength=0.3)
                            dx = self.curr_center[0] - self.prev_center[0]
                            dy = self.curr_center[1] - self.prev_center[1]
                            movement = (dx**2 + dy**2) ** 0.5
                            print(f"Movement vector: dx={dx}, dy={dy}, movement={movement:.2f}")
                            # If movement is significant, update last_movement_time
                            if movement >= 10:
                                self.last_movement_time = time.time()
                        self.last_seen = time.time()
                        # Stop tracking if only small movement (<10) for 3 seconds
                        if time.time() - self.last_movement_time > 3.0:
                            print("[INFO] Only small movement (<10) for 3 seconds, stopping tracker.")
                            self.active = False
                            self.tracker = None
                            self.last_bbox = None
                            self.prev_center = None
                            self.curr_center = None
                    else:
                        self.active = False  # Lost tracking, return to motion detection
                        self.tracker = None
                        self.last_bbox = None
                        self.prev_center = None
                        self.curr_center = None
        return frame

cv_tracker = CVTracker()

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
        # Get frame
        if IS_PI:
            frame = picam2.capture_array()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            rgb_frame = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        # Only run motion detection if not actively tracking
        if not cv_tracker.active:
            # Motion detection
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get all valid contours above threshold
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= MOTION_SENSITIVITY:
                    valid_contours.append(contour)
        else:
            # Skip motion detection while tracking is active
            valid_contours = []
        
        if valid_contours:
            # Convert contours to bounding rectangles
            rects = []
            for contour in valid_contours:
                x, y, w, h = cv2.boundingRect(contour)
                rects.append((x, y, x + w, y + h))
            
            # Combine nearby rectangles
            DISTANCE_THRESHOLD = 100  # pixels - adjust as needed
            combined_rects = []
            used = [False] * len(rects)
            
            for i, rect1 in enumerate(rects):
                if used[i]:
                    continue
                
                # Start with current rectangle
                min_x, min_y, max_x, max_y = rect1
                combined_group = [i]
                
                # Find nearby rectangles to combine
                for j, rect2 in enumerate(rects):
                    if i == j or used[j]:
                        continue
                    
                    # Calculate distance between rectangle centers
                    center1 = ((rect1[0] + rect1[2]) // 2, (rect1[1] + rect1[3]) // 2)
                    center2 = ((rect2[0] + rect2[2]) // 2, (rect2[1] + rect2[3]) // 2)
                    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
                    
                    if distance <= DISTANCE_THRESHOLD:
                        # Expand bounding box to include this rectangle
                        min_x = min(min_x, rect2[0])
                        min_y = min(min_y, rect2[1])
                        max_x = max(max_x, rect2[2])
                        max_y = max(max_y, rect2[3])
                        combined_group.append(j)
                
                # Mark all rectangles in this group as used
                for idx in combined_group:
                    used[idx] = True
                
                # Add combined rectangle
                combined_rects.append((min_x, min_y, max_x, max_y))
            
            # Process all combined rectangles (not just the largest)
            if combined_rects:
                # Sort by area (largest first) and process up to 3 rectangles
                sorted_rects = sorted(combined_rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
                
                for rect_idx, rect in enumerate(sorted_rects[:3]):  # Process up to 3 largest rectangles
                    x, y, max_x, max_y = rect
                    w, h = max_x - x, max_y - y
                    
                    # Add padding to capture more context (10% of width/height)
                    padding_x = max(20, int(w * 0.1))
                    padding_y = max(20, int(h * 0.1))
                    
                    # Apply padding with bounds checking
                    padded_x = max(0, x - padding_x)
                    padded_y = max(0, y - padding_y)
                    padded_w = min(rgb_frame.shape[1] - padded_x, w + 2 * padding_x)
                    padded_h = min(rgb_frame.shape[0] - padded_y, h + 2 * padding_y)
                    
                    roi = rgb_frame[padded_y:padded_y+padded_h, padded_x:padded_x+padded_w]
                    results = model(roi)
                    
                    for r in results:
                        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                            if conf >= 0.30:  # Show all objects with confidence >= 0.30
                                x1, y1, x2, y2 = map(int, box)
                                # Adjust coordinates back to original frame (account for padding)
                                abs_box = (padded_x+x1, padded_y+y1, padded_x+x2, padded_y+y2)
                                if hasattr(model, 'model') and model.model is not None and hasattr(model.model, 'names'):
                                    class_name = model.model.names[int(cls)]  # type: ignore
                                else:
                                    class_name = f"class_{int(cls)}"
                                
                                # Different colors for different object types
                                if int(cls) == object_class_id:
                                    color = (0, 255, 255)  # Yellow for object detected
                                    cv2.rectangle(frame, (abs_box[0], abs_box[1]), (abs_box[2], abs_box[3]), color, 2)
                                    cv2.putText(frame, f"{class_name} {conf:.2f}", (abs_box[0], abs_box[1]-4),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                    # Start OpenCV tracker after object detection
                                    cv_tracker.start(frame, abs_box)
                                    now = time.time()
                                    if now - last_saved_time >= 1.0:
                                        from datetime import datetime
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        filename = os.path.join(save_dir, f"{timestamp}_triple_object.jpg")
                                        if IS_PI:
                                            cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                                        else:
                                            cv2.imwrite(filename, frame)
                                        last_saved_time = now
                                    # Stop processing more rectangles once object is detected and tracking starts
                                    break
                                else:
                                    color = (255, 0, 255)  # Magenta for other objects
                                    cv2.rectangle(frame, (abs_box[0], abs_box[1]), (abs_box[2], abs_box[3]), color, 2)
                                    cv2.putText(frame, f"{class_name} {conf:.2f}", (abs_box[0], abs_box[1]-4),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # If object detected and tracking started, stop processing more rectangles
                        if cv_tracker.active:
                            break
                    
                    # Draw motion detection rectangle AFTER object detection (so it's on top)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for motion detection
                    cv2.putText(frame, f"MOTION {rect_idx+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # After detection, update tracker if active
        frame = cv_tracker.update(frame)
        # Save tracking progress if tracker is active (max 1 per second)
        if cv_tracker.active and cv_tracker.last_bbox is not None:
            now = time.time()
            if now - last_saved_time >= 1.0:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"{timestamp}_object_track.jpg")
                if IS_PI:
                    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(filename, frame)
                last_saved_time = now

        # Calculate and display FPS
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= 1.0:  # Update FPS every second
            fps_display = fps_counter / (current_time - fps_start_time)
            fps_counter = 0
            fps_start_time = current_time
        
        # Display FPS on frame
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if args.preview:
            cv2.imshow("Motion+Object Detection", frame)
        if args.preview and cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = gray
finally:
    if args.preview:
        cv2.destroyAllWindows()
    if IS_PI:
        picam2.close()
    else:
        cap.release()

