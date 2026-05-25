import cv2
import time
import os
import sys
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--preview', action='store_true', help='Enable OpenCV preview window')
parser.add_argument('--camera', type=int, default=0, help='Camera index for OpenCV (macOS/webcam)')
parser.add_argument('--force-pi', action='store_true', help='Force use of PiCamera2 (Raspberry Pi)')
parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold for detection')
parser.add_argument('--fps', type=float, default=0, help='Limit processing FPS (0 = unlimited)')
args = parser.parse_args()

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

# Initialize camera
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

prev_frame = None

save_dir = "motion_captures"
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

# FPS calculation variables
fps_counter = 0
fps_start_time = time.time()
fps_display = 0.0

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
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        # Compute difference between current frame and previous frame
        frame_delta = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours of motion
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

        if motion_detected:
            now = time.time()
            if now - last_saved_time >= 1.0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"motion_{timestamp}.jpg")
                if IS_PI:
                    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(filename, frame)
                last_saved_time = now
        else:
            last_saved_time = 0  # Reset timer when no motion

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
            cv2.imshow("Motion Detection", frame)
        prev_frame = gray

        if args.preview and cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    if args.preview:
        cv2.destroyAllWindows()
    if IS_PI:
        picam2.close()
    else:
        cap.release()



