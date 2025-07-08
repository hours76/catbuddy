import cv2
import numpy as np
import time
import os
from datetime import datetime
import argparse

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--preview', action='store_true', help='Enable OpenCV preview window')
parser.add_argument('--camera', type=int, default=0, help='Camera index for OpenCV (macOS/webcam)')
args = parser.parse_args()

# Initialize camera for MacBook Pro (OpenCV)
cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

prev_frame = None

save_dir = "motion_captures"
os.makedirs(save_dir, exist_ok=True)
last_saved_time = 0

try:
    while True:
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
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(save_dir, f"motion_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                last_saved_time = now
        else:
            last_saved_time = 0  # Reset timer when no motion

        if args.preview:
            cv2.imshow("Motion Detection", frame)
        prev_frame = gray

        if args.preview and cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    cap.release()



