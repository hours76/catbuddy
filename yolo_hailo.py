# hailo.py - Cat detection using Hailo AI Hat + Pi Camera Module 3
# Usage:
#   python hailo.py [--preview] [--confidence FLOAT]
#
# Requirements:
#   sudo apt install hailo-all
#   source ~/work/hailo-rpi5-examples/setup_env.sh
#
# Uses Hailo-8L NPU for inference (~80+ FPS), much faster than CPU YOLO.
# HEF model: yolov8s.hef (hailo8l) from /usr/local/hailo/resources/models/hailo8l/

import os
import sys
import time
import pilot
import argparse
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import hailo
import libcamera
from picamera2 import Picamera2

# ============================= Settings =============================
HEF_PATH        = "/usr/local/hailo/resources/models/hailo8l/yolov8s.hef"
POST_SO_PATH    = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so"
CONF_THRESH     = 0.40          # Only show detections above this confidence
TARGET_LABEL    = "person"      # Label to highlight (all labels shown, target highlighted)
CAMERA_WIDTH    = 1280
CAMERA_HEIGHT   = 720
INFER_SIZE      = 640           # Hailo model input size
SAVE_INTERVAL   = 1.0           # Minimum seconds between saves

os.makedirs("hailo_captures", exist_ok=True)  # base dir

# Session folder named by start datetime
_session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join("hailo_captures", _session_ts)
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================= Args ================================
parser = argparse.ArgumentParser(description="Cat detection with Hailo AI Hat")
parser.add_argument("--preview", action="store_true", help="Show live OpenCV window")
parser.add_argument("--confidence", type=float, default=CONF_THRESH, help="Detection confidence threshold")
args = parser.parse_args()
CONF_THRESH = args.confidence

# ============================= Shared state ========================
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()
det_lock = threading.Lock()
fps_counter = 0
fps_display = 0.0
fps_time = time.time()
last_saved_time = 0.0
running = True

# ============================= Camera ==============================
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
    lores={"size": (INFER_SIZE, INFER_SIZE), "format": "RGB888"},
    transform=libcamera.Transform(hflip=1, vflip=1)  # rotate 180 degrees
)
picam2.configure(config)
picam2.start()
print(f"[INFO] Camera started: {CAMERA_WIDTH}x{CAMERA_HEIGHT}", flush=True)
time.sleep(1)

# ============================= Hailo pipeline ======================
print("[INFO] Initializing GStreamer...", flush=True)
Gst.init(None)
print("[INFO] GStreamer initialized", flush=True)

PIPELINE_STR = (
    f"appsrc name=src is-live=true format=time ! "
    f"video/x-raw,format=RGB,width={INFER_SIZE},height={INFER_SIZE},framerate=30/1 ! "
    f"videoconvert ! "
    f"queue leaky=no max-size-buffers=3 ! "
    f"hailonet hef-path={HEF_PATH} batch-size=1 force-writable=true "
    f"nms-score-threshold={CONF_THRESH} nms-iou-threshold=0.45 "
    f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32 ! "
    f"queue leaky=no max-size-buffers=3 ! "
    f"hailofilter so-path={POST_SO_PATH} function-name=filter qos=false ! "
    f"queue leaky=no max-size-buffers=3 ! "
    f"identity name=det_sink ! "
    f"fakesink"
)

print("[INFO] Parsing pipeline...", flush=True)
pipeline = Gst.parse_launch(PIPELINE_STR)
print("[INFO] Pipeline parsed", flush=True)
appsrc = pipeline.get_by_name("src")
det_sink = pipeline.get_by_name("det_sink")
print("[INFO] Got pipeline elements", flush=True)

_probe_call_count = 0

def on_detection(pad, info):
    global latest_detections, _probe_call_count
    _probe_call_count += 1
    if _probe_call_count <= 3 or _probe_call_count % 100 == 0:
        print(f"[DEBUG] probe called #{_probe_call_count}", flush=True)
    buffer = info.get_buffer()
    if buffer is None:
        print("[DEBUG] buffer is None")
        return Gst.PadProbeReturn.OK
    try:
        roi = hailo.get_roi_from_buffer(buffer)
        objects = roi.get_objects_typed(hailo.HAILO_DETECTION)
        if _probe_call_count <= 3:
            print(f"[DEBUG] detections count: {len(objects)}")
        detections = []
        for det in objects:
            label = det.get_label()
            conf = det.get_confidence()
            bbox = det.get_bbox()
            detections.append({
                "label": label,
                "conf": conf,
                "xmin": bbox.xmin(),
                "ymin": bbox.ymin(),
                "xmax": bbox.xmax(),
                "ymax": bbox.ymax(),
            })
        with det_lock:
            latest_detections = detections
    except Exception as e:
        if _probe_call_count <= 3:
            print(f"[DEBUG] probe exception: {e}")
    return Gst.PadProbeReturn.OK

# Try sink pad (receives buffer with hailo metadata attached)
print("[INFO] Adding probe...", flush=True)
det_sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, on_detection)
print("[INFO] Probe added, starting pipeline...", flush=True)
pipeline.set_state(Gst.State.PLAYING)
print("[INFO] Hailo pipeline started", flush=True)
pilot.start()
print("[INFO] Pilot started", flush=True)

# ============================= Tracking state ======================
object_positions = {}  # {label: [(cx, cy, timestamp), ...]}
ARROW_TIME_WINDOW = 1.0  # seconds

def feed_frames():
    global running
    while running:
        # Grab lores frame for inference
        lores = picam2.capture_array("lores")
        # Also grab main frame for display/save
        main = picam2.capture_array("main")
        with frame_lock:
            global latest_frame
            latest_frame = main.copy()
        # Push lores to GStreamer appsrc
        data = lores.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.pts = buf.dts = int(time.time() * 1e9)
        buf.duration = int(1e9 / 30)
        appsrc.emit("push-buffer", buf)

feed_thread = threading.Thread(target=feed_frames, daemon=True)
feed_thread.start()

# ============================= Main loop ===========================
print("[INFO] Running... Press Ctrl+C to quit")
try:
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue

        # picamera2 RGB888 via capture_array("main") delivers BGR — use directly
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        with det_lock:
            detections = list(latest_detections)

        h, w = frame.shape[:2]
        frame_cx = w // 2
        frame_cy = h // 2
        target_detected = False
        target_cx = 0
        target_cy = 0
        cat_moving = False
        now = time.time()

        for det in detections:
            label = det["label"]
            conf = det["conf"]
            x1 = int(det["xmin"] * w)
            y1 = int(det["ymin"] * h)
            x2 = int(det["xmax"] * w)
            y2 = int(det["ymax"] * h)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            is_cat = label.lower() == TARGET_LABEL
            if is_cat:
                target_detected = True
                target_cx = cx
                target_cy = cy
                box_color = (0, 255, 0)     # green for target
                text_color = (255, 255, 255) # white label
            else:
                box_color = (128, 128, 128)  # grey for others
                text_color = (180, 180, 180) # grey label

            # Track position history — cat only
            if is_cat:
                if label not in object_positions:
                    object_positions[label] = []
                object_positions[label].append((cx, cy, now))

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Label: white text on dark background, top-right corner of bbox
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx = max(x2 - tw, 0)
            ty = max(y1 - 4, th + 2)
            cv2.rectangle(frame, (tx, ty - th - 2), (tx + tw, ty + 2), (0, 0, 0), -1)
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            # Draw movement arrow — cat only
            if is_cat:
                positions = object_positions.get(label, [])
                start_pos = None
                for px, py, pt in positions:
                    if now - pt >= ARROW_TIME_WINDOW:
                        start_pos = (px, py)
                        break
                if start_pos:
                    dist = ((cx - start_pos[0])**2 + (cy - start_pos[1])**2) ** 0.5
                    if dist > 10:
                        cat_moving = True
                        cv2.arrowedLine(frame, start_pos, (cx, cy), (255, 0, 255), 2, tipLength=0.3)
                        cv2.putText(frame, f"1s:{dist:.0f}px", (cx + 8, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        # Clean up old positions (> 2 seconds)
        for key in list(object_positions.keys()):
            object_positions[key] = [(x, y, t) for x, y, t in object_positions[key]
                                     if now - t <= 2.0]

        # FPS
        fps_counter += 1
        if now - fps_time >= 1.0:
            fps_display = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now
        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Update pilot state
        pilot.update_state(
            detected=target_detected,
            cx=target_cx,
            cy=target_cy,
            dx=target_cx - frame_cx if target_detected else 0,
            dy=target_cy - frame_cy if target_detected else 0,
        )

        # Save only when cat is moving
        if cat_moving and (now - last_saved_time) >= SAVE_INTERVAL:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SAVE_DIR, f"{ts}_cat.jpg")
            cv2.imwrite(path, frame)
            last_saved_time = now
            print(f"[INFO] Saved: {path}")

        if args.preview:
            cv2.imshow("Hailo Cat Detection - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        time.sleep(0.01)

except KeyboardInterrupt:
    pass
finally:
    running = False
    pilot.stop_pilot()
    pipeline.set_state(Gst.State.NULL)
    picam2.stop()
    if args.preview:
        cv2.destroyAllWindows()
    print("[INFO] Stopped.")
