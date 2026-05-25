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
from picamera2 import Picamera2

# ============================= Settings =============================
HEF_PATH        = "/usr/local/hailo/resources/models/hailo8l/yolov8s.hef"
POST_SO_PATH    = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so"
CONF_THRESH     = 0.40          # Only show detections above this confidence
TARGET_LABEL    = "cat"         # Label to highlight (all labels shown, cat highlighted)
CAMERA_WIDTH    = 800
CAMERA_HEIGHT   = 600
INFER_SIZE      = 640           # Hailo model input size
SAVE_DIR        = "hailo_captures"
SAVE_INTERVAL   = 1.0           # Minimum seconds between saves

os.makedirs(SAVE_DIR, exist_ok=True)

# Clear existing captures
import glob
for f in glob.glob(f"{SAVE_DIR}/*.jpg"):
    os.remove(f)

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
    lores={"size": (INFER_SIZE, INFER_SIZE), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(1)
print(f"[INFO] Camera started: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

# ============================= Hailo pipeline ======================
Gst.init(None)

PIPELINE_STR = (
    f"appsrc name=src is-live=true format=time ! "
    f"video/x-raw,format=RGB,width={INFER_SIZE},height={INFER_SIZE},framerate=30/1 ! "
    f"videoconvert ! "
    f"queue leaky=no max-size-buffers=3 ! "
    f"hailonet hef-path={HEF_PATH} batch-size=1 "
    f"nms-score-threshold={CONF_THRESH} nms-iou-threshold=0.45 "
    f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32 ! "
    f"queue leaky=no max-size-buffers=3 ! "
    f"hailofilter so-path={POST_SO_PATH} function-name=filter qos=false ! "
    f"queue leaky=no max-size-buffers=3 ! "
    f"identity name=det_sink ! "
    f"fakesink"
)

pipeline = Gst.parse_launch(PIPELINE_STR)
appsrc = pipeline.get_by_name("src")
det_sink = pipeline.get_by_name("det_sink")

def on_detection(pad, info):
    global latest_detections
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    roi = hailo.get_roi_from_buffer(buffer)
    detections = []
    for det in roi.get_objects_typed(hailo.HAILO_DETECTION):
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
    return Gst.PadProbeReturn.OK

det_sink.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_detection)

pipeline.set_state(Gst.State.PLAYING)
print("[INFO] Hailo pipeline started")

# ============================= Feed thread =========================
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

        with det_lock:
            detections = list(latest_detections)

        h, w = frame.shape[:2]
        cat_detected = False

        for det in detections:
            label = det["label"]
            conf = det["conf"]
            x1 = int(det["xmin"] * w)
            y1 = int(det["ymin"] * h)
            x2 = int(det["xmax"] * w)
            y2 = int(det["ymax"] * h)

            is_cat = label.lower() == TARGET_LABEL
            if is_cat:
                cat_detected = True
                color = (0, 255, 0)   # green for cat
            else:
                color = (128, 128, 128)  # gray for others

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # FPS
        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps_display = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now
        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save if cat detected
        if cat_detected and (now - last_saved_time) >= SAVE_INTERVAL:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SAVE_DIR, f"{ts}_cat.jpg")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            last_saved_time = now
            print(f"[INFO] Saved: {path}")

        if args.preview:
            cv2.imshow("Hailo Cat Detection - press q to quit",
                       cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        time.sleep(0.01)

except KeyboardInterrupt:
    pass
finally:
    running = False
    pipeline.set_state(Gst.State.NULL)
    picam2.stop()
    if args.preview:
        cv2.destroyAllWindows()
    print("[INFO] Stopped.")
