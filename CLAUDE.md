# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI robot that plays with cats — computer vision project for motion detection and object tracking using OpenCV and YOLOv8, running on Raspberry Pi 5 with Pi Camera Module 3.

The project consists of four main Python scripts:

- `motion.py`: Basic motion detection that captures frames when motion is detected
- `triple.py`: Advanced motion detection with person recognition using YOLOv8 and OpenCV tracking
- `yolo.py`: Pure YOLO object detection for all 80 COCO classes with movement tracking
- `yolo_tflite.py`: YOLO detection using TensorFlow Lite model (CPU-optimized, legacy)

## Hardware

- **Platform**: Raspberry Pi 5
- **Camera**: Pi Camera Module 3 (via picamera2)
- **AI Accelerator**: Hailo AI Hat (Hailo-8L, 26 TOPS) — Hailo integration planned

## Camera Resolution

All scripts use **800x600** for Pi camera (performance/quality balance on Pi 5).

## Dependencies

```bash
# System packages
sudo apt install python3-picamera2 libcamera-apps

# Python packages
pip install opencv-python numpy ultralytics
```

## Model Files

Located in `models/` (excluded from git — download via ultralytics or manually):

- `models/yolov8n.pt`: Nano model — used by `yolo.py` and `triple.py` (default)
- `models/yolov8l.pt`: Large model — available but slower
- `models/yolov8n_int8.tflite`: TFLite quantized model — used by `yolo_tflite.py`
- `models/convert_to_tflite.py`: Conversion script for generating .tflite from .pt

Models are auto-downloaded by ultralytics on first run if not present locally.

## Running the Scripts

### Motion Detection (motion.py)
```bash
python motion.py [--preview] [--camera INDEX]
```

### Person Detection and Tracking (triple.py)
```bash
python triple.py [--preview] [--camera INDEX] [--force-pi]
```

### YOLO Object Detection (yolo.py)
```bash
python yolo.py [--preview] [--camera INDEX] [--confidence FLOAT]
```

### TFLite Detection (yolo_tflite.py) — legacy, CPU only
```bash
python yolo_tflite.py [--preview] [--camera INDEX]
```

## Architecture

### Platform Detection
Scripts automatically detect Raspberry Pi by checking `/proc/device-tree/model`, using:
- `picamera2` for Pi camera
- `cv2.VideoCapture` for standard webcams

### Motion Detection Flow
1. Capture frame from camera
2. Convert to grayscale and apply Gaussian blur
3. Calculate frame difference from previous frame
4. Apply threshold and dilation to detect motion contours
5. For significant motion areas, run YOLOv8 detection
6. If target detected, start OpenCV tracker
7. Save images when motion/detection occurs

### Tracking System (triple.py)
The `CVTracker` class implements:
- Multiple tracker types (CSRT, KCF, MOSSE) with fallback support
- Movement vector calculation and visualization
- Automatic tracker stopping after 3 seconds of minimal movement

## Output Directories (git-ignored)

- `motion_captures/`: Images saved by motion.py
- `triple_captures/`: Images saved by triple.py
- `yolo_captures/`: Images saved by yolo.py

## Configuration

### Motion Sensitivity
In `triple.py`, adjust `MOTION_SENSITIVITY` (default: 500):
- Lower = more sensitive, Higher = less sensitive

### Detection Thresholds
- Person detection confidence: 0.30 (triple.py)
- YOLO confidence: 0.25 (yolo.py, configurable via --confidence)
- Motion threshold: 25 (motion.py, triple.py)
- Minimum movement for tracking: 10 pixels

## Shared Conventions
- Display FPS in green text at (10, 30)
- Save images with timestamp: `YYYYMMDD_HHMMSS`
- Save at most one image per second
- Clear all capture files before starting each run
- Use .gitignore for captures/ and models/

## Roadmap
- [ ] Hailo AI Hat integration (`yolo_hailo.py`) — replace CPU YOLO with Hailo NPU (~80+ FPS)
