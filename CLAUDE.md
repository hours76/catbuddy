# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computer vision project for motion detection and object tracking using OpenCV and YOLOv8. The project consists of three main Python scripts:

- `motion.py`: Basic motion detection that captures frames when motion is detected
- `triple.py`: Advanced motion detection with person recognition using YOLOv8 and OpenCV tracking
- `yolo.py`: Pure YOLO object detection for all 80 COCO classes with movement tracking

## Dependencies

The project requires the following Python packages:
- `opencv-python` (cv2)
- `numpy`
- `ultralytics` (for YOLOv8)
- `picamera2` (for Raspberry Pi camera support)

Install dependencies with:
```bash
pip install opencv-python numpy ultralytics
# For Raspberry Pi:
pip install picamera2
```

## Model Files

The project uses YOLOv8 model files located in the `models/` directory:
- `models/yolov8n.pt`: Nano model (used by default in triple.py)
- `models/yolov8n_int8.tflite`: TensorFlow Lite quantized model (used by yolo_tflite.py)
- `models/yolov8l.pt`: Large model (available but not used by default)

## Running the Scripts

### Motion Detection (motion.py)
```bash
python motion.py [--preview] [--camera INDEX]
```
- `--preview`: Shows live OpenCV window
- `--camera INDEX`: Specify camera index (default: 0)

### Person Detection and Tracking (triple.py)
```bash
python triple.py [--preview] [--camera INDEX] [--force-pi]
```
- `--preview`: Shows live OpenCV window
- `--camera INDEX`: Specify camera index (default: 0)
- `--force-pi`: Force Raspberry Pi camera mode

### YOLO Object Detection (yolo.py)
```bash
python yolo.py [--preview] [--camera INDEX] [--confidence FLOAT]
```
- `--preview`: Shows live OpenCV window
- `--camera INDEX`: Specify camera index (default: 0)
- `--confidence FLOAT`: Detection confidence threshold (default: 0.25)

## Architecture

### Platform Detection
The `triple.py` script automatically detects if running on Raspberry Pi by checking `/proc/device-tree/model`. It uses:
- `picamera2` for Raspberry Pi camera
- `cv2.VideoCapture` for standard webcams/macOS

### Motion Detection Flow
1. Capture frame from camera
2. Convert to grayscale and apply Gaussian blur
3. Calculate frame difference from previous frame
4. Apply threshold and dilation to detect motion contours
5. For significant motion areas, run YOLOv8 detection
6. If car detected, start OpenCV tracker
7. Save images when motion/detection occurs

### Tracking System
The `CVTracker` class in `triple.py` implements:
- Multiple tracker types (CSRT, KCF, MOSSE) with fallback support
- Movement vector calculation and visualization
- Automatic tracker stopping after 3 seconds of minimal movement
- Robust tracker creation for different OpenCV versions

## Output Directories

- `motion_captures/`: Images saved by motion.py
- `person_captures/`: Images saved by triple.py (includes person detection and tracking)
- `yolo_captures/`: Images saved by yolo.py (all detected objects with timestamps)

## Configuration

### Motion Sensitivity
In `triple.py`, adjust `MOTION_SENSITIVITY` (default: 500) to control motion detection sensitivity:
- Lower values = more sensitive
- Higher values = less sensitive

### Detection Thresholds
- Person detection confidence threshold: 0.30 (in triple.py)
- YOLO detection confidence threshold: 0.25 (in yolo.py, configurable via --confidence)
- Motion detection threshold: 25 (in motion.py and triple.py)
- Minimum movement for tracking: 10 pixels (in triple.py and yolo.py)

## Performance Comparison

### yolo.py (Pure YOLO Detection)
- **Performance**: ~15 FPS
- **Approach**: Direct YOLO detection every frame
- **Best for**: Real-time detection of all object types, consistent performance
- **Features**: Movement arrows showing 1-second displacement, FPS display, color-coded object types

### triple.py (Motion + YOLO + OpenCV Tracking)
- **Performance**: ~8 FPS  
- **Approach**: Motion detection → YOLO → OpenCV tracking
- **Best for**: Focused person tracking with motion pre-filtering
- **Features**: Motion rectangles, OpenCV tracking with movement vectors, automatic cleanup

### motion.py (Basic Motion Detection)
- **Performance**: Highest FPS
- **Approach**: Frame differencing and contour detection only
- **Best for**: Simple motion detection without object classification

### Shared Conventions
- Display FPS in green text at (10, 30)
- Save images with timestamp format: `YYYYMMDD_HHMMSS`
- Save directories: `{type}_captures/`
- save picture as fast as one per second
- clear all files before starting
- Use .gitignore for captures directories and model files (models/)