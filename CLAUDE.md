# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computer vision project for motion detection and car tracking using OpenCV and YOLOv8. The project consists of two main Python scripts:

- `motion.py`: Basic motion detection that captures frames when motion is detected
- `triple.py`: Advanced motion detection with car recognition using YOLOv8 and object tracking

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

The project uses YOLOv8 model files:
- `yolov8n.pt`: Nano model (used by default in triple.py)
- `yolov8l.pt`: Large model (available but not used by default)

## Running the Scripts

### Motion Detection (motion.py)
```bash
python motion.py [--preview] [--camera INDEX]
```
- `--preview`: Shows live OpenCV window
- `--camera INDEX`: Specify camera index (default: 0)

### Car Detection and Tracking (triple.py)
```bash
python triple.py [--preview] [--camera INDEX] [--force-pi]
```
- `--preview`: Shows live OpenCV window
- `--camera INDEX`: Specify camera index (default: 0)
- `--force-pi`: Force Raspberry Pi camera mode

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
- `triple_captures/`: Images saved by triple.py (includes car detection and tracking)

## Configuration

### Motion Sensitivity
In `triple.py`, adjust `MOTION_SENSITIVITY` (default: 500) to control motion detection sensitivity:
- Lower values = more sensitive
- Higher values = less sensitive

### Detection Thresholds
- Car detection confidence threshold: 0.30 (in triple.py:188)
- Motion detection threshold: 25 (in both scripts)
- Minimum movement for tracking: 10 pixels (in triple.py:130)