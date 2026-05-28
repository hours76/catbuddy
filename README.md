# catbuddy

AI robot that chases cats — Raspberry Pi 5 + Hailo AI Hat + Pi Camera Module 3.

## Hardware

- **Platform**: Raspberry Pi 5
- **Camera**: Pi Camera Module 3 (via picamera2), mounted upside-down → rotated 180° in software
- **AI Accelerator**: Hailo AI Hat (Hailo-8L, 26 TOPS)
- **Motors**: 4-wheel drive via PCA9685 PWM controller (I2C 0x40)
- **Camera servo**: Pan (servo 0) + Tilt (servo 1)
- **Obstacle sensor**: Ultrasonic (stops if < 20 cm)
- **Buzzer**: GPIO

## Scripts

### Main: `yolo_hailo.py` — Hailo NPU detection + robot control

The primary script. Runs YOLOv8s on Hailo-8L (~80+ FPS) and drives the robot to chase the detected target.

```bash
source ~/work/hailo-rpi5-examples/setup_env.sh
python yolo_hailo.py [--preview] [--confidence 0.40]
```

**Detection**: target label is `person` (configurable via `TARGET_LABEL`). All detections shown; target highlighted green.

**Save condition**: target detected AND moving (>10px over 1 second) → saves one frame per second to `hailo_captures/<session_timestamp>/`.

**Robot control**: delegates to `pilot.py` + `driver.py` (see below).

---

### `pilot.py` — Steering logic

Reads tracking state from `yolo_hailo.py` and drives the wheels.

| Condition | Action |
|---|---|
| Not detected | Stop |
| Obstacle < 20 cm | Stop |
| `\|dx\|` ≤ 40 px (dead zone) | Forward |
| 40 < `\|dx\|` < 200 px | Blend: forward + differential steer |
| `\|dx\|` ≥ 200 px (spin zone) | Spin in place |

Confirmation delay: 0.3s of continuous detection before moving. Grace period: 0.3s after losing detection before stopping.

---

### `driver.py` — Hardware abstraction

Wraps `Motor`, `Servo`, `Ultrasonic`, `Buzzer` from `opt/Code/Pi5/Server-pi5/`.

```bash
python driver.py        # runs self_test(): beep → servo sweep → motor test
```

Servo calibration (rasp5.local):
- Servo 0 (pan): center=75, right=↓, left=↑, range 40–120
- Servo 1 (tilt): center=110, down=↑, up=↓, range 70–150

---

### Legacy scripts (CPU-based, no robot control)

| Script | Description |
|---|---|
| `motion.py` | Frame-diff motion detection only. Saves to `motion_captures/`. |
| `triple.py` | Motion detection → YOLOv8 CPU inference → OpenCV tracker (CSRT/MIL). Saves to `triple_captures/`. |
| `yolo.py` | Pure YOLOv8 CPU detection on every frame, all 80 COCO classes. Saves to `yolo_captures/`. |
| `yolo_tflite.py` | YOLOv8 TFLite quantized model (CPU, legacy). |

All legacy scripts support `--preview`, `--camera INDEX`, `--force-pi`.

## Dependencies

```bash
# System
sudo apt install python3-picamera2 libcamera-apps hailo-all

# Python
pip install opencv-python numpy ultralytics
```

HEF model: `/usr/local/hailo/resources/models/hailo8l/yolov8s.hef`

## Deploy

From macstudio:

```bash
./deploy.sh
```

Syncs all source files to `hrsung@rasp5.local:~/work/catbuddy/`. Excludes: `models/`, `*_captures/`, `__pycache__/`, `.git/`, `deploy.sh`, `CLAUDE.md`.
