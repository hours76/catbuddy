"""
pilot.py - catbuddy pilot
Reads tracking state from yolo_hailo and decides how to move the robot.

Tracking state (shared dict updated by yolo_hailo):
  {
    "detected": bool,      # target visible in frame
    "cx": int,             # target center x in pixels
    "cy": int,             # target center y in pixels
    "dx": int,             # cx - frame_center_x  (negative=left, positive=right)
    "dy": int,             # cy - frame_center_y  (negative=up, positive=down)
  }

Pilot decisions (wheels only, fixed speed):
  - not detected          → stop
  - obstacle < 20 cm      → stop
  - |dx| <= DEAD_ZONE     → forward (target roughly centered)
  - dx < -DEAD_ZONE       → turn left
  - dx >  DEAD_ZONE       → turn right
"""

import time
import threading
import driver

# ── tuning ────────────────────────────────────────────────────────────────────
DEAD_ZONE        = 40    # pixels — dx within this range → full forward, no steering
SPIN_ZONE        = 200   # pixels — dx beyond this → full spin (no forward component)
BASE_SPEED       = 1000  # forward speed
STEER_K          = 4.0   # dx-to-speed scaling factor
LOOP_HZ          = 10    # pilot update rate
CONFIRM_SECS     = 0.3   # seconds of continuous detection before first move
GRACE_SECS       = 0.3   # seconds to keep moving after losing detection

# ── shared state (written by yolo_hailo, read by pilot) ──────────────────────
tracking_state = {
    "detected": False,
    "cx": 0,
    "cy": 0,
    "dx": 0,
    "dy": 0,
}
_state_lock = threading.Lock()
_first_detected_time = None   # when continuous detection started
_last_detected_time  = None   # when detection was last seen

def update_state(detected, cx=0, cy=0, dx=0, dy=0):
    """Called by yolo_hailo each frame to update tracking state."""
    global _first_detected_time, _last_detected_time
    with _state_lock:
        now = time.time()
        if detected:
            if _first_detected_time is None:
                _first_detected_time = now
            _last_detected_time = now
            confirmed = (now - _first_detected_time) >= CONFIRM_SECS
        else:
            _first_detected_time = None
            # grace period: keep confirmed=True for GRACE_SECS after last detection
            if _last_detected_time is not None and (now - _last_detected_time) < GRACE_SECS:
                confirmed = True
            else:
                confirmed = False

        tracking_state["detected"] = confirmed
        tracking_state["cx"] = cx
        tracking_state["cy"] = cy
        tracking_state["dx"] = dx
        tracking_state["dy"] = dy

def get_state():
    with _state_lock:
        return dict(tracking_state)


# ── pilot loop ────────────────────────────────────────────────────────────────
_running = False

def _calc_motors(dx):
    """
    Convert dx offset to (left_speed, right_speed).
    dx < 0 = target is left, dx > 0 = target is right.

    Zones:
      |dx| <= DEAD_ZONE  → full forward
      DEAD_ZONE < |dx| < SPIN_ZONE → differential (blend forward + turn)
      |dx| >= SPIN_ZONE  → full spin in place
    """
    adx = abs(dx)

    if adx <= DEAD_ZONE:
        # Straight ahead
        return BASE_SPEED, BASE_SPEED

    if adx >= SPIN_ZONE:
        # Full spin
        spin = BASE_SPEED
        if dx < 0:  # target left → spin left
            return -spin, spin
        else:       # target right → spin right
            return spin, -spin

    # Blend zone: forward speed tapers as dx grows
    t = (adx - DEAD_ZONE) / (SPIN_ZONE - DEAD_ZONE)  # 0.0 → 1.0
    fwd = int(BASE_SPEED * (1 - t))
    steer = int(BASE_SPEED * t)

    if dx < 0:  # target left
        left  = fwd - steer
        right = fwd + steer
    else:       # target right
        left  = fwd + steer
        right = fwd - steer

    return left, right

def _loop():
    while _running:
        state = get_state()

        if not state["detected"] or driver.is_obstacle():
            driver.stop()
        else:
            dx = state["dx"]
            left, right = _calc_motors(dx)
            # setMotorModel(FL, BL, FR, BR)
            driver._motor.setMotorModel(left, left, right, right)

        time.sleep(1 / LOOP_HZ)

    driver.stop()


def start():
    """Start the pilot loop in a background thread."""
    global _running
    _running = True
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t

def stop_pilot():
    """Stop the pilot loop."""
    global _running
    _running = False
    driver.stop()


