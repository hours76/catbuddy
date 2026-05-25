"""
driver.py - catbuddy hardware driver
Motor, servo, and ultrasonic control for the cat-chasing robot.

Servo calibration (rasp5.local):
  Servo 0 (pan,  left/right): center=75,  right=smaller, left=larger
  Servo 1 (tilt, up/down):    center=110, down=larger,   up=smaller
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'opt/Code/Pi5/Server-pi5'))

from Motor import Motor
from servo import Servo
from Ultrasonic import Ultrasonic
from Buzzer import Buzzer as _Buzzer

# ── constants ────────────────────────────────────────────────────────────────
MOTOR_SPEED       = 2000          # default drive speed (-4095 ~ 4095)
OBSTACLE_DIST_CM  = 20            # stop if obstacle closer than this

SERVO_PAN_CENTER  = 75            # servo 0 pan  center
SERVO_TILT_CENTER = 110           # servo 1 tilt center
SERVO_PAN_MIN     = 40            # rightmost limit
SERVO_PAN_MAX     = 120           # leftmost  limit
SERVO_TILT_MIN    = 70            # uppermost limit
SERVO_TILT_MAX    = 150           # lowest    limit
SERVO_STEP        = 5             # degrees per nudge call

# ── init ─────────────────────────────────────────────────────────────────────
_motor      = Motor()
_servo      = Servo()
_ultrasonic = Ultrasonic()
_buzzer     = _Buzzer()

# current servo positions
_pan  = SERVO_PAN_CENTER
_tilt = SERVO_TILT_CENTER

def _apply_servo():
    _servo.setServoPwm('0', _pan)
    _servo.setServoPwm('1', _tilt)


# ── buzzer ───────────────────────────────────────────────────────────────────
def beep(duration=0.3):
    """Beep the buzzer for the given duration in seconds."""
    _buzzer.run('1')
    import time; time.sleep(duration)
    _buzzer.run('0')



def get_distance():
    """Return distance in cm from ultrasonic sensor."""
    return _ultrasonic.get_distance()

def is_obstacle():
    """Return True if obstacle is closer than OBSTACLE_DIST_CM."""
    return get_distance() < OBSTACLE_DIST_CM


# ── motor ────────────────────────────────────────────────────────────────────
def stop():
    """Cut motor power immediately."""
    _motor.setMotorModel(0, 0, 0, 0)

def forward(speed=MOTOR_SPEED):
    """Move forward. Stops automatically if obstacle detected."""
    if is_obstacle():
        print(f"[driver] Obstacle detected ({get_distance():.0f} cm) — not moving forward")
        stop()
        return
    _motor.setMotorModel(speed, speed, speed, speed)

def backward(speed=MOTOR_SPEED):
    """Move backward."""
    _motor.setMotorModel(-speed, -speed, -speed, -speed)

def turn_left(speed=MOTOR_SPEED):
    """Turn left in place."""
    if is_obstacle():
        print(f"[driver] Obstacle detected ({get_distance():.0f} cm) — not moving forward")
        stop()
        return
    _motor.setMotorModel(-500, -500, speed, speed)

def turn_right(speed=MOTOR_SPEED):
    """Turn right in place."""
    if is_obstacle():
        print(f"[driver] Obstacle detected ({get_distance():.0f} cm) — not moving forward")
        stop()
        return
    _motor.setMotorModel(speed, speed, -500, -500)


# ── camera servo ─────────────────────────────────────────────────────────────
def camera_center():
    """Reset camera to center position."""
    global _pan, _tilt
    _pan  = SERVO_PAN_CENTER
    _tilt = SERVO_TILT_CENTER
    _apply_servo()

def camera_left(step=SERVO_STEP):
    """Nudge camera left."""
    global _pan
    _pan = min(_pan + step, SERVO_PAN_MAX)
    _apply_servo()

def camera_right(step=SERVO_STEP):
    """Nudge camera right."""
    global _pan
    _pan = max(_pan - step, SERVO_PAN_MIN)
    _apply_servo()

def camera_up(step=SERVO_STEP):
    """Nudge camera up."""
    global _tilt
    _tilt = max(_tilt - step, SERVO_TILT_MIN)
    _apply_servo()

def camera_down(step=SERVO_STEP):
    """Nudge camera down."""
    global _tilt
    _tilt = min(_tilt + step, SERVO_TILT_MAX)
    _apply_servo()

def camera_set(pan, tilt):
    """Set camera to absolute pan/tilt values (clamped to limits)."""
    global _pan, _tilt
    _pan  = max(SERVO_PAN_MIN,  min(SERVO_PAN_MAX,  pan))
    _tilt = max(SERVO_TILT_MIN, min(SERVO_TILT_MAX, tilt))
    _apply_servo()


# ── self-test ─────────────────────────────────────────────────────────────────
def self_test():
    import time

    print("=== driver self-test ===")

    print("Beep!"); beep()

    print(f"Distance: {get_distance():.0f} cm  obstacle={is_obstacle()}")

    print("Camera center"); camera_center(); time.sleep(0.5)
    print("Camera left");   camera_left(15); time.sleep(0.5)
    print("Camera center"); camera_center(); time.sleep(0.5)
    print("Camera right");  camera_right(15); time.sleep(0.5)
    print("Camera center"); camera_center(); time.sleep(0.5)
    print("Camera up");     camera_up(15);   time.sleep(0.5)
    print("Camera center"); camera_center(); time.sleep(0.5)
    print("Camera down");   camera_down(15); time.sleep(0.5)
    print("Camera center"); camera_center(); time.sleep(0.5)

    print("Forward");  forward();   time.sleep(0.5); stop(); time.sleep(0.2)
    print("Backward"); backward();  time.sleep(0.5); stop(); time.sleep(0.2)
    print("Left");     turn_left(); time.sleep(0.5); stop(); time.sleep(0.2)
    print("Right");    turn_right(); time.sleep(0.5); stop()

    print("=== done ===")


if __name__ == '__main__':
    self_test()
