"""
controls.py – Standalone Tello keyboard controller
====================================================
Connects to the drone and lets you fly it from the keyboard.

NOTE: If you are using main.py, do NOT run this file separately –
main.py manages the drone connection and runs the same control
logic inside its own thread. Running both at once will create a
second conflicting UDP connection to the drone.

Key bindings
------------
  [0]       Takeoff          [1]        Land
  [W / S]   Forward / Back   [A / D]    Strafe left / right
  [↑ / ↓]   Ascend / Descend [← / →]   Yaw left / right
  [SPACE]   Exit loop (does NOT land – call drone.land() first if needed)
"""

# Standard library
import time

# Third-party
from djitellopy import tello
import KeyPressModule as kp

# ── Drone setup ───────────────────────────────────────────────────────────────

drone = tello.Tello()
drone.connect()        # Opens UDP connection to the Tello on 192.168.10.1

# ── Key press module ──────────────────────────────────────────────────────────

kp.init()              # Creates a small pygame window to capture key events


# ── Control function ──────────────────────────────────────────────────────────

def getKeyboardInput() -> list[int]:
    """
    Read the current keyboard state and map keys to Tello RC channels.

    Returns
    -------
    [lr, fb, ud, yv] : list of int
        Left/right, forward/back, up/down, yaw velocity.
        Each value is in the range [-speed, +speed] or 0.

    Side effects
    ------------
    Calls drone.land() or drone.takeoff() directly when [1] or [0] is pressed.
    """
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50    # cm/s – tune this to taste (Tello accepts 0–100)

    # Yaw (rotate) – arrow keys left/right
    if   kp.get_key("LEFT"):  yv = -speed
    elif kp.get_key("RIGHT"): yv =  speed

    # Altitude – arrow keys up/down
    if   kp.get_key("UP"):    ud =  speed
    elif kp.get_key("DOWN"):  ud = -speed

    # Forward/back – W / S
    if   kp.get_key("w"):     fb =  speed
    elif kp.get_key("s"):     fb = -speed

    # Strafe left/right – A / D
    if   kp.get_key("a"):     lr = -speed
    elif kp.get_key("d"):     lr =  speed

    # Takeoff / land shortcuts
    if kp.get_key("1"):       drone.land()
    if kp.get_key("0"):       drone.takeoff()

    return [lr, fb, ud, yv]


# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    vals = getKeyboardInput()

    # Send velocity commands on all four RC channels simultaneously.
    # Tello expects these at a minimum of ~20 Hz; we sleep 50 ms (~20 Hz).
    drone.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    time.sleep(0.05)

    # SPACE exits the loop; remember to land the drone before quitting
    if kp.get_key("SPACE"):
        break