"""
main.py – Tello Drone Control System
=====================================
Single entry point. Spin up three concurrent threads:

  controls  – keyboard → RC commands, logs path for retracing
  camera    – live OpenCV video feed from the drone
  monitor   – battery / temperature / altitude watchdog  (warning.py)

Usage
-----
    python main.py

Do NOT run controls.py or imageCapture.py separately anymore;
their logic lives inside the threaded functions below so the
drone object is created exactly once and shared safely.

Key bindings (active after takeoff)
-------------------------------------
  [0]       Takeoff          [1]        Land
  [W / S]   Forward / Back   [A / D]    Strafe left / right
  [↑ / ↓]   Ascend / Descend [← / →]   Yaw left / right
  [SPACE]   Quit (land + exit)
  [Q]       Quit camera window (same effect as SPACE)
"""

import threading
import time

import cv2
from djitellopy import tello

import KeyPressModule as kp
from warning import FlightMonitor

# ── Drone – created once, shared across all threads ──────────────────────────
drone = tello.Tello()
drone.connect()

# ── Shared state ──────────────────────────────────────────────────────────────
# path_log: controls thread appends [lr, fb, ud, yv, duration] for every
#           non-zero RC segment so FlightMonitor can retrace the path.
path_log   = []
stop_event = threading.Event()

SPEED = 50   # cm/s for all axes


# ── Pre-flight summary ────────────────────────────────────────────────────────

def show_preflight_info():
    """Print a flight brief and block until the user presses Enter."""
    battery  = drone.get_battery()
    temp     = drone.get_temperature()
    altitude = drone.get_height()

    border = "═" * 52
    thin   = "─" * 52

    print(f"\n{border}")
    print("         ✈   PRE-FLIGHT INFORMATION   ✈")
    print(border)
    print(f"  Battery           :  {battery} %")
    print(f"  Temperature       :  {temp} °C")
    print(f"  Current altitude  :  {altitude} cm")
    print(thin)
    print("  Controls")
    print("  ──────────────────────────────────────────")
    print("  [0]      Takeoff        [1]      Land")
    print("  [W / S]  Forward/Back   [A / D]  Left/Right")
    print("  [↑ / ↓]  Up/Down        [← / →]  Yaw")
    print("  [SPACE]  Quit (lands drone and exits)")
    print(border)

    # Pre-flight warnings (advisory only – user still decides)
    pre_warnings = []
    if battery < 30:
        pre_warnings.append(
            f"  ⚠  Battery is at {battery}% – consider charging before flying."
        )
    if temp > 70:
        pre_warnings.append(
            f"  ⚠  Temperature is {temp}°C – monitor closely."
        )

    if pre_warnings:
        print("\n  PRE-FLIGHT WARNINGS:")
        for w in pre_warnings:
            print(w)

    print()
    input("  Press Enter to start the flight session… ")
    print()


# ── Thread 1: Controls ────────────────────────────────────────────────────────

def _get_keyboard_input() -> list[int]:
    """
    Read key state and return [lr, fb, ud, yv].
    Also handles takeoff (0), land (1) directly.
    """
    lr, fb, ud, yv = 0, 0, 0, 0

    if   kp.get_key("LEFT"):  yv = -SPEED
    elif kp.get_key("RIGHT"): yv =  SPEED

    if   kp.get_key("UP"):    ud =  SPEED
    elif kp.get_key("DOWN"):  ud = -SPEED

    if   kp.get_key("w"):     fb =  SPEED
    elif kp.get_key("s"):     fb = -SPEED

    if   kp.get_key("a"):     lr = -SPEED
    elif kp.get_key("d"):     lr =  SPEED

    if kp.get_key("1"):
        drone.land()
    if kp.get_key("0"):
        drone.takeoff()

    return [lr, fb, ud, yv]


def controls_thread_fn():
    """
    Keyboard control loop. Runs at ~20 Hz.
    Appends non-zero RC segments to path_log for FlightMonitor's retrace.
    """
    kp.init()

    prev_vals = [0, 0, 0, 0]
    prev_time = time.time()

    while not stop_event.is_set():
        vals = _get_keyboard_input()
        now  = time.time()

        # Log the previous segment if it had any motion
        if any(prev_vals):
            path_log.append([*prev_vals, now - prev_time])

        prev_vals = vals
        prev_time = now

        drone.send_rc_control(vals[0], vals[1], vals[2], vals[3])

        if kp.get_key("0") and not already_taken_off:
            try:
                drone.takeoff()
                already_taken_off = True
                time.sleep(0.5)
            except Exception as e:
                print(f"[CONTROLS] Takeoff failed: {e}")
                print("[CONTROLS] Check battery, surface level, and connection")
                # Thread continues — doesn't crash

        if kp.get_key("1") and not already_landed:
            try:
                drone.land()
                already_landed = True
                already_taken_off = False
                time.sleep(0.5)
            except Exception as e:
                print(f"[CONTROLS] Land failed: {e}")

        if kp.get_key("SPACE"):
            print("\n[CONTROLS] SPACE pressed – landing and quitting.")
            drone.land()
            stop_event.set()
            break

        time.sleep(0.05)

    print("[CONTROLS] Controls thread stopped.")


# ── Thread 2: Camera ──────────────────────────────────────────────────────────

def camera_thread_fn():
    """
    Streams the drone's camera via OpenCV.
    Press [Q] in the window to quit (same effect as SPACE).
    """
    drone.streamon()    # streamon() is the method; stream_on is just a bool attribute

    while not stop_event.is_set():
        frame = drone.get_frame_read().frame
        frame = cv2.resize(frame, (360, 240))
        cv2.imshow("Tello – Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[CAMERA] Q pressed – quitting.")
            drone.land()
            stop_event.set()
            break

    cv2.destroyAllWindows()
    print("[CAMERA] Camera thread stopped.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    show_preflight_info()

    monitor = FlightMonitor(drone, path_log, stop_event)

    t_controls = threading.Thread(
        target=controls_thread_fn, daemon=True, name="controls"
    )
    t_camera = threading.Thread(
        target=camera_thread_fn, daemon=True, name="camera"
    )
    t_monitor = threading.Thread(
        target=monitor.monitor, daemon=True, name="monitor"
    )

    t_controls.start()
    t_camera.start()
    t_monitor.start()

    # Keep main thread alive; all real work is in the daemon threads above
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl-C caught – landing drone.")
        drone.land()
        stop_event.set()

    # Give threads a moment to clean up
    t_controls.join(timeout=3)
    t_camera.join(timeout=3)
    t_monitor.join(timeout=3)

    print("[MAIN] Session ended.") 