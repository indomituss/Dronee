"""
drone.py – Master entry point
==============================
Workflow:
  1. Enter installation details (panel wattage, tariff, sun hours)
  2. Take off → fly to panels → press [I] for each panel (2 frames auto)
  3. Land with [1]
  4. Switch to home WiFi
  5. Press [E] to run analysis → report with financial impact → program exits

Key bindings:
  [0]     Takeoff
  [1]     Land
  [W/S]   Forward/Back
  [A/D]   Left/Right
  [↑/↓]   Up/Down
  [←/→]   Yaw
  [I]     Toggle panel capture (auto-stops after 2 frames)
  [E]     Analyse + generate report (after WiFi switch)
  [SPACE] Emergency land + quit
"""

import threading
import time
import cv2
import os
from datetime import datetime

from djitellopy import tello
import flyingModules.KeyPressModule as kp
from flyingModules.warning import FlightMonitor

# ── Drone ─────────────────────────────────────────────────────────────────────
drone = tello.Tello()
drone.connect()

# ── Shared state ──────────────────────────────────────────────────────────────
path_log           = []
stop_event         = threading.Event()
inspection_mode    = threading.Event()
analysis_triggered = threading.Event()

# ── Session ───────────────────────────────────────────────────────────────────
session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
session_dir = f"inspections/{session_id}"
os.makedirs(session_dir, exist_ok=True)
print(f"[MAIN] Session folder: {session_dir}")

SPEED        = 50
MAX_CAPTURES = 2

# ── Counters ──────────────────────────────────────────────────────────────────
capture_count       = 0
panel_number        = 0
captures_this_panel = 0

# ── Debounce flags ────────────────────────────────────────────────────────────
_takeoff_done    = False
_land_done       = False
_analysis_queued = False

# ── Installation config (filled by get_installation_config) ──────────────────
installation_config = {}


# ── Installation config ───────────────────────────────────────────────────────

def get_installation_config() -> dict:
    """
    Asks operator for basic installation details once before flight.
    Used to calculate financial impact in the report.
    Press Enter at any prompt to accept the default value shown in [ ].
    """
    border = "═" * 52
    print(f"\n{border}")
    print("   INSTALLATION DETAILS  (for financial report)")
    print(border)
    print("  Press Enter to use default values shown in [ ]")
    print(border)

    def ask(prompt, default, cast=float):
        val = input(f"  {prompt} [{default}]: ").strip()
        return cast(val) if val else cast(default)

    config = {}
    config['panel_wattage']  = ask("Panel wattage (W)",         400)
    config['tariff_per_kwh'] = ask("Electricity rate (₹/kWh)",  7.5)
    config['peak_sun_hours'] = ask("Peak sun hours/day",         5.5)
    config['days_per_year']  = ask("Operating days/year",        365, int)

    # Annual revenue per panel at 100% efficiency
    kwh_per_panel_per_year = (
        config['panel_wattage'] / 1000
        * config['peak_sun_hours']
        * config['days_per_year']
    )
    config['annual_revenue_per_panel'] = (
        kwh_per_panel_per_year * config['tariff_per_kwh']
    )

    print(f"\n{border}")
    print("  CONFIRMED:")
    print(f"  Panel wattage    : {config['panel_wattage']}W")
    print(f"  Tariff           : Rs. {config['tariff_per_kwh']}/kWh")
    print(f"  Peak sun hours   : {config['peak_sun_hours']} hrs/day")
    print(f"  Operating days   : {config['days_per_year']} days/year")
    print(
        f"  Revenue/panel    : "
        f"₹{config['annual_revenue_per_panel']:,.0f}/year (at 100%)"
    )
    print(border)
    input("\n  Press Enter to continue to flight… ")

    return config


# ── Pre-flight info ───────────────────────────────────────────────────────────

def show_preflight_info():
    battery = drone.get_battery()
    temp    = drone.get_temperature()

    border = "═" * 52
    print(f"\n{border}")
    print("         ✈   PRE-FLIGHT INFORMATION   ✈")
    print(border)
    print(f"  Battery     : {battery}%")
    print(f"  Temperature : {temp}°C")
    print(border)
    print("  [0]     Takeoff       [1]     Land")
    print("  [W/S]   Forward/Back  [A/D]   Left/Right")
    print("  [↑/↓]   Up/Down       [←/→]   Yaw")
    print("  [I]     Capture panel (auto stops at 2 frames)")
    print("  [E]     Analyse + generate report (after WiFi switch)")
    print("  [SPACE] Emergency land + quit")
    print(border)

    if battery < 30:
        print(f"\n  ⚠  Battery at {battery}% — charge before flying.")
    if temp > 70:
        print(f"  ⚠  Temperature {temp}°C — monitor closely.")

    input("\n  Press Enter to start… ")


# ── Post-flight analysis ──────────────────────────────────────────────────────

def run_post_flight_analysis():
    """
    Loads saved frames, calls Roboflow API, generates report.
    Triggered by [E] keypress after switching to home WiFi.
    Runs in its own thread so it doesn't block the keyboard loop.
    """
    from inspectionModules.detector import PanelDetector
    from inspectionModules.panel_map import PanelMap
    from inspectionModules.reporter import InspectionReporter

    print("\n[ANALYSIS] ── Starting analysis ──")
    print(f"[ANALYSIS] Session: {session_dir}")

    frames_found = sorted([
        f for f in os.listdir(session_dir)
        if f.endswith('.jpg') and '_detected' not in f
    ])

    if not frames_found:
        print("[ANALYSIS] No frames found. Did you capture any panels?")
        stop_event.set()
        return

    print(f"[ANALYSIS] {len(frames_found)} frames to process")

    try:
        detector  = PanelDetector()
        panel_map = PanelMap()
        reporter  = InspectionReporter(session_dir)
    except Exception as e:
        print(f"[ANALYSIS] Failed to initialise detector: {e}")
        print("[ANALYSIS] Make sure you are connected to home WiFi")
        stop_event.set()
        return

    for filename in frames_found:
        filepath = os.path.join(session_dir, filename)
        frame    = cv2.imread(filepath)

        if frame is None:
            print(f"[ANALYSIS] Could not read: {filename}")
            continue

        # Extract panel number from filename: panel_001_cap_1.jpg
        try:
            panel_id = int(filename.split('_')[1])
        except (IndexError, ValueError):
            panel_id = panel_map.next_panel_id()

        print(f"[ANALYSIS] Processing {filename}...")

        result = detector.detect(frame)

        # Save annotated copy
        annotated_path = filepath.replace('.jpg', '_detected.jpg')
        cv2.imwrite(annotated_path, result['annotated_frame'])
        result['image_path'] = annotated_path

        panel_map.record(panel_id, result)

        status = (
            f"  → Panel {panel_id}: {result['worst_class']} "
            f"(severity {result['worst_severity']})"
        )
        if result.get('error'):
            status += f" ⚠ API error: {result['error']}"
        print(status)

    # Pass installation config so reporter can calculate financial impact
    reporter.save_csv(panel_map, installation_config)
    reporter.save_summary(panel_map, installation_config)

    print("\n[ANALYSIS] ── Complete ──")
    print(f"[ANALYSIS] Results in: {session_dir}")

    stop_event.set()


# ── Thread 1: Controls ────────────────────────────────────────────────────────

def controls_thread_fn():
    global _takeoff_done, _land_done, _analysis_queued
    global panel_number, captures_this_panel

    kp.init()
    prev_vals = [0, 0, 0, 0]
    prev_time = time.time()

    while not stop_event.is_set():
        try:
            lr, fb, ud, yv = 0, 0, 0, 0

            if   kp.get_key("LEFT"):  yv = -SPEED
            elif kp.get_key("RIGHT"): yv =  SPEED
            if   kp.get_key("UP"):    ud =  SPEED
            elif kp.get_key("DOWN"):  ud = -SPEED
            if   kp.get_key("w"):     fb =  SPEED
            elif kp.get_key("s"):     fb = -SPEED
            if   kp.get_key("a"):     lr = -SPEED
            elif kp.get_key("d"):     lr =  SPEED

            # Takeoff
            if kp.get_key("0") and not _takeoff_done:
                try:
                    drone.takeoff()
                    _takeoff_done = True
                    _land_done    = False
                    time.sleep(0.5)
                except Exception as e:
                    print(f"[CONTROLS] Takeoff failed: {e}")

            # Land
            if kp.get_key("1") and not _land_done:
                try:
                    drone.land()
                    _land_done    = True
                    _takeoff_done = False
                    time.sleep(0.5)
                    print("\n[CONTROLS] Drone landed.")
                    print(
                        "[CONTROLS] Switch to home WiFi "
                        "then press [E] to analyse."
                    )
                except Exception as e:
                    print(f"[CONTROLS] Land failed: {e}")

            # [I] — toggle panel capture
            if kp.get_key("i"):
                if inspection_mode.is_set():
                    inspection_mode.clear()
                    print(
                        f"[CONTROLS] Capture stopped — "
                        f"panel {panel_number} "
                        f"({captures_this_panel} frames)"
                    )
                    captures_this_panel = 0
                else:
                    panel_number       += 1
                    captures_this_panel = 0
                    inspection_mode.set()
                    print(
                        f"[CONTROLS] Capturing panel {panel_number}..."
                    )
                time.sleep(0.3)  # debounce

            # [E] — trigger analysis after landing and WiFi switch
            if kp.get_key("e") and not _analysis_queued:
                _analysis_queued = True
                print("\n[CONTROLS] [E] pressed — starting analysis...")
                t_analysis = threading.Thread(
                    target=run_post_flight_analysis,
                    daemon=True,
                    name="analysis"
                )
                t_analysis.start()
                time.sleep(0.5)  # debounce

            # Emergency quit
            if kp.get_key("SPACE"):
                print("\n[CONTROLS] SPACE — emergency land.")
                try:
                    drone.land()
                except Exception as e:
                    print(f"[CONTROLS] Emergency land failed: {e}")
                stop_event.set()
                break

            vals = [lr, fb, ud, yv]
            now  = time.time()

            if any(prev_vals):
                path_log.append([*prev_vals, now - prev_time])

            prev_vals = vals
            prev_time = now

            drone.send_rc_control(vals[0], vals[1], vals[2], vals[3])

        except Exception as e:
            print(f"[CONTROLS] Error (continuing): {e}")

        time.sleep(0.05)

    print("[CONTROLS] Stopped.")


# ── Thread 2: Camera ──────────────────────────────────────────────────────────

def camera_thread_fn():
    global capture_count, captures_this_panel, panel_number

    drone.streamon()
    frame_reader = drone.get_frame_read()
    frame_count  = 0

    while not stop_event.is_set():
        _ = frame_reader.frame
        frame = frame_reader.frame

        if frame is None:
            time.sleep(0.033)
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


        frame_count += 1

        # Live preview
        preview = cv2.resize(frame, (360, 240))

        if inspection_mode.is_set():
            cv2.circle(preview, (20, 20), 10, (0, 0, 255), -1)
            cv2.putText(
                preview,
                f"PANEL {panel_number} — {captures_this_panel}/{MAX_CAPTURES}",
                (35, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1
            )
        else:
            colour = (0, 165, 255) if _analysis_queued else (0, 255, 0)
            label  = (
                "ANALYSING..."
                if _analysis_queued
                else f"Panels: {panel_number}  Frames: {capture_count}"
            )
            cv2.circle(preview, (20, 20), 10, colour, -1)
            cv2.putText(
                preview, label,
                (35, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, colour, 1
            )

        cv2.imshow("Tello – Live Feed", preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            try:
                drone.land()
            except Exception:
                pass
            stop_event.set()
            break

        # Capture — only during inspection mode, max 2 frames per panel
        if (inspection_mode.is_set()
                and captures_this_panel < MAX_CAPTURES
                and frame_count % 15 == 0):

            capture_count       += 1
            captures_this_panel += 1

            filename = (
                f"{session_dir}/"
                f"panel_{panel_number:03d}_"
                f"cap_{captures_this_panel}.jpg"
            )
            cv2.imwrite(filename, frame)
            print(
                f"[CAMERA] panel {panel_number} "
                f"cap {captures_this_panel}/{MAX_CAPTURES} saved"
            )

            if captures_this_panel >= MAX_CAPTURES:
                inspection_mode.clear()
                print(
                    f"[CAMERA] Panel {panel_number} done — "
                    f"fly to next, press [I]"
                )

    cv2.destroyAllWindows()
    print(f"[CAMERA] Stopped. {capture_count} frames saved.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    show_preflight_info()

    # Get installation details before flight — used in financial report
    installation_config = get_installation_config()

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

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl-C — landing.")
        try:
            drone.land()
        except Exception:
            pass
        stop_event.set()

    t_controls.join(timeout=5)
    t_camera.join(timeout=5)
    t_monitor.join(timeout=5)

    print("[MAIN] Session ended.")