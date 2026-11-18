"""
warning.py – Flight monitor for DJI Tello
==========================================
Runs in a background thread. Every CHECK_INTERVAL seconds it reads
battery, temperature, and altitude.

Warning level  →  Tk dialog asks user: retrace path and land, or continue?
Critical level →  Immediate emergency action, no dialog (too dangerous to wait)

Path retracing logic:
  The controls thread appends [lr, fb, ud, yv, duration] to path_log
  for every non-zero RC segment. To return home, each segment is
  replayed in reverse order with all axes negated.
"""

import time
import threading
import tkinter as tk
from tkinter import messagebox

# ── Thresholds ────────────────────────────────────────────────────────────────

BATTERY_WARNING  = 15     # % – show dialog
BATTERY_CRITICAL = 5     # % – emergency land immediately

TEMP_WARNING     = 85   # °C – show dialog
TEMP_CRITICAL    = 95   # °C – cut motors immediately

ALT_WARNING      = 2500   # cm (25 m) – show dialog
ALT_CRITICAL     = 2800   # cm (28 m) – force land immediately

CHECK_INTERVAL   = 3      # seconds between telemetry polls


class FlightMonitor:
    """
    Parameters
    ----------
    drone      : connected djitellopy.Tello instance (shared with other threads)
    path_log   : list shared with controls thread;
                 each entry is [lr, fb, ud, yv, duration_seconds]
    stop_event : threading.Event – set by any thread to signal a full stop
    """

    def __init__(self, drone, path_log: list, stop_event: threading.Event):
        self.drone      = drone
        self.path_log   = path_log
        self.stop_event = stop_event
        self._alerted   = set()    # keys of warnings already shown (avoid spam)
        self.retracing  = False    # True while _retrace() is running;
                                   # controls thread checks this and pauses RC

    # ── Private helpers ───────────────────────────────────────────────────────

    def _dialog(self, title: str, message: str) -> bool:
        """
        Blocking yes/no Tk dialog. Always appears on top.
        Returns True if the user clicks Yes.
        """
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        answer = messagebox.askyesno(title, message, parent=root)
        root.destroy()
        return answer

    def _retrace(self):
        """
        Fly back along the recorded path:
          - replay segments in reverse order
          - negate every axis (lr, fb, ud, yv) so each move is mirrored
        Ends with a controlled land at the approximate launch point.
        Note: accuracy depends on flight conditions; wind will introduce drift.

        Sets self.retracing = True for the duration so the controls thread
        knows to stop sending its own RC commands (which would override ours).
        stop_event is NOT checked inside the loop - once retrace starts it
        runs to completion; critical conditions are handled by _cut_motors()
        which calls drone.emergency() and will override everything anyway.
        """
        if not self.path_log:
            print("[MONITOR] No path recorded - landing in place.")
            self.drone.land()
            return

        self.retracing = True
        print(f"[MONITOR] Retracing {len(self.path_log)} path segments...")

        for lr, fb, ud, yv, duration in reversed(self.path_log):
            self.drone.send_rc_control(-lr, -fb, -ud, -yv)
            time.sleep(duration)

        # Stop all motion, then land
        self.drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.3)
        self.drone.land()
        self.retracing = False
        print("[MONITOR] Retrace complete - drone landed.")

    def _cut_motors(self, reason: str):
        """
        Last resort: cuts all motors via emergency(). Drone will drop.
        Only used when continuing to fly poses immediate danger.
        """
        print(f"\n[CRITICAL] {reason}")
        print("[CRITICAL] Cutting motors via emergency(). Drone will drop.")
        self.drone.emergency()
        self.stop_event.set()

    def _controlled_land(self, reason: str):
        """
        Safer fallback when retracing is not possible but the situation is
        not critical enough to cut motors.
        """
        print(f"[MONITOR] {reason} – performing controlled landing in place.")
        self.drone.land()
        self.stop_event.set()

    def _handle_warning(
        self,
        key: str,
        log_msg: str,
        dialog_title: str,
        dialog_msg: str,
        can_retrace: bool,
    ):
        """
        Show a warning once per key. Present a dialog and act on the user's choice.

        can_retrace – False when resources are too low to safely complete a retrace
                      (e.g. battery is at 12% – barely enough for an emergency land,
                      not enough to fly the full path back).
        """
        if key in self._alerted:
            return                 # already warned; don't open another dialog
        self._alerted.add(key)

        print(f"\n[WARNING] {log_msg}")

        confirmed = self._dialog(dialog_title, dialog_msg)

        if confirmed:
            self.stop_event.set()
            if can_retrace:
                self._retrace()
            else:
                self._controlled_land("Insufficient resources for retrace")
        else:
            print("[MONITOR] User chose to continue. Monitoring resumed.")
            print(f"[MONITOR] (Will escalate if condition reaches critical level)")

    # ── Public: main monitoring loop ──────────────────────────────────────────

    def monitor(self):
        """Call this in a daemon thread: threading.Thread(target=monitor.monitor)"""
        print("[MONITOR] Flight monitor active.")

        while not self.stop_event.is_set():
            try:
                battery  = self.drone.get_battery()    # int  0–100 %
                temp     = self.drone.get_temperature() # int  °C
                altitude = self.drone.get_height()      # int  cm

                print(
                    f"[TELEMETRY] Battery: {battery}%  "
                    f"Temp: {temp}°C  "
                    f"Altitude: {altitude} cm ({altitude / 100:.1f} m)"
                )

                # ── Critical: act immediately, no dialog ─────────────────────

                if battery <= BATTERY_CRITICAL:
                    self._cut_motors(
                        f"Battery critically low ({battery}% ≤ {BATTERY_CRITICAL}%)"
                    )
                    break

                if temp >= TEMP_CRITICAL:
                    self._cut_motors(
                        f"Temperature critically high ({temp}°C ≥ {TEMP_CRITICAL}°C)"
                    )
                    break

                if altitude >= ALT_CRITICAL:
                    self._cut_motors(
                        f"Altitude critically high ({altitude} cm ≥ {ALT_CRITICAL} cm)"
                    )
                    break

                # ── Warning: ask user ────────────────────────────────────────

                if battery <= BATTERY_WARNING:
                    # Retrace is safe only if we have enough battery above critical
                    can_retrace = battery > (BATTERY_CRITICAL + 5)
                    self._handle_warning(
                        key          = "battery_low",
                        log_msg      = (
                            f"Battery at {battery}% "
                            f"(warning threshold: {BATTERY_WARNING}%)"
                        ),
                        dialog_title = "⚠  Low Battery",
                        dialog_msg   = (
                            f"Battery is at {battery}%.\n\n"
                            f"Yes  →  Retrace path and land\n"
                            f"No   →  Continue flying\n\n"
                            f"(At {BATTERY_CRITICAL}% motors will be cut automatically)"
                        ),
                        can_retrace  = can_retrace,
                    )

                if temp >= TEMP_WARNING:
                    can_retrace = temp < (TEMP_CRITICAL - 5)
                    self._handle_warning(
                        key          = "temp_high",
                        log_msg      = (
                            f"Temperature at {temp}°C "
                            f"(warning threshold: {TEMP_WARNING}°C)"
                        ),
                        dialog_title = "⚠  High Temperature",
                        dialog_msg   = (
                            f"Drone temperature is {temp}°C.\n\n"
                            f"Yes  →  Retrace path and land\n"
                            f"No   →  Continue flying\n\n"
                            f"(At {TEMP_CRITICAL}°C motors will be cut automatically)"
                        ),
                        can_retrace  = can_retrace,
                    )

                if altitude >= ALT_WARNING:
                    self._handle_warning(
                        key          = "alt_high",
                        log_msg      = (
                            f"Altitude at {altitude} cm "
                            f"({altitude / 100:.1f} m, "
                            f"warning threshold: {ALT_WARNING / 100:.0f} m)"
                        ),
                        dialog_title = "⚠  High Altitude",
                        dialog_msg   = (
                            f"Altitude is {altitude} cm ({altitude / 100:.1f} m).\n\n"
                            f"Yes  →  Retrace path and land\n"
                            f"No   →  Continue flying\n\n"
                            f"(At {ALT_CRITICAL} cm motors will be cut automatically)"
                        ),
                        can_retrace  = True,   # altitude issue doesn't drain resources
                    )

            except Exception as exc:
                print(f"[MONITOR] Telemetry read error: {exc}")

            time.sleep(CHECK_INTERVAL)

        print("[MONITOR] Monitor stopped.")