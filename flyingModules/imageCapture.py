"""
imageCapture.py – Standalone Tello video stream viewer
=======================================================
Connects to the drone, enables the video stream, and displays
the live feed in an OpenCV window at a reduced resolution.

NOTE: If you are using main.py (flyingModules) or drone.py,
do NOT run this file separately – those files manage the drone
connection themselves. Running this alongside them will create
a conflicting UDP connection and crash.

Exit
----
  Close the OpenCV window or press Ctrl-C in the terminal.
"""

from djitellopy import tello
import cv2

# ── Drone setup ───────────────────────────────────────────────────────────────

drone = tello.Tello()
drone.connect()
drone.streamon()  # Fixed: was stream_on() which doesn't exist

# ── Capture loop ──────────────────────────────────────────────────────────────

frame_reader = drone.get_frame_read()

while True:
    # Flush stale frame — discard once then read fresh
    # This reduces the lag caused by buffer accumulation
    _ = frame_reader.frame
    img = frame_reader.frame

    if img is None:
        continue

    img = cv2.resize(img, (360, 240))
    cv2.imshow("Tello – Camera Feed", img)

    # Exit on Q or window close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
drone.streamoff()