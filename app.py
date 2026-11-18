"""
app.py - SPIDrone desktop app
=============================
One window for the whole inspection workflow, no terminal needed:

  Setup tab   : installation details, Roboflow key + model, test on a photo
  Flight tab  : connect, live camera, telemetry, takeoff/land, capture panels
  Report tab  : analyse captured photos, per-panel table, annotated images

Keyboard (when the app window is focused and you're not typing in a box):
  W/S forward/back   A/D left/right   Up/Down climb/descend   Left/Right rotate
  0 takeoff   1 land   I capture panel   SPACE emergency land

Reuses flyingModules.warning.FlightMonitor (safety thresholds + path retrace)
and inspectionModules (detector, panel_map, reporter) unchanged.
"""

import json
import logging
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

from dotenv import dotenv_values  # noqa: E402

import wifi  # noqa: E402

try:
    # djitellopy logs every command at INFO (20 rc lines/second). Left on,
    # that floods the log panel and stalls the UI thread. Warnings only.
    from djitellopy import tello as _tello_mod
    _tello_mod.Tello.LOGGER.setLevel(logging.WARNING)

    class _PrintHandler(logging.Handler):
        # djitellopy's own handler holds the stderr of app start-up, which
        # is None under pythonw and spews "Logging error" tracebacks.
        def emit(self, record):
            try:
                print(f"[DRONE] {record.levelname}: {record.getMessage()}")
            except Exception:
                pass

    _tello_mod.Tello.LOGGER.handlers.clear()
    _tello_mod.Tello.LOGGER.addHandler(_PrintHandler())
    _tello_mod.Tello.LOGGER.propagate = False
except Exception:  # pragma: no cover - library missing
    _tello_mod = None

from flyingModules.warning import FlightMonitor  # noqa: E402
from inspectionModules import detector as detector_module  # noqa: E402
from inspectionModules.detector import PanelDetector  # noqa: E402

CONF_USED = detector_module.CONFIDENCE_THRESHOLD   # guesses below are ignored
from inspectionModules.panel_map import PanelMap  # noqa: E402
from inspectionModules.reporter import InspectionReporter  # noqa: E402

SETTINGS_FILE = os.path.join(APP_DIR, "app_settings.json")
LOG_FILE = os.path.join(APP_DIR, "logs", "spidrone.log")
APP_VERSION = "v9 auto Wi-Fi join + guard"
ENV_FILE = os.path.join(APP_DIR, ".env")
INSPECTIONS_DIR = os.path.join(APP_DIR, "inspections")

VIDEO_W, VIDEO_H = 640, 480
CAPTURES_PER_PANEL = 2
CAPTURE_GAP_MS = 500
CAPTURE_WINDOW_S = 2.5   # how long to watch the video when I is pressed
RC_INTERVAL = 0.05

DEFAULT_SETTINGS = {
    "panel_wattage": 400.0,
    "tariff_per_kwh": 7.5,
    "peak_sun_hours": 5.5,
    "days_per_year": 365,
    "model_id": detector_module.ROBOFLOW_MODEL_ID,
    "speed": 50,
    "cleaning_cost": 40.0,   # ₹ per panel, labour + water (typical Indian rooftop)
    "use_team": True,        # main model + bird-drop & crack specialists
}

MOVE_KEYS = {"w", "a", "s", "d", "up", "down", "left", "right"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            settings.update(json.load(f))
    except (OSError, ValueError):
        pass
    # An earlier trial model was saved as the default by earlier versions;
    # move those users onto the current public model.
    if settings.get("model_id", "").startswith(detector_module.OLD_MODEL_IDS):
        settings["model_id"] = DEFAULT_SETTINGS["model_id"]
    return settings


def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def load_api_key():
    key = (dotenv_values(ENV_FILE).get("ROBOFLOW_API_KEY") or "").strip()
    return "" if key in ("paste_your_key_here", "your_roboflow_api_key_here") else key


def save_api_key(key):
    lines = []
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines()
                     if not ln.strip().startswith("ROBOFLOW_API_KEY")]
    lines.append(f"ROBOFLOW_API_KEY={key}")
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def annual_revenue(settings):
    kwh = (settings["panel_wattage"] / 1000
           * settings["peak_sun_hours"] * settings["days_per_year"])
    return kwh * settings["tariff_per_kwh"]


def installation_config(settings):
    return {
        "panel_wattage": settings["panel_wattage"],
        "tariff_per_kwh": settings["tariff_per_kwh"],
        "peak_sun_hours": settings["peak_sun_hours"],
        "days_per_year": int(settings["days_per_year"]),
        "annual_revenue_per_panel": annual_revenue(settings),
    }


def fit_image(rgb, max_w, max_h):
    """Resize an RGB array to fit max_w x max_h, keeping aspect ratio."""
    h, w = rgb.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def placeholder(text, sub=""):
    img = np.full((VIDEO_H, VIDEO_W, 3), 32, dtype=np.uint8)
    cv2.putText(img, text, (40, VIDEO_H // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
    if sub:
        cv2.putText(img, sub, (40, VIDEO_H // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
    return img


class LogRedirect:
    """Sends print() output from the modules into the app's log panel."""

    def __init__(self, app):
        self.app = app
        self._buf = ""

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip() and not line.startswith("[TELEMETRY]"):
                self.app.log(line)

    def flush(self):
        pass


class AppFlightMonitor(FlightMonitor):
    """FlightMonitor whose yes/no dialogs open in the app window
    (Tk must only be used from the main thread)."""

    def __init__(self, app, drone, path_log, stop_event):
        super().__init__(drone, path_log, stop_event)
        self.app = app

    def _dialog(self, title, message):
        return self.app.ask_yes_no_from_thread(title, message)


_AV_WARNING = 24   # FFmpeg AV_LOG_WARNING; larger numbers = chattier


def _is_damage_msg(entry):
    """True for FFmpeg H.264 messages that mean picture data was lost.

    "Frame num gap" (debug level) is the only sign that a whole small frame
    vanished on Wi-Fi; everything after it is built on the wrong picture.
    Other debug chatter is ignored; errors/warnings use the keyword list."""
    level = entry[0] if isinstance(entry, tuple) and len(entry) == 3 else None
    text = str(entry[-1] if isinstance(entry, tuple) else entry).lower()
    if "frame num gap" in text:
        return True
    if level is not None and level > _AV_WARNING:
        return False
    return any(k in text for k in (
        "conceal", "error while decoding", "corrupt", "missing",
        "out of range", "decode_slice", "left block unavailable",
        "top block unavailable", "invalid", "no frame", "illegal"))


class LowLatencyFrameReader:
    """Reads the Tello's H.264 video and tracks how damaged each frame is.

    Why: Wi-Fi drops packets. The decoder then fills the hole with guesses
    (blocky smears) and every following frame copies that mistake until the
    next intact full picture (key frame). The decoder does not flag those
    frames, but it does print messages ("error while decoding MB ..."), so
    they are captured here and turned into a per-frame damage score:
      0  = clean (nothing lost since an intact key frame)
      >0 = number of damage events since then (bigger = worse)

    Also: single-threaded decoding (frame-threading adds a few frames of
    delay and hides which frame the errors belong to), FFmpeg buffering off,
    big UDP receive buffer, and an on-demand collector used to pick the
    best frames for inspection photos.
    """

    HISTORY = 60   # frames used for the "how damaged is the video" ratio

    def __init__(self, address, grab_timeout=5, mode="all", big_buffers=True):
        import av
        import av.logging
        # Needed so decoder damage messages exist to be captured; keep them
        # out of the app log (they're counted, not shown).
        # DEBUG level is required to see "Frame num gap" (lost whole frames)
        av.logging.set_level(av.logging.DEBUG)
        av.logging.set_skip_repeated(False)
        libav_log = logging.getLogger("libav")
        libav_log.propagate = False
        if not libav_log.handlers:
            libav_log.addHandler(logging.NullHandler())
        from collections import deque
        self.frame = None
        self.frame_id = 0
        self.frame_damage = 0
        self.frame_corrupt = False
        self.recent = deque(maxlen=self.HISTORY)   # True = damaged
        self.decode_errors = 0
        self.hidden_frames = 0
        self.error = None
        self.stopped = False
        self.mode = mode
        self._collect = None
        self._collect_lock = threading.Lock()
        if big_buffers:
            sep = "&" if "?" in address else "?"
            address += (f"{sep}buffer_size=8388608&fifo_size=200000"
                        "&overrun_nonfatal=1")
        self.container = av.open(
            address, timeout=(grab_timeout, None),
            options={"fflags": "nobuffer", "flags": "low_delay"})
        threading.Thread(target=self._run, daemon=True,
                         name="video-decode").start()

    # Share of recent frames that were damaged (0.0 – 1.0)
    @property
    def damaged_ratio(self):
        recent = list(self.recent)
        return sum(recent) / len(recent) if recent else 0.0

    def collect(self, seconds):
        """Start keeping every decoded frame for `seconds`; returns a list
        that fills with (time, frame_id, rgb, damage) tuples."""
        bucket = []
        with self._collect_lock:
            self._collect = (time.time() + seconds, bucket)
        return bucket

    def _run(self):
        import av
        damage = 1          # unknown until the first intact key frame
        try:
            stream = self.container.streams.video[0]
            ctx = stream.codec_context
            try:
                ctx.thread_type = "NONE"
                ctx.thread_count = 1
            except Exception:
                pass
            for packet in self.container.demux(stream):
                if self.stopped:
                    break
                # A damaged packet must not end the stream
                try:
                    with av.logging.Capture(local=False) as logs:
                        frames = packet.decode()
                except Exception:
                    self.decode_errors += 1
                    damage += 1
                    continue
                bad_msgs = sum(1 for m in logs if _is_damage_msg(m))
                self.decode_errors += bad_msgs
                for frame in frames:
                    if frame.key_frame:
                        damage = bad_msgs          # fresh full picture
                    else:
                        damage += bad_msgs
                    if frame.is_corrupt and damage == 0:
                        damage = 1
                    bad_msgs = 0                   # count once per packet
                    self.recent.append(damage > 0)
                    if damage and self.mode == "until_key":
                        self.hidden_frames += 1
                        continue
                    img = frame.to_ndarray(format="rgb24")
                    self.frame_damage = damage
                    self.frame_corrupt = damage > 0
                    self.frame = img
                    self.frame_id += 1
                    with self._collect_lock:
                        col = self._collect
                        if col is not None:
                            if time.time() <= col[0]:
                                col[1].append((time.time(), self.frame_id,
                                               img, damage))
                            else:
                                self._collect = None
        except Exception as exc:
            self.error = str(exc)
        finally:
            try:
                self.container.close()
            except Exception:
                pass

    def stop(self):
        self.stopped = True


# ── App ───────────────────────────────────────────────────────────────────────

class SPIDroneApp:

    def __init__(self, root, tello_factory=None):
        self.root = root
        self.tello_factory = tello_factory
        self.settings = load_settings()
        self.ui_queue = queue.Queue()

        # Drone state
        self.drone = None
        self.frame_reader = None
        self.connected = False
        self.connecting = False
        self.disconnecting = False
        self.tello_ssid = None       # drone network, e.g. TELLO-99ACD8
        self.home_ssid = None        # network to return to for analysis
        self.flying = False
        self.busy_command = False
        self.monitor = None
        self.safety_event = threading.Event()
        self.path_log = []
        self.pressed = set()

        # Session state
        self.session_dir = None
        self.panel_number = 0
        self.frames_saved = 0
        self.capturing = False
        self.capture_progress = 0

        # Analysis state
        self.analysing = False
        self.report_rows = {}
        self.panel_details = {}      # panel_id -> facts per photo
        self.panel_meta = {}         # panel_id -> capture facts (meta json)
        self._report_data = {}
        self.preview_photo = None
        self.video_photo = None

        self._log_lock = threading.Lock()
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

        root.title("SPIDrone – Solar Panel Inspection")
        root.minsize(960, 640)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_ui()
        self._bind_keys()

        sys.stdout = LogRedirect(self)
        sys.stderr = LogRedirect(self)

        # Record crashes in background threads and UI callbacks
        def thread_hook(args):
            import traceback
            self.log("ERROR in " + (args.thread.name if args.thread else "?")
                     + ": " + "".join(traceback.format_exception(
                         args.exc_type, args.exc_value,
                         args.exc_traceback)).strip())
        threading.excepthook = thread_hook

        def tk_hook(exc, val, tb):
            import traceback
            self.log("ERROR: " + "".join(
                traceback.format_exception(exc, val, tb)).strip())
        root.report_callback_exception = tk_hook

        self.log(f"=== SPIDrone app started ({APP_VERSION}) ===")
        self.log("Welcome. Fill in Setup once, then connect to the drone "
                 "on the Flight tab.")
        if not load_api_key():
            self.log("No Roboflow API key yet – add it on the Setup tab.")

        self.root.after(30, self._video_tick)
        self.root.after(50, self._ui_queue_tick)
        self.root.after(1000, self._telemetry_tick)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style(self.root)
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=6)
        style.configure("Danger.TButton", font=("Segoe UI", 11, "bold"),
                        foreground="#b00020", padding=6)
        style.configure("Head.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10))

        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        self.tabs = ttk.Notebook(outer)
        self.tabs.grid(row=0, column=0, sticky="nsew")

        self.flight_tab = ttk.Frame(self.tabs, padding=8)
        self.setup_tab = ttk.Frame(self.tabs, padding=12)
        self.report_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(self.flight_tab, text="  Flight  ")
        self.tabs.add(self.setup_tab, text="  Setup  ")
        self.tabs.add(self.report_tab, text="  Report  ")

        self._build_flight_tab()
        self._build_setup_tab()
        self._build_report_tab()

        log_frame = ttk.LabelFrame(outer, text="Log", padding=4)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=5, wrap="word",
                                state="disabled", font=("Consolas", 9),
                                relief="flat", takefocus=0)
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    def _build_flight_tab(self):
        tab = self.flight_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        # Video
        # Fixed-size container: the label's image must not drive its size,
        # or resizing the frame to the label would grow it every tick.
        video_box = tk.Frame(tab, width=VIDEO_W, height=VIDEO_H, bg="#202020")
        video_box.grid(row=0, column=0, sticky="nsew")
        video_box.grid_propagate(False)
        video_box.columnconfigure(0, weight=1)
        video_box.rowconfigure(0, weight=1)
        self.video_label = tk.Label(video_box, bg="#202020", bd=0,
                                    highlightthickness=0)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Flight buttons under video
        btns = ttk.Frame(tab)
        btns.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.btn_takeoff = ttk.Button(btns, text="Take off  [0]",
                                      style="Big.TButton",
                                      command=self.takeoff, takefocus=0)
        self.btn_land = ttk.Button(btns, text="Land  [1]", style="Big.TButton",
                                   command=self.land, takefocus=0)
        self.btn_capture = ttk.Button(btns, text="📷 Capture panel  [I]",
                                      style="Big.TButton",
                                      command=self.capture_panel, takefocus=0)
        self.btn_emergency = ttk.Button(btns, text="EMERGENCY LAND  [Space]",
                                        style="Danger.TButton",
                                        command=self.emergency_land,
                                        takefocus=0)
        for i, b in enumerate((self.btn_takeoff, self.btn_land,
                               self.btn_capture, self.btn_emergency)):
            b.grid(row=0, column=i, padx=(0, 6), sticky="ew")
            btns.columnconfigure(i, weight=1)

        ttk.Label(
            tab,
            text="Move: W/S forward-back · A/D left-right · ↑/↓ up-down · "
                 "←/→ rotate.   Keys work when this window is in front "
                 "(click the video if they stop).",
            foreground="#555").grid(row=2, column=0, sticky="w", pady=(6, 0))

        # Side panel
        side = ttk.Frame(tab, padding=(12, 0, 0, 0), width=300)
        side.grid(row=0, column=1, rowspan=3, sticky="ns")

        conn = ttk.LabelFrame(side, text="Connection", padding=8)
        conn.pack(fill="x")
        self.conn_status = ttk.Label(conn, text="● Not connected",
                                     foreground="#888", style="Status.TLabel")
        self.conn_status.pack(anchor="w")
        self.conn_help = ttk.Label(conn, text="1. Turn on the Tello\n"
                                              "2. Join its Wi-Fi (TELLO-XXXXXX)\n"
                                              "3. Click Connect",
                                   foreground="#555", justify="left")
        self.conn_help.pack(anchor="w", pady=(4, 6))
        self.btn_connect = ttk.Button(conn, text="Connect to drone",
                                      style="Big.TButton",
                                      command=self.toggle_connect, takefocus=0)
        self.btn_connect.pack(fill="x")

        tele = ttk.LabelFrame(side, text="Drone status", padding=8)
        tele.pack(fill="x", pady=(8, 0))
        tele.columnconfigure(1, weight=1)
        ttk.Label(tele, text="Battery").grid(row=0, column=0, sticky="w")
        self.battery_bar = ttk.Progressbar(tele, maximum=100, length=140)
        self.battery_bar.grid(row=0, column=1, sticky="ew", padx=6)
        self.battery_lbl = ttk.Label(tele, text="–", width=5)
        self.battery_lbl.grid(row=0, column=2)
        self.tele_vars = {}
        for r, (key, label) in enumerate((("height", "Height"),
                                          ("temp", "Temperature"),
                                          ("time", "Flight time"),
                                          ("state", "State")), start=1):
            ttk.Label(tele, text=label).grid(row=r, column=0, sticky="w",
                                             pady=1)
            var = tk.StringVar(value="–")
            ttk.Label(tele, textvariable=var).grid(row=r, column=1,
                                                   columnspan=2, sticky="w",
                                                   padx=6)
            self.tele_vars[key] = var

        speed_box = ttk.LabelFrame(side, text="Speed (start at 30–50)",
                                   padding=8)
        speed_box.pack(fill="x", pady=(8, 0))
        self.speed = int(self.settings.get("speed", 50))
        self.speed_var = tk.IntVar(value=self.speed)
        speed_row = ttk.Frame(speed_box)
        speed_row.pack(fill="x")
        ttk.Scale(speed_row, from_=10, to=100, variable=self.speed_var,
                  command=self._on_speed, takefocus=0).pack(
            side="left", fill="x", expand=True)
        self.speed_lbl = ttk.Label(speed_row, text=f"{self.speed} %", width=5,
                                   anchor="e")
        self.speed_lbl.pack(side="left")

        sess = ttk.LabelFrame(side, text="This session", padding=8)
        sess.pack(fill="x", pady=(8, 0))
        self.session_lbl = ttk.Label(sess, text="Panels: 0   Photos: 0")
        self.session_lbl.pack(anchor="w")
        ttk.Label(sess, text="Panel fills the box → press I → hold still "
                             "~3 s. R retakes the last panel.",
                  foreground="#555", wraplength=260,
                  justify="left").pack(anchor="w", pady=(4, 0))
        ttk.Button(sess, text="↻ Retake last panel  [R]",
                   command=self.retake_panel, takefocus=0).pack(
            fill="x", pady=(6, 0))
        self.btn_go_report = ttk.Button(sess, text="Landed? Go to Report →",
                                        command=self.go_to_report,
                                        takefocus=0)
        self.btn_go_report.pack(fill="x", pady=(6, 0))

    def _build_setup_tab(self):
        tab = self.setup_tab
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)

        inst = ttk.LabelFrame(tab, text="Installation details "
                                        "(used for the money estimates)",
                              padding=12)
        inst.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.setup_vars = {}
        fields = (("panel_wattage", "Panel wattage (W)"),
                  ("tariff_per_kwh", "Electricity rate (₹ per kWh)"),
                  ("peak_sun_hours", "Peak sun hours per day"),
                  ("days_per_year", "Operating days per year"),
                  ("cleaning_cost", "Cleaning cost per panel (₹)"))
        for r, (key, label) in enumerate(fields):
            ttk.Label(inst, text=label).grid(row=r, column=0, sticky="w",
                                             pady=4)
            var = tk.StringVar(value=str(self.settings[key]))
            ttk.Entry(inst, textvariable=var, width=12).grid(
                row=r, column=1, sticky="w", padx=8)
            var.trace_add("write", lambda *_: self._update_revenue_label())
            self.setup_vars[key] = var
        self.revenue_lbl = ttk.Label(inst, text="", style="Head.TLabel")
        self.revenue_lbl.grid(row=len(fields), column=0, columnspan=2,
                              sticky="w", pady=(10, 0))
        self._update_revenue_label()

        api = ttk.LabelFrame(tab, text="Roboflow (the AI that checks the "
                                       "photos)", padding=12)
        api.grid(row=0, column=1, sticky="nsew")
        api.columnconfigure(0, weight=1)
        ttk.Label(api, text="API key  (roboflow.com → Settings → API Keys)"
                  ).grid(row=0, column=0, columnspan=2, sticky="w")
        self.api_key_var = tk.StringVar(value=load_api_key())
        self.api_entry = ttk.Entry(api, textvariable=self.api_key_var,
                                   show="•", width=40)
        self.api_entry.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        self.show_key_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(api, text="Show", variable=self.show_key_var,
                        command=lambda: self.api_entry.configure(
                            show="" if self.show_key_var.get() else "•")
                        ).grid(row=1, column=1, padx=(6, 0), pady=(2, 8))
        ttk.Label(api, text="Model ID  (change only if you trained your own)"
                  ).grid(row=2, column=0, columnspan=2, sticky="w")
        self.model_var = tk.StringVar(value=self.settings["model_id"])
        ttk.Entry(api, textvariable=self.model_var, width=40).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(2, 4))
        self.team_var = tk.BooleanVar(value=bool(self.settings.get("use_team",
                                                                   True)))
        ttk.Checkbutton(
            api, variable=self.team_var, takefocus=0,
            text="Smart analysis: find each panel's outline, measure dirt "
                 "inside it, check bird drops & cracks (4 models, 4× Roboflow "
                 "usage)").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Button(api, text="Test on a photo…", command=self.test_on_photo,
                   takefocus=0).grid(row=5, column=0, sticky="w")
        ttk.Label(api, text="Needs normal internet (not the Tello Wi-Fi). "
                            "Pick any solar panel photo – the result appears "
                            "on the Report tab.",
                  foreground="#555", wraplength=380, justify="left").grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))

        bottom = ttk.Frame(tab)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        ttk.Button(bottom, text="Save settings", style="Big.TButton",
                   command=self.save_setup, takefocus=0).pack(side="left")
        self.setup_saved_lbl = ttk.Label(bottom, text="", foreground="#2e7d32")
        self.setup_saved_lbl.pack(side="left", padx=10)

        tips = ttk.LabelFrame(tab, text="Before your first flight", padding=12)
        tips.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        ttk.Label(tips, justify="left", wraplength=900, text=(
            "• When Windows asks whether Python may use the network, tick BOTH "
            "Private and Public – otherwise the video won't show.\n"
            "• The camera points forward, not down: hover beside/below a panel "
            "so it fills the view.\n"
            "• Safety monitor: at 15% battery or 25 m it offers to fly back; at "
            "5% or 28 m it CUTS THE MOTORS and the drone drops. Land early.\n"
            "• Practise indoors or over grass before flying near real panels.")
        ).pack(anchor="w")

    def _build_report_tab(self):
        tab = self.report_tab
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(2, weight=1)

        actions = ttk.Frame(tab)
        actions.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.btn_analyse = ttk.Button(actions, text="Analyse this flight",
                                      style="Big.TButton",
                                      command=self.analyse_current,
                                      takefocus=0)
        self.btn_analyse.pack(side="left")
        ttk.Button(actions, text="Analyse a folder of photos…",
                   command=self.analyse_folder, takefocus=0).pack(
            side="left", padx=8)
        self.btn_open_folder = ttk.Button(actions, text="Open report folder",
                                          command=self.open_report_folder,
                                          state="disabled", takefocus=0)
        self.btn_open_folder.pack(side="left")
        self.analysis_progress = ttk.Progressbar(actions, length=180,
                                                 mode="determinate")
        self.analysis_progress.pack(side="right")

        self.report_hint = ttk.Label(
            tab, foreground="#555", wraplength=960, justify="left",
            text="After landing, switch your Wi-Fi from TELLO-XXXXXX back to "
                 "your home network, then click Analyse this flight.")
        self.report_hint.grid(row=1, column=0, columnspan=2, sticky="w",
                              pady=(8, 8))

        cols = ("panel", "condition", "priority", "coverage", "loss_pct",
                "annual")
        self.tree = ttk.Treeview(tab, columns=cols, show="headings",
                                 selectmode="browse", height=12)
        for col, title, w, anchor in (
                ("panel", "Panel", 55, "center"),
                ("condition", "Condition", 170, "w"),
                ("priority", "Priority", 170, "w"),
                ("coverage", "AI sure", 70, "e"),
                ("loss_pct", "Power lost (range)", 125, "e"),
                ("annual", "Loss / year", 95, "e")):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=w, anchor=anchor, stretch=False)
        self.tree.grid(row=2, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._on_row_select)
        self.tree.tag_configure("bad", foreground="#b00020")
        self.tree.tag_configure("warn", foreground="#c25e00")
        self.tree.tag_configure("ok", foreground="#2e7d32")
        self.tree.tag_configure("err", foreground="#777")

        preview_outer = ttk.LabelFrame(tab, text="Photo & details", padding=4)
        preview_outer.grid(row=2, column=1, sticky="nsew", padx=(10, 0))
        preview_outer.columnconfigure(0, weight=1)
        preview_outer.rowconfigure(2, weight=1)

        photo_bar = ttk.Frame(preview_outer)
        photo_bar.grid(row=1, column=0, sticky="ew", pady=(4, 2))
        self.photo_btns = []
        for n in (1, 2):
            b = ttk.Button(photo_bar, text=f"Photo {n}", width=9,
                           command=lambda n=n: self._show_panel_photo(n),
                           takefocus=0, state="disabled")
            b.pack(side="left", padx=(0, 4))
            self.photo_btns.append(b)
        self.photo_caption = ttk.Label(photo_bar, text="", foreground="#555")
        self.photo_caption.pack(side="left", padx=6)

        self.details_text = tk.Text(preview_outer, height=14, wrap="word",
                                    font=("Segoe UI", 9), relief="flat",
                                    state="disabled", takefocus=0,
                                    background="#fafafa")
        self.details_text.grid(row=2, column=0, sticky="nsew")
        self.details_text.tag_configure("h", font=("Segoe UI", 10, "bold"))
        self.details_text.tag_configure("good", foreground="#2e7d32")
        self.details_text.tag_configure("warn", foreground="#c25e00")
        self.details_text.tag_configure("bad", foreground="#b00020")
        self.details_text.tag_configure("dim", foreground="#777")

        preview_box = tk.Frame(preview_outer, width=360, height=250,
                               bg="#202020")
        preview_box.grid(row=0, column=0, sticky="nsew")
        preview_box.grid_propagate(False)
        preview_box.columnconfigure(0, weight=1)
        preview_box.rowconfigure(0, weight=1)
        self.preview_label = tk.Label(preview_box, bg="#202020", fg="#bbb",
                                      bd=0, highlightthickness=0,
                                      text="Select a panel to see its photo")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        self.total_lbl = ttk.Label(tab, text="", style="Head.TLabel")
        self.total_lbl.grid(row=3, column=0, columnspan=2, sticky="w",
                            pady=(8, 0))

    # ── Generic UI plumbing ───────────────────────────────────────────────────

    def call_ui(self, fn):
        """Run fn on the Tk main thread (safe from any thread)."""
        self.ui_queue.put(fn)

    def _ui_queue_tick(self):
        try:
            while True:
                self.ui_queue.get_nowait()()
        except queue.Empty:
            pass
        self.root.after(50, self._ui_queue_tick)

    def log(self, msg):
        stamp = datetime.now().strftime("%H:%M:%S")
        try:  # permanent record for troubleshooting: logs/spidrone.log
            with self._log_lock:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now():%Y-%m-%d} {stamp}  {msg}\n")
        except OSError:
            pass

        def _append():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", f"{stamp}  {msg}\n")
            # keep the log short so it never slows the window down
            lines = int(self.log_text.index("end-1c").split(".")[0])
            if lines > 400:
                self.log_text.delete("1.0", f"{lines - 300}.0")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.call_ui(_append)

    def ask_yes_no_from_thread(self, title, message):
        done = threading.Event()
        answer = {}

        def _ask():
            self.root.deiconify()
            self.root.lift()
            answer["yes"] = messagebox.askyesno(title, message,
                                                parent=self.root)
            done.set()

        self.call_ui(_ask)
        done.wait()
        return answer["yes"]

    def run_bg(self, fn, name):
        threading.Thread(target=fn, daemon=True, name=name).start()

    def _on_speed(self, _value):
        v = int(float(self.speed_var.get()))
        self.speed_var.set(v)
        self.speed = v
        self.speed_lbl.configure(text=f"{v} %")
        self.settings["speed"] = v

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _read_setup(self, show_errors=True):
        values = {}
        try:
            values["panel_wattage"] = float(self.setup_vars["panel_wattage"].get())
            values["tariff_per_kwh"] = float(self.setup_vars["tariff_per_kwh"].get())
            values["peak_sun_hours"] = float(self.setup_vars["peak_sun_hours"].get())
            values["days_per_year"] = int(float(self.setup_vars["days_per_year"].get()))
            values["cleaning_cost"] = float(self.setup_vars["cleaning_cost"].get())
        except ValueError:
            if show_errors:
                messagebox.showerror("Setup", "Installation details must be "
                                              "numbers.", parent=self.root)
            return None
        return values

    def _update_revenue_label(self):
        values = self._read_setup(show_errors=False)
        if values is None:
            self.revenue_lbl.configure(text="")
            return
        self.revenue_lbl.configure(
            text=f"One clean panel earns ≈ ₹{annual_revenue(values):,.0f} / year")

    def save_setup(self, quiet=False):
        values = self._read_setup()
        if values is None:
            return False
        self.settings.update(values)
        self.settings["model_id"] = self.model_var.get().strip() or \
            DEFAULT_SETTINGS["model_id"]
        self.settings["speed"] = int(self.speed_var.get())
        self.settings["use_team"] = bool(self.team_var.get())
        save_settings(self.settings)
        key = self.api_key_var.get().strip()
        if key and key != load_api_key():
            save_api_key(key)
        if not quiet:
            self.setup_saved_lbl.configure(text="✓ Saved")
            self.root.after(2500,
                            lambda: self.setup_saved_lbl.configure(text=""))
        return True

    def _make_detector(self):
        if not self.save_setup(quiet=True):
            return None
        key = self.api_key_var.get().strip()
        if not key:
            self.tabs.select(self.setup_tab)
            messagebox.showwarning("Roboflow key needed",
                                   "Paste your Roboflow API key on the Setup "
                                   "tab first.", parent=self.root)
            return None
        return PanelDetector(api_key=key, model_id=self.settings["model_id"],
                             team=bool(self.settings.get("use_team", True)))

    def test_on_photo(self):
        path = filedialog.askopenfilename(
            parent=self.root, title="Pick a solar panel photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
        if not path:
            return
        det = self._make_detector()
        if det is None:
            return
        self.log(f"Testing detection on {os.path.basename(path)}…")
        self.tabs.select(self.report_tab)

        def work():
            frame = cv2.imread(path)
            if frame is None:
                self.call_ui(lambda: messagebox.showerror(
                    "Test", "Couldn't open that image.", parent=self.root))
                return
            result = det.detect(frame)

            def show():
                self._show_preview_bgr(result["annotated_frame"])
                if result.get("error"):
                    messagebox.showerror("Test failed",
                                         self._explain_api_error(
                                             result["error"]),
                                         parent=self.root)
                else:
                    issues = [d["class"] for d in result["detections"]
                              if d["class"] != "panel"]
                    messagebox.showinfo(
                        "Test worked ✓",
                        f"The AI responded.\n\nResult: {result['worst_class']}"
                        f"\nIssues found: {', '.join(issues) or 'none'}"
                        + (("\n\nAI scores:\n" + "\n".join(
                            f"  {n}: {c:.0%}" for n, c in sorted(
                                result['scores'].items(),
                                key=lambda kv: -kv[1])))
                           if result.get('classification') else
                           f"\nPanels outlined: {result['panel_count']}"),
                        parent=self.root)
            self.call_ui(show)

        self.run_bg(work, "test-photo")

    @staticmethod
    def _explain_api_error(err):
        if "401" in err or "403" in err:
            return ("Roboflow refused the request (" + err[:120] + ").\n\n"
                    "Check your API key. If the key is right, the default "
                    "model isn't available to your account: fork the "
                    "'clean-dirty-solarPanel' dataset on Roboflow Universe, "
                    "train it, and put your model ID on the Setup tab.")
        if "404" in err:
            return ("Model not found (" + err[:120] + ").\n\nCheck the Model "
                    "ID on the Setup tab.")
        return ("Couldn't reach Roboflow:\n" + err[:200] + "\n\nAre you on "
                "normal internet? The Tello's Wi-Fi has no internet – switch "
                "back to your home Wi-Fi.")

    # ── Connection ────────────────────────────────────────────────────────────

    def toggle_connect(self):
        if self.connected:
            if self.flying and not messagebox.askyesno(
                    "Disconnect", "The drone is flying. Land it and "
                                  "disconnect?", parent=self.root):
                return
            self.disconnect(land=True)
        elif self.disconnecting:
            self.log("Still disconnecting from the last session – try again "
                     "in a few seconds.")
        elif not self.connecting:
            self.connect()

    def _ensure_drone_wifi(self):
        """Put the laptop on the drone's Wi-Fi (bg thread).
        Returns (ok, message for the user)."""
        ssid = wifi.current_ssid()
        if wifi.on_tello_network():
            self.tello_ssid = ssid or self.tello_ssid
            return True, ""
        if ssid and not wifi.is_tello(ssid):
            self.home_ssid = ssid            # to switch back for analysis
        tello_ssid = wifi.find_tello()
        if not tello_ssid:
            return False, ("The drone's Wi-Fi (TELLO-XXXXXX) isn't visible.\n\n"
                           "Turn the drone on and wait ~20 s for the fast "
                           "yellow blink, then click Connect again.")
        self.tello_ssid = tello_ssid
        self.log(f"Laptop is on '{ssid or 'no network'}' – switching to the "
                 f"drone's Wi-Fi {tello_ssid}…")
        if wifi.join(tello_ssid, wait=15):
            self.log(f"Joined {tello_ssid} (laptop address "
                     f"{wifi.local_ip_toward()}).")
            return True, ""
        return False, (f"Couldn't join the drone's Wi-Fi {tello_ssid}.\n\n"
                       "Click it in the Windows Wi-Fi list (bottom-right), "
                       "wait until it says 'No internet, open', then click "
                       "Connect again.\n\nTip: Windows may keep jumping back "
                       f"to '{ssid}'. If so, right-click that network → "
                       "Properties → turn off 'Connect automatically' while "
                       "flying.")

    def connect(self):
        self.connecting = True
        self.btn_connect.configure(state="disabled", text="Connecting…")
        self.conn_status.configure(text="● Connecting…", foreground="#c25e00")
        self.log("Connecting to the Tello…")

        def work():
            if not self.tello_factory:
                ok, message = self._ensure_drone_wifi()
                if not ok:
                    self.call_ui(lambda: self._connect_failed(message))
                    return
            try:
                if self.tello_factory:
                    drone = self.tello_factory()
                else:
                    from djitellopy import tello
                    # 1 retry (not 3): fail in ~10 s with a clear message
                    drone = tello.Tello(retry_count=1)
                drone.connect()
            except Exception as exc:
                msg = str(exc)
                where = wifi.current_ssid() if not self.tello_factory else "?"
                self.call_ui(lambda: self._connect_failed(
                    "The drone didn't answer.\n\n"
                    f"Laptop Wi-Fi: {where}\n\n"
                    "• If that isn't TELLO-XXXXXX, Windows switched away – "
                    "click Connect again (the app re-joins it).\n"
                    "• If it is, restart the drone (hold the power button "
                    "5 s off, then on) and try again.\n\nDetails: " + msg[:150]))
                return

            self.log("Drone answered. Starting camera…")
            try:
                drone.streamon()
                self.log("Camera stream switched on.")
                self._request_best_video(drone)
                reader = None
                try:
                    reader = LowLatencyFrameReader(
                        drone.get_udp_video_address())
                    t0 = time.time()
                    while reader.frame_id == 0 and time.time() - t0 < 6 \
                            and reader.error is None:
                        time.sleep(0.1)
                    if reader.frame_id == 0:
                        self.log(f"Fast video reader got no picture "
                                 f"(error={reader.error}, damaged packets="
                                 f"{reader.decode_errors}); trying the "
                                 f"standard reader.")
                        reader.stop()
                        time.sleep(0.8)
                        reader = None
                    else:
                        self.log(f"Video OK (first picture after "
                                 f"{time.time() - t0:.1f} s).")
                except Exception as exc:
                    self.log(f"Fast video reader failed to open ({exc}); "
                             "trying the standard reader.")
                    reader = None
                if reader is None:
                    reader = drone.get_frame_read()
                    self.log("Standard video reader started.")
            except Exception as exc:
                msg = str(exc)
                self.log(f"Video failed: {msg}")
                reader = None
                self.call_ui(lambda: messagebox.showwarning(
                    "No video",
                    "Connected, but the camera stream didn't start.\n\n"
                    "Most likely Windows Firewall is blocking Python. Allow "
                    "Python on Public networks (Windows Security → Firewall → "
                    "Allow an app), then reconnect.", parent=self.root))

            self.call_ui(lambda: self._connected(drone, reader))

        self.run_bg(work, "connect")

    def _request_best_video(self, drone):
        """Ask for the highest video quality. Newer firmware (SDK 2.0+/3.0)
        accepts these; older Tellos reply 'error', which is harmless."""
        send = getattr(drone, "send_command_with_return", None)
        if send is None:
            return
        results = []
        for cmd in ("setbitrate 5", "setresolution high", "setfps high"):
            try:
                results.append(f"{cmd} → {send(cmd, timeout=2)}")
            except Exception as exc:
                results.append(f"{cmd} → {str(exc)[:40]}")
        self.log("Video quality request: " + "; ".join(results))

    def _connect_failed(self, message):
        self.connecting = False
        self.btn_connect.configure(state="normal", text="Connect to drone")
        self.conn_status.configure(text="● Not connected", foreground="#888")
        self.log("Connection failed.")
        messagebox.showerror("Connection failed", message, parent=self.root)

    def _connected(self, drone, reader):
        self.drone = drone
        self.frame_reader = reader
        self.connected = True
        self.connecting = False
        self.flying = False
        self.path_log = []
        self.safety_event = threading.Event()
        self.btn_connect.configure(state="normal", text="Disconnect")
        self.conn_status.configure(text="● Connected", foreground="#2e7d32")
        self.conn_help.pack_forget()

        if self.session_dir is None:
            self._new_session()

        self.monitor = AppFlightMonitor(self, drone, self.path_log,
                                        self.safety_event)
        self.run_bg(self.monitor.monitor, "monitor")
        self.run_bg(self._rc_loop, "rc")
        if not self.tello_factory:
            self.run_bg(self._wifi_guard, "wifi-guard")

        try:
            battery = drone.get_battery()
            self.log(f"Connected. Battery {battery}%.")
            if battery < 30:
                messagebox.showwarning("Low battery",
                                       f"Battery is at {battery}%. Charge "
                                       "before flying.", parent=self.root)
        except Exception:
            self.log("Connected.")
        self.root.focus_force()

    def _back_to_internet(self):
        """Analysis needs internet: leave the drone's Wi-Fi for the network
        the laptop was on before (bg thread)."""
        if not wifi.on_tello_network():
            return
        target = self.home_ssid
        if not target:
            visible = set(wifi.visible_ssids())
            target = next((p for p in wifi.saved_profiles()
                           if not wifi.is_tello(p) and p in visible), None)
        if not target:
            self.log("Still on the drone's Wi-Fi – switch to an internet "
                     "network to analyse.")
            return
        self.log(f"Switching Wi-Fi back to '{target}' for the analysis…")
        joined = wifi.join(target, wait=20, need_drone=False)
        t0 = time.time()
        while joined and not wifi.has_internet() and time.time() - t0 < 15:
            time.sleep(1)
        self.log(f"Back on '{target}'." if joined else
                 f"Couldn't switch to '{target}' – do it in the Windows "
                 "Wi-Fi list.")

    def _wifi_guard(self):
        """While flying, keep the laptop on the drone's Wi-Fi: if Windows
        switches to another network, re-join the drone straight away (the
        Tello lands itself after ~15 s without commands)."""
        drone = self.drone
        while self.connected and self.drone is drone:
            if self.flying and self.tello_ssid and not wifi.on_tello_network():
                now = wifi.current_ssid()
                self.log(f"⚠ Wi-Fi dropped to '{now}' – re-joining "
                         f"{self.tello_ssid}…")
                ok = wifi.join(self.tello_ssid, wait=8)
                self.log("Back on the drone's Wi-Fi." if ok else
                         "Couldn't re-join – press SPACE / Land; the drone "
                         "lands itself after ~15 s without commands.")
            time.sleep(1.5)

    def disconnect(self, land=False):
        drone = self.drone
        self.disconnecting = True
        reader = self.frame_reader
        if reader is not None and hasattr(reader, "stop"):
            try:
                reader.stop()
            except Exception:
                pass
        self.connected = False
        self.capturing = False
        self.pressed.clear()
        self.safety_event.set()   # stops the monitor loop
        self.btn_connect.configure(text="Connect to drone")
        self.conn_status.configure(text="● Not connected", foreground="#888")
        self.conn_help.pack(anchor="w", pady=(4, 6), before=self.btn_connect)

        def work():
            try:
                if drone is None:
                    return
                # only talk to the drone if it's reachable (otherwise each
                # command waits for a reply that can never come)
                reachable = bool(self.tello_factory) or wifi.on_tello_network()
                if land and self.flying and reachable:
                    try:
                        drone.send_rc_control(0, 0, 0, 0)
                        drone.land()
                    except Exception as exc:
                        self.log(f"Land failed: {exc}")
                self.flying = False
                if reachable:
                    try:
                        drone.streamoff()
                    except Exception:
                        pass
                try:
                    drone.end()
                except Exception:
                    pass
                self.log("Disconnected.")
            finally:
                self.disconnecting = False

        self.drone = None
        self.frame_reader = None
        self.run_bg(work, "disconnect")

    def _new_session(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(INSPECTIONS_DIR, stamp)
        os.makedirs(self.session_dir, exist_ok=True)
        self.panel_number = 0
        self.frames_saved = 0
        self._update_session_label()
        self.log(f"Photos will be saved in inspections\\{stamp}")

    # ── Flight commands ───────────────────────────────────────────────────────

    def _require_drone(self):
        if not self.connected or self.drone is None:
            self.log("Not connected to the drone.")
            return False
        return True

    def _drone_command(self, name, fn, on_ok=None):
        if self.busy_command:
            return
        self.busy_command = True

        def work():
            try:
                fn()
                if on_ok:
                    on_ok()
            except Exception as exc:
                msg = str(exc)
                self.log(f"{name} failed: {msg}")
            finally:
                self.busy_command = False

        self.run_bg(work, name)

    def takeoff(self):
        if not self._require_drone() or self.flying:
            return
        if self.safety_event.is_set():
            self.log("Safety monitor took over – reconnect before flying "
                     "again.")
            return
        if self.busy_command:
            return
        self.log("Taking off…")
        drone = self.drone
        # Treat the drone as flying as soon as the command goes out: if the
        # drone's "ok" reply is lost over Wi-Fi, djitellopy keeps retrying
        # for up to a minute while the drone is already in the air.
        self.flying = True
        self.busy_command = True

        def work():
            try:
                drone.takeoff()
                self.log("In the air. Fly to the first panel and press I.")
            except Exception as exc:
                time.sleep(1.0)
                try:
                    airborne = drone.get_height() > 0
                except Exception:
                    airborne = True
                if not self.flying:
                    pass              # user already pressed Land
                elif airborne:
                    self.log("Takeoff reply was lost, but the drone is in "
                             "the air – controls are live.")
                else:
                    self.flying = False
                    self.log(f"Takeoff failed: {exc}")
            finally:
                self.busy_command = False

        self.run_bg(work, "takeoff")

    def land(self):
        # Never blocked by a pending takeoff/command – landing always goes out.
        if not self._require_drone():
            return
        self.log("Landing…")
        self.capturing = False
        self.pressed.clear()
        drone = self.drone

        def work():
            self.flying = False
            try:
                drone.send_rc_control(0, 0, 0, 0)
                drone.land()
                self.log("Landed. When you're done, switch Wi-Fi back to your "
                         "home network and use the Report tab.")
            except Exception as exc:
                self.log(f"Land reply not received ({exc}). If the drone is "
                         "still in the air, press Land again.")

        self.run_bg(work, "land")

    def emergency_land(self):
        if not self._require_drone():
            return
        self.log("EMERGENCY LAND")
        self.capturing = False
        self.pressed.clear()
        self.flying = False
        drone = self.drone

        def work():
            try:
                drone.send_rc_control(0, 0, 0, 0)
                drone.land()
            except Exception as exc:
                self.log(f"Emergency land failed: {exc}")

        self.run_bg(work, "emergency-land")

    def _rc_loop(self):
        prev_vals = [0, 0, 0, 0]
        prev_time = time.time()
        drone = self.drone
        moves_sent, last_report = 0, time.time()
        while self.connected and self.drone is drone:
            if moves_sent and time.time() - last_report > 3:
                self.log(f"Movement commands sent to drone: {moves_sent} "
                         f"(last {prev_vals})")
                moves_sent, last_report = 0, time.time()
            try:
                retracing = self.monitor is not None and self.monitor.retracing
                # Always stream RC while connected (the Tello ignores it on
                # the ground), so control never depends on a takeoff reply.
                if not self.safety_event.is_set() and not retracing:
                    vals = self._rc_values()
                    now = time.time()
                    if self.flying and any(prev_vals):
                        self.path_log.append([*prev_vals, now - prev_time])
                    prev_vals, prev_time = vals, now
                    drone.send_rc_control(*vals)
                    if any(vals):
                        if not moves_sent:
                            last_report = time.time()
                        moves_sent += 1
                else:
                    prev_vals, prev_time = [0, 0, 0, 0], time.time()
            except Exception as exc:
                self.log(f"Control error (continuing): {exc}")
            time.sleep(RC_INTERVAL)

    def _rc_values(self):
        keys = set(self.pressed)
        speed = self.speed  # plain int: Tk variables are main-thread only
        lr = fb = ud = yv = 0
        if "left" in keys:
            yv = -speed
        elif "right" in keys:
            yv = speed
        if "up" in keys:
            ud = speed
        elif "down" in keys:
            ud = -speed
        if "w" in keys:
            fb = speed
        elif "s" in keys:
            fb = -speed
        if "a" in keys:
            lr = -speed
        elif "d" in keys:
            lr = speed
        return [lr, fb, ud, yv]

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def _bind_keys(self):
        self.root.bind_all("<KeyPress>", self._on_key_press, add="+")
        self.root.bind_all("<KeyRelease>", self._on_key_release, add="+")
        # Stop moving if the app loses focus (key releases would be missed)
        self.root.bind_all("<FocusOut>", lambda e: self.root.after(
            50, self._clear_keys_if_unfocused), add="+")
        self.video_label.bind("<Button-1>", lambda e: self.root.focus_set())
        # Leaving a Setup text box: give focus back so flight keys work
        self.tabs.bind("<<NotebookTabChanged>>",
                       lambda e: self.root.focus_set(), add="+")

    def _clear_keys_if_unfocused(self):
        try:
            if self.root.focus_get() is None:
                self.pressed.clear()
        except (KeyError, tk.TclError):
            self.pressed.clear()

    def _flight_keys_active(self):
        return self.flying or self.tabs.select() == str(self.flight_tab)

    def _typing(self):
        w = self.root.focus_get()
        return isinstance(w, (tk.Entry, ttk.Entry, tk.Text))

    def _on_key_press(self, event):
        if self._typing() or not self._flight_keys_active():
            return
        key = event.keysym.lower()
        repeat = key in self.pressed
        self.pressed.add(key)
        if repeat:
            return
        if key in MOVE_KEYS and not self.connected:
            self.log(f"Key {key.upper()} pressed, but no drone is connected.")
        elif key in ("0", "1", "i", "r", "space"):
            self.log(f"Key pressed: {key.upper()}")
        if key == "r":
            self.retake_panel()
        if key == "0":
            self.takeoff()
        elif key == "1":
            self.land()
        elif key == "i":
            self.capture_panel()
        elif key == "space":
            self.emergency_land()
            return "break"
        if key in MOVE_KEYS:
            return "break"

    def _on_key_release(self, event):
        self.pressed.discard(event.keysym.lower())

    # ── Camera + capture ──────────────────────────────────────────────────────

    def _current_frame_rgb(self):
        reader = self.frame_reader
        if reader is None:
            return None
        frame = reader.frame
        if frame is None or frame.shape[:2] == (300, 400) and not frame.any():
            return None
        return frame

    def _frame_key(self, frame):
        reader = self.frame_reader
        return getattr(reader, "frame_id", None) or id(frame)

    def _video_tick(self):
        start = time.perf_counter()
        w = max(self.video_label.winfo_width(), 200)
        h = max(self.video_label.winfo_height(), 150)
        frame = self._current_frame_rgb() if self.connected else None

        if frame is None:
            state = ("wait" if self.connected else "off", w, h)
            if state != getattr(self, "_last_video_key", None):
                self._last_video_key = state
                img = (placeholder("Waiting for video…",
                                   "If it never appears, allow Python in "
                                   "Windows Firewall.")
                       if self.connected else
                       placeholder("No drone connected",
                                   "Join the TELLO Wi-Fi, then click Connect."))
                self._show_video(fit_image(img, w, h))
        else:
            overlay_state = (self.capturing, self.capture_progress,
                             self.flying, self.panel_number,
                             self.frames_saved, self._video_quality_level(),
                             time.time() < getattr(
                                 self, "capture_warning_until", 0))
            key = (self._frame_key(frame), w, h, overlay_state)
            if key != getattr(self, "_last_video_key", None):
                self._last_video_key = key
                # One resize, straight from the full-resolution frame
                fh, fw = frame.shape[:2]
                scale = min(w / fw, h / fh)
                size = (max(1, int(fw * scale)), max(1, int(fh * scale)))
                img = cv2.resize(frame, size, interpolation=(
                    cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR))
                self._show_video(self._overlay(img))

        # Keep ~30 fps without hogging the UI thread
        spent = int((time.perf_counter() - start) * 1000)
        self.root.after(max(5, 33 - spent), self._video_tick)

    def _video_quality_level(self):
        ratio = getattr(self.frame_reader, "damaged_ratio", None)
        if ratio is None:
            return None
        return 0 if ratio < 0.1 else (1 if ratio < 0.5 else 2)

    def _show_video(self, img):
        pil = Image.fromarray(img)
        photo = self.video_photo
        if photo is not None and (photo.width(), photo.height()) == pil.size:
            photo.paste(pil)          # reuse the Tk image: much cheaper
        else:
            self.video_photo = ImageTk.PhotoImage(pil)
            self.video_label.configure(image=self.video_photo)

    def _overlay(self, img):
        # img is a fresh array from cv2.resize, safe to draw on
        h, w = img.shape[:2]
        s = max(0.6, h / 600)               # scale text with the window
        bar_h = int(40 * s)
        img[:bar_h] = (img[:bar_h] * 0.35).astype(np.uint8)
        if self.capturing:
            text = (f"CAPTURING PANEL {self.panel_number} - hold still...")
            colour, dot = (255, 90, 90), (230, 30, 30)
        elif time.time() < getattr(self, "capture_warning_until", 0):
            text = "WEAK SIGNAL - photo damaged. Fly closer, press R to retake"
            colour, dot = (255, 170, 60), (255, 140, 0)
        else:
            text = (f"{'FLYING' if self.flying else 'ON GROUND'}   "
                    f"Panels: {self.panel_number}  Photos: {self.frames_saved}")
            colour = (255, 255, 255)
            dot = (40, 200, 60) if self.flying else (200, 200, 200)
        cv2.circle(img, (int(20 * s), bar_h // 2), int(9 * s), dot, -1,
                   cv2.LINE_AA)
        cv2.putText(img, text, (int(38 * s), bar_h // 2 + int(7 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, colour,
                    max(1, int(s + 0.3)), cv2.LINE_AA)
        # Video signal quality (share of recently damaged frames)
        level = self._video_quality_level()
        if level is not None:
            q_text, q_col = {
                0: ("Video: clear", (90, 220, 90)),
                1: ("Video: some damage", (240, 210, 60)),
                2: ("Video: weak signal - fly closer", (255, 140, 0)),
            }[level]
            (tw, _), _ = cv2.getTextSize(q_text, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.5 * s, 1)
            if w - tw - int(14 * s) > int(38 * s) + 10:
                cv2.putText(img, q_text,
                            (w - tw - int(12 * s), bar_h // 2 + int(6 * s)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, q_col,
                            max(1, int(s + 0.3)), cv2.LINE_AA)
        # centre guide to help frame the panel
        cv2.rectangle(img, (w // 6, h // 6), (w * 5 // 6, h * 5 // 6),
                      (255, 255, 255), 1)
        return img

    def capture_panel(self):
        if not self._require_drone():
            return
        if self.capturing:
            return
        if self._current_frame_rgb() is None:
            self.log("No video yet – can't capture.")
            return
        self.panel_number += 1
        self.capturing = True
        self.capture_progress = 0
        self.capture_warning_until = 0
        reader = self.frame_reader
        if hasattr(reader, "collect"):
            # Watch the video for a moment and keep the best frames, instead
            # of whatever (possibly Wi-Fi-damaged) frame is on screen now.
            self.log(f"Capturing panel {self.panel_number} – hold still…")
            bucket = reader.collect(CAPTURE_WINDOW_S)
            panel = self.panel_number
            self.root.after(int(CAPTURE_WINDOW_S * 1000) + 150,
                            lambda: self.run_bg(
                                lambda: self._finish_best_capture(bucket,
                                                                  panel),
                                "pick-photos"))
        else:
            self.log(f"Capturing panel {self.panel_number}…")
            self.root.after(300, self._capture_one)

    @staticmethod
    def _sharpness(rgb):
        small = cv2.resize(rgb, (480, 360), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _finish_best_capture(self, bucket, panel):
        """Runs in a background thread: choose and save the best frames."""
        frames = list(bucket)
        if not frames:
            self.call_ui(lambda: self._capture_done(panel, [], None))
            return
        least = min(f[3] for f in frames)
        # Only judge sharpness among the least-damaged frames (blocky damage
        # adds fake "sharp" edges, so damage always ranks first).
        pool = [f for f in frames if f[3] <= least + (0 if least == 0 else 1)]
        scored = sorted(((f[3], -self._sharpness(f[2]), f) for f in pool),
                        key=lambda s: (s[0], s[1]))
        chosen = [scored[0][2]]
        for _, _, f in scored[1:]:
            if abs(f[0] - chosen[0][0]) >= 0.3:
                chosen.append(f)
                break
        if len(chosen) < CAPTURES_PER_PANEL:
            rest = sorted(frames, key=lambda f: (f[3], -f[0]))
            for f in rest:
                if f is not chosen[0]:
                    chosen.append(f)
                    break
        saved = []
        for n, (_, _, rgb, damage) in enumerate(chosen[:CAPTURES_PER_PANEL],
                                                start=1):
            path = os.path.join(self.session_dir,
                                f"panel_{panel:03d}_cap_{n}.jpg")
            cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 97])
            saved.append(damage)
        clean_share = sum(1 for f in frames if f[3] == 0) / len(frames)
        # Capture facts the report explains later (video signal, damage)
        try:
            with open(os.path.join(self.session_dir,
                                   f"panel_{panel:03d}_meta.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"clean_frame_share": round(clean_share, 3),
                           "frames_watched": len(frames),
                           "photo_damage": saved}, fh)
        except OSError:
            pass
        self.call_ui(lambda: self._capture_done(panel, saved, clean_share))

    def _capture_done(self, panel, damages, clean_share):
        self.capturing = False
        self.capture_progress = len(damages)
        # count files on disk so retakes (which overwrite) aren't counted twice
        self.frames_saved = len(self._frames_in(self.session_dir))
        self._update_session_label()
        if not damages:
            self.log(f"Panel {panel}: no video received – press I again.")
            self.capture_warning_until = time.time() + 5
            return
        if max(damages) == 0:
            self.log(f"Panel {panel} saved – {len(damages)} clean photos "
                     f"({clean_share:.0%} of frames were clean). "
                     "Fly to the next panel and press I.")
        else:
            self.capture_warning_until = time.time() + 6
            self.log(f"Panel {panel}: WEAK SIGNAL – the best photos still have "
                     f"Wi-Fi damage. Fly closer to the panel (and keep the "
                     f"laptop near), then press R to retake panel {panel}.")

    def retake_panel(self):
        """Capture the last panel again, overwriting its photos."""
        if self.capturing or self.panel_number == 0:
            return
        self.log(f"Retaking panel {self.panel_number}…")
        self.panel_number -= 1
        self.capture_panel()

    def _capture_one(self):
        if not self.capturing:
            return
        frame = self._current_frame_rgb()
        if frame is not None:
            self.capture_progress += 1
            self.frames_saved += 1
            path = os.path.join(
                self.session_dir,
                f"panel_{self.panel_number:03d}_cap_{self.capture_progress}.jpg")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self._update_session_label()
        if self.capture_progress >= CAPTURES_PER_PANEL:
            self.capturing = False
            self.log(f"Panel {self.panel_number} done – fly to the next one "
                     f"and press I.")
        else:
            self.root.after(CAPTURE_GAP_MS, self._capture_one)

    def _update_session_label(self):
        self.session_lbl.configure(
            text=f"Panels: {self.panel_number}   "
                 f"Photos: {self.frames_saved}")

    # ── Telemetry ─────────────────────────────────────────────────────────────

    def _telemetry_tick(self):
        drone = self.drone
        if self.connected and drone is not None:
            try:
                battery = drone.get_battery()
                self.battery_bar["value"] = battery
                self.battery_lbl.configure(
                    text=f"{battery}%",
                    foreground="#b00020" if battery <= 20 else "")
                height = drone.get_height()
                self.tele_vars["height"].set(f"{height / 100:.1f} m")
                self.tele_vars["temp"].set(f"{drone.get_temperature():.0f} °C")
                self.tele_vars["time"].set(f"{drone.get_flight_time()} s")
            except Exception:
                pass
            if self.safety_event.is_set():
                state = "Safety monitor took over"
            elif self.monitor is not None and self.monitor.retracing:
                state = "Flying back (retrace)"
            else:
                state = "Flying" if self.flying else "On the ground"
            self.tele_vars["state"].set(state)
        else:
            self.battery_bar["value"] = 0
            self.battery_lbl.configure(text="–")
            for var in self.tele_vars.values():
                var.set("–")
        self.btn_analyse.configure(
            state="disabled" if self.analysing else "normal")
        self.root.after(1000, self._telemetry_tick)

    # ── Analysis ──────────────────────────────────────────────────────────────

    def go_to_report(self):
        self.tabs.select(self.report_tab)

    def analyse_current(self):
        folder = self.session_dir
        if folder is None or not self._frames_in(folder):
            folder = self._latest_session_with_photos()
            if folder is None:
                messagebox.showinfo("Nothing to analyse",
                                    "No captured photos yet. Fly, press I at "
                                    "each panel, then come back here.",
                                    parent=self.root)
                return
        self._start_analysis(folder, folder)

    def analyse_folder(self):
        folder = filedialog.askdirectory(parent=self.root,
                                         title="Folder of panel photos",
                                         initialdir=INSPECTIONS_DIR
                                         if os.path.isdir(INSPECTIONS_DIR)
                                         else APP_DIR)
        if not folder:
            return
        if not self._frames_in(folder):
            messagebox.showinfo("No photos", "That folder has no .jpg or .png "
                                             "photos.", parent=self.root)
            return
        folder = os.path.normpath(folder)
        if os.path.normpath(os.path.dirname(folder)) == \
                os.path.normpath(INSPECTIONS_DIR):
            out_dir = folder
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(INSPECTIONS_DIR, f"{stamp}_imported")
        self._start_analysis(folder, out_dir)

    @staticmethod
    def _frames_in(folder):
        try:
            return sorted(f for f in os.listdir(folder)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))
                          and "_detected" not in f)
        except OSError:
            return []

    def _latest_session_with_photos(self):
        if not os.path.isdir(INSPECTIONS_DIR):
            return None
        for name in sorted(os.listdir(INSPECTIONS_DIR), reverse=True):
            path = os.path.join(INSPECTIONS_DIR, name)
            if os.path.isdir(path) and self._frames_in(path):
                return path
        return None

    def _start_analysis(self, src_dir, out_dir):
        if self.analysing:
            return
        if self.flying:
            messagebox.showwarning("Still flying",
                                   "Land the drone before analysing.",
                                   parent=self.root)
            return
        det = self._make_detector()
        if det is None:
            return
        if self.connected and not self.tello_factory:
            self.log("Disconnecting from the drone for the analysis "
                     "(it needs internet).")
            self.disconnect()
        config = installation_config(self.settings)
        files = self._frames_in(src_dir)
        os.makedirs(out_dir, exist_ok=True)

        self.analysing = True
        self.report_dir = out_dir
        self.btn_analyse.configure(state="disabled")
        self.tree.delete(*self.tree.get_children())
        self.report_rows = {}
        self.total_lbl.configure(text="")
        self.analysis_progress.configure(maximum=len(files), value=0)
        self.report_hint.configure(
            text=f"Analysing {len(files)} photos from {src_dir} …")
        self.log(f"Analysing {len(files)} photos…")

        def work():
            if not self.tello_factory:
                self._back_to_internet()
            panel_map = PanelMap()
            reporter = InspectionReporter(out_dir)
            used_ids = set()
            for name in files:
                m = re.match(r"panel_(\d+)", name)
                if m:
                    used_ids.add(int(m.group(1)))
            next_id = max(used_ids, default=0)
            errors = 0
            details = {}
            self.panel_details = details
            self.panel_meta = {}
            for pid in used_ids:
                try:
                    with open(os.path.join(src_dir, f"panel_{pid:03d}_meta.json"),
                              encoding="utf-8") as fh:
                        self.panel_meta[pid] = json.load(fh)
                except (OSError, ValueError):
                    pass

            for i, name in enumerate(files):
                frame = cv2.imread(os.path.join(src_dir, name))
                if frame is None:
                    self.log(f"Couldn't read {name}")
                    continue
                m = re.match(r"panel_(\d+)", name)
                if m:
                    panel_id = int(m.group(1))
                else:
                    next_id += 1
                    panel_id = next_id

                result = det.detect(frame)
                stem = os.path.splitext(name)[0]
                annotated = os.path.join(out_dir, f"{stem}_detected.jpg")
                cv2.imwrite(annotated, result["annotated_frame"])
                result["image_path"] = annotated
                result.pop("annotated_frame", None)
                panel_map.record(panel_id, result)
                details.setdefault(panel_id, []).append(
                    self._photo_facts(name, frame, result))

                if result.get("error"):
                    errors += 1
                    if i == 0:
                        err = result["error"]
                        self.call_ui(lambda: self._analysis_failed(err))
                        return
                self.call_ui(lambda v=i + 1: self.analysis_progress
                             .configure(value=v))

            reporter.save_csv(panel_map, config)
            reporter.save_summary(panel_map, config)
            self.call_ui(lambda: self._analysis_done(panel_map, reporter,
                                                     config, errors, out_dir))

        self.run_bg(work, "analysis")

    def _analysis_failed(self, err):
        self.analysing = False
        self.report_hint.configure(text="Analysis stopped – see the message.")
        self.log(f"Analysis stopped: {err[:150]}")
        messagebox.showerror("Analysis failed", self._explain_api_error(err),
                             parent=self.root)

    def _analysis_done(self, panel_map, reporter, config, errors, out_dir):
        self.analysing = False
        self.btn_open_folder.configure(state="normal")
        self._report_ctx = (reporter, config)
        self._report_data = dict(panel_map.panel_data)
        rev = config["annual_revenue_per_panel"]
        total = 0.0
        attention = 0
        for panel_id, data in sorted(panel_map.panel_data.items()):
            fin = reporter._calculate_financial_impact(data, rev)
            sev = data["worst_severity"]
            total += fin["loss_per_year"]
            if sev > 0:
                attention += 1
            tag = {-1: "err", 0: "ok", 2: "warn"}.get(sev, "bad")
            self.tree.insert("", "end", iid=str(panel_id), tags=(tag,), values=(
                panel_id,
                self._condition_text(data, reporter),
                self._priority_text(data, fin, reporter),
                self._ai_confidence_text(data, panel_id),
                self._loss_text(data, fin) if sev > 0 else "–",
                f"₹{fin['loss_per_year']:,.0f}" if sev > 0 else "–",
            ))
            self.report_rows[str(panel_id)] = data.get("image_path")

        n = len(panel_map.panel_data)
        self.total_lbl.configure(
            text=f"{n} panels · {attention} need attention · estimated loss "
                 f"₹{total:,.0f}/year (₹{total / 12:,.0f}/month)")
        note = f" {errors} photo(s) failed – check internet." if errors else ""
        self.report_hint.configure(
            text=f"Done. Report saved in {out_dir} (inspection_report.csv "
                 f"opens in Excel). Click a row to see its photos and why the "
                 f"AI decided that.{note}")
        self.log("Analysis complete.")
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])

    # ── Per-panel details ("why did it say that?") ───────────────────────────

    BLURRY_BELOW = 100     # sharpness score; flight photos: blurry ≈ 75,
    SOFT_BELOW = 150       # sharp 150–350 (measured on real Tello frames)

    def _photo_facts(self, name, frame_bgr, result):
        """Everything the details panel needs about one photo (bg thread)."""
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = detector_module.inspection_area(w, h)
        guesses = []
        smart = bool(result.get("smart"))
        for p in result.get("raw_predictions") or []:
            if not isinstance(p, dict) or "x" not in p:
                continue
            cls = p.get("class", "?")
            conf = float(p.get("confidence", 0))
            if smart:
                src = p.get("source", "")
                off_topic = ((src == "bird model" and cls not in ("bird_drop", "white_blob"))
                             or (src == "crack model" and cls != "cracked")
                             or src == "outline model" or cls in ("panel", "clean"))
                if off_topic or conf < CONF_USED:
                    continue
                why = ("used for the decision" if p.get("used") else
                       "not used (outside the panel, too big to be a dropping, "
                       "or not backed up by the photo)")
                guesses.append((cls, conf, why, src))
                continue
            inside = detector_module.in_inspection_area(p, w, h)
            speciality = p.get("speciality")
            if (speciality and cls != speciality) or cls == "panel":
                continue   # off-topic specialist guesses / vague labels: noise
            if not inside:
                why = "ignored (outside the checked area)"
            elif conf < CONF_USED:
                why = "ignored (AI less than 50% sure)"
            else:
                why = "used"
            guesses.append((cls, conf, why, p.get("source", "")))
        # used first, then by confidence
        guesses.sort(key=lambda g: (not g[2].startswith("used"), -g[1]))
        summary = None
        if smart:
            lvl = result.get("level") if result.get("worst_class") == "dusty" else ""
            summary = (f"{self._nice(result.get('worst_class'))}"
                       + (f" ({lvl})" if lvl else "")
                       + f" – {result.get('dirt_share', 0):.0%} of the panel "
                         f"looks dirty")
        return {
            "summary": summary,
            "file": name,
            "image_path": result.get("image_path"),
            "verdict": result.get("worst_class"),
            "decided_by": result.get("decided_by"),
            "team_errors": result.get("team_errors") or [],
            "error": result.get("error"),
            "sharpness": self._sharpness(frame_bgr),
            "guesses": guesses,
        }

    @staticmethod
    def _nice(cls):
        return {"clean": "no defect", "dusty": "dust", "bird_drop":
                "bird dropping", "cracked": "damage / crack",
                "white_blob": "white spot (possible dropping)",
                "panel": "'defective' (vague)"}.get(cls, cls)

    def _write_details(self, panel_id):
        t = self.details_text
        t.configure(state="normal")
        t.delete("1.0", "end")

        def put(text, tag=None):
            t.insert("end", text, tag)

        data = self._report_data.get(panel_id)
        photos = self.panel_details.get(panel_id, [])
        reporter, config = self._report_ctx
        fin = reporter._calculate_financial_impact(
            data, config["annual_revenue_per_panel"])
        cls, sev = data.get("worst_class"), data.get("worst_severity", 0)
        conf = self._ai_confidence_text(data, panel_id)

        # 1. Verdict in plain words
        put(f"Panel {panel_id}: ", "h")
        if data.get("smart") and sev >= 0:
            name = {"clean": "looks clean", "dusty": f"{data.get('level', '')} dust",
                    "bird_drop": "bird dropping", "cracked": "damage / crack"
                    }.get(cls, cls)
            if data.get("mixed"):
                name = "heavy soiling – dust or many bird droppings"
            elif cls == "bird_drop" and data.get("level") == "many":
                name = "covered in bird droppings"
            tag = {"clean": "good", "dusty": "warn"}.get(cls, "bad")
            loss = data.get("worst_eff_loss", 0)
            put(f"{name}" + (f" (≈{loss:.0%} power lost)" if loss else "") + "\n",
                tag)
            put(f"Why: {data.get('why', '')}.\n")
            put(f"Panel found using the {data.get('panel_source', '?')}; only "
                "what's inside that outline (green on the photo) was judged. "
                "Red tint = what was counted as dirt.\n", "dim")
        elif sev < 0:
            put("could not be analysed\n", "bad")
            put(f"{data.get('error', '')[:160]}\n", "dim")
        elif sev > 0:
            put(f"{self._nice(cls)} found ({conf} sure)\n", "bad" if sev >= 3
                else "warn")
            who = next((p.get("decided_by") for p in photos
                        if p.get("verdict") == cls and p.get("decided_by")),
                       None)
            put("The AI marked this problem inside the area you framed" +
                (f" (found by the {who})" if who else "") + ".\n")
        else:
            used_clean = any(g[0] == "clean" and g[2] == "used"
                             for p in photos for g in p["guesses"])
            if used_clean:
                put(f"looks clean ({conf} sure no defect)\n", "good")
            else:
                put("no problem found – but the AI wasn't sure of anything\n",
                    "warn")
                put("Nothing inside the checked area reached 50% confidence. "
                    "Treat this as 'not checked': fly closer and retake (R).\n")

        if photos and all(p["sharpness"] < self.BLURRY_BELOW for p in photos):
            put("⚠ All photos of this panel are blurry, so this result isn't "
                "reliable. Retake (R) while the drone hovers still.\n", "bad")

        # Doubts: weaker problem guesses inside the area, or photos disagree
        doubts = sorted({(self._nice(g[0]), round(g[1] * 100))
                         for p in photos for g in p["guesses"]
                         if g[0] not in ("clean", "panel") and g[2] !=
                         "used" and "outside" not in g[2] and g[1] >= 0.3},
                        key=lambda d: -d[1])
        if sev == 0 and doubts and not data.get("smart"):
            put("Unsure: the AI also half-saw " + ", ".join(
                f"{n} {c}%" for n, c in doubts[:3]) +
                ". Check this panel by eye or retake closer.\n", "warn")
        verdicts = {p["verdict"] for p in photos if not p["error"]}
        if len(verdicts) > 1:
            put("The two photos disagree (" + " vs ".join(
                self._nice(v) for v in sorted(verdicts)) +
                "); the worse result is used to be safe.\n", "warn")

        # 2. Money
        put("\nMoney\n", "h")
        if sev > 0:
            day = fin["loss_per_day"]
            put(f"Loses about ₹{day:,.2f}/day · ₹{fin['loss_per_month']:,.0f}"
                f"/month · ₹{fin['loss_per_year']:,.0f}/year\n")
            if data.get("smart") and data.get("loss_range"):
                lo, hi = data["loss_range"]
                year = fin["loss_per_year"]
                frac = max(data.get("worst_eff_loss", 0), 1e-6)
                put(f"Power lost: ≈{frac:.0%} (likely {lo:.0%}–{hi:.0%}), i.e. "
                    f"₹{year * lo / frac:,.0f}–₹{year * hi / frac:,.0f} per year.\n")
                if data.get("loss_note"):
                    put(f"How: {data['loss_note']}.\n", "dim")
            else:
                put(f"(≈{fin['efficiency_loss_pct']:.0f}% power – the typical loss "
                    f"for {self._nice(cls)}, not measured from the photo)\n", "dim")
            cost = float(self.settings.get("cleaning_cost", 40))
            if cls == "cracked":
                put("Action: call the installer – cracks can't be cleaned and "
                    "are a shock/fire risk.\n", "bad")
            elif day > 0:
                payback = cost / day
                when = ("clean now" if payback <= 14 else "clean this month"
                        if payback <= 45 else "clean at the next routine wash")
                put(f"Cleaning (₹{cost:,.0f}) pays for itself in "
                    f"{payback:,.0f} days → {when}.\n",
                    "bad" if payback <= 14 else "warn")
                if cls == "bird_drop" or data.get("mixed"):
                    put("Remove droppings soon: they cause hot-spots that can "
                        "permanently damage the cell.\n", "warn")
        elif sev == 0:
            put("No loss estimated for this panel.\n", "good")

        # 3. What the AI saw, photo by photo
        put("\nWhat the AI saw\n", "h")
        for n, p in enumerate(photos, start=1):
            put(f"Photo {n}: ")
            if p["error"]:
                put(f"error – {p['error'][:80]}\n", "bad")
                continue
            if p.get("summary"):
                put(p["summary"] + "\n")
                for g_cls, g_conf, why, src in p["guesses"][:6]:
                    who = f" [{src}]" if src else ""
                    put(f"   • {self._nice(g_cls)} {g_conf:.0%}{who} – ")
                    put(f"{why}\n", None if why.startswith("used") else "dim")
            elif not p["guesses"]:
                put("nothing detected\n", "dim")
            else:
                put("\n")
                for g_cls, g_conf, why, src in p["guesses"][:8]:
                    who = f" [{src}]" if src else ""
                    put(f"   • {self._nice(g_cls)} {g_conf:.0%}{who} – ")
                    put(f"{why}\n", None if why == "used" else "dim")
            for err in p.get("team_errors", []):
                put(f"   ⚠ {err[:90]}\n", "warn")

        # 4. Photo quality
        put("\nPhoto quality\n", "h")
        for n, p in enumerate(photos, start=1):
            s = p["sharpness"]
            if s < self.BLURRY_BELOW:
                put(f"Photo {n}: blurry (score {s:.0f}) – the drone was moving. "
                    "Retake while hovering still.\n", "warn")
            elif s < self.SOFT_BELOW:
                put(f"Photo {n}: slightly soft (score {s:.0f})\n")
            else:
                put(f"Photo {n}: sharp (score {s:.0f})\n", "good")
        meta = self.panel_meta.get(panel_id)
        if meta:
            share = meta.get("clean_frame_share", 0)
            tag = "good" if share >= 0.5 else ("warn" if share < 0.2 else None)
            label = ("good" if share >= 0.5 else
                     "weak – fly/stand closer" if share < 0.2 else "fair")
            put(f"Video signal while capturing: {share:.0%} of frames arrived "
                f"undamaged ({label}).\n", tag)
        t.configure(state="disabled")

    def _show_panel_photo(self, n):
        sel = self.tree.selection()
        if not sel:
            return
        photos = self.panel_details.get(int(sel[0]), [])
        if 1 <= n <= len(photos) and photos[n - 1].get("image_path"):
            img = cv2.imread(photos[n - 1]["image_path"])
            if img is not None:
                self._show_preview_bgr(img)
                self.photo_caption.configure(
                    text=f"{photos[n - 1]['file']} · AI: "
                         f"{self._nice(photos[n - 1]['verdict'])}")

    @staticmethod
    def _condition_text(data, reporter):
        sev = data.get("worst_severity", 0)
        label = reporter.SEVERITY_LABEL.get(sev, "Unknown")
        if data.get("smart") and data.get("mixed"):
            return "Dust or droppings – heavy"
        if data.get("smart") and data.get("worst_class") == "dusty" \
                and data.get("level"):
            return f"Dust – {data['level']}"
        if data.get("smart") and data.get("worst_class") == "bird_drop" \
                and data.get("level") == "many":
            return "Bird droppings – many"
        return label

    @staticmethod
    def _loss_text(data, fin):
        rng = data.get("loss_range") if data.get("smart") else None
        if rng and rng[1] > rng[0]:
            return (f"≈{fin['efficiency_loss_pct']:.0f}% "
                    f"({rng[0] * 100:.0f}–{rng[1] * 100:.0f})")
        return f"≈{fin['efficiency_loss_pct']:.0f}%"

    def _priority_text(self, data, fin, reporter):
        """Priority from what cleaning is worth (smart mode), else by type."""
        cls = data.get("worst_class")
        if not data.get("smart") or cls not in ("dusty", "bird_drop"):
            return reporter.PRIORITY_LABEL.get(data.get("worst_severity", 0),
                                               "Unknown")
        day = fin.get("loss_per_day", 0)
        cost = float(self.settings.get("cleaning_cost", 40))
        payback = cost / day if day > 0 else 1e9
        if cls == "bird_drop" or payback <= 14:
            return "High – clean now"
        if payback <= 45:
            return "Medium – clean this month"
        return "Low – next routine wash"

    def _ai_confidence_text(self, data, panel_id=None):
        """How sure the AI was about the condition shown for this panel,
        taking the best photo of the panel."""
        cls = data.get("worst_class")
        confs = [d["confidence"] for d in data.get("detections", [])
                 if d.get("class") == cls]
        for p in self.panel_details.get(panel_id, []):
            confs += [g[1] for g in p["guesses"]
                      if g[0] == cls and g[2].startswith("used")]
        if not confs:
            # smart mode judges dust/clean from the panel's pixels
            return "measured" if data.get("smart") and cls in (
                "dusty", "clean") else "–"
        return f"{max(confs):.0%}"

    def _on_row_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        panel_id = int(sel[0])
        photos = self.panel_details.get(panel_id, [])
        for n, b in enumerate(self.photo_btns, start=1):
            b.configure(state="normal" if n <= len(photos) else "disabled")
        # Start on the photo that decided the verdict
        path = self.report_rows.get(sel[0])
        first = next((n for n, p in enumerate(photos, start=1)
                      if p.get("image_path") == path), 1)
        if photos:
            self._show_panel_photo(first)
        else:
            img = cv2.imread(path) if path else None
            if img is not None:
                self._show_preview_bgr(img)
        if panel_id in getattr(self, "_report_data", {}):
            self._write_details(panel_id)

    def _show_preview_bgr(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        w = max(self.preview_label.winfo_width(), 320)
        h = max(self.preview_label.winfo_height(), 240)
        self.preview_photo = ImageTk.PhotoImage(
            Image.fromarray(fit_image(rgb, w - 8, h - 8)))
        self.preview_label.configure(image=self.preview_photo, text="")

    def open_report_folder(self):
        folder = getattr(self, "report_dir", None)
        if folder and os.path.isdir(folder):
            os.startfile(folder)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def on_close(self):
        if self.connected and self.flying:
            if not messagebox.askyesno("Quit",
                                       "The drone is flying. Land it and "
                                       "quit?", parent=self.root):
                return
            try:
                self.drone.send_rc_control(0, 0, 0, 0)
                self.drone.land()
            except Exception:
                pass
        if self.connected:
            self.disconnect()
        self.save_setup(quiet=True)
        self.root.after(300, self.root.destroy)


def main():
    try:  # crisp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    SPIDroneApp(root)
    try:
        root.state("zoomed")
    except tk.TclError:
        pass
    root.mainloop()


if __name__ == "__main__":
    main()
