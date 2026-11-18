"""
wifi.py - keep the laptop on the drone's Wi-Fi (Windows, via netsh)
====================================================================
Why: the Tello is its own Wi-Fi access point with no internet. Windows sees
"No internet" and quietly switches back to a saved network (home/office
Wi-Fi) that has internet - the drone then never receives commands.

This module lets the app:
  - see which network the laptop is on and whether it can reach the drone
  - join the drone's network (TELLO-XXXXXX) itself
  - switch back to the previous network afterwards (for the analysis)
"""
import re
import socket
import subprocess
import time

DRONE_IP = "192.168.10.1"
NO_WINDOW = 0x08000000          # don't flash a console window (pythonw)


def _netsh(*args, timeout=10):
    try:
        out = subprocess.run(["netsh", "wlan", *args], capture_output=True,
                             timeout=timeout, creationflags=NO_WINDOW)
        return out.stdout.decode("utf-8", errors="replace")
    except Exception:
        return ""


def current_ssid():
    """SSID the Wi-Fi adapter is connected to, or None."""
    text = _netsh("show", "interfaces")
    if not re.search(r"^\s*State\s*:\s*connected", text, re.M | re.I):
        return None
    m = re.search(r"^\s*SSID\s*:\s*(.+?)\s*$", text, re.M)
    return m.group(1) if m else None


def visible_ssids():
    text = _netsh("show", "networks")
    return sorted({m.group(1).strip() for m in
                   re.finditer(r"^SSID \d+\s*:\s*(.+?)\s*$", text, re.M)
                   if m.group(1).strip()})


def saved_profiles():
    text = _netsh("show", "profiles")
    return [m.group(1).strip() for m in
            re.finditer(r"All User Profile\s*:\s*(.+?)\s*$", text, re.M)]


def is_tello(ssid):
    return bool(ssid) and ssid.upper().startswith(("TELLO-", "RMTT-"))


def find_tello():
    """Drone network to use: a visible one first, else a saved one."""
    for ssid in visible_ssids():
        if is_tello(ssid):
            return ssid
    for ssid in saved_profiles():
        if is_tello(ssid):
            return ssid
    return None


def local_ip_toward(host=DRONE_IP):
    """The laptop's own address on the route to `host` (no packet sent)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect((host, 8889))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def on_tello_network():
    ip = local_ip_toward()
    return bool(ip) and ip.startswith("192.168.10.")


def _ensure_open_profile(ssid):
    """The Tello network is open (no password). Windows can only 'connect'
    to networks it has a profile for, so create one if missing."""
    if ssid in saved_profiles():
        return True
    import os
    import tempfile
    hex_name = ssid.encode("utf-8").hex().upper()
    xml = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>{ssid}</name>
  <SSIDConfig><SSID><hex>{hex_name}</hex><name>{ssid}</name></SSID></SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>manual</connectionMode>
  <MSM><security><authEncryption><authentication>open</authentication>
  <encryption>none</encryption><useOneX>false</useOneX></authEncryption>
  </security></MSM>
</WLANProfile>"""
    path = os.path.join(tempfile.gettempdir(), "spidrone_tello_profile.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    _netsh("add", "profile", f"filename={path}", "user=current")
    return ssid in saved_profiles()


def join(ssid, wait=15, need_drone=None):
    """Connect to `ssid` and wait until connected (and, for the drone's
    network, until the laptop has a 192.168.10.x address)."""
    if need_drone is None:
        need_drone = is_tello(ssid)
    if need_drone:
        _ensure_open_profile(ssid)
    _netsh("connect", f"name={ssid}", f"ssid={ssid}")
    end = time.time() + wait
    while time.time() < end:
        if current_ssid() == ssid and (not need_drone or on_tello_network()):
            return True
        time.sleep(0.5)
    return False


def has_internet(host="serverless.roboflow.com", timeout=3):
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False
