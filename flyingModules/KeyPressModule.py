"""
KeyPressModule.py – Thin pygame wrapper for keyboard polling
=============================================================
Provides two functions used by controls.py (and main.py's controls thread):

  init()        – Start pygame and open a small capture window
  get_key(name) – Return True if the named key is currently held down

Why a separate module?
  Pygame's key-state API requires an active display surface and a processed
  event queue. Wrapping both in these helpers keeps the control code clean
  and avoids repeating the boilerplate in every file that needs key input.

Supported key names
-------------------
  Any pygame constant suffix works, e.g. "a", "LEFT", "SPACE", "UP", "0".
  Internally the name is looked up as pygame.K_<name>, so it must match
  exactly (case-sensitive) what pygame expects.
"""

# Third-party
import pygame


def init():
    """
    Initialise pygame and open a small window to receive keyboard events.

    Must be called once before any get_key() calls. The 400×400 window is
    intentionally small – it just needs to exist so the OS routes key events
    to pygame's event queue.
    """
    pygame.init()
    pygame.display.set_mode((400, 400))
    pygame.display.set_caption("Tello – Key Capture")


def get_key(key_name: str) -> bool:
    """
    Check whether a key is currently held down.

    Parameters
    ----------
    key_name : str
        Pygame key constant suffix, e.g. "a", "LEFT", "UP", "SPACE".

    Returns
    -------
    bool
        True if the key is pressed, False otherwise.

    How it works
    ------------
    1. Flush the pygame event queue (required or the OS marks the window
       as unresponsive and key state stops updating).
    2. Snapshot all key states with get_pressed().
    3. Look up the pygame constant for key_name via getattr().
    4. Refresh the display so the window stays alive.
    """
    # Drain the event queue – without this, get_pressed() goes stale
    for _ in pygame.event.get():
        pass

    key_input = pygame.key.get_pressed()

    # Dynamically resolve e.g. "LEFT" → pygame.K_LEFT
    my_key = getattr(pygame, f"K_{key_name}")

    # Redraw the display to prevent the window from freezing
    pygame.display.update()

    return bool(key_input[my_key])