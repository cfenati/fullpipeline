"""Touch helpers shared by the OpenCV measure tools, the Tk capture GUIs and the menu.

Nothing here talks to hardware. The pieces are small and clock-injectable so their
timing rules (tap vs drag, tap-again-to-confirm, save debounce) are unit-testable.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

import numpy as np

TAP_MAX_MOVE_PX = 12
CONFIRM_TIMEOUT_S = 3.0
MESSAGE_TIMEOUT_S = 2.5
ROW_HEIGHT = 60
BAR_HEIGHT = 2 * ROW_HEIGHT
SAVE_DEBOUNCE_S = 1.0


class ConfirmGate:
    """Two-tap confirmation: the first press arms, a second within the timeout fires.

    A stray touch must not end a session or wipe every point, so destructive buttons
    go through one of these.
    """

    def __init__(self, timeout_s: float = CONFIRM_TIMEOUT_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._timeout_s = timeout_s
        self._clock = clock
        self._armed_at: Optional[float] = None

    def armed(self) -> bool:
        return (self._armed_at is not None
                and self._clock() - self._armed_at <= self._timeout_s)

    def press(self) -> bool:
        """True when this press confirms; otherwise arms the gate and returns False."""
        if self.armed():
            self._armed_at = None
            return True
        self._armed_at = self._clock()
        return False

    def disarm(self) -> None:
        self._armed_at = None


class Debouncer:
    """Rejects events that arrive within ``min_gap_s`` of the last ``mark()``."""

    def __init__(self, min_gap_s: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._min_gap_s = min_gap_s
        self._clock = clock
        self._last: Optional[float] = None

    def mark(self) -> None:
        self._last = self._clock()

    def ready(self) -> bool:
        return self._last is None or self._clock() - self._last >= self._min_gap_s


class GestureTracker:
    """Classifies one press -> release as a tap or a drag."""

    def __init__(self, tap_max_move_px: float = TAP_MAX_MOVE_PX) -> None:
        self._tap_max = tap_max_move_px
        self._start: Optional[Tuple[int, int]] = None
        self._last: Optional[Tuple[int, int]] = None
        self._dragging = False

    @property
    def active(self) -> bool:
        return self._start is not None

    def press(self, x: int, y: int) -> None:
        self._start = (x, y)
        self._last = (x, y)
        self._dragging = False

    def move(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """Pan delta since the previous call once the touch has become a drag, else None."""
        if self._start is None or self._last is None:
            return None
        if not self._dragging:
            if np.hypot(x - self._start[0], y - self._start[1]) <= self._tap_max:
                return None
            self._dragging = True
        delta = (x - self._last[0], y - self._last[1])
        self._last = (x, y)
        return delta

    def release(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """(x, y) if the touch was a tap; None for a drag or a release with no press."""
        was_tap = self._start is not None and not self._dragging
        self._start = None
        self._last = None
        self._dragging = False
        return (x, y) if was_tap else None


def apply_touch_style(root, maximize: bool = True) -> None:
    """Big-finger ttk styles for the Tk screens, and a maximized window (X11 ``-zoomed``)."""
    import tkinter as tk
    from tkinter import ttk

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("Touch.TButton", font=("TkDefaultFont", 20, "bold"), padding=(28, 22))
    style.configure("TouchHuge.TButton", font=("TkDefaultFont", 34, "bold"), padding=(40, 40))
    style.configure("Tile.TButton", font=("TkDefaultFont", 30, "bold"), padding=(20, 40))
    style.configure("Key.TButton", font=("TkDefaultFont", 24, "bold"), padding=(10, 18))
    style.configure("Title.TLabel", font=("TkDefaultFont", 36, "bold"))
    style.configure("Notice.TLabel", font=("TkDefaultFont", 20), foreground="#8a5a00")
    style.configure("Body.TLabel", font=("TkDefaultFont", 22))
    if maximize:
        try:
            root.attributes("-zoomed", True)
        except tk.TclError:
            pass  # window manager without -zoomed support: stay at the requested size
