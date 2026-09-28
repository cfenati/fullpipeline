"""Touch helpers shared by the OpenCV measure tools, the Tk capture GUIs and the menu.

Nothing here talks to hardware. The pieces are small and clock-injectable so their
timing rules (tap vs drag, tap-again-to-confirm, save debounce) are unit-testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

TAP_MAX_MOVE_PX = 12
CONFIRM_TIMEOUT_S = 3.0
MESSAGE_TIMEOUT_S = 2.5
ROW_HEIGHT = 60
BAR_HEIGHT = 2 * ROW_HEIGHT
SAVE_DEBOUNCE_S = 1.0
ZOOM_IN_FACTOR = 1.25
ZOOM_OUT_FACTOR = 0.8
CROSSHAIR_COLOR = (255, 0, 255)  # BGR magenta: distinct from the A/B/pending marker colours
FONT = cv2.FONT_HERSHEY_SIMPLEX


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


@dataclass(frozen=True)
class Button:
    key: str
    label: str
    confirm: bool = False


ARROW_KEYS = ("left", "right", "up", "down")

_FILLS = {"place": (60, 150, 60), "cancel": (60, 60, 150)}
_DEFAULT_FILL = (70, 70, 70)
_ARMED_FILL = (0, 150, 230)


def bar_rows(show_lock: bool) -> List[List[Button]]:
    """Two rows: aiming controls on top, session controls below."""
    aim = [Button("place", "Place"), Button("cancel", "Cancel"),
           Button("left", ""), Button("right", ""), Button("up", ""), Button("down", "")]
    session = [Button("undo", "Undo"), Button("reset", "Reset", confirm=True),
               Button("zoom_in", "Zoom +"), Button("zoom_out", "Zoom -"), Button("fit", "Fit")]
    if show_lock:
        session.append(Button("lock", "Lock rim"))
    session.append(Button("finish", "Finish", confirm=True))
    return [aim, session]


class ButtonBar:
    """Layout and hit-testing for the touch button strip (rows of equal-width buttons)."""

    def __init__(self, width: int, rows: Sequence[Sequence[Button]]) -> None:
        self.width = width
        self.rows = [list(row) for row in rows]

    @property
    def height(self) -> int:
        return ROW_HEIGHT * len(self.rows)

    def rects(self) -> List[Tuple[Button, Tuple[int, int, int, int]]]:
        out: List[Tuple[Button, Tuple[int, int, int, int]]] = []
        for row_index, row in enumerate(self.rows):
            count = len(row)
            for i, button in enumerate(row):
                x0 = self.width * i // count
                x1 = self.width * (i + 1) // count
                out.append((button, (x0, row_index * ROW_HEIGHT, x1, (row_index + 1) * ROW_HEIGHT)))
        return out

    def hit(self, x: int, y: int) -> Optional[Button]:
        for button, (x0, y0, x1, y1) in self.rects():
            if x0 <= x < x1 and y0 <= y < y1:
                return button
        return None


def _draw_label(image: np.ndarray, text: str, cx: int, cy: int, max_width: int) -> None:
    scale = 0.8
    (width, height), _ = cv2.getTextSize(text, FONT, scale, 2)
    if width > max_width > 0:
        scale *= max_width / width
        (width, height), _ = cv2.getTextSize(text, FONT, scale, 2)
    cv2.putText(image, text, (cx - width // 2, cy + height // 2), FONT, scale,
                (245, 245, 245), 2, cv2.LINE_AA)


def _draw_arrow(image: np.ndarray, key: str, cx: int, cy: int, size: int = 14) -> None:
    triangles = {
        "left": [(cx - size, cy), (cx + size, cy - size), (cx + size, cy + size)],
        "right": [(cx + size, cy), (cx - size, cy - size), (cx - size, cy + size)],
        "up": [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)],
        "down": [(cx, cy + size), (cx - size, cy - size), (cx + size, cy - size)],
    }
    cv2.fillPoly(image, [np.array(triangles[key], np.int32)], (245, 245, 245), cv2.LINE_AA)


def draw_bar(bar: ButtonBar, armed: Dict[str, bool]) -> np.ndarray:
    """Render the button strip; ``armed`` marks confirm buttons waiting for their second tap."""
    image = np.full((bar.height, bar.width, 3), 24, np.uint8)
    for button, (x0, y0, x1, y1) in bar.rects():
        is_armed = armed.get(button.key, False)
        fill = _ARMED_FILL if is_armed else _FILLS.get(button.key, _DEFAULT_FILL)
        cv2.rectangle(image, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), fill, -1)
        cv2.rectangle(image, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), (200, 200, 200), 1)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if button.key in ARROW_KEYS:
            _draw_arrow(image, button.key, cx, cy)
        else:
            _draw_label(image, "Tap again" if is_armed else button.label, cx, cy, x1 - x0 - 16)
    return image


class TouchController:
    """Turns finger gestures into what ``run_interactive`` already understands.

    A tap only aims a tentative crosshair (a finger hides the feature it points at, and
    there is no hover); ``Place`` then sends the tool's own left-click at the crosshair,
    so blob-snap, epipolar snapping and click recording run untouched. Drags pan and the
    zoom buttons zoom the panel's own view. Other buttons queue the key code the keyboard
    path would have produced, drained through ``pop_key`` into ``handle_interactive_key``.
    """

    def __init__(self, state: Dict[str, Any], status_height: int, show_lock: bool,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._state = state
        self._clock = clock
        self._panel_w = int(state["panel_a"].width)
        self._panel_h = int(state["panel_a"].height)
        self._bar_top = self._panel_h + status_height
        self._bar = ButtonBar(2 * self._panel_w, bar_rows(show_lock))
        self._tracker = GestureTracker()
        self._gates = {button.key: ConfirmGate(clock=clock)
                       for row in self._bar.rows for button in row if button.confirm}
        self._keys: List[int] = []
        self._tentative: Optional[Tuple[str, np.ndarray]] = None
        self._touch_panel = "a"
        self._message = ""
        self._message_until = 0.0

    # -- public surface ---------------------------------------------------- #
    @property
    def tentative(self) -> Optional[Tuple[str, np.ndarray]]:
        return self._tentative

    @property
    def bar(self) -> ButtonBar:
        return self._bar

    @property
    def bar_top(self) -> int:
        return self._bar_top

    def wrap(self, inner: Callable[..., None]) -> Callable[..., None]:
        def callback(event, x, y, flags, param):
            self.handle_mouse(event, x, y, flags, inner)
        return callback

    def pop_key(self) -> int:
        return self._keys.pop(0) if self._keys else -1

    def handle_mouse(self, event: int, x: int, y: int, flags: int,
                     inner: Callable[..., None]) -> None:
        if self._state["text_mode"]:
            return
        # OpenCV reports a quick second tap as a double-click, not a second button-down.
        if event == cv2.EVENT_LBUTTONDBLCLK:
            event = cv2.EVENT_LBUTTONDOWN
        if y >= self._bar_top:
            if event == cv2.EVENT_LBUTTONDOWN:
                button = self._bar.hit(x, y - self._bar_top)
                if button is not None:
                    self._press_button(button, inner)
            return
        if y >= self._panel_h:  # status strip
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._tracker.press(x, y)
            self._touch_panel = self._panel_name_at(x)
        elif event == cv2.EVENT_MOUSEMOVE:
            delta = self._tracker.move(x, y)
            if delta is not None:
                self._panel(self._touch_panel).pan(delta)
        elif event == cv2.EVENT_LBUTTONUP:
            tap = self._tracker.release(x, y)
            if tap is not None:
                self._aim(*tap)

    def sync(self) -> None:
        """Point the loupe at the crosshair so the operator sees what Place will record."""
        if self._tentative is None:
            return
        name, point = self._tentative
        sx, sy = self._panel(name).to_screen(point)
        self._state["cursor"] = (sx + self._offset_x(name), sy)

    def compose(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the crosshair/message on the rendered frame and append the button bar."""
        self._draw_crosshair(frame)
        self._draw_message(frame)
        armed = {key: gate.armed() for key, gate in self._gates.items()}
        return np.vstack([frame, draw_bar(self._bar, armed)])

    # -- internals --------------------------------------------------------- #
    def _panel(self, name: str):
        return self._state["panel_b" if name == "b" else "panel_a"]

    def _panel_name_at(self, x: int) -> str:
        return "b" if x >= self._panel_w else "a"

    def _offset_x(self, name: str) -> int:
        return self._panel_w if name == "b" else 0

    def _aim(self, x: int, y: int) -> None:
        name = self._panel_name_at(x)
        local = (x - self._offset_x(name), y)
        self._tentative = (name, np.asarray(self._panel(name).to_image(local), dtype=np.float64))

    def _say(self, text: str) -> None:
        self._message = text
        self._message_until = self._clock() + MESSAGE_TIMEOUT_S

    def _press_button(self, button: Button, inner: Callable[..., None]) -> None:
        for key, gate in self._gates.items():
            if key != button.key:
                gate.disarm()
        if button.confirm and not self._gates[button.key].press():
            return
        key = button.key
        if key == "place":
            self._place(inner)
        elif key == "cancel":
            self._tentative = None
        elif key in ARROW_KEYS:
            self._nudge(key)
        elif key == "undo":
            if self._tentative is not None:
                self._tentative = None
            else:
                self._keys.append(ord("u"))
        elif key == "reset":
            self._tentative = None
            self._keys.append(ord("r"))
        elif key == "fit":
            self._keys.append(ord("0"))
        elif key == "lock":
            self._keys.append(ord("n"))
        elif key == "finish":
            self._keys.append(ord("q"))
        elif key == "zoom_in":
            self._zoom(ZOOM_IN_FACTOR)
        elif key == "zoom_out":
            self._zoom(ZOOM_OUT_FACTOR)

    def _place(self, inner: Callable[..., None]) -> None:
        if self._tentative is None:
            self._say("Tap a spot on the image first")
            return
        name, point = self._tentative
        wanted = "b" if self._state["pending_a"] is not None else "a"
        if name != wanted:
            self._say(f"Tap in camera {wanted.upper()}")
            return
        panel = self._panel(name)
        sx, sy = panel.to_screen(point)
        if not (0 <= sx < panel.width and 0 <= sy < panel.height):
            self._say("Crosshair is off screen - tap the image again")
            return
        inner(cv2.EVENT_LBUTTONDOWN, sx + self._offset_x(name), sy, 0, None)
        self._tentative = None

    def _nudge(self, direction: str) -> None:
        if self._tentative is None:
            return
        name, point = self._tentative
        step = 1.0 / self._panel(name).scale
        dx, dy = {"left": (-step, 0.0), "right": (step, 0.0),
                  "up": (0.0, -step), "down": (0.0, step)}[direction]
        self._tentative = (name, point + np.array([dx, dy]))

    def _zoom(self, factor: float) -> None:
        panel = self._panel(self._touch_panel)
        anchor: Tuple[float, float] = (panel.width / 2.0, panel.height / 2.0)
        if self._tentative is not None and self._tentative[0] == self._touch_panel:
            sx, sy = panel.to_screen(self._tentative[1])
            if 0 <= sx < panel.width and 0 <= sy < panel.height:
                anchor = (float(sx), float(sy))
        panel.zoom_at(anchor, factor)

    def _draw_crosshair(self, frame: np.ndarray) -> None:
        if self._tentative is None:
            return
        name, point = self._tentative
        panel = self._panel(name)
        sx, sy = panel.to_screen(point)
        if not (0 <= sx < panel.width and 0 <= sy < panel.height):
            return
        x = sx + self._offset_x(name)
        cv2.circle(frame, (x, sy), 16, CROSSHAIR_COLOR, 2, cv2.LINE_AA)
        cv2.line(frame, (x - 26, sy), (x + 26, sy), CROSSHAIR_COLOR, 1, cv2.LINE_AA)
        cv2.line(frame, (x, sy - 26), (x, sy + 26), CROSSHAIR_COLOR, 1, cv2.LINE_AA)

    def _draw_message(self, frame: np.ndarray) -> None:
        if not self._message or self._clock() > self._message_until:
            return
        (width, height), _ = cv2.getTextSize(self._message, FONT, 0.8, 2)
        x, y = (frame.shape[1] - width) // 2, 40
        cv2.rectangle(frame, (x - 10, y - height - 10), (x + width + 10, y + 10), (20, 20, 20), -1)
        cv2.putText(frame, self._message, (x, y), FONT, 0.8, (0, 200, 255), 2, cv2.LINE_AA)


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
