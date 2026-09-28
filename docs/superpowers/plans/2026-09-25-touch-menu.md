# Touchscreen Menu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A touchscreen home screen (`menu.py`) that launches the existing capture, live-calibration, wound-depth, length and registration scripts on the Jetson, plus a `--touch` mode for the tools it launches so none of them needs a keyboard.

**Architecture:** `menu.py` (Tkinter screens only) runs each stage as a subprocess via display-free logic in `menu_logic.py`. Touch support is a translation layer (`touch_controls.py`) that turns finger gestures into the mouse/key events the OpenCV measure tools already understand; the Tk capture/calibration GUIs get a `--touch` styling mode. One unrelated bug (capture-folder collision on double-tap) is fixed in its own commit.

**Tech Stack:** Python 3.9, Tkinter/ttk, OpenCV (`cv2`), NumPy, Pillow, PyYAML, stdlib `unittest` + `unittest.mock`. No new dependencies.

Spec: `docs/superpowers/specs/2026-09-25-touch-menu-design.md` (approved 2026-09-25).

## Global Constraints

- Python **3.9** (no `match`, no `X | Y` outside annotations); every new module starts with `from __future__ import annotations` and has type hints on function signatures (CLAUDE.md conventions).
- New scripts/flags follow the argparse pattern; `--touch` is opt-in. **With `--touch` absent every existing script behaves exactly as before** — the only CLI-visible change is the capture-folder `_2`/`_3` suffix (Task 5, its own commit).
- Tests use stdlib `unittest` only. Run everything with `python -m unittest discover -s tests -t . -v` from the project root. No new dependency.
- Design floor: screen at least **1280×720**; the menu passes `--window W H` sized inside the screen (an oversized cv2 window sits partly offscreen and never gets input — see `DEFAULT_MAX_WINDOW` comment in `measure_points.py`).
- Touch constants (from the spec): tap-vs-drag threshold **12 px**; "tap again to confirm" timeout **3 s** for Reset, Finish, Discard Worst, Discard & Restart; button bar **120 px** (two rows of 60 px); zoom factors **1.25 / 0.8**; Take-photo debounce **1 s after the previous save finishes**.
- Stage logs go to `logs/menu/<YYYYmmdd_HHMMSS>_<stage>.log`; `logs/` is git-ignored.
- Data hygiene: never commit `captures/`, `calibration/results/`, `color_calibration/`, `color_reports/`, `design/out/`, `registration/results/`, `logs/`. Stage files **explicitly** (`git add <paths>`), never `git add -A`. Leave the pre-existing untracked `calibration/report.txt` alone.
- `config.yaml` has an **unrelated uncommitted edit by the user** (`output_dir: /mnt/usbdrive/captures`): never stage, revert or format it, and never write tests that read the real `config.yaml`. Consequence for the menu: the captures folder may be an absolute path on removable storage, so the menu must **never** `mkdir(parents=True)` it and must say so plainly when the drive is missing (Tasks 8 and 10).
- Camera code and real touchscreen input **cannot be exercised** here. Say so in the final summary; do not claim they work.
- Commit trailer: end every commit message with the attribution line your session specifies (currently `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`).

## Spec refinements made while planning (all found by reading the code)

1. `apply_touch_style()` lives in `touch_controls.py`, not `gui.py`, so `menu.py` never imports `gui.py` (which pulls in camera modules).
2. Pan/zoom buttons call the panel's own `pan` / `zoom_at` instead of synthesizing right-drag / wheel events (simpler, same effect).
3. The Save debounce keys off save **completion**: `save_capture` blocks the Tk loop for seconds, so a queued second tap is only delivered after the save ends.
4. The Result screen also accepts `report_features.txt` / `preview_features.jpg` (what `register_features.py` writes).
5. `EVENT_LBUTTONDBLCLK` is treated as a press: OpenCV reports a quick second tap (e.g. "tap again to confirm") as a double-click, not a second button-down.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `touch_controls.py` | create | `ConfirmGate`, `Debouncer`, `GestureTracker`, `ButtonBar` + drawing, `TouchController` (cv2 touch layer), `apply_touch_style` (Tk). |
| `menu_logic.py` | create | Display-free: stages, session discovery, pagination, command building, name sanitizing, window sizing, `StageRun` subprocess wrapper, friendly errors, result lookup, desktop shortcut. |
| `menu.py` | create | Tkinter screens (home, session picker, name keyboard, running, error, result), `--install-shortcut`. |
| `measure_points.py` | modify | `touch` param on `run_interactive` and `build_status_lines`; `--touch` flag. |
| `measure_wound_depth.py` | modify | `--touch` flag; `RimAndWoundSession.status_lines()`. |
| `gui.py` | modify | `CaptureGUI(touch=)`: big Take-photo button, save debounce. |
| `live_gui.py` | modify | `LiveCaptureGUI(touch=)`: two-row big toolbar, confirm on destructive buttons. |
| `capture_pipeline.py` | modify | `unique_session_dir()` collision fix; `--touch` flag. |
| `calibrate_live.py` | modify | `--touch` flag. |
| `tests/` | create | `__init__.py`, `helpers.py`, one `test_*.py` per unit. |
| `.gitignore`, `README.md` | modify | `logs/`; "Touchscreen menu" section. |

---

### Task 0: Branch and commit the spec and plan

**Files:**
- Create: none (commits existing `docs/superpowers/specs/2026-09-25-touch-menu-design.md`, `docs/superpowers/plans/2026-09-25-touch-menu.md`)

- [ ] **Step 1: Create the branch from local HEAD**

Run: `git switch -c touch-menu`
Expected: `Switched to a new branch 'touch-menu'`. (Do not use `EnterWorktree`'s default: it branches from a stale `origin/master` here.)

- [ ] **Step 2: Commit the two docs only**

```bash
git add docs/superpowers/specs/2026-09-25-touch-menu-design.md docs/superpowers/plans/2026-09-25-touch-menu.md
git commit -m "docs: add touchscreen menu design spec and implementation plan"
```

---

### Task 1: Touch primitives (`ConfirmGate`, `Debouncer`, `GestureTracker`, `apply_touch_style`)

**Files:**
- Create: `touch_controls.py`, `tests/__init__.py`, `tests/test_touch_primitives.py`

**Interfaces:**
- Produces (later tasks import these exact names):
  - constants `TAP_MAX_MOVE_PX = 12`, `CONFIRM_TIMEOUT_S = 3.0`, `MESSAGE_TIMEOUT_S = 2.5`, `ROW_HEIGHT = 60`, `BAR_HEIGHT = 120`, `SAVE_DEBOUNCE_S = 1.0`
  - `ConfirmGate(timeout_s=CONFIRM_TIMEOUT_S, clock=time.monotonic)` with `press() -> bool`, `armed() -> bool`, `disarm() -> None`
  - `Debouncer(min_gap_s: float, clock=time.monotonic)` with `mark() -> None`, `ready() -> bool`
  - `GestureTracker(tap_max_move_px=TAP_MAX_MOVE_PX)` with `press(x, y)`, `move(x, y) -> Optional[Tuple[int, int]]` (pan delta once dragging), `release(x, y) -> Optional[Tuple[int, int]]` (tap position, or `None` for a drag), `active` property
  - `apply_touch_style(root, maximize: bool = True) -> None` (defines ttk styles `Touch.TButton`, `TouchHuge.TButton`, `Tile.TButton`, `Key.TButton`, `Title.TLabel`, `Notice.TLabel`, `Body.TLabel`)

- [ ] **Step 1: Write the failing tests**

Create empty `tests/__init__.py`. Create `tests/test_touch_primitives.py`:

```python
from __future__ import annotations

import unittest

from touch_controls import ConfirmGate, Debouncer, GestureTracker


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class ConfirmGateTest(unittest.TestCase):
    def test_second_press_inside_timeout_confirms(self):
        clock = FakeClock()
        gate = ConfirmGate(3.0, clock)
        self.assertFalse(gate.press())
        self.assertTrue(gate.armed())
        clock.now += 2.0
        self.assertTrue(gate.press())
        self.assertFalse(gate.armed())

    def test_press_after_timeout_rearms_instead_of_confirming(self):
        clock = FakeClock()
        gate = ConfirmGate(3.0, clock)
        gate.press()
        clock.now += 3.5
        self.assertFalse(gate.armed())
        self.assertFalse(gate.press())
        self.assertTrue(gate.armed())

    def test_a_confirmation_is_consumed(self):
        gate = ConfirmGate(3.0, FakeClock())
        gate.press()
        gate.press()
        self.assertFalse(gate.press())

    def test_disarm_cancels_a_pending_confirmation(self):
        gate = ConfirmGate(3.0, FakeClock())
        gate.press()
        gate.disarm()
        self.assertFalse(gate.press())


class DebouncerTest(unittest.TestCase):
    def test_ready_until_marked_then_not_until_gap_passes(self):
        clock = FakeClock()
        debounce = Debouncer(1.0, clock)
        self.assertTrue(debounce.ready())
        debounce.mark()
        self.assertFalse(debounce.ready())
        clock.now += 0.5
        self.assertFalse(debounce.ready())
        clock.now += 0.5
        self.assertTrue(debounce.ready())


class GestureTrackerTest(unittest.TestCase):
    def test_small_movement_is_a_tap_at_the_release_point(self):
        tracker = GestureTracker(12)
        tracker.press(100, 100)
        self.assertIsNone(tracker.move(105, 103))
        self.assertEqual(tracker.release(105, 103), (105, 103))

    def test_large_movement_is_a_drag_with_incremental_deltas(self):
        tracker = GestureTracker(12)
        tracker.press(100, 100)
        self.assertEqual(tracker.move(130, 100), (30, 0))  # includes the threshold distance
        self.assertEqual(tracker.move(140, 110), (10, 10))
        self.assertIsNone(tracker.release(140, 110))

    def test_release_without_press_is_nothing(self):
        self.assertIsNone(GestureTracker().release(5, 5))

    def test_state_resets_after_a_drag(self):
        tracker = GestureTracker(12)
        tracker.press(0, 0)
        tracker.move(50, 0)
        tracker.release(50, 0)
        self.assertFalse(tracker.active)
        tracker.press(10, 10)
        self.assertEqual(tracker.release(10, 10), (10, 10))

    def test_move_before_press_is_ignored(self):
        self.assertIsNone(GestureTracker().move(9, 9))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m unittest tests.test_touch_primitives -v`
Expected: `ModuleNotFoundError: No module named 'touch_controls'`

- [ ] **Step 3: Implement**

Create `touch_controls.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m unittest tests.test_touch_primitives -v`
Expected: 10 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add touch_controls.py tests/__init__.py tests/test_touch_primitives.py
git commit -m "feat: add touch primitives (confirm gate, debouncer, gesture tracker)"
```

---

### Task 2: Button bar and `TouchController`

**Files:**
- Modify: `touch_controls.py` (append)
- Create: `tests/test_touch_controller.py`

**Interfaces:**
- Consumes: `ConfirmGate`, `GestureTracker`, `BAR_HEIGHT`, `ROW_HEIGHT`, `MESSAGE_TIMEOUT_S` from Task 1; a `Panel`-like object (`measure_points.Panel`) with `.width .height .scale .to_image(screen) .to_screen(image) .zoom_at(screen, factor) .pan(delta)`.
- Produces:
  - `Button(key: str, label: str, confirm: bool = False)`, `ARROW_KEYS`
  - `bar_rows(show_lock: bool) -> List[List[Button]]` — row 1: `place, cancel, left, right, up, down`; row 2: `undo, reset(confirm), zoom_in, zoom_out, fit, [lock], finish(confirm)`
  - `ButtonBar(width, rows)` with `.height`, `.rects() -> List[Tuple[Button, (x0,y0,x1,y1)]]`, `.hit(x, y) -> Optional[Button]` (y relative to bar top)
  - `draw_bar(bar, armed: Dict[str, bool]) -> np.ndarray`
  - `TouchController(state, status_height, show_lock, clock=time.monotonic)` with `.wrap(inner_on_mouse)`, `.handle_mouse(event, x, y, flags, inner)`, `.sync()`, `.compose(frame) -> np.ndarray`, `.pop_key() -> int` (`-1` when none), `.tentative` (`None` or `(panel_name, image_xy)`), `.bar`, `.bar_top`
  - `state` keys it reads/writes: `panel_a`, `panel_b`, `pending_a`, `text_mode`, `cursor`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_touch_controller.py`:

```python
from __future__ import annotations

import unittest

import cv2
import numpy as np

from measure_points import Panel
from touch_controls import BAR_HEIGHT, ButtonBar, TouchController, bar_rows, draw_bar

PANEL = (400, 330)
STATUS = 150


def make_state():
    image = np.zeros((480, 640, 3), np.uint8)
    return {"panel_a": Panel(image, PANEL), "panel_b": Panel(image, PANEL),
            "pending_a": None, "text_mode": False, "cursor": None}


class Rig:
    """A TouchController plus a recorder standing in for run_interactive's on_mouse."""

    def __init__(self, show_lock: bool = False) -> None:
        self.state = make_state()
        self.controller = TouchController(self.state, STATUS, show_lock=show_lock)
        self.inner_calls = []
        self._callback = self.controller.wrap(lambda *args: self.inner_calls.append(args))

    def mouse(self, event: int, x: int, y: int) -> None:
        self._callback(event, x, y, 0, None)

    def tap(self, x: int, y: int) -> None:
        self.mouse(cv2.EVENT_LBUTTONDOWN, x, y)
        self.mouse(cv2.EVENT_LBUTTONUP, x, y)

    def press(self, key: str, event: int = cv2.EVENT_LBUTTONDOWN) -> None:
        for button, (x0, y0, x1, y1) in self.controller.bar.rects():
            if button.key == key:
                self.mouse(event, (x0 + x1) // 2, self.controller.bar_top + (y0 + y1) // 2)
                return
        raise KeyError(key)


class ButtonBarTest(unittest.TestCase):
    def test_buttons_tile_each_row_without_gaps(self):
        bar = ButtonBar(800, bar_rows(True))
        for row_index in range(2):
            rects = [r for _b, r in bar.rects() if r[1] == row_index * 60]
            self.assertEqual(rects[0][0], 0)
            self.assertEqual(rects[-1][2], 800)
            for left, right in zip(rects, rects[1:]):
                self.assertEqual(left[2], right[0])

    def test_hit_finds_the_button_under_a_point_and_none_outside(self):
        bar = ButtonBar(800, bar_rows(False))
        button, (x0, y0, x1, y1) = bar.rects()[0]
        self.assertIs(bar.hit((x0 + x1) // 2, (y0 + y1) // 2), button)
        self.assertIsNone(bar.hit(800, 0))
        self.assertIsNone(bar.hit(0, bar.height))

    def test_lock_button_only_when_requested(self):
        keys = lambda show: {b.key for b, _r in ButtonBar(800, bar_rows(show)).rects()}
        self.assertIn("lock", keys(True))
        self.assertNotIn("lock", keys(False))

    def test_draw_bar_size_and_armed_state_is_visible(self):
        bar = ButtonBar(800, bar_rows(False))
        plain = draw_bar(bar, {})
        self.assertEqual(plain.shape, (BAR_HEIGHT, 800, 3))
        self.assertFalse(np.array_equal(plain, draw_bar(bar, {"finish": True})))


class TapAndDragTest(unittest.TestCase):
    def test_tap_sets_a_tentative_crosshair_without_recording(self):
        rig = Rig()
        rig.tap(200, 150)
        name, point = rig.controller.tentative
        self.assertEqual(name, "a")
        np.testing.assert_allclose(point, rig.state["panel_a"].to_image((200, 150)))
        self.assertEqual(rig.inner_calls, [])

    def test_tap_on_the_right_half_is_panel_b(self):
        rig = Rig()
        rig.tap(600, 150)
        name, point = rig.controller.tentative
        self.assertEqual(name, "b")
        np.testing.assert_allclose(point, rig.state["panel_b"].to_image((200, 150)))

    def test_tap_in_the_status_strip_is_ignored(self):
        rig = Rig()
        rig.tap(200, 340)  # panel height is 330, so this is the status strip
        self.assertIsNone(rig.controller.tentative)

    def test_drag_pans_the_touched_panel_and_leaves_no_crosshair(self):
        rig = Rig()
        before_a = rig.state["panel_a"].offset.copy()
        before_b = rig.state["panel_b"].offset.copy()
        rig.mouse(cv2.EVENT_LBUTTONDOWN, 200, 150)
        rig.mouse(cv2.EVENT_MOUSEMOVE, 230, 150)
        rig.mouse(cv2.EVENT_LBUTTONUP, 230, 150)
        moved = before_a - rig.state["panel_a"].offset
        self.assertAlmostEqual(moved[0], 30 / rig.state["panel_a"].scale)
        np.testing.assert_allclose(rig.state["panel_b"].offset, before_b)
        self.assertIsNone(rig.controller.tentative)

    def test_text_mode_ignores_touches(self):
        rig = Rig()
        rig.state["text_mode"] = True
        rig.tap(200, 150)
        self.assertIsNone(rig.controller.tentative)


class PlaceTest(unittest.TestCase):
    def test_place_forwards_a_left_click_at_the_crosshair(self):
        rig = Rig()
        rig.tap(200, 150)
        rig.press("place")
        self.assertEqual(len(rig.inner_calls), 1)
        event, x, y, _flags, _param = rig.inner_calls[0]
        self.assertEqual((event, x, y), (cv2.EVENT_LBUTTONDOWN, 200, 150))
        self.assertIsNone(rig.controller.tentative)

    def test_place_in_b_uses_canvas_coordinates_once_a_is_pending(self):
        rig = Rig()
        rig.state["pending_a"] = [1.0, 2.0]
        rig.tap(600, 150)
        rig.press("place")
        _event, x, y, _flags, _param = rig.inner_calls[0]
        self.assertEqual((x, y), (600, 150))

    def test_place_in_the_wrong_panel_is_refused_and_keeps_the_crosshair(self):
        rig = Rig()
        rig.tap(600, 150)  # B, but nothing is pending in A yet
        rig.press("place")
        self.assertEqual(rig.inner_calls, [])
        self.assertIsNotNone(rig.controller.tentative)

    def test_place_without_a_crosshair_does_nothing(self):
        rig = Rig()
        rig.press("place")
        self.assertEqual(rig.inner_calls, [])

    def test_place_is_refused_when_the_crosshair_was_panned_off_screen(self):
        rig = Rig()
        rig.tap(200, 150)
        rig.state["panel_a"].pan((-5000, 0))
        rig.press("place")
        self.assertEqual(rig.inner_calls, [])
        self.assertIsNotNone(rig.controller.tentative)


class NudgeUndoZoomTest(unittest.TestCase):
    def test_arrows_nudge_one_screen_pixel(self):
        rig = Rig()
        rig.tap(200, 150)
        scale = rig.state["panel_a"].scale
        _n, start = rig.controller.tentative
        start = start.copy()
        rig.press("right")
        rig.press("down")
        _n, end = rig.controller.tentative
        self.assertAlmostEqual(end[0] - start[0], 1 / scale)
        self.assertAlmostEqual(end[1] - start[1], 1 / scale)
        rig.press("left")
        rig.press("up")
        _n, back = rig.controller.tentative
        np.testing.assert_allclose(back, start)

    def test_undo_clears_the_crosshair_before_sending_a_key(self):
        rig = Rig()
        rig.tap(200, 150)
        rig.press("undo")
        self.assertIsNone(rig.controller.tentative)
        self.assertEqual(rig.controller.pop_key(), -1)
        rig.press("undo")
        self.assertEqual(rig.controller.pop_key(), ord("u"))

    def test_cancel_clears_the_crosshair(self):
        rig = Rig()
        rig.tap(200, 150)
        rig.press("cancel")
        self.assertIsNone(rig.controller.tentative)

    def test_fit_and_lock_send_their_keys(self):
        rig = Rig(show_lock=True)
        rig.press("fit")
        rig.press("lock")
        self.assertEqual(rig.controller.pop_key(), ord("0"))
        self.assertEqual(rig.controller.pop_key(), ord("n"))
        self.assertEqual(rig.controller.pop_key(), -1)
        with self.assertRaises(KeyError):
            Rig(show_lock=False).press("lock")

    def test_zoom_buttons_scale_the_last_touched_panel_only(self):
        rig = Rig()
        rig.tap(600, 150)  # touch B
        a0, b0 = rig.state["panel_a"].scale, rig.state["panel_b"].scale
        rig.press("zoom_in")
        self.assertAlmostEqual(rig.state["panel_b"].scale, b0 * 1.25)
        self.assertAlmostEqual(rig.state["panel_a"].scale, a0)
        rig.press("zoom_out")
        self.assertAlmostEqual(rig.state["panel_b"].scale, b0)

    def test_zoom_keeps_the_crosshair_fixed_on_screen(self):
        rig = Rig()
        rig.tap(200, 150)
        rig.press("zoom_in")
        _n, point = rig.controller.tentative
        self.assertEqual(rig.state["panel_a"].to_screen(point), (200, 150))


class ConfirmTest(unittest.TestCase):
    def test_reset_and_finish_need_two_taps(self):
        rig = Rig()
        rig.press("reset")
        self.assertEqual(rig.controller.pop_key(), -1)
        rig.press("reset")
        self.assertEqual(rig.controller.pop_key(), ord("r"))
        rig.press("finish")
        self.assertEqual(rig.controller.pop_key(), -1)
        rig.press("finish")
        self.assertEqual(rig.controller.pop_key(), ord("q"))

    def test_pressing_another_button_disarms_a_pending_confirm(self):
        rig = Rig()
        rig.press("finish")
        rig.press("fit")
        self.assertEqual(rig.controller.pop_key(), ord("0"))
        rig.press("finish")
        self.assertEqual(rig.controller.pop_key(), -1)

    def test_a_quick_second_tap_arrives_as_double_click_and_still_confirms(self):
        rig = Rig()
        rig.press("finish")
        rig.press("finish", event=cv2.EVENT_LBUTTONDBLCLK)
        self.assertEqual(rig.controller.pop_key(), ord("q"))


class SyncAndComposeTest(unittest.TestCase):
    def test_sync_points_the_loupe_cursor_at_the_crosshair(self):
        rig = Rig()
        rig.tap(600, 150)
        rig.controller.sync()
        self.assertEqual(rig.state["cursor"], (600, 150))

    def test_compose_appends_the_bar_and_keeps_the_width(self):
        rig = Rig()
        frame = np.zeros((PANEL[1] + STATUS, 800, 3), np.uint8)
        out = rig.controller.compose(frame)
        self.assertEqual(out.shape, (PANEL[1] + STATUS + BAR_HEIGHT, 800, 3))

    def test_compose_draws_the_crosshair_where_the_operator_tapped(self):
        rig = Rig()
        rig.tap(200, 150)
        frame = np.zeros((PANEL[1] + STATUS, 800, 3), np.uint8)
        out = rig.controller.compose(frame)
        self.assertTrue(out[150, 200].any())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m unittest tests.test_touch_controller -v`
Expected: `ImportError: cannot import name 'ButtonBar' from 'touch_controls'`

- [ ] **Step 3: Implement**

In `touch_controls.py`, change the top imports to:

```python
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
```

Add these constants under `SAVE_DEBOUNCE_S = 1.0`:

```python
ZOOM_IN_FACTOR = 1.25
ZOOM_OUT_FACTOR = 0.8
CROSSHAIR_COLOR = (255, 0, 255)  # BGR magenta: distinct from the A/B/pending marker colours
FONT = cv2.FONT_HERSHEY_SIMPLEX
```

Append (before `apply_touch_style`) the following:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m unittest tests.test_touch_controller tests.test_touch_primitives -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add touch_controls.py tests/test_touch_controller.py
git commit -m "feat: add touch button bar and TouchController for the measure tools"
```

---

### Task 3: Wire touch into `run_interactive` (`measure_points.py`)

**Files:**
- Modify: `measure_points.py` (imports near line 50; `build_status_lines` ~437; `run_interactive` ~549-702; `parse_args` ~802; `main` ~849)
- Create: `tests/helpers.py`, `tests/test_measure_points_touch.py`

**Interfaces:**
- Consumes: `TouchController`, `BAR_HEIGHT` (Task 2).
- Produces: `run_interactive(..., on_status=None, touch: bool = False)`; `build_status_lines(state, on_status, touch: bool = False)`; CLI `--touch` on `measure_points.py`. `check_depth_accuracy.py`'s call to `run_interactive` (line ~793) is unchanged and must keep working.

- [ ] **Step 1: Write the test helpers and failing tests**

Create `tests/helpers.py`:

```python
"""Shared test helpers: a synthetic stereo rig and a headless stand-in for cv2's window calls."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, List
from unittest import mock

import cv2
import numpy as np

from calibration.stereo import StereoExtrinsics


def synthetic_rig() -> StereoExtrinsics:
    """Two identical pinhole cameras 60 mm apart, no distortion, no rotation."""
    k = np.array([[1000.0, 0, 320], [0, 1000.0, 240], [0, 0, 1]])
    r = np.eye(3)
    t = np.array([-0.06, 0.0, 0.0])
    skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return StereoExtrinsics(
        name_a="a", name_b="b", image_size_a=(640, 480), image_size_b=(640, 480),
        camera_matrix_a=k, distortion_a=np.zeros(5),
        camera_matrix_b=k.copy(), distortion_b=np.zeros(5),
        R=r, T=t, essential=skew @ r, fundamental=np.zeros((3, 3)),
        reprojection_error_px=0.1, views_used=1, points_used=1,
    )


class HeadlessCv2:
    """Stands in for OpenCV's window calls so run_interactive runs with no display.

    ``script`` steps run one per ``waitKey`` call; each returns the key code to hand back
    (-1 for "no key"). A step may fire the registered mouse callback first. If the script
    runs dry before the session finishes, the test fails loudly instead of hanging.
    """

    def __init__(self) -> None:
        self.callback = None
        self.frames: List[np.ndarray] = []
        self.script: List[Callable[[], int]] = []
        self.destroyed = False

    def _wait_key(self, _delay: int) -> int:
        if self.destroyed:
            return -1
        if not self.script:
            raise AssertionError("scripted interaction ended before the session finished")
        return self.script.pop(0)()

    @contextmanager
    def patched(self) -> Iterator["HeadlessCv2"]:
        with mock.patch.object(cv2, "namedWindow"), \
             mock.patch.object(cv2, "setMouseCallback",
                               lambda _w, cb: setattr(self, "callback", cb)), \
             mock.patch.object(cv2, "imshow", lambda _w, f: self.frames.append(f.copy())), \
             mock.patch.object(cv2, "waitKey", self._wait_key), \
             mock.patch.object(cv2, "getWindowProperty", lambda *_a: 1), \
             mock.patch.object(cv2, "destroyAllWindows",
                               lambda: setattr(self, "destroyed", True)):
            yield self

    def mouse(self, event: int, x: int, y: int) -> Callable[[], int]:
        def step() -> int:
            self.callback(event, x, y, 0, None)
            return -1
        return step

    def tap(self, x: int, y: int) -> List[Callable[[], int]]:
        return [self.mouse(cv2.EVENT_LBUTTONDOWN, x, y), self.mouse(cv2.EVENT_LBUTTONUP, x, y)]

    def key(self, code: int) -> Callable[[], int]:
        return lambda: code
```

Create `tests/test_measure_points_touch.py`:

```python
from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_points
from tests.helpers import HeadlessCv2, synthetic_rig
from touch_controls import ButtonBar, bar_rows

WINDOW = (800, 600)          # panels 400 wide; touch: 330 tall + 150 status + 120 bar
BAR_TOP = 330 + measure_points.STATUS_HEIGHT


def bar_press(headless: HeadlessCv2, key: str, event: int = cv2.EVENT_LBUTTONDOWN):
    for button, (x0, y0, x1, y1) in ButtonBar(800, bar_rows(False)).rects():
        if button.key == key:
            return headless.mouse(event, (x0 + x1) // 2, BAR_TOP + (y0 + y1) // 2)
    raise KeyError(key)


def run(headless: HeadlessCv2, touch: bool):
    image = np.random.RandomState(0).randint(0, 255, (480, 640, 3), np.uint8)
    with headless.patched():
        return measure_points.run_interactive(
            image, image, synthetic_rig(), (0.05, 0.5), 8, WINDOW, 0, 0.15, touch=touch)


class TouchRunInteractiveTest(unittest.TestCase):
    def test_tap_then_place_records_points_and_finish_needs_two_taps(self):
        h = HeadlessCv2()
        h.script = (h.tap(200, 150) + [bar_press(h, "place")]      # A, point 0
                    + h.tap(600, 150) + [bar_press(h, "place")]    # B, point 0
                    + h.tap(260, 200) + [bar_press(h, "place")]    # A, point 1
                    + h.tap(660, 200) + [bar_press(h, "place")]    # B, point 1
                    + [bar_press(h, "finish"), bar_press(h, "finish")])
        result = run(h, touch=True)
        self.assertEqual(len(result["clicks_a"]), 2)
        self.assertEqual(h.frames[0].shape, (600, 800, 3))  # window size unchanged

    def test_a_single_finish_tap_does_not_end_the_session(self):
        h = HeadlessCv2()
        h.script = [bar_press(h, "finish")]
        with self.assertRaises(AssertionError):  # the loop was still running when the script ran dry
            run(h, touch=True)

    def test_placing_in_the_wrong_panel_records_nothing(self):
        h = HeadlessCv2()
        h.script = (h.tap(600, 150) + [bar_press(h, "place")]
                    + [bar_press(h, "finish"), bar_press(h, "finish")])
        self.assertEqual(run(h, touch=True), {})

    def test_undo_button_removes_the_last_point(self):
        h = HeadlessCv2()
        h.script = (h.tap(200, 150) + [bar_press(h, "place")] + h.tap(600, 150) + [bar_press(h, "place")]
                    + h.tap(260, 200) + [bar_press(h, "place")] + h.tap(660, 200) + [bar_press(h, "place")]
                    + h.tap(300, 250) + [bar_press(h, "place")] + h.tap(700, 250) + [bar_press(h, "place")]
                    + [bar_press(h, "undo"), bar_press(h, "finish"), bar_press(h, "finish")])
        self.assertEqual(len(run(h, touch=True)["clicks_a"]), 2)

    def test_keyboard_and_mouse_path_is_unchanged_without_touch(self):
        h = HeadlessCv2()
        down = cv2.EVENT_LBUTTONDOWN
        h.script = [h.mouse(down, 200, 150), h.mouse(down, 600, 150),
                    h.mouse(down, 260, 200), h.mouse(down, 660, 200), h.key(ord("q"))]
        result = run(h, touch=False)
        self.assertEqual(len(result["clicks_a"]), 2)
        self.assertEqual(h.frames[0].shape, (600, 800, 3))


class BuildStatusLinesTest(unittest.TestCase):
    def _state(self, pending=None):
        return {"text_mode": False, "clicks_a": [], "pending_a": pending,
                "consecutive": [], "linked": True}

    def test_non_touch_lines_are_unchanged(self):
        lines = measure_points.build_status_lines(self._state(), None)
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "point 0: click a feature in camera A")
        self.assertIn("wheel = zoom", lines[2])

    def test_non_touch_on_status_still_replaces_everything(self):
        self.assertEqual(measure_points.build_status_lines(self._state(), lambda: ["x"]), ["x"])

    def test_touch_lines_name_the_next_camera_and_drop_keyboard_hints(self):
        lines = measure_points.build_status_lines(self._state(), None, touch=True)
        self.assertIn("tap a feature in camera A", lines[0])
        self.assertNotIn("wheel", " ".join(lines))
        pending = measure_points.build_status_lines(self._state([1.0, 2.0]), None, touch=True)
        self.assertIn("camera B", pending[0])

    def test_touch_prepends_the_prompt_to_a_callers_status_lines(self):
        lines = measure_points.build_status_lines(self._state(), lambda: ["Rim: 1 of 5"], touch=True)
        self.assertEqual(lines[1:], ["Rim: 1 of 5"])
        self.assertIn("camera A", lines[0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m unittest tests.test_measure_points_touch -v`
Expected: `TypeError: run_interactive() got an unexpected keyword argument 'touch'` (and `build_status_lines` `touch` errors).

- [ ] **Step 3: Implement — edit `measure_points.py`**

3a. Import (after `from calibrate_cameras import load_config, resolve_path  # noqa: E402`):

```python
from touch_controls import BAR_HEIGHT as TOUCH_BAR_HEIGHT, TouchController  # noqa: E402
```

3b. `build_status_lines` signature — replace
```python
def build_status_lines(
    state: Dict[str, Any], on_status: Optional[Callable[[], List[str]]],
) -> List[str]:
```
with
```python
def build_status_lines(
    state: Dict[str, Any], on_status: Optional[Callable[[], List[str]]],
    touch: bool = False,
) -> List[str]:
```
and replace the tail (from `    if on_status is not None:` through the closing `]` of the returned 4-line list) with:

```python
    if on_status is not None and not touch:
        return on_status()
    placed = len(state["clicks_a"])
    if touch:
        head = (f"point {placed}: tap the SAME feature in camera B, then Place"
                if state["pending_a"] is not None
                else f"point {placed}: tap a feature in camera A, then Place")
        if on_status is not None:
            return [head] + on_status()
    else:
        head = (f"point {placed}: click the SAME feature in camera B, near the blue line"
                if state["pending_a"] is not None
                else f"point {placed}: click a feature in camera A")
    recent = "   ".join(f"{i}->{i+1}: {d:.3f}mm"
                        for i, d in enumerate(state["consecutive"]))[-140:]
    if touch:
        return [
            head,
            f"measurements: {recent}" if recent else "measurements: (need two points)",
            "drag = pan   Zoom +/- = zoom   arrows nudge the crosshair   Place = record the point",
            "Undo | Reset | Lock rim (wound depth) | Finish = save and show the result",
        ]
    return [
        head,
        f"measurements: {recent}" if recent else "measurements: (need two points)",
        ("wheel = zoom (fully independent)   right-drag = pan (fully independent)   "
         f"0 = fit both   l = aim B at each new A-click "
         f"{'ON' if state['linked'] else 'OFF'}"),
        "u undo | r reset | n advance/label | q or Esc = finish and print the report",
    ]
```
(Keep the original four returned strings byte-for-byte in the non-touch return.)

3c. `run_interactive`: add the parameter — replace
```python
                    on_status: Optional[Callable[[], List[str]]] = None,
                    ) -> Dict[str, Any]:
```
with
```python
                    on_status: Optional[Callable[[], List[str]]] = None,
                    touch: bool = False,
                    ) -> Dict[str, Any]:
```

3d. Panel size — replace
```python
    panel_size = (max(320, max_window[0] // 2), max(320, max_window[1] - STATUS_HEIGHT))
```
with
```python
    # Touch mode adds a button bar under the status strip; the requested window size stays
    # the total, so the canvas is still exactly 1:1 with screen pixels.
    bar_height = TOUCH_BAR_HEIGHT if touch else 0
    panel_size = (max(320, max_window[0] // 2),
                  max(320, max_window[1] - STATUS_HEIGHT - bar_height))
```

3e. In `run_interactive`, after the 4-space-indented call (NOT the deeper-indented one in `handle_interactive_key`'s `l` branch, which appears earlier in the file), i.e. between
```python
    link_view(state["panel_a"], state["panel_b"], extrinsics, state["depth_hint"])

    def recompute() -> None:
```
insert after the `link_view(...)` line:

```python
    touch_controller: Optional[TouchController] = (
        TouchController(state, STATUS_HEIGHT, show_lock=on_advance is not None)
        if touch else None)
```

3f. Replace `    cv2.setMouseCallback(window, on_mouse)` with:

```python
    cv2.setMouseCallback(
        window, on_mouse if touch_controller is None else touch_controller.wrap(on_mouse))
```

3g. Replace
```python
        state["status"] = build_status_lines(state, on_status)
        cv2.imshow(window, render(state))
        key = cv2.waitKey(20)
```
with
```python
        state["status"] = build_status_lines(state, on_status, touch=touch)
        if touch_controller is not None:
            touch_controller.sync()
        frame = render(state)
        if touch_controller is not None:
            frame = touch_controller.compose(frame)
        cv2.imshow(window, frame)
        key = cv2.waitKey(20)
```

3h. Replace
```python
        if key == -1:
            continue
        key &= 0xFF
```
with
```python
        if key == -1 and touch_controller is not None:
            key = touch_controller.pop_key()
        if key == -1:
            continue
        key &= 0xFF
```

3i. `parse_args`: before `    return parser.parse_args()` add

```python
    parser.add_argument("--touch", action="store_true",
                        help="Touchscreen mode: on-screen buttons (place, undo, zoom, finish) and "
                             "tap-to-aim instead of keyboard keys and the mouse wheel.")
```
and in `main()`'s `run_interactive(...)` call add `touch=args.touch,` as the last keyword argument (after the `float(reg_config.get("default_depth", DEFAULT_REFERENCE_DEPTH)),` line).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m unittest tests.test_measure_points_touch -v`
Expected: all PASS.

- [ ] **Step 5: Regression checks (CLI + the other `run_interactive` caller)**

Run:
```bash
python -c "import measure_points, measure_wound_depth, check_depth_accuracy; print('imports ok')"
python measure_points.py --help | grep -A2 -- "--touch"
python -m unittest discover -s tests -t . -v
```
Expected: `imports ok`; `--touch` help text printed; all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add measure_points.py tests/helpers.py tests/test_measure_points_touch.py
git commit -m "feat: touch mode for run_interactive and measure_points (--touch)"
```

---

### Task 4: Wound-depth `--touch` and live status

**Files:**
- Modify: `measure_wound_depth.py` (`RimAndWoundSession` ~199-273, `parse_args` ~342-383, `main` ~440)
- Create: `tests/test_wound_depth_status.py`

**Interfaces:**
- Consumes: `run_interactive(..., on_status=..., touch=...)` (Task 3).
- Produces: `RimAndWoundSession.status_lines() -> List[str]` and attribute `last_live: Optional[str]`; CLI `--touch`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_wound_depth_status.py`:

```python
from __future__ import annotations

import contextlib
import io
import unittest

from measure_wound_depth import RimAndWoundSession

# mm, camera looking down +z: a flat rim at z=200 and one point 3 mm farther away (a recession)
POINTS = [[0, 0, 200], [10, 0, 200], [0, 10, 200], [10, 10, 200], [5, 5, 200], [5, 5, 203]]


class RimAndWoundStatusLinesTest(unittest.TestCase):
    def test_counts_the_rim_then_locks_then_reports_a_live_depth(self):
        session = RimAndWoundSession(neighbors=5)
        result = {"points_mm": POINTS}
        self.assertEqual(session.status_lines(),
                         ["Rim: 0 of 5 needed", "Tap rim points around the area"])
        with contextlib.redirect_stdout(io.StringIO()):
            for index in range(4):
                session.on_point(index, result)
            self.assertEqual(session.status_lines()[0], "Rim: 4 of 5 needed")
            session.on_point(4, result)
            self.assertEqual(session.status_lines()[0], "Rim: 5 points - tap Lock rim when done")
            session.on_advance()
            self.assertEqual(session.status_lines(),
                             ["Rim locked: 5 points", "Tap points to measure"])
            session.on_point(5, result)
        self.assertEqual(session.status_lines()[1], "point 1: depth 3.00 mm (live estimate)")

    def test_undo_clears_the_live_estimate(self):
        session = RimAndWoundSession(neighbors=5)
        result = {"points_mm": POINTS}
        with contextlib.redirect_stdout(io.StringIO()):
            for index in range(5):
                session.on_point(index, result)
            session.on_advance()
            session.on_point(5, result)
            session.on_undo(5)
        self.assertIsNone(session.last_live)
        self.assertEqual(session.status_lines()[1], "Tap points to measure")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m unittest tests.test_wound_depth_status -v`
Expected: `AttributeError: 'RimAndWoundSession' object has no attribute 'status_lines'`

- [ ] **Step 3: Implement — edit `measure_wound_depth.py`**

3a. In `RimAndWoundSession.__init__`, after `self.points_mm: Dict[int, np.ndarray] = {}` add:
```python
        # Latest live-estimate line for the touch status strip (print() output is invisible there).
        self.last_live: Optional[str] = None
```

3b. In `on_point`, replace
```python
        except ValueError as exc:
            print(f"  wound point {len(self.point_indices) - 1}: {exc}")
            return
```
with
```python
        except ValueError as exc:
            self.last_live = f"point {len(self.point_indices)}: {exc}"
            print(f"  wound point {len(self.point_indices) - 1}: {exc}")
            return
        self.last_live = (f"point {len(self.point_indices)}: "
                          f"depth {live['depth_mm']:.2f} mm (live estimate)")
```

3c. In `on_undo`, add as its first statement: `self.last_live = None`.

3d. After the `on_advance` method (after its final `return None`), add:

```python
    def status_lines(self) -> List[str]:
        """Two status-strip lines for touch mode, where print() output is not visible."""
        if self.rim_closed:
            rim = f"Rim locked: {len(self.rim_indices)} points"
        elif len(self.rim_indices) >= self.neighbors:
            rim = f"Rim: {len(self.rim_indices)} points - tap Lock rim when done"
        else:
            rim = f"Rim: {len(self.rim_indices)} of {self.neighbors} needed"
        idle = "Tap points to measure" if self.rim_closed else "Tap rim points around the area"
        return [rim, self.last_live or idle]
```

3e. `parse_args`: before `    return parser.parse_args()` add
```python
    parser.add_argument("--touch", action="store_true",
                        help="Touchscreen mode: on-screen buttons (place, undo, zoom, lock rim, "
                             "finish) and tap-to-aim instead of keyboard keys and the mouse wheel.")
```

3f. In `main()`, in the `run_interactive(...)` call, replace
```python
            on_advance=session.on_advance,
        )
```
with
```python
            on_advance=session.on_advance,
            on_status=session.status_lines if args.touch else None,
            touch=args.touch,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m unittest tests.test_wound_depth_status -v && python measure_wound_depth.py --help | grep -A2 -- "--touch"`
Expected: PASS; help text printed.

- [ ] **Step 5: Commit**

```bash
git add measure_wound_depth.py tests/test_wound_depth_status.py
git commit -m "feat: --touch for measure_wound_depth with a live rim/depth status strip"
```

---

### Task 5: Capture-folder collision fix (own commit)

**Files:**
- Modify: `capture_pipeline.py` (before `save_capture` ~line 262; inside `save_capture` ~278)
- Create: `tests/test_capture_session_dir.py`

**Interfaces:**
- Produces: `unique_session_dir(output_dir: Path, timestamp: str) -> Path` — `output_dir/timestamp`, or `..._2`, `..._3` if taken (same scheme as `calibration/live_capture.py:_new_session_dir`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_capture_session_dir.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from capture_pipeline import unique_session_dir


class UniqueSessionDirTest(unittest.TestCase):
    def test_second_and_third_capture_in_one_second_get_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = unique_session_dir(root, "20260925_101500")
            self.assertEqual(first.name, "20260925_101500")
            first.mkdir()
            second = unique_session_dir(root, "20260925_101500")
            self.assertEqual(second.name, "20260925_101500_2")
            second.mkdir()
            self.assertEqual(unique_session_dir(root, "20260925_101500").name, "20260925_101500_3")

    def test_different_seconds_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "20260925_101500").mkdir()
            self.assertEqual(unique_session_dir(root, "20260925_101501").name, "20260925_101501")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m unittest tests.test_capture_session_dir -v`
Expected: `ImportError: cannot import name 'unique_session_dir'`

- [ ] **Step 3: Implement — edit `capture_pipeline.py`**

Insert immediately above `def save_capture(`:

```python
def unique_session_dir(output_dir: Path, timestamp: str) -> Path:
    """``output_dir/timestamp``, or ``timestamp_2``, ``timestamp_3``... if that name is taken.

    The timestamp only has second precision, so two captures in the same second used to
    share one folder and the second silently overwrote the first.
    """
    candidate = output_dir / timestamp
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = output_dir / f"{timestamp}_{suffix}"
    return candidate


```

In `save_capture`, replace
```python
    session_dir = output_dir / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    rgb1_path = session_dir / "rgb_cam1.jpg"
```
with
```python
    session_dir = unique_session_dir(output_dir, timestamp)
    session_dir.mkdir(parents=True, exist_ok=False)

    rgb1_path = session_dir / "rgb_cam1.jpg"
```
(Leave `capture_gain_sweep`'s identical-looking `session_dir = output_dir / timestamp` alone — it is not reachable from the menu.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m unittest tests.test_capture_session_dir -v && python -c "import capture_pipeline; print('import ok')"`
Expected: PASS; `import ok`. (`save_capture` itself needs cameras; not exercised here.)

- [ ] **Step 5: Commit (separate commit — this one changes plain-CLI behaviour)**

```bash
git add capture_pipeline.py tests/test_capture_session_dir.py
git commit -m "fix: don't overwrite a capture when two land in the same second

save_capture reused the timestamp folder (exist_ok=True) with only
second-precision names, so a second capture within the same second
overwrote the first while the GUI counter still incremented. Add a
_2/_3 suffix, the same scheme calibrate_live.py already uses."
```

---

### Task 6: Capture GUI `--touch` (`gui.py`, `capture_pipeline.py`)

**Files:**
- Modify: `gui.py` (imports; `CaptureGUI.__init__`, `_build_layout`, `request_save`, `note_capture_saved`), `capture_pipeline.py` (`run_pipeline`, `parse_args`, `main`)
- Create: `tests/test_gui_touch.py`

**Interfaces:**
- Consumes: `apply_touch_style`, `Debouncer`, `SAVE_DEBOUNCE_S` (Task 1).
- Produces: `CaptureGUI(output_dir, window_size="1400x920", touch=False)`; CLI `capture_pipeline.py --touch`; `run_pipeline(..., touch: bool = False)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui_touch.py` (Tk tests are skipped when there is no display; they open a real Tk root briefly):

```python
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from touch_controls import Debouncer


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@unittest.skipUnless(os.environ.get("DISPLAY"), "needs a display")
class CaptureGuiTouchTest(unittest.TestCase):
    def setUp(self):
        from gui import CaptureGUI
        self.tmp = tempfile.TemporaryDirectory()
        self.gui = CaptureGUI(Path(self.tmp.name), "800x600", touch=True)
        self.clock = FakeClock()
        self.gui._save_gate = Debouncer(1.0, self.clock)

    def tearDown(self):
        self.gui.close()
        self.tmp.cleanup()

    def test_a_tap_queued_during_a_slow_save_is_ignored_after_it_finishes(self):
        self.gui.request_save()
        self.assertTrue(self.gui.consume_save_request())
        # ... save_capture runs for seconds; a second tap is delivered only once it ends ...
        self.gui.note_capture_saved(Path("20260925_101500"))
        self.gui.request_save()
        self.assertFalse(self.gui.consume_save_request())

    def test_a_deliberate_second_capture_a_moment_later_works(self):
        self.gui.request_save()
        self.gui.consume_save_request()
        self.gui.note_capture_saved(Path("20260925_101500"))
        self.clock.now += 1.2
        self.gui.request_save()
        self.assertTrue(self.gui.consume_save_request())

    def test_touch_mode_has_no_open_folder_button(self):
        self.gui.pump()
        texts = []

        def walk(widget):
            for child in widget.winfo_children():
                try:
                    texts.append(str(child.cget("text")))
                except Exception:
                    pass
                walk(child)
        walk(self.gui.root)
        self.assertIn("Take photo", texts)
        self.assertNotIn("Open Output Folder", texts)


@unittest.skipUnless(os.environ.get("DISPLAY"), "needs a display")
class CaptureGuiDesktopUnchangedTest(unittest.TestCase):
    def test_desktop_mode_does_not_debounce_and_keeps_its_buttons(self):
        from gui import CaptureGUI
        with tempfile.TemporaryDirectory() as tmp:
            gui = CaptureGUI(Path(tmp), "800x600")
            try:
                gui.note_capture_saved(Path("x"))
                gui.request_save()
                self.assertTrue(gui.consume_save_request())
            finally:
                gui.close()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m unittest tests.test_gui_touch -v`
Expected: `TypeError: __init__() got an unexpected keyword argument 'touch'` (or SKIP if no display — then rely on Step 4's import check plus a manual run on a machine with a display).

- [ ] **Step 3: Implement — edit `gui.py`**

3a. Add to the imports (after `from cameras.thermal_camera import ThermalFrame`):
```python
from touch_controls import SAVE_DEBOUNCE_S, Debouncer, apply_touch_style
```

3b. `CaptureGUI.__init__` signature and start:
```python
    def __init__(
        self,
        output_dir: Path,
        window_size: str = "1400x920",
        touch: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self._touch = touch
        self._save_gate = Debouncer(SAVE_DEBOUNCE_S)
        self._closed = False
```
(keep the rest of the existing assignments), and replace `self.root.minsize(1000, 720)` with
```python
        self.root.minsize(*((640, 400) if touch else (1000, 720)))
```

3c. `_build_layout`: replace the style setup and toolbar buttons — i.e. replace
```python
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(fill=tk.X)

        ttk.Button(
            toolbar,
            text="Save Capture",
            command=self.request_save,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            toolbar,
            text="Open Output Folder",
            command=self.open_output_folder,
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            toolbar,
            text="Quit",
            command=self.request_quit,
        ).pack(side=tk.RIGHT)
```
with
```python
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(fill=tk.X)

        if self._touch:
            apply_touch_style(self.root)
            ttk.Button(
                toolbar, text="Take photo", style="TouchHuge.TButton",
                command=self.request_save,
            ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 12))
            ttk.Button(
                toolbar, text="Done", style="Touch.TButton", command=self.request_quit,
            ).pack(side=tk.RIGHT)
        else:
            style = ttk.Style()
            if "clam" in style.theme_names():
                style.theme_use("clam")
            ttk.Button(
                toolbar,
                text="Save Capture",
                command=self.request_save,
            ).pack(side=tk.LEFT, padx=(0, 6))
            ttk.Button(
                toolbar,
                text="Open Output Folder",
                command=self.open_output_folder,
            ).pack(side=tk.LEFT, padx=(0, 6))

            ttk.Button(
                toolbar,
                text="Quit",
                command=self.request_quit,
            ).pack(side=tk.RIGHT)
```

3d. Replace `request_save` and `note_capture_saved`:
```python
    def request_save(self) -> None:
        # A save blocks this loop for seconds, so a double-tap's second tap is only delivered
        # after the save ends; refusing taps for a moment after completion drops it.
        if self._touch and not self._save_gate.ready():
            return
        self._save_requested = True
```
```python
    def note_capture_saved(self, session_dir: Path) -> None:
        self._save_gate.mark()
        self._capture_count += 1
        self._count_var.set(f"Captures: {self._capture_count}")
        self._save_var.set(f"Last save: {session_dir.name}")
        self.set_status(
            "Saved. Tap Take photo for another." if self._touch
            else "Ready — press Save Capture or S"
        )
```

- [ ] **Step 4: Implement — edit `capture_pipeline.py`**

4a. `run_pipeline` signature: replace
```python
    sync_print: bool = False,
    output: Optional[str] = None,
) -> int:
    config = load_config(config_path)
```
with
```python
    sync_print: bool = False,
    output: Optional[str] = None,
    touch: bool = False,
) -> int:
    config = load_config(config_path)
```
4b. Replace
```python
            gui = CaptureGUI(
                output_dir=output_dir,
                window_size=window_size,
            )
```
with
```python
            gui = CaptureGUI(
                output_dir=output_dir,
                window_size=window_size,
                touch=touch,
            )
```
4c. Replace `            gui.set_status("Ready — Save button, S in terminal, or Ctrl+S")` with
```python
            gui.set_status(
                "Ready — tap Take photo" if touch
                else "Ready — Save button, S in terminal, or Ctrl+S"
            )
```
4d. In `parse_args`, replace
```python
        help="Print grab timings on each save (or each sample with --sync-test)",
    )
    return parser.parse_args()
```
with
```python
        help="Print grab timings on each save (or each sample with --sync-test)",
    )
    parser.add_argument(
        "--touch",
        action="store_true",
        help="Touchscreen mode: one large Take photo button, no keyboard hints",
    )
    return parser.parse_args()
```
4e. In `main()`, replace
```python
        sync_print=sync_print,
        output=args.output,
    )
```
with
```python
        sync_print=sync_print,
        output=args.output,
        touch=args.touch,
    )
```

- [ ] **Step 5: Run to verify**

Run:
```bash
python -m unittest tests.test_gui_touch -v
python -c "import gui, capture_pipeline; print('imports ok')"
python capture_pipeline.py --help | grep -A2 -- "--touch"
```
Expected: PASS (or SKIP without a display); `imports ok`; help printed. **Not exercised:** the live capture loop and real cameras.

- [ ] **Step 6: Commit**

```bash
git add gui.py capture_pipeline.py tests/test_gui_touch.py
git commit -m "feat: --touch for capture_pipeline (big Take photo button, save debounce)"
```

---

### Task 7: Live-calibration GUI `--touch` (`live_gui.py`, `calibrate_live.py`)

**Files:**
- Modify: `live_gui.py` (imports; `__init__`; `_build_layout` toolbar/hint; `toggle_pause`), `calibrate_live.py` (`parse_args`, GUI construction ~278)
- Create: `tests/test_live_gui_touch.py`

**Interfaces:**
- Consumes: `apply_touch_style`, `ConfirmGate`, `CONFIRM_TIMEOUT_S` (Task 1).
- Produces: `LiveCaptureGUI(captures_dir, camera_a, camera_b, window_size="1500x950", touch=False)`; `LiveCaptureGUI._confirming(button, label, action) -> Callable[[], None]`; CLI `calibrate_live.py --touch`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_live_gui_touch.py`:

```python
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from tkinter import ttk


@unittest.skipUnless(os.environ.get("DISPLAY"), "needs a display")
class LiveGuiTouchTest(unittest.TestCase):
    def setUp(self):
        from live_gui import LiveCaptureGUI
        self.tmp = tempfile.TemporaryDirectory()
        self.gui = LiveCaptureGUI(Path(self.tmp.name), "rgb_cam1", "rgb_cam2", "900x700", touch=True)

    def tearDown(self):
        self.gui.close()
        self.tmp.cleanup()

    def _button_texts(self):
        texts = []

        def walk(widget):
            for child in widget.winfo_children():
                if isinstance(child, ttk.Button):
                    texts.append(str(child.cget("text")))
                walk(child)
        walk(self.gui.root)
        return texts

    def test_touch_toolbar_has_no_keyboard_hints_or_folder_button(self):
        texts = self._button_texts()
        self.assertIn("Discard Worst", texts)
        self.assertIn("Discard & Restart", texts)
        self.assertFalse([t for t in texts if "(r)" in t or "(space)" in t or "(q)" in t])
        self.assertNotIn("Open Captures Folder", texts)

    def test_destructive_button_needs_a_second_tap(self):
        called = []
        button = ttk.Button(self.gui.root, text="Discard")
        handler = self.gui._confirming(button, "Discard", lambda: called.append(1))
        handler()
        self.assertEqual(called, [])
        self.assertEqual(button.cget("text"), "Tap again to confirm")
        handler()
        self.assertEqual(called, [1])
        self.assertEqual(button.cget("text"), "Discard")

    def test_pause_label_has_no_keyboard_hint_in_touch_mode(self):
        self.gui.toggle_pause()
        self.assertEqual(self.gui._pause_button.cget("text"), "Resume Auto-Capture")


@unittest.skipUnless(os.environ.get("DISPLAY"), "needs a display")
class LiveGuiDesktopUnchangedTest(unittest.TestCase):
    def test_desktop_toolbar_keeps_its_hints(self):
        from live_gui import LiveCaptureGUI
        with tempfile.TemporaryDirectory() as tmp:
            gui = LiveCaptureGUI(Path(tmp), "rgb_cam1", "rgb_cam2", "900x700")
            try:
                gui.toggle_pause()
                self.assertEqual(gui._pause_button.cget("text"), "Resume Auto-Capture (space)")
            finally:
                gui.close()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m unittest tests.test_live_gui_touch -v`
Expected: `TypeError: __init__() takes from 4 to 5 positional arguments but 6 were given` (or SKIP without a display).

- [ ] **Step 3: Implement — edit `live_gui.py`**

3a. Imports (after `from gui import fit_to_box`):
```python
from touch_controls import CONFIRM_TIMEOUT_S, ConfirmGate, apply_touch_style
```

3b. `__init__`: add the parameter and store it — replace
```python
        camera_b: str,
        window_size: str = "1500x950",
    ) -> None:
```
with
```python
        camera_b: str,
        window_size: str = "1500x950",
        touch: bool = False,
    ) -> None:
```
add `self._touch = touch` right after `self.camera_b = camera_b`, and replace `self.root.minsize(1100, 760)` with
```python
        self.root.minsize(*((640, 400) if touch else (1100, 760)))
```

3c. `_build_layout`: replace the block from `        style = ttk.Style()` through the `Quit (q)` button line (i.e. lines that build `style` and the toolbar) with:

```python
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(fill=tk.X)
        if self._touch:
            apply_touch_style(self.root)
            self._build_touch_toolbar(toolbar)
        else:
            style = ttk.Style()
            if "clam" in style.theme_names():
                style.theme_use("clam")
            self._build_desktop_toolbar(toolbar)
```
and add these methods to the class (directly after `_build_layout`):

```python
    def _build_desktop_toolbar(self, toolbar: ttk.Frame) -> None:
        ttk.Button(toolbar, text="Refit Now (r)", command=self.request_refit).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Capture More (u)", command=self.request_top_up).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Continue (c)", command=self.request_continue).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Discard Worst (w)", command=self.request_discard_worst).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(toolbar, text="Discard & Restart (d)", command=self.request_discard).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self._pause_button = ttk.Button(
            toolbar, text="Pause Auto-Capture (space)", command=self.toggle_pause
        )
        self._pause_button.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Open Captures Folder", command=self.open_captures_folder).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(toolbar, text="Quit (q)", command=self.request_quit).pack(side=tk.RIGHT)

    def _build_touch_toolbar(self, toolbar: ttk.Frame) -> None:
        """Two rows of big buttons; the destructive ones need a second tap."""
        top, bottom = ttk.Frame(toolbar), ttk.Frame(toolbar)
        top.pack(fill=tk.X, pady=(0, 6))
        bottom.pack(fill=tk.X)

        def add(row: ttk.Frame, text: str, command, confirm: bool = False) -> ttk.Button:
            button = ttk.Button(row, text=text, style="Touch.TButton")
            button.configure(command=self._confirming(button, text, command) if confirm else command)
            button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
            return button

        add(top, "Refit Now", self.request_refit)
        add(top, "Capture More", self.request_top_up)
        add(top, "Continue", self.request_continue)
        self._pause_button = add(top, "Pause Auto-Capture", self.toggle_pause)
        add(bottom, "Discard Worst", self.request_discard_worst, confirm=True)
        add(bottom, "Discard & Restart", self.request_discard, confirm=True)
        add(bottom, "Quit", self.request_quit)

    def _confirming(self, button: ttk.Button, label: str, action):
        """Wrap ``action`` so the first tap only relabels the button; a second tap runs it."""
        gate = ConfirmGate()

        def handler() -> None:
            if gate.press():
                button.configure(text=label)
                action()
                return
            button.configure(text="Tap again to confirm")
            self.root.after(int(CONFIRM_TIMEOUT_S * 1000) + 100,
                            lambda: button.configure(text=label))
        return handler
```

3d. Hide the long keyboard-oriented hint in touch mode — wrap the `hint = ttk.Frame(...)` … `.pack(anchor="w")` block:
```python
        if not self._touch:
            hint = ttk.Frame(self.root, padding=(10, 0, 10, 6))
            hint.pack(fill=tk.X)
            ttk.Label( ... unchanged ... ).pack(anchor="w")
```
(indent the existing three statements under the `if`; content unchanged).

3e. `toggle_pause`: replace the label line with
```python
        suffix = "" if self._touch else " (space)"
        label = f"Resume Auto-Capture{suffix}" if self._paused else f"Pause Auto-Capture{suffix}"
```

- [ ] **Step 4: Implement — edit `calibrate_live.py`**

Replace
```python
    parser.add_argument("--no-preview", action="store_true", help="Headless mode: terminal keys only.")
    return parser.parse_args()
```
with
```python
    parser.add_argument("--no-preview", action="store_true", help="Headless mode: terminal keys only.")
    parser.add_argument("--touch", action="store_true",
                        help="Touchscreen mode: big two-row toolbar, no keyboard hints, "
                             "tap-again-to-confirm on the discard buttons.")
    return parser.parse_args()
```
and replace `gui = LiveCaptureGUI(captures_dir, args.camera_a, args.camera_b)` with
```python
            gui = LiveCaptureGUI(captures_dir, args.camera_a, args.camera_b, touch=args.touch)
```

- [ ] **Step 5: Run to verify**

Run:
```bash
python -m unittest tests.test_live_gui_touch -v
python -c "import live_gui, calibrate_live; print('imports ok')"
python calibrate_live.py --help | grep -A2 -- "--touch"
```
Expected: PASS (or SKIP without a display); `imports ok`; help printed. **Not exercised:** live calibration with real cameras.

- [ ] **Step 6: Commit**

```bash
git add live_gui.py calibrate_live.py tests/test_live_gui_touch.py
git commit -m "feat: --touch for calibrate_live (two-row toolbar, confirm on discard)"
```

---

### Task 8: `menu_logic.py` part A — stages, sessions, commands

**Files:**
- Create: `menu_logic.py`, `tests/test_menu_logic.py`

**Interfaces:**
- Produces (consumed by `menu.py`, Task 10):
  - `PROJECT_ROOT`, `CAMERA_FILES = ("rgb_cam1.jpg", "rgb_cam2.jpg")`, `UNNAMED_FOLDER = ""`
  - `Session(path, folder, name, mtime)` with `.display`
  - `Stage(key, title, script, touch=False, needs_session=False, window=False, naming=False, result=False)`; `STAGES: Dict[str, Stage]`, `STAGE_ORDER: List[str]`
  - `Page(items, index, count)`; `paginate(items, page, size) -> Page`
  - `find_sessions(captures_dir, hidden=(), cameras=CAMERA_FILES) -> List[Session]` (newest first), `latest_session(sessions) -> Optional[Session]`, `group_by_folder(sessions) -> List[Tuple[str, List[Session]]]`
  - `load_menu_config(project_root) -> dict`, `captures_dir_from(config, project_root)`, `results_dir_from(config, project_root)`, `hidden_capture_dirs(config, project_root) -> List[Path]`
  - `sanitize_session_name(text) -> str`, `window_for_screen(width, height) -> Tuple[int, int]`
  - `storage_problem(captures_dir) -> Optional[str]` — plain-language message when the captures folder and its parent are both missing (an unplugged USB drive), else `None`
  - `build_command(stage, python, project_root, session=None, name=None, window=None) -> List[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_menu_logic.py`:

```python
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from menu_logic import (
    PROJECT_ROOT, STAGE_ORDER, STAGES, UNNAMED_FOLDER, build_command, captures_dir_from,
    find_sessions, group_by_folder, hidden_capture_dirs, latest_session, paginate,
    sanitize_session_name, storage_problem, window_for_screen,
)


def make_session(root: Path, *parts: str, mtime: float = 1000.0) -> Path:
    path = root.joinpath(*parts)
    path.mkdir(parents=True)
    for name in ("rgb_cam1.jpg", "rgb_cam2.jpg"):
        (path / name).write_bytes(b"x")
    os.utime(path, (mtime, mtime))
    return path


class FindSessionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_session(self.root, "20260916_154103", mtime=1000)                 # unnamed capture
        make_session(self.root, "wound_a", "20260916_100000", mtime=2000)
        make_session(self.root, "wound_a", "20260917_090000", mtime=3000)
        make_session(self.root, "stereo", "s1", mtime=9000)                     # hidden folder
        (self.root / "empty").mkdir()
        half = self.root / "half" / "x"
        half.mkdir(parents=True)
        (half / "rgb_cam1.jpg").write_bytes(b"x")                               # only one camera

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_sessions_newest_first_and_skips_hidden_and_incomplete(self):
        sessions = find_sessions(self.root, hidden=[self.root / "stereo"])
        self.assertEqual([s.name for s in sessions],
                         ["20260917_090000", "20260916_100000", "20260916_154103"])

    def test_folder_is_blank_for_unnamed_captures(self):
        by_name = {s.name: s.folder for s in find_sessions(self.root)}
        self.assertEqual(by_name["20260916_154103"], UNNAMED_FOLDER)
        self.assertEqual(by_name["20260917_090000"], "wound_a")

    def test_latest_session_and_empty(self):
        sessions = find_sessions(self.root, hidden=[self.root / "stereo"])
        self.assertEqual(latest_session(sessions).name, "20260917_090000")
        self.assertIsNone(latest_session([]))

    def test_groups_are_ordered_by_their_newest_session(self):
        groups = group_by_folder(find_sessions(self.root, hidden=[self.root / "stereo"]))
        self.assertEqual([name for name, _ in groups], ["wound_a", UNNAMED_FOLDER])
        self.assertEqual(len(groups[0][1]), 2)

    def test_missing_captures_dir_is_empty(self):
        self.assertEqual(find_sessions(self.root / "nope"), [])

    def test_display_formats_a_timestamp_name(self):
        sessions = find_sessions(self.root, hidden=[self.root / "stereo"])
        unnamed = [s for s in sessions if s.folder == UNNAMED_FOLDER][0]
        self.assertEqual(unnamed.display, "16 Sep 2026  15:41:03")
        named = [s for s in sessions if s.folder == "wound_a"][0]
        self.assertTrue(named.display.startswith("wound_a / "))


class ConfigHelpersTest(unittest.TestCase):
    def test_hidden_capture_dirs_come_from_the_calibration_config(self):
        config = {"geometric_calibration": {
            "intrinsics_captures": "captures/stereo",
            "stereo_captures": ["captures/stereo", "captures/other"],
            "cross_validation_captures": "captures/cross-validation",
        }}
        hidden = hidden_capture_dirs(config, Path("/proj"))
        self.assertIn(Path("/proj/captures/stereo"), hidden)
        self.assertIn(Path("/proj/captures/cross-validation"), hidden)
        self.assertIn(Path("/proj/captures/other"), hidden)

    def test_no_calibration_config_hides_nothing(self):
        self.assertEqual(hidden_capture_dirs({}, Path("/proj")), [])

    def test_an_absolute_output_dir_is_kept_and_the_default_is_project_relative(self):
        self.assertEqual(captures_dir_from({"output_dir": "/mnt/usbdrive/captures"}, Path("/proj")),
                         Path("/mnt/usbdrive/captures"))
        self.assertEqual(captures_dir_from({}, Path("/proj")), Path("/proj/captures"))


class StorageProblemTest(unittest.TestCase):
    def test_an_existing_folder_or_one_whose_parent_exists_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(storage_problem(Path(tmp)))
            self.assertIsNone(storage_problem(Path(tmp) / "captures"))  # will be created there

    def test_a_missing_mount_is_reported_with_the_path_and_the_usb_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "usbdrive" / "captures"
            message = storage_problem(missing)
            self.assertIn(str(missing), message)
            self.assertIn("USB drive", message)


class NameAndWindowTest(unittest.TestCase):
    def test_sanitize_keeps_only_safe_characters_and_caps_length(self):
        self.assertEqual(sanitize_session_name("wound a/../x!"), "woundax")
        self.assertEqual(sanitize_session_name("Pig_skin-2"), "Pig_skin-2")
        self.assertEqual(len(sanitize_session_name("a" * 100)), 40)

    def test_window_fits_inside_the_screen_with_a_floor(self):
        self.assertEqual(window_for_screen(1920, 1080), (1880, 940))
        self.assertEqual(window_for_screen(1280, 720), (1240, 580))
        self.assertEqual(window_for_screen(800, 480), (960, 540))


class PaginateTest(unittest.TestCase):
    def test_slices_and_clamps(self):
        items = list(range(10))
        first = paginate(items, 0, 4)
        self.assertEqual((first.items, first.index, first.count), ([0, 1, 2, 3], 0, 3))
        last = paginate(items, 99, 4)
        self.assertEqual((last.items, last.index), ([8, 9], 2))
        self.assertEqual(paginate(items, -5, 4).index, 0)

    def test_empty_list_is_one_empty_page(self):
        page = paginate([], 0, 6)
        self.assertEqual((page.items, page.index, page.count), ([], 0, 1))


class BuildCommandTest(unittest.TestCase):
    ROOT = Path("/proj")

    def test_capture_with_and_without_a_name(self):
        named = build_command(STAGES["capture"], "py", self.ROOT, name="wound_a")
        self.assertEqual(named, ["py", "/proj/capture_pipeline.py", "--touch", "--output", "wound_a"])
        self.assertEqual(build_command(STAGES["capture"], "py", self.ROOT),
                         ["py", "/proj/capture_pipeline.py", "--touch"])

    def test_calibrate_takes_no_session(self):
        self.assertEqual(build_command(STAGES["calibrate"], "py", self.ROOT),
                         ["py", "/proj/calibrate_live.py", "--touch"])

    def test_measure_tools_get_session_and_window(self):
        cmd = build_command(STAGES["depth"], "py", self.ROOT,
                            session=Path("/data/s1"), window=(1240, 580))
        self.assertEqual(cmd, ["py", "/proj/measure_wound_depth.py", "--touch",
                               "--session", "/data/s1", "--window", "1240", "580"])

    def test_register_has_no_touch_flag_and_no_window(self):
        cmd = build_command(STAGES["register"], "py", self.ROOT,
                            session=Path("/data/s1"), window=(1240, 580))
        self.assertEqual(cmd, ["py", "/proj/register_features.py", "--session", "/data/s1"])

    def test_a_session_stage_without_a_session_is_an_error(self):
        with self.assertRaises(ValueError):
            build_command(STAGES["length"], "py", self.ROOT)


class StageTableTest(unittest.TestCase):
    def test_every_stage_script_exists_and_touch_stages_accept_the_flag(self):
        for key in STAGE_ORDER:
            stage = STAGES[key]
            script = PROJECT_ROOT / stage.script
            self.assertTrue(script.is_file(), f"{stage.script} is missing")
            if stage.touch:
                self.assertIn('"--touch"', script.read_text(encoding="utf-8"),
                              f"{stage.script} has no --touch flag")
            if stage.window:
                self.assertIn('"--window"', script.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m unittest tests.test_menu_logic -v`
Expected: `ModuleNotFoundError: No module named 'menu_logic'`

- [ ] **Step 3: Implement**

Create `menu_logic.py`:

```python
"""Display-free logic behind menu.py: which stages exist, how to launch them, where captures
and results live. Kept free of Tk and of any camera import so it is unit-testable anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CAMERA_FILES = ("rgb_cam1.jpg", "rgb_cam2.jpg")
UNNAMED_FOLDER = ""
MAX_NAME_LEN = 40
SCREEN_MARGIN_W = 40
SCREEN_MARGIN_H = 140  # top bar + window title bar + slack
MIN_WINDOW = (960, 540)
_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    script: str
    touch: bool = False           # script accepts --touch
    needs_session: bool = False   # script needs --session <dir>
    window: bool = False          # script accepts --window W H
    naming: bool = False          # operator may name the capture folder
    result: bool = False          # show the Result screen afterwards


STAGES: Dict[str, Stage] = {
    "capture": Stage("capture", "Take images", "capture_pipeline.py", touch=True, naming=True),
    "calibrate": Stage("calibrate", "Calibrate cameras", "calibrate_live.py", touch=True),
    "depth": Stage("depth", "Measure wound depth", "measure_wound_depth.py",
                   touch=True, needs_session=True, window=True, result=True),
    "length": Stage("length", "Measure length", "measure_points.py",
                    touch=True, needs_session=True, window=True, result=True),
    "register": Stage("register", "Register images", "register_features.py",
                      needs_session=True, result=True),
}
STAGE_ORDER: List[str] = ["capture", "calibrate", "depth", "length", "register"]


def build_command(stage: Stage, python: str, project_root: Path,
                  session: Optional[Path] = None, name: Optional[str] = None,
                  window: Optional[Tuple[int, int]] = None) -> List[str]:
    command = [python, str(project_root / stage.script)]
    if stage.touch:
        command.append("--touch")
    if stage.needs_session:
        if session is None:
            raise ValueError(f"{stage.title} needs a capture session")
        command += ["--session", str(session)]
    if stage.window and window is not None:
        command += ["--window", str(window[0]), str(window[1])]
    if stage.naming and name:
        command += ["--output", name]
    return command


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_menu_config(project_root: Path) -> Dict[str, Any]:
    with (project_root / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def captures_dir_from(config: Dict[str, Any], project_root: Path) -> Path:
    return (project_root / config.get("output_dir", "captures")).resolve()


def results_dir_from(config: Dict[str, Any], project_root: Path) -> Path:
    registration = config.get("registration") or {}
    return (project_root / registration.get("output_dir", "registration/results")).resolve()


def hidden_capture_dirs(config: Dict[str, Any], project_root: Path) -> List[Path]:
    """Calibration capture folders (hundreds of sessions) that the measure picker must hide."""
    geo = config.get("geometric_calibration") or {}
    names = [geo.get("intrinsics_captures"), geo.get("cross_validation_captures")]
    names += list(geo.get("stereo_captures") or [])
    return [(project_root / name).resolve() for name in names if name]


def storage_problem(captures_dir: Path) -> Optional[str]:
    """A plain-language message if the captures folder cannot be used, else None.

    ``output_dir`` may point at removable storage (e.g. a USB drive at /mnt/usbdrive). When
    that is not mounted neither the folder nor its parent exists, and creating it with
    ``mkdir(parents=True)`` could silently write to the internal disk instead.
    """
    if captures_dir.exists() or captures_dir.parent.is_dir():
        return None
    return (f"The captures folder {captures_dir} was not found. If captures are saved to a "
            "USB drive, check that it is plugged in, then try again.")


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Session:
    path: Path
    folder: str   # UNNAMED_FOLDER for captures/<timestamp>/, else the folder name
    name: str     # the timestamp directory name
    mtime: float

    @property
    def display(self) -> str:
        try:
            stamp = datetime.strptime(self.name[:15], "%Y%m%d_%H%M%S").strftime("%d %b %Y  %H:%M:%S")
        except ValueError:
            stamp = self.name
        return f"{self.folder} / {stamp}" if self.folder else stamp


def _is_session(directory: Path, cameras: Sequence[str]) -> bool:
    return all((directory / camera).is_file() for camera in cameras)


def find_sessions(captures_dir: Path, hidden: Sequence[Path] = (),
                  cameras: Sequence[str] = CAMERA_FILES) -> List[Session]:
    """Capture sessions at most two levels under ``captures_dir``, newest first.

    ``captures/<timestamp>/`` (unnamed) and ``captures/<name>/<timestamp>/`` (named) both
    count. ``hidden`` folders are skipped without being listed.
    """
    if not captures_dir.is_dir():
        return []
    skip = {Path(path).resolve() for path in hidden}
    sessions: List[Session] = []
    for child in captures_dir.iterdir():
        if not child.is_dir() or child.resolve() in skip:
            continue
        if _is_session(child, cameras):
            sessions.append(Session(child, UNNAMED_FOLDER, child.name, child.stat().st_mtime))
            continue
        for sub in child.iterdir():
            if sub.is_dir() and _is_session(sub, cameras):
                sessions.append(Session(sub, child.name, sub.name, sub.stat().st_mtime))
    sessions.sort(key=lambda s: (s.mtime, s.name), reverse=True)
    return sessions


def latest_session(sessions: Sequence[Session]) -> Optional[Session]:
    return max(sessions, key=lambda s: (s.mtime, s.name)) if sessions else None


def group_by_folder(sessions: Sequence[Session]) -> List[Tuple[str, List[Session]]]:
    ordered = sorted(sessions, key=lambda s: (s.mtime, s.name), reverse=True)
    groups: Dict[str, List[Session]] = {}
    for session in ordered:
        groups.setdefault(session.folder, []).append(session)
    return sorted(groups.items(), key=lambda item: item[1][0].mtime, reverse=True)


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
class Page(NamedTuple):
    items: List[Any]
    index: int
    count: int


def paginate(items: Sequence[Any], page: int, size: int) -> Page:
    count = max(1, -(-len(items) // size))
    index = min(max(page, 0), count - 1)
    return Page(list(items[index * size:(index + 1) * size]), index, count)


def sanitize_session_name(text: str) -> str:
    return _NAME_UNSAFE.sub("", text)[:MAX_NAME_LEN]


def window_for_screen(width: int, height: int) -> Tuple[int, int]:
    """A cv2 window size that fits inside the screen (an oversized one never gets input)."""
    return (max(MIN_WINDOW[0], width - SCREEN_MARGIN_W),
            max(MIN_WINDOW[1], height - SCREEN_MARGIN_H))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m unittest tests.test_menu_logic -v`
Expected: all PASS (the `StageTableTest` also confirms Tasks 3–7 added every `--touch`/`--window` flag the menu relies on).

- [ ] **Step 5: Commit**

```bash
git add menu_logic.py tests/test_menu_logic.py
git commit -m "feat: menu_logic stages, session discovery and command building"
```

---

### Task 9: `menu_logic.py` part B — `StageRun`, errors, results, shortcut

**Files:**
- Modify: `menu_logic.py` (append), `.gitignore`
- Create: `tests/test_menu_runtime.py`

**Interfaces:**
- Produces (consumed by `menu.py`):
  - `StageRun(command, log_path, cwd, term_after_s=5.0, kill_after_s=8.0, clock=time.monotonic)` with `.start()`, `.poll() -> Optional[int]`, `.stop()`, `.stopped`, `.elapsed() -> float`, `.log_text() -> str`, `.started_at` (epoch seconds)
  - `GENERIC_FAILURE`, `friendly_error(log_text) -> str`, `tail_lines(text, count=20) -> str`
  - `Report(report_path, image_path)`, `find_newest_report(results_dir, since) -> Optional[Report]`
  - `desktop_entry_text(python, menu_script, project_root) -> str`, `install_shortcut(python, menu_script, project_root, home=None) -> List[Path]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_menu_runtime.py`:

```python
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from menu_logic import (
    GENERIC_FAILURE, StageRun, desktop_entry_text, find_newest_report, friendly_error,
    install_shortcut, tail_lines,
)


def wait_for_exit(run: StageRun, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code = run.poll()
        if code is not None:
            return code
        time.sleep(0.05)
    raise AssertionError("child did not exit in time")


class StageRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, code: str, **kwargs) -> StageRun:
        run = StageRun([sys.executable, "-c", code], self.root / "logs" / "x.log", self.root, **kwargs)
        run.start()
        return run

    def test_success_records_output_in_the_log(self):
        run = self._run("print('hello from stage')")
        self.assertEqual(wait_for_exit(run), 0)
        self.assertIn("hello from stage", run.log_text())
        self.assertFalse(run.stopped)

    def test_failure_returns_the_exit_code_and_captures_stderr(self):
        run = self._run("import sys; sys.stderr.write('boom\\n'); sys.exit(3)")
        self.assertEqual(wait_for_exit(run), 3)
        self.assertIn("boom", run.log_text())

    def test_stop_interrupts_a_long_running_child(self):
        run = self._run("import time; time.sleep(60)")
        run.stop()
        self.assertTrue(run.stopped)
        wait_for_exit(run)

    def test_stop_escalates_when_the_child_ignores_interrupts(self):
        run = self._run("import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)",
                        term_after_s=0.3, kill_after_s=0.6)
        time.sleep(0.5)  # let the child install its handler
        run.stop()
        self.assertNotEqual(wait_for_exit(run), 0)

    def test_elapsed_and_started_at(self):
        run = self._run("import time; time.sleep(0.3)")
        self.assertLess(abs(run.started_at - time.time()), 5)
        wait_for_exit(run)
        self.assertGreaterEqual(run.elapsed(), 0.0)


class FriendlyErrorTest(unittest.TestCase):
    def test_known_failures_get_plain_language(self):
        self.assertIn("RGB camera", friendly_error("Pipeline error: Cannot open rgb_cam1 at /dev/video0"))
        self.assertIn("thermal", friendly_error("RuntimeError: evo_irimager_usb_init failed with code -1"))
        self.assertIn("FLIR", friendly_error("RuntimeError: No FLIR/Spinnaker cameras found"))
        self.assertIn("calibrated", friendly_error("No stereo extrinsics at /x/extrinsics.json."))
        self.assertIn("missing a camera image", friendly_error("/x/s1 is missing rgb_cam2.jpg."))

    def test_unknown_failure_falls_back_to_the_generic_message(self):
        self.assertEqual(friendly_error("something odd"), GENERIC_FAILURE)

    def test_tail_lines(self):
        text = "\n".join(str(i) for i in range(50)) + "\n"
        self.assertEqual(tail_lines(text, 3), "47\n48\n49")
        self.assertEqual(tail_lines("", 3), "")


class FindNewestReportTest(unittest.TestCase):
    def _write(self, path: Path, mtime: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
        os.utime(path, (mtime, mtime))

    def test_picks_the_newest_report_since_the_run_started_with_its_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root / "old" / "wound_depth" / "report.txt", 500)
            self._write(root / "s1" / "wound_depth" / "report.txt", 2000)
            self._write(root / "s2" / "point_measurements" / "report.txt", 2500)
            self._write(root / "s2" / "point_measurements" / "measured.jpg", 2500)
            found = find_newest_report(root, since=1000)
            self.assertEqual(found.report_path, root / "s2" / "point_measurements" / "report.txt")
            self.assertEqual(found.image_path, root / "s2" / "point_measurements" / "measured.jpg")

    def test_registration_report_and_preview_are_recognised(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root / "s3" / "report_features.txt", 3000)
            self._write(root / "s3" / "preview_features.jpg", 3000)
            found = find_newest_report(root, since=1000)
            self.assertEqual(found.report_path.name, "report_features.txt")
            self.assertEqual(found.image_path.name, "preview_features.jpg")

    def test_nothing_new_or_missing_dir_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root / "s1" / "report.txt", 500)
            self.assertIsNone(find_newest_report(root, since=1000))
            self.assertIsNone(find_newest_report(root / "missing", since=0))


class ShortcutTest(unittest.TestCase):
    def test_desktop_entry_has_absolute_exec_and_path(self):
        text = desktop_entry_text("/venv/bin/python", Path("/proj/menu.py"), Path("/proj"))
        self.assertIn('Exec="/venv/bin/python" "/proj/menu.py"', text)
        self.assertIn("Path=/proj", text)
        self.assertIn("Terminal=false", text)
        self.assertTrue(text.startswith("[Desktop Entry]"))

    def test_install_writes_executable_launchers(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = install_shortcut("/venv/bin/python", Path("/proj/menu.py"), Path("/proj"), home=Path(tmp))
            self.assertEqual({p.parent.name for p in written}, {"Desktop", "applications"})
            for path in written:
                self.assertTrue(path.is_file())
                self.assertTrue(os.access(path, os.X_OK))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m unittest tests.test_menu_runtime -v`
Expected: `ImportError: cannot import name 'GENERIC_FAILURE' from 'menu_logic'`

- [ ] **Step 3: Implement — append to `menu_logic.py`**

Extend the imports at the top of `menu_logic.py`:
```python
import os
import signal
import subprocess
import time
from typing import Callable
```
(merge with the existing `from typing import ...` line: add `Callable`).

Append:

```python
# --------------------------------------------------------------------------- #
# Running a stage
# --------------------------------------------------------------------------- #
class StageRun:
    """One stage as a subprocess: output to a log file, polled (never blocked on) by the UI.

    The child runs in its own process group so Stop reaches everything it spawned. Stop
    sends SIGINT first (Python ``finally`` blocks release the cameras), then escalates to
    SIGTERM and SIGKILL if the child ignores it.
    """

    def __init__(self, command: List[str], log_path: Path, cwd: Path,
                 term_after_s: float = 5.0, kill_after_s: float = 8.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._command = command
        self._log_path = log_path
        self._cwd = cwd
        self._term_after_s = term_after_s
        self._kill_after_s = kill_after_s
        self._clock = clock
        self._proc: Optional[subprocess.Popen] = None
        self._t0 = 0.0
        self._stop_at: Optional[float] = None
        self._termed = False
        self._killed = False
        self.started_at = 0.0

    @property
    def stopped(self) -> bool:
        return self._stop_at is not None

    def start(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("wb") as log:
            self._proc = subprocess.Popen(
                self._command, cwd=str(self._cwd), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        self._t0 = self._clock()
        self.started_at = time.time()

    def _signal(self, sig: int) -> None:
        if self._proc is None:
            return
        try:
            os.killpg(self._proc.pid, sig)  # pgid == pid because of start_new_session
        except ProcessLookupError:
            pass

    def stop(self) -> None:
        if self._stop_at is None:
            self._stop_at = self._clock()
            self._signal(signal.SIGINT)

    def poll(self) -> Optional[int]:
        """Exit code, or None while running. Also escalates a Stop the child is ignoring."""
        if self._proc is None:
            return None
        code = self._proc.poll()
        if code is None and self._stop_at is not None:
            waited = self._clock() - self._stop_at
            if waited >= self._kill_after_s and not self._killed:
                self._killed = True
                self._signal(signal.SIGKILL)
            elif waited >= self._term_after_s and not self._termed:
                self._termed = True
                self._signal(signal.SIGTERM)
        return code

    def elapsed(self) -> float:
        return self._clock() - self._t0

    def log_text(self) -> str:
        try:
            return self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


# --------------------------------------------------------------------------- #
# Failure text and results
# --------------------------------------------------------------------------- #
GENERIC_FAILURE = "Something went wrong. Tap Show details for the technical message."

# (substring found in the log, plain-language message); first match wins. The substrings are
# the real messages raised in cameras/*.py, capture_pipeline.py and the measure tools.
_FAILURE_HINTS: Tuple[Tuple[str, str], ...] = (
    ("Cannot open rgb_cam", "An RGB camera could not be opened. Check both USB cables, then try again."),
    ("evo_irimager_usb_init failed", "The thermal camera could not be started. Check its USB cable and power, then try again."),
    ("Thermal config not found", "The thermal camera settings file is missing. Ask whoever maintains the rig."),
    ("No FLIR/Spinnaker cameras found", "The FLIR camera was not found. Check its USB cable, then try again."),
    ("Failed to grab from one or both RGB cameras", "The cameras stopped sending images. Check the USB cables, then try again."),
    ("No stereo extrinsics", "This rig has not been calibrated yet. Run Calibrate cameras first."),
    ("is missing rgb_cam", "That capture is missing a camera image. Pick a different capture."),
    ("Failed to decode images", "That capture's images could not be read. Pick a different capture."),
)


def friendly_error(log_text: str) -> str:
    for needle, message in _FAILURE_HINTS:
        if needle in log_text:
            return message
    return GENERIC_FAILURE


def tail_lines(text: str, count: int = 20) -> str:
    return "\n".join(text.rstrip("\n").splitlines()[-count:])


REPORT_GLOB = "report*.txt"
REPORT_NAMES = ("report.txt", "report_features.txt")
RESULT_IMAGE_NAMES = ("annotated.jpg", "measured.jpg", "preview_features.jpg")


@dataclass(frozen=True)
class Report:
    report_path: Path
    image_path: Optional[Path]


def find_newest_report(results_dir: Path, since: float) -> Optional[Report]:
    """The newest report written at or after ``since`` (epoch s), with the image beside it."""
    if not results_dir.is_dir():
        return None
    newest: Optional[Tuple[float, Path]] = None
    for path in results_dir.rglob(REPORT_GLOB):
        if path.name not in REPORT_NAMES or not path.is_file():
            continue
        mtime = path.stat().st_mtime
        if mtime >= since - 1.0 and (newest is None or mtime > newest[0]):
            newest = (mtime, path)
    if newest is None:
        return None
    folder = newest[1].parent
    image = next((folder / name for name in RESULT_IMAGE_NAMES if (folder / name).is_file()), None)
    return Report(newest[1], image)


# --------------------------------------------------------------------------- #
# Desktop shortcut
# --------------------------------------------------------------------------- #
def desktop_entry_text(python: str, menu_script: Path, project_root: Path) -> str:
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        "Name=FullPipeline",
        "Comment=Capture, calibrate and measure",
        f'Exec="{python}" "{menu_script}"',
        f"Path={project_root}",
        "Terminal=false",
        "Categories=Utility;",
        "",
    ])


def install_shortcut(python: str, menu_script: Path, project_root: Path,
                     home: Optional[Path] = None) -> List[Path]:
    home = home or Path.home()
    text = desktop_entry_text(python, menu_script, project_root)
    written: List[Path] = []
    for directory in (home / "Desktop", home / ".local" / "share" / "applications"):
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "FullPipeline.desktop"
        target.write_text(text, encoding="utf-8")
        target.chmod(0o755)
        written.append(target)
    return written
```

In `.gitignore`, add under the "Personal Claude Code settings" block (or at the end):
```
# Menu stage logs (one file per launched stage)
logs/
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m unittest tests.test_menu_runtime tests.test_menu_logic -v`
Expected: all PASS (the process tests take ~1–2 s).

- [ ] **Step 5: Commit**

```bash
git add menu_logic.py tests/test_menu_runtime.py .gitignore
git commit -m "feat: menu_logic StageRun, friendly errors, result lookup and desktop shortcut"
```

---

### Task 10: `menu.py` screens, smoke test, README, final verification

**Files:**
- Create: `menu.py`, `tests/test_menu_smoke.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything from Tasks 1, 8, 9.
- Produces: `MenuApp(project_root=PROJECT_ROOT, maximize=True)` with screen methods `show_home(notice="")`, `show_session_picker(stage, folder=None, page=0)`, `show_name_keyboard(stage)`, `show_running(stage)`, `show_error(stage, message, details)`, `show_result(stage, report)`, `start_stage(stage, session=None, name=None)`, `open_output_folder()`; script entry `python menu.py [--install-shortcut]`.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_menu_smoke.py` (skipped without a display; it opens a real Tk window for a moment):

```python
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from menu_logic import STAGES, Report


@unittest.skipUnless(os.environ.get("DISPLAY"), "needs a display")
class MenuSmokeTest(unittest.TestCase):
    def setUp(self):
        from menu import MenuApp
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "config.yaml").write_text("output_dir: captures\n", encoding="utf-8")
        for stamp in ("20260916_100000", "20260917_090000"):
            session = root / "captures" / "wound_a" / stamp
            session.mkdir(parents=True)
            for camera in ("rgb_cam1.jpg", "rgb_cam2.jpg"):
                cv2.imwrite(str(session / camera), np.full((60, 80, 3), 128, np.uint8))
        self.root_dir = root
        self.app = MenuApp(project_root=root, maximize=False)

    def tearDown(self):
        self.app.root.destroy()
        self.tmp.cleanup()

    def _texts(self):
        out = []

        def walk(widget):
            for child in widget.winfo_children():
                try:
                    out.append(str(child.cget("text")))
                except Exception:
                    pass
                walk(child)
        walk(self.app.body)
        return out

    def test_home_has_a_tile_per_stage_plus_output_folder_and_quit(self):
        self.app.root.update()
        texts = self._texts()
        for key in ("capture", "calibrate", "depth", "length", "register"):
            self.assertIn(STAGES[key].title, texts)
        self.assertIn("Output folder", texts)
        self.assertIn("Quit", texts)

    def test_every_screen_builds(self):
        app = self.app
        app.show_session_picker(STAGES["depth"])
        app.root.update()
        self.assertTrue(any(t.startswith("Latest capture") for t in self._texts()))
        app.show_session_picker(STAGES["depth"], folder="wound_a")
        app.root.update()
        app.show_name_keyboard(STAGES["capture"])
        app.root.update()
        self.assertIn("Skip (auto-name)", self._texts())
        app.show_error(STAGES["capture"], "An RGB camera could not be opened.", "log line")
        app.root.update()
        self.assertIn("Retry", self._texts())
        report = self.root_dir / "report.txt"
        report.write_text("DEPTH 3.0 mm\n", encoding="utf-8")
        app.show_result(STAGES["depth"], Report(report, None))
        app.root.update()
        app.show_result(STAGES["depth"], None)
        app.root.update()
        self.assertIn("Done", self._texts())

    def test_empty_captures_shows_a_message_not_a_crash(self):
        from menu import MenuApp
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text("output_dir: captures\n", encoding="utf-8")
            app = MenuApp(project_root=Path(tmp), maximize=False)
            try:
                app.show_session_picker(STAGES["length"])
                app.root.update()
                texts = []

                def walk(widget):
                    for child in widget.winfo_children():
                        try:
                            texts.append(str(child.cget("text")))
                        except Exception:
                            pass
                        walk(child)
                walk(app.body)
                self.assertIn("No captures yet. Use Take images first.", texts)
            finally:
                app.root.destroy()

    def test_a_missing_usb_drive_is_reported_instead_of_being_silently_created(self):
        from menu import MenuApp
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text(
                f"output_dir: {Path(tmp) / 'unplugged' / 'captures'}\n", encoding="utf-8")
            app = MenuApp(project_root=Path(tmp), maximize=False)
            try:
                app.choose(STAGES["capture"])
                app.root.update()
                self.assertFalse((Path(tmp) / "unplugged").exists())
                texts = []

                def walk(widget):
                    for child in widget.winfo_children():
                        try:
                            texts.append(str(child.cget("text")))
                        except Exception:
                            pass
                        walk(child)
                walk(app.body)
                self.assertTrue(any("USB drive" in t for t in texts))
            finally:
                app.root.destroy()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m unittest tests.test_menu_smoke -v`
Expected: `ModuleNotFoundError: No module named 'menu'` (or SKIP without a display — then create `menu.py` and rely on Step 5's import check and a manual run on a machine with a display).

- [ ] **Step 3: Implement `menu.py`**

Create `menu.py`:

```python
#!/usr/bin/env python3
"""Touchscreen home screen: launch capture, calibration, measurement and registration.

Every stage is an existing script run as a subprocess (see menu_logic.STAGES); this process
only draws screens, so it never imports a camera SDK and a crashing stage cannot take it
down. Only one stage runs at a time, so two stages can never fight over the cameras.

    python menu.py                     # open the menu
    python menu.py --install-shortcut  # put a launcher icon on the desktop, then exit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, List, Optional, Tuple

from PIL import Image, ImageTk

from menu_logic import (
    PROJECT_ROOT, STAGE_ORDER, STAGES, Page, Report, Session, Stage, StageRun,
    build_command, captures_dir_from, find_newest_report, find_sessions, friendly_error,
    group_by_folder, hidden_capture_dirs, install_shortcut, latest_session,
    load_menu_config, paginate, results_dir_from, sanitize_session_name, storage_problem,
    tail_lines, window_for_screen,
)
from touch_controls import apply_touch_style

POLL_MS = 300
FOLDERS_PER_PAGE = 8
SESSIONS_PER_PAGE = 6
NAME_CHIPS = 6
THUMB_BOX = (260, 195)
KEY_ROWS = ("1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm_-")


class MenuApp:
    def __init__(self, project_root: Path = PROJECT_ROOT, maximize: bool = True) -> None:
        self.project_root = project_root
        config = load_menu_config(project_root)
        self.captures_dir = captures_dir_from(config, project_root)
        self.results_dir = results_dir_from(config, project_root)
        self.hidden_dirs = hidden_capture_dirs(config, project_root)

        self.root = tk.Tk()
        self.root.title("FullPipeline")
        self.root.geometry("1280x720")
        apply_touch_style(self.root, maximize=maximize)
        self.body = ttk.Frame(self.root, padding=24)
        self.body.pack(fill=tk.BOTH, expand=True)

        self._photos: List[ImageTk.PhotoImage] = []
        self._run: Optional[StageRun] = None
        self._last_launch: Optional[Tuple[Stage, Optional[Path], Optional[str]]] = None
        self._elapsed = tk.StringVar(value="")
        self._name = tk.StringVar(value="")
        self.show_home()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _reset(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()
        self._photos.clear()

    def _title(self, text: str) -> None:
        ttk.Label(self.body, text=text, style="Title.TLabel").pack(anchor="w", pady=(0, 12))

    def _button(self, parent, text: str, command: Callable[[], None],
                style: str = "Touch.TButton") -> ttk.Button:
        return ttk.Button(parent, text=text, style=style, command=command)

    def _load_photo(self, path: Path, box: Tuple[int, int]) -> Optional[ImageTk.PhotoImage]:
        try:
            with Image.open(path) as image:
                image.draft("RGB", (box[0] * 2, box[1] * 2))  # fast reduced-size JPEG decode
                thumb = image.convert("RGB")
            thumb.thumbnail(box)
            photo = ImageTk.PhotoImage(thumb)
        except (OSError, ValueError):
            return None
        self._photos.append(photo)
        return photo

    def _nav_bar(self, page: Page, goto: Callable[[int], None], back: Callable[[], None]) -> None:
        # Packed before the expanding content so it always keeps its space at the bottom.
        bar = ttk.Frame(self.body)
        bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))
        self._button(bar, "Back", back).pack(side=tk.LEFT)
        if page.count > 1:
            nxt = self._button(bar, "Next >", lambda: goto(page.index + 1))
            nxt.pack(side=tk.RIGHT)
            prev = self._button(bar, "< Prev", lambda: goto(page.index - 1))
            prev.pack(side=tk.RIGHT, padx=(0, 12))
            ttk.Label(bar, text=f"{page.index + 1} / {page.count}", style="Body.TLabel").pack(
                side=tk.RIGHT, padx=16)
            if page.index == 0:
                prev.state(["disabled"])
            if page.index >= page.count - 1:
                nxt.state(["disabled"])

    # ------------------------------------------------------------------ #
    # Screens
    # ------------------------------------------------------------------ #
    def show_home(self, notice: str = "") -> None:
        self._reset()
        self._title("FullPipeline")
        if notice:
            ttk.Label(self.body, text=notice, style="Notice.TLabel", wraplength=1100,
                      justify=tk.LEFT).pack(anchor="w")
        self._button(self.body, "Quit", self.root.destroy).pack(side=tk.BOTTOM, anchor="e", pady=(12, 0))
        grid = ttk.Frame(self.body)
        grid.pack(fill=tk.BOTH, expand=True)
        tiles: List[Tuple[str, Callable[[], None]]] = [
            (STAGES[key].title, lambda key=key: self.choose(STAGES[key])) for key in STAGE_ORDER
        ]
        tiles.append(("Output folder", self.open_output_folder))
        for index, (title, command) in enumerate(tiles):
            self._button(grid, title, command, style="Tile.TButton").grid(
                row=index // 3, column=index % 3, sticky="nsew", padx=10, pady=10)
        grid.columnconfigure((0, 1, 2), weight=1, uniform="tile")
        grid.rowconfigure((0, 1), weight=1, uniform="tile")

    def choose(self, stage: Stage) -> None:
        if stage.naming:  # a stage that writes new captures: refuse if the drive is missing
            problem = storage_problem(self.captures_dir)
            if problem:
                self.show_home(problem)
                return
        if stage.needs_session:
            self.show_session_picker(stage)
        elif stage.naming:
            self.show_name_keyboard(stage)
        else:
            self.start_stage(stage)

    def show_session_picker(self, stage: Stage, folder: Optional[str] = None, page: int = 0) -> None:
        self._reset()
        sessions = find_sessions(self.captures_dir, self.hidden_dirs)
        if folder is None:
            self._show_folder_list(stage, sessions, page)
        else:
            members = [s for s in sessions if s.folder == folder]
            self._show_folder_sessions(stage, members, folder, page)

    def _show_folder_list(self, stage: Stage, sessions: List[Session], page: int) -> None:
        self._title(f"{stage.title}: pick a capture")
        if not sessions:
            self._button(self.body, "Back", self.show_home).pack(side=tk.BOTTOM, anchor="w")
            ttk.Label(self.body, text=storage_problem(self.captures_dir)
                      or "No captures yet. Use Take images first.",
                      style="Body.TLabel", wraplength=1000, justify=tk.LEFT).pack(anchor="w", pady=24)
            return
        newest = latest_session(sessions)
        self._button(self.body, f"Latest capture\n{newest.display}",
                     lambda: self.start_stage(stage, session=newest.path),
                     style="TouchHuge.TButton").pack(fill=tk.X, pady=(0, 16))
        chunk = paginate(group_by_folder(sessions), page, FOLDERS_PER_PAGE)
        self._nav_bar(chunk, lambda p: self.show_session_picker(stage, None, p), self.show_home)
        grid = ttk.Frame(self.body)
        grid.pack(fill=tk.BOTH, expand=True)
        for index, (name, members) in enumerate(chunk.items):
            label = f"{name or 'Unnamed captures'}  ({len(members)})"
            self._button(grid, label, lambda name=name: self.show_session_picker(stage, name)).grid(
                row=index // 2, column=index % 2, sticky="nsew", padx=6, pady=6)
        grid.columnconfigure((0, 1), weight=1, uniform="folders")

    def _show_folder_sessions(self, stage: Stage, members: List[Session], folder: str, page: int) -> None:
        self._title(f"{folder or 'Unnamed captures'}: pick a capture")
        chunk = paginate(members, page, SESSIONS_PER_PAGE)
        self._nav_bar(chunk, lambda p: self.show_session_picker(stage, folder, p),
                      lambda: self.show_session_picker(stage))
        grid = ttk.Frame(self.body)
        grid.pack(fill=tk.BOTH, expand=True)
        for index, session in enumerate(chunk.items):
            photo = self._load_photo(session.path / "rgb_cam1.jpg", THUMB_BOX)
            button = ttk.Button(grid, text=session.display, style="Touch.TButton",
                                command=lambda s=session: self.start_stage(stage, session=s.path))
            if photo is not None:
                button.configure(image=photo, compound=tk.TOP)
            button.grid(row=index // 3, column=index % 3, sticky="nsew", padx=6, pady=6)
        grid.columnconfigure((0, 1, 2), weight=1, uniform="cards")

    def show_name_keyboard(self, stage: Stage) -> None:
        self._reset()
        self._name.set("")
        self._title("Name this capture (optional)")
        ttk.Label(self.body, textvariable=self._name, style="Title.TLabel",
                  relief="sunken", anchor="w", padding=8).pack(fill=tk.X)
        existing = [name for name, _ in group_by_folder(find_sessions(self.captures_dir, self.hidden_dirs))
                    if name][:NAME_CHIPS]
        if existing:
            chips = ttk.Frame(self.body)
            chips.pack(fill=tk.X, pady=8)
            for name in existing:
                self._button(chips, name, lambda name=name: self._name.set(name)).pack(side=tk.LEFT, padx=4)

        keys = ttk.Frame(self.body)
        keys.pack(expand=True)
        for row, letters in enumerate(KEY_ROWS):
            for column, char in enumerate(letters):
                ttk.Button(keys, text=char, style="Key.TButton", width=3,
                           command=lambda char=char: self._name.set(
                               sanitize_session_name(self._name.get() + char))
                           ).grid(row=row, column=column, padx=3, pady=3)
        ttk.Button(keys, text="Backspace", style="Key.TButton",
                   command=lambda: self._name.set(self._name.get()[:-1])
                   ).grid(row=len(KEY_ROWS), column=0, columnspan=5, sticky="nsew", padx=3, pady=3)
        ttk.Button(keys, text="Clear", style="Key.TButton",
                   command=lambda: self._name.set("")
                   ).grid(row=len(KEY_ROWS), column=5, columnspan=5, sticky="nsew", padx=3, pady=3)

        actions = ttk.Frame(self.body)
        actions.pack(fill=tk.X, pady=(8, 0))
        self._button(actions, "Back", self.show_home).pack(side=tk.LEFT)
        self._button(actions, "Skip (auto-name)", lambda: self.start_stage(stage)).pack(side=tk.RIGHT)
        self._button(actions, "Start", lambda: self.start_stage(
            stage, name=sanitize_session_name(self._name.get()) or None)).pack(side=tk.RIGHT, padx=(0, 12))

    def show_running(self, stage: Stage) -> None:
        self._reset()
        self._title(f"Running: {stage.title}")
        ttk.Label(self.body, style="Body.TLabel", wraplength=1000, justify=tk.LEFT,
                  text="The tool's window opens on top of this screen. Use its Done / "
                       "Finish button when you are finished.").pack(anchor="w", pady=8)
        ttk.Label(self.body, textvariable=self._elapsed, style="Body.TLabel").pack(anchor="w")
        progress = ttk.Progressbar(self.body, mode="indeterminate")
        progress.pack(fill=tk.X, pady=16)
        progress.start(15)
        self._button(self.body, "Stop", self._stop_run, style="TouchHuge.TButton").pack(
            side=tk.BOTTOM, fill=tk.X)
        self._poll(stage)

    def show_error(self, stage: Stage, message: str, details: str) -> None:
        self._reset()
        self._title(f"{stage.title} did not finish")
        ttk.Label(self.body, text=message, style="Body.TLabel", wraplength=1000,
                  justify=tk.LEFT).pack(anchor="w", pady=12)
        actions = ttk.Frame(self.body)
        actions.pack(side=tk.BOTTOM, fill=tk.X)
        box = tk.Text(self.body, height=10, font=("TkFixedFont", 14), wrap="word")
        box.insert("1.0", details or "(no output was recorded)")
        box.configure(state="disabled")

        def toggle() -> None:
            if box.winfo_manager():
                box.pack_forget()
            else:
                box.pack(fill=tk.BOTH, expand=True, pady=8)

        self._button(actions, "Home", self.show_home).pack(side=tk.LEFT)
        self._button(actions, "Retry", self._retry).pack(side=tk.RIGHT)
        self._button(actions, "Show details", toggle).pack(side=tk.RIGHT, padx=(0, 12))

    def show_result(self, stage: Stage, report: Optional[Report]) -> None:
        self._reset()
        self._title(f"{stage.title}: result")
        actions = ttk.Frame(self.body)
        actions.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))
        self._button(actions, "Done", self.show_home).pack(side=tk.LEFT)
        if report is None:
            ttk.Label(self.body, style="Body.TLabel", wraplength=1000, justify=tk.LEFT,
                      text="No result was saved. (The measure tools only save once points "
                           "have been placed before Finish.)").pack(anchor="w", pady=12)
            return
        content = ttk.Frame(self.body)
        content.pack(fill=tk.BOTH, expand=True)
        text = tk.Text(content, font=("TkFixedFont", 16), wrap="none", width=44)
        try:
            text.insert("1.0", report.report_path.read_text(encoding="utf-8", errors="replace"))
        except OSError as error:
            text.insert("1.0", f"Could not read {report.report_path}: {error}")
        text.configure(state="disabled")
        self._button(actions, "Page down", lambda: text.yview_scroll(1, "pages")).pack(side=tk.RIGHT)
        self._button(actions, "Page up", lambda: text.yview_scroll(-1, "pages")).pack(
            side=tk.RIGHT, padx=(0, 12))
        if report.image_path is not None:
            box = (int(self.root.winfo_screenwidth() * 0.5), int(self.root.winfo_screenheight() * 0.6))
            photo = self._load_photo(report.image_path, box)
            if photo is not None:
                ttk.Label(content, image=photo).pack(side=tk.LEFT, padx=(0, 16))
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------ #
    # Running stages
    # ------------------------------------------------------------------ #
    def start_stage(self, stage: Stage, session: Optional[Path] = None, name: Optional[str] = None) -> None:
        self._last_launch = (stage, session, name)
        window = window_for_screen(self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        command = build_command(stage, sys.executable, self.project_root,
                                session=session, name=name, window=window)
        log_path = self.project_root / "logs" / "menu" / f"{time.strftime('%Y%m%d_%H%M%S')}_{stage.key}.log"
        run = StageRun(command, log_path, self.project_root)
        try:
            run.start()
        except OSError as error:
            self.show_error(stage, f"Could not start {stage.title}.", str(error))
            return
        self._run = run
        self._elapsed.set("0 s")
        self.show_running(stage)

    def _stop_run(self) -> None:
        if self._run is not None:
            self._run.stop()
            self._elapsed.set("Stopping...")

    def _retry(self) -> None:
        if self._last_launch is not None:
            stage, session, name = self._last_launch
            self.start_stage(stage, session, name)

    def _poll(self, stage: Stage) -> None:
        run = self._run
        if run is None:
            return
        code = run.poll()
        if code is None:
            if not run.stopped:
                self._elapsed.set(f"{int(run.elapsed())} s")
            self.root.after(POLL_MS, lambda: self._poll(stage))
            return
        self._run = None
        self.root.lift()
        if run.stopped:
            self.show_home(f"{stage.title} stopped.")
            return
        log = run.log_text()
        if code != 0:
            self.show_error(stage, friendly_error(log), tail_lines(log))
            return
        if stage.result:
            self.show_result(stage, find_newest_report(self.results_dir, run.started_at))
            return
        self.show_home(f"{stage.title}: finished.")

    def open_output_folder(self) -> None:
        problem = storage_problem(self.captures_dir)
        if problem:  # never mkdir(parents=True) a missing mount point
            self.show_home(problem)
            return
        try:
            self.captures_dir.mkdir(exist_ok=True)
            subprocess.Popen(["xdg-open", str(self.captures_dir)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as error:
            self.show_home(f"Could not open the output folder: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Touchscreen home screen for the FullPipeline rig.")
    parser.add_argument("--install-shortcut", action="store_true",
                        help="Write a FullPipeline launcher to ~/Desktop and "
                             "~/.local/share/applications, then exit.")
    args = parser.parse_args()
    if args.install_shortcut:
        for path in install_shortcut(sys.executable, Path(__file__).resolve(), PROJECT_ROOT):
            print(f"Wrote {path}")
        return 0
    MenuApp().root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m unittest tests.test_menu_smoke -v`
Expected: all PASS (or SKIP without a display). Briefly opens Tk windows on the current display.

- [ ] **Step 5: README**

Add to `README.md`, after the "Guided live capture (recommended)" section and before "## Calibration", a short section (do not duplicate content from elsewhere in the README):

````markdown
## Touchscreen menu

```bash
python menu.py                     # home screen: capture, calibrate, measure, register, output folder
python menu.py --install-shortcut  # desktop + application-menu launcher (run once on the Jetson)
```

A launcher for the touchscreen: every tile runs the existing script with `--touch`
(`capture_pipeline.py`, `calibrate_live.py`, `measure_wound_depth.py`, `measure_points.py`
take it; `register_features.py` has no window). `--touch` swaps keyboard/mouse-wheel controls
for on-screen buttons (aim with a tap, nudge with the arrows, then **Place**; Reset, Finish
and the calibration discard buttons need a second tap) and leaves everything else unchanged.
Stage output goes to `logs/menu/`; the measure/registration tiles finish on a Result screen
showing the newest `report.txt`/`report_features.txt` from `registration/results/`.
Sessions from the calibration folders in `config.yaml` are hidden from the session picker.
````

- [ ] **Step 6: Final verification**

Run, from the project root:
```bash
python -m unittest discover -s tests -t . -v
python -m py_compile menu.py menu_logic.py touch_controls.py gui.py live_gui.py capture_pipeline.py calibrate_live.py measure_points.py measure_wound_depth.py check_depth_accuracy.py
for s in capture_pipeline calibrate_live measure_points measure_wound_depth; do python $s.py --help | grep -c -- "--touch"; done
python menu.py --install-shortcut --help | head -3
git status --short
```
Expected: every test PASS (Tk tests SKIP only if there is no display); `py_compile` silent; each `grep -c` prints `1`; `--help` prints usage; `git status` shows only intended files (`calibration/report.txt` still untracked, the pre-existing modified `config.yaml` untouched and unstaged, nothing under `captures/`, `logs/`, `registration/results/`).

Do **not** run `python menu.py --install-shortcut` for real here (it writes into the home directory); it is verified by `tests.test_menu_runtime.ShortcutTest` against a temp home.

- [ ] **Step 7: Commit**

```bash
git add menu.py tests/test_menu_smoke.py README.md
git commit -m "feat: touchscreen menu (home, session picker, name keyboard, running, error, result)"
```

- [ ] **Step 8: Report honestly**

State plainly in the summary what was **not** exercised: any real camera code path (capture loop, live calibration), real touchscreen input reaching an OpenCV window, window stacking under the Jetson's window manager, and the `.desktop` "Allow Launching" first-run step. List the spec's "Assumptions to check on the Jetson" as the first things to try on the device.

---

## Self-Review (run against the spec)

**Spec coverage**

| Spec section | Task |
|---|---|
| Home screen, 6 tiles + Quit, maximized (not kiosk), adapts to screen | 10 (`show_home`, `apply_touch_style`), 8 (`window_for_screen`) |
| Stage commands table (`--touch`, `--window`, `--output`, register untouched) | 8 (`build_command`, `StageTableTest`) |
| Running screen with Stop, no hide; logs to `logs/menu/`; friendly errors + details + Retry | 9 (`StageRun`, `friendly_error`), 10 (`show_running`, `show_error`) |
| Session picker: Latest, folders, thumbnails, hidden calibration folders | 8 (`find_sessions`, `hidden_capture_dirs`), 10 |
| Naming keyboard with chips and Skip | 8 (`sanitize_session_name`), 10 (`show_name_keyboard`) |
| Result screen (incl. `report_features.txt`) | 9 (`find_newest_report`), 10 (`show_result`) |
| Touch layer: tap→crosshair, drag pan, nudge, Place, Cancel, button bar, confirm gates, status strip | 1, 2, 3 |
| Wound-depth `on_status` (rim count / locked / live depth) | 4 |
| Capture `--touch` (Take photo, Done, hide folder button) + debounce | 6 |
| Live calibration `--touch` (two rows, hints dropped, confirm on discards) | 7 |
| Collision fix (`_2`/`_3`, own commit) | 5 |
| `--install-shortcut` | 9 (`install_shortcut`), 10 (`main`) |
| `logs/` gitignore, README section | 9, 10 |
| Never `mkdir(parents=True)` a missing captures mount (Global Constraints: `output_dir` may be removable-storage-absolute) | 8 (`storage_problem`), 10 (`choose`, `_show_folder_list`, `open_output_folder`) |
| Verification statement / Jetson assumptions | 10 Step 6–8 |

**Placeholder scan:** no TBD/TODO; every code step has code. The two "wrap existing lines under `if not self._touch:`" edits (Task 7 §3d) describe an indentation-only change to existing statements whose content is unchanged.

**Type/name consistency:** `Button/ButtonBar/bar_rows/draw_bar/TouchController` (Task 2) match their use in Task 3's test (`ButtonBar`, `bar_rows`) and Task 3's `TouchController(state, STATUS_HEIGHT, show_lock=...)`. `run_interactive(..., touch=)` (Task 3) matches Task 4's call. `Debouncer/SAVE_DEBOUNCE_S/apply_touch_style` (Task 1) match Tasks 6, 7, 10. `Session.name/.folder/.display`, `Stage` fields, `Page.items/.index/.count`, `Report(report_path, image_path)`, `StageRun.started_at/.stopped/.poll/.stop/.elapsed/.log_text` (Tasks 8–9) match their use in `menu.py` (Task 10). `hidden_capture_dirs(config, project_root)` / `captures_dir_from` / `results_dir_from` match `MenuApp.__init__`.
