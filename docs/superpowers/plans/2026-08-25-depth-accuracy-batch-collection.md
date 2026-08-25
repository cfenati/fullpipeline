# Fix GUI freeze, switch to batched multi-point plane collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a real hardware-reproduced bug (the interactive GUI freezes and gets force-killed because `LabelingSession` calls blocking `input()` from inside an OpenCV mouse callback) and switch from one click per block to click-many/press-n-to-advance batches per plane, averaging along the reference plane's already-established normal.

**Architecture:** `measure_points.run_interactive` gains a modal in-window text-entry mode (digits/comma/backspace/enter/esc, all handled inside the same `cv2.waitKey()` loop that already pumps window events -- no blocking I/O anywhere) plus two new hooks, `on_advance` (fires on **n**, closes out the current batch) and `on_text_submit` (fires on Enter during text entry). `check_depth_accuracy.py`'s `PlaneCollectionSession` replaces `LabelingSession`, tracking an ordered list of batches (the first is always the reference; every one after is a labeled block) instead of one label per click. `measure_depth_session`/`aggregate_depth_results`/`write_report` are untouched -- a batch is just expanded into the same parallel per-click arrays those functions already consume.

**Tech Stack:** Python 3.9, `opencv-contrib-python`, `numpy`; this repo's `measure_points.py`, `calibration.depth_grid_target.DepthGridTarget`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-25-depth-accuracy-batch-collection-design.md` (primary); `docs/superpowers/specs/2026-08-25-depth-accuracy-interactive-fix-design.md` and `docs/superpowers/specs/2026-08-24-check-depth-accuracy-design.md` (background).
- No test suite or linter in this repo -- verify with throwaway `python3` heredoc scripts, run during implementation and **not committed**, same precedent as every prior plan touching this tool.
- `from __future__ import annotations` + type hints on every touched signature.
- No blocking terminal I/O (`input()`) anywhere in the interactive session -- this is the whole point of this plan. Every prompt goes through the new in-window text-entry mode.
- A block batch needs `>= 1` point (not more); the reference batch needs `>= 3`. Neither is tied to a fixed count read from config.
- A block's measured height is the mean of its points' perpendicular distances to the reference plane's normal -- never an independently-fit plane per block (see the design doc's "Why the reference plane comes first" reasoning, generalized).
- `measure_depth_session`, `aggregate_depth_results`, `write_report` are not modified except the one necessary relaxation in Task 3 Step 1 (reference count `>= 3` instead of an exact match) -- everything else about their shape and behavior stays exactly as-is.
- The actual click-through-a-window path cannot be exercised by script (this bug was only found by running on real hardware, and this plan can't re-run that either) -- verify the state-machine and driver logic directly with synthetic inputs, and say explicitly that the real GUI path remains unverified by script.

---

### Task 1: Modal text-entry mode + `on_advance`/`on_text_submit` hooks in `measure_points.py`

**Files:**
- Modify: `measure_points.py`

**Interfaces:**
- Produces: a new module-level function `handle_interactive_key(state: Dict[str, Any], key: int, extrinsics: StereoExtrinsics, on_undo: Optional[Callable[[int], None]], on_advance: Optional[Callable[[], Optional[str]]], on_text_submit: Optional[Callable[[str], Optional[str]]], recompute: Callable[[], None]) -> bool` -- handles one already-`&0xFF`-masked key code; returns `True` if the session should end. Extracted from `run_interactive`'s main loop so it's testable without a real window.
- Produces: `run_interactive(..., on_advance: Optional[Callable[[], Optional[str]]] = None, on_text_submit: Optional[Callable[[str], Optional[str]]] = None) -> Dict[str, Any]` -- two new trailing optional params, appended after the existing `on_point`/`on_undo`. `on_advance()` fires on **n**; if it returns a prompt string, the window enters text-entry mode (digits 0-9, `,`, Backspace, Enter, Esc; the typed text renders live in the status bar). Enter calls `on_text_submit(text)`: `None` return accepts and exits text-entry mode, a string return is an error message that's printed and the buffer clears for another attempt. Esc cancels text-entry with no hook call. While `text_mode` is on, mouse clicks are ignored entirely.

- [ ] **Step 1: Extract `handle_interactive_key`, add the two new hook parameters, add text-entry state**

In `measure_points.py`, change `run_interactive`'s signature:

```python
def run_interactive(image_a: np.ndarray, image_b: np.ndarray,
                    extrinsics: StereoExtrinsics, depth_range: Tuple[float, float],
                    zoom: int, max_window: Tuple[int, int], blob_radius: int,
                    depth_m: float,
                    on_point: Optional[Callable[[int, Dict[str, Any]], None]] = None,
                    on_undo: Optional[Callable[[int], None]] = None,
                    on_advance: Optional[Callable[[], Optional[str]]] = None,
                    on_text_submit: Optional[Callable[[str], Optional[str]]] = None,
                    ) -> Dict[str, Any]:
```

Add three fields to the `state` dict literal, right after `"click_offsets_px": [],`:

```python
        "click_offsets_px": [],
        # Modal text entry (row,col labels etc.), driven entirely through this
        # same key-handling loop -- NEVER through input(), which would block
        # cv2's event pump and freeze the window until it gets force-killed.
        # This is not a hypothetical: it's exactly what happened running the
        # previous (input()-based) design on real hardware.
        "text_mode": False, "text_prompt": "", "text_buffer": "",
```

Replace `on_mouse`'s opening line:

```python
    def on_mouse(event, x, y, flags, _param):
        state["cursor"] = (x, y)
```

with:

```python
    def on_mouse(event, x, y, flags, _param):
        if state["text_mode"]:
            return
        state["cursor"] = (x, y)
```

(so a click during text entry is silently ignored rather than registering as a point).

Add a new module-level function, placed right before `def run_interactive(`:

```python
def handle_interactive_key(
    state: Dict[str, Any], key: int, extrinsics: StereoExtrinsics,
    on_undo: Optional[Callable[[int], None]],
    on_advance: Optional[Callable[[], Optional[str]]],
    on_text_submit: Optional[Callable[[str], Optional[str]]],
    recompute: Callable[[], None],
) -> bool:
    """Handle one already-``& 0xFF``-masked key code. Returns True if the
    session should end (q/Esc outside text-entry mode).

    Extracted from run_interactive's main loop so the key-handling state
    machine -- including the text-entry mode -- is testable with a synthetic
    state dict and key codes, without a real window.
    """
    if state["text_mode"]:
        if key in (13, 10):
            if on_text_submit is not None:
                error = on_text_submit(state["text_buffer"])
                if error is None:
                    state["text_mode"] = False
                    state["text_prompt"] = ""
                    state["text_buffer"] = ""
                else:
                    print(f"  {error}")
                    state["text_buffer"] = ""
        elif key == 27:
            state["text_mode"] = False
            state["text_prompt"] = ""
            state["text_buffer"] = ""
        elif key in (8, 127):
            state["text_buffer"] = state["text_buffer"][:-1]
        elif 48 <= key <= 57 or key == ord(","):
            state["text_buffer"] += chr(key)
        return False

    if key in (ord("q"), ord("Q"), 27):
        return True
    if key == ord("u"):
        if state["pending_a"] is not None:
            state["pending_a"] = None
        elif state["clicks_b"]:
            state["clicks_a"].pop(); state["clicks_b"].pop()
            state["click_offsets_px"].pop()
            recompute()
            if on_undo is not None:
                on_undo(len(state["clicks_a"]))
    elif key == ord("r"):
        state["clicks_a"].clear(); state["clicks_b"].clear()
        state["click_offsets_px"].clear()
        state["pending_a"] = None; recompute()
        if on_undo is not None:
            on_undo(0)
    elif key == ord("n"):
        if on_advance is not None:
            prompt = on_advance()
            if prompt is not None:
                state["text_mode"] = True
                state["text_prompt"] = prompt
                state["text_buffer"] = ""
    elif key == ord("0"):
        state["panel_a"].fit(); state["panel_b"].fit()
    elif key == ord("l"):
        state["linked"] = not state["linked"]
        if state["linked"]:
            link_view(state["panel_a"], state["panel_b"], extrinsics, state["depth_hint"])
    elif key in (ord("+"), ord("=")):
        state["zoom"] = min(32, state["zoom"] * 2)
    elif key == ord("-"):
        state["zoom"] = max(2, state["zoom"] // 2)
    return False
```

- [ ] **Step 2: Update the status-bar text to show the text-entry prompt when active, and use `handle_interactive_key` in the main loop**

Replace the status-building block and the key-handling chain (from `while True:` through the end of the `elif key == ord("-"):` branch) with:

```python
    while True:
        placed = len(state["clicks_a"])
        if state["text_mode"]:
            state["status"] = [
                f"{state['text_prompt']}{state['text_buffer']}_",
                "type digits and , then Enter to confirm, Esc to cancel",
            ]
        else:
            head = (f"point {placed}: click the SAME feature in camera B, near the blue line"
                    if state["pending_a"] is not None
                    else f"point {placed}: click a feature in camera A")
            recent = "   ".join(f"{i}->{i+1}: {d:.3f}mm"
                                for i, d in enumerate(state["consecutive"]))[-140:]
            state["status"] = [
                head,
                f"measurements: {recent}" if recent else "measurements: (need two points)",
                ("wheel = zoom (camera A drags B along)   right-drag = pan   "
                 f"0 = fit   l = link {'ON' if state['linked'] else 'OFF'}"),
                "u undo | r reset | n advance/label | q or Esc = finish and print the report",
            ]
        cv2.imshow(window, render(state))
        key = cv2.waitKey(20)
        # Closing the window with its X button must end the session too, or the
        # loop would spin forever on an invisible window.
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == -1:
            continue
        key &= 0xFF
        if handle_interactive_key(
            state, key, extrinsics, on_undo, on_advance, on_text_submit, recompute,
        ):
            break
```

(This deletes the old inline `if key in (ord("q")...` through `elif key == ord("-"):` chain entirely -- it now lives inside `handle_interactive_key`.)

- [ ] **Step 3: Verify the key-handling state machine directly (no window needed)**

```bash
python3 - <<'EOF'
import numpy as np
from pathlib import Path
from calibration.stereo import StereoExtrinsics
from measure_points import handle_interactive_key

extrinsics = StereoExtrinsics.load_json(
    Path("calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json")
)

def fresh_state():
    return {
        "clicks_a": [], "clicks_b": [], "click_offsets_px": [], "pending_a": None,
        "text_mode": False, "text_prompt": "", "text_buffer": "",
        "zoom": 8, "linked": False,
    }

def noop():
    pass

# n with no on_advance registered: no-op, stays in click mode.
state = fresh_state()
ended = handle_interactive_key(state, ord("n"), extrinsics, None, None, None, noop)
assert not ended and not state["text_mode"]

# n with on_advance returning a prompt: enters text mode.
state = fresh_state()
handle_interactive_key(
    state, ord("n"), extrinsics, None, lambda: "cell row,col > ", None, noop,
)
assert state["text_mode"] and state["text_prompt"] == "cell row,col > "

# n with on_advance declining (not enough points): stays in click mode.
state = fresh_state()
handle_interactive_key(state, ord("n"), extrinsics, None, lambda: None, None, noop)
assert not state["text_mode"]

# Typing digits and a comma while in text mode builds the buffer; other keys
# (e.g. 'u') are NOT treated as hotkeys while text_mode is on.
state = fresh_state()
state["text_mode"] = True
for ch in "2,3":
    handle_interactive_key(state, ord(ch), extrinsics, None, None, None, noop)
assert state["text_buffer"] == "2,3", state["text_buffer"]
handle_interactive_key(state, ord("u"), extrinsics, None, None, None, noop)
assert state["text_buffer"] == "2,3u", state["text_buffer"]  # 'u' is just a (rejected) char here

# Backspace removes the last character.
handle_interactive_key(state, 8, extrinsics, None, None, None, noop)
assert state["text_buffer"] == "2,3", state["text_buffer"]

# Enter with on_text_submit accepting (returns None): exits text mode, clears buffer.
submitted = []
def accept(text):
    submitted.append(text)
    return None
handle_interactive_key(state, 13, extrinsics, None, None, accept, noop)
assert submitted == ["2,3"]
assert not state["text_mode"] and state["text_buffer"] == ""

# Enter with on_text_submit rejecting (returns an error string): stays in text
# mode, buffer clears for a retry, error is not silently dropped.
state = fresh_state()
state["text_mode"] = True
state["text_buffer"] = "99,99"
handle_interactive_key(
    state, 13, extrinsics, None, None, lambda text: "out of range", noop,
)
assert state["text_mode"] and state["text_buffer"] == ""

# Esc cancels text mode without calling on_text_submit at all.
state = fresh_state()
state["text_mode"] = True
state["text_buffer"] = "1,2"
called = []
handle_interactive_key(
    state, 27, extrinsics, None, None, lambda text: called.append(text), noop,
)
assert not called
assert not state["text_mode"]

# q outside text mode ends the session; q typed AS TEXT during text mode does not.
state = fresh_state()
assert handle_interactive_key(state, ord("q"), extrinsics, None, None, None, noop) is True
state = fresh_state()
state["text_mode"] = True
assert handle_interactive_key(state, ord("q"), extrinsics, None, None, None, noop) is False
assert state["text_mode"]  # 'q' isn't a valid text-entry char (not digit/comma) -- silently ignored

print("Task 1 verification OK")
EOF
```

Expected: `Task 1 verification OK`.

- [ ] **Step 4: Syntax check**

```bash
python3 -c "import measure_points"
```

Expected: no output, exit code 0. Note explicitly (per Global Constraints): this confirms the file imports and the extracted state machine behaves correctly under test -- it is not a claim that a real window was opened or that the fix has been confirmed on hardware.

- [ ] **Step 5: Commit**

```bash
git add measure_points.py
git commit -m "$(cat <<'EOF'
Fix GUI freeze: replace input() with an in-window modal text-entry mode

LabelingSession called blocking input() from inside an OpenCV mouse
callback -- reproduced on real hardware as exactly what that predicts:
the window stops processing events the moment input() blocks, and gets
force-killed as unresponsive. Adds on_advance/on_text_submit hooks and a
text-entry mode driven entirely through the same cv2.waitKey() loop that
already pumps window events, so no blocking terminal I/O exists anywhere
in the interactive session. handle_interactive_key is extracted as a
standalone function so this state machine is testable without a window.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `PlaneCollectionSession` replaces `LabelingSession`

**Files:**
- Modify: `check_depth_accuracy.py`

**Interfaces:**
- Consumes: `measure_points.run_interactive`'s `on_point`/`on_undo`/`on_advance`/`on_text_submit` hooks (Task 1); unchanged `fit_plane_3d`, `perpendicular_distance_to_plane`, `DepthGridTarget.row_count`/`col_count`/`height_at`.
- Produces: `PlaneCollectionSession(target: DepthGridTarget)` with `.on_point(index, result)`, `.on_undo(new_count)`, `.on_advance() -> Optional[str]`, `.on_text_submit(text) -> Optional[str]`, and a `.batches: List[Tuple[Optional[Tuple[int,int]], List[int]]]` attribute -- `batches[0]` is always the reference (label `None`); every entry after is `(row_col_label, [click indices])`. This is what Task 3's `main()` wiring reads once the window closes, in place of the old `LabelingSession.labels`.

- [ ] **Step 1: Replace `LabelingSession` with `PlaneCollectionSession`**

Replace the entire `LabelingSession` class (from `class LabelingSession:` through the end of its `on_undo` method) with:

```python
class PlaneCollectionSession:
    """Drives batched multi-point plane collection via run_interactive's
    on_point/on_undo/on_advance/on_text_submit hooks.

    Click freely for a plane's worth of points; press n to close the batch
    out. The first batch (index 0) is always the reference -- needs >= 3
    points, no label, fits the datum plane the moment it's closed. Every
    batch after that is a block, identified by row,col typed through the
    in-window text-entry mode (never a fixed count, never fewer than 1
    point); its measured height is the mean of its points' perpendicular
    distances to the reference plane's ALREADY-established normal, not an
    independent per-block plane fit -- every block is part of the same
    rigid printed object as the reference corners, so it shares that one
    normal exactly. See docs/superpowers/specs/2026-08-25-depth-accuracy-
    batch-collection-design.md.
    """

    def __init__(self, target: DepthGridTarget) -> None:
        self.target = target
        self.plane: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.batches: List[Tuple[Optional[Tuple[int, int]], List[int]]] = [(None, [])]
        self.points_mm: Dict[int, np.ndarray] = {}

    def on_point(self, index: int, result: Dict[str, Any]) -> None:
        self.points_mm[index] = np.asarray(result["points_mm"][index], dtype=np.float64)
        label, indices = self.batches[-1]
        indices.append(index)
        if label is None:
            print(f"  reference point {len(indices)} recorded")
        else:
            print(f"  point {len(indices)} recorded for block {label}")

    def on_undo(self, new_count: int) -> None:
        for index in list(self.points_mm):
            if index >= new_count:
                del self.points_mm[index]
        # Drop indices >= new_count from the current batch, back to front; an
        # emptied batch that isn't the reference (index 0) is discarded
        # outright, so the "current batch" pointer falls back to whichever
        # batch actually still owns the removed point (reopening it, even if
        # it was already finalized -- no relabeling needed, the label is
        # still recorded).
        while self.batches:
            label, indices = self.batches[-1]
            indices[:] = [i for i in indices if i < new_count]
            if indices or len(self.batches) == 1:
                break
            self.batches.pop()
        if new_count == 0:
            self.plane = None

    def on_advance(self) -> Optional[str]:
        label, indices = self.batches[-1]
        if label is None:
            if len(indices) < 3:
                print(f"  need at least 3 reference points, have {len(indices)}")
                return None
            points_mm = np.array([self.points_mm[i] for i in indices])
            centroid, normal, rms_m = fit_plane_3d(points_mm / 1000.0)
            if normal @ (-centroid) < 0.0:
                normal = -normal
            self.plane = (centroid, normal)
            print(f"  reference plane fit from {len(indices)} points -- "
                  f"rms {rms_m * 1000.0:.4f} mm")
        else:
            if len(indices) < 1:
                print("  click at least one point before pressing n")
                return None
            centroid, normal = self.plane
            values = np.array([
                perpendicular_distance_to_plane(self.points_mm[i] / 1000.0, centroid, normal)
                * 1000.0
                for i in indices
            ])
            mean = float(values.mean())
            if values.size > 1:
                spread = float(np.sqrt(np.mean((values - mean) ** 2)))
                print(f"  block {label}: {values.size} points, mean {mean:.3f}mm "
                      f"(spread {spread:.4f}mm rms)")
            else:
                print(f"  block {label}: measured {mean:.3f}mm")
        return "cell row,col > "

    def on_text_submit(self, text: str) -> Optional[str]:
        try:
            row_text, col_text = text.split(",")
            row, col = int(row_text), int(col_text)
            if not (0 <= row < self.target.row_count and 0 <= col < self.target.col_count):
                raise ValueError
        except ValueError:
            return (f"enter as ROW,COL within 0-{self.target.row_count - 1},"
                    f"0-{self.target.col_count - 1}")
        self.batches.append(((row, col), []))
        return None
```

- [ ] **Step 2: Verify the batch-driver logic directly (no GUI)**

This mirrors how `LabelingSession` was verified before: simulate a full click sequence -- 4 reference points, 3 points for block (0,0), 2 points for block (0,1) -- by calling the hooks exactly as `run_interactive` would, including the minimum-point declines and an undo that reopens an already-finalized batch.

```bash
python3 - <<'EOF'
import numpy as np
from calibration.depth_grid_target import DepthGridTarget
from check_depth_accuracy import PlaneCollectionSession

target = DepthGridTarget(heights_mm=((3.6, 7.0),), pitch_mm=10.0,
                          reference_corner_count=4, measured_by="synthetic test")
session = PlaneCollectionSession(target)

# 4 flat (z=180, no tilt -- fine for testing bookkeeping, not plane precision)
# reference points.
ref_points = [
    [-24.0, -24.0, 180.0], [24.0, -24.0, 180.0],
    [-24.0, 24.0, 180.0], [24.0, 24.0, 180.0],
]
for i in range(4):
    session.on_point(i, {"points_mm": ref_points[: i + 1]})

# Pressing n too early (only 2 points -- below the minimum of 3) must decline.
partial = PlaneCollectionSession(target)
for i in range(2):
    partial.on_point(i, {"points_mm": ref_points[:i + 1]})
assert partial.on_advance() is None

# The real session: n after all 4 lands, fits the plane.
prompt = session.on_advance()
assert prompt == "cell row,col > "
assert session.plane is not None

# Reject a malformed / out-of-range row,col; accept a valid one.
assert session.on_text_submit("bogus") is not None
assert session.on_text_submit("9,9") is not None  # out of range for a 1x2 grid
assert session.on_text_submit("0,0") is None
assert session.batches[-1] == ((0, 0), [])

# 3 points for block (0,0), exactly on the z=180 plane offset by 3.6mm.
block_points = ref_points + [[0.0, 0.0, 176.4], [1.0, 1.0, 176.4], [2.0, 2.0, 176.4]]
for i in range(4, 7):
    session.on_point(i, {"points_mm": block_points[: i + 1]})
assert session.batches[-1][1] == [4, 5, 6]

# n with zero points in a fresh batch must decline (not silently accept an
# empty block).
empty_next = session.on_advance()
assert empty_next == "cell row,col > "  # this call finalizes (0,0), which HAS points
session.on_text_submit("0,1")
assert session.on_advance() is None  # NOW zero points in the (0,1) batch -- declines
print("  (confirmed: n with 0 points in a fresh block batch is declined)")

# Give it 2 points at 173.0 (7.0mm above the plane) and finalize. on_advance
# finalizes IN PLACE -- it does not itself open a new batch (only
# on_text_submit does), so batches[-1] is still ((0,1), [7,8]) right after.
block_points += [[5.0, 5.0, 173.0], [5.0, 5.0, 173.0]]
for i in range(7, 9):
    session.on_point(i, {"points_mm": block_points[: i + 1]})
prompt2 = session.on_advance()
assert prompt2 == "cell row,col > "
assert session.batches[-1] == ((0, 1), [7, 8])

# Advance past (0,1) by opening a new (empty) batch -- revisiting (0,0) is
# legitimate, a block can be labeled again across separate batches -- then
# undo before clicking anything into it. This must discard the empty
# pending batch outright and reopen (0,1) at its last point: no relabeling
# needed, and nothing downstream ever sees a zero-point batch.
session.on_text_submit("0,0")
assert session.batches[-1] == ((0, 0), [])
session.on_undo(8)  # click index 8 (the most recent) is undone
assert session.batches[-1] == ((0, 1), [7])

# Undo all the way back to zero: everything resets, including the plane.
session.on_undo(0)
assert session.batches == [(None, [])]
assert session.plane is None

print("Task 2 verification OK")
EOF
```

Expected: `Task 2 verification OK` (plus the printed live-feedback lines from `on_point`/`on_advance`, which is expected noise, not an error).

- [ ] **Step 3: Commit**

```bash
git add check_depth_accuracy.py
git commit -m "$(cat <<'EOF'
Replace LabelingSession with batched PlaneCollectionSession

Click many points for a plane, press n to close the batch -- the
reference batch needs >= 3 points and fits the datum plane on close;
every batch after that is a block (row,col typed via the new in-window
text-entry mode from Task 1, never input()), needing only >= 1 point,
measured as the mean of its points' perpendicular distances to the
reference plane's already-established normal rather than an
independent per-block plane fit. Not yet wired into main() -- that's
the next task.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Relax `measure_depth_session`'s reference-count check, wire `PlaneCollectionSession` into `main()`

**Files:**
- Modify: `check_depth_accuracy.py`

**Interfaces:**
- Consumes: Task 2's `PlaneCollectionSession` (`.batches`, `.on_point`, `.on_undo`, `.on_advance`, `.on_text_submit`); Task 1's `run_interactive(..., on_advance=, on_text_submit=)`.
- Produces: nothing new consumed elsewhere in this plan -- `measure_depth_session`'s only change is accepting `>= 3` reference clicks instead of exactly `target.reference_corner_count`; its return shape is unchanged.

- [ ] **Step 1: Relax the reference-count check in `measure_depth_session`**

Replace:

```python
    if len(ref_clicks_a) != target.reference_corner_count:
        raise ValueError(
            f"expected {target.reference_corner_count} reference clicks, "
            f"got {len(ref_clicks_a)}"
        )
```

with:

```python
    if len(ref_clicks_a) < 3:
        raise ValueError(
            f"need at least 3 reference clicks for a plane fit, got {len(ref_clicks_a)}"
        )
```

- [ ] **Step 2: Rewrite `main()`'s interactive branch around `PlaneCollectionSession`**

Replace this whole region of `main()` (from `else:` -- the interactive branch, through the end of building `cell_offsets_px`):

```python
        else:
            print(f"{label}: click the {target.reference_corner_count} reference corners "
                  "first (any order among themselves) -- the plane fits itself in as soon "
                  "as the last one lands. After that, click a block and type the height "
                  "engraved on it when prompted; click the same block again anytime for a "
                  "repeatability sample. Press q/Esc to finish.")
            session = LabelingSession(target)
            raw_result = run_interactive(
                image_a, image_b, extrinsics, depth_range,
                max(2, args.zoom), (int(args.window[0]), int(args.window[1])),
                max(3, args.blob_radius) if args.blob_snap else 0,
                float(reg_config.get("default_depth", 0.168)),
                on_point=session.on_point, on_undo=session.on_undo,
            )
            n_ref = target.reference_corner_count
            total_clicks = len(raw_result.get("clicks_a", [])) if raw_result else 0
            if total_clicks < n_ref + 1 or not session.labels:
                results.append({
                    "label": label,
                    "skipped": f"fewer than {n_ref} reference + 1 labeled block click",
                })
                print(f"{label}: skipped: not enough points clicked")
                continue
            ref_clicks_a = np.array(raw_result["clicks_a"][:n_ref])
            ref_clicks_b = np.array(raw_result["clicks_b_snapped"][:n_ref])
            ref_offsets_px = np.array(raw_result["epipolar_offset_px"][:n_ref])
            labeled_indices = sorted(session.labels)
            cell_labels = [session.labels[i] for i in labeled_indices]
            cell_clicks_a = np.array([raw_result["clicks_a"][i] for i in labeled_indices])
            cell_clicks_b = np.array(
                [raw_result["clicks_b_snapped"][i] for i in labeled_indices]
            )
            cell_offsets_px = np.array(
                [raw_result["epipolar_offset_px"][i] for i in labeled_indices]
            )
```

with:

```python
        else:
            print(f"{label}: click reference points on the flat baseplate (at least 3, "
                  "spread out) then press n -- the plane fits and its rms prints "
                  "immediately. Then click a block's points (as many as you like, even "
                  "just one) and press n again: type its row,col when prompted, Enter "
                  "to confirm. Repeat for every block you can see. Press q/Esc to finish.")
            session = PlaneCollectionSession(target)
            raw_result = run_interactive(
                image_a, image_b, extrinsics, depth_range,
                max(2, args.zoom), (int(args.window[0]), int(args.window[1])),
                max(3, args.blob_radius) if args.blob_snap else 0,
                float(reg_config.get("default_depth", 0.168)),
                on_point=session.on_point, on_undo=session.on_undo,
                on_advance=session.on_advance, on_text_submit=session.on_text_submit,
            )
            if not raw_result.get("clicks_a", []):
                results.append({"label": label, "skipped": "no points clicked"})
                print(f"{label}: skipped: no points clicked")
                continue
            labeled_batches = [
                (batch_label, indices) for batch_label, indices in session.batches
                if batch_label is not None and indices
            ]
            if len(session.batches[0][1]) < 3 or not labeled_batches:
                results.append({
                    "label": label,
                    "skipped": "fewer than 3 reference points or no labeled block batches",
                })
                print(f"{label}: skipped: not enough points clicked")
                continue
            ref_indices = session.batches[0][1]
            ref_clicks_a = np.array([raw_result["clicks_a"][i] for i in ref_indices])
            ref_clicks_b = np.array([raw_result["clicks_b_snapped"][i] for i in ref_indices])
            ref_offsets_px = np.array(
                [raw_result["epipolar_offset_px"][i] for i in ref_indices]
            )
            cell_labels = [
                batch_label for batch_label, indices in labeled_batches for _ in indices
            ]
            cell_clicks_a = np.array([
                raw_result["clicks_a"][i] for _, indices in labeled_batches for i in indices
            ])
            cell_clicks_b = np.array([
                raw_result["clicks_b_snapped"][i]
                for _, indices in labeled_batches for i in indices
            ])
            cell_offsets_px = np.array([
                raw_result["epipolar_offset_px"][i]
                for _, indices in labeled_batches for i in indices
            ])
```

- [ ] **Step 3: Verify against real calibrated extrinsics with a forward-projected synthetic session, driven through `measure_depth_session` directly with a variable reference count**

```bash
python3 - <<'EOF'
import numpy as np
from pathlib import Path
from calibration.stereo import StereoExtrinsics
from calibration.depth_grid_target import DepthGridTarget
from check_depth_accuracy import measure_depth_session

extrinsics = StereoExtrinsics.load_json(
    Path("calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json")
)
Ka, Kb, R = extrinsics.camera_matrix_a, extrinsics.camera_matrix_b, extrinsics.R
T = np.asarray(extrinsics.T).reshape(3)

def project(point_m, K):
    x, y, z = point_m
    return np.array([K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]])

def project_pair(point_a_m):
    return project(point_a_m, Ka), project(R @ point_a_m + T, Kb)

normal = np.array([0.05, -0.03, -0.998]); normal /= np.linalg.norm(normal)
centre = np.array([0.0, 0.0, 0.178])

def on_plane(dx_m, dy_m):
    p = centre + np.array([dx_m, dy_m, 0.0])
    return p - ((p - centre) @ normal) * normal

# 5 reference points (not the historical 4), to prove the relaxed check
# genuinely accepts a variable count, not just a different fixed one.
ref_corners_m = [
    on_plane(dx, dy) for dx, dy in
    [(-0.024, -0.024), (0.024, -0.024), (-0.024, 0.024), (0.024, 0.024), (0.0, 0.0)]
]
ref_clicks_a, ref_clicks_b = [], []
for p in ref_corners_m:
    a, b = project_pair(p)
    ref_clicks_a.append(a); ref_clicks_b.append(b)

target = DepthGridTarget(heights_mm=((3.6, 7.0),), pitch_mm=10.0,
                          reference_corner_count=4, measured_by="synthetic test")

# Block (0,0), 3 points (mean, not an independent plane fit -- Global
# Constraints); block (0,1), 1 point (the new minimum).
block36_m = on_plane(0.005, 0.0) + normal * 0.0036
block70_m = on_plane(-0.005, 0.005) + normal * 0.0070
cell_labels = [(0, 0), (0, 0), (0, 0), (0, 1)]
cell_clicks_a, cell_clicks_b = [], []
for point_m, jitter_px in ((block36_m, 0.0), (block36_m, 0.3), (block36_m, -0.2), (block70_m, 0.0)):
    a, b = project_pair(point_m)
    a = a + np.array([jitter_px, 0.0])
    cell_clicks_a.append(a); cell_clicks_b.append(b)

result = measure_depth_session(
    np.array(ref_clicks_a), np.array(ref_clicks_b), cell_labels,
    np.array(cell_clicks_a), np.array(cell_clicks_b), extrinsics, target,
)
print("plane_point_count", result["plane_point_count"])
print("block_labels", result["block_labels"])
print("block_samples", result["block_samples"])
print("block_measured_mm", result["block_measured_mm"])

assert result["plane_point_count"] == 5
assert result["block_labels"] == [(0, 0), (0, 1)]
assert result["block_samples"].tolist() == [3, 1]
assert abs(result["block_measured_mm"][0] - 3.6) < 0.05
assert abs(result["block_measured_mm"][1] - 7.0) < 0.05

# Below the new minimum (2 reference clicks) must still raise.
try:
    measure_depth_session(
        np.array(ref_clicks_a[:2]), np.array(ref_clicks_b[:2]), [(0, 0)],
        np.array(cell_clicks_a[:1]), np.array(cell_clicks_b[:1]), extrinsics, target,
    )
    raise AssertionError("expected ValueError for fewer than 3 reference clicks")
except ValueError as exc:
    assert "at least 3" in str(exc), exc

print("Task 3 verification OK")
EOF
```

Expected: `Task 3 verification OK`, `plane_point_count` is `5` (not the old hardcoded `4`), `block_samples` is `[3, 1]`.

- [ ] **Step 4: Syntax check**

```bash
python3 -c "import check_depth_accuracy"
```

Expected: no output, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add check_depth_accuracy.py
git commit -m "$(cat <<'EOF'
Wire PlaneCollectionSession into main(), relax reference count to >= 3

measure_depth_session no longer requires exactly target.reference_
corner_count reference clicks -- just enough for a plane fit (>= 3),
since the new click-many/press-n batch flow doesn't have a fixed count
to enforce up front. main()'s interactive branch expands each finalized
batch into the same parallel per-click arrays measure_depth_session
already consumed under the old per-click-label design, so its own
grouping/pairwise/aggregation logic is untouched.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Revert `--cell` CLI to `ROW,COL,AX,AY,BX,BY`, update docstrings

**Files:**
- Modify: `check_depth_accuracy.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `parse_cell(text: str) -> Tuple[int, int, float, float, float, float]` (was `Tuple[float, float, float, float, float]`). `--cell`'s CLI shape changes back to `ROW,COL,AX,AY,BX,BY`.

- [ ] **Step 1: Revert `parse_cell`**

Replace:

```python
def parse_cell(text: str) -> Tuple[float, float, float, float, float]:
    parts = text.replace(" ", "").split(",")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"--cell wants HEIGHT,AX,AY,BX,BY, got '{text}'"
        )
    try:
        height, ax, ay, bx, by = (float(value) for value in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--cell could not parse '{text}': {exc}") from exc
    return height, ax, ay, bx, by
```

with:

```python
def parse_cell(text: str) -> Tuple[int, int, float, float, float, float]:
    parts = text.replace(" ", "").split(",")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            f"--cell wants ROW,COL,AX,AY,BX,BY, got '{text}'"
        )
    try:
        row, col = int(parts[0]), int(parts[1])
        ax, ay, bx, by = (float(value) for value in parts[2:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--cell could not parse '{text}': {exc}") from exc
    return row, col, ax, ay, bx, by
```

- [ ] **Step 2: Revert `--cell`'s argparse help text**

Replace:

```python
    parser.add_argument("--cell", type=parse_cell, action="append", default=None,
                        metavar="HEIGHT,AX,AY,BX,BY",
                        help="Non-interactive: one block click pair, labeled by the height "
                             "(mm) engraved on it -- must match one of the target's "
                             "heights_mm values. Repeat for every block visible this "
                             "session (a block may repeat for a repeatability sample); "
                             "occluded blocks are simply omitted, not required.")
```

with:

```python
    parser.add_argument("--cell", type=parse_cell, action="append", default=None,
                        metavar="ROW,COL,AX,AY,BX,BY",
                        help="Non-interactive: one block click pair, labeled by its "
                             "(row, col) in the target's heights_mm grid. Repeat for every "
                             "block visible this session (a block may repeat for a "
                             "repeatability sample); occluded blocks are simply omitted, "
                             "not required.")
```

- [ ] **Step 3: Revert `--ref`'s validation and the non-interactive branch's `cell_labels` construction**

Replace:

```python
        if not args.ref or len(args.ref) != target.reference_corner_count:
            raise SystemExit(
                f"{target_path.name} needs exactly {target.reference_corner_count} --ref "
                f"clicks, got {len(args.ref) if args.ref else 0}."
            )
        if not args.cell:
            raise SystemExit("Need at least 1 --cell click to measure a block's depth.")
```

with:

```python
        if not args.ref or len(args.ref) < 3:
            raise SystemExit(
                f"Need at least 3 --ref clicks for a plane fit, got "
                f"{len(args.ref) if args.ref else 0}."
            )
        if not args.cell:
            raise SystemExit("Need at least 1 --cell click to measure a block's depth.")
        for row, col, *_ in args.cell:
            if not (0 <= row < target.row_count and 0 <= col < target.col_count):
                raise SystemExit(
                    f"--cell {row},{col} is out of range for a {target.row_count}x"
                    f"{target.col_count} grid."
                )
```

Replace:

```python
        if args.ref:
            ref_clicks_a = np.array([[point[0], point[1]] for point in args.ref])
            ref_clicks_b = np.array([[point[2], point[3]] for point in args.ref])
            try:
                cell_labels = [target.cell_at_height(cell[0]) for cell in args.cell]
            except ValueError as exc:
                raise SystemExit(str(exc))
            cell_clicks_a = np.array([[cell[1], cell[2]] for cell in args.cell])
            cell_clicks_b = np.array([[cell[3], cell[4]] for cell in args.cell])
            ref_offsets_px = None
            cell_offsets_px = None
```

with:

```python
        if args.ref:
            ref_clicks_a = np.array([[point[0], point[1]] for point in args.ref])
            ref_clicks_b = np.array([[point[2], point[3]] for point in args.ref])
            cell_labels = [(cell[0], cell[1]) for cell in args.cell]
            cell_clicks_a = np.array([[cell[2], cell[3]] for cell in args.cell])
            cell_clicks_b = np.array([[cell[4], cell[5]] for cell in args.cell])
            ref_offsets_px = None
            cell_offsets_px = None
```

- [ ] **Step 4: Revert the module docstring's usage example and "Method, per session" prose**

Replace:

```
    click a block, type the height engraved on it when prompted (repeat
    clicks on the same block are repeatability samples, not new blocks;
    some blocks may be self-occluded from one or both cameras -- that is
    expected, not a failure; see the target's module docstring) ->
    triangulate -> signed perpendicular distance from each block's point
    to the fitted plane
```

with:

```
    click at least 3 spread-out reference points on the flat baseplate,
    press n -> triangulate -> fit a plane through them (this is the depth
    datum, NOT any assumed camera-to-plate standoff -- the plate's mounting
    angle to the camera is unknown and is never trusted)
    click a block's points (even just one; repeat clicks average together
    for a repeatability sample, not a new block), press n, type its
    row,col when prompted -- some blocks may be self-occluded from one or
    both cameras, that is expected, not a failure; see the target's module
    docstring -> triangulate -> mean perpendicular distance from the
    block's points to the fitted plane
```

Replace:

```
Usage:
    python check_depth_accuracy.py --captures captures/depth_target
    python check_depth_accuracy.py --session captures/depth_target/<timestamp>
    python check_depth_accuracy.py --session captures/depth_target/<timestamp> \\
        --ref 100,200,90,205 --ref 900,200,890,205 \\
        --ref 100,900,90,905 --ref 900,900,890,905 \\
        --cell 0.6,300,400,290,405 --cell 30.0,700,600,690,605   # non-interactive;
        # HEIGHT,AX,AY,BX,BY -- identify a block by the number engraved on it
"""
```

with:

```
Usage:
    python check_depth_accuracy.py --captures captures/depth_target
    python check_depth_accuracy.py --session captures/depth_target/<timestamp>
    python check_depth_accuracy.py --session captures/depth_target/<timestamp> \\
        --ref 100,200,90,205 --ref 900,200,890,205 \\
        --ref 100,900,90,905 --ref 900,900,890,905 \\
        --cell 0,0,300,400,290,405 --cell 4,4,700,600,690,605   # non-interactive
"""
```

- [ ] **Step 5: Syntax check**

```bash
python3 -c "import check_depth_accuracy"
```

Expected: no output, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add check_depth_accuracy.py
git commit -m "$(cat <<'EOF'
Revert --cell to ROW,COL,AX,AY,BX,BY for consistency with interactive mode

The interactive session now identifies a block by row,col (typed
through the new text-entry mode), not the height engraved on it -- so
the non-interactive --ref/--cell path reverts to match, keeping both
paths identifying a block the same way. Updates the module docstring's
usage example and method description to match.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Remove `cell_at_height` and the tolerance-uniqueness check from `DepthGridTarget`

**Files:**
- Modify: `calibration/depth_grid_target.py`

**Interfaces:**
- Produces: nothing -- this task only removes now-dead code. `height_at`, `pair_depths_mm`, `row_count`, `col_count`, `from_dict`/`from_yaml`/`to_dict` are all unchanged.

- [ ] **Step 1: Remove the tolerance-uniqueness check from `__post_init__`**

Replace:

```python
        if any(height < 0.0 for row in self.heights_mm for height in row):
            raise ValueError("heights_mm must not contain negative heights")
        flat_heights = [height for row in self.heights_mm for height in row]
        # Check that all heights are sufficiently distinct (within tolerance 1e-6).
        # Use sorted adjacent-diff approach: O(n log n) and catches any pair too close.
        sorted_heights = sorted(flat_heights)
        tol = 1e-6
        for i in range(len(sorted_heights) - 1):
            if sorted_heights[i + 1] - sorted_heights[i] < tol:
                raise ValueError(
                    f"heights_mm must not contain heights closer than {tol} together -- "
                    f"found {sorted_heights[i]} and {sorted_heights[i + 1]} which differ by "
                    f"{sorted_heights[i + 1] - sorted_heights[i]}. Block identification by "
                    f"engraved height requires every height to be unique (within tolerance)"
                )
        if self.pitch_mm <= 0.0:
```

with:

```python
        if any(height < 0.0 for row in self.heights_mm for height in row):
            raise ValueError("heights_mm must not contain negative heights")
        if self.pitch_mm <= 0.0:
```

- [ ] **Step 2: Remove `cell_at_height`**

Remove the whole method (from `def cell_at_height(` through its closing `raise ValueError(...)`), i.e. replace:

```python
    def cell_at_height(self, height_mm: float, tol: float = 1e-6) -> Tuple[int, int]:
        """Reverse lookup: which (row, col) carries this engraved height.

        Every height in the grid is unique (enforced in __post_init__), so
        this is well-defined. Raises with the sorted list of valid heights on
        a miss, since a typo here would otherwise be silently indistinguishable
        from a real measurement.
        """
        for row in range(self.row_count):
            for col in range(self.col_count):
                if abs(self.heights_mm[row][col] - height_mm) <= tol:
                    return row, col
        valid = sorted({height for row in self.heights_mm for height in row})
        raise ValueError(
            f"no block at height {height_mm} mm (tol {tol}); valid heights: {valid}"
        )

    def pair_depths_mm(
```

with:

```python
    def pair_depths_mm(
```

- [ ] **Step 3: Update the module docstring's mention of `cell_at_height`**

Replace:

```
Unlike the line ladder, the number of points measured in a session is not
fixed. This target's own rig-specific geometry (a 30 mm height range packed
into a 50 mm footprint) makes self-occlusion from one or both cameras a real
possibility for some cells -- a session that only measures 15 of the grid's
cells is a normal, valid result, not a partial failure. Internally
correspondences are keyed by ``(row, col)`` grid index (never by click
order or count), but the user-facing identity is the height engraved on
each block -- ``cell_at_height`` resolves one to the other.
```

with:

```
Unlike the line ladder, the number of points measured in a session is not
fixed. This target's own rig-specific geometry (a 30 mm height range packed
into a 50 mm footprint) makes self-occlusion from one or both cameras a real
possibility for some cells -- a session that only measures 15 of the grid's
cells is a normal, valid result, not a partial failure. Correspondences are
keyed by ``(row, col)`` grid index, not by click order or count.
```

- [ ] **Step 4: Verify the real config still loads and duplicate heights are no longer rejected**

```bash
python3 - <<'EOF'
from calibration.depth_grid_target import DepthGridTarget

real = DepthGridTarget.from_yaml("calibration/config/depth_grid_target.yaml")
assert real.height_at(2, 1) == 7.0

# A grid with duplicate heights must now construct successfully -- the
# uniqueness constraint existed only to support the removed reverse lookup.
duplicate = DepthGridTarget(heights_mm=((1.0, 1.0),), pitch_mm=10.0,
                             reference_corner_count=3)
assert duplicate.height_at(0, 0) == duplicate.height_at(0, 1) == 1.0

assert not hasattr(DepthGridTarget, "cell_at_height")

print("Task 5 verification OK")
EOF
```

Expected: `Task 5 verification OK`.

- [ ] **Step 5: Commit**

```bash
git add calibration/depth_grid_target.py
git commit -m "$(cat <<'EOF'
Remove cell_at_height and its uniqueness check -- no remaining caller

check_depth_accuracy.py identifies blocks by row,col again (Task 4), so
the height-based reverse lookup this validation existed to support has
no caller left. Deleted rather than kept around unused.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: End-to-end verification against real calibrated extrinsics

**Files:** none (verification only; no code changes).

**Interfaces:** Consumes the finished CLI from Tasks 1-5 as a black box (subprocess), plus this rig's real `calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json`.

- [ ] **Step 1: Build a synthetic session and run the CLI subprocess end-to-end via the reverted `--ref`/`--cell ROW,COL,...` format**

Same forward-projection approach as every prior verification of this tool, driven through the real CLI as a subprocess (dummy images at this rig's real calibrated resolution, a small temp target file, `--ref`/`--cell` in the reverted `ROW,COL,AX,AY,BX,BY` format, and now 5 `--ref` clicks instead of the historical 4, proving the relaxed reference-count check works end-to-end too).

```bash
python3 - <<'EOF'
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from calibration.stereo import StereoExtrinsics

extrinsics = StereoExtrinsics.load_json(
    Path("calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json")
)
Ka, Kb, R = extrinsics.camera_matrix_a, extrinsics.camera_matrix_b, extrinsics.R
T = np.asarray(extrinsics.T).reshape(3)
width, height = extrinsics.image_size_a

def project(point_m, K):
    x, y, z = point_m
    return np.array([K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]])

def project_pair(point_a_m):
    return project(point_a_m, Ka), project(R @ point_a_m + T, Kb)

normal = np.array([0.05, -0.03, -0.998]); normal /= np.linalg.norm(normal)
centre = np.array([0.0, 0.0, 0.178])

def on_plane(dx_m, dy_m):
    p = centre + np.array([dx_m, dy_m, 0.0])
    return p - ((p - centre) @ normal) * normal

ref_corners_m = [
    on_plane(dx, dy) for dx, dy in
    [(-0.024, -0.024), (0.024, -0.024), (-0.024, 0.024), (0.024, 0.024), (0.0, 0.0)]
]
block36_m = on_plane(0.005, 0.0) + normal * 0.0036
block70_m = on_plane(-0.005, 0.005) + normal * 0.0070

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    session_dir = tmp / "20990101_000000"
    session_dir.mkdir()
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(session_dir / "rgb_cam1.jpg"), blank)
    cv2.imwrite(str(session_dir / "rgb_cam2.jpg"), blank)

    target_path = tmp / "target.yaml"
    target_path.write_text(yaml.safe_dump({
        "target_type": "depth_grid",
        "heights_mm": [[3.6, 7.0]],
        "pitch_mm": 10.0,
        "reference_corner_count": 4,
        "height_uncertainty_mm": 0.05,
        "measured_by": "synthetic test",
    }))

    ref_args = []
    for p in ref_corners_m:
        a, b = project_pair(p)
        ref_args += ["--ref", f"{a[0]},{a[1]},{b[0]},{b[1]}"]

    cell_args = []
    for point_m, row, col in ((block36_m, 0, 0), (block36_m, 0, 0), (block70_m, 0, 1)):
        a, b = project_pair(point_m)
        cell_args += ["--cell", f"{row},{col},{a[0]},{a[1]},{b[0]},{b[1]}"]

    out_dir = tmp / "out"
    proc = subprocess.run(
        [sys.executable, "check_depth_accuracy.py",
         "--session", str(session_dir), "--target", str(target_path),
         "--out", str(out_dir), *ref_args, *cell_args],
        cwd=Path.cwd(), capture_output=True, text=True,
    )
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    assert proc.returncode == 0, f"exit code {proc.returncode}"

    report = (out_dir / "depth_accuracy" / "report.txt").read_text()
    assert report.startswith("=" * 78 + "\nDEPTH-GRID RELATIVE ACCURACY\n" + "=" * 78 + "\nRESULT: scale error")
    assert "nan" not in report.lower()

    result = json.loads((out_dir / "depth_accuracy" / "result.json").read_text())
    summary = result["summary"]
    assert summary["scale_fit"] is not None
    assert abs(summary["scale_fit"]["scale_error_pct"]) < 1.0
    session_entry = next(s for s in result["sessions"] if s["label"] == "20990101_000000")
    assert session_entry["plane_point_count"] == 5  # not the old hardcoded 4
    per_block = {(entry["row"], entry["col"]): entry for entry in summary["per_block"]}
    assert abs(per_block[(0, 0)]["error_mm"]) < 0.05
    assert abs(per_block[(0, 1)]["error_mm"]) < 0.05

    assert (out_dir / "depth_accuracy" / "20990101_000000_correspondences.jpg").exists()

print("Task 6 verification OK")
EOF
```

Expected: `Task 6 verification OK`, subprocess stdout showing a `scale error` line close to `+0.000 %`, no `nan`, `plane_point_count` equal to `5`.

- [ ] **Step 2: Confirm no unintended files were left behind**

```bash
git status --short
```

Expected: clean (the subprocess test ran entirely inside a `tempfile.TemporaryDirectory`).

No commit for this task -- it verifies Tasks 1-5's combined commits, and touches no files of its own.
