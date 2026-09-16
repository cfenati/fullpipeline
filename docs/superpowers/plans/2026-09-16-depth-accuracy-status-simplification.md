# Unified on-screen status for check_depth_accuracy.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `check_depth_accuracy.py`'s scattered per-click terminal prints with one live, 3-line status strip inside the interactive session's own OpenCV window, and show a block's ground-truth height on screen the instant its row,col is typed (not buried in the final report).

**Architecture:** `measure_points.run_interactive` gains one new optional hook, `on_status`, which -- only when supplied -- replaces the window's generic length-measurement status text wholesale; the status-building logic is extracted into a standalone `build_status_lines` function so it's testable without a real window, same precedent as `handle_interactive_key`. `PlaneCollectionSession` (in `check_depth_accuracy.py`) implements `status_lines()` against this hook: it shows the current batch's task (with ground truth, for a block) plus the most recently completed step's result, instead of printing either to the terminal.

**Tech Stack:** Python 3.9, `opencv-contrib-python`, `numpy`; this repo's `measure_points.py`, `check_depth_accuracy.py`, `calibration.depth_grid_target.DepthGridTarget`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-16-depth-accuracy-status-simplification-design.md` (primary); `docs/superpowers/specs/2026-08-25-depth-accuracy-batch-collection-design.md` (background: `PlaneCollectionSession`'s existing batch/label bookkeeping, unchanged by this plan).
- No test suite or linter in this repo -- verify with throwaway `python3` heredoc scripts, run during implementation and **not committed**, same precedent as every prior plan touching this tool.
- `from __future__ import annotations` + type hints on every touched signature.
- `on_status` defaults to `None` everywhere; only `check_depth_accuracy.py`'s interactive branch passes it. `measure_points.py`'s own CLI and `measure_wound_depth.py` are not modified and stay behaviorally identical.
- `status_lines()` always returns exactly 3 lines (the middle "last result" line is `""` before anything has finished) -- the keys line must never shift position.
- No CLI flag gates this: it is a pure display/output-channel change (what's printed where), not a numerical method whose correctness could be in question, so there's nothing to fall back to.
- `measure_depth_session`, `aggregate_depth_results`, `write_report`, and the non-interactive `--ref`/`--cell` CLI path are not touched -- this plan only changes what's shown/printed during the interactive session, never the measurement math or the final report.
- The actual in-window rendering (legibility, text fitting `STATUS_HEIGHT=150` at font scale 0.52) cannot be exercised by script -- say so explicitly rather than claiming it's confirmed; it remains the user's next real-hardware step.

---

### Task 1: `on_status` hook + `build_status_lines` extraction in `measure_points.py`

**Files:**
- Modify: `measure_points.py`

**Interfaces:**
- Produces: `build_status_lines(state: Dict[str, Any], on_status: Optional[Callable[[], List[str]]]) -> List[str]` -- pure function, extracted from `run_interactive`'s main loop so it's testable with a synthetic state dict, no window needed. In text-entry mode, always returns the existing 2-line prompt/buffer display regardless of `on_status`. Otherwise: if `on_status` is not `None`, returns `on_status()`'s result wholesale; if `None`, falls back to the existing 4-line generic display unchanged.
- Produces: `run_interactive(..., on_status: Optional[Callable[[], List[str]]] = None) -> Dict[str, Any]` -- one new trailing optional parameter, appended after `on_text_submit`. When supplied, also suppresses the per-click `"point N -> N+1 : X mm ..."` terminal print in `on_mouse` (meaningless once the caller owns its own status display).
- Consumes: nothing new (no dependency on Task 2 within this plan).

- [ ] **Step 1: Extract `build_status_lines`, add `on_status` to `run_interactive`'s signature**

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
                    on_status: Optional[Callable[[], List[str]]] = None,
                    ) -> Dict[str, Any]:
```

Add a new module-level function immediately before `def run_interactive(`:

```python
def build_status_lines(
    state: Dict[str, Any], on_status: Optional[Callable[[], List[str]]],
) -> List[str]:
    """Build the window's bottom status-strip lines for one frame.

    Extracted from run_interactive's main loop so it's testable with a
    synthetic state dict, without a real window -- same reasoning as
    handle_interactive_key. Text-entry mode always shows the same 2-line
    prompt/buffer display regardless of on_status: that's generic
    input-mechanism chrome a caller's on_status has no reason to reimplement.
    Outside text-entry mode, on_status (when supplied) replaces the default
    4-line generic display wholesale -- a caller with its own session
    semantics (e.g. check_depth_accuracy.py's PlaneCollectionSession) owns
    the whole status strip instead of measure_points.py's own point-by-point
    narration, which is meaningless once clicks no longer describe a single
    running length measurement.
    """
    if state["text_mode"]:
        return [
            f"{state['text_prompt']}{state['text_buffer']}_",
            "type digits and , then Enter to confirm, Esc to cancel",
        ]
    if on_status is not None:
        return on_status()
    placed = len(state["clicks_a"])
    head = (f"point {placed}: click the SAME feature in camera B, near the blue line"
            if state["pending_a"] is not None
            else f"point {placed}: click a feature in camera A")
    recent = "   ".join(f"{i}->{i+1}: {d:.3f}mm"
                        for i, d in enumerate(state["consecutive"]))[-140:]
    return [
        head,
        f"measurements: {recent}" if recent else "measurements: (need two points)",
        ("wheel = zoom (fully independent)   right-drag = pan (fully independent)   "
         f"0 = fit both   l = aim B at each new A-click "
         f"{'ON' if state['linked'] else 'OFF'}"),
        "u undo | r reset | n advance/label | q or Esc = finish and print the report",
    ]
```

Replace the main loop's inline status-building block:

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
                ("wheel = zoom (fully independent)   right-drag = pan (fully independent)   "
                 f"0 = fit both   l = aim B at each new A-click "
                 f"{'ON' if state['linked'] else 'OFF'}"),
                "u undo | r reset | n advance/label | q or Esc = finish and print the report",
            ]
        cv2.imshow(window, render(state))
        key = cv2.waitKey(20)
```

with:

```python
    while True:
        state["status"] = build_status_lines(state, on_status)
        cv2.imshow(window, render(state))
        key = cv2.waitKey(20)
```

- [ ] **Step 2: Gate the generic per-click distance print on `on_status is None`**

Replace:

```python
            if state["consecutive"]:
                index = len(state["clicks_a"]) - 1
                print(f"  point {index - 1} -> {index} : "
                      f"{state['consecutive'][-1]:.3f} mm    "
                      f"depth {state['result']['depth_mm'][-1]:.1f} mm, "
                      f"click offset {raw_offset:.1f} px",
                      flush=True)
```

with:

```python
            if state["consecutive"] and on_status is None:
                index = len(state["clicks_a"]) - 1
                print(f"  point {index - 1} -> {index} : "
                      f"{state['consecutive'][-1]:.3f} mm    "
                      f"depth {state['result']['depth_mm'][-1]:.1f} mm, "
                      f"click offset {raw_offset:.1f} px",
                      flush=True)
```

- [ ] **Step 3: Verify `build_status_lines` directly (no window needed)**

```bash
python3 - <<'EOF'
from measure_points import build_status_lines

def fresh_state(**overrides):
    state = {
        "text_mode": False, "text_prompt": "", "text_buffer": "",
        "clicks_a": [], "pending_a": None, "consecutive": [], "linked": True,
    }
    state.update(overrides)
    return state

# Text-entry mode: always the 2-line prompt/buffer display, regardless of on_status.
state = fresh_state(text_mode=True, text_prompt="cell row,col > ", text_buffer="2,3")
lines = build_status_lines(state, on_status=lambda: ["should not be used"])
assert lines == ["cell row,col > 2,3_", "type digits and , then Enter to confirm, Esc to cancel"]

# No on_status: falls back to the existing generic 4-line display.
state = fresh_state()
lines = build_status_lines(state, on_status=None)
assert len(lines) == 4
assert lines[0] == "point 0: click a feature in camera A"

# on_status supplied, not in text mode: wholesale replacement, exactly what
# the callback returns.
custom = ["BLOCK (2, 3) -- ground truth 2.500mm -- 1 point(s) clicked, press n to finish",
          "last: reference plane fit, rms 0.0123mm from 4 points",
          "u undo | n finish batch | q/Esc quit"]
state = fresh_state()
lines = build_status_lines(state, on_status=lambda: custom)
assert lines == custom

print("Task 1 verification OK")
EOF
```

Expected: `Task 1 verification OK`.

- [ ] **Step 4: Syntax check**

```bash
python3 -c "import measure_points"
```

Expected: no output, exit code 0. This confirms the file imports and the extracted function behaves correctly under test -- it is not a claim that a real window was opened.

- [ ] **Step 5: Commit**

```bash
git add measure_points.py
git commit -m "$(cat <<'EOF'
Add on_status hook to run_interactive, extract build_status_lines

A caller with its own session semantics (check_depth_accuracy.py's
PlaneCollectionSession, next commit) needs to own the window's status
strip instead of measure_points.py's generic length-measurement text,
which is meaningless once clicks no longer describe one running length
measurement. build_status_lines is extracted from the main loop so this
is testable without a window, same precedent as handle_interactive_key.
Also suppresses the generic per-click distance print when on_status is
supplied, for the same reason.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `PlaneCollectionSession.status_lines()`, wired into `check_depth_accuracy.py`'s `main()`

**Files:**
- Modify: `check_depth_accuracy.py`

**Interfaces:**
- Consumes: Task 1's `run_interactive(..., on_status=)` and `build_status_lines`'s contract (wholesale replacement, exactly 3 lines expected by convention here).
- Produces: `PlaneCollectionSession.status_lines(self) -> List[str]`; new instance attribute `PlaneCollectionSession._last_result: str`, set by `on_advance`. Nothing else in this file changes shape -- `on_point`, `on_undo`, `on_text_submit`'s signatures and `batches` structure are exactly as before (see the 2026-08-25 batch-collection plan).

- [ ] **Step 1: Track `_last_result`, drop the now-redundant per-click/per-advance prints**

Replace `PlaneCollectionSession.__init__`:

```python
    def __init__(self, target: DepthGridTarget) -> None:
        self.target = target
        self.plane: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.batches: List[Tuple[Optional[Tuple[int, int]], List[int]]] = [(None, [])]
        self.points_mm: Dict[int, np.ndarray] = {}
```

with:

```python
    def __init__(self, target: DepthGridTarget) -> None:
        self.target = target
        self.plane: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.batches: List[Tuple[Optional[Tuple[int, int]], List[int]]] = [(None, [])]
        self.points_mm: Dict[int, np.ndarray] = {}
        # Most recently completed step's one-line summary (reference-plane
        # fit, or a finished block), shown by status_lines() -- overwritten
        # each time on_advance succeeds, never accumulated.
        self._last_result: str = ""
```

Replace `on_point`:

```python
    def on_point(self, index: int, result: Dict[str, Any]) -> None:
        self.points_mm[index] = np.asarray(result["points_mm"][index], dtype=np.float64)
        label, indices = self.batches[-1]
        indices.append(index)
        if label is None:
            print(f"  reference point {len(indices)} recorded")
        else:
            print(f"  point {len(indices)} recorded for block {label}")
```

with:

```python
    def on_point(self, index: int, result: Dict[str, Any]) -> None:
        self.points_mm[index] = np.asarray(result["points_mm"][index], dtype=np.float64)
        label, indices = self.batches[-1]
        indices.append(index)
```

Replace `on_advance`:

```python
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
```

with:

```python
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
            self._last_result = (
                f"reference plane fit, rms {rms_m * 1000.0:.4f}mm from {len(indices)} points"
            )
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
            truth = self.target.height_at(*label)
            if values.size > 1:
                spread = float(np.sqrt(np.mean((values - mean) ** 2)))
                self._last_result = (
                    f"block {label}: truth {truth:.3f}mm, measured {mean:.3f}mm "
                    f"(delta {mean - truth:+.3f}mm, {values.size} pts, spread {spread:.4f}mm rms)"
                )
            else:
                self._last_result = (
                    f"block {label}: truth {truth:.3f}mm, measured {mean:.3f}mm "
                    f"(delta {mean - truth:+.3f}mm)"
                )
        return "cell row,col > "
```

- [ ] **Step 2: Add `status_lines()`**

Insert immediately after `on_text_submit`'s closing `return None` (i.e. right before the class's end):

```python
    def status_lines(self) -> List[str]:
        """The window's whole status strip while this session drives it --
        see docs/superpowers/specs/2026-09-16-depth-accuracy-status-
        simplification-design.md. Always exactly 3 lines, so the keys line
        never shifts position: current task (with the block's ground truth
        height, visible the instant its row,col batch becomes current -- no
        separate print needed), the most recently completed step's result
        (empty until the first one finishes), then the key hints.
        """
        label, indices = self.batches[-1]
        if label is None:
            head = (f"REFERENCE PLANE -- {len(indices)} point(s) clicked (need >= 3), "
                    "press n when done")
        else:
            truth = self.target.height_at(*label)
            head = (f"BLOCK {label} -- ground truth {truth:.3f}mm -- "
                    f"{len(indices)} point(s) clicked, press n to finish")
        return [
            head,
            f"last: {self._last_result}" if self._last_result else "",
            "u undo | n finish batch | q/Esc quit",
        ]
```

- [ ] **Step 3: Wire `on_status` into `main()`'s `run_interactive` call, update the one-time instructions print**

Replace:

```python
            print(f"{label}: click reference points on the flat baseplate (at least 3, "
                  "spread out) then press n -- the plane fits and its rms prints "
                  "immediately, and you'll be prompted for a block's row,col right away. "
                  "Type it, Enter, then click that block's points (as many as you like, "
                  "even just one) and press n: it finalizes that block and immediately "
                  "prompts for the next block's row,col. Repeat for every block you can "
                  "see. Press q/Esc to finish.")
            session = PlaneCollectionSession(target)
            raw_result = run_interactive(
                image_a, image_b, extrinsics, depth_range,
                max(2, args.zoom), (int(args.window[0]), int(args.window[1])),
                max(3, args.blob_radius) if args.blob_snap else 0,
                float(reg_config.get("default_depth", 0.168)),
                on_point=session.on_point, on_undo=session.on_undo,
                on_advance=session.on_advance, on_text_submit=session.on_text_submit,
            )
```

with:

```python
            print(f"{label}: click reference points on the flat baseplate (at least 3, "
                  "spread out) then press n -- the plane fit and its rms appear in the "
                  "window's status bar, and you'll be prompted for a block's row,col "
                  "right away. Type it, Enter, and its ground truth height appears on "
                  "screen immediately -- double check it before clicking. Click that "
                  "block's points (as many as you like, even just one) and press n: the "
                  "measured-vs-truth result appears in the status bar and it immediately "
                  "prompts for the next block's row,col. Repeat for every block you can "
                  "see. Press q/Esc to finish.")
            session = PlaneCollectionSession(target)
            raw_result = run_interactive(
                image_a, image_b, extrinsics, depth_range,
                max(2, args.zoom), (int(args.window[0]), int(args.window[1])),
                max(3, args.blob_radius) if args.blob_snap else 0,
                float(reg_config.get("default_depth", 0.168)),
                on_point=session.on_point, on_undo=session.on_undo,
                on_advance=session.on_advance, on_text_submit=session.on_text_submit,
                on_status=session.status_lines,
            )
```

- [ ] **Step 4: Verify `status_lines()` against a synthetic driven session (no window needed)**

Same driving approach as the 2026-08-25 plan's `PlaneCollectionSession` verification: call the hooks directly, in the order `run_interactive` would.

```bash
python3 - <<'EOF'
import numpy as np
from calibration.depth_grid_target import DepthGridTarget
from check_depth_accuracy import PlaneCollectionSession

target = DepthGridTarget(heights_mm=((3.6, 7.0),), pitch_mm=10.0,
                          reference_corner_count=4, measured_by="synthetic test")
session = PlaneCollectionSession(target)

# Before anything: reference phase, 0 points, no "last" line yet.
lines = session.status_lines()
assert lines[0] == "REFERENCE PLANE -- 0 point(s) clicked (need >= 3), press n when done", lines[0]
assert lines[1] == ""
assert lines[2] == "u undo | n finish batch | q/Esc quit"

ref_points = [
    [-24.0, -24.0, 180.0], [24.0, -24.0, 180.0],
    [-24.0, 24.0, 180.0], [24.0, 24.0, 180.0],
]
for i in range(4):
    session.on_point(i, {"points_mm": ref_points[: i + 1]})
lines = session.status_lines()
assert lines[0] == "REFERENCE PLANE -- 4 point(s) clicked (need >= 3), press n when done"

prompt = session.on_advance()
assert prompt == "cell row,col > "
lines = session.status_lines()
assert lines[1].startswith("last: reference plane fit, rms "), lines[1]
assert "from 4 points" in lines[1]

# Submit a row,col: ground truth height must appear on screen IMMEDIATELY,
# before any of this block's points are clicked.
assert session.on_text_submit("0,0") is None
lines = session.status_lines()
assert lines[0] == "BLOCK (0, 0) -- ground truth 3.600mm -- 0 point(s) clicked, press n to finish", lines[0]

# 3 points for block (0,0), exactly on the z=180 plane offset by 3.6mm.
block_points = ref_points + [[0.0, 0.0, 176.4], [1.0, 1.0, 176.4], [2.0, 2.0, 176.4]]
for i in range(4, 7):
    session.on_point(i, {"points_mm": block_points[: i + 1]})
lines = session.status_lines()
assert lines[0] == "BLOCK (0, 0) -- ground truth 3.600mm -- 3 point(s) clicked, press n to finish"

session.on_advance()
lines = session.status_lines()
assert lines[1].startswith("last: block (0, 0): truth 3.600mm, measured "), lines[1]
assert "delta" in lines[1] and "spread" in lines[1]  # 3 points -> repeatability spread shown

# A single-point block: no "spread" text (matches the mean-only branch).
session.on_text_submit("0,1")
session.on_point(7, {"points_mm": block_points + [[5.0, 5.0, 173.0]]})
session.on_advance()
lines = session.status_lines()
assert lines[1].startswith("last: block (0, 1): truth 7.000mm, measured "), lines[1]
assert "spread" not in lines[1]

print("Task 2 verification OK")
EOF
```

Expected: `Task 2 verification OK`.

- [ ] **Step 5: Syntax check**

```bash
python3 -c "import check_depth_accuracy"
```

Expected: no output, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add check_depth_accuracy.py
git commit -m "$(cat <<'EOF'
Show ground truth + a unified live status in the depth-accuracy window

PlaneCollectionSession.status_lines() replaces the window's generic
length-measurement status text (via Task 1's on_status hook) with the
current batch's task -- including a block's ground-truth height, shown
the instant its row,col is typed, not just in the final report -- plus
the most recently completed step's result. The per-click/per-advance
terminal prints this replaces ("reference point N recorded", "block
(r,c): N points, mean...") are removed; that information now lives
permanently in the window instead of scrolling past in the terminal.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Notes for whoever runs this

Both tasks' verification is state-machine logic against synthetic inputs -- no camera or window required, consistent with every prior plan touching this tool. The actual in-window appearance (does the 3-line strip fit/read well at `STATUS_HEIGHT=150`) can only be judged by running `check_depth_accuracy.py --session <real capture>` on the rig; that is the user's own next step, not something this plan can verify.
