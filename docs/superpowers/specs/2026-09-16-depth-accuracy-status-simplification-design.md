# Design: unified on-screen status for check_depth_accuracy.py's interactive session

Date: 2026-09-16
Status: Proposed

Builds on `docs/superpowers/specs/2026-08-25-depth-accuracy-batch-collection-design.md`
(the batched multi-point `PlaneCollectionSession` this doc modifies) and the
shared window/status-bar machinery in `measure_points.py`'s `run_interactive`.
Read those first.

## Purpose

Two complaints about the current interactive session:

1. When you type a block's `ROW,COL`, nothing tells you the target's known
   ground-truth height for that cell. You can't sanity-check that you typed
   the label you meant to until the final report, long after the clicks are
   done.
2. The window's bottom status strip still shows `run_interactive`'s generic,
   length-measurement-flavored text (`"point N: click a feature in camera
   A"`, consecutive point-to-point distances) which is meaningless for the
   depth-grid flow -- clicks here span different blocks and planes, so a
   distance between consecutive clicks is not a quantity anyone cares about.
   Meanwhile the information that *does* matter (which batch you're on, how
   many points you've clicked for it, the last completed result) only exists
   as terminal `print()` calls from `PlaneCollectionSession`'s hooks, so it
   scrolls past and you have to look away from the window to see it.

Both are fixed the same way: give `PlaneCollectionSession` control over the
window's status strip instead of the terminal, and make the ground-truth
height part of that live status rather than a one-off print.

## Method

### 1. `on_status` hook on `run_interactive` (measure_points.py)

Add one new optional parameter:

```python
def run_interactive(..., on_status: Optional[Callable[[], List[str]]] = None) -> Dict[str, Any]:
```

In the main loop, the existing non-text-mode branch that builds
`state["status"]` (the `head` / `measurements` / view-controls-hint / keys
4-line block) only runs when `on_status is None`; when a caller supplies it,
`state["status"] = on_status()` instead, wholesale. Text-entry mode's status
(the `{prompt}{buffer}_` line + "type digits and , then Enter...") is
unchanged either way -- it's generic input-mechanism chrome, not something
`check_depth_accuracy.py` needs to own.

`on_mouse`'s existing per-click terminal print (`"point N -> N+1 : X mm ...
depth ... click offset ..."`) is gated on `on_status is None` too: it is
exactly the length-measurement artifact described above, and only meaningful
when `on_status` is absent (i.e. plain `measure_points.py` usage, or
`measure_wound_depth.py`, neither of which passes this new hook).

Nothing else in `measure_points.py` changes. `measure_points.py`'s own CLI and
`measure_wound_depth.py` do not pass `on_status`, so their behavior is
unaffected.

### 2. `PlaneCollectionSession` (check_depth_accuracy.py)

- New instance field `_last_result: str = ""`, holding a one-line human
  summary of the most recently completed step (reference-plane fit, or a
  block's finished measurement). Overwritten each time `on_advance` succeeds;
  never accumulates history.
- `on_point` no longer prints anything (removes `"reference point N
  recorded"` / `"point N recorded for block ..."` -- that count is now always
  visible live in `status_lines()`).
- `on_advance` no longer prints the success messages (`"reference plane fit
  from N points..."` / `"block (r,c): N points, mean ...")`; instead it sets
  `self._last_result` to the equivalent line. The existing failure-path
  prints (`"need at least 3 reference points..."`, `"click at least one point
  before pressing n"`) are unchanged -- they're rare validation messages tied
  to a specific keypress, not routine per-click narration, and match the
  pattern used elsewhere in this same file (e.g. `on_text_submit`'s invalid-
  label message).
- New method `status_lines(self) -> List[str]`, returning exactly 3 lines:
  1. Current task line:
     - Reference phase: `f"REFERENCE PLANE -- {n} point(s) clicked (need >= 3), press n when done"`
     - Block phase: `f"BLOCK {label} -- ground truth {truth:.3f}mm -- {n} point(s) clicked, press n to finish"`
       (`truth = self.target.height_at(*label)`, looked up fresh every frame,
       so it's on screen the instant the block's `ROW,COL` batch becomes
       current -- no separate print needed to satisfy "show me the ground
       truth as soon as I type it in.")
  2. Last-result line (empty string before anything has finished -- still
     returned, so the line count is always exactly 3 and the keys line
     doesn't shift position):
     `f"last: reference plane fit, rms {rms:.4f}mm from {n} points"`, later
     replaced by
     `f"last: block {label}: truth {truth:.3f}mm, measured {mean:.3f}mm (delta {mean-truth:+.3f}mm)"`
     (repeatability spread appended when the block had > 1 point, same as
     today's terminal message).
  3. Keys line: `"u undo | n finish batch | q/Esc quit"`.
- `check_depth_accuracy.py`'s call to `run_interactive` gains
  `on_status=session.status_lines`.

### 3. Explicitly unchanged

- The one-time instructional print before the window opens (`"{label}: click
  reference points on the flat baseplate..."`) stays -- it's a one-time
  orientation the 3-line status bar has no room for, not per-click noise.
- Non-interactive `--ref`/`--cell` CLI usage: untouched, never opens a window.
- `measure_points.py` standalone CLI and `measure_wound_depth.py`: untouched,
  neither passes `on_status`.
- Final end-of-session terminal summary (`"{label}: scale ..., blocks ...,
  plane rms ..., depth ..."`, printed once after the window closes) and
  `report.txt`/`result.json`: unchanged, still the authoritative record.
- No CLI flag added to toggle this. Unlike the registration heuristics this
  project has previously gated behind `--no-X` flags, this is a pure
  display/output-channel change (what's printed where), not a numerical
  method whose correctness is in question -- there's nothing to fall back to
  if it turns out wrong, and the underlying measurement math
  (`measure_depth_session`, `fit_plane_3d`) is untouched.

## Testing

No hardware needed: `status_lines()` and the updated `on_advance`/`on_point`
are pure state-machine logic over a `PlaneCollectionSession` instance and
synthetic point data, exactly like the existing coverage for this class (see
the 2026-08-25 batch-collection plan's verification approach). `on_status`
being `None` by default means `measure_points.py --session ...` and
`measure_wound_depth.py`'s existing non-interactive/CLI verification paths
are unaffected and don't need re-checking beyond a quick import/syntax check.

The actual in-window rendering (does the text fit, is it legible at
`STATUS_HEIGHT=150`/font scale 0.52) cannot be exercised without the rig's
cameras -- per this repo's `CLAUDE.md`, that remains the user's next real-
hardware step, same as the rest of this script's interactive path.
