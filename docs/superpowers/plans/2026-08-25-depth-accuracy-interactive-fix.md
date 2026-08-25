# Fix check_depth_accuracy.py interactive labeling and diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `check_depth_accuracy.py` so a real capture session produces a trustworthy result: label each block live (by the height engraved on it) instead of blind post-hoc typing, fix the epipolar-offset diagnostic that always read 0.0, and report one clear headline + one core per-block table instead of the noisy multi-table report that hid a `scale = nan` bug.

**Architecture:** `measure_points.run_interactive` gains two small optional hooks (`on_point`, `on_undo`) so a caller can label and grade each click live without forking the click/loupe/epipolar UI. `check_depth_accuracy.py`'s `LabelingSession` uses those hooks to fit the reference plane the instant the last corner lands, then prompts for each block's engraved height and grades it immediately. `measure_depth_session` groups same-labeled clicks into one block entry (mean + repeatability RMS) before any pairwise/scale-fit math runs, which is what makes the `truth @ truth == 0` NaN structurally unreachable.

**Tech Stack:** Python 3.9, `opencv-contrib-python`, `numpy`, `PyYAML` (already in `requirements.txt`); this repo's `calibration.stereo.StereoExtrinsics`, `check_line_accuracy.fit_scale`, `triangulate.py`.

## Global Constraints

- Specs: `docs/superpowers/specs/2026-08-25-depth-accuracy-interactive-fix-design.md` (primary) and `docs/superpowers/specs/2026-08-24-check-depth-accuracy-design.md` (background — physical grid, plane-fit method).
- No test suite or linter in this repo (per `CLAUDE.md`) — verify with throwaway `python3` heredoc scripts against synthetic data and this rig's real calibrated extrinsics, run during implementation and **not committed** — same precedent as `docs/superpowers/plans/2026-08-24-check-depth-accuracy.md` and `2026-08-19-check-line-accuracy.md`.
- `from __future__ import annotations` + type hints on every signature.
- RGB cameras only, defaults `rgb_cam1`/`rgb_cam2`.
- Depth stays relative-only: every reported quantity is a difference (block-to-plane or block-to-block), never a raw absolute Z.
- The reference-plane fit must complete (all `reference_corner_count` corners clicked) before any block measurement is meaningful — the plane isolates "height" from "sideways" grid-position offset; see the 2026-08-25 spec's "Why the reference plane comes first."
- Every height in a `DepthGridTarget` must be unique — block identity is now looked up by engraved height, so a duplicate would be silently ambiguous. Enforce at load time, don't just assume it.
- Do not modify `check_line_accuracy.py` (only `fit_scale` is imported, unmodified).
- GUI/mouse-driven code (the `on_mouse` wiring inside `run_interactive`, `LabelingSession` as invoked through a real window) cannot be exercised headlessly. Verify the math and the callback logic directly (calling hook methods with synthetic arguments); say explicitly that the real click-through-a-window path is untested by script, per `CLAUDE.md`'s rule against claiming unverified GUI code works.

---

### Task 1: Fix the epipolar-offset diagnostic and add interactive hooks

**Files:**
- Modify: `triangulate.py` (add `distance_to_line`, refactor `snap_to_line` to use it)
- Modify: `measure_points.py` (use `distance_to_line`; add `on_point`/`on_undo` hooks to `run_interactive`; capture the real pre-snap offset)

**Interfaces:**
- Produces: `triangulate.distance_to_line(points: np.ndarray, lines: np.ndarray) -> np.ndarray` — signed perpendicular distance (pixels) from each point to its normalized line.
- Produces: `measure_points.run_interactive(..., on_point: Optional[Callable[[int, Dict[str, Any]], None]] = None, on_undo: Optional[Callable[[int], None]] = None) -> Dict[str, Any]` — `on_point(index, result)` fires right after point `index` completes; `result["epipolar_offset_px"]` is now the real per-point offset, not a trivial recomputation. `on_undo(new_count)` fires whenever `u`/`r` changes the completed point count.

- [ ] **Step 1: Add `distance_to_line` to `triangulate.py`, refactor `snap_to_line` to use it**

In `triangulate.py`, replace:

```python
def snap_to_line(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Closest point on each line: drops the click error perpendicular to it."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 3)
    offset = np.einsum("ij,ij->i", np.column_stack([points, np.ones(len(points))]), lines)
    return points - offset[:, None] * lines[:, :2]
```

with:

```python
def distance_to_line(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Signed perpendicular distance from each point to its line, in pixels.

    Lines are normalised (a^2 + b^2 = 1, as epipolar_lines returns), so this
    homogeneous residual is a true distance, not just a scaled one.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 3)
    return np.einsum("ij,ij->i", np.column_stack([points, np.ones(len(points))]), lines)


def snap_to_line(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Closest point on each line: drops the click error perpendicular to it."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 3)
    offset = distance_to_line(points, lines)
    return points - offset[:, None] * lines[:, :2]
```

- [ ] **Step 2: Verify `distance_to_line` against a known distance**

Run (not committed):

```bash
python3 - <<'EOF'
import numpy as np
from triangulate import distance_to_line, snap_to_line

# Line y=0 normalised as (a,b,c)=(0,1,0). Point (5,3) is 3px from it.
line = np.array([[0.0, 1.0, 0.0]])
point = np.array([[5.0, 3.0]])
d = distance_to_line(point, line)
assert abs(d[0] - 3.0) < 1e-9, d
snapped = snap_to_line(point, line)
assert np.allclose(snapped, [[5.0, 0.0]]), snapped
print("Task 1 step 2 OK:", d, snapped)
EOF
```

Expected: `Task 1 step 2 OK: [3.] [[5. 0.]]`

- [ ] **Step 3: Use `distance_to_line` in `measure_points.py`'s non-interactive path**

In `measure_points.py`, update the import:

```python
from triangulate import (  # noqa: E402
    distance_to_line,
    epipolar_lines,
    fundamental_for_undistorted,
    snap_to_line,
    triangulate_points,
)
```

In `measure_points()`, replace:

```python
    offsets = np.abs(np.einsum(
        "ij,ij->i", np.column_stack([clicks_b, np.ones(len(clicks_b))]), lines,
    ))
```

with:

```python
    offsets = np.abs(distance_to_line(clicks_b, lines))
```

- [ ] **Step 4: Add `Callable`/`Optional` to typing imports**

`measure_points.py`'s typing import currently reads
`from typing import Any, Dict, List, Optional, Sequence, Tuple` — add `Callable`:

```python
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
```

- [ ] **Step 5: Add `click_offsets_px` to interactive state, capture the raw offset before snapping**

In `run_interactive`'s `state` dict literal, add `"click_offsets_px": []` right after `"consecutive": [],`:

```python
    state: Dict[str, Any] = {
        "panel_a": Panel(image_a, panel_size), "panel_b": Panel(image_b, panel_size),
        "zoom": zoom, "clicks_a": [], "clicks_b": [], "pending_a": None,
        "pending_line": None, "cursor": None, "status": [], "consecutive": [],
        "click_offsets_px": [],
        "drag": None, "linked": True,
        "depth_hint": depth_m,
    }
```

Update `recompute()` to override the trivial recomputed offset with the real captured one:

```python
    def recompute() -> None:
        state["consecutive"] = []
        if len(state["clicks_b"]) >= 1:
            result = measure_points(
                np.array(state["clicks_a"]), np.array(state["clicks_b"]), extrinsics,
            )
            result["epipolar_offset_px"] = list(state["click_offsets_px"])
            points = np.array(result["points_mm"])
            state["consecutive"] = [
                float(np.linalg.norm(points[i] - points[i - 1]))
                for i in range(1, len(points))
            ]
            state["result"] = result
```

- [ ] **Step 6: Add the `on_point`/`on_undo` parameters and capture the raw offset at click time**

Change `run_interactive`'s signature (add the two trailing optional params):

```python
def run_interactive(image_a: np.ndarray, image_b: np.ndarray,
                    extrinsics: StereoExtrinsics, depth_range: Tuple[float, float],
                    zoom: int, max_window: Tuple[int, int], blob_radius: int,
                    depth_m: float,
                    on_point: Optional[Callable[[int, Dict[str, Any]], None]] = None,
                    on_undo: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
```

In `on_mouse`'s `on_b` branch, replace:

```python
        elif on_b:
            # Centroid first (the dot's true centre), then the epipolar line
            # (which discards whatever click error is left across the line).
            blob = snap_to_blob(gray_b, full, blob_radius)
            snapped = snap_to_line(blob[None, :], state["pending_line"][None, :])[0]
            state["clicks_a"].append(state["pending_a"])
            state["clicks_b"].append(snapped.tolist())
            state["pending_a"] = None
            recompute()
            state["depth_hint"] = float(state["result"]["depth_mm"][-1]) / 1000.0
            if state["consecutive"]:
                index = len(state["clicks_a"]) - 1
                print(f"  point {index - 1} -> {index} : "
                      f"{state['consecutive'][-1]:.3f} mm    "
                      f"depth {state['result']['depth_mm'][-1]:.1f} mm, "
                      f"click offset {state['result']['epipolar_offset_px'][-1]:.1f} px",
                      flush=True)
```

with:

```python
        elif on_b:
            # Centroid first (the dot's true centre), then the epipolar line
            # (which discards whatever click error is left across the line) --
            # capture the distance to that line HERE, before it's discarded.
            blob = snap_to_blob(gray_b, full, blob_radius)
            raw_offset = float(np.abs(
                distance_to_line(blob[None, :], state["pending_line"][None, :])[0]
            ))
            snapped = snap_to_line(blob[None, :], state["pending_line"][None, :])[0]
            state["clicks_a"].append(state["pending_a"])
            state["clicks_b"].append(snapped.tolist())
            state["click_offsets_px"].append(raw_offset)
            state["pending_a"] = None
            recompute()
            state["depth_hint"] = float(state["result"]["depth_mm"][-1]) / 1000.0
            if state["consecutive"]:
                index = len(state["clicks_a"]) - 1
                print(f"  point {index - 1} -> {index} : "
                      f"{state['consecutive'][-1]:.3f} mm    "
                      f"depth {state['result']['depth_mm'][-1]:.1f} mm, "
                      f"click offset {raw_offset:.1f} px",
                      flush=True)
            if on_point is not None:
                on_point(len(state["clicks_a"]) - 1, state["result"])
```

- [ ] **Step 7: Keep `click_offsets_px` in sync on undo/reset, call `on_undo`**

Replace the `u` key handler:

```python
        if key == ord("u"):
            if state["pending_a"] is not None:
                state["pending_a"] = None
            elif state["clicks_b"]:
                state["clicks_a"].pop(); state["clicks_b"].pop(); recompute()
```

with:

```python
        if key == ord("u"):
            if state["pending_a"] is not None:
                state["pending_a"] = None
            elif state["clicks_b"]:
                state["clicks_a"].pop(); state["clicks_b"].pop()
                state["click_offsets_px"].pop()
                recompute()
                if on_undo is not None:
                    on_undo(len(state["clicks_a"]))
```

Replace the `r` key handler:

```python
        elif key == ord("r"):
            state["clicks_a"].clear(); state["clicks_b"].clear()
            state["pending_a"] = None; recompute()
```

with:

```python
        elif key == ord("r"):
            state["clicks_a"].clear(); state["clicks_b"].clear()
            state["click_offsets_px"].clear()
            state["pending_a"] = None; recompute()
            if on_undo is not None:
                on_undo(0)
```

- [ ] **Step 8: Override the final return's `epipolar_offset_px` too**

Replace:

```python
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if len(state["clicks_b"]) < 2:
        return {}
    return measure_points(
        np.array(state["clicks_a"]), np.array(state["clicks_b"]), extrinsics,
    )
```

with:

```python
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if len(state["clicks_b"]) < 2:
        return {}
    result = measure_points(
        np.array(state["clicks_a"]), np.array(state["clicks_b"]), extrinsics,
    )
    result["epipolar_offset_px"] = list(state["click_offsets_px"])
    return result
```

- [ ] **Step 9: Syntax check**

```bash
python3 -c "import measure_points, triangulate"
```

Expected: no output, exit code 0. Note explicitly: the `on_mouse`/hook wiring itself requires a real window and mouse events to exercise — this check only confirms the file imports; it is not a claim that the interactive path was run.

- [ ] **Step 10: Commit**

```bash
git add triangulate.py measure_points.py
git commit -m "$(cat <<'EOF'
Fix epipolar-offset diagnostic, add on_point/on_undo hooks to run_interactive

run_interactive stored clicks_b only after snapping onto the epipolar
line, so every later recomputation of epipolar_offset_px read ~0
regardless of how far off the original click was. distance_to_line
(factored out of snap_to_line) now captures the real offset at click
time, before it's discarded. Also adds optional on_point/on_undo hooks
so a caller can label and grade each click live, without forking the
click/loupe/epipolar UI -- needed by check_depth_accuracy.py next.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `DepthGridTarget` reverse height lookup + uniqueness validation

**Files:**
- Modify: `calibration/depth_grid_target.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `DepthGridTarget.cell_at_height(height_mm: float, tol: float = 1e-6) -> Tuple[int, int]`. `__post_init__` now raises `ValueError` if any two heights in the grid are equal.

- [ ] **Step 1: Reject a grid with duplicate heights in `__post_init__`**

In `calibration/depth_grid_target.py`, right after the existing negative-height check, add:

```python
        if any(height < 0.0 for row in self.heights_mm for height in row):
            raise ValueError("heights_mm must not contain negative heights")
        flat_heights = [height for row in self.heights_mm for height in row]
        if len(set(flat_heights)) != len(flat_heights):
            raise ValueError(
                "heights_mm must not contain duplicate heights -- block "
                "identification by engraved height requires every height to "
                "be unique"
            )
```

- [ ] **Step 2: Add `cell_at_height`, right after `height_at`**

```python
    def height_at(self, row: int, col: int) -> float:
        return self.heights_mm[row][col]

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
```

- [ ] **Step 3: Verify against a synthetic target and the real config**

```bash
python3 - <<'EOF'
from calibration.depth_grid_target import DepthGridTarget

target = DepthGridTarget(heights_mm=((1.0, 2.0), (3.0, 4.0)), pitch_mm=10.0,
                          reference_corner_count=4)
assert target.cell_at_height(3.0) == (1, 0)
assert target.cell_at_height(2.0000001) == (0, 1)  # within default tol
try:
    target.cell_at_height(9.9)
    raise AssertionError("expected ValueError for a height not in the grid")
except ValueError as exc:
    assert "valid heights" in str(exc), exc

try:
    DepthGridTarget(heights_mm=((1.0, 1.0),), pitch_mm=10.0, reference_corner_count=3)
    raise AssertionError("expected ValueError for duplicate heights")
except ValueError as exc:
    assert "duplicate" in str(exc), exc

# The real config must still load: all 25 heights there are already distinct.
real = DepthGridTarget.from_yaml("calibration/config/depth_grid_target.yaml")
assert real.cell_at_height(7.0) == (2, 1)
print("Task 2 verification OK")
EOF
```

Expected: `Task 2 verification OK`.

- [ ] **Step 4: Commit**

```bash
git add calibration/depth_grid_target.py
git commit -m "$(cat <<'EOF'
Add DepthGridTarget.cell_at_height, enforce unique heights

check_depth_accuracy.py is about to identify a block by the number
engraved on it rather than by (row, col) -- that only works if every
height in the grid is unique, so it's now enforced at load time instead
of just assumed.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Rework `measure_depth_session` to group same-label clicks

**Files:**
- Modify: `check_depth_accuracy.py`

**Interfaces:**
- Consumes: `DepthGridTarget.cell_at_height` (Task 2, not used inside this function but by its caller later); unchanged `fit_plane_3d`, `perpendicular_distance_to_plane`, `measure_points.measure_points` (`triangulate_clicks`).
- Produces: `measure_depth_session(...) -> Dict[str, Any]` now returns block-level keys instead of per-click cell keys: `block_labels: List[Tuple[int,int]]`, `block_truth_mm: np.ndarray`, `block_measured_mm: np.ndarray`, `block_samples: np.ndarray[int]`, `block_repeatability_rms_mm: List[Optional[float]]` (parallel to `block_labels`, `None` when that block had only 1 sample), plus unchanged `pairs`, `pair_truth_mm`, `pair_measured_mm`, `plane_rms_mm`, `plane_point_count`, `ref_epipolar_offset_max_px`, `cell_epipolar_offset_max_px`, `order_mismatch_count`, `depth_mean_m`, `ref_clicks_a`, `ref_clicks_b`, `ref_points_mm`, `cell_clicks_a`, `cell_clicks_b`, `cell_points_mm` (the last three keep the raw per-click arrays, still needed for the annotated-JPEG output). The minimum-count error now allows 1 labeled click, not 2.

- [ ] **Step 1: Replace `measure_depth_session`'s body**

In `check_depth_accuracy.py`, replace the whole function (from `def measure_depth_session(` through its closing `return {...}` block) with:

```python
def measure_depth_session(
    ref_clicks_a: np.ndarray,
    ref_clicks_b: np.ndarray,
    cell_labels: Sequence[Tuple[int, int]],
    cell_clicks_a: np.ndarray,
    cell_clicks_b: np.ndarray,
    extrinsics: StereoExtrinsics,
    target: DepthGridTarget,
) -> Dict[str, Any]:
    """Fit the baseplate plane from the reference clicks, then measure every
    labeled block's signed perpendicular distance to it.

    ``cell_labels`` pairs 1:1 by index with ``cell_clicks_a``/``cell_clicks_b``
    -- that pairing IS the correspondence, not click order. A label repeated
    across multiple clicks is a repeatability sample of the SAME block, not a
    second block -- clicks are grouped by label before any pairwise or
    scale-fit computation runs, so a pairwise truth of 0 mm (which broke the
    scale fit before this rewrite) can no longer happen from a mislabeled
    repeat.
    """
    ref_clicks_a = np.asarray(ref_clicks_a, dtype=np.float64).reshape(-1, 2)
    ref_clicks_b = np.asarray(ref_clicks_b, dtype=np.float64).reshape(-1, 2)
    cell_clicks_a = np.asarray(cell_clicks_a, dtype=np.float64).reshape(-1, 2)
    cell_clicks_b = np.asarray(cell_clicks_b, dtype=np.float64).reshape(-1, 2)

    if len(ref_clicks_a) != target.reference_corner_count:
        raise ValueError(
            f"expected {target.reference_corner_count} reference clicks, "
            f"got {len(ref_clicks_a)}"
        )
    if not (len(cell_labels) == len(cell_clicks_a) == len(cell_clicks_b)):
        raise ValueError("cell_labels, cell_clicks_a, cell_clicks_b must be the same length")
    if len(cell_labels) < 1:
        raise ValueError(f"need at least 1 labeled block click, got {len(cell_labels)}")

    ref_result = triangulate_clicks(ref_clicks_a, ref_clicks_b, extrinsics)
    reference_points_mm = np.asarray(ref_result["points_mm"], dtype=np.float64)
    centroid, normal, plane_rms_m = fit_plane_3d(reference_points_mm / 1000.0)
    # Orient toward camera A's optical centre (the origin in this frame), so a
    # block protruding toward the camera reads a POSITIVE depth, matching the
    # target's own height_mm sign convention.
    if normal @ (-centroid) < 0.0:
        normal = -normal

    cell_result = triangulate_clicks(cell_clicks_a, cell_clicks_b, extrinsics)
    cell_points_mm = np.asarray(cell_result["points_mm"], dtype=np.float64)
    cell_measured_mm = np.array([
        perpendicular_distance_to_plane(point / 1000.0, centroid, normal) * 1000.0
        for point in cell_points_mm
    ])

    # Group repeat clicks on the same block into one entry: mean measured
    # height, sample count, and repeatability RMS (None for a single sample).
    unique_labels: List[Tuple[int, int]] = []
    block_measured_list: List[float] = []
    block_truth_list: List[float] = []
    block_samples_list: List[int] = []
    block_repeatability_list: List[Any] = []
    for label in cell_labels:
        if label in unique_labels:
            continue
        indices = [index for index, other in enumerate(cell_labels) if other == label]
        samples = cell_measured_mm[indices]
        unique_labels.append(label)
        block_measured_list.append(float(samples.mean()))
        block_truth_list.append(target.height_at(*label))
        block_samples_list.append(int(samples.size))
        block_repeatability_list.append(
            float(np.sqrt(np.mean((samples - samples.mean()) ** 2)))
            if samples.size > 1 else None
        )
    block_measured_mm = np.array(block_measured_list)
    block_truth_mm = np.array(block_truth_list)
    block_samples = np.array(block_samples_list)

    pairs, pair_truth_mm = target.pair_depths_mm(unique_labels)
    pair_measured_mm = block_measured_mm[pairs[:, 1]] - block_measured_mm[pairs[:, 0]]

    # Best-effort order-mismatch warning, at block granularity: all of the
    # target's heights are distinct, so the clicked blocks have a well-defined
    # true rank order to check the measured order against.
    if len(unique_labels) > 1:
        rank = np.argsort(block_truth_mm)
        ranked_measured = block_measured_mm[rank]
        order_mismatch_count = int(np.sum(np.diff(ranked_measured) < 0))
    else:
        order_mismatch_count = 0

    all_points_m = np.concatenate([reference_points_mm, cell_points_mm]) / 1000.0

    return {
        "block_labels": unique_labels,
        "block_truth_mm": block_truth_mm,
        "block_measured_mm": block_measured_mm,
        "block_samples": block_samples,
        "block_repeatability_rms_mm": block_repeatability_list,
        "pairs": pairs,
        "pair_truth_mm": pair_truth_mm,
        "pair_measured_mm": pair_measured_mm,
        "plane_rms_mm": plane_rms_m * 1000.0,
        "plane_point_count": len(ref_clicks_a),
        "ref_epipolar_offset_max_px": float(np.max(ref_result["epipolar_offset_px"])),
        "cell_epipolar_offset_max_px": float(np.max(cell_result["epipolar_offset_px"])),
        "order_mismatch_count": order_mismatch_count,
        "depth_mean_m": float(np.mean(all_points_m[:, 2])),
        "ref_clicks_a": ref_clicks_a,
        "ref_clicks_b": np.asarray(ref_result["clicks_b_snapped"]),
        "ref_points_mm": reference_points_mm,
        "cell_clicks_a": cell_clicks_a,
        "cell_clicks_b": np.asarray(cell_result["clicks_b_snapped"]),
        "cell_points_mm": cell_points_mm,
    }
```

- [ ] **Step 2: Add `Optional` to `check_depth_accuracy.py`'s typing import**

Current: `from typing import Any, Dict, List, Sequence, Tuple` — change to:

```python
from typing import Any, Dict, List, Optional, Sequence, Tuple
```

(`Optional` is needed by Task 5's `LabelingSession`; adding it now keeps this task's diff self-contained since it touches the same import line.)

- [ ] **Step 3: Verify against real calibrated extrinsics with a forward-projected synthetic session**

This projects synthetic 3D points (on a deliberately tilted plane, so the geometry isn't trivially axis-aligned) through this rig's real `K`/`R`/`T` to get pixel clicks, then checks the new grouping logic recovers the right blocks with the right repeatability behaviour.

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
    pa = project(point_a_m, Ka)
    pb = project(R @ point_a_m + T, Kb)
    return pa, pb

normal = np.array([0.05, -0.03, -0.998]); normal /= np.linalg.norm(normal)
centre = np.array([0.0, 0.0, 0.178])

def on_plane(dx_m, dy_m):
    p = centre + np.array([dx_m, dy_m, 0.0])
    return p - ((p - centre) @ normal) * normal

corners_m = [on_plane(dx, dy) for dx in (-0.024, 0.024) for dy in (-0.024, 0.024)]
ref_clicks_a, ref_clicks_b = [], []
for p in corners_m:
    a, b = project_pair(p)
    ref_clicks_a.append(a); ref_clicks_b.append(b)

target = DepthGridTarget(heights_mm=((3.6, 7.0),), pitch_mm=10.0,
                          reference_corner_count=4, height_uncertainty_mm=0.05,
                          measured_by="synthetic test")

# normal is oriented toward the camera (origin) by construction here, since
# normal @ (-centre) > 0 already -- matches measure_depth_session's own
# orientation fix, so +height moves a point toward the camera.
block36_m = on_plane(0.005, 0.0) + normal * 0.0036
block70_m = on_plane(-0.005, 0.005) + normal * 0.0070
cell_labels = [(0, 0), (0, 0), (0, 1)]
cell_clicks_a, cell_clicks_b = [], []
for point_m, jitter_px in ((block36_m, 0.0), (block36_m, 0.4), (block70_m, 0.0)):
    a, b = project_pair(point_m)
    a = a + np.array([jitter_px, 0.0])
    cell_clicks_a.append(a); cell_clicks_b.append(b)

result = measure_depth_session(
    np.array(ref_clicks_a), np.array(ref_clicks_b), cell_labels,
    np.array(cell_clicks_a), np.array(cell_clicks_b), extrinsics, target,
)
print("block_labels", result["block_labels"])
print("block_samples", result["block_samples"])
print("block_measured_mm", result["block_measured_mm"])
print("block_repeatability_rms_mm", result["block_repeatability_rms_mm"])
print("pair_truth_mm", result["pair_truth_mm"])

assert result["block_labels"] == [(0, 0), (0, 1)]
assert result["block_samples"].tolist() == [2, 1]
assert abs(result["block_measured_mm"][0] - 3.6) < 0.05
assert abs(result["block_measured_mm"][1] - 7.0) < 0.05
assert result["block_repeatability_rms_mm"][0] is not None
assert result["block_repeatability_rms_mm"][0] > 0.0
assert result["block_repeatability_rms_mm"][1] is None
assert 0.0 not in [round(t, 6) for t in result["pair_truth_mm"]]
assert abs(result["pair_truth_mm"][0] - 3.4) < 1e-9  # 7.0 - 3.6

# A single-block session (pure repeatability probe) must succeed, not raise --
# this was previously a ValueError ("need at least 2 cells").
one_block = measure_depth_session(
    np.array(ref_clicks_a), np.array(ref_clicks_b), [(0, 0), (0, 0)],
    np.array(cell_clicks_a[:2]), np.array(cell_clicks_b[:2]), extrinsics, target,
)
assert len(one_block["pairs"]) == 0
assert one_block["block_labels"] == [(0, 0)]
print("Task 3 verification OK")
EOF
```

Expected: `Task 3 verification OK`, with `block_measured_mm` close to `[3.6, 7.0]` and a nonzero `block_repeatability_rms_mm[0]`.

- [ ] **Step 4: Commit**

```bash
git add check_depth_accuracy.py
git commit -m "$(cat <<'EOF'
Group same-label clicks in measure_depth_session before pairwise math

Repeat clicks on one block (intentional, for repeatability, or an old
mislabeling accident) used to feed the pairwise truth/measured table as
if they were separate blocks, producing 0mm-truth pairs that made
fit_scale's truth @ truth divide-by-zero into nan. Grouping by label
first makes that structurally unreachable, and turns repeats into a
first-class block_repeatability_rms_mm instead of silent noise. Also
relaxes the minimum from 2 labeled clicks to 1, since a single-block
repeatability-only session is now a valid, scored result.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Rework `aggregate_depth_results` and `write_report`

**Files:**
- Modify: `check_depth_accuracy.py`

**Interfaces:**
- Consumes: Task 3's `measure_depth_session` return shape (`block_labels`, `block_truth_mm`, `block_measured_mm`, `pairs`, `pair_truth_mm`, `pair_measured_mm`, `plane_rms_mm`, `plane_point_count`, `order_mismatch_count`, `depth_mean_m`); `check_line_accuracy.fit_scale` (unchanged).
- Produces: `aggregate_depth_results(...) -> Dict[str, Any]` with keys `sessions_scored`, `per_pair` (list of `{truth_mm, samples, measured_mean_mm, error_mm}`), `per_block` (list of `{row, col, truth_mm, samples, measured_mean_mm, repeatability_rms_mm (float or None), error_mm}`), `per_session` (list of `{label, blocks_measured, plane_rms_mm, plane_point_count, order_mismatch_count, depth_mean_m}`), `scale_fit` (the `fit_scale` dict, or `None` if no session produced any pair), `rms_error_mm`, `max_abs_error_mm`, `repeatability_rms_mm` (float or `None`), `plane_rms_mean_mm`, `order_mismatch_total`. `write_report(summary, results, target, context, path)` signature is unchanged.

- [ ] **Step 1: Replace `aggregate_depth_results`**

Replace the whole function with:

```python
def aggregate_depth_results(
    results: Sequence[Dict[str, Any]], target: DepthGridTarget,
) -> Dict[str, Any]:
    """Pool per-session pairwise and per-block measurements, then fit scale.

    Each session already deduplicates repeat clicks on the same block into
    one mean (measure_depth_session) -- this pools those per-session means
    across sessions, so the reported repeatability_rms_mm answers "if I
    measure this block on separate occasions, how much does the answer
    vary," on top of (not instead of) each session's own within-session
    block_repeatability_rms_mm.
    """
    scored = [result for result in results if "skipped" not in result]

    pair_buckets: Dict[float, List[float]] = {}
    for result in scored:
        for truth, measured in zip(result["pair_truth_mm"], result["pair_measured_mm"]):
            pair_buckets.setdefault(round(float(truth), 6), []).append(float(measured))

    per_pair: List[Dict[str, Any]] = []
    for truth in sorted(pair_buckets):
        values = np.asarray(pair_buckets[truth], dtype=np.float64)
        mean = float(values.mean())
        per_pair.append({
            "truth_mm": truth,
            "samples": int(values.size),
            "measured_mean_mm": mean,
            "error_mm": mean - truth,
        })

    block_buckets: Dict[Tuple[int, int], List[float]] = {}
    for result in scored:
        for label, measured in zip(result["block_labels"], result["block_measured_mm"]):
            block_buckets.setdefault(tuple(label), []).append(float(measured))

    per_block: List[Dict[str, Any]] = []
    for label in sorted(block_buckets):
        truth = target.height_at(*label)
        values = np.asarray(block_buckets[label], dtype=np.float64)
        mean = float(values.mean())
        per_block.append({
            "row": label[0], "col": label[1],
            "truth_mm": truth,
            "samples": int(values.size),
            "measured_mean_mm": mean,
            "repeatability_rms_mm": (
                float(np.sqrt(np.mean((values - mean) ** 2))) if values.size > 1 else None
            ),
            "error_mm": mean - truth,
        })

    block_errors = np.array([entry["error_mm"] for entry in per_block])
    repeatability_values = np.array([
        entry["repeatability_rms_mm"] for entry in per_block
        if entry["repeatability_rms_mm"] is not None
    ])

    has_pairs = any(len(result["pair_truth_mm"]) for result in scored)
    scale_fit = None
    if has_pairs:
        flat_truth = np.concatenate([result["pair_truth_mm"] for result in scored])
        flat_measured = np.concatenate([result["pair_measured_mm"] for result in scored])
        scale_fit = fit_scale(flat_truth, flat_measured)

    per_session = [{
        "label": result["label"],
        "blocks_measured": len(result["block_labels"]),
        "plane_rms_mm": result["plane_rms_mm"],
        "plane_point_count": result["plane_point_count"],
        "order_mismatch_count": result["order_mismatch_count"],
        "depth_mean_m": result["depth_mean_m"],
    } for result in scored]

    return {
        "sessions_scored": len(scored),
        "per_pair": per_pair,
        "per_block": per_block,
        "per_session": per_session,
        "scale_fit": scale_fit,
        "rms_error_mm": float(np.sqrt(np.mean(block_errors ** 2))) if len(block_errors) else float("nan"),
        "max_abs_error_mm": float(np.abs(block_errors).max()) if len(block_errors) else float("nan"),
        "repeatability_rms_mm": (
            float(np.sqrt(np.mean(repeatability_values ** 2))) if len(repeatability_values) else None
        ),
        "plane_rms_mean_mm": float(np.mean([result["plane_rms_mm"] for result in scored])),
        "order_mismatch_total": int(sum(result["order_mismatch_count"] for result in scored)),
    }
```

- [ ] **Step 2: Replace `write_report`**

Replace the whole function with:

```python
def write_report(
    summary: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    target: DepthGridTarget,
    context: Dict[str, Any],
    path: Path,
) -> None:
    rule = "=" * 78
    thin = "-" * 78

    fit = summary["scale_fit"]
    if fit is not None:
        headline = (
            f"RESULT: scale error {fit['scale_error_pct']:+.3f} %  "
            f"(residual {fit['residual_rms_mm']:.4f} mm rms)  --  "
            f"{len(summary['per_block'])} block(s) across {summary['sessions_scored']} "
            f"session(s)"
        )
    else:
        headline = (
            f"RESULT: no pairwise data this run (every session measured only one "
            f"distinct block) -- {len(summary['per_block'])} block(s) across "
            f"{summary['sessions_scored']} session(s)"
        )
    if summary["repeatability_rms_mm"] is not None:
        headline += f", repeatability {summary['repeatability_rms_mm']:.4f} mm rms"

    lines = [
        rule,
        "DEPTH-GRID RELATIVE ACCURACY",
        rule,
        headline,
        "",
        f"extrinsics       : {context['extrinsics']}",
        f"target           : {context['target']}",
        f"ground truth     : {target.measured_by}",
        f"uncertainty      : +/- {target.height_uncertainty_mm:.3f} mm per block",
        f"cameras          : {context['camera_a']} (A) / {context['camera_b']} (B)",
        "",
    ]

    skipped = [result for result in results if "skipped" in result]
    if skipped:
        lines.extend([thin, "SKIPPED SESSIONS", thin])
        lines.extend(f"  {result['label']}: {result['skipped']}" for result in skipped)
        lines.append("")

    lines.extend([
        thin,
        "PER BLOCK  (pooled across sessions -- accuracy vs. truth, and repeatability "
        "from repeated clicks/sessions on the same block)",
        thin,
        f"{'truth mm':>9} {'n':>3} {'measured mm':>12} {'repeat. rms mm':>15} {'err mm':>9}",
    ])
    for entry in summary["per_block"]:
        repeat = (
            f"{entry['repeatability_rms_mm']:.4f}"
            if entry["repeatability_rms_mm"] is not None else "--"
        )
        lines.append(
            f"{entry['truth_mm']:>9.3f} {entry['samples']:>3} "
            f"{entry['measured_mean_mm']:>12.3f} {repeat:>15} {entry['error_mm']:>+9.3f}"
        )
    lines.append("")

    if fit is not None:
        lines.extend([
            thin,
            "PAIRWISE SCALE FIT  (block-to-block separations, immune to any offset "
            "in where the reference plane sits)",
            thin,
            f"  measured = a * true          a = {fit['scale']:.5f}  "
            f"-> {fit['scale_error_pct']:+.3f} %",
            f"  residual after scale         {fit['residual_rms_mm']:.4f} mm rms",
            f"  measured = a * true + b      a = {fit['affine_slope']:.5f}  "
            f"({fit['affine_slope_error_pct']:+.3f} %), b = {fit['affine_intercept_mm']:+.4f} mm",
            f"  residual after affine        {fit['affine_residual_rms_mm']:.4f} mm rms",
            "",
            "  a-1 is systematic depth-scale error. A large b points at a fixed",
            "  plane-fit bias rather than a proportional one.",
            "",
        ])

    lines.extend([thin, "PER SESSION", thin])
    for result in results:
        if "skipped" in result:
            continue
        lines.append(
            f"  {result['label']}: {len(result['block_labels'])} block(s), "
            f"plane rms {result['plane_rms_mm']:.4f} mm from {result['plane_point_count']} "
            f"reference points, depth {result['depth_mean_m'] * 1000:.0f} mm"
            + (f", {result['order_mismatch_count']} order mismatch(es)"
               if result["order_mismatch_count"] else "")
        )

    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 3: Verify with hand-built session dicts (no GUI, no real images needed)**

```bash
python3 - <<'EOF'
import numpy as np
from pathlib import Path
import tempfile
from calibration.depth_grid_target import DepthGridTarget
from check_depth_accuracy import aggregate_depth_results, write_report

target = DepthGridTarget(heights_mm=((3.6, 7.0),), pitch_mm=10.0,
                          reference_corner_count=4, measured_by="synthetic test")

def session(label, block_labels, block_measured_mm, pair_truth_mm, pair_measured_mm):
    block_measured_mm = np.array(block_measured_mm)
    block_truth_mm = np.array([target.height_at(*l) for l in block_labels])
    pairs = np.array([[0, 1]]) if len(block_labels) == 2 else np.zeros((0, 2), dtype=int)
    return {
        "label": label, "block_labels": block_labels,
        "block_truth_mm": block_truth_mm, "block_measured_mm": block_measured_mm,
        "block_samples": np.array([1] * len(block_labels)),
        "block_repeatability_rms_mm": [None] * len(block_labels),
        "pairs": pairs, "pair_truth_mm": np.array(pair_truth_mm),
        "pair_measured_mm": np.array(pair_measured_mm),
        "plane_rms_mm": 0.05, "plane_point_count": 4,
        "ref_epipolar_offset_max_px": 0.2, "cell_epipolar_offset_max_px": 0.3,
        "order_mismatch_count": 0, "depth_mean_m": 0.18,
    }

# Two ordinary sessions, both measuring both blocks.
results = [
    session("s1", [(0, 0), (0, 1)], [3.62, 6.98], [3.4], [3.36]),
    session("s2", [(0, 0), (0, 1)], [3.58, 7.03], [3.4], [3.45]),
]
summary = aggregate_depth_results(results, target)
assert summary["scale_fit"] is not None
assert len(summary["per_block"]) == 2
assert summary["repeatability_rms_mm"] is not None  # block (0,0) has 2 session means
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "report.txt"
    write_report(summary, results, target, {
        "extrinsics": "x", "target": "y", "camera_a": "rgb_cam1", "camera_b": "rgb_cam2",
    }, out)
    text = out.read_text()
    assert text.startswith("=" * 78 + "\nDEPTH-GRID RELATIVE ACCURACY\n" + "=" * 78 + "\nRESULT:")
    assert "PAIRWISE SCALE FIT" in text

# A single-block-only run: scale_fit must be None, and write_report must not crash.
one_block_results = [session("s3", [(0, 0)], [3.61], [], [])]
summary2 = aggregate_depth_results(one_block_results, target)
assert summary2["scale_fit"] is None
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "report.txt"
    write_report(summary2, one_block_results, target, {
        "extrinsics": "x", "target": "y", "camera_a": "rgb_cam1", "camera_b": "rgb_cam2",
    }, out)
    text = out.read_text()
    assert "RESULT: no pairwise data this run" in text
    assert "PAIRWISE SCALE FIT" not in text

print("Task 4 verification OK")
EOF
```

Expected: `Task 4 verification OK`.

- [ ] **Step 4: Commit**

```bash
git add check_depth_accuracy.py
git commit -m "$(cat <<'EOF'
Trim the depth-accuracy report to a headline + one core per-block table

Replaces the old report's separate per-session cell tables, standalone
per-cell section, and measurement-quality section (which showed the
same numbers split across several places) with one headline line, one
per-block table (truth, samples, measured mean, repeatability, error),
and the pairwise scale fit -- kept, since it's the one view that
separates a proportional scale error from a fixed offset. Handles the
now-valid case of zero pairwise data (every session single-block) by
reporting scale_fit=None instead of a stray nan.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Replace `_prompt_cell_labels` with `LabelingSession`

**Files:**
- Modify: `check_depth_accuracy.py`

**Interfaces:**
- Consumes: `measure_points.run_interactive`'s `on_point`/`on_undo` hooks (Task 1); `DepthGridTarget.cell_at_height` (Task 2); `fit_plane_3d`, `perpendicular_distance_to_plane` (unchanged, already in this file).
- Produces: `LabelingSession(target: DepthGridTarget)` with `.on_point(index: int, result: Dict[str, Any]) -> None`, `.on_undo(new_count: int) -> None`, and a `.labels: List[Tuple[int, int]]` attribute that always has exactly `len(labels) == max(0, points_completed - target.reference_corner_count)` entries, in click order -- this is what the caller (Task 6) reads once the window closes, in place of the old `_prompt_cell_labels`'s return value.

- [ ] **Step 1: Delete `_prompt_cell_labels`**

Remove the whole function (`def _prompt_cell_labels(...)` through its closing `return labels`).

- [ ] **Step 2: Add `LabelingSession`, in its place**

```python
class LabelingSession:
    """Drives live labeling via run_interactive's on_point/on_undo hooks.

    Counts off the reference corners as they land, fits the datum plane the
    instant the last one completes, then prompts for each subsequent
    block's engraved height and grades it immediately against that plane --
    replacing the old blind, post-hoc, click-order-matched labeling prompt.
    """

    def __init__(self, target: DepthGridTarget) -> None:
        self.target = target
        self.n_ref = target.reference_corner_count
        self.labels: List[Tuple[int, int]] = []
        self.samples: Dict[Tuple[int, int], List[float]] = {}
        self.plane: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def on_point(self, index: int, result: Dict[str, Any]) -> None:
        if index < self.n_ref - 1:
            print(f"  reference corner {index + 1}/{self.n_ref} recorded")
            return
        if index == self.n_ref - 1:
            points_mm = np.asarray(result["points_mm"][:self.n_ref], dtype=np.float64)
            centroid, normal, rms_m = fit_plane_3d(points_mm / 1000.0)
            if normal @ (-centroid) < 0.0:
                normal = -normal
            self.plane = (centroid, normal)
            print(f"  reference corner {self.n_ref}/{self.n_ref} recorded -- "
                  f"plane fit rms {rms_m * 1000.0:.4f} mm")
            return

        centroid, normal = self.plane
        point_mm = np.asarray(result["points_mm"][index], dtype=np.float64)
        measured_mm = perpendicular_distance_to_plane(
            point_mm / 1000.0, centroid, normal,
        ) * 1000.0

        while True:
            raw = input("  engraved height (mm) on this block > ").strip()
            try:
                height = float(raw)
                label = self.target.cell_at_height(height)
            except ValueError as exc:
                print(f"    {exc}")
                continue
            break

        self.labels.append(label)
        history = self.samples.setdefault(label, [])
        history.append(measured_mm)
        if len(history) == 1:
            print(f"  {height}mm: measured {measured_mm:.3f}mm "
                  f"(delta {measured_mm - height:+.3f}mm)")
        else:
            values = np.asarray(history)
            spread = float(np.sqrt(np.mean((values - values.mean()) ** 2)))
            print(f"  {height}mm, sample {len(history)}: measured {measured_mm:.3f}mm "
                  f"(spread so far: {spread:.4f}mm rms)")

    def on_undo(self, new_count: int) -> None:
        n_labels = max(0, new_count - self.n_ref)
        removed = self.labels[n_labels:]
        self.labels = self.labels[:n_labels]
        for label in removed:
            if self.samples.get(label):
                self.samples[label].pop()
        if new_count < self.n_ref:
            self.plane = None
```

- [ ] **Step 3: Verify the callback logic directly (no GUI)**

This simulates a full click sequence -- 4 reference corners then 3 block clicks (one repeated) -- by calling `on_point`/`on_undo` exactly as `run_interactive` would, with `input()` mocked to supply the engraved heights a human would type. This is the boundary this task can test without a real window; the mouse-to-hook wiring itself was verified by code inspection in Task 1.

```bash
python3 - <<'EOF'
from unittest.mock import patch
import numpy as np
from calibration.depth_grid_target import DepthGridTarget
from check_depth_accuracy import LabelingSession

target = DepthGridTarget(heights_mm=((3.6, 7.0),), pitch_mm=10.0,
                          reference_corner_count=4, measured_by="synthetic test")
session = LabelingSession(target)

# 4 coplanar reference corners (z=0.18 flat, no tilt -- fine for this test,
# which only checks the labeling/bookkeeping, not plane-fit precision).
ref_points = [
    [-24.0, -24.0, 180.0], [24.0, -24.0, 180.0],
    [-24.0, 24.0, 180.0], [24.0, 24.0, 180.0],
]
for i in range(4):
    fake_result = {"points_mm": ref_points[: i + 1]}
    session.on_point(i, fake_result)
assert session.plane is not None

# Block (0,0) truth 3.6mm: two clicks (repeatability sample), then block
# (0,1) truth 7.0mm: one click. Points chosen so the perpendicular distance
# to the (flat, z=180) plane is exactly the intended height.
block_points = ref_points + [
    [0.0, 0.0, 180.0 - 3.6], [0.0, 0.0, 180.0 - 3.5], [5.0, 5.0, 180.0 - 7.0],
]
with patch("builtins.input", side_effect=["3.6", "3.6", "7.0"]):
    for i in range(4, 7):
        fake_result = {"points_mm": block_points[: i + 1]}
        session.on_point(i, fake_result)

assert session.labels == [(0, 0), (0, 0), (0, 1)], session.labels
assert len(session.samples[(0, 0)]) == 2
assert abs(session.samples[(0, 0)][0] - 3.6) < 1e-6
assert abs(session.samples[(0, 0)][1] - 3.5) < 1e-6

# Undo the last click (the (0,1) block) -- label list must shrink back.
session.on_undo(6)
assert session.labels == [(0, 0), (0, 0)]

# A bad height must re-prompt, not crash or silently mislabel.
with patch("builtins.input", side_effect=["9.9", "7.0"]):
    session.on_point(6, {"points_mm": block_points})
assert session.labels[-1] == (0, 1)

print("Task 5 verification OK")
EOF
```

Expected: `Task 5 verification OK`. Note explicitly (per Global Constraints): this exercises the callback logic directly with synthetic `on_point`/`on_undo` calls, not a real click-through-a-window session -- that path needs the physical target and a human clicking, which isn't available to verify by script.

- [ ] **Step 4: Commit**

```bash
git add check_depth_accuracy.py
git commit -m "$(cat <<'EOF'
Replace blind post-hoc cell labeling with live LabelingSession

The old flow collected every click first, then asked the user to type
(row, col) for each one afterward, matched purely by click order with
nothing on screen linking a label to what was actually clicked -- this
produced the duplicate-labeled clicks seen in the real capture
sessions. LabelingSession uses run_interactive's new on_point/on_undo
hooks to fit the reference plane the instant the last corner lands and
prompt for each block's engraved height right after it's clicked,
grading it immediately against the already-fit plane.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `--cell` flag format change and `main()` wiring

**Files:**
- Modify: `check_depth_accuracy.py`

**Interfaces:**
- Consumes: Task 2's `DepthGridTarget.cell_at_height`; Task 3's `measure_depth_session`/Task 4's `aggregate_depth_results`/`write_report` return shapes; Task 5's `LabelingSession`; Task 1's `run_interactive(..., on_point=, on_undo=)`.
- Produces: nothing new consumed elsewhere -- this task finishes wiring the script end to end. `--cell` now takes `HEIGHT,AX,AY,BX,BY` instead of `ROW,COL,AX,AY,BX,BY`.

- [ ] **Step 1: Update the module docstring's usage example**

Replace:

```
Usage:
    python check_depth_accuracy.py --captures captures/depth_target
    python check_depth_accuracy.py --session captures/depth_target/<timestamp>
    python check_depth_accuracy.py --session captures/depth_target/<timestamp> \\
        --ref 100,200,90,205 --ref 900,200,890,205 \\
        --ref 100,900,90,905 --ref 900,900,890,905 \\
        --cell 0,0,300,400,290,405 --cell 4,4,700,600,690,605   # non-interactive
```

with:

```
Usage:
    python check_depth_accuracy.py --captures captures/depth_target
    python check_depth_accuracy.py --session captures/depth_target/<timestamp>
    python check_depth_accuracy.py --session captures/depth_target/<timestamp> \\
        --ref 100,200,90,205 --ref 900,200,890,205 \\
        --ref 100,900,90,905 --ref 900,900,890,905 \\
        --cell 0.6,300,400,290,405 --cell 30.0,700,600,690,605   # non-interactive;
        # HEIGHT,AX,AY,BX,BY -- identify a block by the number engraved on it
```

- [ ] **Step 2: Replace `parse_cell`**

Replace:

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

with:

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

- [ ] **Step 3: Update `--cell`'s argparse help text**

Replace:

```python
    parser.add_argument("--cell", type=parse_cell, action="append", default=None,
                        metavar="ROW,COL,AX,AY,BX,BY",
                        help="Non-interactive: one grid-cell click pair, labeled by its "
                             "(row, col) in the target's heights_mm grid. Repeat for every "
                             "cell visible this session -- occluded cells are simply "
                             "omitted, not required.")
```

with:

```python
    parser.add_argument("--cell", type=parse_cell, action="append", default=None,
                        metavar="HEIGHT,AX,AY,BX,BY",
                        help="Non-interactive: one block click pair, labeled by the height "
                             "(mm) engraved on it -- must match one of the target's "
                             "heights_mm values. Repeat for every block visible this "
                             "session (a block may repeat for a repeatability sample); "
                             "occluded blocks are simply omitted, not required.")
```

- [ ] **Step 4: Update the `--ref`/`--cell` validation block in `main()`**

Replace:

```python
    if args.ref or args.cell:
        if not args.session:
            raise SystemExit("--ref/--cell require --session (they supply coordinates for "
                             "exactly one session).")
        if args.captures:
            raise SystemExit("--ref/--cell cannot be combined with --captures.")
        if not args.ref or len(args.ref) != target.reference_corner_count:
            raise SystemExit(
                f"{target_path.name} needs exactly {target.reference_corner_count} --ref "
                f"clicks, got {len(args.ref) if args.ref else 0}."
            )
        if not args.cell or len(args.cell) < 2:
            raise SystemExit("Need at least 2 --cell clicks to measure a depth difference.")
        for row, col, *_ in args.cell:
            if not (0 <= row < target.row_count and 0 <= col < target.col_count):
                raise SystemExit(
                    f"--cell {row},{col} is out of range for a {target.row_count}x"
                    f"{target.col_count} grid."
                )
```

with:

```python
    if args.ref or args.cell:
        if not args.session:
            raise SystemExit("--ref/--cell require --session (they supply coordinates for "
                             "exactly one session).")
        if args.captures:
            raise SystemExit("--ref/--cell cannot be combined with --captures.")
        if not args.ref or len(args.ref) != target.reference_corner_count:
            raise SystemExit(
                f"{target_path.name} needs exactly {target.reference_corner_count} --ref "
                f"clicks, got {len(args.ref) if args.ref else 0}."
            )
        if not args.cell:
            raise SystemExit("Need at least 1 --cell click to measure a block's depth.")
```

- [ ] **Step 5: Update the non-interactive branch that builds `cell_labels`**

Replace:

```python
        if args.ref:
            ref_clicks_a = np.array([[point[0], point[1]] for point in args.ref])
            ref_clicks_b = np.array([[point[2], point[3]] for point in args.ref])
            cell_labels = [(cell[0], cell[1]) for cell in args.cell]
            cell_clicks_a = np.array([[cell[2], cell[3]] for cell in args.cell])
            cell_clicks_b = np.array([[cell[4], cell[5]] for cell in args.cell])
```

with:

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
```

- [ ] **Step 6: Replace the interactive branch's labeling call**

Replace:

```python
        else:
            print(f"{label}: click the {target.reference_corner_count} reference corners "
                  "first (any order among themselves), then click every grid cell you can "
                  "see -- skip ones you can't. Press q/Esc to finish.")
            raw_result = run_interactive(
                image_a, image_b, extrinsics, depth_range,
                max(2, args.zoom), (int(args.window[0]), int(args.window[1])),
                max(3, args.blob_radius) if args.blob_snap else 0,
                float(reg_config.get("default_depth", 0.168)),
            )
            n_ref = target.reference_corner_count
            total_clicks = len(raw_result.get("clicks_a", [])) if raw_result else 0
            if total_clicks < n_ref + 2:
                results.append({
                    "label": label,
                    "skipped": f"fewer than {n_ref} reference + 2 cell clicks",
                })
                print(f"{label}: skipped: not enough points clicked")
                continue
            ref_clicks_a = np.array(raw_result["clicks_a"][:n_ref])
            ref_clicks_b = np.array(raw_result["clicks_b_snapped"][:n_ref])
            remaining_a = raw_result["clicks_a"][n_ref:]
            remaining_b = raw_result["clicks_b_snapped"][n_ref:]
            cell_labels = _prompt_cell_labels(len(remaining_a), target)
            cell_clicks_a = np.array(remaining_a)
            cell_clicks_b = np.array(remaining_b)
```

with:

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
            if total_clicks < n_ref + 1:
                results.append({
                    "label": label,
                    "skipped": f"fewer than {n_ref} reference + 1 labeled block click",
                })
                print(f"{label}: skipped: not enough points clicked")
                continue
            ref_clicks_a = np.array(raw_result["clicks_a"][:n_ref])
            ref_clicks_b = np.array(raw_result["clicks_b_snapped"][:n_ref])
            cell_labels = session.labels
            cell_clicks_a = np.array(raw_result["clicks_a"][n_ref:])
            cell_clicks_b = np.array(raw_result["clicks_b_snapped"][n_ref:])
```

- [ ] **Step 7: Update the per-session print after `measure_depth_session`**

Replace:

```python
        session_scale = fit_scale(measured["pair_truth_mm"], measured["pair_measured_mm"])
        print(f"{label}: scale {session_scale['scale_error_pct']:+.3f} %, "
              f"cells {len(measured['cell_labels'])}, "
              f"plane rms {measured['plane_rms_mm']:.4f} mm, "
              f"depth {measured['depth_mean_m'] * 1000:.0f} mm")
```

with:

```python
        if len(measured["pair_truth_mm"]):
            session_scale = fit_scale(measured["pair_truth_mm"], measured["pair_measured_mm"])
            scale_text = f"scale {session_scale['scale_error_pct']:+.3f} %"
        else:
            scale_text = "scale n/a (1 block)"
        print(f"{label}: {scale_text}, blocks {len(measured['block_labels'])}, "
              f"plane rms {measured['plane_rms_mm']:.4f} mm, "
              f"depth {measured['depth_mean_m'] * 1000:.0f} mm")
```

- [ ] **Step 8: Update the `result.json` payload**

Replace the `"sessions"` list comprehension inside the `payload` dict:

```python
        "sessions": [
            {"label": result["label"], "skipped": result["skipped"]}
            if "skipped" in result else
            {
                "label": result["label"],
                "cell_labels": result["cell_labels"],
                "cell_truth_mm": result["cell_truth_mm"].tolist(),
                "cell_measured_mm": result["cell_measured_mm"].tolist(),
                "pairs": result["pairs"].tolist(),
                "pair_truth_mm": result["pair_truth_mm"].tolist(),
                "pair_measured_mm": result["pair_measured_mm"].tolist(),
                "plane_rms_mm": result["plane_rms_mm"],
                "plane_point_count": result["plane_point_count"],
                "ref_epipolar_offset_max_px": result["ref_epipolar_offset_max_px"],
                "cell_epipolar_offset_max_px": result["cell_epipolar_offset_max_px"],
                "order_mismatch_count": result["order_mismatch_count"],
                "depth_mean_m": result["depth_mean_m"],
            }
            for result in results
        ],
```

with:

```python
        "sessions": [
            {"label": result["label"], "skipped": result["skipped"]}
            if "skipped" in result else
            {
                "label": result["label"],
                "block_labels": result["block_labels"],
                "block_truth_mm": result["block_truth_mm"].tolist(),
                "block_measured_mm": result["block_measured_mm"].tolist(),
                "block_samples": result["block_samples"].tolist(),
                "block_repeatability_rms_mm": result["block_repeatability_rms_mm"],
                "pairs": result["pairs"].tolist(),
                "pair_truth_mm": result["pair_truth_mm"].tolist(),
                "pair_measured_mm": result["pair_measured_mm"].tolist(),
                "plane_rms_mm": result["plane_rms_mm"],
                "plane_point_count": result["plane_point_count"],
                "ref_epipolar_offset_max_px": result["ref_epipolar_offset_max_px"],
                "cell_epipolar_offset_max_px": result["cell_epipolar_offset_max_px"],
                "order_mismatch_count": result["order_mismatch_count"],
                "depth_mean_m": result["depth_mean_m"],
            }
            for result in results
        ],
```

- [ ] **Step 9: Update the final summary print**

Replace:

```python
    fit = summary["scale_fit"]
    print()
    print(f"scale error   {fit['scale_error_pct']:+.3f} %  "
          f"(a = {fit['scale']:.5f}, residual {fit['residual_rms_mm']:.4f} mm rms)")
    print(f"mean abs err  {summary['mean_abs_relative_pct']:.3f} %  "
          f"({summary['rms_error_mm']:.4f} mm rms, max {summary['max_abs_error_mm']:.4f} mm)")
    print(f"Saved report.txt / result.json / *_correspondences.jpg to {output_dir}")
```

with:

```python
    fit = summary["scale_fit"]
    print()
    if fit is not None:
        print(f"scale error   {fit['scale_error_pct']:+.3f} %  "
              f"(a = {fit['scale']:.5f}, residual {fit['residual_rms_mm']:.4f} mm rms)")
    else:
        print("scale error   n/a (no pairwise data -- every session measured only one block)")
    print(f"rms error     {summary['rms_error_mm']:.4f} mm  (max {summary['max_abs_error_mm']:.4f} mm)")
    if summary["repeatability_rms_mm"] is not None:
        print(f"repeatability {summary['repeatability_rms_mm']:.4f} mm rms")
    print(f"Saved report.txt / result.json / *_correspondences.jpg to {output_dir}")
```

- [ ] **Step 10: Syntax and import check**

```bash
python3 -c "import check_depth_accuracy"
```

Expected: no output, exit code 0.

- [ ] **Step 11: Commit**

```bash
git add check_depth_accuracy.py
git commit -m "$(cat <<'EOF'
Wire LabelingSession and block-based --cell into check_depth_accuracy CLI

--cell now takes HEIGHT,AX,AY,BX,BY instead of ROW,COL,AX,AY,BX,BY, so
scripted and interactive sessions identify a block the same way. The
interactive branch drives LabelingSession through run_interactive's new
hooks instead of _prompt_cell_labels; main()'s printing and result.json
payload follow measure_depth_session/aggregate_depth_results' new
block_* field names.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: End-to-end verification against real calibrated extrinsics

**Files:** none (verification only; no code changes).

**Interfaces:** Consumes the finished CLI from Tasks 1-6 as a black box (subprocess), plus this rig's real `calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json`.

- [ ] **Step 1: Build a synthetic session and run the CLI subprocess end-to-end**

Same forward-projection approach as Task 3's verification, but driven through the real CLI as a subprocess (dummy images at this rig's real calibrated resolution, a small temp target file, `--ref`/`--cell` in the new `HEIGHT,AX,AY,BX,BY` format), checking the whole pipeline -- argument parsing, `measure_depth_session`, `aggregate_depth_results`, `write_report`, `result.json`, the annotated JPEG -- together, not each piece in isolation.

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

corners_m = [on_plane(dx, dy) for dx in (-0.024, 0.024) for dy in (-0.024, 0.024)]
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
    for p in corners_m:
        a, b = project_pair(p)
        ref_args += ["--ref", f"{a[0]},{a[1]},{b[0]},{b[1]}"]

    cell_args = []
    for point_m, height_mm in ((block36_m, 3.6), (block36_m, 3.6), (block70_m, 7.0)):
        a, b = project_pair(point_m)
        cell_args += ["--cell", f"{height_mm},{a[0]},{a[1]},{b[0]},{b[1]}"]

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
    assert "PAIRWISE SCALE FIT" in report
    assert "nan" not in report.lower()

    result = json.loads((out_dir / "depth_accuracy" / "result.json").read_text())
    summary = result["summary"]
    assert summary["scale_fit"] is not None
    assert abs(summary["scale_fit"]["scale_error_pct"]) < 1.0
    assert summary["repeatability_rms_mm"] is not None
    per_block = {(entry["row"], entry["col"]): entry for entry in summary["per_block"]}
    assert abs(per_block[(0, 0)]["error_mm"]) < 0.05
    assert abs(per_block[(0, 1)]["error_mm"]) < 0.05

    assert (out_dir / "depth_accuracy" / "20990101_000000_correspondences.jpg").exists()

print("Task 7 verification OK")
EOF
```

Expected: `Task 7 verification OK`, with the printed subprocess stdout showing a `scale error` line close to `+0.000 %` and no `nan` anywhere.

- [ ] **Step 2: Confirm no unintended files were left behind**

```bash
git status --short
```

Expected: only the files touched by Tasks 1-6 (`triangulate.py`, `measure_points.py`, `calibration/depth_grid_target.py`, `check_depth_accuracy.py`) show as already committed; nothing untracked from this task (the subprocess test ran entirely inside a `tempfile.TemporaryDirectory`).

No commit for this task -- it verifies Tasks 1-6's combined commits, and touches no files of its own.
