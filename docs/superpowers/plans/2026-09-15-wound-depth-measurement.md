# measure_wound_depth.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New standalone script, `measure_wound_depth.py`, that measures how far a real specimen's point (e.g. a wound) sits below its own immediate surroundings, without assuming any single global reference plane and without any calibration-target/ground-truth machinery.

**Architecture:** Click points along the specimen's rim (intact surface immediately around it), press **n** once to lock the rim in. Every click after that is a measured point, reported the instant you click it: its `k` nearest rim points (3D mm distance, `--neighbors`, default 5) are found, a local plane is fit through just those, and the point's signed perpendicular distance to *that* local plane is its depth. Different points get different local planes, so a curved surface is tolerated without ever assuming one flat reference. Reuses `measure_points.run_interactive`'s click/undo/epipolar-snap machinery and `check_depth_accuracy.fit_plane_3d`/`perpendicular_distance_to_plane` (imported, not duplicated); drops `DepthGridTarget`, ground-truth lookup, and pairwise scale-fit entirely, none of which apply to an unknown specimen.

**Tech Stack:** Python 3.9, `opencv-contrib-python`, `numpy`; this repo's `measure_points.py`, `check_depth_accuracy.py`, `calibration.stereo.StereoExtrinsics`, `registration_io`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-15-wound-depth-measurement-design.md`.
- No test suite or linter in this repo (per `CLAUDE.md`) -- verify with throwaway `python3` heredoc scripts, run during implementation and **not committed**, same precedent as every prior plan touching `measure_points.py`/`check_depth_accuracy.py`.
- `from __future__ import annotations` + type hints on every new signature.
- `fit_plane_3d` and `perpendicular_distance_to_plane` are imported from `check_depth_accuracy.py`, never redefined. `run_interactive`, `handle_interactive_key`, `measure_points` (aliased `triangulate_clicks`), `parse_point`, `Panel`, `draw_marker` are imported from `measure_points.py`, never redefined. No changes to either of those files.
- Sign convention (load-bearing, document it everywhere it appears): `perpendicular_distance_to_plane` reports a point *protruding toward the camera* as positive (`check_depth_accuracy.py`'s convention, unchanged there). This script flips that sign before reporting, so a specimen *recessed* relative to its rim reads a **positive** `depth_mm` -- the intuitive convention for "how deep is this wound." The flip happens in exactly one place (`nearest_rim_plane`), not re-derived anywhere else.
- One session, one specimen, one output folder -- no multi-session scanning/aggregation like `check_depth_accuracy.py`'s `--captures`. This mirrors `measure_points.py`'s single-`--session` shape, not `check_depth_accuracy.py`'s multi-session one.
- **Deviation from the approved design doc's Undo section, found while writing this plan:** the design doc said undo "refuses" to reach into a closed rim batch. That's not actually implementable -- `handle_interactive_key` already pops the click from `state["clicks_a"]`/`clicks_b"]`/`click_offsets_px"]` and calls `recompute()` *before* the driver's `on_undo` hook is even invoked (see `measure_points.py`'s `u` handler), so by the time `on_undo` runs there is nothing left to "refuse." The fix that preserves the design's actual intent (never show a stale, silently-wrong number as if it were still authoritative) without requiring an impossible refusal: **every printed number during the session is an explicit live estimate**, and the real numbers written to `report.txt`/`result.json` are always recomputed exactly once, at the very end, from the FINAL rim/point sets after all undos -- via `measure_wound_session`, the same function the non-interactive `--rim`/`--wound` path calls directly. `on_undo` therefore never refuses anything; it just keeps `rim_indices`/`point_indices` in sync with whatever `run_interactive` already did. See Task 2's class docstring.
- Output goes to `registration.output_dir` (config) / `<session>` / `wound_depth` -- matching `measure_points.py`'s location convention, not `check_depth_accuracy.py`'s `calibration/results/.../depth_accuracy`.

---

### Task 1: Module scaffold + pure geometry functions (`nearest_rim_plane`, `measure_wound_session`)

**Files:**
- Create: `measure_wound_depth.py`

**Interfaces:**
- Consumes: `check_depth_accuracy.fit_plane_3d(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]`, `check_depth_accuracy.perpendicular_distance_to_plane(point, centroid, normal) -> float`, `measure_points.measure_points(clicks_a, clicks_b, extrinsics) -> Dict[str, Any]` (imported as `triangulate_clicks`).
- Produces:
  - `nearest_rim_plane(point_mm: np.ndarray, rim_points_mm: np.ndarray, neighbors: int) -> Dict[str, float]` -- keys `depth_mm`, `plane_rms_mm`, `neighbor_count`, `farthest_neighbor_mm`. Raises `ValueError` if `len(rim_points_mm) < neighbors`.
  - `measure_wound_session(rim_clicks_a, rim_clicks_b, point_clicks_a, point_clicks_b, extrinsics, neighbors, rim_offsets_px=None, point_offsets_px=None) -> Dict[str, Any]` -- keys `neighbors`, `rim_clicks_a`, `rim_clicks_b`, `rim_points_mm`, `rim_epipolar_offset_max_px`, `point_clicks_a`, `point_clicks_b`, `points_mm`, `measurements` (a list of dicts, one per measured point, each with `index`, `depth_mm`, `plane_rms_mm`, `neighbor_count`, `farthest_neighbor_mm`, `epipolar_offset_px`). Raises `ValueError` if fewer than `neighbors` rim clicks or fewer than 1 point click. This is the single authoritative computation both the interactive session (Task 2/4) and the non-interactive `--rim`/`--wound` path (Task 4) call.

- [ ] **Step 1: Write the module skeleton and both functions**

Create `measure_wound_depth.py`:

```python
#!/usr/bin/env python3
"""Measure a real specimen's local depth against its own immediate surroundings.

Built for a case check_depth_accuracy.py's plane-fit machinery cannot cover: a
real specimen (e.g. a wound) has no known ground-truth height and, unlike a
rigid calibration plate, no guarantee of being flat even locally. So there is
no single global reference plane here. Instead, click points along the rim
(intact surface immediately surrounding the area of interest) and, for every
point you then measure, only the k NEAREST rim points (by 3D distance, not
click order) are used to fit that point's own small local plane -- different
points, especially across a curved region, get different local planes drawn
from whichever rim points are actually close to them.

Method, per session::

    click points along the rim (at least --neighbors of them, spread around
    the area of interest, close to it -- not distant surface), press n to
    lock the rim in
    click every point you want measured (any number, no further n needed) --
    each is immediately triangulated, its k nearest rim points found, a local
    plane fit through just those, and its signed distance to that local plane
    reported as this point's depth. A point recessed relative to its
    immediate surroundings reads a POSITIVE depth (opposite sign convention
    from check_depth_accuracy.py, which reports a block PROTRUDING toward the
    camera as positive -- here the specimen is expected to be a depression,
    and "depth in mm below the surface" reading positive is the intuitive
    convention for that)

Reported depths print live as you click, using whatever rim points exist at
that moment -- a convenient running estimate, not the final answer. The
authoritative numbers saved to report.txt/result.json are always recomputed
once, at the end of the session, from the FINAL rim and point sets (after any
undos) -- see measure_wound_session. This also means undoing a rim point
after the rim is already "locked" is always safe: it only changes what that
one final recomputation uses, it can never leave a half-updated number on
screen.

Usage:
    python measure_wound_depth.py --session captures/<timestamp>
    python measure_wound_depth.py --session captures/<timestamp> \\
        --rim 100,200,90,205 --rim 900,200,890,205 --rim 100,900,90,905 \\
        --rim 900,900,890,905 --rim 500,150,490,155 \\
        --wound 500,500,490,505   # non-interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_cameras import load_config, resolve_path  # noqa: E402
from calibration.stereo import StereoExtrinsics  # noqa: E402
from check_depth_accuracy import fit_plane_3d, perpendicular_distance_to_plane  # noqa: E402
from measure_points import (  # noqa: E402
    DEFAULT_BLOB_RADIUS_PX,
    DEFAULT_LOUPE_ZOOM,
    DEFAULT_MAX_WINDOW,
    Panel,
    draw_marker,
    measure_points as triangulate_clicks,
    parse_point,
    run_interactive,
)
from registration_io import default_extrinsics_path, undistort_pair  # noqa: E402

DEFAULT_OUTPUT_SUBDIR = "wound_depth"
DEFAULT_NEIGHBORS = 5


# --------------------------------------------------------------------------- #
# Geometry: local plane from a point's nearest rim neighbors
# --------------------------------------------------------------------------- #

def nearest_rim_plane(
    point_mm: np.ndarray, rim_points_mm: np.ndarray, neighbors: int,
) -> Dict[str, float]:
    """Fit a plane through a point's ``neighbors`` nearest rim points (3D mm
    distance) and return that point's signed depth below it.

    Every measured point gets its OWN local plane from whichever rim points
    are actually close to it -- this is what tolerates a curved surface
    without assuming one global plane fits everywhere, unlike
    check_depth_accuracy.py's single reference plane (valid there only
    because a calibration plate is rigid and flat by construction).
    """
    point_mm = np.asarray(point_mm, dtype=np.float64)
    rim_points_mm = np.asarray(rim_points_mm, dtype=np.float64).reshape(-1, 3)
    if len(rim_points_mm) < neighbors:
        raise ValueError(
            f"need at least {neighbors} rim points, got {len(rim_points_mm)}"
        )
    distances_mm = np.linalg.norm(rim_points_mm - point_mm, axis=1)
    nearest = np.argsort(distances_mm)[:neighbors]
    centroid, normal, rms_m = fit_plane_3d(rim_points_mm[nearest] / 1000.0)
    if normal @ (-centroid) < 0.0:
        normal = -normal
    signed_mm = perpendicular_distance_to_plane(
        point_mm / 1000.0, centroid, normal,
    ) * 1000.0
    return {
        # protruding toward the camera is positive under
        # perpendicular_distance_to_plane's convention -- flipped here since
        # a specimen of interest is expected to be a RECESSION relative to
        # its rim, and "depth below the surrounding surface" reading
        # positive is the intuitive convention for that.
        "depth_mm": -signed_mm,
        "plane_rms_mm": rms_m * 1000.0,
        "neighbor_count": int(neighbors),
        "farthest_neighbor_mm": float(distances_mm[nearest].max()),
    }


def measure_wound_session(
    rim_clicks_a: np.ndarray, rim_clicks_b: np.ndarray,
    point_clicks_a: np.ndarray, point_clicks_b: np.ndarray,
    extrinsics: StereoExtrinsics, neighbors: int,
    rim_offsets_px: Optional[np.ndarray] = None,
    point_offsets_px: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Triangulate the rim and the measured points, then compute every
    measured point's depth against its own local rim-derived plane.

    This is the single authoritative computation -- both the interactive
    session (recomputing once at the end, from the FINAL rim/point sets
    after any undos) and the non-interactive --rim/--wound path call this
    same function, the way check_depth_accuracy.measure_depth_session is
    shared between its own two entry points.

    ``rim_offsets_px``/``point_offsets_px``, when given (the interactive
    path), override the epipolar-offset diagnostic with the real pre-snap
    value captured live by run_interactive; when None (the non-interactive
    path), it's recomputed from the already-snapped clicks, which is correct
    there since those clicks are genuinely raw.
    """
    rim_clicks_a = np.asarray(rim_clicks_a, dtype=np.float64).reshape(-1, 2)
    rim_clicks_b = np.asarray(rim_clicks_b, dtype=np.float64).reshape(-1, 2)
    point_clicks_a = np.asarray(point_clicks_a, dtype=np.float64).reshape(-1, 2)
    point_clicks_b = np.asarray(point_clicks_b, dtype=np.float64).reshape(-1, 2)

    if len(rim_clicks_a) < neighbors:
        raise ValueError(
            f"need at least {neighbors} rim clicks, got {len(rim_clicks_a)}"
        )
    if len(point_clicks_a) < 1:
        raise ValueError(f"need at least 1 measured point, got {len(point_clicks_a)}")

    rim_result = triangulate_clicks(rim_clicks_a, rim_clicks_b, extrinsics)
    rim_points_mm = np.asarray(rim_result["points_mm"], dtype=np.float64)

    point_result = triangulate_clicks(point_clicks_a, point_clicks_b, extrinsics)
    points_mm = np.asarray(point_result["points_mm"], dtype=np.float64)

    measurements: List[Dict[str, Any]] = []
    for index, point_mm in enumerate(points_mm):
        local = nearest_rim_plane(point_mm, rim_points_mm, neighbors)
        local["index"] = index
        local["epipolar_offset_px"] = float(
            point_offsets_px[index] if point_offsets_px is not None
            else point_result["epipolar_offset_px"][index]
        )
        measurements.append(local)

    return {
        "neighbors": neighbors,
        "rim_clicks_a": rim_clicks_a,
        "rim_clicks_b": np.asarray(rim_result["clicks_b_snapped"]),
        "rim_points_mm": rim_points_mm,
        "rim_epipolar_offset_max_px": float(
            np.max(rim_offsets_px) if rim_offsets_px is not None
            else np.max(rim_result["epipolar_offset_px"])
        ),
        "point_clicks_a": point_clicks_a,
        "point_clicks_b": np.asarray(point_result["clicks_b_snapped"]),
        "points_mm": points_mm,
        "measurements": measurements,
    }
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import measure_wound_depth"
```

Expected: no output, exit code 0.

- [ ] **Step 3: Verify `nearest_rim_plane` tolerates curvature (no extrinsics needed)**

Two locally-flat patches at different tilts (simulating a curved rim), well separated so each patch's own 4 points are unambiguously the 4 nearest neighbors for a point near that patch. This is the core claim of the whole design -- a point near patch A must recover patch A's own tilt, not an average with patch B's.

```bash
python3 - <<'EOF'
import numpy as np
from measure_wound_depth import nearest_rim_plane

def tilted_offset(dx, dy, normal):
    # A point at (dx, dy, 0) projected onto a plane through the origin with
    # the given normal -- i.e. a point "on" that tilted patch.
    p = np.array([dx, dy, 0.0])
    return p - (p @ normal) * normal

normal_a = np.array([0.15, 0.0, -0.988]); normal_a /= np.linalg.norm(normal_a)
normal_b = np.array([-0.15, 0.0, -0.988]); normal_b /= np.linalg.norm(normal_b)
centre_a = np.array([-20.0, 0.0, 180.0])
centre_b = np.array([20.0, 0.0, 180.0])

patch_a = [centre_a + tilted_offset(dx, dy, normal_a)
           for dx, dy in [(-3, -3), (3, -3), (-3, 3), (3, 3)]]
patch_b = [centre_b + tilted_offset(dx, dy, normal_b)
           for dx, dy in [(-3, -3), (3, -3), (-3, 3), (3, 3)]]
rim_points_mm = np.array(patch_a + patch_b)

# A point 4.0mm INTO the surface (toward camera A, i.e. protruding) at each
# patch's own centre and tilt -- protrusion means a NEGATIVE depth_mm under
# this script's flipped convention. Protrusion = moving TOWARD the camera =
# SMALLER Z = adding the (already camera-oriented) normal, not subtracting it.
point_near_a = centre_a + normal_a * 4.0
point_near_b = centre_b + normal_b * 4.0

result_a = nearest_rim_plane(point_near_a, rim_points_mm, neighbors=4)
result_b = nearest_rim_plane(point_near_b, rim_points_mm, neighbors=4)
print(result_a)
print(result_b)
assert abs(result_a["depth_mm"] - (-4.0)) < 1e-6, result_a
assert abs(result_b["depth_mm"] - (-4.0)) < 1e-6, result_b
assert result_a["plane_rms_mm"] < 1e-9  # patch A's own 4 points are EXACTLY coplanar
assert result_a["farthest_neighbor_mm"] < 10.0  # nowhere near patch B's ~40mm distance

# Using ALL 8 rim points (k=8) instead of the nearest 4 mixes both patches'
# tilts and must NOT recover the same clean -4.0mm -- this is the failure
# mode a single global plane has, and exactly what per-point nearest-k
# neighbors is designed to avoid.
mixed = nearest_rim_plane(point_near_a, rim_points_mm, neighbors=8)
print(mixed)
assert abs(mixed["depth_mm"] - (-4.0)) > 0.5, "k=8 should NOT match the local-only result"

# Too few rim points for the requested k raises, doesn't silently degrade.
try:
    nearest_rim_plane(point_near_a, rim_points_mm[:2], neighbors=4)
    raise AssertionError("expected ValueError")
except ValueError as exc:
    assert "at least 4" in str(exc), exc

print("Task 1 curvature verification OK")
EOF
```

Expected: `Task 1 curvature verification OK`, both patch depths within `1e-6` of `-4.0`, the `k=8` mixed result clearly different.

- [ ] **Step 4: Verify `measure_wound_session` end-to-end against real calibrated extrinsics**

Same forward-projection technique used to verify `check_depth_accuracy.py` (see `docs/superpowers/plans/2026-08-25-depth-accuracy-batch-collection.md` Task 3) -- project known 3D points through this rig's real saved extrinsics to get synthetic pixel clicks, then confirm the whole pipeline (triangulate -> nearest-k -> local plane -> signed, flipped depth) recovers the known offset.

```bash
python3 - <<'EOF'
import numpy as np
from pathlib import Path
from calibration.stereo import StereoExtrinsics
from measure_wound_depth import measure_wound_session

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

rim_points_m = [
    on_plane(dx, dy) for dx, dy in
    [(-0.02, -0.02), (0.02, -0.02), (-0.02, 0.02), (0.02, 0.02), (0.0, 0.0)]
]
rim_clicks_a, rim_clicks_b = [], []
for p in rim_points_m:
    a, b = project_pair(p)
    rim_clicks_a.append(a); rim_clicks_b.append(b)

# One measured point recessed 3.0mm (a "wound") -- away from the camera, so
# depth_mm must read POSITIVE (opposite sign from a check_depth_accuracy.py
# block, which would read negative for the same physical direction). Recession
# = moving AWAY from the camera = LARGER Z = subtracting the (already
# camera-oriented) normal, not adding it.
wound_m = on_plane(0.0, 0.0) - normal * 0.0030
point_clicks_a, point_clicks_b = [project_pair(wound_m)[0]], [project_pair(wound_m)[1]]

result = measure_wound_session(
    np.array(rim_clicks_a), np.array(rim_clicks_b),
    np.array(point_clicks_a), np.array(point_clicks_b),
    extrinsics, neighbors=5,
)
print("measurements", result["measurements"])
assert len(result["measurements"]) == 1
measured = result["measurements"][0]
assert abs(measured["depth_mm"] - 3.0) < 0.05, measured
assert measured["neighbor_count"] == 5
assert measured["index"] == 0

# Fewer than `neighbors` rim clicks raises.
try:
    measure_wound_session(
        np.array(rim_clicks_a[:3]), np.array(rim_clicks_b[:3]),
        np.array(point_clicks_a), np.array(point_clicks_b), extrinsics, neighbors=5,
    )
    raise AssertionError("expected ValueError")
except ValueError as exc:
    assert "at least 5" in str(exc), exc

# Zero point clicks raises.
try:
    measure_wound_session(
        np.array(rim_clicks_a), np.array(rim_clicks_b),
        np.zeros((0, 2)), np.zeros((0, 2)), extrinsics, neighbors=5,
    )
    raise AssertionError("expected ValueError")
except ValueError as exc:
    assert "at least 1" in str(exc), exc

print("Task 1 measure_wound_session verification OK")
EOF
```

Expected: `Task 1 measure_wound_session verification OK`, measured `depth_mm` within `0.05` of `3.0`.

- [ ] **Step 5: Commit**

```bash
git add measure_wound_depth.py
git commit -m "$(cat <<'EOF'
Add measure_wound_depth.py: per-point local-plane depth vs. a specimen's own rim

check_depth_accuracy.py's plane-fit machinery assumes one global flat
reference plane and a known-ground-truth grid -- neither holds for a
real specimen (e.g. a wound), which has no ground truth and no
guarantee of being flat even locally. nearest_rim_plane fits each
measured point its OWN local plane from just its k nearest rim points
(3D distance), so a curved rim is tolerated instead of assumed flat.
measure_wound_session wires triangulation + this local-plane math into
one function shared by the (not yet built) interactive and
non-interactive entry points.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `RimAndWoundSession` interactive driver

**Files:**
- Modify: `measure_wound_depth.py`

**Interfaces:**
- Consumes: Task 1's `nearest_rim_plane`. `measure_points.run_interactive`'s existing `on_point(index, result)`/`on_undo(new_count)`/`on_advance() -> Optional[str]` hook contract (no `on_text_submit` needed -- points are never labeled).
- Produces: `RimAndWoundSession(neighbors: int)` with `.on_point`, `.on_undo`, `.on_advance`, and public attributes `.rim_closed: bool`, `.rim_indices: List[int]`, `.point_indices: List[int]` -- Task 4's `main()` reads `rim_indices`/`point_indices` once `run_interactive` returns, to pull the FINAL click coordinates for `measure_wound_session`.

- [ ] **Step 1: Add the class**

Append to `measure_wound_depth.py`, after `measure_wound_session`:

```python
# --------------------------------------------------------------------------- #
# Interactive session
# --------------------------------------------------------------------------- #

class RimAndWoundSession:
    """Drives the two-phase rim-then-point collection via run_interactive's
    on_point/on_undo/on_advance hooks.

    Click rim points freely; press n once to lock the rim (needs >=
    ``neighbors`` points). Every click after that is a measured point,
    reported immediately using nearest_rim_plane against whatever rim points
    currently exist -- a live, convenient estimate only. The authoritative
    numbers always come from measure_wound_session, called once at the end
    of the session (see main()) against the FINAL rim_indices/point_indices
    after any undos.

    Undo never "refuses" to reach into an already-locked rim: by the time
    on_undo runs, run_interactive has already popped the click from its own
    state (see measure_points.py's `u` handler) -- there is nothing left
    here to protect. Reaching back into the rim just shrinks rim_indices;
    rim_closed stays True (the next click is still treated as a measured
    point, not a new rim point), and it's simply what the final
    recomputation in main() uses. This is why every printed number during
    the session is explicitly a "live estimate."
    """

    def __init__(self, neighbors: int) -> None:
        self.neighbors = neighbors
        self.rim_closed = False
        self.rim_indices: List[int] = []
        self.point_indices: List[int] = []
        self.points_mm: Dict[int, np.ndarray] = {}

    def on_point(self, index: int, result: Dict[str, Any]) -> None:
        self.points_mm[index] = np.asarray(result["points_mm"][index], dtype=np.float64)
        if not self.rim_closed:
            self.rim_indices.append(index)
            print(f"  rim point {len(self.rim_indices)} recorded")
            return
        self.point_indices.append(index)
        rim_points_mm = np.array([self.points_mm[i] for i in self.rim_indices])
        try:
            live = nearest_rim_plane(self.points_mm[index], rim_points_mm, self.neighbors)
        except ValueError as exc:
            print(f"  wound point {len(self.point_indices) - 1}: {exc}")
            return
        print(f"  wound point {len(self.point_indices) - 1}: depth {live['depth_mm']:.3f} mm "
              f"(local plane rms {live['plane_rms_mm']:.4f} mm from {live['neighbor_count']} "
              f"rim points, farthest {live['farthest_neighbor_mm']:.1f} mm) [live estimate]")

    def on_undo(self, new_count: int) -> None:
        for index in list(self.points_mm):
            if index >= new_count:
                del self.points_mm[index]
        self.rim_indices = [i for i in self.rim_indices if i < new_count]
        self.point_indices = [i for i in self.point_indices if i < new_count]
        if new_count == 0:
            self.rim_closed = False

    def on_advance(self) -> Optional[str]:
        if self.rim_closed:
            print("  rim already locked -- every click is measured immediately, "
                  "no need to press n again")
            return None
        if len(self.rim_indices) < self.neighbors:
            print(f"  need at least {self.neighbors} rim points, have {len(self.rim_indices)}")
            return None
        self.rim_closed = True
        print(f"  rim locked: {len(self.rim_indices)} points")
        return None
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import measure_wound_depth"
```

Expected: no output, exit code 0.

- [ ] **Step 3: Verify the session's bookkeeping directly (no window)**

```bash
python3 - <<'EOF'
from measure_wound_depth import RimAndWoundSession

rim_points = [
    [-5.0, -5.0, 180.0], [5.0, -5.0, 180.0],
    [-5.0, 5.0, 180.0], [5.0, 5.0, 180.0],
]

session = RimAndWoundSession(neighbors=3)
for i in range(4):
    session.on_point(i, {"points_mm": rim_points[: i + 1]})
assert session.rim_indices == [0, 1, 2, 3]
assert not session.rim_closed

# n with only 2 points (neighbors=3) declines.
partial = RimAndWoundSession(neighbors=3)
for i in range(2):
    partial.on_point(i, {"points_mm": rim_points[:i + 1]})
assert partial.on_advance() is None
assert not partial.rim_closed

# n with 4 (>= 3) locks the rim.
assert session.on_advance() is None
assert session.rim_closed

# A second n press is a harmless no-op.
assert session.on_advance() is None
assert session.rim_closed

# Every click after locking is a measured point, tracked separately from
# rim_indices, and reported live without raising.
all_points = rim_points + [[0.0, 0.0, 176.0]]
session.on_point(4, {"points_mm": all_points})
assert session.point_indices == [4]
assert session.rim_indices == [0, 1, 2, 3]

# Undoing a POINT does not touch the rim or reopen it.
session.on_undo(4)
assert session.point_indices == []
assert session.rim_indices == [0, 1, 2, 3]
assert session.rim_closed

# Undoing INTO the rim itself: rim shrinks, stays "closed" by design (see
# class docstring -- there is nothing left here to protect).
session.on_undo(2)
assert session.rim_indices == [0, 1]
assert session.rim_closed

# Undoing to zero resets everything, including rim_closed.
session.on_undo(0)
assert session.rim_indices == []
assert session.point_indices == []
assert not session.rim_closed

# If undo shrinks the rim below `neighbors` while still locked, the live
# per-click estimate must print nearest_rim_plane's error, never crash.
tight = RimAndWoundSession(neighbors=3)
for i in range(3):
    tight.on_point(i, {"points_mm": rim_points[:i + 1]})
tight.on_advance()
assert tight.rim_closed
tight.on_undo(1)  # rim shrinks to 1 point, below neighbors=3; stays closed
assert tight.rim_indices == [0]
tight.on_point(1, {"points_mm": rim_points[:1] + [[0.0, 0.0, 176.0]]})  # must not raise
print("  (confirmed: live estimate with too few rim points prints, doesn't crash)")

print("Task 2 verification OK")
EOF
```

Expected: `Task 2 verification OK` (plus the printed live-feedback lines, which are expected noise, not an error).

- [ ] **Step 4: Commit**

```bash
git add measure_wound_depth.py
git commit -m "$(cat <<'EOF'
Add RimAndWoundSession: click-freely, immediate-feedback interactive driver

Phase 1 (rim): click freely, press n once to lock (needs >= neighbors
points). Phase 2 (points): every click after that is auto-numbered and
measured immediately via nearest_rim_plane against the CURRENT rim --
a live estimate only. Undo never refuses to reach into a locked rim
(run_interactive has already popped the underlying click by the time
on_undo runs, so there's nothing left to protect); it just keeps
rim_indices/point_indices in sync, and the authoritative numbers are
always recomputed once at session end from whatever survives (see
Global Constraints' documented deviation from the design doc's Undo
section). Not yet wired into a CLI -- that's Task 4.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Output -- `annotate` and `write_report`

**Files:**
- Modify: `measure_wound_depth.py`

**Interfaces:**
- Consumes: Task 1's `measure_wound_session` return shape (`rim_clicks_a`/`rim_clicks_b`, `point_clicks_a`/`point_clicks_b`, `measurements`, `neighbors`, `rim_epipolar_offset_max_px`). `measure_points.Panel`, `measure_points.draw_marker`.
- Produces: `annotate(image_a, image_b, result, max_window) -> np.ndarray`; `write_report(result, context, path) -> None` where `context` is `{"extrinsics": str, "session": str, "camera_a": str, "camera_b": str}`. Task 4's `main()` calls both.

- [ ] **Step 1: Add the two functions**

Append to `measure_wound_depth.py`, after `RimAndWoundSession`:

```python
# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

COLOR_RIM = (219, 99, 37)
COLOR_POINT = (40, 220, 255)


def annotate(
    image_a: np.ndarray, image_b: np.ndarray, result: Dict[str, Any],
    max_window: Tuple[int, int],
) -> np.ndarray:
    """Saved picture of exactly what was clicked: rim points in one color,
    measured points in another, each measured point's depth labeled next to
    its marker.
    """
    panel_size = (max(320, max_window[0] // 2), max(320, max_window[1]))
    panel_a, panel_b = Panel(image_a, panel_size), Panel(image_b, panel_size)
    canvas = np.hstack([panel_a.render(), panel_b.render()])
    split = panel_a.width

    for index, point in enumerate(result["rim_clicks_a"]):
        draw_marker(canvas, panel_a, point, COLOR_RIM, index)
    for index, point in enumerate(result["rim_clicks_b"]):
        draw_marker(canvas, panel_b, point, COLOR_RIM, index, split)

    for measurement, point_a, point_b in zip(
        result["measurements"], result["point_clicks_a"], result["point_clicks_b"],
    ):
        index = measurement["index"]
        draw_marker(canvas, panel_a, point_a, COLOR_POINT, index)
        draw_marker(canvas, panel_b, point_b, COLOR_POINT, index, split)
        label = f"{measurement['depth_mm']:+.2f}mm"
        x, y = panel_a.to_screen(point_a)
        cv2.putText(canvas, label, (x + 22, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_POINT, 1, cv2.LINE_AA)
    return canvas


def write_report(
    result: Dict[str, Any], context: Dict[str, Any], path: Path,
) -> None:
    rule = "=" * 78
    thin = "-" * 78
    lines = [
        rule,
        "RELATIVE DEPTH -- point vs. its own local rim neighborhood",
        rule,
        f"extrinsics       : {context['extrinsics']}",
        f"session          : {context['session']}",
        f"cameras          : {context['camera_a']} (A) / {context['camera_b']} (B)",
        f"neighbors (k)    : {result['neighbors']}",
        f"rim points       : {len(result['rim_clicks_a'])}  "
        f"(max epipolar offset {result['rim_epipolar_offset_max_px']:.2f} px)",
        "",
        thin,
        f"{'pt':>3} {'depth mm':>10} {'plane rms mm':>13} {'neighbors':>9} "
        f"{'farthest mm':>12} {'click off px':>12}",
        thin,
    ]
    for measurement in result["measurements"]:
        lines.append(
            f"{measurement['index']:>3} {measurement['depth_mm']:>+10.3f} "
            f"{measurement['plane_rms_mm']:>13.4f} {measurement['neighbor_count']:>9} "
            f"{measurement['farthest_neighbor_mm']:>12.1f} "
            f"{measurement['epipolar_offset_px']:>12.1f}"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import measure_wound_depth"
```

Expected: no output, exit code 0.

- [ ] **Step 3: Verify both functions with a synthetic result dict (no camera/extrinsics needed)**

```bash
python3 - <<'EOF'
import tempfile
from pathlib import Path

import numpy as np
from measure_wound_depth import annotate, write_report

result = {
    "neighbors": 5,
    "rim_clicks_a": np.array([[50.0, 50.0], [150.0, 50.0], [50.0, 150.0], [150.0, 150.0], [100.0, 30.0]]),
    "rim_clicks_b": np.array([[48.0, 51.0], [148.0, 51.0], [48.0, 151.0], [148.0, 151.0], [98.0, 31.0]]),
    "rim_epipolar_offset_max_px": 1.2,
    "point_clicks_a": np.array([[100.0, 100.0]]),
    "point_clicks_b": np.array([[98.0, 101.0]]),
    "points_mm": np.array([[0.0, 0.0, 178.0]]),
    "measurements": [
        {"index": 0, "depth_mm": 3.456, "plane_rms_mm": 0.0123, "neighbor_count": 5,
         "farthest_neighbor_mm": 25.4, "epipolar_offset_px": 0.8},
    ],
}
context = {
    "extrinsics": "calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json",
    "session": "captures/wound/20990101_000000",
    "camera_a": "rgb_cam1", "camera_b": "rgb_cam2",
}

image_a = image_b = np.zeros((200, 200, 3), dtype=np.uint8)
canvas = annotate(image_a, image_b, result, (400, 300))
assert canvas.shape[0] > 0 and canvas.shape[1] > 0
assert canvas.dtype == np.uint8
assert canvas.max() > 0  # markers/text were actually drawn, not a blank frame

with tempfile.TemporaryDirectory() as tmp:
    report_path = Path(tmp) / "report.txt"
    write_report(result, context, report_path)
    text = report_path.read_text()
    assert "RELATIVE DEPTH" in text
    assert "neighbors (k)    : 5" in text
    assert "+3.456" in text
    assert "0.0123" in text

print("Task 3 verification OK")
EOF
```

Expected: `Task 3 verification OK`.

- [ ] **Step 4: Commit**

```bash
git add measure_wound_depth.py
git commit -m "$(cat <<'EOF'
Add annotate/write_report output helpers for measure_wound_depth.py

annotate draws rim points and measured points in distinct colors, with
each measured point's depth labeled next to its marker -- reuses
measure_points.Panel/draw_marker rather than duplicating them.
write_report is a plain-text summary (one row per measured point: depth,
local plane rms, neighbor count, farthest neighbor, click offset).
Verified against a synthetic result dict; not yet wired into a CLI.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: CLI (`parse_args`, `main`) -- interactive and non-interactive entry points

**Files:**
- Modify: `measure_wound_depth.py`

**Interfaces:**
- Consumes: Task 1's `measure_wound_session`, `DEFAULT_NEIGHBORS`; Task 2's `RimAndWoundSession`; Task 3's `annotate`, `write_report`; `measure_points.parse_point`, `DEFAULT_BLOB_RADIUS_PX`/`DEFAULT_LOUPE_ZOOM`/`DEFAULT_MAX_WINDOW`, `run_interactive`; `calibrate_cameras.load_config`/`resolve_path`; `registration_io.default_extrinsics_path`/`undistort_pair`.
- Produces: a runnable CLI. `python measure_wound_depth.py --session <dir>` (interactive) and `... --rim ... --wound ...` (non-interactive) both end by writing `report.txt`, `result.json`, `annotated.jpg` under `<registration.output_dir>/<session>/wound_depth/`.

- [ ] **Step 1: Add `parse_args` and `main`**

Append to `measure_wound_depth.py`, after `write_report`:

```python
# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure how far a point sits below (or above) its own local "
                    "surroundings, from nearby rim clicks -- no assumed global plane.",
    )
    parser.add_argument("--session", required=True,
                        help="Capture folder holding <camera-a>.jpg and <camera-b>.jpg.")
    parser.add_argument("--camera-a", default="rgb_cam1")
    parser.add_argument("--camera-b", default="rgb_cam2")
    parser.add_argument("--extrinsics", default=None,
                        help="Stereo extrinsics JSON (default: geometric_calibration."
                             "extrinsics_<a>_<b> in config, else "
                             "calibration/results/stereo_<a>_<b>/extrinsics.json).")
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS,
                        help="How many nearest rim points fit each measured point's own "
                             "local plane (default: %(default)s, minimum 3).")
    parser.add_argument("--rim", type=parse_point, action="append", default=None,
                        metavar="AX,AY,BX,BY",
                        help="Non-interactive: one rim click pair, in undistorted "
                             "full-res pixels. Repeat at least --neighbors times, spread "
                             "around the area of interest.")
    parser.add_argument("--wound", type=parse_point, action="append", default=None,
                        metavar="AX,AY,BX,BY",
                        help="Non-interactive: one measured-point click pair. Repeat for "
                             "every point you want depth for.")
    parser.add_argument("--zoom", type=int, default=DEFAULT_LOUPE_ZOOM,
                        help="Initial loupe magnification (default: %(default)s).")
    parser.add_argument("--window", type=int, nargs=2, default=list(DEFAULT_MAX_WINDOW),
                        metavar=("W", "H"),
                        help="Window size in pixels (default: %(default)s).")
    parser.add_argument("--blob-radius", type=int, default=DEFAULT_BLOB_RADIUS_PX,
                        help="Search radius when snapping a click to a dot's centroid, "
                             "in full-resolution pixels (default: %(default)s).")
    parser.add_argument("--blob-snap", action="store_true",
                        help="Move each click onto the intensity centroid of the marker "
                             "underneath it. Off by default so a click lands exactly "
                             "where you put it.")
    parser.add_argument("--out", "--output", dest="output", default=None,
                        help="Output directory (default: registration.output_dir in "
                             f"config) / <session> / {DEFAULT_OUTPUT_SUBDIR}.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.neighbors < 3:
        raise SystemExit(
            f"--neighbors must be at least 3 for a plane fit, got {args.neighbors}."
        )
    config = load_config()
    reg_config = config.get("registration", {}) or {}
    depth_range = reg_config.get("depth_range", [0.11, 0.21])
    depth_range = (float(depth_range[0]), float(depth_range[1]))

    extrinsics_path = resolve_path(
        args.extrinsics or default_extrinsics_path(args.camera_a, args.camera_b)
    )
    if not extrinsics_path.exists():
        raise SystemExit(f"No stereo extrinsics at {extrinsics_path}.")
    extrinsics = StereoExtrinsics.load_json(extrinsics_path)

    if args.rim or args.wound:
        if not args.rim or len(args.rim) < args.neighbors:
            raise SystemExit(
                f"Need at least --neighbors ({args.neighbors}) --rim clicks, got "
                f"{len(args.rim) if args.rim else 0}."
            )
        if not args.wound:
            raise SystemExit("Need at least 1 --wound click to measure a point's depth.")

    session_dir = resolve_path(args.session)
    paths = [session_dir / f"{camera}.jpg" for camera in (args.camera_a, args.camera_b)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"{session_dir} is missing {', '.join(missing)}.")
    raw_a, raw_b = cv2.imread(str(paths[0])), cv2.imread(str(paths[1]))
    if raw_a is None or raw_b is None:
        raise SystemExit(f"Failed to decode images in {session_dir}.")
    image_a, image_b = undistort_pair(raw_a, raw_b, extrinsics)

    output_dir = resolve_path(
        args.output or reg_config.get("output_dir", "registration/results")
    ) / session_dir.name / DEFAULT_OUTPUT_SUBDIR

    if args.rim:
        rim_clicks_a = np.array([[point[0], point[1]] for point in args.rim])
        rim_clicks_b = np.array([[point[2], point[3]] for point in args.rim])
        point_clicks_a = np.array([[point[0], point[1]] for point in args.wound])
        point_clicks_b = np.array([[point[2], point[3]] for point in args.wound])
        rim_offsets_px = None
        point_offsets_px = None
    else:
        print(f"{session_dir.name}: click points along the rim (at least "
              f"{args.neighbors}, spread around the area of interest) then press n to "
              "lock it in. Every click after that is measured immediately against its "
              "own nearest rim points -- no further n presses needed. Press q or Esc "
              "(or close the window) to finish and get the report.")
        session = RimAndWoundSession(args.neighbors)
        raw_result = run_interactive(
            image_a, image_b, extrinsics, depth_range,
            max(2, args.zoom), (int(args.window[0]), int(args.window[1])),
            max(3, args.blob_radius) if args.blob_snap else 0,
            float(reg_config.get("default_depth", 0.168)),
            on_point=session.on_point, on_undo=session.on_undo,
            on_advance=session.on_advance,
        )
        if len(session.rim_indices) < args.neighbors or not session.point_indices:
            print(f"{session_dir.name}: no points measured (need the rim locked with "
                  f"at least {args.neighbors} points, and at least 1 point clicked "
                  "after that).")
            return 0
        rim_clicks_a = np.array([raw_result["clicks_a"][i] for i in session.rim_indices])
        rim_clicks_b = np.array(
            [raw_result["clicks_b_snapped"][i] for i in session.rim_indices]
        )
        rim_offsets_px = np.array(
            [raw_result["epipolar_offset_px"][i] for i in session.rim_indices]
        )
        point_clicks_a = np.array(
            [raw_result["clicks_a"][i] for i in session.point_indices]
        )
        point_clicks_b = np.array(
            [raw_result["clicks_b_snapped"][i] for i in session.point_indices]
        )
        point_offsets_px = np.array(
            [raw_result["epipolar_offset_px"][i] for i in session.point_indices]
        )

    result = measure_wound_session(
        rim_clicks_a, rim_clicks_b, point_clicks_a, point_clicks_b,
        extrinsics, args.neighbors,
        rim_offsets_px=rim_offsets_px, point_offsets_px=point_offsets_px,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "extrinsics": str(extrinsics_path), "session": str(session_dir),
        "camera_a": args.camera_a, "camera_b": args.camera_b,
    }
    write_report(result, context, output_dir / "report.txt")
    cv2.imwrite(
        str(output_dir / "annotated.jpg"),
        annotate(image_a, image_b, result, (int(args.window[0]), int(args.window[1]))),
    )
    payload = {
        "extrinsics": str(extrinsics_path),
        "session": str(session_dir),
        "neighbors": result["neighbors"],
        "rim_points_mm": result["rim_points_mm"].tolist(),
        "rim_epipolar_offset_max_px": result["rim_epipolar_offset_max_px"],
        "points_mm": result["points_mm"].tolist(),
        "measurements": result["measurements"],
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print()
    for measurement in result["measurements"]:
        print(f"point {measurement['index']}: depth {measurement['depth_mm']:+.3f} mm  "
              f"(plane rms {measurement['plane_rms_mm']:.4f} mm)")
    print(f"Saved report.txt / result.json / annotated.jpg to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import measure_wound_depth"
```

Expected: no output, exit code 0.

- [ ] **Step 3: `--help` sanity check**

```bash
python3 measure_wound_depth.py --help
```

Expected: exit code 0, usage text listing `--session`, `--neighbors`, `--rim`, `--wound`, `--out`, etc.

- [ ] **Step 4: End-to-end CLI subprocess check via `--rim`/`--wound`, using real calibrated extrinsics**

Same forward-projection approach as Task 1 Step 4, this time driven through the actual CLI as a subprocess against dummy images at this rig's real calibrated resolution and a real temp session folder.

```bash
python3 - <<'EOF'
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

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

rim_points_m = [
    on_plane(dx, dy) for dx, dy in
    [(-0.02, -0.02), (0.02, -0.02), (-0.02, 0.02), (0.02, 0.02), (0.0, 0.0)]
]
wound_m = on_plane(0.0, 0.0) - normal * 0.0030  # recessed 3.0mm (subtract: away from camera)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    session_dir = tmp / "20990101_000000"
    session_dir.mkdir()
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(session_dir / "rgb_cam1.jpg"), blank)
    cv2.imwrite(str(session_dir / "rgb_cam2.jpg"), blank)

    rim_args = []
    for p in rim_points_m:
        a, b = project_pair(p)
        rim_args += ["--rim", f"{a[0]},{a[1]},{b[0]},{b[1]}"]
    a, b = project_pair(wound_m)
    wound_args = ["--wound", f"{a[0]},{a[1]},{b[0]},{b[1]}"]

    out_dir = tmp / "out"
    proc = subprocess.run(
        [sys.executable, "measure_wound_depth.py",
         "--session", str(session_dir), "--out", str(out_dir),
         *rim_args, *wound_args],
        cwd=Path.cwd(), capture_output=True, text=True,
    )
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    assert proc.returncode == 0, f"exit code {proc.returncode}"

    # --out is a BASE directory, same convention as measure_points.py's own
    # main() -- session_dir.name and wound_depth are still appended.
    result_dir = out_dir / session_dir.name / "wound_depth"
    report = (result_dir / "report.txt").read_text()
    assert "RELATIVE DEPTH" in report
    assert "neighbors (k)    : 5" in report

    result = json.loads((result_dir / "result.json").read_text())
    assert len(result["measurements"]) == 1
    assert abs(result["measurements"][0]["depth_mm"] - 3.0) < 0.05
    assert (result_dir / "annotated.jpg").exists()

    # Argument validation: too few --rim clicks must exit non-zero, before
    # ever touching the images.
    bad = subprocess.run(
        [sys.executable, "measure_wound_depth.py",
         "--session", str(session_dir), "--out", str(out_dir),
         *rim_args[:6], *wound_args],  # only 3 of the 5 --rim pairs
        cwd=Path.cwd(), capture_output=True, text=True,
    )
    assert bad.returncode != 0
    assert "at least" in bad.stderr.lower()

    # --neighbors below 3 must also exit non-zero.
    bad2 = subprocess.run(
        [sys.executable, "measure_wound_depth.py",
         "--session", str(session_dir), "--neighbors", "2",
         *rim_args, *wound_args],
        cwd=Path.cwd(), capture_output=True, text=True,
    )
    assert bad2.returncode != 0
    assert "at least 3" in bad2.stderr

print("Task 4 verification OK")
EOF
```

Expected: `Task 4 verification OK`, measured `depth_mm` within `0.05` of `3.0`, both bad-argument subprocess calls exit non-zero.

- [ ] **Step 5: Confirm no unintended files were left behind**

```bash
git status --short
```

Expected: clean (the subprocess test ran entirely inside a `tempfile.TemporaryDirectory`).

- [ ] **Step 6: Commit**

```bash
git add measure_wound_depth.py
git commit -m "$(cat <<'EOF'
Wire measure_wound_depth.py's CLI: interactive and --rim/--wound entry points

Both paths converge on the same measure_wound_session (Task 1) for the
authoritative numbers -- the interactive path pulls the FINAL
rim_indices/point_indices from RimAndWoundSession only after the window
closes, never trusting the live per-click estimates. Validates --rim/
--wound counts and --neighbors >= 3 up front, before touching any
images. Verified end-to-end via subprocess against this rig's real
calibrated extrinsics with forward-projected synthetic points.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## What this plan does not cover

Same caveat as every interactive tool in this repo: the actual click-through-a-window path, and any real photograph of an actual specimen, cannot be exercised by script. A real hardware session -- ideally on a test surface with a known curvature and a known recess first, before a real wound -- is the user's own next step after this lands.
