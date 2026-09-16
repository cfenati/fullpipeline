# check_line_accuracy.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `check_line_accuracy.py`, a new diagnostic script that measures how accurately this rig's RGB stereo pair recovers real-world distance, by triangulating a known printed line-ladder target and comparing measured gaps to their ground-truth mm values.

**Architecture:** A new `calibration/line_target.py:LineLadderTarget` dataclass (mirrors `target_board.py`'s `TargetBoard`) describes the physical target. `check_line_accuracy.py` is a single top-level script (matching `check_color.py`/`check_registration_error.py`'s convention) that: detects the target's parallel lines independently in both cameras' images (Canny + HoughLinesP + clustering), gets a geometrically valid point correspondence for samples along each line via the epipolar constraint (this rig's cameras are a verged ~18°-toe-in pair, not a rectifiable parallel pair, so "same pixel row in both images" is not a safe shortcut), triangulates real 3D points using the rig's existing calibrated `R`/`T`, and reports measured-vs-ground-truth mm error per gap.

**Tech Stack:** Python 3.9, `opencv-contrib-python`, `numpy`, `PyYAML` (all already in `requirements.txt`), this repo's `calibration.stereo.StereoExtrinsics`/`collect_session_pairs`, `register_pipeline.default_extrinsics_path`, `calibrate_cameras.resolve_path`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-19-check-line-accuracy-design.md` (read this first).
- No test suite or linter exists in this repo (per `CLAUDE.md`) — do not add pytest or a `tests/` directory. Verify with standalone `python3 -c` / heredoc scripts using plain `assert`s against synthetic data, matching the precedent in `docs/superpowers/plans/2026-08-18-record-kalibr-bag.md`. Because this script's entire pipeline (line detection, epipolar geometry, triangulation) is pure image/math with no hardware dependency, every task here is fully and genuinely testable synthetically — no camera or physical target needed until real-world validation.
- `from __future__ import annotations` + type hints on every function signature (project convention).
- RGB cameras only, defaults `rgb_cam1`/`rgb_cam2`.
- Target is a **single-orientation** ladder (`orientation: vertical|horizontal` + `gaps_mm` list) — not a grid, no line intersections.
- Distance is **never** a CLI input — always the result of stereo triangulation using `calibration/results/stereo_<a>_<b>/extrinsics.json`'s `R`/`T`/`fundamental`.
- `T` in `extrinsics.json` is in metres (documented convention) — triangulated points must be ×1000'd to mm.
- Epipolar line convention (must match `calibration/stereo.py:817-857`'s `epipolar_residuals`, the existing precedent): points undistorted first via `cv2.undistortPoints(pts, K, dist, P=K)`; `F = extrinsics.fundamental`; epipolar line in B for a point in A is `[x, y, 1] @ F.T`.
- Triangulation convention (must match `register_features.py`'s `triangulate_matches`, lines 498-519): `P_a = K_a @ [I|0]`, `P_b = K_b @ [R|T]`, `cv2.triangulatePoints`.
- Report convention: `report.txt` (human-readable) + `result.json` (machine-readable), matching `check_registration_error.py`'s pair.
- Output dir default `line_accuracy_reports/` — must be added to `.gitignore` (git-ignored generated data, per this repo's data-hygiene convention).
- Session/capture handling reuses `calibration.stereo.collect_session_pairs` and `register_pipeline.default_extrinsics_path` rather than re-deriving them.
- Do not commit unless the plan's own commit steps say so, and never with `--no-verify`.

---

### Task 1: `LineLadderTarget` target-description dataclass + config

**Files:**
- Create: `calibration/line_target.py`
- Create: `calibration/config/line_target.yaml`

**Interfaces:**
- Produces: `DEFAULT_LINE_TARGET_CONFIG: str`, `LineLadderTarget` dataclass with fields `orientation: str`, `gaps_mm: List[float]`; property `line_count: int`; classmethods `from_dict(data: Dict[str, Any]) -> LineLadderTarget`, `from_yaml(path: Path) -> LineLadderTarget`; method `to_dict() -> Dict[str, Any]`.

- [ ] **Step 1: Write `calibration/line_target.py`**

```python
"""Printed line-ladder target used to validate RGB stereo distance measurement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

DEFAULT_LINE_TARGET_CONFIG = "calibration/config/line_target.yaml"


@dataclass(frozen=True)
class LineLadderTarget:
    orientation: str      # "vertical" | "horizontal"
    gaps_mm: List[float]  # consecutive real-world gaps between lines, mm; not required to be uniform

    def __post_init__(self) -> None:
        if self.orientation not in ("vertical", "horizontal"):
            raise ValueError(f"orientation must be 'vertical' or 'horizontal', got {self.orientation!r}")
        if len(self.gaps_mm) < 1:
            raise ValueError("gaps_mm must have at least one gap")

    @property
    def line_count(self) -> int:
        return len(self.gaps_mm) + 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineLadderTarget":
        target_type = str(data.get("target_type", "line_ladder")).lower()
        if target_type != "line_ladder":
            raise ValueError(f"Unsupported target_type '{target_type}'")
        return cls(
            orientation=str(data["orientation"]).lower(),
            gaps_mm=[float(v) for v in data["gaps_mm"]],
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "LineLadderTarget":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Line target config not found: {path}")
        with path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
        try:
            return cls.from_dict(data)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{exc} in {path}") from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_type": "line_ladder",
            "orientation": self.orientation,
            "gaps_mm": list(self.gaps_mm),
        }
```

- [ ] **Step 2: Write `calibration/config/line_target.yaml`**

```yaml
# Printed line-ladder target used by check_line_accuracy.py to validate RGB
# stereo distance measurement. A single orientation of parallel lines (not a
# crossing grid) -- see docs/superpowers/specs/2026-08-19-check-line-accuracy-design.md.
target_type: line_ladder
orientation: vertical   # vertical | horizontal -- the direction the printed lines run

# Consecutive real-world gaps (mm) between lines, in order: left-to-right for
# vertical lines, top-to-bottom for horizontal lines. Not required to be
# uniform. THESE ARE PLACEHOLDERS -- replace with calipers-measured distances
# from the actual printed target before any report using this config means
# anything.
gaps_mm: [10.0, 15.0, 20.0, 25.0, 30.0]
```

- [ ] **Step 3: Verify round-trip, validation, and YAML loading**

Run:

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
import tempfile
import yaml
from calibration.line_target import LineLadderTarget, DEFAULT_LINE_TARGET_CONFIG

# round-trip via from_dict/to_dict
t = LineLadderTarget(orientation='vertical', gaps_mm=[10.0, 20.0, 15.0])
assert t.line_count == 4
d = t.to_dict()
assert d == {'target_type': 'line_ladder', 'orientation': 'vertical', 'gaps_mm': [10.0, 20.0, 15.0]}
t2 = LineLadderTarget.from_dict(d)
assert t2 == t
print('from_dict/to_dict round-trip OK')

# invalid orientation raises
try:
    LineLadderTarget(orientation='diagonal', gaps_mm=[1.0])
    assert False, 'expected ValueError'
except ValueError:
    pass
print('orientation validation OK')

# from_yaml against a temp file
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / 'target.yaml'
    p.write_text(yaml.safe_dump({'target_type': 'line_ladder', 'orientation': 'horizontal', 'gaps_mm': [5.0, 6.0]}))
    loaded = LineLadderTarget.from_yaml(p)
    assert loaded.orientation == 'horizontal'
    assert loaded.gaps_mm == [5.0, 6.0]
    assert loaded.line_count == 3
print('from_yaml OK')

# the real shipped config loads without error
real = LineLadderTarget.from_yaml(Path(DEFAULT_LINE_TARGET_CONFIG))
assert real.orientation in ('vertical', 'horizontal')
assert len(real.gaps_mm) == real.line_count - 1
print('shipped calibration/config/line_target.yaml loads OK:', real)
"
```

Expected: four `... OK` lines printed, no assertion errors or tracebacks.

- [ ] **Step 4: Commit**

```bash
git add calibration/line_target.py calibration/config/line_target.yaml
git commit -m "$(cat <<'EOF'
Add LineLadderTarget dataclass and config for check_line_accuracy.py

Mirrors target_board.py's TargetBoard pattern (from_dict/from_yaml/
to_dict) for a printed line-ladder target with known, not-necessarily-
uniform gaps. gaps_mm in the shipped config are placeholders pending
a real printed target.
EOF
)"
```

---

### Task 2: Line detection and clustering

**Files:**
- Create: `check_line_accuracy.py` (this task only adds detection/geometry-in-image-space functions; CLI and orchestration come in later tasks)

**Interfaces:**
- Consumes: nothing from other tasks yet.
- Produces: `detect_line_segments(gray: np.ndarray, canny_low: int, canny_high: int, hough_threshold: int, hough_min_line_length: float, hough_max_line_gap: float) -> np.ndarray` (shape `(N, 4)`, columns `x1,y1,x2,y2`), `filter_segments_by_orientation(segments: np.ndarray, orientation: str, angle_tolerance_deg: float) -> np.ndarray`, `cluster_segments(segments: np.ndarray, orientation: str, cluster_distance_px: float) -> List[np.ndarray]` (each element an `(M, 2)` array of that cluster's segment endpoints, clusters sorted ascending by position), `fit_line(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]` (returns `(direction (2,), point_on_line (2,))`), `line_to_abc(direction: np.ndarray, point: np.ndarray) -> Tuple[float, float, float]`, `sample_along_line(points: np.ndarray, direction: np.ndarray, point: np.ndarray, n_samples: int, margin_fraction: float) -> np.ndarray` (shape `(n_samples, 2)`).

- [ ] **Step 1: Write `check_line_accuracy.py` header and detection/clustering functions**

```python
#!/usr/bin/env python3
"""Measure this rig's RGB stereo distance accuracy against a known printed
line-ladder target.

Detects a single-orientation ladder of parallel lines independently in both
RGB cameras, gets a geometrically valid correspondence for sample points
along each line via the epipolar constraint (this rig's cameras are a
verged ~18deg-toe-in pair, not a rectifiable parallel pair), triangulates
real 3D points using the rig's calibrated R/T, and reports measured-vs-
ground-truth mm error per gap. See
docs/superpowers/specs/2026-08-19-check-line-accuracy-design.md.

Usage:
    python check_line_accuracy.py --session captures/line_target/shot1
    python check_line_accuracy.py --captures captures/line_target
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_cameras import resolve_path  # noqa: E402
from calibration.line_target import DEFAULT_LINE_TARGET_CONFIG, LineLadderTarget  # noqa: E402
from calibration.stereo import StereoExtrinsics, collect_session_pairs  # noqa: E402
from register_pipeline import default_extrinsics_path  # noqa: E402

DEFAULT_CAPTURES = "captures/line_target"
DEFAULT_OUTPUT_DIR = "line_accuracy_reports"


def detect_line_segments(
    gray: np.ndarray, canny_low: int, canny_high: int,
    hough_threshold: int, hough_min_line_length: float, hough_max_line_gap: float,
) -> np.ndarray:
    """Grayscale -> blur -> Canny -> HoughLinesP. Returns (N, 4) [x1,y1,x2,y2]
    segments, or an empty (0, 4) array if none are found."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    segments = cv2.HoughLinesP(
        edges, 1, np.pi / 180, hough_threshold,
        minLineLength=hough_min_line_length, maxLineGap=hough_max_line_gap,
    )
    if segments is None:
        return np.zeros((0, 4), dtype=np.float64)
    return segments.reshape(-1, 4).astype(np.float64)


def filter_segments_by_orientation(
    segments: np.ndarray, orientation: str, angle_tolerance_deg: float,
) -> np.ndarray:
    """Keep only segments whose angle matches `orientation` within
    angle_tolerance_deg; others are discarded as noise."""
    if len(segments) == 0:
        return segments
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    angle_deg = np.mod(np.degrees(np.arctan2(dy, dx)), 180.0)  # line direction is unsigned
    target_angle = 90.0 if orientation == "vertical" else 0.0
    diff = np.abs(angle_deg - target_angle)
    diff = np.minimum(diff, 180.0 - diff)  # wrap around the 0/180 seam
    return segments[diff <= angle_tolerance_deg]


def cluster_segments(
    segments: np.ndarray, orientation: str, cluster_distance_px: float,
) -> List[np.ndarray]:
    """Group segments into per-line clusters by perpendicular coordinate (x
    for vertical lines, y for horizontal), collapsing a stripe's two edges
    or duplicate detections into one line. Returns point-array clusters
    (each segment contributes both endpoints), sorted ascending by the
    cluster's mean perpendicular coordinate."""
    if len(segments) == 0:
        return []
    axis = 0 if orientation == "vertical" else 1  # x for vertical, y for horizontal
    coords = (segments[:, axis] + segments[:, axis + 2]) / 2.0
    order = np.argsort(coords)
    member_lists: List[List[int]] = []
    running_means: List[float] = []
    for idx in order:
        coord = coords[idx]
        if member_lists and abs(coord - running_means[-1]) <= cluster_distance_px:
            member_lists[-1].append(int(idx))
            running_means[-1] = float(np.mean(coords[member_lists[-1]]))
        else:
            member_lists.append([int(idx)])
            running_means.append(float(coord))

    point_clusters = []
    for members in member_lists:
        endpoints_1 = segments[members][:, [0, 1]]
        endpoints_2 = segments[members][:, [2, 3]]
        point_clusters.append(np.vstack([endpoints_1, endpoints_2]))
    return point_clusters


def fit_line(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """cv2.fitLine on a cluster's points. Returns (direction (2,) unit
    vector, a point on the fitted line (2,))."""
    vx, vy, x0, y0 = cv2.fitLine(
        points.reshape(-1, 1, 2).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01,
    ).reshape(4)
    return np.array([vx, vy], dtype=np.float64), np.array([x0, y0], dtype=np.float64)


def line_to_abc(direction: np.ndarray, point: np.ndarray) -> Tuple[float, float, float]:
    """Implicit form a*x + b*y + c = 0, a^2 + b^2 = 1, from a direction and
    a point on the line (normal = direction rotated 90 degrees)."""
    a, b = float(direction[1]), float(-direction[0])
    norm = math.hypot(a, b)
    a, b = a / norm, b / norm
    c = -(a * float(point[0]) + b * float(point[1]))
    return a, b, c


def sample_along_line(
    points: np.ndarray, direction: np.ndarray, point: np.ndarray,
    n_samples: int, margin_fraction: float,
) -> np.ndarray:
    """n_samples points evenly spaced along the fitted line, spanning the
    cluster's own projected extent with margin_fraction trimmed off each end
    (Hough segment endpoints are noisy). Returns (n_samples, 2)."""
    projections = (points - point) @ direction
    lo, hi = float(projections.min()), float(projections.max())
    span = hi - lo
    lo, hi = lo + span * margin_fraction, hi - span * margin_fraction
    t = np.linspace(lo, hi, n_samples)
    return point[np.newaxis, :] + t[:, np.newaxis] * direction[np.newaxis, :]
```

- [ ] **Step 2: Syntax-check the file**

Run: `python3 -m py_compile check_line_accuracy.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Verify detection and clustering against a synthetic image**

Run:

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
import numpy as np
import cv2
import check_line_accuracy as m

image = np.zeros((400, 600), dtype=np.uint8)
expected_x = [50, 150, 300, 500]
for x in expected_x:
    cv2.line(image, (x, 0), (x, 399), 255, 3)

segments = m.detect_line_segments(
    image, canny_low=50, canny_high=150,
    hough_threshold=50, hough_min_line_length=100.0, hough_max_line_gap=10.0,
)
assert len(segments) > 0, 'no segments detected'

vertical = m.filter_segments_by_orientation(segments, 'vertical', angle_tolerance_deg=10.0)
assert len(vertical) > 0, 'no vertical segments survived the angle filter'

horizontal = m.filter_segments_by_orientation(segments, 'horizontal', angle_tolerance_deg=10.0)
assert len(horizontal) == 0, 'a vertical-line image should have no horizontal survivors'

clusters = m.cluster_segments(vertical, 'vertical', cluster_distance_px=15.0)
assert len(clusters) == 4, f'expected 4 clusters, got {len(clusters)}'

detected_x = sorted(float(np.mean(c[:, 0])) for c in clusters)
for got, want in zip(detected_x, expected_x):
    assert abs(got - want) < 3.0, f'got {got}, want {want}'
print('detect_line_segments / filter_segments_by_orientation / cluster_segments OK:', detected_x)

direction, origin = m.fit_line(clusters[0])
assert abs(abs(direction[1]) - 1.0) < 0.05, f'expected near-vertical direction, got {direction}'

a, b, c = m.line_to_abc(direction, origin)
assert abs(a * a + b * b - 1.0) < 1e-6
residual = a * expected_x[0] + b * 200.0 + c
assert abs(residual) < 3.0, f'fitted line does not pass near the known line, residual={residual}'
print('fit_line / line_to_abc OK')

samples = m.sample_along_line(clusters[0], direction, origin, n_samples=5, margin_fraction=0.1)
assert samples.shape == (5, 2)
assert np.all(samples[:, 1] > 0) and np.all(samples[:, 1] < 400)
assert np.all(np.abs(samples[:, 0] - expected_x[0]) < 3.0)
print('sample_along_line OK')
"
```

Expected: three `... OK` lines printed, no assertion errors.

- [ ] **Step 4: Commit**

```bash
git add check_line_accuracy.py
git commit -m "$(cat <<'EOF'
Add line detection and clustering to check_line_accuracy.py

Canny + HoughLinesP + angle filter + perpendicular-coordinate
clustering, verified against a synthetic image with known line
positions -- no camera hardware needed since this stage is pure
image processing.
EOF
)"
```

---

### Task 3: Epipolar correspondence and triangulation

**Files:**
- Modify: `check_line_accuracy.py` (add functions after Task 2's)

**Interfaces:**
- Consumes: nothing from Task 2 (this task's functions operate purely on point coordinates and camera parameters).
- Produces: `undistort_points(points: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray) -> np.ndarray`, `epipolar_line(point: np.ndarray, fundamental: np.ndarray) -> Tuple[float, float, float]`, `intersect_lines(line1: Tuple[float, float, float], line2: Tuple[float, float, float]) -> np.ndarray` (raises `ValueError` if parallel), `triangulate_point(p_a: np.ndarray, p_b: np.ndarray, camera_matrix_a: np.ndarray, camera_matrix_b: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray` (returns `(X, Y, Z)` metres, camera-A frame).

- [ ] **Step 1: Add the functions to `check_line_accuracy.py`**

Insert after `sample_along_line` (before any CLI code, which doesn't exist yet):

```python
def undistort_points(points: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    """cv2.undistortPoints with P=camera_matrix -- same convention as
    check_registration_error.py's undistort_points and
    calibration/stereo.py's _undistorted_pixels."""
    undistorted = cv2.undistortPoints(
        points.reshape(-1, 1, 2).astype(np.float64), camera_matrix, distortion, P=camera_matrix,
    )
    return undistorted.reshape(-1, 2)


def epipolar_line(point: np.ndarray, fundamental: np.ndarray) -> Tuple[float, float, float]:
    """Epipolar line in image B for one undistorted point in image A. Same
    convention as calibration/stereo.py's epipolar_residuals:
    line_b = [x, y, 1] @ F.T."""
    homogeneous = np.array([point[0], point[1], 1.0])
    a, b, c = homogeneous @ fundamental.T
    return float(a), float(b), float(c)


def intersect_lines(
    line1: Tuple[float, float, float], line2: Tuple[float, float, float],
) -> np.ndarray:
    """Intersection of two implicit-form (a, b, c) lines via the homogeneous
    cross product. Raises ValueError if the lines are parallel."""
    cross = np.cross(np.array(line1), np.array(line2))
    if abs(cross[2]) < 1e-9:
        raise ValueError("Lines are parallel; no intersection")
    return np.array([cross[0] / cross[2], cross[1] / cross[2]])


def triangulate_point(
    p_a: np.ndarray, p_b: np.ndarray,
    camera_matrix_a: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray,
) -> np.ndarray:
    """Triangulate one undistorted point pair into camera-A-frame 3D
    (X, Y, Z), metres. Same P_a/P_b convention as register_features.py's
    triangulate_matches, but keeps the full point rather than only depth."""
    projection_a = camera_matrix_a @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_b = camera_matrix_b @ np.hstack([R, T.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(
        projection_a, projection_b,
        p_a.reshape(2, 1).astype(np.float64), p_b.reshape(2, 1).astype(np.float64),
    )
    homogeneous = homogeneous / homogeneous[3]
    return homogeneous[:3].reshape(3)
```

- [ ] **Step 2: Syntax-check**

Run: `python3 -m py_compile check_line_accuracy.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Verify the geometry with a synthetic (non-trivial R,T) stereo pair**

This is pure linear algebra — no images, no hardware, and it uses a rotated `R` (not just identity) so it doesn't accidentally only work for an axis-aligned special case.

Run:

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
import numpy as np
import cv2
import check_line_accuracy as m

K_a = np.array([[1000.0, 0, 320.0], [0, 1000.0, 240.0], [0, 0, 1.0]])
K_b = K_a.copy()
R = cv2.Rodrigues(np.array([0.0, np.radians(15.0), 0.0]))[0]  # 15deg toe-in, not identity
T = np.array([0.05, 0.0, 0.01])

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

E = skew(T) @ R
F = np.linalg.inv(K_b).T @ E @ np.linalg.inv(K_a)

X_true = np.array([0.02, 0.01, 0.5])  # metres, camera-A frame

P_a = K_a @ np.hstack([np.eye(3), np.zeros((3, 1))])
P_b = K_b @ np.hstack([R, T.reshape(3, 1)])

x_a_h = P_a @ np.append(X_true, 1.0)
x_a = x_a_h[:2] / x_a_h[2]
x_b_h = P_b @ np.append(X_true, 1.0)
x_b = x_b_h[:2] / x_b_h[2]

# undistort_points is identity when distortion is zero
distortion = np.zeros(5)
undist = m.undistort_points(np.array([[100.0, 50.0], [200.0, 150.0]]), K_a, distortion)
assert np.allclose(undist, [[100.0, 50.0], [200.0, 150.0]], atol=1e-3)
print('undistort_points OK (identity check with zero distortion)')

line_b = m.epipolar_line(x_a, F)
residual = line_b[0] * x_b[0] + line_b[1] * x_b[1] + line_b[2]
assert abs(residual) < 1e-6, f'point not on its own epipolar line, residual={residual}'
print('epipolar_line OK')

line_vertical_b = (1.0, 0.0, -x_b[0])  # a known line through x_b, standing in for a detected target line
p_b_recovered = m.intersect_lines(line_b, line_vertical_b)
assert np.allclose(p_b_recovered, x_b, atol=1e-6), f'{p_b_recovered} vs {x_b}'
print('intersect_lines OK')

try:
    m.intersect_lines((1.0, 0.0, 0.0), (1.0, 0.0, 5.0))
    assert False, 'expected ValueError for parallel lines'
except ValueError:
    pass
print('intersect_lines parallel-line guard OK')

X_recovered = m.triangulate_point(x_a, p_b_recovered, K_a, K_b, R, T)
assert np.allclose(X_recovered, X_true, atol=1e-6), f'{X_recovered} vs {X_true}'
print('triangulate_point OK')
"
```

Expected: five `... OK` lines printed, no assertion errors. This confirms the full epipolar-correspondence-and-triangulation chain recovers the exact synthetic 3D point for a genuinely rotated (non-identity) `R`.

- [ ] **Step 4: Commit**

```bash
git add check_line_accuracy.py
git commit -m "$(cat <<'EOF'
Add epipolar correspondence and triangulation to check_line_accuracy.py

undistort_points/epipolar_line follow the exact convention
calibration/stereo.py's epipolar_residuals already established;
triangulate_point follows register_features.py's triangulate_matches
convention but keeps the full 3D point. Verified with a synthetic
stereo pair using a genuinely rotated R (not just identity), which
exactly recovers a known 3D point.
EOF
)"
```

---

### Task 4: Per-session line-gap measurement and annotation

**Files:**
- Modify: `check_line_accuracy.py` (add functions after Task 3's)

**Interfaces:**
- Consumes: `fit_line`, `line_to_abc`, `sample_along_line`, `detect_line_segments`, `filter_segments_by_orientation`, `cluster_segments` (Task 2); `undistort_points`, `epipolar_line`, `intersect_lines`, `triangulate_point` (Task 3).
- Produces: `detect_and_cluster(image: np.ndarray, orientation: str, args: argparse.Namespace) -> List[np.ndarray]`, `measure_line_gaps(clusters_a: List[np.ndarray], clusters_b: List[np.ndarray], camera_matrix_a: np.ndarray, distortion_a: np.ndarray, camera_matrix_b: np.ndarray, distortion_b: np.ndarray, R: np.ndarray, T: np.ndarray, fundamental: np.ndarray, n_samples: int, margin_fraction: float) -> List[List[float]]` (one list per gap, each a list of mm distance samples), `save_annotated(image: np.ndarray, clusters: List[np.ndarray], path: Path) -> None`, `process_session(label: str, path_a: Path, path_b: Path, extrinsics: StereoExtrinsics, target: LineLadderTarget, args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]` (returns `{"label", "skipped": str}` or `{"label", "gaps_mm": List[List[float]]}`).

- [ ] **Step 1: Add the functions to `check_line_accuracy.py`**

Insert after `triangulate_point`:

```python
def detect_and_cluster(image: np.ndarray, orientation: str, args: argparse.Namespace) -> List[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segments = detect_line_segments(
        gray, args.canny_low, args.canny_high,
        args.hough_threshold, args.hough_min_line_length, args.hough_max_line_gap,
    )
    segments = filter_segments_by_orientation(segments, orientation, args.angle_tolerance_deg)
    return cluster_segments(segments, orientation, args.cluster_distance_px)


def measure_line_gaps(
    clusters_a: List[np.ndarray], clusters_b: List[np.ndarray],
    camera_matrix_a: np.ndarray, distortion_a: np.ndarray,
    camera_matrix_b: np.ndarray, distortion_b: np.ndarray,
    R: np.ndarray, T: np.ndarray, fundamental: np.ndarray,
    n_samples: int, margin_fraction: float,
) -> List[List[float]]:
    """For each matched line index (assumes len(clusters_a) == len(clusters_b),
    checked by the caller), sample points along line i in A, find their
    epipolar-consistent correspondence on line i in B, triangulate, then
    return one list per gap of the 3D distances (mm) between consecutive
    lines' triangulated points -- one entry per sample index that didn't hit
    a degenerate (parallel) epipolar intersection."""
    line_points_3d: List[List[np.ndarray]] = []
    for cluster_a, cluster_b in zip(clusters_a, clusters_b):
        undistorted_a = undistort_points(cluster_a, camera_matrix_a, distortion_a)
        undistorted_b = undistort_points(cluster_b, camera_matrix_b, distortion_b)
        direction_a, origin_a = fit_line(undistorted_a)
        direction_b, origin_b = fit_line(undistorted_b)
        line_b_abc = line_to_abc(direction_b, origin_b)
        samples_a = sample_along_line(undistorted_a, direction_a, origin_a, n_samples, margin_fraction)

        points_3d = []
        for p_a in samples_a:
            line_in_b = epipolar_line(p_a, fundamental)
            try:
                p_b = intersect_lines(line_in_b, line_b_abc)
            except ValueError:
                continue
            points_3d.append(triangulate_point(p_a, p_b, camera_matrix_a, camera_matrix_b, R, T))
        line_points_3d.append(points_3d)

    gaps_mm: List[List[float]] = []
    for i in range(len(line_points_3d) - 1):
        pts_i, pts_i1 = line_points_3d[i], line_points_3d[i + 1]
        n = min(len(pts_i), len(pts_i1))
        gaps_mm.append([float(np.linalg.norm(pts_i1[k] - pts_i[k]) * 1000.0) for k in range(n)])
    return gaps_mm


def save_annotated(image: np.ndarray, clusters: List[np.ndarray], path: Path) -> None:
    annotated = image.copy()
    for index, cluster in enumerate(clusters):
        direction, origin = fit_line(cluster)
        projections = (cluster - origin) @ direction
        p1 = origin + projections.min() * direction
        p2 = origin + projections.max() * direction
        cv2.line(annotated, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 0), 2)
        midpoint = tuple(((p1 + p2) / 2).astype(int))
        cv2.putText(annotated, str(index), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def process_session(
    label: str, path_a: Path, path_b: Path,
    extrinsics: StereoExtrinsics, target: LineLadderTarget,
    args: argparse.Namespace, output_dir: Path,
) -> Dict[str, Any]:
    image_a = cv2.imread(str(path_a))
    image_b = cv2.imread(str(path_b))
    if image_a is None or image_b is None:
        return {"label": label, "skipped": f"could not read {path_a} or {path_b}"}

    clusters_a = detect_and_cluster(image_a, target.orientation, args)
    clusters_b = detect_and_cluster(image_b, target.orientation, args)

    save_annotated(image_a, clusters_a, output_dir / f"{label}_{args.camera_a}_lines.jpg")
    save_annotated(image_b, clusters_b, output_dir / f"{label}_{args.camera_b}_lines.jpg")

    if len(clusters_a) != target.line_count or len(clusters_b) != target.line_count:
        return {
            "label": label,
            "skipped": f"expected {target.line_count} lines, found "
                       f"{len(clusters_a)} in {args.camera_a}, {len(clusters_b)} in {args.camera_b}",
        }

    gaps_mm = measure_line_gaps(
        clusters_a, clusters_b,
        extrinsics.camera_matrix_a, extrinsics.distortion_a,
        extrinsics.camera_matrix_b, extrinsics.distortion_b,
        extrinsics.R, extrinsics.T, extrinsics.fundamental,
        args.samples_per_line, 0.1,
    )
    return {"label": label, "gaps_mm": gaps_mm}
```

- [ ] **Step 2: Syntax-check**

Run: `python3 -m py_compile check_line_accuracy.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Verify `process_session` end-to-end against a synthetic two-camera render**

This builds fake camera-A / camera-B images in memory (a straightforward stereo pair, `R = I`, baseline along X — deliberately simpler than Task 3's rotated-R check, since this task validates the *image-processing* wiring, not the geometry, which Task 3 already proved works for a general `R`), with 4 known vertical lines at known 3D positions giving known mm gaps, and confirms `process_session` recovers them.

Run:

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
import argparse
import tempfile
from pathlib import Path

import numpy as np
import cv2

import check_line_accuracy as m
from calibration.stereo import StereoExtrinsics
from calibration.line_target import LineLadderTarget

K = np.array([[1000.0, 0, 320.0], [0, 1000.0, 240.0], [0, 0, 1.0]])
R = np.eye(3)
T = np.array([0.05, 0.0, 0.0])

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

E = skew(T) @ R
F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

# 4 lines at X = -0.06, -0.04, -0.01, 0.02 m, Z = 0.5 m -> gaps 20, 30, 30 mm
xs_world = [-0.06, -0.04, -0.01, 0.02]
gaps_truth = [1000.0 * (xs_world[i + 1] - xs_world[i]) for i in range(len(xs_world) - 1)]
depth = 0.5

def project_x(x_world, K_mat, R_mat, T_vec):
    X = np.array([x_world, 0.0, depth])
    Xc = R_mat @ X + T_vec
    x_px = K_mat[0, 0] * Xc[0] / Xc[2] + K_mat[0, 2]
    return x_px

image_a = np.full((480, 640, 3), 40, dtype=np.uint8)
image_b = np.full((480, 640, 3), 40, dtype=np.uint8)
for x_world in xs_world:
    xa = int(round(project_x(x_world, K, np.eye(3), np.zeros(3))))
    xb = int(round(project_x(x_world, K, R, T)))
    cv2.line(image_a, (xa, 0), (xa, 479), (255, 255, 255), 3)
    cv2.line(image_b, (xb, 0), (xb, 479), (255, 255, 255), 3)

extrinsics = StereoExtrinsics(
    name_a='rgb_cam1', name_b='rgb_cam2',
    image_size_a=(640, 480), image_size_b=(640, 480),
    camera_matrix_a=K, distortion_a=np.zeros(5),
    camera_matrix_b=K, distortion_b=np.zeros(5),
    R=R, T=T, essential=E, fundamental=F,
    reprojection_error_px=0.0, views_used=1, points_used=8,
)
target = LineLadderTarget(orientation='vertical', gaps_mm=gaps_truth)
args = argparse.Namespace(
    canny_low=50, canny_high=150, hough_threshold=50,
    hough_min_line_length=100.0, hough_max_line_gap=10.0,
    angle_tolerance_deg=10.0, cluster_distance_px=15.0, samples_per_line=5,
    camera_a='rgb_cam1', camera_b='rgb_cam2',
)

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    path_a = tmp_path / 'rgb_cam1.jpg'
    path_b = tmp_path / 'rgb_cam2.jpg'
    cv2.imwrite(str(path_a), image_a)
    cv2.imwrite(str(path_b), image_b)
    output_dir = tmp_path / 'output'

    result = m.process_session('synthetic', path_a, path_b, extrinsics, target, args, output_dir)
    assert 'skipped' not in result, f\"unexpectedly skipped: {result.get('skipped')}\"
    assert len(result['gaps_mm']) == 3, f\"expected 3 gaps, got {len(result['gaps_mm'])}\"
    for i, samples in enumerate(result['gaps_mm']):
        assert samples, f'gap {i} has no successful samples'
        mean_mm = float(np.mean(samples))
        truth = gaps_truth[i]
        tolerance = max(3.0, 0.15 * truth)
        assert abs(mean_mm - truth) < tolerance, f'gap {i}: measured {mean_mm:.2f}mm, truth {truth:.2f}mm'
        print(f'gap {i}: measured {mean_mm:.2f}mm (truth {truth:.2f}mm) OK')

    assert (output_dir / 'synthetic_rgb_cam1_lines.jpg').exists()
    assert (output_dir / 'synthetic_rgb_cam2_lines.jpg').exists()
    print('save_annotated output files OK')
"
```

Expected: three `gap N: measured ... OK` lines with measured values close to 20.00mm/30.00mm/30.00mm, plus `save_annotated output files OK`, no assertion errors.

- [ ] **Step 4: Commit**

```bash
git add check_line_accuracy.py
git commit -m "$(cat <<'EOF'
Add per-session line-gap measurement and annotation to check_line_accuracy.py

process_session wires detection (Task 2) and epipolar triangulation
(Task 3) together per capture session, and saves an annotated JPEG per
camera for visual QA. Verified end-to-end against a synthetic rendered
stereo pair with known 3D line positions -- recovers the known mm gaps
without needing a real camera or printed target.
EOF
)"
```

---

### Task 5: CLI, session collection, reporting, and `main()`

**Files:**
- Modify: `check_line_accuracy.py` (add CLI/orchestration functions and the `if __name__ == "__main__":` entry point)

**Interfaces:**
- Consumes: `process_session` (Task 4); `LineLadderTarget`, `DEFAULT_LINE_TARGET_CONFIG` (Task 1, imported in Task 2's header); `StereoExtrinsics`, `collect_session_pairs`, `default_extrinsics_path`, `resolve_path` (imported in Task 2's header).
- Produces: `parse_args() -> argparse.Namespace`, `aggregate_results(results: List[Dict[str, Any]], target: LineLadderTarget) -> List[Dict[str, Any]]`, `write_report(output_dir: Path, args: argparse.Namespace, extrinsics_path: Path, target_path: Path, results: List[Dict[str, Any]], gap_summaries: List[Dict[str, Any]]) -> None`, `main() -> int`.

- [ ] **Step 1: Add the functions to `check_line_accuracy.py`**

Insert after `process_session`:

```python
def aggregate_results(
    results: List[Dict[str, Any]], target: LineLadderTarget,
) -> List[Dict[str, Any]]:
    """Combine every scored session's per-gap sample lists into one flat
    list per gap index, and compute mean/std/error against ground truth."""
    scored = [r for r in results if "skipped" not in r]
    gap_summaries = []
    for i, ground_truth in enumerate(target.gaps_mm):
        samples: List[float] = []
        for r in scored:
            samples.extend(r["gaps_mm"][i])
        if samples:
            mean_mm = float(np.mean(samples))
            std_mm = float(np.std(samples))
            error_mm = mean_mm - ground_truth
            error_pct = 100.0 * error_mm / ground_truth
        else:
            mean_mm = std_mm = error_mm = error_pct = float("nan")
        gap_summaries.append({
            "index": i,
            "ground_truth_mm": ground_truth,
            "n_samples": len(samples),
            "mean_mm": mean_mm,
            "std_mm": std_mm,
            "error_mm": error_mm,
            "error_pct": error_pct,
        })
    return gap_summaries


def write_report(
    output_dir: Path, args: argparse.Namespace,
    extrinsics_path: Path, target_path: Path,
    results: List[Dict[str, Any]], gap_summaries: List[Dict[str, Any]],
) -> None:
    scored = [r for r in results if "skipped" not in r]
    lines = [
        f"Line accuracy check ({args.camera_a} / {args.camera_b}, stereo triangulation)",
        "=" * 66,
        "",
        f"  extrinsics:       {extrinsics_path}",
        f"  target:           {target_path}",
        f"  samples per line: {args.samples_per_line}",
        f"  sessions scored:  {len(scored)}/{len(results)}",
        "",
        "Per-session",
        "-" * 66,
    ]
    for r in results:
        if "skipped" in r:
            lines.append(f"  {r['label']:<24} skipped: {r['skipped']}")
        else:
            lines.append(f"  {r['label']:<24} ok")
    lines.append("")
    lines.append("Per-gap (mm)")
    lines.append("-" * 66)
    lines.append(f"  {'idx':<4} {'truth':>8} {'measured':>9} {'std':>7} {'error':>8} {'error%':>8}")
    for g in gap_summaries:
        lines.append(
            f"  {g['index']:<4} {g['ground_truth_mm']:>8.2f} {g['mean_mm']:>9.2f} "
            f"{g['std_mm']:>7.2f} {g['error_mm']:>8.2f} {g['error_pct']:>7.1f}%"
        )
    errors = [abs(g["error_mm"]) for g in gap_summaries if not math.isnan(g["error_mm"])]
    if errors:
        lines.append("")
        lines.append(f"  mean abs error: {np.mean(errors):.2f} mm    max abs error: {np.max(errors):.2f} mm")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "camera_a": args.camera_a,
        "camera_b": args.camera_b,
        "extrinsics": str(extrinsics_path),
        "target": str(target_path),
        "samples_per_line": args.samples_per_line,
        "sessions": [
            {"label": r["label"], "skipped": r.get("skipped"), "gaps_mm": r.get("gaps_mm")}
            for r in results
        ],
        "gaps": gap_summaries,
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure RGB stereo distance accuracy against a known line-ladder target.",
    )
    parser.add_argument(
        "--captures", nargs="+", default=None,
        help=f"Directories holding capture sessions (default: {DEFAULT_CAPTURES}).",
    )
    parser.add_argument(
        "--session", default=None,
        help="Restrict to one session folder instead of every session under --captures.",
    )
    parser.add_argument("--camera-a", default="rgb_cam1")
    parser.add_argument("--camera-b", default="rgb_cam2")
    parser.add_argument("--extrinsics", default=None,
                        help="Stereo extrinsics JSON (default: resolved via "
                             "register_pipeline.default_extrinsics_path).")
    parser.add_argument("--target", default=None,
                        help=f"Line target YAML (default: {DEFAULT_LINE_TARGET_CONFIG}).")
    parser.add_argument("--output", default=None,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--hough-threshold", type=int, default=50)
    parser.add_argument("--hough-min-line-length", type=float, default=100.0)
    parser.add_argument("--hough-max-line-gap", type=float, default=10.0)
    parser.add_argument("--angle-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--cluster-distance-px", type=float, default=15.0)
    parser.add_argument("--samples-per-line", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    extrinsics_path = resolve_path(
        args.extrinsics or default_extrinsics_path(args.camera_a, args.camera_b)
    )
    if not extrinsics_path.exists():
        raise SystemExit(f"No stereo extrinsics at {extrinsics_path}.")
    extrinsics = StereoExtrinsics.load_json(extrinsics_path)

    target_path = resolve_path(args.target or DEFAULT_LINE_TARGET_CONFIG)
    target = LineLadderTarget.from_yaml(target_path)

    if args.session:
        session_dir = resolve_path(args.session)
        capture_dirs = [session_dir.parent]
    else:
        capture_values = args.captures or [DEFAULT_CAPTURES]
        capture_dirs = [resolve_path(value) for value in capture_values]

    session_pairs, notes = collect_session_pairs(capture_dirs, args.camera_a, args.camera_b)
    for note in notes:
        print(f"  {note}")
    if args.session:
        session_pairs = [pair for pair in session_pairs if pair[0] == session_dir.name]
    if not session_pairs:
        raise SystemExit("No usable session found.")

    output_dir = resolve_path(args.output or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for label, path_a, path_b in session_pairs:
        print(f"Processing {label}...")
        result = process_session(label, path_a, path_b, extrinsics, target, args, output_dir)
        if "skipped" in result:
            print(f"  skipped: {result['skipped']}")
        results.append(result)

    gap_summaries = aggregate_results(results, target)
    write_report(output_dir, args, extrinsics_path, target_path, results, gap_summaries)

    for g in gap_summaries:
        print(f"  gap {g['index']}: truth {g['ground_truth_mm']:.2f} mm, "
              f"measured {g['mean_mm']:.2f}+/-{g['std_mm']:.2f} mm, error {g['error_mm']:+.2f} mm")
    print(f"Saved report.txt / result.json to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Syntax-check and `--help`**

Run: `python3 -m py_compile check_line_accuracy.py`
Expected: no output, exit code 0.

Run: `python3 check_line_accuracy.py --help`
Expected: argparse usage text listing all flags, exit code 0.

- [ ] **Step 3: End-to-end verification via the real CLI on synthetic files on disk**

Builds the same synthetic stereo pair as Task 4, but this time writes real `rgb_cam1.jpg`/`rgb_cam2.jpg`, a real `extrinsics.json` (via `StereoExtrinsics.save_json`), and a real `line_target.yaml` to a temp directory, then invokes `check_line_accuracy.py` as a subprocess exactly as a user would, and checks its actual file outputs.

Run:

```bash
python3 -c "
import sys, subprocess, json, tempfile
from pathlib import Path

import numpy as np
import cv2
import yaml

sys.path.insert(0, '.')
from calibration.stereo import StereoExtrinsics

K = np.array([[1000.0, 0, 320.0], [0, 1000.0, 240.0], [0, 0, 1.0]])
R = np.eye(3)
T = np.array([0.05, 0.0, 0.0])

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

E = skew(T) @ R
F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

xs_world = [-0.06, -0.04, -0.01, 0.02]
gaps_truth = [1000.0 * (xs_world[i + 1] - xs_world[i]) for i in range(len(xs_world) - 1)]
depth = 0.5

def project_x(x_world, R_mat, T_vec):
    X = np.array([x_world, 0.0, depth])
    Xc = R_mat @ X + T_vec
    return K[0, 0] * Xc[0] / Xc[2] + K[0, 2]

image_a = np.full((480, 640, 3), 40, dtype=np.uint8)
image_b = np.full((480, 640, 3), 40, dtype=np.uint8)
for x_world in xs_world:
    xa = int(round(project_x(x_world, np.eye(3), np.zeros(3))))
    xb = int(round(project_x(x_world, R, T)))
    cv2.line(image_a, (xa, 0), (xa, 479), (255, 255, 255), 3)
    cv2.line(image_b, (xb, 0), (xb, 479), (255, 255, 255), 3)

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    session_dir = tmp_path / 'captures' / 'shot1'
    session_dir.mkdir(parents=True)
    cv2.imwrite(str(session_dir / 'rgb_cam1.jpg'), image_a)
    cv2.imwrite(str(session_dir / 'rgb_cam2.jpg'), image_b)

    extrinsics_path = tmp_path / 'extrinsics.json'
    StereoExtrinsics(
        name_a='rgb_cam1', name_b='rgb_cam2',
        image_size_a=(640, 480), image_size_b=(640, 480),
        camera_matrix_a=K, distortion_a=np.zeros(5),
        camera_matrix_b=K, distortion_b=np.zeros(5),
        R=R, T=T, essential=E, fundamental=F,
        reprojection_error_px=0.0, views_used=1, points_used=8,
    ).save_json(extrinsics_path)

    target_path = tmp_path / 'line_target.yaml'
    target_path.write_text(yaml.safe_dump({
        'target_type': 'line_ladder', 'orientation': 'vertical', 'gaps_mm': gaps_truth,
    }))

    output_dir = tmp_path / 'output'

    proc = subprocess.run(
        [sys.executable, 'check_line_accuracy.py',
         '--session', str(session_dir),
         '--extrinsics', str(extrinsics_path),
         '--target', str(target_path),
         '--output', str(output_dir),
         '--camera-a', 'rgb_cam1', '--camera-b', 'rgb_cam2'],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f'exit {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}'

    report_txt = output_dir / 'report.txt'
    result_json = output_dir / 'result.json'
    assert report_txt.exists(), proc.stdout + proc.stderr
    assert result_json.exists(), proc.stdout + proc.stderr
    assert 'Line accuracy check' in report_txt.read_text()

    summary = json.loads(result_json.read_text())
    assert len(summary['gaps']) == 3
    for i, gap in enumerate(summary['gaps']):
        truth = gaps_truth[i]
        tolerance = max(3.0, 0.15 * truth)
        assert abs(gap['mean_mm'] - truth) < tolerance, f\"gap {i}: {gap['mean_mm']} vs {truth}\"

    assert (output_dir / 'shot1_rgb_cam1_lines.jpg').exists()
    assert (output_dir / 'shot1_rgb_cam2_lines.jpg').exists()
    print('End-to-end CLI run OK -- report.txt/result.json/annotated JPEGs all correct:')
    print(proc.stdout)
"
```

Expected: `End-to-end CLI run OK ...` followed by the script's own stdout (per-gap measured/truth/error lines), no assertion errors.

- [ ] **Step 4: Commit**

```bash
git add check_line_accuracy.py
git commit -m "$(cat <<'EOF'
Add CLI, session collection, and reporting to check_line_accuracy.py

Wires parse_args/main() around process_session, reusing
collect_session_pairs and default_extrinsics_path from the existing
calibration/registration scripts. Writes report.txt + result.json,
matching check_registration_error.py's convention. Verified end-to-end
by invoking the real CLI as a subprocess against synthetic files on
disk (rendered images, extrinsics.json, line_target.yaml) -- no real
camera or printed target needed.
EOF
)"
```

---

### Task 6: Repo integration — README and `.gitignore`

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`

**Interfaces:** none (documentation/config only).

- [ ] **Step 1: Add `line_accuracy_reports/` to `.gitignore`**

Add this block after the existing `# Registration outputs` section (matches the existing grouped-by-purpose style):

```
# Line-accuracy check outputs
line_accuracy_reports/
```

- [ ] **Step 2: Add a stage-table row to `README.md`**

In the stage table (the block starting `| Stage               | Script                 | Output |`), add a new row after the `Cross-validation` row:

```
| Line accuracy        | `check_line_accuracy.py` | `line_accuracy_reports/`                                |
```

- [ ] **Step 3: Add a short usage section to `README.md`**

Near the `check_registration_error.py` mention (search for `check_registration_error.py` in `README.md` to find the right spot), add:

```markdown
`check_line_accuracy.py` measures how accurately the RGB stereo pair
recovers real-world distance, against a printed line-ladder target with
known (calipers-measured) gaps. Both cameras must capture the target at
once (a session dir, like every other stereo script); it detects the
lines, gets a valid point correspondence between the two views via the
epipolar constraint (this rig's cameras are a verged ~18deg-toe-in pair,
so a rectified-pair shortcut isn't safe), and triangulates real 3D
distances using the calibrated extrinsics -- distance is never a manual
input.

```bash
python check_line_accuracy.py --session captures/line_target/shot1
```

Edit `calibration/config/line_target.yaml`'s `gaps_mm` to match your actual
printed target before trusting the report.
```

- [ ] **Step 4: Verify the README changes render sensibly**

Run: `grep -n "check_line_accuracy" README.md`
Expected: three matches (stage table row, usage section heading text, and the `python check_line_accuracy.py` example line).

Run: `grep -n "line_accuracy_reports" .gitignore`
Expected: one match.

- [ ] **Step 5: Commit**

```bash
git add README.md .gitignore
git commit -m "$(cat <<'EOF'
Document check_line_accuracy.py in README and ignore its output dir

Adds the stage-table row and usage section README.md's other
calibration/validation scripts already have, and git-ignores
line_accuracy_reports/ as generated data (matching color_reports/,
calibration/results/, etc).
EOF
)"
```
