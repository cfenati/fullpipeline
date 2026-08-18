# Stereo Cross-Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `cross_validate_stereo.py` script that checks the already-fitted
`rgb_cam1`/`rgb_cam2` stereo extrinsics against a second, independently
captured checkerboard pose set (`captures/cross-validation`), by triangulating
held-out corners through the frozen `R`/`T`/`K`/distortion and comparing the
reconstructed corner-to-corner distances against the board's known printed
square size — a check that in-sample reprojection/epipolar error cannot
perform, since that error is graded on the very poses that set the scale.

**Architecture:** Two small, reusable functions land in
`calibration/stereo.py` (a triangulation helper factored out of the existing
`triangulation_closure`, and a new `adjacent_corner_distance_errors`); a thin
argparse CLI (`cross_validate_stereo.py`, sibling to `stereo_calibrate.py`)
wires them to the held-out capture directory and writes results to a new
`cross_validation/` subfolder. The script only ever *reads*
`extrinsics.json`; every write happens inside the new subfolder, whose path is
built in code (`<out>/stereo_<a>_<b>/cross_validation/`), not taken verbatim
from `--out`, so it cannot be pointed at `extrinsics.json` even by mistake.

**Tech Stack:** Python 3.9, OpenCV (`cv2.triangulatePoints`, already a
dependency), NumPy, Matplotlib (already a dependency) — no new packages.

## Global Constraints

- **Never write to an existing calibration artifact.** `extrinsics.json`,
  `intrinsics.json`, `report.txt`/figures under `calibration/results/stereo_<a>_<b>/`
  are read-only inputs to this feature. The new script must not open any of
  them for writing, and Task 2's verification step must prove this with a
  checksum, not just an eyeball check.
- **No test suite exists in this repo** (confirmed in `CLAUDE.md`). Task
  verification steps below use real, already-captured data
  (`captures/cross-validation` and `calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json`
  both already exist on disk) instead of `pytest` — this is geometry-only
  code with no hardware dependency, so it is fully runnable now, unlike the
  camera-facing scripts.
- **Commit narrowly, per task.** Working directly on `master` (user declined
  a worktree — substantial unrelated pre-existing uncommitted work is already
  there and the plan was written against it). Each task commits *only* the
  files its own task touches (e.g. `calibration/stereo.py`,
  `calibration/target_board.py` for Task 1) — never `git add -A` / `git add .`.
  The user's other pre-existing uncommitted changes (`cameras/`,
  `capture_pipeline.py`, `design/`, etc.) must stay untouched and still
  uncommitted throughout.
- **Follow existing conventions:** `from __future__ import annotations`, type
  hints, `argparse` CLI with `config.yaml` defaults resolved via
  `calibrate_cameras.load_config`/`resolve_path`, docstrings that explain the
  *why* the way the rest of `calibration/stereo.py` does.

---

### Task 1: Library support — `TargetBoard.from_dict` and `adjacent_corner_distance_errors`

**Files:**
- Modify: `calibration/target_board.py`
- Modify: `calibration/stereo.py`

**Interfaces:**
- Produces: `TargetBoard.from_dict(data: Dict[str, Any]) -> TargetBoard` (classmethod)
- Produces: `calibration.stereo._triangulate_observation(observation, projection_a, projection_b, camera_matrix_a, distortion_a, camera_matrix_b, distortion_b) -> Optional[Tuple[np.ndarray, np.ndarray]]` — `(points_3d, board_points)` for valid points, or `None` if fewer than 4 triangulate cleanly.
- Produces: `calibration.stereo.adjacent_corner_distance_errors(extrinsics: StereoExtrinsics, observations: Sequence[StereoObservation], board: TargetBoard, tolerance_fraction: float = 0.1) -> Dict[str, Any]` with keys `square_size_mm, tolerance_mm, per_view, pairs_total, mean_error_mm, rms_error_mm, max_abs_error_mm, std_error_mm, mean_relative_error_pct, rms_relative_error_pct`.
- Consumes: existing `StereoExtrinsics`, `StereoObservation`, `_undistorted_pixels`, `_fit_rigid` already in `calibration/stereo.py`.

- [ ] **Step 1: Add `TargetBoard.from_dict`, refactor `from_yaml` to use it**

In `calibration/target_board.py`, replace the body of `from_yaml` and add
`from_dict` right above it:

```python
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetBoard":
        target_type = str(data.get("target_type", "charuco")).lower()
        if target_type != "charuco":
            raise ValueError(f"Unsupported target_type '{target_type}'")

        return cls(
            squares_x=int(data["squares_x"]),
            squares_y=int(data["squares_y"]),
            square_size_m=float(data["square_size_m"]),
            marker_size_m=float(data["marker_size_m"]),
            dictionary=str(data.get("dictionary", "DICT_4X4_50")),
            legacy_pattern=bool(data.get("legacy_pattern", True)),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "TargetBoard":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Board config not found: {path}")

        with path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}

        try:
            return cls.from_dict(data)
        except ValueError as exc:
            raise ValueError(f"{exc} in {path}") from exc
```

This preserves `from_yaml`'s exact existing error text
(`"Unsupported target_type 'xxx' in <path>"`) and behavior — it's a pure
extract-method refactor.

- [ ] **Step 2: Verify the refactor**

```bash
python -c "
from calibration.target_board import TargetBoard
b = TargetBoard.from_yaml('calibration/config/charuco_11x8.yaml')
b2 = TargetBoard.from_dict(b.to_dict())
assert b == b2, (b, b2)
print('TargetBoard.from_dict OK:', b2)
"
```
Expected: prints `TargetBoard.from_dict OK: TargetBoard(...)`, no traceback.

- [ ] **Step 3: Factor `_triangulate_observation` out of `triangulation_closure`**

In `calibration/stereo.py`, insert this helper immediately above
`triangulation_closure` (which currently starts at line 1261):

```python
def _triangulate_observation(
    observation: StereoObservation,
    projection_a: np.ndarray,
    projection_b: np.ndarray,
    camera_matrix_a: np.ndarray,
    distortion_a: np.ndarray,
    camera_matrix_b: np.ndarray,
    distortion_b: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Triangulated 3D corners and their known board-space coordinates.

    Points whose triangulated homogeneous weight is too small to trust are
    dropped. Returns ``None`` when fewer than 4 points survive — too few to
    say anything about the view.
    """
    points_a = _undistorted_pixels(observation.left_points, camera_matrix_a, distortion_a)
    points_b = _undistorted_pixels(observation.right_points, camera_matrix_b, distortion_b)
    homogeneous = cv2.triangulatePoints(projection_a, projection_b, points_a.T, points_b.T)
    w = homogeneous[3]
    valid = np.abs(w) > 1e-12
    if valid.sum() < 4:
        return None
    points_3d = (homogeneous[:3, valid] / w[valid]).T
    board_points = observation.object_points.reshape(-1, 3)[valid]
    return points_3d, board_points
```

Then replace the body of `triangulation_closure`'s loop (the block from
`for observation in observations:` through `per_view.append(...)`) with:

```python
    for observation in observations:
        result = _triangulate_observation(
            observation, projection_a, projection_b,
            K_a, extrinsics.distortion_a, K_b, extrinsics.distortion_b,
        )
        if result is None:
            continue
        points_3d, board_points = result

        R, t, scale = _fit_rigid(board_points, points_3d)
        fitted = board_points @ R.T + t
        error = np.linalg.norm(points_3d - fitted, axis=1)
        residuals.append(error)
        per_view.append(
            {
                "label": observation.label,
                "depth_m": float(points_3d[:, 2].mean()),
                "rms_mm": float(np.sqrt(np.mean(error ** 2)) * 1000.0),
                "max_mm": float(error.max() * 1000.0),
                "scale": scale,
                "points": int(len(points_3d)),
            }
        )
```

Everything else in `triangulation_closure` (the docstring, the
`projection_a`/`projection_b` setup above the loop, and the return statement
below it) stays unchanged. This is behavior-preserving: `len(points_3d) ==
valid.sum()` from the old code, so `"points"` is identical.

- [ ] **Step 4: Add `adjacent_corner_distance_errors`**

Insert this function directly after `triangulation_closure` (before the
"Naming and removing the views that spoil a set" section comment):

```python
def adjacent_corner_distance_errors(
    extrinsics: StereoExtrinsics,
    observations: Sequence[StereoObservation],
    board: TargetBoard,
    tolerance_fraction: float = 0.1,
) -> Dict[str, Any]:
    """Triangulated corner-to-corner distance vs. the board's printed square size.

    :func:`triangulation_closure` fits a rigid transform, plus a free scale,
    onto the known board and reports what is left over — a systematic scale
    error is exactly what that fitted scale is built to absorb, so it barely
    moves the residual. This instead compares the triangulated distance
    between every pair of corners one square apart directly against
    ``board.square_size_m``, with nothing fitted that could hide a scale bias.

    Run on views that were never part of the stereo fit (see
    ``cross_validate_stereo.py``), this is a genuine out-of-sample check of
    the metric scale baked into ``extrinsics.T`` — the one thing in-sample
    reprojection error cannot see, because it is graded on the same poses
    that set the scale in the first place.

    ``tolerance_fraction`` selects pairs whose *known* board-space distance is
    within that fraction of one square size; a diagonal pair differs by ~41%
    and a two-square pair by 100%, so the default of 0.1 cleanly picks out
    only true grid-adjacent (horizontal or vertical) pairs.
    """
    K_a, K_b = extrinsics.camera_matrix_a, extrinsics.camera_matrix_b
    projection_a = K_a @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_b = K_b @ np.hstack([extrinsics.R, extrinsics.T.reshape(3, 1)])
    tolerance_m = tolerance_fraction * board.square_size_m

    per_view: List[Dict[str, Any]] = []
    all_errors_mm: List[np.ndarray] = []

    for observation in observations:
        result = _triangulate_observation(
            observation, projection_a, projection_b,
            K_a, extrinsics.distortion_a, K_b, extrinsics.distortion_b,
        )
        if result is None:
            continue
        points_3d, board_points = result
        if len(points_3d) < 2:
            continue

        known = np.linalg.norm(
            board_points[:, None, :] - board_points[None, :, :], axis=-1
        )
        measured = np.linalg.norm(
            points_3d[:, None, :] - points_3d[None, :, :], axis=-1
        )
        rows, cols = np.triu_indices(len(points_3d), k=1)
        known_pairs = known[rows, cols]
        measured_pairs = measured[rows, cols]

        adjacent = np.abs(known_pairs - board.square_size_m) < tolerance_m
        if not adjacent.any():
            continue

        errors_mm = (measured_pairs[adjacent] - known_pairs[adjacent]) * 1000.0
        all_errors_mm.append(errors_mm)
        per_view.append(
            {
                "label": observation.label,
                "pairs": int(adjacent.sum()),
                "mean_error_mm": float(errors_mm.mean()),
                "rms_error_mm": float(np.sqrt(np.mean(errors_mm ** 2))),
                "max_abs_error_mm": float(np.abs(errors_mm).max()),
            }
        )

    all_errors = np.concatenate(all_errors_mm) if all_errors_mm else np.zeros(0)
    square_mm = board.square_size_m * 1000.0
    return {
        "square_size_mm": square_mm,
        "tolerance_mm": tolerance_m * 1000.0,
        "per_view": per_view,
        "pairs_total": int(all_errors.size),
        "mean_error_mm": float(all_errors.mean()) if all_errors.size else float("nan"),
        "rms_error_mm": float(np.sqrt(np.mean(all_errors ** 2))) if all_errors.size else float("nan"),
        "max_abs_error_mm": float(np.abs(all_errors).max()) if all_errors.size else float("nan"),
        "std_error_mm": float(all_errors.std(ddof=1)) if all_errors.size > 1 else 0.0,
        "mean_relative_error_pct": (
            float(all_errors.mean() / square_mm * 100.0) if all_errors.size else float("nan")
        ),
        "rms_relative_error_pct": (
            float(np.sqrt(np.mean(all_errors ** 2)) / square_mm * 100.0)
            if all_errors.size else float("nan")
        ),
    }
```

`TargetBoard` is already imported at the top of `calibration/stereo.py`
(`from calibration.target_board import TargetBoard`), so no new import is
needed there.

- [ ] **Step 5: Verify against real, already-captured data (read-only)**

This exercises both the refactor and the new function against the actual
`captures/cross-validation` set and the actual frozen
`calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json` — safe,
because `StereoExtrinsics.load_json` and detection are pure reads; nothing
here calls `calibrate_stereo` (the function that fits/rewrites extrinsics).

```bash
python -c "
from pathlib import Path
from calibration.stereo import (
    StereoExtrinsics, collect_session_pairs, detect_stereo_observations,
    triangulation_closure, adjacent_corner_distance_errors,
)
from calibration.target_board import TargetBoard

extrinsics = StereoExtrinsics.load_json(
    'calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json'
)
board = TargetBoard.from_dict(extrinsics.board)
pairs, notes = collect_session_pairs(
    [Path('captures/cross-validation')], 'rgb_cam1', 'rgb_cam2'
)
observations, size_a, size_b, skipped = detect_stereo_observations(pairs, board)
print(f'{len(observations)}/{len(pairs)} held-out sessions usable')

closure = triangulation_closure(extrinsics, observations)
print('closure rms_mm', closure['rms_mm'], 'scale_mean', closure['scale_mean'])

errors = adjacent_corner_distance_errors(extrinsics, observations, board)
print('pairs_total', errors['pairs_total'])
print('mean/rms mm', errors['mean_error_mm'], errors['rms_error_mm'])
print('rms relative pct', errors['rms_relative_error_pct'])
assert errors['pairs_total'] > 0
print('OK')
"
```
Expected: no traceback, `pairs_total > 0`, finite mean/rms numbers printed,
ends with `OK`. Note whatever numbers print — they inform Task 2's report
wording but are not a pass/fail gate here (this is a diagnostic, not a unit
test with a known answer).

---

### Task 2: `cross_validate_stereo.py` CLI + `config.yaml` key

**Files:**
- Create: `cross_validate_stereo.py`
- Modify: `config.yaml`

**Interfaces:**
- Consumes: `calibrate_cameras.load_config`, `calibrate_cameras.resolve_path`
  (already used the same way by `stereo_calibrate.py`); `calibration.stereo.{StereoExtrinsics,
  collect_session_pairs, detect_stereo_observations, epipolar_residuals,
  triangulation_closure, adjacent_corner_distance_errors, DEFAULT_MIN_SHARED_CORNERS}`;
  `calibration.opencv_calibrate.DEFAULT_MIN_CORNERS`; `calibration.target_board.TargetBoard`.
- Produces: `calibration/results/stereo_<a>_<b>/cross_validation/{report.txt,cross_validation.json,corner_errors.png}`.

- [ ] **Step 1: Add the config key**

In `config.yaml`, inside the `geometric_calibration:` block, right after the
`stereo_captures:` entry (currently ending at `- captures/stereo`, line 68)
and before `extrinsics_rgb_cam1_rgb_cam2:` (line 69), insert:

```yaml
  # Held-out second checkerboard pose set for cross_validate_stereo.py. Never
  # used to fit extrinsics, only to check them: a stereo fit is graded by
  # stereo_calibrate.py against the very poses that set its scale, so it can
  # look perfect there and still be wrong. Triangulating this separate set and
  # comparing recovered corner-to-corner distances against the board's known
  # square size catches that in a way in-sample reprojection error cannot.
  cross_validation_captures: captures/cross-validation
```

- [ ] **Step 2: Write `cross_validate_stereo.py`**

```python
#!/usr/bin/env python3
"""Check a fitted stereo pair against a held-out checkerboard pose set.

``stereo_calibrate.py`` grades its own fit on the poses it was fit from, so a
stereo calibration that happens to fit that particular set of board poses
well looks perfect in its own report even when it does not generalise — the
classic overfitting blind spot that in-sample reprojection error cannot see.
This script reuses a *second*, independently captured set of checkerboard
poses (default ``captures/cross-validation``) that never went into the fit,
and checks the frozen extrinsics against it: it triangulates every shared
corner and compares the reconstructed corner-to-corner distances against the
board's printed square size (nothing fitted here to hide a scale bias behind),
and it reports the epipolar and triangulation-closure residuals on these
unseen views for comparison with the numbers in
``calibration/results/stereo_<a>_<b>/report.txt``.

This script is read-only with respect to every existing calibration
artifact — it never re-fits or rewrites ``extrinsics.json``,
``intrinsics.json``, or anything else already under
``calibration/results/<pair>/``. Results land in a new ``cross_validation/``
subfolder next to — never over — the calibration they check.

    python cross_validate_stereo.py
    python cross_validate_stereo.py --in captures/cross-validation --camera-a rgb_cam1 --camera-b rgb_cam2
    python cross_validate_stereo.py --extrinsics calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from calibrate_cameras import load_config, resolve_path  # noqa: E402
from calibration.opencv_calibrate import DEFAULT_MIN_CORNERS  # noqa: E402
from calibration.stereo import (  # noqa: E402
    DEFAULT_MIN_SHARED_CORNERS,
    StereoExtrinsics,
    adjacent_corner_distance_errors,
    collect_session_pairs,
    detect_stereo_observations,
    epipolar_residuals,
    triangulation_closure,
)
from calibration.target_board import TargetBoard  # noqa: E402

DEFAULT_OUTPUT_DIR = "calibration/results"
DEFAULT_CROSS_VALIDATION_CAPTURES = "captures/cross-validation"
# A calibration whose out-of-sample corner-distance error exceeds this is
# reproducing the training board's geometry rather than the true metric
# scale; see adjacent_corner_distance_errors in calibration/stereo.py.
WARN_RELATIVE_ERROR_PCT = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a fitted stereo pair against a held-out checkerboard pose set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Read-only: never re-fits or rewrites extrinsics.json / intrinsics.json.\n"
               "Results land in calibration/results/stereo_<a>_<b>/cross_validation/.",
    )
    parser.add_argument(
        "--in", "--captures", dest="captures", nargs="+", default=None,
        help="Directories holding held-out capture session subfolders "
             "(default: geometric_calibration.cross_validation_captures, else "
             f"{DEFAULT_CROSS_VALIDATION_CAPTURES}).",
    )
    parser.add_argument("--camera-a", default="rgb_cam1",
                        help="Reference camera (default: %(default)s).")
    parser.add_argument("--camera-b", default="rgb_cam2",
                        help="Second camera of the pair (default: %(default)s).")
    parser.add_argument(
        "--extrinsics", default=None,
        help="Extrinsics JSON to validate (default: geometric_calibration."
             "extrinsics_<a>_<b>, else calibration/results/stereo_<a>_<b>/extrinsics.json). "
             "Always read-only.",
    )
    parser.add_argument(
        "--out", "--output", dest="output", default=None,
        help="Calibration results root (default: geometric_calibration.output_dir, else "
             f"{DEFAULT_OUTPUT_DIR}). Output always lands in "
             "<out>/stereo_<a>_<b>/cross_validation/, so this can never collide with "
             "extrinsics.json.",
    )
    parser.add_argument("--min-corners", type=int, default=DEFAULT_MIN_CORNERS,
                        help="Minimum corners per image for a view to be considered "
                             "(default: %(default)s).")
    parser.add_argument("--min-shared-corners", type=int, default=DEFAULT_MIN_SHARED_CORNERS,
                        help="Minimum corners seen by both cameras (default: %(default)s).")
    parser.add_argument("--tolerance-fraction", type=float, default=0.1,
                        help="How close (as a fraction of the square size) a pair's known "
                             "distance must be to one square to count as adjacent "
                             "(default: %(default)s).")
    parser.add_argument("--no-figure", action="store_true", help="Skip the error figure.")
    return parser.parse_args()


def load_extrinsics(path: Path) -> StereoExtrinsics:
    if not path.exists():
        raise SystemExit(
            f"No extrinsics at {path}.\n"
            "Run stereo_calibrate.py first — this script only validates an existing fit, "
            "it does not create one."
        )
    return StereoExtrinsics.load_json(path)


def board_from_extrinsics(extrinsics: StereoExtrinsics, extrinsics_path: Path) -> TargetBoard:
    if not extrinsics.board:
        raise SystemExit(
            f"{extrinsics_path} carries no board info; re-run stereo_calibrate.py "
            "(current version always records it) before cross-validating."
        )
    return TargetBoard.from_dict(extrinsics.board)


def write_report(
    pair_name: str,
    extrinsics_path: Path,
    capture_dirs: List[Path],
    board: TargetBoard,
    n_sessions_total: int,
    skipped: List[Tuple[str, str]],
    notes: List[str],
    epipolar: Dict[str, Any],
    closure: Dict[str, Any],
    corner_errors: Dict[str, Any],
) -> str:
    lines: List[str] = []
    add = lines.append
    rule = "-" * 74

    add(f"Cross-validation: {pair_name}")
    add("=" * 74)
    add("")
    add(f"  calibration checked:  {extrinsics_path}  (read-only, not modified)")
    for capture_dir in capture_dirs:
        add(f"  held-out captures:    {capture_dir}")
    add(f"  sessions usable:      {len(corner_errors['per_view'])}/{n_sessions_total}")
    add(f"  board:                {board.squares_x}x{board.squares_y} ChArUco, "
        f"square {board.square_size_m * 1000:.2f} mm")
    add("")

    add("EPIPOLAR RESIDUAL   (out-of-sample; compare with the fit's own report.txt)")
    add(rule)
    add(f"  rms / p95 / max       {epipolar['rms_px']:.4f} / {epipolar['p95_px']:.3f} / "
        f"{epipolar['max_px']:.3f} px")
    add("")

    add("TRIANGULATION CLOSURE   (rigid fit + free scale onto the known board)")
    add(rule)
    add(f"  rms / max             {closure['rms_mm']:.3f} / {closure['max_mm']:.3f} mm")
    add(f"  scale                 {closure['scale_mean']:.5f} +/- {closure['scale_std']:.5f}"
        "   (1.0 = consistent)")
    add("  On the calibration set this scale is close to 1 almost by construction. On this")
    add("  held-out set it is a genuine check — though the fitted scale here can still")
    add("  absorb a global bias; the next section is the number that cannot.")
    add("")

    add(f"ADJACENT-CORNER DISTANCE vs. KNOWN SQUARE SIZE ({corner_errors['square_size_mm']:.2f} mm)")
    add(rule)
    add(f"  pairs compared        {corner_errors['pairs_total']}")
    add(f"  mean / rms error      {corner_errors['mean_error_mm']:.4f} / "
        f"{corner_errors['rms_error_mm']:.4f} mm  "
        f"({corner_errors['mean_relative_error_pct']:.3f} / "
        f"{corner_errors['rms_relative_error_pct']:.3f} %)")
    add(f"  max abs / std         {corner_errors['max_abs_error_mm']:.4f} / "
        f"{corner_errors['std_error_mm']:.4f} mm")
    add("")
    add("  Nothing here is fitted, so a stereo calibration that reproduces the training")
    add("  board's geometry without the true metric scale shows up here even though its")
    add("  own in-sample reprojection error looked fine.")
    add("")

    if corner_errors["per_view"]:
        add("Per-session breakdown")
        add(rule)
        add(f"  {'session':<24}{'pairs':>7}{'mean mm':>10}{'rms mm':>9}{'max mm':>9}")
        for view in sorted(
            corner_errors["per_view"], key=lambda v: v["rms_error_mm"], reverse=True
        ):
            add(f"  {view['label']:<24}{view['pairs']:>7}{view['mean_error_mm']:>10.4f}"
                f"{view['rms_error_mm']:>9.4f}{view['max_abs_error_mm']:>9.4f}")
        add("")

    if skipped:
        add("Sessions without a usable stereo view")
        add(rule)
        for label, reason in skipped:
            add(f"  {label:<24} {reason}")
        add("")

    if notes:
        add("Capture notes")
        add(rule)
        for note in notes:
            add(f"  {note}")
        add("")

    if (
        corner_errors["pairs_total"] > 0
        and abs(corner_errors["rms_relative_error_pct"]) > WARN_RELATIVE_ERROR_PCT
    ):
        add("Quality warning")
        add(rule)
        add(f"  RMS relative error is {corner_errors['rms_relative_error_pct']:.2f} %, above the "
            f"{WARN_RELATIVE_ERROR_PCT:.1f} % that a 1 % error in the printed square implies for "
            "the baseline (see the calibration/stereo.py module docstring). The fit may be "
            "overfitting stereo_captures rather than measuring the true rig geometry.")
        add("")

    return "\n".join(lines) + "\n"


def figure_corner_errors(corner_errors: Dict[str, Any], path: Path) -> None:
    per_view = corner_errors["per_view"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    axis = axes[0]
    means = [view["mean_error_mm"] for view in per_view]
    axis.axvline(0.0, color="k", lw=1.0)
    axis.hist(means, bins=min(20, max(5, len(means))), color="#2563eb", alpha=0.8)
    axis.set_xlabel("per-session mean corner-distance error [mm]")
    axis.set_ylabel("sessions")
    axis.set_title("Distribution across held-out sessions")
    axis.grid(alpha=0.3, axis="y")

    axis = axes[1]
    labels = [view["label"][-6:] for view in per_view]
    rms = [view["rms_error_mm"] for view in per_view]
    axis.bar(range(len(labels)), rms, color="#d97706")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=6)
    axis.set_ylabel("rms error [mm]")
    axis.set_title("Per-session rms corner-distance error")
    axis.grid(alpha=0.3, axis="y")

    figure.suptitle(
        f"Cross-validation: adjacent-corner distance vs "
        f"{corner_errors['square_size_mm']:.2f} mm square", fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    config = load_config().get("geometric_calibration", {}) or {}

    output_root = args.output or config.get("output_dir", DEFAULT_OUTPUT_DIR)
    pair_name = f"stereo_{args.camera_a}_{args.camera_b}"

    extrinsics_path = resolve_path(
        args.extrinsics
        or config.get(f"extrinsics_{args.camera_a}_{args.camera_b}")
        or f"{output_root}/{pair_name}/extrinsics.json"
    )
    extrinsics = load_extrinsics(extrinsics_path)
    if extrinsics.name_a != args.camera_a or extrinsics.name_b != args.camera_b:
        print(f"Note: extrinsics file names the pair {extrinsics.name_a}/{extrinsics.name_b}; "
              f"validating it as {args.camera_a}/{args.camera_b}.")

    board = board_from_extrinsics(extrinsics, extrinsics_path)

    capture_values = (
        args.captures
        or config.get("cross_validation_captures")
        or DEFAULT_CROSS_VALIDATION_CAPTURES
    )
    if isinstance(capture_values, str):
        capture_values = [capture_values]
    capture_dirs = [resolve_path(value) for value in capture_values]

    print(f"Cross-validating {args.camera_a} -> {args.camera_b} against {extrinsics_path}")
    for capture_dir in capture_dirs:
        print(f"Held-out source: {capture_dir}/*/{{{args.camera_a},{args.camera_b}}}.jpg")

    session_pairs, notes = collect_session_pairs(capture_dirs, args.camera_a, args.camera_b)
    observations, size_a, size_b, skipped = detect_stereo_observations(
        session_pairs, board,
        min_corners=args.min_corners,
        min_shared_corners=args.min_shared_corners,
    )
    print(f"Both cameras saw the board in {len(observations)}/{len(session_pairs)} held-out "
          f"sessions ({args.camera_a} at {size_a[0]}x{size_a[1]}, "
          f"{args.camera_b} at {size_b[0]}x{size_b[1]})")
    for label, reason in skipped:
        print(f"  discarded {label}: {reason}")

    if (size_a, size_b) != (extrinsics.image_size_a, extrinsics.image_size_b):
        raise SystemExit(
            f"Held-out captures are {size_a[0]}x{size_a[1]}/{size_b[0]}x{size_b[1]} but the "
            f"extrinsics were fit at {extrinsics.image_size_a[0]}x{extrinsics.image_size_a[1]}/"
            f"{extrinsics.image_size_b[0]}x{extrinsics.image_size_b[1]}. Capture the held-out "
            "set at the same resolution as the calibration."
        )
    if not observations:
        raise SystemExit(
            "No held-out session had the board visible in both cameras; nothing to validate."
        )

    epipolar = epipolar_residuals(extrinsics, observations)
    closure = triangulation_closure(extrinsics, observations)
    corner_errors = adjacent_corner_distance_errors(
        extrinsics, observations, board, tolerance_fraction=args.tolerance_fraction
    )
    if corner_errors["pairs_total"] == 0:
        print("Warning: no grid-adjacent corner pairs found in any held-out view; "
              "--min-shared-corners may be too low, or the board/resolution do not match.")

    output_dir = resolve_path(output_root) / pair_name / "cross_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_text = write_report(
        pair_name, extrinsics_path, capture_dirs, board,
        n_sessions_total=len(session_pairs), skipped=skipped, notes=notes,
        epipolar=epipolar, closure=closure, corner_errors=corner_errors,
    )
    (output_dir / "report.txt").write_text(report_text, encoding="utf-8")

    payload = {
        "camera_a": args.camera_a,
        "camera_b": args.camera_b,
        "extrinsics_path": str(extrinsics_path),
        "capture_dirs": [str(d) for d in capture_dirs],
        "board": board.to_dict(),
        "sessions_total": len(session_pairs),
        "sessions_used": len(observations),
        "skipped": skipped,
        "notes": notes,
        "epipolar": {k: v for k, v in epipolar.items() if k != "residuals_px"},
        "triangulation_closure": {k: v for k, v in closure.items() if k != "residuals_mm"},
        "adjacent_corner_distance_errors": corner_errors,
    }
    (output_dir / "cross_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    figure_path = None
    if not args.no_figure and corner_errors["per_view"]:
        figure_path = output_dir / "corner_errors.png"
        figure_corner_errors(corner_errors, figure_path)

    print("")
    print(f"  epipolar rms:                 {epipolar['rms_px']:.4f} px")
    print(f"  triangulation closure:        {closure['rms_mm']:.3f} mm rms, "
          f"scale {closure['scale_mean']:.5f}")
    print(f"  adjacent-corner distance:     {corner_errors['mean_error_mm']:+.4f} mm mean, "
          f"{corner_errors['rms_error_mm']:.4f} mm rms "
          f"({corner_errors['rms_relative_error_pct']:.3f} % of "
          f"{corner_errors['square_size_mm']:.2f} mm square)")
    if (
        corner_errors["pairs_total"] > 0
        and abs(corner_errors["rms_relative_error_pct"]) > WARN_RELATIVE_ERROR_PCT
    ):
        print(f"  WARNING: exceeds the {WARN_RELATIVE_ERROR_PCT:.1f} % noticeable-bias threshold")
    print("")
    print(f"Saved cross-validation results to {output_dir}")
    print(f"  {output_dir / 'report.txt'}")
    print(f"  {output_dir / 'cross_validation.json'}")
    if figure_path is not None:
        print(f"  {figure_path}")
    print("")
    print(f"{extrinsics_path} was only read, never written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Import-check**

```bash
python -c "import cross_validate_stereo; print('import OK')"
```
Expected: `import OK`, no traceback.

- [ ] **Step 4: Prove read-only safety, then run for real**

```bash
cd /home/cfenati/projects/MasterThesis/FullPipeline
sha256sum calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json \
          calibration/results/stereo_rgb_cam1_rgb_cam2/report.txt \
          calibration/results/stereo_rgb_cam1_rgb_cam2/rig_pose.yaml \
  > /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/79823f6d-4146-426a-b953-feea1cffa70e/scratchpad/before.sha256

python cross_validate_stereo.py

sha256sum -c /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/79823f6d-4146-426a-b953-feea1cffa70e/scratchpad/before.sha256
```
Expected: the script prints a normal run (session counts, epipolar/closure/
corner-distance summary, the "was only read, never written" line, and writes
under `calibration/results/stereo_rgb_cam1_rgb_cam2/cross_validation/`), and
the final `sha256sum -c` reports all three existing files as `OK` — proving
`extrinsics.json` and its siblings are byte-for-byte unchanged.

- [ ] **Step 5: Inspect the output**

```bash
ls -la calibration/results/stereo_rgb_cam1_rgb_cam2/cross_validation/
cat calibration/results/stereo_rgb_cam1_rgb_cam2/cross_validation/report.txt
```
Expected: `report.txt`, `cross_validation.json`, `corner_errors.png` present;
report reads coherently with real numbers (no `nan` unless a session
genuinely produced none, no traceback).

---

### Task 3: README documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a stage-table row**

After the `Extrinsics` row (line 15: `| Extrinsics ... stereo_calibrate.py ... |`)
and before `Rig eval / optimize` (line 16), insert:

```markdown
| Cross-validation    | `cross_validate_stereo.py` | `calibration/results/stereo_<a>_<b>/cross_validation/` |
```

- [ ] **Step 2: Document it in the Calibration section**

In the `## Calibration` section, after the existing paragraph that ends
`Stereo holds intrinsics fixed and writes ``extrinsics.json``, figures, and
measured poses into ``rig_as_built.yaml``.` (around line 200), add:

```markdown
```bash
python cross_validate_stereo.py --camera-a rgb_cam1 --camera-b rgb_cam2
```

Checks the fitted extrinsics against a second, independently captured
checkerboard pose set (`captures/cross-validation` by default) instead of the
set they were fit from. Triangulates held-out corners through the frozen
`R`/`T`/intrinsics and compares the reconstructed corner-to-corner distances
to the board's known square size — catching overfitting to `stereo_captures`
that in-sample reprojection/epipolar error can't see, since that error is
graded on the very poses that set the scale. Read-only: never rewrites
`extrinsics.json`; results land in a separate `cross_validation/` subfolder.
```

- [ ] **Step 3: Update the config-highlights table**

Change the existing `geometric_calibration.*` row (line 233) from:
```markdown
| `geometric_calibration.*` | Board, results dir, stereo capture dirs |
```
to:
```markdown
| `geometric_calibration.*` | Board, results dir, stereo + cross-validation capture dirs |
```

- [ ] **Step 4: Verify**

```bash
git diff README.md
```
Expected: the diff shows exactly the stage-table row, the new paragraph +
command block, and the one-word table-cell change above — read it over for
typos and that it renders as valid Markdown (no broken table pipes).

---

## Self-Review

- **Spec coverage:** held-out second pose set (Task 2, `captures/cross-validation`
  default) — ✓; triangulate corners and compare to known square size (Task 1
  Step 4 + Task 2's `adjacent_corner_distance_errors` call) — ✓; save to
  another folder (Task 2, `cross_validation/` subfolder, path built in code) —
  ✓; never override existing calibration values (Global Constraints +
  Task 2 Step 4's checksum proof) — ✓.
- **`solvePnP` vs. triangulation:** the user offered either. A `solvePnP`-based
  "recovered inter-corner distance" would be circular — `solvePnP` fits a
  rigid pose from the very object-space points being compared, so the
  distances it implies match the known board by construction regardless of
  any calibration error. Triangulation from independent stereo correspondence
  is the one of the two that can actually disagree with the known geometry,
  so it's the one implemented; `adjacent_corner_distance_errors`'s docstring
  explains this.
- **Placeholder scan:** no TODOs/TBDs; every step has literal code or a
  literal shell command with an expected result.
- **Type consistency:** `adjacent_corner_distance_errors` and
  `_triangulate_observation` signatures in Task 1 match exactly how they're
  called in Task 2's `cross_validate_stereo.py` and in the refactored
  `triangulation_closure`.
