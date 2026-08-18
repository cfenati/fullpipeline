# register_features.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `register_features.py`, a new standalone entry point that
registers `rgb_cam2` onto `rgb_cam1` via sparse LightGlue feature matching and
a single fitted plane depth, per the approved design in
`docs/superpowers/specs/2026-08-18-feature-based-registration-design.md`.

**Architecture:** Reuse `register_pipeline.py`'s calibration-facing geometry
(`plane_homography`, `candidate_depths`, `singular_depth`,
`load_session_images`, `undistort_pair`, `downscale_pair`,
`compose_warped_output`) unchanged. New code is only: DISK feature
extraction, LightGlue matching, a 1-DOF (scalar depth) RANSAC-by-grid-search
+ golden-section refinement, and this method's own report/preview outputs.

**Tech Stack:** PyTorch + kornia (`kornia.feature.DISK`,
`kornia.feature.LightGlueMatcher`), OpenCV, NumPy. Python 3.9,
`from __future__ import annotations`, argparse CLI — same conventions as
every other script in this repo.

## Global Constraints

- Reuse verbatim from `register_pipeline.py` (do not reimplement): `DEFAULT_DEPTH_MIN`,
  `DEFAULT_DEPTH_MAX`, `DEFAULT_STEPS`, `DEFAULT_REGISTRATION_OUTPUT_DIR`,
  `PREVIEW_PANEL_WIDTH`, `default_extrinsics_path`, `load_session_images`,
  `undistort_pair`, `downscale_pair`, `plane_homography`, `candidate_depths`,
  `singular_depth`, `compose_warped_output`, `_labelled`, `_resize_to_width`.
  This mirrors the approved design doc's explicit instruction: "this script
  does not reimplement calibration-facing geometry."
- Config loaded once via `calibrate_cameras.load_config()` /
  `calibrate_cameras.resolve_path()`, `PROJECT_ROOT = Path(__file__).resolve().parent`
  — same pattern as every other entry-point script (see CLAUDE.md).
- `from __future__ import annotations` + type hints on every function signature.
- Argparse CLI, `--help`-documented flags, matching `register_pipeline.py`'s style.
- **No test suite or linter exists in this repo** (CLAUDE.md, confirmed
  current). Verification in this plan therefore uses (a) one-off Python
  scripts run via `python3` against real session data already in the repo
  (`captures/cross-validation/20260817_164509_218202` has
  `rgb_cam1.jpg`/`rgb_cam2.jpg`) or synthetic geometry, run once during
  development and not committed, and (b) `python -c "import register_features"`
  syntax checks — the same approach CLAUDE.md prescribes for `check_color.py`.
  Do not introduce a `tests/` directory or `pytest` dependency; that would be
  inconsistent with the rest of the repo.
- New dependencies for `requirements.txt`: `torch`, `kornia` — both already
  installed and confirmed working in the current environment (`torch
  2.8.0+cu128`, `kornia 0.8.2`, Python 3.9.19). `kornia.feature.DISK` and
  `kornia.feature.LightGlueMatcher` download pretrained weights from kornia's
  model hub on first run (needs internet once, then cached under
  `~/.cache/torch/hub/checkpoints/`) — confirmed during planning (a ~45 MB
  download of `disk_lightglue.pth`).
- **Deviation from the design doc, evidence-based:** the design doc lists
  `--downscale` default `1.0`. Measured during planning on this machine
  (14 GB RAM): full-resolution DISK extraction (4656x3496, scale 1.0) was
  **OOM-killed** (exit 137). Scale 0.5 (2328x1748) completed but used 9.4 GB
  RSS — too close to the ~8-9 GB typically free to be a safe default. Scale
  0.3 (1397x1049) completed in 1.8 s using ~4 GB RSS and, end-to-end on
  session `captures/cross-validation/20260817_164509_218202`, produced a
  clean fit: 294/841 matches survived `--min-confidence 0.2`, 25 were
  inliers at the fitted depth, median reprojection error 1.6 px after
  golden-section refinement. **`DEFAULT_DOWNSCALE = 0.3`**, not `1.0`. Users
  with more RAM or a GPU can raise `--downscale`; document the memory
  behavior in the module docstring the same way `register_pipeline.py`
  documents its own runtime characteristics.
- Output directory is the **same** `registration/results/<session>/` folder
  `register_pipeline.py` writes to — filenames disambiguate
  (`warped_features.jpg`, `preview_features.jpg`, `matches.jpg`,
  `report_features.txt`), per the design doc.

---

## File Structure

- Create: `register_features.py` — the only new file, mirroring
  `register_pipeline.py`'s single-file structure (that file is 755 lines of
  clearly separated functions; this repo's convention is one script per
  entry point, not a package split).
- Modify: `requirements.txt` — add `torch`, `kornia`.
- Modify: `README.md` — add `register_features.py` to the stage table,
  Registration section, and Layout block, mirroring how `register_pipeline.py`
  is documented there.

---

### Task 1: CLI scaffold + reused I/O

**Files:**
- Create: `register_features.py`
- Test: ad hoc script at
  `/tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task1.sh`
  (not committed)

**Interfaces:**
- Produces: `parse_args() -> argparse.Namespace`; `main() -> int` (partial —
  loads/undistorts/downscales the pair and prints diagnostics; later tasks
  extend it in place).

- [ ] **Step 1: Write the verification script**

```bash
cat > /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task1.sh <<'EOF'
set -e
python3 -c "import register_features" 2>&1
python3 register_features.py --help 2>&1
python3 register_features.py \
  --session captures/cross-validation/20260817_164509_218202 \
  --downscale 0.3 2>&1
EOF
chmod +x /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task1.sh
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `bash /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task1.sh`
Expected: `ModuleNotFoundError: No module named 'register_features'` (the file
does not exist yet).

- [ ] **Step 3: Implement `register_features.py`**

```python
#!/usr/bin/env python3
"""Register camera B onto camera A via sparse LightGlue feature matching and a
single fitted plane depth.

Separate from and does not replace register_pipeline.py's dense plane-sweep
approach -- it addresses the case where that script's dense ZNCC correlation
degenerates. On the existing captures/cross-validation and finger-closeup
sessions, register_pipeline.py only scored 27-32% of the cameras' overlap
region as confident (registration/results/*/report.txt); preview.jpg shows
salt-and-pepper noise in the depth map even over the ChArUco board itself,
which is high-texture -- not just in the low-texture background. The
extrinsics have been independently re-verified (DIYer22/calibrating) and are
trusted, so the problem is dense per-pixel patch correlation, not the
calibration. See docs/superpowers/specs/2026-08-18-feature-based-registration-design.md.

The subject in the target use case (close-range hand/finger shots) is close
enough to a single plane that one homography per capture is an acceptable
simplification. Rather than fit a free 8-DOF homography that discards the
calibration, this fits the one unknown that matters -- a scalar plane depth
d -- through register_pipeline.py's existing analytic model,
H(d) = K_b @ (R + T @ n^T / d) @ inv(K_a). plane_homography(), candidate_depths(),
singular_depth(), load_session_images(), undistort_pair(), downscale_pair(),
compose_warped_output(), and the preview-panel helpers are imported from
register_pipeline.py and reused verbatim -- this script does not reimplement
calibration-facing geometry.

Memory: DISK feature extraction on CPU is memory-hungry per pixel. Measured on
a 14 GB machine: full resolution (4656x3496, --downscale 1.0) was OOM-killed;
--downscale 0.5 (2328x1748) used 9.4 GB RSS; --downscale 0.3 (1397x1049, the
default) used ~4 GB and 1.8 s, and produced a clean fit end-to-end. Raise
--downscale only with enough free RAM (or a GPU) to back it.

Example:
    python register_features.py --session captures/hand
    python register_features.py --session captures/hand --downscale 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_cameras import load_config, resolve_path  # noqa: E402
from calibration.stereo import StereoExtrinsics  # noqa: E402
from register_pipeline import (  # noqa: E402
    DEFAULT_DEPTH_MIN,
    DEFAULT_DEPTH_MAX,
    DEFAULT_STEPS,
    DEFAULT_REGISTRATION_OUTPUT_DIR,
    PREVIEW_PANEL_WIDTH,
    _labelled,
    _resize_to_width,
    candidate_depths,
    compose_warped_output,
    default_extrinsics_path,
    downscale_pair,
    load_session_images,
    plane_homography,
    singular_depth,
    undistort_pair,
)

DEFAULT_DOWNSCALE = 0.3  # see module docstring: 1.0 OOMs on CPU, 0.3 is measured-safe
DEFAULT_MAX_KEYPOINTS = 2048
DEFAULT_MIN_MATCHES = 20
DEFAULT_MIN_CONFIDENCE = 0.2
DEFAULT_RANSAC_THRESHOLD = 3.0
MIN_INLIERS = 8  # minimum to trust a 1-DOF fit robustly, per design doc
DISK_WINDOW_SIZE = 5
DISK_SCORE_THRESHOLD = 0.0
BOUNDARY_CLIP_TOLERANCE = 1e-9
DEPTH_VS_DEFAULT_NOTE_FRACTION = 0.15


# --------------------------------------------------------------------------- #
# CLI / config
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    reg_config = load_config().get("registration", {}) or {}
    depth_range = reg_config.get("depth_range", [DEFAULT_DEPTH_MIN, DEFAULT_DEPTH_MAX])

    parser = argparse.ArgumentParser(
        description="Register camera B onto camera A via sparse LightGlue feature "
                     "matching and a single fitted plane depth.",
    )
    parser.add_argument(
        "--session", required=True,
        help="Capture session folder holding <camera-a>.jpg and <camera-b>.jpg.",
    )
    parser.add_argument("--camera-a", default="rgb_cam1",
                        help="Target frame; the output is warped into this camera's view "
                             "(default: %(default)s).")
    parser.add_argument("--camera-b", default="rgb_cam2",
                        help="Source camera, warped onto camera A (default: %(default)s).")
    parser.add_argument(
        "--extrinsics", default=None,
        help="Stereo extrinsics JSON (default: geometric_calibration.extrinsics_<a>_<b> "
             "in config, else calibration/results/stereo_<a>_<b>/extrinsics.json).",
    )
    parser.add_argument("--depth-min", type=float, default=depth_range[0],
                        help="Near edge of the depth search range, metres "
                             "(default: registration.depth_range[0] in config, "
                             f"else {DEFAULT_DEPTH_MIN}).")
    parser.add_argument("--depth-max", type=float, default=depth_range[1],
                        help="Far edge of the depth search range, metres "
                             "(default: registration.depth_range[1] in config, "
                             f"else {DEFAULT_DEPTH_MAX}).")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="Coarse grid resolution for the depth search, sampled "
                             "uniformly in inverse depth (default: %(default)s).")
    parser.add_argument("--downscale", type=float, default=DEFAULT_DOWNSCALE,
                        help="Resize factor applied to the undistorted images before "
                             "feature extraction. Default is 0.3, not 1.0 -- see module "
                             "docstring, full resolution OOMs on CPU (default: %(default)s).")
    parser.add_argument("--max-keypoints", type=int, default=DEFAULT_MAX_KEYPOINTS,
                        help="Maximum DISK keypoints extracted per image (default: %(default)s).")
    parser.add_argument("--min-matches", type=int, default=DEFAULT_MIN_MATCHES,
                        help="Minimum raw LightGlue matches required to proceed "
                             "(default: %(default)s).")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                        help="Minimum LightGlue match confidence (1 - descriptor distance) "
                             "kept before depth fitting (default: %(default)s).")
    parser.add_argument("--ransac-threshold", type=float, default=DEFAULT_RANSAC_THRESHOLD,
                        help="Inlier reprojection threshold in pixels, at the working "
                             "(possibly downscaled) resolution (default: %(default)s).")
    parser.add_argument("--out", "--output", dest="output", default=None,
                        help="Output directory (default: registration.output_dir in config, "
                             f"else {DEFAULT_REGISTRATION_OUTPUT_DIR}) / <session name>.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    if args.depth_min <= 0 or args.depth_max <= args.depth_min:
        raise SystemExit(
            f"--depth-min/--depth-max must satisfy 0 < min < max, got "
            f"{args.depth_min} / {args.depth_max}"
        )

    reg_config = load_config().get("registration", {}) or {}
    default_depth = reg_config.get("default_depth", 0.168)

    extrinsics_path = resolve_path(
        args.extrinsics or default_extrinsics_path(args.camera_a, args.camera_b)
    )
    if not extrinsics_path.exists():
        raise SystemExit(
            f"No stereo extrinsics at {extrinsics_path}.\n"
            f"Run:  python stereo_calibrate.py --camera-a {args.camera_a} "
            f"--camera-b {args.camera_b}"
        )
    extrinsics = StereoExtrinsics.load_json(extrinsics_path)

    session_dir = resolve_path(args.session)
    output_root = args.output or reg_config.get("output_dir", DEFAULT_REGISTRATION_OUTPUT_DIR)
    output_dir = resolve_path(output_root) / session_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    image_a, image_b = load_session_images(session_dir, args.camera_a, args.camera_b)
    undistorted_a, undistorted_b = undistort_pair(image_a, image_b, extrinsics)
    color_a, color_b, camera_matrix_a, camera_matrix_b = downscale_pair(
        undistorted_a, undistorted_b,
        extrinsics.camera_matrix_a, extrinsics.camera_matrix_b, args.downscale,
    )
    size_a = (color_a.shape[1], color_a.shape[0])
    size_b = (color_b.shape[1], color_b.shape[0])

    print(
        f"Registering {args.camera_b} -> {args.camera_a}: {size_a[0]}x{size_a[1]} "
        f"(downscale {args.downscale}), depth search {args.depth_min:.3f}-"
        f"{args.depth_max:.3f} m over {args.steps} steps"
    )
    print(f"Saving outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the verification script again, confirm it passes**

Run: `bash /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task1.sh`
Expected: the `--help` text prints all flags; the session run prints a
`Registering rgb_cam2 -> rgb_cam1: ...` line with size `1397x1049` (downscale
0.3 applied to the 4656x3496 source) and a `Saving outputs to ...` line, exit
code 0. No `outputs written yet — this task does not save files inside
output_dir beyond creating the directory.

- [ ] **Step 5: Commit**

```bash
git add register_features.py
git commit -m "Add register_features.py CLI scaffold reusing register_pipeline.py's I/O"
```

---

### Task 2: DISK feature extraction

**Files:**
- Modify: `register_features.py`
- Test: ad hoc script (not committed)

**Interfaces:**
- Consumes: nothing new from Task 1 besides the module's own imports.
- Produces: `load_disk(device: "torch.device") -> "kornia.feature.DISK"`;
  `extract_features(disk, image_bgr: np.ndarray, max_keypoints: int, device) -> "kornia.feature.DISKFeatures"`.
  `DISKFeatures.keypoints` is `(N, 2)` float32 `(x, y)` pixel coordinates,
  `.descriptors` is `(N, 128)` float32, `.n` is the keypoint count — this is
  kornia's own type, used as-is by Task 3.

- [ ] **Step 1: Write the verification script**

```bash
cat > /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task2.py <<'EOF'
import sys
sys.path.insert(0, ".")
import cv2
import torch
from register_features import load_disk, extract_features

device = torch.device("cpu")
disk = load_disk(device)
image = cv2.imread("captures/cross-validation/20260817_164509_218202/rgb_cam1.jpg")
image = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
features = extract_features(disk, image, max_keypoints=2048, device=device)
print("keypoints:", features.n)
assert features.n > 0
assert features.keypoints.shape == (features.n, 2)
assert features.descriptors.shape == (features.n, 128)
print("PASS")
EOF
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `python3 /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task2.py`
Expected: `ImportError: cannot import name 'load_disk' from 'register_features'`.

- [ ] **Step 3: Add the imports and functions**

Add to the top-level imports in `register_features.py` (after the existing
`import numpy as np`):

```python
import torch
import kornia.feature as KF
```

Add these functions (place after the constants block, before `parse_args`):

```python
# --------------------------------------------------------------------------- #
# Feature extraction (DISK)
# --------------------------------------------------------------------------- #
def select_device() -> "torch.device":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_disk(device: "torch.device") -> "KF.DISK":
    return KF.DISK.from_pretrained("depth").to(device).eval()


def load_matcher(device: "torch.device") -> "KF.LightGlueMatcher":
    return KF.LightGlueMatcher("disk").to(device).eval()


def image_to_tensor(image_bgr: np.ndarray, device: "torch.device") -> "torch.Tensor":
    """BGR uint8 HxWx3 -> normalized RGB float32 tensor, shape (1, 3, H, W)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0).to(device)


def extract_features(
    disk: "KF.DISK", image_bgr: np.ndarray, max_keypoints: int, device: "torch.device",
) -> "KF.DISKFeatures":
    """Detect DISK keypoints + descriptors in one image.

    pad_if_not_divisible=True handles DISK's requirement that input dimensions
    be a multiple of 16 -- undistorted/downscaled captures are not guaranteed
    to be, and this avoids a manual crop/pad step here.
    """
    tensor = image_to_tensor(image_bgr, device)
    with torch.no_grad():
        features = disk(
            tensor, n=max_keypoints, window_size=DISK_WINDOW_SIZE,
            score_threshold=DISK_SCORE_THRESHOLD, pad_if_not_divisible=True,
        )
    return features[0]
```

- [ ] **Step 4: Run the verification script again, confirm it passes**

Run: `python3 /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task2.py`
Expected: `keypoints: 2048` (or close to it) and `PASS`. First run downloads
`disk_lightglue.pth` (~45 MB) if not already cached from planning.

- [ ] **Step 5: Commit**

```bash
git add register_features.py
git commit -m "Add DISK feature extraction to register_features.py"
```

---

### Task 3: LightGlue matching

**Files:**
- Modify: `register_features.py`
- Test: ad hoc script (not committed)

**Interfaces:**
- Consumes: `extract_features()` and `DISKFeatures` from Task 2.
- Produces: `match_features(matcher, feat_a, feat_b, size_a: Tuple[int, int], size_b: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
  returning `(points_a, points_b, confidence)` as `(M, 2)`, `(M, 2)`, `(M,)`
  float64/float64/float32 NumPy arrays — `size_a`/`size_b` are `(width,
  height)` tuples, matching `register_pipeline.py`'s convention throughout.

- [ ] **Step 1: Write the verification script**

```bash
cat > /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task3.py <<'EOF'
import sys
sys.path.insert(0, ".")
import cv2
import torch
from register_features import load_disk, load_matcher, extract_features, match_features

device = torch.device("cpu")
disk = load_disk(device)
matcher = load_matcher(device)

image_a = cv2.imread("captures/cross-validation/20260817_164509_218202/rgb_cam1.jpg")
image_b = cv2.imread("captures/cross-validation/20260817_164509_218202/rgb_cam2.jpg")
image_a = cv2.resize(image_a, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
image_b = cv2.resize(image_b, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

feat_a = extract_features(disk, image_a, 2048, device)
feat_b = extract_features(disk, image_b, 2048, device)
size_a = (image_a.shape[1], image_a.shape[0])
size_b = (image_b.shape[1], image_b.shape[0])

points_a, points_b, confidence = match_features(matcher, feat_a, feat_b, size_a, size_b)
print("raw matches:", len(points_a))
print("confidence >= 0.2:", int((confidence >= 0.2).sum()))
assert points_a.shape == points_b.shape
assert points_a.shape[1] == 2
assert confidence.shape == (len(points_a),)
assert len(points_a) > 500  # measured 841 on this exact pair during planning
assert int((confidence >= 0.2).sum()) > 200  # measured 294
print("PASS")
EOF
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `python3 /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task3.py`
Expected: `ImportError: cannot import name 'match_features' from 'register_features'`.

- [ ] **Step 3: Add `match_features`**

Add after `extract_features`:

```python
# --------------------------------------------------------------------------- #
# Matching (LightGlue)
# --------------------------------------------------------------------------- #
def match_features(
    matcher: "KF.LightGlueMatcher", feat_a: "KF.DISKFeatures", feat_b: "KF.DISKFeatures",
    size_a: Tuple[int, int], size_b: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match two DISK feature sets with LightGlue.

    DISK has no scale/orientation, so laf_from_center_scale_ori's defaults
    (scale=1, orientation=0) are used -- LightGlue was trained expecting
    exactly this for DISK features (kornia's own LightGlueMatcher wires DISK
    the same way internally).

    LightGlueMatcher.forward returns a descriptor *distance*, not a
    confidence; kornia's own convention (kornia/feature/integrated.py) is
    confidence = 1 - distance, reused here for the same reason: dists are not
    guaranteed in [0, 1] against a nonsense input, but 1 - dist against a
    real DISK/LightGlue pair empirically falls in [0, 1] (confirmed against
    this rig's captures during planning: min ~5e-5, max ~0.90).

    Returns points_a, points_b as float64 (x, y) pixel arrays and confidence
    as float32, all length M (the number of raw LightGlue matches -- no
    confidence filtering here, callers filter with --min-confidence).
    """
    lafs_a = KF.laf_from_center_scale_ori(feat_a.keypoints[None])
    lafs_b = KF.laf_from_center_scale_ori(feat_b.keypoints[None])
    width_a, height_a = size_a
    width_b, height_b = size_b
    with torch.no_grad():
        dists, idxs = matcher(
            feat_a.descriptors, feat_b.descriptors, lafs_a, lafs_b,
            hw1=(height_a, width_a), hw2=(height_b, width_b),
        )
    points_a = feat_a.keypoints[idxs[:, 0]].double().numpy()
    points_b = feat_b.keypoints[idxs[:, 1]].double().numpy()
    confidence = (1.0 - dists).reshape(-1).float().numpy()
    return points_a, points_b, confidence
```

- [ ] **Step 4: Run the verification script again, confirm it passes**

Run: `python3 /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task3.py`
Expected: `raw matches: 841`, `confidence >= 0.2: 294`, `PASS` (matches the
numbers measured during planning on this exact session/downscale).

- [ ] **Step 5: Commit**

```bash
git add register_features.py
git commit -m "Add LightGlue matching to register_features.py"
```

---

### Task 4: Plane-depth fitting (coarse grid search + golden-section refine)

**Files:**
- Modify: `register_features.py`
- Test: ad hoc script (not committed) — pure-geometry synthetic test, no
  neural net involved.

**Interfaces:**
- Consumes: `points_a`, `points_b`, `confidence` from `match_features()`
  (Task 3); `plane_homography`, `candidate_depths` (reused from
  `register_pipeline`, Task 1 imports).
- Produces: `fit_plane_depth(points_a, points_b, confidence, min_confidence, camera_matrix_a, camera_matrix_b, R, T, depths, ransac_threshold) -> Dict[str, Any]`
  with keys `points_a`, `points_b` (the confidence-filtered `(M, 2)` arrays
  actually used for fitting), `coarse_depth: float`, `depth: float` (the
  refined fit), `errors: (M,) float64` (reprojection error of every filtered
  match at the refined depth), `inlier_mask: (M,) bool` (`errors <
  ransac_threshold`, evaluated at the refined depth — Task 5/6 rely on this
  exact key set).

- [ ] **Step 1: Write the verification script**

```bash
cat > /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task4.py <<'EOF'
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from register_pipeline import plane_homography, candidate_depths
from register_features import fit_plane_depth

K_a = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]])
K_b = np.array([[790.0, 0, 315.0], [0, 790.0, 235.0], [0, 0, 1.0]])
R = cv2.Rodrigues(np.array([0.02, 0.30, 0.01]))[0]
T = np.array([[0.050], [0.0], [0.0]])
true_depth = 0.16

rng = np.random.default_rng(0)
points_a = rng.uniform([50, 50], [590, 430], size=(60, 2))
K_a_inv = np.linalg.inv(K_a)
H_true = plane_homography(K_a_inv, K_b, R, T, true_depth)
homogeneous = np.concatenate([points_a, np.ones((60, 1))], axis=1)
projected = (H_true @ homogeneous.T).T
points_b = (projected[:, :2] / projected[:, 2:3]).copy()

outlier_idx = rng.choice(60, size=15, replace=False)
points_b[outlier_idx] += rng.uniform(-80, 80, size=(15, 2))

confidence = np.ones(60)
depths = candidate_depths(0.11, 0.21, 60)
fit = fit_plane_depth(points_a, points_b, confidence, 0.0, K_a, K_b, R, T, depths, ransac_threshold=3.0)

print("recovered depth:", fit["depth"], "vs true:", true_depth)
depth_error = abs(fit["depth"] - true_depth)
print("depth error (m):", depth_error)
recovered_inliers = set(np.where(fit["inlier_mask"])[0].tolist())
true_inliers = set(range(60)) - set(outlier_idx.tolist())
print("inlier set matches:", recovered_inliers == true_inliers)

assert depth_error < 1e-6, depth_error  # measured ~1.7e-10 during planning (noiseless synthetic case)
assert recovered_inliers == true_inliers
print("PASS")
EOF
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `python3 /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task4.py`
Expected: `ImportError: cannot import name 'fit_plane_depth' from 'register_features'`.

- [ ] **Step 3: Add the depth-fitting functions**

Add after `match_features`:

```python
# --------------------------------------------------------------------------- #
# Plane-depth fitting
# --------------------------------------------------------------------------- #
def inlier_reprojection_errors(
    points_a: np.ndarray, points_b: np.ndarray,
    camera_matrix_a_inv: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray, depth_m: float,
) -> np.ndarray:
    """Reprojection error in pixels of every A/B match against plane_homography(depth_m)."""
    H = plane_homography(camera_matrix_a_inv, camera_matrix_b, R, T, depth_m)
    homogeneous_a = np.concatenate([points_a, np.ones((len(points_a), 1))], axis=1)
    projected = (H @ homogeneous_a.T).T
    projected_xy = projected[:, :2] / projected[:, 2:3]
    return np.linalg.norm(projected_xy - points_b, axis=1)


def coarse_search_depth(
    points_a: np.ndarray, points_b: np.ndarray,
    camera_matrix_a_inv: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray, depths: np.ndarray, ransac_threshold: float,
) -> Tuple[int, np.ndarray]:
    """Grid search over `depths`: index with the most inlier matches.

    RANSAC over a single parameter, done by grid search since the parameter
    is 1-D and the range is already bounded by the rig's known working
    distance (design doc) -- not the O(steps*W*H) per-pixel sweep
    register_pipeline.py needs; here `depths` is swept against a few hundred
    match pairs, negligible cost per step.
    """
    best_idx, best_count, best_errors = 0, -1, None
    for idx, depth in enumerate(depths):
        errors = inlier_reprojection_errors(
            points_a, points_b, camera_matrix_a_inv, camera_matrix_b, R, T, float(depth)
        )
        count = int((errors < ransac_threshold).sum())
        if count > best_count:
            best_idx, best_count, best_errors = idx, count, errors
    return best_idx, best_errors


def refine_depth_golden_section(
    points_a: np.ndarray, points_b: np.ndarray,
    camera_matrix_a_inv: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray, depths: np.ndarray, coarse_idx: int,
    inlier_mask: np.ndarray, iterations: int = 25,
) -> float:
    """Golden-section search between the coarse winner's grid neighbors,
    minimizing median reprojection error over its inlier set -- lands the fit
    between grid steps without a second, separately-tuned optimizer."""
    lo = depths[max(0, coarse_idx - 1)]
    hi = depths[min(len(depths) - 1, coarse_idx + 1)]
    inlier_a, inlier_b = points_a[inlier_mask], points_b[inlier_mask]

    def median_error(depth: float) -> float:
        errors = inlier_reprojection_errors(
            inlier_a, inlier_b, camera_matrix_a_inv, camera_matrix_b, R, T, depth
        )
        return float(np.median(errors))

    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    for _ in range(iterations):
        if median_error(c) < median_error(d):
            b = d
        else:
            a = c
        c = b - invphi * (b - a)
        d = a + invphi * (b - a)
    return (a + b) / 2.0


def fit_plane_depth(
    points_a: np.ndarray, points_b: np.ndarray, confidence: np.ndarray,
    min_confidence: float, camera_matrix_a: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray, depths: np.ndarray, ransac_threshold: float,
) -> Dict[str, Any]:
    """Fit the single scalar plane depth that best explains the matches.

    Inliers are recomputed at the *refined* depth for the returned
    inlier_mask/errors (not the coarse grid depth the golden-section search
    started from) -- that is the fit callers actually use downstream, so it
    is what --min-matches/MIN_INLIERS gating and the report should describe.
    """
    keep = confidence >= min_confidence
    filtered_a, filtered_b = points_a[keep], points_b[keep]
    camera_matrix_a_inv = np.linalg.inv(camera_matrix_a)

    coarse_idx, coarse_errors = coarse_search_depth(
        filtered_a, filtered_b, camera_matrix_a_inv, camera_matrix_b, R, T,
        depths, ransac_threshold,
    )
    coarse_inlier_mask = coarse_errors < ransac_threshold
    refined_depth = refine_depth_golden_section(
        filtered_a, filtered_b, camera_matrix_a_inv, camera_matrix_b, R, T,
        depths, coarse_idx, coarse_inlier_mask,
    )
    final_errors = inlier_reprojection_errors(
        filtered_a, filtered_b, camera_matrix_a_inv, camera_matrix_b, R, T, refined_depth
    )
    final_inlier_mask = final_errors < ransac_threshold

    return {
        "points_a": filtered_a,
        "points_b": filtered_b,
        "coarse_depth": float(depths[coarse_idx]),
        "depth": float(refined_depth),
        "errors": final_errors,
        "inlier_mask": final_inlier_mask,
        "n_raw_matches": int(len(points_a)),
        "n_filtered_matches": int(len(filtered_a)),
    }
```

- [ ] **Step 4: Run the verification script again, confirm it passes**

Run: `python3 /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task4.py`
Expected: `depth error (m): 1.66...e-10`, `inlier set matches: True`, `PASS`.

- [ ] **Step 5: Commit**

```bash
git add register_features.py
git commit -m "Add plane-depth fitting (grid search + golden-section refine) to register_features.py"
```

---

### Task 5: Compose warp + match/preview visualizations

**Files:**
- Modify: `register_features.py`
- Test: ad hoc script (not committed), run against real session data.

**Interfaces:**
- Consumes: `fit_plane_depth()`'s return dict (Task 4); `compose_warped_output`,
  `_labelled`, `_resize_to_width`, `PREVIEW_PANEL_WIDTH` (reused from
  `register_pipeline`, Task 1 imports).
- Produces: `save_matches_visualization(path, color_a, color_b, points_a, points_b, inlier_mask) -> None`;
  `points_inlier_panel(color_a, points_a, inlier_mask) -> np.ndarray`;
  `save_preview(path, color_a, warped_color, points_panel) -> None`.

- [ ] **Step 1: Write the verification script**

```bash
cat > /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task5.py <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
import torch
import cv2
from calibration.stereo import StereoExtrinsics
from register_pipeline import load_session_images, undistort_pair, downscale_pair, compose_warped_output, candidate_depths
from register_features import (
    load_disk, load_matcher, extract_features, match_features, fit_plane_depth,
    save_matches_visualization, points_inlier_panel, save_preview,
)

out_dir = Path("/tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/task5_out")
out_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")
disk = load_disk(device)
matcher = load_matcher(device)

extrinsics = StereoExtrinsics.load_json(Path("calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json"))
session_dir = Path("captures/cross-validation/20260817_164509_218202")
image_a, image_b = load_session_images(session_dir, "rgb_cam1", "rgb_cam2")
undistorted_a, undistorted_b = undistort_pair(image_a, image_b, extrinsics)
color_a, color_b, K_a, K_b = downscale_pair(
    undistorted_a, undistorted_b, extrinsics.camera_matrix_a, extrinsics.camera_matrix_b, 0.3
)
size_a = (color_a.shape[1], color_a.shape[0])
size_b = (color_b.shape[1], color_b.shape[0])

feat_a = extract_features(disk, color_a, 2048, device)
feat_b = extract_features(disk, color_b, 2048, device)
points_a, points_b, confidence = match_features(matcher, feat_a, feat_b, size_a, size_b)
depths = candidate_depths(0.11, 0.21, 60)
fit = fit_plane_depth(points_a, points_b, confidence, 0.2, K_a, K_b, extrinsics.R, extrinsics.T, depths, 3.0)
print("fitted depth:", fit["depth"], "inliers:", int(fit["inlier_mask"].sum()), "/", fit["n_filtered_matches"])

depth_map = np.full((size_a[1], size_a[0]), fit["depth"], dtype=np.float64)
warped_color, remap_valid = compose_warped_output(
    depth_map, color_b, K_a, K_b, extrinsics.R, extrinsics.T, size_a, size_b
)
cv2.imwrite(str(out_dir / "warped_features.jpg"), warped_color)

save_matches_visualization(
    out_dir / "matches.jpg", color_a, color_b, fit["points_a"], fit["points_b"], fit["inlier_mask"]
)
panel = points_inlier_panel(color_a, fit["points_a"], fit["inlier_mask"])
save_preview(out_dir / "preview_features.jpg", color_a, warped_color, panel)

for name in ("warped_features.jpg", "matches.jpg", "preview_features.jpg"):
    path = out_dir / name
    assert path.exists() and path.stat().st_size > 0, name
print("PASS -- inspect", out_dir, "by eye")
EOF
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `python3 /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task5.py`
Expected: `ImportError: cannot import name 'save_matches_visualization' from 'register_features'`.

- [ ] **Step 3: Add the visualization functions**

Add after `fit_plane_depth`:

```python
# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def save_matches_visualization(
    path: Path, color_a: np.ndarray, color_b: np.ndarray,
    points_a: np.ndarray, points_b: np.ndarray, inlier_mask: np.ndarray,
) -> None:
    """Side-by-side A|B canvas with a line per match: green = inlier at the
    fitted depth, red = outlier -- this method's own debug output, separate
    from preview_features.jpg's same-aspect-ratio panels."""
    height = max(color_a.shape[0], color_b.shape[0])
    width_a = color_a.shape[1]
    canvas = np.zeros((height, width_a + color_b.shape[1], 3), dtype=np.uint8)
    canvas[: color_a.shape[0], :width_a] = color_a
    canvas[: color_b.shape[0], width_a:] = color_b

    for (xa, ya), (xb, yb), is_inlier in zip(points_a, points_b, inlier_mask):
        color = (0, 200, 0) if is_inlier else (0, 0, 200)
        point_a = (int(round(xa)), int(round(ya)))
        point_b = (int(round(xb)) + width_a, int(round(yb)))
        cv2.line(canvas, point_a, point_b, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, point_a, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point_b, 2, color, -1, cv2.LINE_AA)

    cv2.imwrite(str(path), _resize_to_width(canvas, PREVIEW_PANEL_WIDTH * 2))


def points_inlier_panel(
    color_a: np.ndarray, points_a: np.ndarray, inlier_mask: np.ndarray,
) -> np.ndarray:
    """Camera A with its matched keypoints marked -- same shape as color_a, so
    it can sit alongside the other preview_features.jpg panels."""
    panel = color_a.copy()
    for (x, y), is_inlier in zip(points_a, inlier_mask):
        color = (0, 200, 0) if is_inlier else (0, 0, 200)
        cv2.circle(panel, (int(round(x)), int(round(y))), 4, color, -1, cv2.LINE_AA)
    return panel


def save_preview(
    path: Path, color_a: np.ndarray, warped_color: np.ndarray, points_panel: np.ndarray,
) -> None:
    panels = [
        _resize_to_width(_labelled(color_a, "camera A"), PREVIEW_PANEL_WIDTH),
        _resize_to_width(_labelled(warped_color, "B warped onto A (single depth)"), PREVIEW_PANEL_WIDTH),
        _resize_to_width(_labelled(points_panel, "matches: inlier (green) / outlier (red)"), PREVIEW_PANEL_WIDTH),
    ]
    preview = np.hstack(panels)
    cv2.imwrite(str(path), preview)
```

- [ ] **Step 4: Run the verification script again, confirm it passes**

Run: `python3 /tmp/claude-1000/-home-cfenati-projects-MasterThesis-FullPipeline/9b380bed-4707-4fe9-98ee-86308f0c8eb2/scratchpad/verify_task5.py`
Expected: `fitted depth: 0.166...`, `inliers: 25 / 294` (matching the numbers
measured during planning), `PASS`. Then actually view
`.../scratchpad/task5_out/preview_features.jpg` and `matches.jpg` — confirm
the warped panel visibly aligns with camera A's subject and the match lines
mostly land on real correspondences (green), not scattered noise.

- [ ] **Step 5: Commit**

```bash
git add register_features.py
git commit -m "Add warp compose and match/preview visualizations to register_features.py"
```

---

### Task 6: Report, quality gates, and full `main()` wiring

**Files:**
- Modify: `register_features.py` (replace the `main()` body from Task 1 with
  the full pipeline; add `write_report` and `aggregate_quality_warnings`)
- Test: end-to-end run against all 4 existing sessions under
  `registration/results/` (real data, already in the repo).

**Interfaces:**
- Consumes: every function from Tasks 1-5.
- Produces: a complete `register_features.py` CLI.

- [ ] **Step 1: Add `aggregate_quality_warnings` and `write_report`**

Add before `main()`:

```python
def aggregate_quality_warnings(
    fit: Dict[str, Any], depths: np.ndarray, default_depth: float,
) -> List[str]:
    warnings: List[str] = []
    if (
        np.isclose(fit["depth"], depths[0], atol=BOUNDARY_CLIP_TOLERANCE)
        or np.isclose(fit["depth"], depths[-1], atol=BOUNDARY_CLIP_TOLERANCE)
    ):
        warnings.append(
            f"Fitted depth {fit['depth']:.4f} m landed on the edge of the search range "
            f"({depths[0]:.4f}-{depths[-1]:.4f} m). Widen --depth-min/--depth-max."
        )
    relative_diff = abs(fit["depth"] - default_depth) / default_depth
    if relative_diff > DEPTH_VS_DEFAULT_NOTE_FRACTION:
        warnings.append(
            f"Fitted depth {fit['depth']:.4f} m is {relative_diff * 100:.1f}% away from "
            f"registration.default_depth ({default_depth:.4f} m) -- expected if the "
            "subject really is at a different distance than the reference plane, "
            "otherwise worth a second look."
        )
    return warnings


def write_report(
    path: Path,
    session_dir: Path, camera_a: str, camera_b: str, extrinsics_path: Path,
    extrinsics: StereoExtrinsics, size_a: Tuple[int, int], downscale: float,
    depths: np.ndarray, max_keypoints: int, min_confidence: float,
    ransac_threshold: float, min_matches: int, default_depth: float, d_critical: float,
    fit: Dict[str, Any], warnings: List[str],
) -> None:
    inlier_mask = fit["inlier_mask"]
    n_inliers = int(inlier_mask.sum())
    n_filtered = fit["n_filtered_matches"]
    inlier_errors = fit["errors"][inlier_mask]

    def pct(count: int, denominator: int) -> str:
        return f"{count / denominator * 100:.1f}%" if denominator else "n/a"

    lines = [
        f"Registration: {camera_b} -> {camera_a} (LightGlue feature-based, single fitted depth)",
        "=" * 66,
        "",
        f"  session:            {session_dir}",
        f"  extrinsics:         {extrinsics_path}",
        f"  resolution:         {size_a[0]}x{size_a[1]} (downscale {downscale})",
        f"  baseline:           {extrinsics.baseline_m * 1000:.2f} mm",
        f"  convergence:        {extrinsics.optical_axis_angle_deg:.2f} deg",
        "",
        "MATCHING",
        "-" * 66,
        f"  max keypoints:       {max_keypoints}",
        f"  raw matches:         {fit['n_raw_matches']}",
        f"  min-confidence:      {min_confidence}",
        f"  matches after filter: {n_filtered}",
        "",
        "DEPTH FIT",
        "-" * 66,
        f"  search range:        {depths[0]:.4f} - {depths[-1]:.4f} m ({len(depths)} steps)",
        f"  ransac threshold:    {ransac_threshold} px",
        f"  coarse depth:        {fit['coarse_depth']:.4f} m",
        f"  refined depth:       {fit['depth']:.4f} m",
        f"  default (reference) depth: {default_depth:.4f} m",
        f"  homography singular at:  {d_critical * 1000:.2f} mm "
        f"({'well clear of' if abs(d_critical) < depths[0] / 5 else 'CHECK: close to'} "
        "the search range)",
        f"  inliers:             {n_inliers} / {n_filtered} ({pct(n_inliers, n_filtered)})",
    ]
    if n_inliers > 0:
        lines.append(
            f"  reprojection error (inliers, px): median {np.median(inlier_errors):.3f} / "
            f"mean {inlier_errors.mean():.3f} / max {inlier_errors.max():.3f}"
        )

    if warnings:
        lines += ["", "Quality warnings", "-" * 66]
        lines += [f"  - {warning}" for warning in warnings]

    lines += [
        "",
        "OUTPUT FILES",
        "-" * 66,
        "  warped_features.jpg    camera B warped into camera A's frame (single homography)",
        "  preview_features.jpg   camera A | warped B | matches, side by side",
        "  matches.jpg             inlier (green) / outlier (red) correspondence lines",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 2: Replace `main()`**

Replace the `main()` function added in Task 1 with:

```python
def main() -> int:
    args = parse_args()
    if args.depth_min <= 0 or args.depth_max <= args.depth_min:
        raise SystemExit(
            f"--depth-min/--depth-max must satisfy 0 < min < max, got "
            f"{args.depth_min} / {args.depth_max}"
        )

    reg_config = load_config().get("registration", {}) or {}
    default_depth = reg_config.get("default_depth", 0.168)

    extrinsics_path = resolve_path(
        args.extrinsics or default_extrinsics_path(args.camera_a, args.camera_b)
    )
    if not extrinsics_path.exists():
        raise SystemExit(
            f"No stereo extrinsics at {extrinsics_path}.\n"
            f"Run:  python stereo_calibrate.py --camera-a {args.camera_a} "
            f"--camera-b {args.camera_b}"
        )
    extrinsics = StereoExtrinsics.load_json(extrinsics_path)

    session_dir = resolve_path(args.session)
    output_root = args.output or reg_config.get("output_dir", DEFAULT_REGISTRATION_OUTPUT_DIR)
    output_dir = resolve_path(output_root) / session_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    image_a, image_b = load_session_images(session_dir, args.camera_a, args.camera_b)
    undistorted_a, undistorted_b = undistort_pair(image_a, image_b, extrinsics)
    color_a, color_b, camera_matrix_a, camera_matrix_b = downscale_pair(
        undistorted_a, undistorted_b,
        extrinsics.camera_matrix_a, extrinsics.camera_matrix_b, args.downscale,
    )
    size_a = (color_a.shape[1], color_a.shape[0])
    size_b = (color_b.shape[1], color_b.shape[0])

    print(
        f"Registering {args.camera_b} -> {args.camera_a}: {size_a[0]}x{size_a[1]} "
        f"(downscale {args.downscale}), depth search {args.depth_min:.3f}-"
        f"{args.depth_max:.3f} m over {args.steps} steps"
    )

    device = select_device()
    disk = load_disk(device)
    matcher = load_matcher(device)
    feat_a = extract_features(disk, color_a, args.max_keypoints, device)
    feat_b = extract_features(disk, color_b, args.max_keypoints, device)
    points_a, points_b, confidence = match_features(matcher, feat_a, feat_b, size_a, size_b)

    if len(points_a) < args.min_matches:
        raise SystemExit(
            f"Only {len(points_a)} raw LightGlue matches, need at least {args.min_matches}.\n"
            "Check --camera-a/--camera-b order, the extrinsics file, and whether the "
            "subject has enough texture; try --max-keypoints higher or --downscale higher."
        )

    depths = candidate_depths(args.depth_min, args.depth_max, args.steps)
    fit = fit_plane_depth(
        points_a, points_b, confidence, args.min_confidence,
        camera_matrix_a, camera_matrix_b, extrinsics.R, extrinsics.T,
        depths, args.ransac_threshold,
    )
    n_inliers = int(fit["inlier_mask"].sum())
    if n_inliers < MIN_INLIERS:
        raise SystemExit(
            f"Only {n_inliers} inliers at the fitted depth {fit['depth']:.4f} m, need at "
            f"least {MIN_INLIERS} to trust a 1-DOF fit.\n"
            "Try --min-confidence lower, --ransac-threshold higher, or --max-keypoints higher."
        )

    depth_map = np.full((size_a[1], size_a[0]), fit["depth"], dtype=np.float64)
    warped_color, remap_valid = compose_warped_output(
        depth_map, color_b, camera_matrix_a, camera_matrix_b,
        extrinsics.R, extrinsics.T, size_a, size_b,
    )
    d_critical = singular_depth(extrinsics.R, extrinsics.T)
    warnings = aggregate_quality_warnings(fit, depths, default_depth)

    cv2.imwrite(str(output_dir / "warped_features.jpg"), warped_color)
    save_matches_visualization(
        output_dir / "matches.jpg", color_a, color_b, fit["points_a"], fit["points_b"],
        fit["inlier_mask"],
    )
    panel = points_inlier_panel(color_a, fit["points_a"], fit["inlier_mask"])
    save_preview(output_dir / "preview_features.jpg", color_a, warped_color, panel)

    write_report(
        output_dir / "report_features.txt",
        session_dir, args.camera_a, args.camera_b, extrinsics_path, extrinsics, size_a,
        args.downscale, depths, args.max_keypoints, args.min_confidence,
        args.ransac_threshold, args.min_matches, default_depth, d_critical, fit, warnings,
    )

    inlier_pct = n_inliers / fit["n_filtered_matches"] * 100 if fit["n_filtered_matches"] else 0.0
    print(
        f"Fitted depth: {fit['depth']:.4f} m "
        f"({n_inliers}/{fit['n_filtered_matches']} inliers, {inlier_pct:.1f}%)"
    )
    if warnings:
        print("\nQuality warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print(f"\nSaved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run end-to-end against all 4 real sessions**

```bash
for session in \
  captures/cross-validation/20260817_164509_218202 \
  captures/20260817_164635_577763 \
  captures/20260817_164642_887223 \
  captures/20260818_152110_605174
do
  echo "=== $session ==="
  python3 register_features.py --session "$session"
  echo
done
```

Expected for each: exits 0, prints `Fitted depth: ... m (.../... inliers,
...%)`, and writes `registration/results/<session-name>/{warped_features.jpg,
preview_features.jpg, matches.jpg, report_features.txt}` alongside the
existing plane-sweep outputs.

- [ ] **Step 4: Verify against the design doc's own success criteria**

```bash
grep "refined depth" registration/results/*/report_features.txt
grep "min / mean / max" registration/results/*/report.txt
```

Compare: the design doc's verification plan calls for the fitted depth to be
in the same ballpark as plane-sweep's confident-pixel depth mean
(~0.148-0.153 m across these three older sessions, per the existing
`report.txt` files). Confirm the new `report_features.txt` numbers land in a
comparable range (not off by 2x or landing on a search-range edge) — if any
session's `report_features.txt` carries the "landed on the edge" warning or a
wildly different depth, investigate before considering this task done
(do not silently accept an outlier fit).

Then view at least one `preview_features.jpg` and `matches.jpg` by eye:
confirm the warped panel visibly aligns with camera A's subject and most
inlier (green) lines connect real corresponding points, not scattered noise
— the same check the plane-sweep script's own quality-warning text
recommends ("check preview.jpg by eye").

- [ ] **Step 5: Commit**

```bash
git add register_features.py
git commit -m "Wire register_features.py's full pipeline: report, quality gates, main()"
```

---

### Task 7: Dependencies and docs

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`

**Interfaces:** none (docs/config only).

- [ ] **Step 1: Add dependencies**

In `requirements.txt`, after the `PyYAML>=6.0` line and before the `# Multi-camera rig design` comment block, add a new section:

```
# Feature-based registration (register_features.py)
torch>=2.0
kornia>=0.8
```

- [ ] **Step 2: Verify**

Run: `python3 -c "import torch, kornia; print(torch.__version__, kornia.__version__)"`
Expected: prints version strings, no error (already confirmed working in
this environment during planning: `2.8.0+cu128 0.8.2`).

- [ ] **Step 3: Update README's stage table**

In `README.md`, after the `| Registration | register_pipeline.py | ... |` row
(around line 18), add:

```
| Registration (features) | `register_features.py` | LightGlue feature match + single fitted plane depth |
```

- [ ] **Step 4: Update README's Registration section**

In `README.md`, after the existing `## Registration` section's closing
paragraph (the one ending "Outputs under `registration/results/<session>/`:
... `preview.jpg`, `report.txt`.", around line 244), add:

```markdown
When plane-sweep's dense ZNCC correlation degenerates (low match confidence
across most of the frame; see `report.txt`'s "Only N% ... scored
confidently" warning), `register_features.py` is an alternative: sparse
LightGlue feature matches fit a single scalar plane depth against the same
calibrated `H(d) = K_b @ (R + T @ n^T / d) @ inv(K_a)` model, instead of a
free 8-DOF homography. Not a replacement — both scripts remain available and
write into the same output directory with disambiguated filenames.

```bash
python register_features.py --session captures/hand
python register_features.py --session captures/hand --downscale 0.5
```

Outputs under `registration/results/<session>/`: `warped_features.jpg`,
`preview_features.jpg`, `matches.jpg` (inlier/outlier correspondence lines),
`report_features.txt`.
```

- [ ] **Step 5: Update README's Layout block**

In `README.md`'s `## Layout` section (around line 266), add a line after
`register_pipeline.py`:

```
register_features.py                       LightGlue feature-based registration
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt README.md
git commit -m "Document register_features.py: requirements, README stage table and Registration section"
```

---

## Self-Review Notes

- **Spec coverage:** every section of the design doc maps to a task —
  Motivation/Approach/Algorithm → Tasks 2-4; "Why not a free homography" →
  encoded in `fit_plane_depth`'s scalar-depth design (Task 4); Dependencies →
  Task 7; CLI → Task 1 (plus `--max-keypoints`, a necessary addition the
  design doc's CLI list omitted since `DISK.forward`'s `n` parameter has no
  other source); Outputs → Tasks 5-6; Error handling/quality warnings → Task
  6; Verification plan → Task 6 Steps 3-4 run the exact 3 sessions the design
  doc names plus the 4th session already present in the repo, and cross-check
  fitted depth against plane-sweep's own reported depths.
- **Deviations from the design doc, both evidence-based and disclosed above
  in Global Constraints:** `--downscale` default `0.3` not `1.0` (measured
  OOM at 1.0 on this machine); `--max-keypoints` added (implementation
  necessity, not a scope change).
- **Type consistency:** `fit_plane_depth`'s return dict keys
  (`points_a`, `points_b`, `coarse_depth`, `depth`, `errors`, `inlier_mask`,
  `n_raw_matches`, `n_filtered_matches`) are used identically in Task 5
  (`save_matches_visualization`, `points_inlier_panel`, `compose_warped_output`
  call) and Task 6 (`write_report`, `aggregate_quality_warnings`, the
  `main()` inlier/gate checks) — checked against each other while writing.
  `size_a`/`size_b` are `(width, height)` everywhere, matching
  `register_pipeline.py`'s own convention throughout.
