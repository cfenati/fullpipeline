# Feature-Based Registration (`register_features.py`) Implementation Plan

> **Superseded.** This documents the plane-fit design: a single fitted plane depth, reusing `register_pipeline.py`'s `plane_homography`, `candidate_depths` and `singular_depth`. `register_pipeline.py` was deleted in `ff8c992`, and `register_features.py` now triangulates every match with no assumed plane depth (`triangulate.py`; see the `register_features.py` docstring and the README "Registration" section). Kept as a design record only.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `register_features.py`, a sibling to `register_pipeline.py` that
registers `rgb_cam2` onto `rgb_cam1` using LightGlue sparse feature matching
and a single depth fit through the verified calibration, instead of dense
plane-sweep ZNCC correlation (which only scored 27-32% of the frame
confident on real captures and produced visibly noisy warps).

**Architecture:** LightGlue (kornia's `DISK` extractor + `LightGlueMatcher`)
finds sparse correspondences between the undistorted pair. Those
correspondences are reduced to a single scalar unknown — plane depth `d` —
fit through the *existing*, unmodified `register_pipeline.py` calibration
math (`candidate_depths`, `compose_warped_output`), via a coarse
inlier-counting grid search (1-D RANSAC) followed by golden-section
refinement. The one new formula this script introduces
(`project_points_a_to_b`) mirrors `register_pipeline.reproject_a_to_b`'s
per-point math exactly, but for a sparse `(N, 2)` point array — that function
can't be reused directly because it hardcodes a reshape to a dense
`(height_a, width_a)` raster.

**Tech Stack:** Python 3.9, OpenCV (already a dependency), NumPy (already a
dependency), PyTorch + kornia (**new** dependencies — see Global Constraints).

## Global Constraints

- **No test suite exists in this repo** (confirmed in `CLAUDE.md`). Every
  task below verifies with real, already-captured images
  (`captures/20260818_152110_605174/rgb_cam1.jpg` /
  `rgb_cam2.jpg` and `captures/cross-validation/20260817_164509_218202/...`
  both already exist on disk) via `python -c` snippets, the same pattern
  `docs/superpowers/plans/2026-08-17-stereo-cross-validation.md` used — not
  `pytest`. Task 2's synthetic-geometry check is the one exception: it needs
  no images at all, only NumPy, so it runs first and fastest.
- **Validated kornia API — use these exact calls, not alternatives found in
  older kornia docs/tutorials online:**
  - `kornia.feature.DISK.from_pretrained("depth")` — loads the pretrained
    extractor (auto-downloads weights on first use; needs internet once).
  - `disk(tensor, n=N, pad_if_not_divisible=True)` where `tensor` is
    `(1, 3, H, W)` float in `[0, 1]`, RGB channel order — returns a
    length-1 list of `DISKFeatures`, each with `.keypoints` `(N, 2)` in
    `(x, y)` pixel order, `.descriptors` `(N, D)`.
  - `kornia.feature.LightGlueMatcher("disk")` — the matcher. Feed it LAFs
    built with `kornia.feature.laf_from_center_scale_ori(keypoints[None])`
    (no scale/orientation info needed for DISK keypoints; defaults to
    scale=1, ori=0).
  - `matcher(desc_a, desc_b, lafs_a, lafs_b, hw1=(H, W), hw2=(H, W))` returns
    `(scores, matches)`: `scores` is `(M, 1)`, **higher is better** (a
    confidence in roughly `[0.1, 1.0]` on real data, not a distance despite
    the docstring's generic wording); `matches` is `(M, 2)` int64, column 0
    indexes into `desc_a`/`feats_a.keypoints`, column 1 into
    `desc_b`/`feats_b.keypoints`.
  - This was run end-to-end against real rig captures during design research
    (962-1301 raw matches, 172 inliers on the finger-closeup session,
    fitted depth 0.1745 m against a calibrated default of 0.168 m) —
    numbers below reflect that, not estimates.
- **`torch` is already installed in this dev environment** (`2.8.0+cu128`)
  and `kornia==0.8.2` was installed here during design research, so imports
  already succeed on this machine. Task 1 still must add pinned entries to
  `requirements.txt` for reproducibility elsewhere (e.g. the actual capture
  rig), and its verification step doubles as confirming nothing here was a
  fluke of this machine's pre-existing install.
- **Disk space note for other machines:** default `pip install torch` on
  Linux pulls the CUDA build (~2GB with bundled `nvidia-*` packages). If a
  fresh install machine is disk-constrained, use
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  instead (CPU-only, ~200MB) — this script never touches CUDA. Note this in
  `requirements.txt` as a comment; don't hardcode the CPU index into the
  file itself (that would force CPU-only even on machines that want CUDA
  elsewhere).
- **Reuse, don't reimplement, calibration math.** `plane_homography`,
  `candidate_depths`, `singular_depth`, `load_session_images`,
  `undistort_pair`, `downscale_pair`, `compose_warped_output`,
  `default_extrinsics_path`, `PREVIEW_PANEL_WIDTH`,
  `DEFAULT_REGISTRATION_OUTPUT_DIR`, `DEFAULT_DEPTH_MIN`, `DEFAULT_DEPTH_MAX`,
  `DEFAULT_STEPS` are imported from `register_pipeline.py`, never
  copy-pasted. `register_pipeline.py` itself is not modified by this plan.
- **Commit narrowly, per task.** The working tree already has substantial
  unrelated pre-existing uncommitted work (`cameras/`, `capture_pipeline.py`,
  `design/`, `config.yaml`, etc. — visible in `git status`). Each task commits
  *only* the files its own task touches (`git add register_features.py
  requirements.txt` etc., never `git add -A` / `git add .`). That unrelated
  work must stay untouched and still uncommitted throughout.
- **Follow existing conventions:** `from __future__ import annotations`, type
  hints on function signatures, `argparse` CLI with `config.yaml` defaults
  resolved via `calibrate_cameras.load_config`/`resolve_path`, docstrings
  that explain *why* the way `register_pipeline.py`'s do — this file sits
  right next to it and should read like it belongs there.
- **Full design context:** `docs/superpowers/specs/2026-08-18-feature-based-registration-design.md`.

---

### Task 1: Add PyTorch + kornia dependency

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `torch`, `kornia` importable from any script in this repo.

- [ ] **Step 1: Add the dependency lines**

Append to `requirements.txt` (after the existing `PyQt5>=5.15` line):

```
# LightGlue feature matching (register_features.py)
torch>=2.0            # CPU-only build recommended on disk-constrained machines:
                       #   pip install torch --index-url https://download.pytorch.org/whl/cpu
kornia>=0.8            # provides both the DISK extractor and LightGlueMatcher
```

- [ ] **Step 2: Verify import**

```bash
python3 -c "
import torch, kornia
print('torch', torch.__version__)
print('kornia', kornia.__version__)
import kornia.feature as KF
assert hasattr(KF, 'DISK') and hasattr(KF, 'LightGlueMatcher')
print('kornia.feature.DISK / LightGlueMatcher OK')
"
```
Expected: prints both version strings and `kornia.feature.DISK /
LightGlueMatcher OK`, no traceback. (Already confirmed working on this
machine — `torch 2.8.0+cu128`, `kornia 0.8.2` — this step just re-verifies
after the `requirements.txt` edit.)

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "$(cat <<'EOF'
Add torch/kornia dependency for LightGlue-based registration

register_features.py (next commits) needs kornia's DISK extractor and
LightGlueMatcher for sparse feature matching.
EOF
)"
```

---

### Task 2: Depth-fitting geometry core, with synthetic verification

**Files:**
- Create: `register_features.py`

**Interfaces:**
- Produces: `register_features.golden_section_minimize(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-5, max_iter: int = 100) -> float`
- Produces: `register_features.project_points_a_to_b(depth: float, points_a: np.ndarray, camera_matrix_a_inv: np.ndarray, camera_matrix_b: np.ndarray, R: np.ndarray, T: np.ndarray, size_b: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]` — returns `(predicted_b (N,2), valid (N,) bool)`.
- Produces: `register_features.fit_depth(pts_a: np.ndarray, pts_b: np.ndarray, camera_matrix_a_inv: np.ndarray, camera_matrix_b: np.ndarray, R: np.ndarray, T: np.ndarray, size_b: Tuple[int, int], depths: np.ndarray, ransac_threshold: float) -> Tuple[float, float, np.ndarray, np.ndarray]` — returns `(refined_depth, coarse_depth, inliers (N,) bool, errors (N,) float)`.
- Consumes: nothing from other tasks (pure NumPy). `depths` is expected to come from `register_pipeline.candidate_depths` (wired up in Task 5), but this task's own verification builds it the same way.

- [ ] **Step 1: Create the file**

```python
#!/usr/bin/env python3
"""Register camera B onto camera A via LightGlue sparse matching and a single
fitted plane depth.

Sibling to register_pipeline.py, not a replacement: that script's dense
plane-sweep ZNCC correlation only scored 27-32% of the cameras' overlap
region confident on real captures (see registration/results/*/report.txt),
falling back to a default depth almost everywhere and producing visibly
noisy warps. This script targets that failure case for subjects close
enough to a single plane (the scoping decision recorded in
docs/superpowers/specs/2026-08-18-feature-based-registration-design.md):
find sparse correspondences with LightGlue, then fit the *one* unknown that
matters -- a scalar plane depth -- through register_pipeline.py's own
calibration math, rather than a free 8-DOF homography that would discard
the verified calibration.

Algorithm:
    1. Undistort + optionally downscale the pair (register_pipeline.py's
       own load_session_images/undistort_pair/downscale_pair).
    2. LightGlue match (kornia DISK + LightGlueMatcher) -> sparse (pts_a,
       pts_b) correspondences.
    3. Fit depth d: coarse grid search over candidate_depths(), scoring
       each hypothesis by inlier count under a pixel-reprojection
       threshold (this is exactly RANSAC over one parameter) -- then
       golden-section refinement over the full search range, minimizing
       median reprojection error over the coarse inlier set.
    4. Compose the final dense warp with register_pipeline.compose_warped_output,
       passing the single fitted depth directly (it accepts a scalar or a
       dense (H, W) array; register_pipeline.py itself only ever uses the
       dense form).

Example:
    python register_features.py --session captures/hand
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Depth fit
# --------------------------------------------------------------------------- #
GOLDEN_RATIO = (np.sqrt(5.0) - 1.0) / 2.0


def golden_section_minimize(f, lo: float, hi: float, tol: float = 1e-5, max_iter: int = 100) -> float:
    """Bounded 1-D minimization, no scipy dependency.

    Standard golden-section search: no derivatives, no assumptions beyond
    (approximate) unimodality on [lo, hi], which holds here since
    reprojection error is smooth and has a single minimum near the true
    depth once obvious outliers are excluded (fit_depth's coarse stage).
    """
    c = hi - GOLDEN_RATIO * (hi - lo)
    d = lo + GOLDEN_RATIO * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(hi - lo) < tol:
            break
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - GOLDEN_RATIO * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + GOLDEN_RATIO * (hi - lo)
            fd = f(d)
    return (lo + hi) / 2.0


def project_points_a_to_b(
    depth: float, points_a: np.ndarray,
    camera_matrix_a_inv: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray, size_b: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project camera-A pixel points at a hypothesized plane depth into camera B.

    Mirrors register_pipeline.reproject_a_to_b's per-point formula exactly
    (back-project to the plane at `depth`, transform by R/T, project into
    B, chirality + bounds check), but for a sparse (N, 2) point array
    instead of a dense per-pixel raster -- reproject_a_to_b hardcodes a
    reshape to (height_a, width_a), so it cannot be reused directly for an
    arbitrary sparse point count. Any future change to the projection
    model must be mirrored in both places.

    Returns (predicted_b (N, 2), valid (N,) bool) -- valid is False wherever
    the projected point falls outside B's frame OR behind camera B, same
    chirality-via-w check reproject_a_to_b uses (a point behind the camera
    can otherwise divide by a negative w and land back inside frame bounds
    by coincidence).
    """
    width_b, height_b = size_b
    ones = np.ones((points_a.shape[0], 1))
    homogeneous_a = np.concatenate([points_a, ones], axis=1).T  # (3, N)

    rays_a = camera_matrix_a_inv @ homogeneous_a  # (3, N), normalized ray directions
    points_3d = rays_a * depth  # (3, N), points at the hypothesized depth
    points_b = R @ points_3d + T.reshape(3, 1)  # (3, N), camera B frame
    projected = camera_matrix_b @ points_b  # (3, N), unnormalized B-image coords

    w = projected[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        predicted = (projected[:2] / w).T  # (N, 2)

    valid = (
        (w > 1e-9)
        & (predicted[:, 0] >= 0) & (predicted[:, 0] <= width_b - 1)
        & (predicted[:, 1] >= 0) & (predicted[:, 1] <= height_b - 1)
    )
    predicted = np.nan_to_num(predicted, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return predicted, valid


def fit_depth(
    pts_a: np.ndarray, pts_b: np.ndarray,
    camera_matrix_a_inv: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray, size_b: Tuple[int, int],
    depths: np.ndarray, ransac_threshold: float,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Fit the single plane depth that best explains matched (pts_a, pts_b).

    Coarse stage: score every depth in `depths` by counting inliers (matches
    whose reprojection lands within `ransac_threshold` px) -- this is 1-D
    RANSAC via grid search, robust to however many outlier matches LightGlue
    lets through. Refine stage: golden-section search over the *full*
    [depths[0], depths[-1]] range (not just the coarse winner's neighboring
    grid cells -- an early version bracketed too tightly and consistently
    undershot the true depth by ~1-2mm in testing) minimizing median
    reprojection error over the coarse-stage inlier set.

    Returns (refined_depth, coarse_depth, inliers, errors) -- inliers/errors
    are evaluated at refined_depth, over ALL of pts_a/pts_b (not just the
    coarse inlier set), so a match the coarse stage missed can still count
    once refinement lands closer to the true depth.
    """
    best_depth = float(depths[0])
    best_inlier_count = -1
    best_errors = np.full(len(pts_a), np.inf)
    for depth in depths:
        predicted_b, valid = project_points_a_to_b(
            float(depth), pts_a, camera_matrix_a_inv, camera_matrix_b, R, T, size_b
        )
        errors = np.linalg.norm(predicted_b - pts_b, axis=1)
        errors = np.where(valid, errors, np.inf)
        inlier_count = int(np.sum(errors < ransac_threshold))
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_depth = float(depth)
            best_errors = errors

    coarse_depth = best_depth
    coarse_inliers = best_errors < ransac_threshold
    if int(coarse_inliers.sum()) < 2:
        # Too few inliers to refine meaningfully; caller (main) gates on
        # MIN_INLIERS_TO_TRUST and will SystemExit before trusting this.
        return best_depth, coarse_depth, coarse_inliers, best_errors

    lo, hi = float(depths[0]), float(depths[-1])

    def objective(depth: float) -> float:
        predicted_b, valid = project_points_a_to_b(
            depth, pts_a[coarse_inliers], camera_matrix_a_inv, camera_matrix_b, R, T, size_b
        )
        errors = np.linalg.norm(predicted_b - pts_b[coarse_inliers], axis=1)
        errors = np.where(valid, errors, ransac_threshold * 10.0)
        return float(np.median(errors))

    refined_depth = golden_section_minimize(objective, lo, hi)
    predicted_b, valid = project_points_a_to_b(
        refined_depth, pts_a, camera_matrix_a_inv, camera_matrix_b, R, T, size_b
    )
    final_errors = np.where(valid, np.linalg.norm(predicted_b - pts_b, axis=1), np.inf)
    final_inliers = final_errors < ransac_threshold
    return refined_depth, coarse_depth, final_inliers, final_errors
```

- [ ] **Step 2: Verify with a synthetic geometry test (no images, no torch)**

```bash
python3 -c "
import numpy as np
import cv2
import sys
sys.path.insert(0, '.')
from register_features import fit_depth, project_points_a_to_b

rng = np.random.default_rng(0)
camera_matrix_a = np.array([[1000.0, 0, 500.0], [0, 1000.0, 400.0], [0, 0, 1.0]])
camera_matrix_b = np.array([[1000.0, 0, 500.0], [0, 1000.0, 400.0], [0, 0, 1.0]])
R = cv2.Rodrigues(np.array([0.01, 0.02, 0.01]))[0]
T = np.array([0.015, 0.0, 0.0])
true_depth = 0.15

camera_matrix_a_inv = np.linalg.inv(camera_matrix_a)
pts_a = rng.uniform([300, 200], [700, 600], size=(60, 2))
predicted_b, valid = project_points_a_to_b(true_depth, pts_a, camera_matrix_a_inv, camera_matrix_b, R, T, (1000, 800))
assert valid.all(), 'synthetic setup should keep all points in front of/inside camera B'
pts_b = predicted_b.copy()

outlier_idx = rng.choice(60, size=10, replace=False)
pts_b[outlier_idx] += rng.uniform(-200, 200, size=(10, 2))

depths = np.sort(1.0 / np.linspace(1.0 / 0.25, 1.0 / 0.10, 60))
refined, coarse, inliers, errors = fit_depth(
    pts_a, pts_b, camera_matrix_a_inv, camera_matrix_b, R, T, (1000, 800), depths, ransac_threshold=3.0
)
print(f'true={true_depth} refined={refined:.5f} coarse={coarse:.5f} inliers={inliers.sum()}/60')
assert abs(refined - true_depth) < 1e-3, f'depth fit off by {abs(refined - true_depth)}'
assert inliers.sum() == 50, f'expected 50 inliers (10 planted outliers), got {inliers.sum()}'
assert not inliers[outlier_idx].any(), 'outliers should never be marked inlier'
print('PASS')
"
```
Expected: prints `true=0.15 refined=0.15000 coarse=...` and `PASS`, no
assertion error. (Confirmed exact match — `refined=0.15000` — during design
research with this same setup.)

- [ ] **Step 3: Commit**

```bash
git add register_features.py
git commit -m "$(cat <<'EOF'
Add register_features.py depth-fitting geometry core

Sparse-point counterpart to register_pipeline.reproject_a_to_b (which
can't be reused directly -- it hardcodes a dense-raster reshape) plus
a coarse-grid + golden-section fit for the single plane depth that
best explains a set of matched point pairs. Verified against a
synthetic scene with planted outliers; LightGlue wiring comes next.
EOF
)"
```

---

### Task 3: LightGlue feature extraction & matching wrapper

**Files:**
- Modify: `register_features.py`

**Interfaces:**
- Produces: `register_features.load_models(device: torch.device) -> Tuple[KF.DISK, KF.LightGlueMatcher]`
- Produces: `register_features.extract_features(disk: KF.DISK, image_bgr: np.ndarray, device: torch.device, max_keypoints: int) -> Tuple[KF.DISKFeatures, Tuple[int, int]]` — second element is `(H, W)` of the tensor fed to DISK.
- Produces: `register_features.match_features(matcher: KF.LightGlueMatcher, feats_a: KF.DISKFeatures, feats_b: KF.DISKFeatures, hw_a: Tuple[int, int], hw_b: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` — `(pts_a (M,2), pts_b (M,2), scores (M,))`, all matches kornia returned, unfiltered by confidence (caller applies `--min-confidence`).
- Consumes: nothing from Task 2 directly (this task is independent of the geometry core; both feed into Task 5's `main`).

- [ ] **Step 1: Add imports and the three functions**

At the top of `register_features.py`, change:
```python
from __future__ import annotations

from typing import Tuple

import numpy as np
```
to:
```python
from __future__ import annotations

from typing import Tuple

import cv2
import kornia.feature as KF
import numpy as np
import torch
```

Then insert this new section between the import block and the existing
`# --- Depth fit ---` section header (i.e., it becomes the first section in
the file, ahead of depth-fitting):

```python
DEFAULT_MAX_KEYPOINTS = 2048


# --------------------------------------------------------------------------- #
# Feature extraction & matching
# --------------------------------------------------------------------------- #
def load_models(device: torch.device) -> Tuple[KF.DISK, KF.LightGlueMatcher]:
    """Load the pretrained DISK extractor and LightGlue matcher once.

    Weights auto-download from kornia's model hub on first call (needs
    internet once; cached locally after that).
    """
    disk = KF.DISK.from_pretrained("depth").to(device).eval()
    matcher = KF.LightGlueMatcher("disk").to(device).eval()
    return disk, matcher


def extract_features(
    disk: KF.DISK, image_bgr: np.ndarray, device: torch.device, max_keypoints: int,
) -> Tuple["KF.DISKFeatures", Tuple[int, int]]:
    """DISK keypoints + descriptors for one BGR image.

    Returns (features, (H, W)) -- the (H, W) is the tensor shape fed to
    DISK, needed unchanged as LightGlueMatcher's hw1/hw2 argument later
    (pad_if_not_divisible pads internally but keypoints stay in this
    original, unpadded coordinate frame).
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        features = disk(tensor, n=max_keypoints, pad_if_not_divisible=True)[0]
    return features, (tensor.shape[2], tensor.shape[3])


def match_features(
    matcher: KF.LightGlueMatcher,
    feats_a: "KF.DISKFeatures", feats_b: "KF.DISKFeatures",
    hw_a: Tuple[int, int], hw_b: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match two DISK feature sets with LightGlue.

    Returns (pts_a, pts_b, scores) for every match kornia returned --
    unfiltered by confidence, since --min-confidence is a user-tunable CLI
    threshold applied by the caller, not baked into this wrapper. `scores`
    is LightGlue's own matching confidence in [0, 1]; higher is better.
    """
    lafs_a = KF.laf_from_center_scale_ori(feats_a.keypoints[None])
    lafs_b = KF.laf_from_center_scale_ori(feats_b.keypoints[None])
    with torch.no_grad():
        scores, matches = matcher(
            feats_a.descriptors, feats_b.descriptors, lafs_a, lafs_b, hw1=hw_a, hw2=hw_b
        )
    matches_np = matches.cpu().numpy()
    scores_np = scores.reshape(-1).cpu().numpy()
    pts_a = feats_a.keypoints.cpu().numpy()[matches_np[:, 0]]
    pts_b = feats_b.keypoints.cpu().numpy()[matches_np[:, 1]]
    return pts_a, pts_b, scores_np
```

- [ ] **Step 2: Verify against real capture images**

```bash
python3 -c "
import sys, time
sys.path.insert(0, '.')
import cv2, torch
from register_features import load_models, extract_features, match_features

image_a = cv2.imread('captures/20260818_152110_605174/rgb_cam1.jpg')
image_b = cv2.imread('captures/20260818_152110_605174/rgb_cam2.jpg')
assert image_a is not None and image_b is not None, 'session images not found -- check the path'

scale = 0.25
small_a = cv2.resize(image_a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
small_b = cv2.resize(image_b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

device = torch.device('cpu')
disk, matcher = load_models(device)
t0 = time.time()
feats_a, hw_a = extract_features(disk, small_a, device, max_keypoints=2048)
feats_b, hw_b = extract_features(disk, small_b, device, max_keypoints=2048)
pts_a, pts_b, scores = match_features(matcher, feats_a, feats_b, hw_a, hw_b)
print(f'{feats_a.keypoints.shape[0]} / {feats_b.keypoints.shape[0]} keypoints, '
      f'{len(scores)} matches, score range [{scores.min():.2f}, {scores.max():.2f}], '
      f'{time.time()-t0:.1f}s')
assert len(scores) >= 500, f'expected >=500 raw matches on this session, got {len(scores)}'
assert pts_a.shape == pts_b.shape == (len(scores), 2)
print('PASS')
"
```
Expected: prints keypoint/match counts (design research saw 1301 raw
matches on this exact session/downscale) and `PASS`. This is *not*
run against synthetic data on purpose — DISK/LightGlue's whole value is
real-image robustness, and this is the actual session the extrinsics were
already verified against.

- [ ] **Step 3: Commit**

```bash
git add register_features.py
git commit -m "$(cat <<'EOF'
Add LightGlue feature extraction and matching to register_features.py

kornia's DISK extractor + LightGlueMatcher, wired up and verified
against a real capture pair (1301 raw matches on
captures/20260818_152110_605174 at 0.25x downscale).
EOF
)"
```

---

### Task 4: Visualization & report writers

**Files:**
- Modify: `register_features.py`

**Interfaces:**
- Produces: `register_features.render_match_visualization(color_a: np.ndarray, color_b: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray, inliers: np.ndarray) -> np.ndarray` — side-by-side image with correspondence lines.
- Produces: `register_features.save_preview(path: Path, color_a: np.ndarray, warped_color: np.ndarray, match_viz: np.ndarray) -> None`
- Produces: `register_features.write_report(path: Path, session_dir: Path, camera_a: str, camera_b: str, extrinsics_path: Path, extrinsics: StereoExtrinsics, size_a: Tuple[int, int], n_raw_matches: int, min_confidence: float, n_matches: int, depths: np.ndarray, ransac_threshold: float, fitted_depth: float, inliers: np.ndarray, errors: np.ndarray, default_depth: float, d_critical: float, warnings: List[str]) -> None`
- Consumes: nothing new from Tasks 2/3 directly at the function-signature level (this task's functions take plain arrays/scalars) — Task 5 wires the real values in.

- [ ] **Step 1: Add imports and the three functions**

Change the import block again, from:
```python
from __future__ import annotations

from typing import Tuple

import cv2
import kornia.feature as KF
import numpy as np
import torch
```
to:
```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import kornia.feature as KF
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.stereo import StereoExtrinsics  # noqa: E402
from register_pipeline import PREVIEW_PANEL_WIDTH  # noqa: E402
```

Then add this new section at the end of the file (after `fit_depth`):

```python
# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def render_match_visualization(
    color_a: np.ndarray, color_b: np.ndarray,
    pts_a: np.ndarray, pts_b: np.ndarray, inliers: np.ndarray,
) -> np.ndarray:
    """Side-by-side correspondence visualization: green lines/dots for inlier
    matches, red for outliers. A and B keep their native size here (no
    PREVIEW_PANEL_WIDTH resize) so matches.jpg is legible at full detail;
    save_preview resizes a copy for the combined strip."""
    height = max(color_a.shape[0], color_b.shape[0])
    canvas = np.zeros((height, color_a.shape[1] + color_b.shape[1], 3), dtype=np.uint8)
    canvas[: color_a.shape[0], : color_a.shape[1]] = color_a
    canvas[: color_b.shape[0], color_a.shape[1] :] = color_b
    offset_x = color_a.shape[1]

    for i in range(len(pts_a)):
        color = (0, 200, 0) if inliers[i] else (0, 0, 200)
        point_a = (int(round(pts_a[i, 0])), int(round(pts_a[i, 1])))
        point_b = (int(round(pts_b[i, 0] + offset_x)), int(round(pts_b[i, 1])))
        cv2.line(canvas, point_a, point_b, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, point_a, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point_b, 3, color, -1, cv2.LINE_AA)
    return canvas


def _labelled(image: np.ndarray, text: str) -> np.ndarray:
    frame = image.copy()
    cv2.putText(frame, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def _resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] <= width:
        return image
    scale = width / image.shape[1]
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def save_preview(
    path: Path, color_a: np.ndarray, warped_color: np.ndarray, match_viz: np.ndarray,
) -> None:
    """camera A | warped B | match visualization, side by side.

    Unlike register_pipeline.save_preview's three panels (which all share
    size_a's aspect ratio), match_viz is roughly twice as wide as the other
    two, so after independent _resize_to_width scaling the panels can end
    up different heights -- pad each to the tallest before hstacking, or
    np.hstack raises on mismatched shapes.
    """
    panels = [
        _resize_to_width(_labelled(color_a, "camera A"), PREVIEW_PANEL_WIDTH),
        _resize_to_width(_labelled(warped_color, "B warped onto A"), PREVIEW_PANEL_WIDTH),
        _resize_to_width(_labelled(match_viz, "LightGlue matches"), PREVIEW_PANEL_WIDTH),
    ]
    max_height = max(panel.shape[0] for panel in panels)
    padded = [
        cv2.copyMakeBorder(panel, 0, max_height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for panel in panels
    ]
    preview = np.hstack(padded)
    cv2.imwrite(str(path), preview)


def write_report(
    path: Path,
    session_dir: Path, camera_a: str, camera_b: str, extrinsics_path: Path,
    extrinsics: StereoExtrinsics, size_a: Tuple[int, int],
    n_raw_matches: int, min_confidence: float, n_matches: int,
    depths: np.ndarray, ransac_threshold: float,
    fitted_depth: float, inliers: np.ndarray, errors: np.ndarray,
    default_depth: float, d_critical: float, warnings: List[str],
) -> None:
    """n_matches and inliers.sum() are both guaranteed > 0 here -- main()
    SystemExits before calling this if --min-matches or MIN_INLIERS_TO_TRUST
    aren't met, so no n/a branches are needed (unlike register_pipeline's
    write_report, which isn't gated the same way for its DEPTH MAP section)."""
    n_inliers = int(inliers.sum())
    inlier_errors = errors[inliers]

    lines = [
        f"Registration: {camera_b} -> {camera_a} (LightGlue sparse match + single fitted plane depth)",
        "=" * 66,
        "",
        f"  session:            {session_dir}",
        f"  extrinsics:         {extrinsics_path}",
        f"  resolution:         {size_a[0]}x{size_a[1]}",
        f"  baseline:           {extrinsics.baseline_m * 1000:.2f} mm",
        f"  convergence:        {extrinsics.optical_axis_angle_deg:.2f} deg",
        "",
        "MATCHING",
        "-" * 66,
        f"  raw LightGlue matches:      {n_raw_matches}",
        f"  above min-confidence {min_confidence}: {n_matches}",
        "",
        "DEPTH FIT",
        "-" * 66,
        f"  search range:        {depths[0]:.4f} - {depths[-1]:.4f} m "
        f"({len(depths)} steps, uniform in 1/depth)",
        f"  ransac threshold:    {ransac_threshold} px",
        f"  fitted depth:        {fitted_depth:.4f} m",
        f"  inliers:             {n_inliers} / {n_matches} ({n_inliers / n_matches * 100:.1f}%)",
        f"  reprojection error (inliers, px): median {np.median(inlier_errors):.2f}, "
        f"mean {inlier_errors.mean():.2f}, max {inlier_errors.max():.2f}",
        f"  calibrated default depth: {default_depth:.4f} m "
        f"(delta {abs(fitted_depth - default_depth) * 1000:.1f} mm)",
        f"  homography singular at:  {d_critical * 1000:.2f} mm "
        f"({'well clear of' if abs(d_critical) < depths[0] / 5 else 'CHECK: close to'} "
        "the search range)",
    ]
    if warnings:
        lines += ["", "Quality warnings", "-" * 66]
        lines += [f"  - {warning}" for warning in warnings]
    lines += [
        "",
        "OUTPUT FILES",
        "-" * 66,
        "  warped_features.jpg   camera B warped onto camera A via the fitted single-depth homography",
        "  matches.jpg           inlier (green) / outlier (red) correspondence lines",
        "  preview_features.jpg  camera A | warped B | match visualization, side by side",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 2: Verify with real images + Task 2/3's real outputs**

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
import cv2, numpy as np
from pathlib import Path
from register_features import render_match_visualization, save_preview

color_a = cv2.imread('captures/20260818_152110_605174/rgb_cam1.jpg')
color_b = cv2.imread('captures/20260818_152110_605174/rgb_cam2.jpg')
color_a = cv2.resize(color_a, None, fx=0.25, fy=0.25)
color_b = cv2.resize(color_b, None, fx=0.25, fy=0.25)

rng = np.random.default_rng(0)
pts_a = rng.uniform([0, 0], [color_a.shape[1], color_a.shape[0]], size=(40, 2))
pts_b = rng.uniform([0, 0], [color_b.shape[1], color_b.shape[0]], size=(40, 2))
inliers = np.zeros(40, dtype=bool); inliers[:25] = True

match_viz = render_match_visualization(color_a, color_b, pts_a, pts_b, inliers)
assert match_viz.shape[1] == color_a.shape[1] + color_b.shape[1]

out = Path('/tmp/preview_features_test.jpg')
save_preview(out, color_a, color_b, match_viz)
assert out.exists() and out.stat().st_size > 10_000, 'preview file missing or suspiciously small'
print('preview size', out.stat().st_size, 'bytes -- PASS')
"
```
Expected: prints a preview file size in the tens-of-KB range and `PASS`.
(`write_report` is exercised for real as part of Task 5's end-to-end
verification, once real `StereoExtrinsics`/depth-fit values exist to feed
it — no point synthesizing a fake `StereoExtrinsics` here.)

- [ ] **Step 3: Commit**

```bash
git add register_features.py
git commit -m "$(cat <<'EOF'
Add visualization and report writers to register_features.py

Match-correspondence visualization, a 3-panel preview (mirroring
register_pipeline.py's layout), and a text report -- all specific to
the single-depth-fit output shape (no dense depth map to render).
EOF
)"
```

---

### Task 5: CLI entry point, full wiring, README

**Files:**
- Modify: `register_features.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `register_features.parse_args() -> argparse.Namespace`
- Produces: `register_features.main() -> int`
- Consumes: everything from Tasks 2-4, plus `register_pipeline.{DEFAULT_DEPTH_MIN, DEFAULT_DEPTH_MAX, DEFAULT_STEPS, DEFAULT_REGISTRATION_OUTPUT_DIR, candidate_depths, compose_warped_output, default_extrinsics_path, downscale_pair, load_session_images, singular_depth, undistort_pair}` and `calibrate_cameras.{load_config, resolve_path}`.

- [ ] **Step 1: Extend imports to the final set**

Change:
```python
from calibration.stereo import StereoExtrinsics  # noqa: E402
from register_pipeline import PREVIEW_PANEL_WIDTH  # noqa: E402
```
to:
```python
from calibrate_cameras import load_config, resolve_path  # noqa: E402
from calibration.stereo import StereoExtrinsics  # noqa: E402
from register_pipeline import (  # noqa: E402
    DEFAULT_DEPTH_MIN,
    DEFAULT_DEPTH_MAX,
    DEFAULT_REGISTRATION_OUTPUT_DIR,
    DEFAULT_STEPS,
    PREVIEW_PANEL_WIDTH,
    candidate_depths,
    compose_warped_output,
    default_extrinsics_path,
    downscale_pair,
    load_session_images,
    singular_depth,
    undistort_pair,
)
```
Also add `import argparse` as the first import line (before `import sys`).

- [ ] **Step 2: Add CLI/config constants and `parse_args`**

Insert this new section between the import block and the existing
`# --- Feature extraction & matching ---` section header added in Task 3
(i.e., it becomes the first section in the file, ahead of everything else —
mirroring `register_pipeline.py`, where `parse_args` is the first thing
defined after imports):

```python
DEFAULT_MIN_MATCHES = 20
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_RANSAC_THRESHOLD = 3.0
MIN_INLIERS_TO_TRUST = 8  # minimum to trust a 1-DOF (single depth) fit


# --------------------------------------------------------------------------- #
# CLI / config
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    reg_config = load_config().get("registration", {}) or {}
    depth_range = reg_config.get("depth_range", [DEFAULT_DEPTH_MIN, DEFAULT_DEPTH_MAX])

    parser = argparse.ArgumentParser(
        description="Register camera B onto camera A via LightGlue sparse matching "
                    "and a single fitted plane depth.",
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
                        help="Near edge of the depth-fit search range, metres "
                             "(default: registration.depth_range[0] in config, "
                             f"else {DEFAULT_DEPTH_MIN}).")
    parser.add_argument("--depth-max", type=float, default=depth_range[1],
                        help="Far edge of the depth-fit search range, metres "
                             "(default: registration.depth_range[1] in config, "
                             f"else {DEFAULT_DEPTH_MAX}).")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="Coarse grid resolution for the depth fit, sampled uniformly "
                             "in inverse depth (default: %(default)s).")
    parser.add_argument("--downscale", type=float, default=1.0,
                        help="Resize factor applied to the undistorted images before "
                             "feature extraction, e.g. 0.25 for the native ~4656x3496 "
                             "sensor -- DISK is CPU-heavy at full resolution "
                             "(default: %(default)s).")
    parser.add_argument("--min-matches", type=int, default=DEFAULT_MIN_MATCHES,
                        help="Minimum raw LightGlue matches (post --min-confidence) "
                             "required to attempt a depth fit (default: %(default)s).")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                        help="Minimum LightGlue match confidence to keep, in [0, 1] "
                             "(default: %(default)s).")
    parser.add_argument("--ransac-threshold", type=float, default=DEFAULT_RANSAC_THRESHOLD,
                        help="Inlier pixel-reprojection threshold for the depth fit, in "
                             "the working (possibly downscaled) resolution "
                             "(default: %(default)s).")
    parser.add_argument("--out", "--output", dest="output", default=None,
                        help="Output directory (default: registration.output_dir in config, "
                             f"else {DEFAULT_REGISTRATION_OUTPUT_DIR}) / <session name>.")
    return parser.parse_args()
```

- [ ] **Step 3: Add `main` and the entry point**

Add at the end of the file:

```python
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

    device = torch.device("cpu")
    print(f"Registering {args.camera_b} -> {args.camera_a}: LightGlue matching at "
          f"{size_a[0]}x{size_a[1]} (CPU)")
    disk, matcher = load_models(device)
    feats_a, hw_a = extract_features(disk, color_a, device, DEFAULT_MAX_KEYPOINTS)
    feats_b, hw_b = extract_features(disk, color_b, device, DEFAULT_MAX_KEYPOINTS)
    pts_a_all, pts_b_all, scores_all = match_features(matcher, feats_a, feats_b, hw_a, hw_b)

    keep = scores_all >= args.min_confidence
    pts_a, pts_b = pts_a_all[keep], pts_b_all[keep]
    n_raw_matches, n_matches = len(scores_all), int(keep.sum())
    print(f"LightGlue: {n_raw_matches} raw matches, {n_matches} above confidence "
          f"{args.min_confidence}")
    if n_matches < args.min_matches:
        raise SystemExit(
            f"Only {n_matches} matches above --min-confidence {args.min_confidence} "
            f"(need >= {args.min_matches}). Check --camera-a/--camera-b order, exposure "
            "match between cameras, or lower --min-confidence."
        )

    camera_matrix_a_inv = np.linalg.inv(camera_matrix_a)
    depths = candidate_depths(args.depth_min, args.depth_max, args.steps)
    refined_depth, coarse_depth, inliers, errors = fit_depth(
        pts_a, pts_b, camera_matrix_a_inv, camera_matrix_b,
        extrinsics.R, extrinsics.T, size_b, depths, args.ransac_threshold,
    )
    n_inliers = int(inliers.sum())
    if n_inliers > 0:
        print(f"Fitted depth: {refined_depth:.4f} m ({n_inliers}/{n_matches} inliers, "
              f"median error {np.median(errors[inliers]):.2f}px)")
    if n_inliers < MIN_INLIERS_TO_TRUST:
        raise SystemExit(
            f"Only {n_inliers} inlier matches at the fitted depth (need >= "
            f"{MIN_INLIERS_TO_TRUST} to trust a 1-DOF plane fit). Check calibration, "
            "--ransac-threshold, or capture a more textured scene."
        )

    warped_color, _remap_valid = compose_warped_output(
        refined_depth, color_b, camera_matrix_a, camera_matrix_b,
        extrinsics.R, extrinsics.T, size_a, size_b,
    )
    match_viz = render_match_visualization(color_a, color_b, pts_a, pts_b, inliers)

    cv2.imwrite(str(output_dir / "warped_features.jpg"), warped_color)
    cv2.imwrite(str(output_dir / "matches.jpg"), match_viz)
    save_preview(output_dir / "preview_features.jpg", color_a, warped_color, match_viz)

    d_critical = singular_depth(extrinsics.R, extrinsics.T)
    warnings: List[str] = []
    if np.isclose(coarse_depth, depths[0]) or np.isclose(coarse_depth, depths[-1]):
        warnings.append(
            f"Fitted depth landed on the edge of the search range ({depths[0]:.3f} or "
            f"{depths[-1]:.3f} m). Widen --depth-min/--depth-max."
        )

    write_report(
        output_dir / "report_features.txt",
        session_dir, args.camera_a, args.camera_b, extrinsics_path, extrinsics, size_a,
        n_raw_matches, args.min_confidence, n_matches,
        depths, args.ransac_threshold,
        refined_depth, inliers, errors,
        default_depth, d_critical, warnings,
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

- [ ] **Step 4: Update README.md**

In the stage table (around line 18), add a row right after the existing
Registration row:

```
| Registration (sparse)| `register_features.py` | LightGlue-fitted single-plane warp                           |
```

At the end of the existing `## Registration` section (after its example
commands, before `## Config highlights`), add:

```markdown
### Sparse alternative: `register_features.py`

For subjects close enough to a single plane, `register_features.py` is an
alternative to the dense plane-sweep above: it matches sparse features with
LightGlue and fits the one scalar depth that best explains those matches
through the same calibrated geometry, instead of scoring every pixel with
ZNCC. Useful when plane-sweep's dense correlation degenerates (e.g. low
per-pixel texture) — check `report_features.txt`'s inlier count/ratio and
`preview_features.jpg` the same way you'd check plane-sweep's confident
fraction. Needs `torch`/`kornia` (see `requirements.txt`); CPU-only, no GPU
required.

```bash
python register_features.py --session captures/hand
python register_features.py --session captures/hand --downscale 0.25
```

Outputs under `registration/results/<session>/`: `warped_features.jpg`,
`matches.jpg` (inlier/outlier correspondence lines), `preview_features.jpg`,
`report_features.txt`.
```

Also update the `Layout` section's tree near the bottom (around where
`register_pipeline.py` is listed) to add:
```
register_features.py                       LightGlue sparse match + single-depth warp
```

- [ ] **Step 5: End-to-end verification against real sessions**

```bash
python3 register_features.py --session captures/20260818_152110_605174 --downscale 0.25
```
Expected: prints raw/confident match counts, fitted depth, inlier count,
`Saved outputs to .../registration/results/20260818_152110_605174`, exit
code 0. (Design research on this exact session/downscale: 1301 raw matches,
1250 above 0.5 confidence, fitted depth 0.1745 m, 172 inliers, median error
1.50px — expect numbers in this neighborhood, not necessarily bit-identical.)

```bash
ls registration/results/20260818_152110_605174/
```
Expected: `warped_features.jpg`, `matches.jpg`, `preview_features.jpg`,
`report_features.txt` all present and non-empty.

```bash
cat registration/results/20260818_152110_605174/report_features.txt
```
Expected: a populated report — inspect it directly to sanity-check the
fitted depth is within `[0.11, 0.21]` (the physical working range this rig
was calibrated for) and reasonably close to the calibrated default (0.168
m); read the delta rather than asserting an exact threshold, since a real
subject legitimately sits at a different distance than the reference plane.

Then visually inspect `preview_features.jpg` (e.g. open it, or in an
environment with image display, read it back) — the warped-B panel should
show recognizable alignment with camera A (fingernails/skin features
overlapping, not offset or badly distorted), the concrete check the design
doc's verification plan calls for.

Also re-run against the second real session as a cross-check:
```bash
python3 register_features.py --session captures/20260818_152114_103039 --downscale 0.25
```
Expected: same shape of output, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add register_features.py README.md
git commit -m "$(cat <<'EOF'
Wire up register_features.py CLI and document it in README

Full LightGlue-based registration pipeline: argparse CLI matching
register_pipeline.py's conventions, end-to-end main() gated by
--min-matches and a minimum inlier count before trusting the fit.
Verified end-to-end against two real capture sessions.
EOF
)"
```

---

## Out of scope (confirmed in the design doc)

- No changes to `register_pipeline.py` itself.
- No dense per-pixel depth output from this script — single depth + one
  homography only.
- No GPU-specific code path (CPU `torch.device` is hardcoded).
- Thermal registration remains unhandled by both scripts.
