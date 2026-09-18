# Design: `register_features.py`

> **Superseded.** This documents the plane-fit design: a single fitted plane depth, reusing `register_pipeline.py`'s `plane_homography`, `candidate_depths` and `singular_depth`. `register_pipeline.py` was deleted in `ff8c992`, and `register_features.py` now triangulates every match with no assumed plane depth (`triangulate.py`; see the `register_features.py` docstring and the README "Registration" section). Kept as a design record only.

**Date:** 2026-08-18
**Status:** Approved, then superseded (see note above)

## Purpose

A new standalone entry-point script that registers `rgb_cam2` onto `rgb_cam1`
using sparse deep-learned feature matching (LightGlue) instead of the dense
plane-sweep ZNCC approach in `register_pipeline.py`. This is separate from and
does not replace `register_pipeline.py` — it addresses the case where that
script's dense correlation degenerates.

## Motivation

`register_pipeline.py`'s plane-sweep only scored 27-32% of the cameras'
overlap region as confident on the existing `captures/cross-validation` and
finger-closeup sessions (see `registration/results/*/report.txt`); most
pixels fell back to the default reference-plane depth, and `preview.jpg`
shows visibly noisy, misaligned warps on the textured subject (finger
surfaces) against the ZNCC patch matcher. The extrinsics themselves have
since been independently re-verified against a different calibration
pipeline (DIYer22/calibrating) and are trusted, so the problem is specific to
plane-sweep's per-pixel patch correlation, not the calibration.

## Approach

The subject in the target use case (close-range hand/finger shots) is close
enough to a single plane that one homography per capture is an acceptable
simplification. Rather than fit a free 8-DOF homography (`cv2.findHomography`)
that discards the calibration, this script fits the **one unknown that
matters** — a scalar plane depth `d` — through `register_pipeline.py`'s
existing analytic model:

```
H(d) = K_b @ (R + T @ n^T / d) @ inv(K_a)
```

`plane_homography()`, `candidate_depths()`, `singular_depth()`,
`load_session_images()`, `undistort_pair()`, `downscale_pair()`, and
`scale_camera_matrix()` are imported from `register_pipeline.py` and reused
verbatim — this script does not reimplement calibration-facing geometry.

### Algorithm

1. Load + undistort the pair (`load_session_images`, `undistort_pair`,
   optional `downscale_pair`), exactly as `register_pipeline.py` does.
2. Extract features and match with LightGlue: kornia's `DISK` extractor +
   `LightGlueMatcher` (pretrained `"disk"` weights), filtered to matches
   above `--min-confidence`.
3. Fit `d`:
   - Coarse stage: score every depth in `candidate_depths(depth_min,
     depth_max, steps)` by counting inliers (matches whose
     `H(d)`-reprojected point lands within `--ransac-threshold` px of its
     matched point) — this is RANSAC over a single parameter, done by grid
     search since the parameter is 1-D and the range is already bounded by
     the rig's known working distance.
   - Refine stage: golden-section search (hand-rolled, no new dependency)
     around the coarse winner, minimizing the median reprojection error over
     that depth's inlier set, to land between grid steps.
4. Compose the final warp with `plane_homography(d_refined)` and
   `cv2.warpPerspective`.

### Why not a free homography

Considered and rejected: an unconstrained `cv2.findHomography` would need
enough well-distributed inlier matches to constrain 8 unknowns reliably, and
loses the physical depth estimate. Fitting 1 unknown against the calibrated
model is more robust when LightGlue returns few matches (e.g. on the
low-texture background visible in the existing captures) and keeps the
result grounded in the calibration the user just re-verified.

## Dependencies

Adds to `requirements.txt`:

- `torch` — LightGlue/DISK run on it. CPU-only is fine for single-pair
  registration; no CUDA requirement.
- `kornia` — provides both the `DISK` feature extractor and
  `LightGlueMatcher`, both plain PyPI packages (no git-clone of the original
  cvg/LightGlue research repo needed).

This is a real cost worth flagging explicitly: a few hundred MB of install,
and kornia downloads pretrained DISK/LightGlue weights from its model hub on
first run, which needs internet access once (cached locally after that, the
same way OpenCV's ChArUco/aruco assets don't need re-fetching each run).

## CLI

Mirrors `register_pipeline.py`'s argparse conventions:

- `--session` (required) — same session folder convention
  (`<camera>.jpg` per camera).
- `--camera-a` / `--camera-b` — default `rgb_cam1` / `rgb_cam2`.
- `--extrinsics` — default resolution identical to `register_pipeline.py`'s
  `default_extrinsics_path()`.
- `--depth-min` / `--depth-max` — search bounds for the depth fit; same
  config-driven defaults (`registration.depth_range`) as `register_pipeline.py`.
- `--steps` — coarse grid resolution (default matches `register_pipeline.py`'s
  `DEFAULT_STEPS`).
- `--downscale` — resize factor before feature extraction (default `1.0`).
- `--min-matches` — minimum raw LightGlue matches required to proceed
  (`SystemExit` below this, mirroring `register_pipeline.py`'s guard style).
- `--min-confidence` — LightGlue match confidence threshold.
- `--ransac-threshold` — inlier pixel-reprojection threshold, in the
  (possibly downscaled) working resolution.
- `--out` / `--output` — output directory, default
  `registration.output_dir` in config / `<session name>`, same as
  `register_pipeline.py`.

## Outputs

Under `registration/results/<session>/` (same root as
`register_pipeline.py`; filenames disambiguate the method):

- `warped_features.jpg` — camera B warped onto camera A via the fitted
  single-depth homography.
- `preview_features.jpg` — camera A | warped B | match visualization,
  side by side (same layout convention as `register_pipeline.py`'s
  `preview.jpg`).
- `matches.jpg` — inlier (green) vs. outlier (red) correspondence lines
  between the raw pair, for debugging this method specifically.
- `report_features.txt` — fitted depth, raw match count, inlier count and
  ratio, reprojection error stats (median/mean/max px) over inliers,
  comparison against `registration.default_depth` from config, and quality
  warnings.

## Error handling / quality warnings

Mirrors `register_pipeline.py`'s tone:

- Fewer than `--min-matches` raw matches → `SystemExit` with a clear message
  (don't silently fit garbage).
- Fewer than 8 inliers at the fitted depth (the minimum to trust a 1-DOF fit
  robustly) → `SystemExit`, same reasoning. Inlier *ratio* is reported but
  not gated on — a real image pair full of textureless background is
  expected to produce a low ratio even when the plane fit itself is good.
- Fitted depth landing on the edge of `[depth-min, depth-max]` → warning
  (mirrors `register_pipeline.py`'s `BOUNDARY_CLIP_WARN_FRACTION` check),
  since it suggests the search range should widen.
- Fitted depth far from `registration.default_depth` → informational note,
  not a hard warning (a real subject at a different distance than the
  reference plane is expected, not an error).

## Verification plan

No hardware needed — this operates on already-saved JPEGs. Validation:

1. Run against real capture sessions that still have raw
   `rgb_cam1.jpg`/`rgb_cam2.jpg` on disk. Of the three sessions originally
   referenced under `registration/results/`, only
   `captures/cross-validation/20260817_164509_218202` still has its raw
   captures — the other two were since pruned/deleted, leaving only their
   old plane-sweep outputs. Use that session plus the two fresh sessions
   captured during this feature's own design research
   (`captures/20260818_152110_605174`, `captures/20260818_152114_103039`,
   both finger closeups) instead.
2. Compare `preview_features.jpg` against the corresponding plane-sweep
   `preview.jpg` where one exists — the finger edges that were visibly
   noisy in the plane-sweep warp are the concrete thing to check. (Already
   done informally during design research: LightGlue + single-depth-fit
   produced a visibly clean, well-aligned warp on both the ChArUco-board
   cross-validation session and a real finger-closeup session, where
   plane-sweep's warp was noisy — see the implementation plan's captured
   numbers.)
3. Compare the fitted depth against plane-sweep's confident-pixel depth
   mean where a plane-sweep report exists (~0.153 m on the
   cross-validation session) as a sanity cross-check between the two
   independent methods.
4. `python -c "import register_features"` for a basic import/syntax check
   before the above.

## Out of scope

- Not a replacement for `register_pipeline.py`; both remain available.
- No dense per-pixel depth output — single scalar depth + one homography
  only, per the "single plane is fine" scoping decision.
- No GPU-specific optimization; CPU inference is acceptable for
  single-pair, offline registration.
- Thermal registration remains out of scope for both scripts, as already
  noted in `register_pipeline.py`'s docstring.
