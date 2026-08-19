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
