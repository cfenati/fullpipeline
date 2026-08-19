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
