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

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import kornia.feature as KF
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


DEFAULT_MAX_KEYPOINTS = 2048
DEFAULT_MIN_MATCHES = 20
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_RANSAC_THRESHOLD = 3.0
MIN_INLIERS_TO_TRUST = 8  # minimum to trust a 1-DOF (single depth) fit
AMBIGUOUS_DEPTH_GAP_STEPS = 3  # grid points closer than this to the winner aren't a
                               # competing hypothesis, just its own neighborhood
AMBIGUOUS_INLIER_RATIO = 0.85  # runner-up inlier count / winner inlier count threshold
                               # to flag a genuinely competitive alternate depth


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
                        help="Minimum matches above --min-confidence required to attempt "
                             "a depth fit (default: %(default)s).")
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


def golden_section_minimize(
    f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-5, max_iter: int = 100,
) -> float:
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
) -> Tuple[float, float, np.ndarray, np.ndarray, Optional[float]]:
    """Fit the single plane depth that best explains matched (pts_a, pts_b).

    Coarse stage: score every depth in `depths` by counting inliers (matches
    whose reprojection lands within `ransac_threshold` px) -- this is 1-D
    RANSAC via grid search, robust to however many outlier matches LightGlue
    lets through. Refine stage: golden-section search over the *full*
    [depths[0], depths[-1]] range (not just the coarse winner's neighboring
    grid cells -- an early version bracketed too tightly and consistently
    undershot the true depth by ~1-2mm in testing) minimizing median
    reprojection error over the coarse-stage inlier set.

    Ambiguity check: every candidate depth's coarse inlier count is kept (not
    just the running best/second, discarded in earlier versions), so after
    the coarse loop we can look for a "runner-up" -- among grid points whose
    index is more than AMBIGUOUS_DEPTH_GAP_STEPS away from the winner's index
    (so a merely-adjacent grid point isn't mistaken for a competing
    hypothesis), the one with the highest inlier count. If that runner-up's
    inlier count is at least AMBIGUOUS_INLIER_RATIO times the winner's, the
    scene has two near-equally-good planar explanations and the single-plane
    fit is reported as ambiguous rather than silently picking whichever one
    the grid happened to favor.

    Returns (refined_depth, coarse_depth, inliers, errors, ambiguous_depth):
    inliers/errors are evaluated at refined_depth, over ALL of pts_a/pts_b
    (not just the coarse inlier set), so a match the coarse stage missed can
    still count once refinement lands closer to the true depth -- EXCEPT on
    the early-return branch below (fewer than 2 coarse inliers), which skips
    refinement entirely and returns the unrefined coarse depth with
    inliers/errors evaluated at THAT depth, not a golden-section-refined one.
    ambiguous_depth is the competitive runner-up depth in metres if the
    ambiguity check above fired, else None.
    """
    inlier_counts = np.empty(len(depths), dtype=np.int64)
    best_depth = float(depths[0])
    best_inlier_count = -1
    best_index = 0
    best_errors = np.full(len(pts_a), np.inf)
    for i, depth in enumerate(depths):
        predicted_b, valid = project_points_a_to_b(
            float(depth), pts_a, camera_matrix_a_inv, camera_matrix_b, R, T, size_b
        )
        errors = np.linalg.norm(predicted_b - pts_b, axis=1)
        errors = np.where(valid, errors, np.inf)
        inlier_count = int(np.sum(errors < ransac_threshold))
        inlier_counts[i] = inlier_count
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_depth = float(depth)
            best_index = i
            best_errors = errors

    # Runner-up search: mask out grid points near the winner, then take the
    # highest inlier count among what's left. np.argmax picks the first
    # occurrence of the max, same tie-breaking convention as the running-best
    # loop above (only a strict `>` replaces the incumbent).
    far_from_winner = np.abs(np.arange(len(depths)) - best_index) > AMBIGUOUS_DEPTH_GAP_STEPS
    ambiguous_depth: Optional[float] = None
    if np.any(far_from_winner):
        masked_counts = np.where(far_from_winner, inlier_counts, -1)
        runner_up_index = int(np.argmax(masked_counts))
        runner_up_count = int(masked_counts[runner_up_index])
        if runner_up_count >= AMBIGUOUS_INLIER_RATIO * best_inlier_count:
            ambiguous_depth = float(depths[runner_up_index])

    coarse_depth = best_depth
    coarse_inliers = best_errors < ransac_threshold
    if int(coarse_inliers.sum()) < 2:
        # Too few inliers to refine meaningfully; caller (main) gates on
        # MIN_INLIERS_TO_TRUST and will SystemExit before trusting this.
        return best_depth, coarse_depth, coarse_inliers, best_errors, ambiguous_depth

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
    return refined_depth, coarse_depth, final_inliers, final_errors, ambiguous_depth


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
    default_depth: float, d_critical: float, n_outside_fov: int,
    ambiguous_depth: Optional[float], warnings: List[str],
) -> None:
    """n_matches and inliers.sum() are both guaranteed > 0 here -- main()
    SystemExits before calling this if --min-matches or MIN_INLIERS_TO_TRUST
    aren't met, so no n/a branches are needed (unlike register_pipeline's
    write_report, which isn't gated the same way for its DEPTH MAP section)."""
    n_inliers = int(inliers.sum())
    inlier_errors = errors[inliers]
    n_pixels = size_a[0] * size_a[1]

    def pct(count: int, denominator: int) -> str:
        return f"{count / denominator * 100:.1f}%" if denominator else "n/a"

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
    if ambiguous_depth is not None:
        lines.append(
            f"  ambiguous alternate depth: {ambiguous_depth:.4f} m "
            "(comparable inlier count to the fitted depth -- see Quality warnings)"
        )
    lines += [
        "",
        "COVERAGE",
        "-" * 66,
        f"  camera A pixels:              {n_pixels}",
        f"  outside shared field of view (post-fill): {n_outside_fov} "
        f"({pct(n_outside_fov, n_pixels)})",
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
        "  fit_result.json       machine-readable summary: fitted/coarse/ambiguous depth, "
        "match/inlier counts, outside-FOV pixels",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    refined_depth, coarse_depth, inliers, errors, ambiguous_depth = fit_depth(
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

    warped_color, remap_valid = compose_warped_output(
        refined_depth, color_b, camera_matrix_a, camera_matrix_b,
        extrinsics.R, extrinsics.T, size_a, size_b,
    )
    match_viz = render_match_visualization(color_a, color_b, pts_a, pts_b, inliers)
    n_outside_fov = int((~remap_valid).sum())

    cv2.imwrite(str(output_dir / "warped_features.jpg"), warped_color)
    cv2.imwrite(str(output_dir / "matches.jpg"), match_viz)
    save_preview(output_dir / "preview_features.jpg", color_a, warped_color, match_viz)

    fit_result = {
        "fitted_depth_m": float(refined_depth),
        "coarse_depth_m": float(coarse_depth),
        "ambiguous_alternate_depth_m": (
            float(ambiguous_depth) if ambiguous_depth is not None else None
        ),
        "n_matches": n_matches,
        "n_inliers": n_inliers,
        "outside_fov_pixels": n_outside_fov,
        "outside_fov_fraction": n_outside_fov / (size_a[0] * size_a[1]),
    }
    with (output_dir / "fit_result.json").open("w", encoding="utf-8") as f:
        json.dump(fit_result, f, indent=2)
        f.write("\n")

    d_critical = singular_depth(extrinsics.R, extrinsics.T)
    warnings: List[str] = []
    if np.isclose(coarse_depth, depths[0]) or np.isclose(coarse_depth, depths[-1]):
        warnings.append(
            f"Fitted depth landed on the edge of the search range ({depths[0]:.3f} or "
            f"{depths[-1]:.3f} m). Widen --depth-min/--depth-max."
        )
    if ambiguous_depth is not None:
        warnings.append(
            f"A competing plane at {ambiguous_depth:.4f} m had a comparable inlier count "
            f"to the fitted depth {refined_depth:.4f} m -- the single-plane assumption "
            "may be ambiguous for this scene. Inspect matches.jpg / preview_features.jpg, "
            "or narrow --depth-min/--depth-max if you know the true working range."
        )

    write_report(
        output_dir / "report_features.txt",
        session_dir, args.camera_a, args.camera_b, extrinsics_path, extrinsics, size_a,
        n_raw_matches, args.min_confidence, n_matches,
        depths, args.ransac_threshold,
        refined_depth, inliers, errors,
        default_depth, d_critical, n_outside_fov, ambiguous_depth, warnings,
    )

    if warnings:
        print("\nQuality warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print(f"\nSaved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
