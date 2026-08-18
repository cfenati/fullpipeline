#!/usr/bin/env python3
"""Register camera B onto camera A via plane-sweep depth-aware warping.

Replaces the old SGBM/rectification-based registration: at this rig's close range
(confirmed working distance 0.13-0.19 m) and small baseline with strong toe-in
(currently ~50 mm / ~17.5 deg), a single fixed-plane homography is only accurate
to <2 px within +/-0.1 mm of the reference depth, and rectifying the pair for dense
SGBM throws away most of each frame (only ~36%/35% of the rectified frames reach
real pixels even at alpha=1.0) because the convergence angle is too large. Plane
sweeping avoids both problems: it needs no rectification (correspondences are found
directly for images that contain the epipoles, which is exactly this pair's
situation), and by sweeping many candidate depths through the known working range it
recovers real per-pixel depth instead of assuming one plane for the whole frame.

Algorithm: for each of --steps candidate depths spanning [--depth-min, --depth-max],
compute the plane-induced homography straight from the calibrated stereo extrinsics,
warp camera B onto camera A's frame, score the match with ZNCC (robust to the
brightness/gain difference between the two cameras), and keep the best-scoring depth
per pixel (winner-take-all over a streamed running best/second-best, not a
materialized cost volume -- see sweep_cost_volume). Pixels whose winning match is
untrustworthy (textureless, ambiguous, or never valid across the whole sweep) fall
back to the calibrated reference-plane depth and are flagged in filled_mask, so
downstream numeric use can weight or exclude them instead of trusting a silent guess.

Expected runtime: plane sweeping is inherently O(steps * width * height) -- each
hypothesis does one cv2.remap plus a few cv2.boxFilter passes over the full frame.
At the defaults (60 steps, native 4656x3496) expect roughly tens of seconds to a
couple of minutes; pass --downscale 0.5 (or lower) for faster interactive iteration.

Example:
    python register_pipeline.py --session captures/hand
    python register_pipeline.py --session captures/hand --depth-min 0.12 --depth-max 0.20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_cameras import DEFAULT_OUTPUT_DIR, load_config, resolve_path  # noqa: E402
from calibration.stereo import StereoExtrinsics  # noqa: E402

DEFAULT_REGISTRATION_OUTPUT_DIR = "registration/results"
DEFAULT_DEPTH_MIN = 0.11
DEFAULT_DEPTH_MAX = 0.21
DEFAULT_STEPS = 60
DEFAULT_PATCH = 9
DEFAULT_MIN_SCORE = 0.5
DEFAULT_MIN_MARGIN = 0.05  # ZNCC is in [-1, 1]; 0.05 = 5% of the full range

# ZNCC scores are always in [-1, 1]; this sentinel is finite but unreachable by a
# real score, so a pixel whose best score is still this value never had a single
# valid (in-bounds, in-front-of-camera-B) hypothesis across the whole sweep.
INVALID_SCORE = -2.0
TEXTURELESS_STD_FLOOR = 2.0  # gray levels, 0-255 scale
ZNCC_EPS = 1e-6

CONFIDENT_FRACTION_WARN = 0.5  # warn if <50% of the cameras' overlap is confident
BOUNDARY_CLIP_WARN_FRACTION = 0.05  # warn if >=5% of confident pixels sit on the sweep edge

NORMAL_A = np.array([[0.0], [0.0], [1.0]])  # plane normal, fronto-parallel to camera A
PREVIEW_PANEL_WIDTH = 1000


# --------------------------------------------------------------------------- #
# CLI / config
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    reg_config = load_config().get("registration", {}) or {}
    depth_range = reg_config.get("depth_range", [DEFAULT_DEPTH_MIN, DEFAULT_DEPTH_MAX])

    parser = argparse.ArgumentParser(
        description="Register camera B onto camera A via plane-sweep depth-aware warping.",
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
                        help="Near edge of the swept depth range, metres "
                             "(default: registration.depth_range[0] in config, "
                             f"else {DEFAULT_DEPTH_MIN}).")
    parser.add_argument("--depth-max", type=float, default=depth_range[1],
                        help="Far edge of the swept depth range, metres "
                             "(default: registration.depth_range[1] in config, "
                             f"else {DEFAULT_DEPTH_MAX}).")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="Number of candidate depths, sampled uniformly in inverse "
                             "depth (default: %(default)s).")
    parser.add_argument("--patch", type=int, default=DEFAULT_PATCH,
                        help="ZNCC window size in pixels; must be odd (default: %(default)s).")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help="Minimum ZNCC score to trust a pixel's winning depth "
                             "(default: %(default)s).")
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN,
                        help="Minimum ZNCC gap between the best and second-best depth to "
                             "trust the winner, rather than call it ambiguous "
                             "(default: %(default)s).")
    parser.add_argument("--downscale", type=float, default=1.0,
                        help="Resize factor applied to the undistorted images before the "
                             "sweep, e.g. 0.5 for a faster/coarser pass (default: %(default)s).")
    parser.add_argument("--out", "--output", dest="output", default=None,
                        help="Output directory (default: registration.output_dir in config, "
                             f"else {DEFAULT_REGISTRATION_OUTPUT_DIR}) / <session name>.")
    return parser.parse_args()


def default_extrinsics_path(camera_a: str, camera_b: str) -> str:
    config = load_config().get("geometric_calibration", {}) or {}
    output_root = config.get("output_dir", DEFAULT_OUTPUT_DIR)
    key = f"extrinsics_{camera_a}_{camera_b}"
    return config.get(key, f"{output_root}/stereo_{camera_a}_{camera_b}/extrinsics.json")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_session_images(
    session_dir: Path, camera_a: str, camera_b: str
) -> Tuple[np.ndarray, np.ndarray]:
    if not session_dir.exists():
        raise SystemExit(f"Session directory not found: {session_dir}")

    paths = {camera: session_dir / f"{camera}.jpg" for camera in (camera_a, camera_b)}
    for camera, path in paths.items():
        if not path.exists():
            raise SystemExit(
                f"No {path.name} in {session_dir}.\n"
                "Expected the literal filename capture_pipeline.py writes for this camera."
            )

    image_a = cv2.imread(str(paths[camera_a]))
    image_b = cv2.imread(str(paths[camera_b]))
    if image_a is None or image_b is None:
        raise SystemExit(f"Failed to decode images in {session_dir}.")
    return image_a, image_b


def undistort_pair(
    image_a: np.ndarray, image_b: np.ndarray, extrinsics: StereoExtrinsics
) -> Tuple[np.ndarray, np.ndarray]:
    """Undistort with cv2.undistort(img, K, dist) -- no newCameraMatrix argument.

    Deliberately not CameraIntrinsics.undistort() (calibration/opencv_calibrate.py):
    that helper calls cv2.getOptimalNewCameraMatrix internally and returns only the
    corrected image, discarding the new K it computed. The plane-sweep homography
    below needs to know exactly which K applies to the undistorted images. Omitting
    newCameraMatrix makes cv2.undistort reuse the input K unchanged, so K_a/K_b loaded
    straight from extrinsics.json stay valid for these images with no extra bookkeeping.
    """
    size_a = (image_a.shape[1], image_a.shape[0])
    size_b = (image_b.shape[1], image_b.shape[0])
    if size_a != tuple(extrinsics.image_size_a):
        raise SystemExit(
            f"Camera A image is {size_a}, but the extrinsics were fit at "
            f"{tuple(extrinsics.image_size_a)}. Recalibrate or check --camera-a/--session."
        )
    if size_b != tuple(extrinsics.image_size_b):
        raise SystemExit(
            f"Camera B image is {size_b}, but the extrinsics were fit at "
            f"{tuple(extrinsics.image_size_b)}. Recalibrate or check --camera-b/--session."
        )

    undistorted_a = cv2.undistort(image_a, extrinsics.camera_matrix_a, extrinsics.distortion_a)
    undistorted_b = cv2.undistort(image_b, extrinsics.camera_matrix_b, extrinsics.distortion_b)
    return undistorted_a, undistorted_b


def scale_camera_matrix(camera_matrix: np.ndarray, scale: float) -> np.ndarray:
    scaling = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]])
    return scaling @ camera_matrix


def downscale_pair(
    image_a: np.ndarray, image_b: np.ndarray,
    camera_matrix_a: np.ndarray, camera_matrix_b: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if scale == 1.0:
        return image_a, image_b, camera_matrix_a, camera_matrix_b
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_a = cv2.resize(image_a, None, fx=scale, fy=scale, interpolation=interpolation)
    resized_b = cv2.resize(image_b, None, fx=scale, fy=scale, interpolation=interpolation)
    return (
        resized_a, resized_b,
        scale_camera_matrix(camera_matrix_a, scale),
        scale_camera_matrix(camera_matrix_b, scale),
    )


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def candidate_depths(depth_min: float, depth_max: float, steps: int) -> np.ndarray:
    """Depths sampled uniformly in inverse depth (1/d), ascending.

    H(d) depends on d only through 1/d, and the calibrated registration error is
    steeply nonlinear in depth (105 px at 0.157 m vs. 1759 px at 0.097 m for the same
    fixed plane, per calibration/results/stereo_rgb_cam1_rgb_cam2/report.txt) -- an
    inverse-depth grid spends hypotheses where sensitivity is high instead of
    oversampling the far end, the standard sampling choice for plane-sweep/MVS.
    """
    inverse_depths = np.linspace(1.0 / depth_max, 1.0 / depth_min, steps)
    return np.sort(1.0 / inverse_depths).astype(np.float64)


def plane_homography(
    camera_matrix_a_inv: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray, depth_m: float,
) -> np.ndarray:
    """3x3 homography mapping homogeneous pixels in undistorted A to undistorted B,
    for the plane fronto-parallel to camera A at depth_m (Z_a = depth_m).

    X_b = R @ X_a + T (this repo's convention, calibration/stereo.py), and for points
    on this plane n^T X_a = depth_m exactly (n = [0,0,1]^T), so
        T = (T @ n^T / depth_m) @ X_a
    exactly, giving X_b = (R + T @ n^T / depth_m) @ X_a. Composing with the pinhole
    projections and cancelling the common scale depth_m:
        H(depth_m) = K_b @ (R + T @ n^T / depth_m) @ inv(K_a)
    This is a PLUS, not the "R - t n^T/d" form in some references (e.g. Hartley &
    Zisserman) -- that form assumes the opposite plane-equation sign, n^T X = -d.
    Verified numerically against ground-truth reprojection using the real
    calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json: the plus form
    matches to ~1e-12 px, the minus form is off by thousands of px on the same test.
    """
    mixing = R + (T.reshape(3, 1) @ NORMAL_A.T) / depth_m
    return camera_matrix_b @ mixing @ camera_matrix_a_inv


def singular_depth(R: np.ndarray, T: np.ndarray) -> float:
    """Depth at which plane_homography's matrix becomes singular.

    det(R + T n^T/d) = 1 + n^T R^T T / d (matrix determinant lemma, det(R)=1,
    R^-1=R^T for a rotation), zero at d = -n^T R^T T. Reported so the sweep range can
    be checked against it -- for the current rig this comes out a couple of orders of
    magnitude below any usable working distance, i.e. nowhere near the sweep.
    """
    return float(-(R.T @ T.reshape(3, 1))[2, 0])


def pixel_grid(width: int, height: int) -> np.ndarray:
    """Homogeneous pixel-center coordinates for an HxW image, as a (3, H*W) array."""
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    ones = np.ones_like(u)
    return np.stack([u.ravel(), v.ravel(), ones.ravel()], axis=0)


def reproject_a_to_b(
    depth, camera_matrix_a_inv: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray, grid: np.ndarray,
    size_a: Tuple[int, int], size_b: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project every A pixel at `depth`, transform to B's frame, project into B.

    `depth` is a scalar (uniform-depth hypothesis) or an (H, W) array (a resolved
    per-pixel depth map); both broadcast against grid's (3, H*W) shape, so this one
    function serves both the per-hypothesis sweep and the final compose step -- there
    is no second, separately-maintained implementation of the same projection math.

    Returns (map_x, map_y) float32 ready for cv2.remap, and a bool `valid` mask that
    is False wherever the projected point falls outside B's frame OR behind camera B
    (a chirality check on the homogeneous w coordinate, not just frame bounds -- a
    point behind the camera can otherwise divide by a negative w and land back inside
    the frame bounds by coincidence, which a bounds-only test would miss).

    Computed in float64 throughout except the final cast: fx ~ 8600 and pixel
    coordinates up to ~4656 multiply to intermediate values beyond float32's exact-
    integer range (2**24), so doing the chain in float32 could leave ~1 px of
    avoidable noise in the sampled coordinates.
    """
    width_a, height_a = size_a
    width_b, height_b = size_b
    depth_flat = np.asarray(depth, dtype=np.float64).reshape(1, -1)

    points_a = camera_matrix_a_inv @ grid  # 3xHW, normalized ray directions
    points_a = points_a * depth_flat  # 3xHW, points at the hypothesized/resolved depth
    points_b = R @ points_a + T.reshape(3, 1)  # 3xHW, camera B frame
    projected = camera_matrix_b @ points_b  # 3xHW, unnormalized B-image coords

    w = projected[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        map_x = (projected[0] / w).reshape(height_a, width_a)
        map_y = (projected[1] / w).reshape(height_a, width_a)

    valid = (
        (w.reshape(height_a, width_a) > 1e-9)
        & (map_x >= 0) & (map_x <= width_b - 1)
        & (map_y >= 0) & (map_y <= height_b - 1)
    )
    map_x = np.nan_to_num(map_x, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)
    map_y = np.nan_to_num(map_y, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)
    return map_x, map_y, valid


def warp_hypothesis(
    gray_b: np.ndarray, depth_scalar: float,
    camera_matrix_a_inv: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray, grid: np.ndarray,
    size_a: Tuple[int, int], size_b: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    map_x, map_y, valid = reproject_a_to_b(
        depth_scalar, camera_matrix_a_inv, camera_matrix_b, R, T, grid, size_a, size_b
    )
    warped = cv2.remap(
        gray_b, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    return warped.astype(np.float32), valid


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def local_mean_std(image: np.ndarray, patch: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel local mean and std over a patch x patch box, via box filters --
    vectorized over the whole image, no Python pixel loop."""
    ksize = (patch, patch)
    mean = cv2.boxFilter(image, ddepth=-1, ksize=ksize, normalize=True)
    mean_sq = cv2.boxFilter(image * image, ddepth=-1, ksize=ksize, normalize=True)
    variance = np.clip(mean_sq - mean * mean, 0.0, None)  # float rounding can dip
                                                             # slightly negative
    return mean, np.sqrt(variance)


def zncc_score(
    gray_a: np.ndarray, mean_a: np.ndarray, std_a: np.ndarray,
    warped_b: np.ndarray, patch: int,
) -> np.ndarray:
    """Zero-mean normalized cross-correlation, robust to the brightness/gain offset
    between the two cameras (cam2 collects noticeably less light than cam1).

    A true ZNCC/correlation coefficient is always in [-1, 1] (Cauchy-Schwarz:
    |covariance| <= std_a * std_b), but that bound only holds under exact
    arithmetic. Near-flat patches (common wherever a camera's local contrast is
    low, e.g. skin without ridge lines, or just generally more of the frame when
    one camera runs lower-contrast than the other) push std_a * std_b toward the
    ZNCC_EPS floor, and float32 rounding noise in the box-filtered covariance no
    longer cancels cleanly against so small a denominator -- observed scores up to
    ~244 on real captures, not the theoretical +-1. Clipping after the divide is a
    strictly-correct safety net (any value outside [-1, 1] is provably numerical
    error, never real signal), not an approximation.
    """
    mean_b, std_b = local_mean_std(warped_b, patch)
    cross = cv2.boxFilter(gray_a * warped_b, ddepth=-1, ksize=(patch, patch), normalize=True)
    covariance = cross - mean_a * mean_b
    score = covariance / (std_a * std_b + ZNCC_EPS)
    return np.clip(score, -1.0, 1.0)


def sweep_cost_volume(
    gray_a: np.ndarray, gray_b: np.ndarray,
    camera_matrix_a: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray, depths: np.ndarray, patch: int,
) -> Dict[str, np.ndarray]:
    """Stream a running best/second-best score+depth per pixel across hypotheses,
    instead of materializing a full (steps, H, W) cost volume -- at defaults (60
    steps, native 4656x3496) that array would be ~3.9 GB, a real OOM risk. This keeps
    memory at O(H*W) regardless of step count.

    The top-2 update is exact, not approximate: before each hypothesis, the invariant
    best_score >= second_score >= every other score seen holds. If the new score beats
    best_score, then new > old_best >= old_second >= others, so old_best is provably
    still the correct new second-best.

    Hypotheses that fail the bounds/chirality check get the explicit INVALID_SCORE
    sentinel rather than being skipped -- a pixel where every hypothesis is invalid
    ends the loop with best_score == INVALID_SCORE, which is how "no hypothesis ever
    matched here" is detected explicitly, rather than silently defaulting to hypothesis
    0 (depth == --depth-min) the way an unguarded argmin over a stacked array would.
    """
    height_a, width_a = gray_a.shape
    height_b, width_b = gray_b.shape
    camera_matrix_a_inv = np.linalg.inv(camera_matrix_a)
    grid = pixel_grid(width_a, height_a)

    mean_a, std_a = local_mean_std(gray_a, patch)

    best_score = np.full((height_a, width_a), INVALID_SCORE, dtype=np.float32)
    second_score = np.full((height_a, width_a), INVALID_SCORE, dtype=np.float32)
    best_depth = np.full((height_a, width_a), np.nan, dtype=np.float64)
    any_valid = np.zeros((height_a, width_a), dtype=bool)

    for depth in depths:
        warped_b, valid = warp_hypothesis(
            gray_b, float(depth), camera_matrix_a_inv, camera_matrix_b, R, T, grid,
            (width_a, height_a), (width_b, height_b),
        )
        score = zncc_score(gray_a, mean_a, std_a, warped_b, patch)
        score = np.where(valid, score, INVALID_SCORE)

        any_valid |= valid
        improve = score > best_score
        second_score = np.where(improve, best_score, np.maximum(second_score, score))
        best_depth = np.where(improve, depth, best_depth)
        best_score = np.where(improve, score, best_score)

    return {
        "best_depth": best_depth,
        "best_score": best_score,
        "second_score": second_score,
        "std_a": std_a,
        "any_valid": any_valid,
    }


def resolve_depth_map(
    sweep: Dict[str, np.ndarray], min_score: float, min_margin: float, default_depth: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Combine four independent guards into one filled_mask, each alone sufficient to
    distrust a pixel's winning depth:
      - out_of_bounds: best score is still the sentinel (no hypothesis was ever valid)
      - textureless: camera A's local patch std is below a floor -- ZNCC is numerically
        meaningless on a flat patch regardless of its nominal score
      - low_score: best ZNCC below --min-score
      - ambiguous: margin between best and second-best score below --min-margin
        (periodic/repetitive texture -- the argmin pick would be arbitrary even though
        the absolute score looks fine)
    Any pixel failing a guard gets the calibrated reference-plane depth instead of its
    untrusted winning depth, and is flagged True in filled_mask.
    """
    best_score, second_score = sweep["best_score"], sweep["second_score"]
    best_depth, std_a, any_valid = sweep["best_depth"], sweep["std_a"], sweep["any_valid"]

    out_of_bounds = best_score <= INVALID_SCORE
    textureless = std_a < TEXTURELESS_STD_FLOOR
    low_score = best_score < min_score
    ambiguous = (best_score - second_score) < min_margin
    filled_mask = out_of_bounds | textureless | low_score | ambiguous

    depth_map = np.where(filled_mask, default_depth, best_depth)
    depth_map = np.nan_to_num(depth_map, nan=default_depth)

    reasons = {
        "out_of_bounds": out_of_bounds,
        "textureless": textureless,
        "low_score": low_score,
        "ambiguous": ambiguous,
        "any_valid": any_valid,
    }
    return depth_map, filled_mask, reasons


def compose_warped_output(
    depth_map: np.ndarray, color_b: np.ndarray,
    camera_matrix_a: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray,
    size_a: Tuple[int, int], size_b: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """One call to reproject_a_to_b with the *resolved* (post-fallback-fill) depth map
    gives the final dense warp directly -- the per-pixel formula and the per-hypothesis
    homography are the same math (reproject_a_to_b), so no separate warp pass at
    default_depth is needed for the filled pixels.

    remap_valid is False where even the resolved depth reprojects outside B's frame --
    a genuine "outside shared field of view" hole, a different cause from the matching
    guards in resolve_depth_map (that calls for reframing the shot; a matching guard
    calls for retuning thresholds), so it's reported separately.
    """
    width_a, height_a = size_a
    camera_matrix_a_inv = np.linalg.inv(camera_matrix_a)
    grid = pixel_grid(width_a, height_a)
    map_x, map_y, remap_valid = reproject_a_to_b(
        depth_map, camera_matrix_a_inv, camera_matrix_b, R, T, grid, size_a, size_b
    )
    warped_color = cv2.remap(
        color_b, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    return warped_color, remap_valid


def aggregate_quality_warnings(
    reasons: Dict[str, np.ndarray], best_depth: np.ndarray, depths: np.ndarray,
) -> List[str]:
    """Whole-volume sanity checks, distinguishing ordinary per-pixel texture noise
    from a systematic problem (wrong camera order, stale extrinsics, or a depth range
    that misses the actual subject) -- same tone as calibration/stereo.py's existing
    'Quality warnings' report sections."""
    warnings: List[str] = []
    overlap = reasons["any_valid"]
    n_overlap = int(overlap.sum())
    if n_overlap == 0:
        warnings.append(
            "No pixel in camera A ever reprojected inside camera B's frame across the "
            "whole sweep. Check --camera-a/--camera-b order and the extrinsics file."
        )
        return warnings

    confident = overlap & ~(
        reasons["out_of_bounds"] | reasons["textureless"]
        | reasons["low_score"] | reasons["ambiguous"]
    )
    confident_fraction = confident.sum() / n_overlap
    if confident_fraction < CONFIDENT_FRACTION_WARN:
        warnings.append(
            f"Only {confident_fraction * 100:.1f}% of the cameras' overlap region "
            f"scored confidently (< {CONFIDENT_FRACTION_WARN * 100:.0f}%). This points "
            "to a systematic problem (texture, exposure mismatch, wrong camera order, "
            "or a stale extrinsics file), not ordinary per-pixel noise -- check "
            "preview.jpg by eye."
        )

    n_confident = int(confident.sum())
    if n_confident > 0:
        boundary = confident & (
            np.isclose(best_depth, depths[0]) | np.isclose(best_depth, depths[-1])
        )
        boundary_fraction = boundary.sum() / n_confident
        if boundary_fraction >= BOUNDARY_CLIP_WARN_FRACTION:
            warnings.append(
                f"{boundary_fraction * 100:.1f}% of confident pixels landed exactly on "
                f"the swept depth range's edge ({depths[0]:.3f} or {depths[-1]:.3f} m). "
                "The true depth is likely outside --depth-min/--depth-max; widen the sweep."
            )
    return warnings


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_report(
    path: Path,
    session_dir: Path, camera_a: str, camera_b: str, extrinsics_path: Path,
    extrinsics: StereoExtrinsics, size_a: Tuple[int, int],
    depths: np.ndarray, patch: int, min_score: float, min_margin: float,
    default_depth: float, d_critical: float,
    reasons: Dict[str, np.ndarray], remap_valid: np.ndarray, depth_map: np.ndarray,
    warnings: List[str],
) -> None:
    overlap = reasons["any_valid"]
    n_pixels = size_a[0] * size_a[1]
    n_overlap = int(overlap.sum())
    confident = overlap & ~(
        reasons["out_of_bounds"] | reasons["textureless"]
        | reasons["low_score"] | reasons["ambiguous"]
    )
    n_confident = int(confident.sum())

    def pct(count: int, denominator: int) -> str:
        return f"{count / denominator * 100:.1f}%" if denominator else "n/a"

    lines = [
        f"Registration: {camera_b} -> {camera_a} (plane-sweep depth-aware warp)",
        "=" * 66,
        "",
        f"  session:            {session_dir}",
        f"  extrinsics:         {extrinsics_path}",
        f"  resolution:         {size_a[0]}x{size_a[1]}",
        f"  baseline:           {extrinsics.baseline_m * 1000:.2f} mm",
        f"  convergence:        {extrinsics.optical_axis_angle_deg:.2f} deg",
        "",
        "SWEEP",
        "-" * 66,
        f"  depth range:        {depths[0]:.4f} - {depths[-1]:.4f} m ({len(depths)} steps, "
        "uniform in 1/depth)",
        f"  patch:               {patch}px, min-score {min_score}, min-margin {min_margin}",
        f"  default (fallback) depth: {default_depth:.4f} m",
        f"  homography singular at:  {d_critical * 1000:.2f} mm "
        f"({'well clear of' if abs(d_critical) < depths[0] / 5 else 'CHECK: close to'} "
        "the swept range)",
        "",
        "COVERAGE",
        "-" * 66,
        f"  camera A pixels:              {n_pixels}",
        f"  in cameras' shared overlap:   {n_overlap} ({pct(n_overlap, n_pixels)})",
        f"  confident match:              {n_confident} ({pct(n_confident, n_overlap)} of overlap)",
        f"    out of bounds / behind cam: {int(reasons['out_of_bounds'].sum())}",
        f"    textureless:                {int(reasons['textureless'].sum())}",
        f"    low score:                  {int(reasons['low_score'].sum())}",
        f"    ambiguous (small margin):   {int(reasons['ambiguous'].sum())}",
        f"  outside shared field of view (post-fill): {int((~remap_valid).sum())} "
        f"({pct(int((~remap_valid).sum()), n_pixels)})",
        "",
        "DEPTH MAP (confident pixels only)",
        "-" * 66,
    ]
    if n_confident > 0:
        confident_depths = depth_map[confident]
        lines.append(
            f"  min / mean / max:   {confident_depths.min():.4f} / "
            f"{confident_depths.mean():.4f} / {confident_depths.max():.4f} m"
        )
    else:
        lines.append("  no confident pixels")

    if warnings:
        lines += ["", "Quality warnings", "-" * 66]
        lines += [f"  - {warning}" for warning in warnings]

    lines += [
        "",
        "OUTPUT FILES",
        "-" * 66,
        "  depth_m.npy / depth_m.png   per-pixel depth, metres",
        "  filled_mask.npy / .png      True where the reference-plane fallback was used",
        "  confidence.npy              raw ZNCC score per pixel",
        "  warped.jpg                  camera B warped into camera A's frame",
        "  preview.jpg                 camera A | warped B | depth, side by side",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    path: Path, color_a: np.ndarray, warped_color: np.ndarray, depth_map: np.ndarray,
    depth_min: float, depth_max: float,
) -> np.ndarray:
    normalized = np.clip((depth_map - depth_min) / (depth_max - depth_min), 0.0, 1.0)
    depth_colormap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    panels = [
        _resize_to_width(_labelled(color_a, f"camera A"), PREVIEW_PANEL_WIDTH),
        _resize_to_width(_labelled(warped_color, "B warped onto A"), PREVIEW_PANEL_WIDTH),
        _resize_to_width(_labelled(depth_colormap, "depth"), PREVIEW_PANEL_WIDTH),
    ]
    preview = np.hstack(panels)
    cv2.imwrite(str(path), preview)
    return depth_colormap


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    if args.patch < 3 or args.patch % 2 == 0:
        raise SystemExit(f"--patch must be odd and >= 3, got {args.patch}")
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
    gray_a = cv2.cvtColor(color_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(color_b, cv2.COLOR_BGR2GRAY).astype(np.float32)

    depths = candidate_depths(args.depth_min, args.depth_max, args.steps)
    print(
        f"Registering {args.camera_b} -> {args.camera_a}: sweeping {args.steps} depths "
        f"from {args.depth_min:.3f} to {args.depth_max:.3f} m at {size_a[0]}x{size_a[1]} "
        f"(patch {args.patch})"
    )

    sweep = sweep_cost_volume(
        gray_a, gray_b, camera_matrix_a, camera_matrix_b,
        extrinsics.R, extrinsics.T, depths, args.patch,
    )
    depth_map, filled_mask, reasons = resolve_depth_map(
        sweep, args.min_score, args.min_margin, default_depth
    )
    warped_color, remap_valid = compose_warped_output(
        depth_map, color_b, camera_matrix_a, camera_matrix_b,
        extrinsics.R, extrinsics.T, size_a, size_b,
    )
    warnings = aggregate_quality_warnings(reasons, sweep["best_depth"], depths)
    d_critical = singular_depth(extrinsics.R, extrinsics.T)

    np.save(output_dir / "depth_m.npy", depth_map.astype(np.float32))
    np.save(output_dir / "filled_mask.npy", filled_mask)
    np.save(output_dir / "confidence.npy", sweep["best_score"].astype(np.float32))
    cv2.imwrite(str(output_dir / "warped.jpg"), warped_color)
    cv2.imwrite(str(output_dir / "filled_mask.png"), (filled_mask * 255).astype(np.uint8))
    depth_colormap = save_preview(
        output_dir / "preview.jpg", color_a, warped_color, depth_map,
        args.depth_min, args.depth_max,
    )
    cv2.imwrite(str(output_dir / "depth_m.png"), depth_colormap)

    write_report(
        output_dir / "report.txt",
        session_dir, args.camera_a, args.camera_b, extrinsics_path, extrinsics, size_a,
        depths, args.patch, args.min_score, args.min_margin, default_depth, d_critical,
        reasons, remap_valid, depth_map, warnings,
    )

    overlap_count = int(reasons["any_valid"].sum())
    confident_count = int((
        reasons["any_valid"] & ~(
            reasons["out_of_bounds"] | reasons["textureless"]
            | reasons["low_score"] | reasons["ambiguous"]
        )
    ).sum())
    confident_pct = confident_count / overlap_count * 100 if overlap_count else 0.0
    print(f"Confident: {confident_pct:.1f}% of the cameras' overlap region")
    if warnings:
        print("\nQuality warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print(f"\nSaved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
