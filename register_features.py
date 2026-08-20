#!/usr/bin/env python3
"""Register camera B onto camera A via LightGlue sparse matching and a
piecewise-affine warp from those matches.

Sibling to register_pipeline.py, not a replacement: that script's dense
plane-sweep ZNCC correlation only scored 27-32% of the cameras' overlap
region confident on real captures (see registration/results/*/report.txt).
This script finds sparse correspondences with LightGlue and warps B onto A
by interpolating the 2-D correspondence field (Delaunay piecewise affine).
Matches are also triangulated through the calibrated extrinsics as a
sanity check (depths should land in the working range).

A single global homography -- 1-DOF plane or free 8-DOF -- cannot register
this subject. LightGlue matches on a close-range hand are geometrically
valid but span several centimetres of depth, so they do not lie on one
plane. One H either rejects most matches or shears the whole frame, and
the side-by-side preview then looks like the images never overlapped.
A piecewise affine from the matches is exact at every correspondence and
continuous across the convex hull, which is what actually overlays the
fingers. Outside the hull, pixels fall back to a calibrated median-depth
plane so the rest of the frame is still filled.

Algorithm:
    1. Load + undistort the pair (register_pipeline.py's own
       load_session_images/undistort_pair/downscale_pair).
    2. LightGlue match (kornia DISK + LightGlueMatcher) -> sparse (pts_a,
       pts_b) correspondences, filtered to --min-confidence.
    3. Triangulate with P_a = K_a [I|0], P_b = K_b [R|T]; keep points
       whose Z_a sits in the configured working range.
    4. Piecewise-affine warp: mesh the inlier matches plus synthetic anchor
       points at the frame perimeter (project_via_plane, see
       border_anchor_points -- pulls the mesh out to the frame edge instead
       of stopping wherever real matches thin out), Delaunay-interpolate
       pts_b as a function of pts_a and cv2.remap, after masking out
       degenerate triangles (huge area or thin slivers -- see
       degenerate_triangle_mask) so an outlier vertex can't smear a real
       depth discontinuity into a smooth, wrong blend. Uncovered pixels
       (outside the mesh, or inside a rejected triangle) blend from this
       warp into compose_warped_output's median-inlier-depth plane over a
       feathered transition band (feather_alpha) instead of a hard switch.

Memory: DISK on CPU is hungry per pixel. Native 4656x3496 (--downscale 1.0)
OOMs; the default --downscale 0.25 (1164x874) is the resolution the existing
captures were last matched at. Raise it only with enough free RAM.

Example:
    python register_features.py --session captures/hand
    python register_features.py --session captures/hand --downscale 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import cv2
import kornia.feature as KF
import matplotlib.tri as mtri
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_cameras import load_config, resolve_path  # noqa: E402
from calibration.stereo import StereoExtrinsics  # noqa: E402
from register_pipeline import (  # noqa: E402
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_REGISTRATION_OUTPUT_DIR,
    PREVIEW_PANEL_WIDTH,
    compose_warped_output,
    default_extrinsics_path,
    downscale_pair,
    load_session_images,
    plane_homography,
    undistort_pair,
)


DEFAULT_DOWNSCALE = 0.25  # DISK at native 4656x3496 OOMs on CPU; 0.25 is the
                          # resolution the existing captures were last matched at
                          # (1164x874). Raise --downscale only with enough RAM.
DEFAULT_MAX_KEYPOINTS = 4096
DEFAULT_MIN_MATCHES = 20
DEFAULT_MIN_CONFIDENCE = 0.3
MIN_INLIERS_TO_TRUST = 30
MAX_TRIANGLE_AREA_RATIO = 20.0  # reject Delaunay triangles bigger than N x the median
                                # triangle area -- a triangle this much larger than its
                                # neighbors usually means its vertices straddle a real
                                # depth/object discontinuity (e.g. a fingertip silhouette
                                # edge, where no match exists ON the edge itself), so
                                # linear interpolation across it draws a smooth blend
                                # over what should be a sharp boundary.
MAX_TRIANGLE_SLENDERNESS = 10.0  # reject triangles whose longest edge^2 / (2 * area)
                                  # exceeds this -- catches thin sliver triangles from
                                  # near-collinear vertices, including the zero-area
                                  # degenerate case (slenderness -> inf there).
BORDER_ANCHOR_SPACING_PX = 80  # spacing between synthetic anchor points placed around
                                # camera A's frame perimeter, working-resolution px.
                                # Dense enough to pull the mesh out to the frame edge
                                # without adding many extra triangles.
FEATHER_WIDTH_PX = 25  # match_warp's blend weight ramps from 0 at the covered/uncovered
                       # boundary to 1 this many px inside the covered region, instead
                       # of switching to plane_warp in one step.


# --------------------------------------------------------------------------- #
# CLI / config
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register camera B onto camera A via LightGlue sparse matching "
                    "and a piecewise-affine warp from those matches.",
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
        help="Stereo extrinsics JSON (undistortion + triangulation). Default: "
             "geometric_calibration.extrinsics_<a>_<b> in config, else "
             "calibration/results/stereo_<a>_<b>/extrinsics.json.",
    )
    parser.add_argument("--downscale", type=float, default=DEFAULT_DOWNSCALE,
                        help="Resize factor applied to the undistorted images before "
                             "feature extraction. Native ~4656x3496 OOMs DISK on CPU; "
                             "0.25 (1164x874) is the measured-safe default "
                             "(default: %(default)s).")
    parser.add_argument("--min-matches", type=int, default=DEFAULT_MIN_MATCHES,
                        help="Minimum matches above --min-confidence required to attempt "
                             "triangulation (default: %(default)s).")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                        help="Minimum LightGlue match confidence to keep, in [0, 1] "
                             "(default: %(default)s).")
    parser.add_argument(
        "--no-reject-degenerate-triangles", action="store_true",
        help="Keep huge-area/thin-sliver Delaunay triangles in the piecewise-affine warp "
             "instead of masking them out to the fallback plane (default: rejected). "
             f"Thresholds: area > {MAX_TRIANGLE_AREA_RATIO}x median, or "
             f"longest_edge^2/(2*area) > {MAX_TRIANGLE_SLENDERNESS}.",
    )
    parser.add_argument(
        "--no-border-anchors", action="store_true",
        help="Don't add synthetic frame-perimeter anchor points (projected through the "
             "calibrated fallback plane) to the piecewise-affine mesh. Default: added, "
             "so the mesh reaches the frame edge instead of stopping wherever real "
             f"matches happen to thin out (spacing: {BORDER_ANCHOR_SPACING_PX} px).",
    )
    parser.add_argument(
        "--no-feather-blend", action="store_true",
        help="Hard-switch between the piecewise-affine warp and the fallback plane at "
             "the covered/uncovered boundary instead of ramping smoothly across it "
             f"(default: feathered over {FEATHER_WIDTH_PX} px).",
    )
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
# Geometry
# --------------------------------------------------------------------------- #
def triangulate_matches(
    pts_a: np.ndarray, pts_b: np.ndarray,
    camera_matrix_a: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray,
    depth_min: float, depth_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate LightGlue matches into camera-A depth.

    P_a = K_a [I | 0], P_b = K_b [R | T] using this repo's X_b = R X_a + T
    convention. Returns (depths (N,), inliers (N,) bool) -- inliers are
    matches whose triangulated Z_a sits in (depth_min, depth_max).
    """
    projection_a = camera_matrix_a @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_b = camera_matrix_b @ np.hstack([R, T.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(
        projection_a, projection_b,
        pts_a.T.astype(np.float64), pts_b.T.astype(np.float64),
    )
    homogeneous /= homogeneous[3]
    depths = homogeneous[2]
    inliers = (depths > depth_min) & (depths < depth_max)
    return depths, inliers


def degenerate_triangle_mask(
    points: np.ndarray, triangles: np.ndarray,
    max_area_ratio: float, max_slenderness: float,
) -> np.ndarray:
    """Flag Delaunay triangles too large or too thin to trust for interpolation.

    area = 0.5 |cross(v1-v0, v2-v0)|. slenderness uses the fact that the
    altitude to a triangle's longest edge is always its shortest altitude
    (area is fixed, altitude ~ 1/base), so longest_edge / shortest_altitude
    reduces to longest_edge^2 / (2 * area) -- no need to compute all three
    altitudes. Equilateral triangles score ~1.15 (the minimum); it grows
    without bound as a triangle thins toward collinear, so the zero-area
    limit is naturally caught by the same threshold via the division guard.

    Returns a boolean mask aligned with `triangles`' rows, True = reject.
    """
    vertices = points[triangles]  # (M, 3, 2)
    edge_lengths = np.linalg.norm(
        vertices[:, [1, 2, 0], :] - vertices, axis=2
    )  # (M, 3): |v1-v0|, |v2-v1|, |v0-v2|
    longest_edge = edge_lengths.max(axis=1)
    e1 = vertices[:, 1] - vertices[:, 0]
    e2 = vertices[:, 2] - vertices[:, 0]
    cross_z = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]  # 2-D cross product (scalar)
    area = 0.5 * np.abs(cross_z)

    too_big = area > max_area_ratio * np.median(area)
    with np.errstate(divide="ignore", invalid="ignore"):
        slenderness = longest_edge**2 / (2.0 * area)
    too_thin = ~np.isfinite(slenderness) | (slenderness > max_slenderness)
    return too_big | too_thin


def build_mesh_interpolators(
    pts_a: np.ndarray, pts_b: np.ndarray, reject_degenerate: bool = True,
) -> Tuple["mtri.LinearTriInterpolator", "mtri.LinearTriInterpolator", int, int]:
    """Delaunay-triangulate pts_a and build linear interpolators for pts_b as a
    function of it (exact at every vertex), after masking out degenerate
    triangles (see degenerate_triangle_mask) unless reject_degenerate is
    False, so a single stretched-out triangle can't smear a real depth
    discontinuity into a smooth (wrong) blend.

    Split out of warp_with_match_field so other callers (e.g. a registration
    accuracy check evaluating the same mesh at specific query points, not a
    full pixel grid) can reuse the exact mesh main() builds instead of
    re-deriving it. Returns (interpolate_x, interpolate_y, n_rejected,
    n_triangles); n_rejected is 0 when reject_degenerate is False. Querying
    either interpolator outside a valid triangle returns a masked value.
    """
    triangulation = mtri.Triangulation(pts_a[:, 0], pts_a[:, 1])
    n_rejected = 0
    if reject_degenerate:
        bad_triangles = degenerate_triangle_mask(
            pts_a, triangulation.triangles, MAX_TRIANGLE_AREA_RATIO, MAX_TRIANGLE_SLENDERNESS,
        )
        triangulation.set_mask(bad_triangles)
        n_rejected = int(bad_triangles.sum())
    interpolate_x = mtri.LinearTriInterpolator(triangulation, pts_b[:, 0].astype(np.float64))
    interpolate_y = mtri.LinearTriInterpolator(triangulation, pts_b[:, 1].astype(np.float64))
    return interpolate_x, interpolate_y, n_rejected, len(triangulation.triangles)


def warp_with_match_field(
    pts_a: np.ndarray, pts_b: np.ndarray, color_b: np.ndarray, size_a: Tuple[int, int],
    reject_degenerate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Piecewise-affine warp of camera B onto camera A from sparse matches,
    via build_mesh_interpolators, sampled over the full pixel grid and
    remapped. Returns (warped, covered, n_rejected, n_triangles): covered is
    False outside the convex hull of the matches OR inside a rejected
    triangle -- those pixels have no trustworthy source location and the
    caller fills them with a calibrated plane warp.
    """
    width, height = size_a
    interpolate_x, interpolate_y, n_rejected, n_triangles = build_mesh_interpolators(
        pts_a, pts_b, reject_degenerate,
    )
    rows, cols = np.mgrid[0:height, 0:width]
    source_x = interpolate_x(cols, rows)
    source_y = interpolate_y(cols, rows)
    covered = ~np.ma.getmaskarray(source_x)
    map_x = np.ma.filled(source_x, -1.0).astype(np.float32)
    map_y = np.ma.filled(source_y, -1.0).astype(np.float32)
    warped = cv2.remap(
        color_b, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    return warped, covered, n_rejected, n_triangles


def border_anchor_points(size_a: Tuple[int, int], spacing_px: float) -> np.ndarray:
    """Points spaced around camera A's frame perimeter, corners included."""
    width, height = size_a
    n_x = max(2, int(round(width / spacing_px)) + 1)
    n_y = max(2, int(round(height / spacing_px)) + 1)
    top = np.stack([np.linspace(0, width - 1, n_x), np.zeros(n_x)], axis=1)
    bottom = np.stack([np.linspace(0, width - 1, n_x), np.full(n_x, height - 1)], axis=1)
    left = np.stack([np.zeros(n_y), np.linspace(0, height - 1, n_y)], axis=1)
    right = np.stack([np.full(n_y, width - 1), np.linspace(0, height - 1, n_y)], axis=1)
    return np.unique(np.concatenate([top, bottom, left, right]), axis=0)


def project_via_plane(
    points_a: np.ndarray, camera_matrix_a: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray, depth_m: float, size_b: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project camera-A points into camera B via the calibrated single-depth plane
    homography (register_pipeline.plane_homography). Used for synthetic border
    anchors, not real matches: gives the mesh a physically-grounded vertex to
    interpolate toward near the frame edge, instead of stopping wherever real
    matches happen to thin out. Same chirality-via-w + bounds check as
    register_pipeline.reproject_a_to_b. Returns (points_b, valid); valid is
    False where the projection lands outside B's frame or behind camera B --
    genuinely outside this rig's shared field of view, not a bug.
    """
    homography = plane_homography(np.linalg.inv(camera_matrix_a), camera_matrix_b, R, T, depth_m)
    homogeneous_a = np.concatenate([points_a, np.ones((len(points_a), 1))], axis=1).T  # (3, N)
    projected = homography @ homogeneous_a
    w = projected[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        points_b = (projected[:2] / w).T
    width_b, height_b = size_b
    valid = (
        (w > 1e-9)
        & (points_b[:, 0] >= 0) & (points_b[:, 0] <= width_b - 1)
        & (points_b[:, 1] >= 0) & (points_b[:, 1] <= height_b - 1)
    )
    points_b = np.nan_to_num(points_b, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return points_b, valid


def feather_alpha(covered: np.ndarray, width_px: int) -> np.ndarray:
    """Blend weight for match_warp vs. plane_warp: 0 right at the covered/uncovered
    boundary, ramping to 1 a full width_px inside the covered region.

    One-sided by design (always 0 outside `covered`, never partially blended
    in): match_warp's pixels outside the hull are meaningless remap fill
    (black, from cv2.remap's borderValue), not a smooth extrapolation of real
    content, so blending any nonzero weight of them in would draw a dark
    fringe just outside the boundary instead of removing the seam.
    """
    covered_u8 = covered.astype(np.uint8) * 255
    dist_inside = cv2.distanceTransform(covered_u8, cv2.DIST_L2, 5)
    return np.clip(dist_inside / width_px, 0.0, 1.0).astype(np.float32)


def match_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-channel histogram match so overlay panels are not dominated by the
    exposure/gain offset between the two cameras (cam2 collects less light)."""
    matched = source.copy()
    for channel in range(source.shape[2]):
        source_values, bin_index, source_counts = np.unique(
            source[:, :, channel].ravel(), return_inverse=True, return_counts=True
        )
        reference_values, reference_counts = np.unique(
            reference[:, :, channel].ravel(), return_counts=True
        )
        source_quantiles = np.cumsum(source_counts).astype(np.float64)
        source_quantiles /= source_quantiles[-1]
        reference_quantiles = np.cumsum(reference_counts).astype(np.float64)
        reference_quantiles /= reference_quantiles[-1]
        mapped = np.interp(source_quantiles, reference_quantiles, reference_values)
        matched[:, :, channel] = mapped[bin_index].reshape(source[:, :, channel].shape).astype(np.uint8)
    return matched


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


def render_checker(
    color_a: np.ndarray, warped: np.ndarray, covered: np.ndarray, n_cells: int = 12,
) -> np.ndarray:
    """Checkerboard of camera A and warped B, only inside the match hull.

    Outside the hull the warp is a fallback plane, not a correspondence, so
    including it in the checker makes the fingers look broken even when the
    matched surface overlapped. Uncovered pixels stay as camera A, dimmed.
    """
    height, width = color_a.shape[:2]
    cell = max(32, min(height, width) // n_cells)
    rows, cols = np.ogrid[:height, :width]
    board = ((cols // cell) + (rows // cell)) % 2 == 0
    checker = color_a.copy()
    checker[covered & board] = warped[covered & board]
    checker[~covered] = (color_a[~covered] // 2)
    return checker


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
    path: Path, color_a: np.ndarray, warped_color: np.ndarray, checker: np.ndarray,
) -> None:
    """camera A | warped B | checker overlay, side by side.

    The checker is the overlap diagnostic: if registration worked, finger
    edges and skin creases continue across square boundaries. A and warped
    B share size_a's aspect; pad anyway so a future layout change cannot
    make np.hstack raise.
    """
    panels = [
        _resize_to_width(_labelled(color_a, "camera A"), PREVIEW_PANEL_WIDTH),
        _resize_to_width(_labelled(warped_color, "B warped onto A"), PREVIEW_PANEL_WIDTH),
        _resize_to_width(_labelled(checker, "overlap (checker)"), PREVIEW_PANEL_WIDTH),
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
    size_a: Tuple[int, int],
    n_raw_matches: int, min_confidence: float, n_matches: int,
    n_inliers: int, depths: np.ndarray, inliers: np.ndarray,
    covered: np.ndarray, n_outside_fov: int,
    reject_degenerate: bool, n_rejected_triangles: int, n_triangles: int,
    use_border_anchors: bool, n_anchors: int, use_feather_blend: bool,
) -> None:
    inlier_depths = depths[inliers]
    n_pixels = size_a[0] * size_a[1]
    n_covered = int(covered.sum())

    def pct(count: int, denominator: int) -> str:
        return f"{count / denominator * 100:.1f}%" if denominator else "n/a"

    lines = [
        f"Registration: {camera_b} -> {camera_a} (LightGlue sparse match + piecewise-affine warp)",
        "=" * 66,
        "",
        f"  session:            {session_dir}",
        f"  extrinsics:         {extrinsics_path}",
        f"  resolution:         {size_a[0]}x{size_a[1]}",
        "",
        "MATCHING",
        "-" * 66,
        f"  raw LightGlue matches:      {n_raw_matches}",
        f"  above min-confidence {min_confidence}: {n_matches}",
        "",
        "TRIANGULATION",
        "-" * 66,
        f"  inliers (Z in working range): {n_inliers} / {n_matches} "
        f"({n_inliers / n_matches * 100:.1f}%)",
        f"  triangulated Z (inliers, m): min {inlier_depths.min():.4f}, "
        f"median {np.median(inlier_depths):.4f}, max {inlier_depths.max():.4f}",
        f"  match-hull coverage: {n_covered} ({pct(n_covered, n_pixels)}) "
        "-- uncovered pixels use a calibrated median-depth plane",
        (
            f"  degenerate triangles rejected: {n_rejected_triangles} / {n_triangles} "
            f"({pct(n_rejected_triangles, n_triangles)}) -- huge-area or thin-sliver, "
            "excluded from the piecewise warp"
        ) if reject_degenerate else (
            "  degenerate triangle rejection: disabled "
            "(--no-reject-degenerate-triangles)"
        ),
        (
            f"  border anchors: {n_anchors} synthetic points added at the frame "
            "perimeter (projected via the calibrated plane) so the mesh reaches "
            "the edge instead of stopping wherever real matches thin out"
        ) if use_border_anchors else (
            "  border anchors: disabled (--no-border-anchors)"
        ),
        (
            f"  covered/uncovered blend: feathered over {FEATHER_WIDTH_PX} px "
            "(match_warp -> plane_warp ramp, no hard seam)"
        ) if use_feather_blend else (
            "  covered/uncovered blend: hard switch (--no-feather-blend)"
        ),
        "",
        "COVERAGE",
        "-" * 66,
        f"  camera A pixels:              {n_pixels}",
        f"  outside shared field of view (post-fill): {n_outside_fov} "
        f"({pct(n_outside_fov, n_pixels)})",
        "",
        "OUTPUT FILES",
        "-" * 66,
        "  warped_features.jpg   camera B warped onto camera A (piecewise affine from matches)",
        "  overlay_checker.jpg   checkerboard of camera A vs warped B (overlap diagnostic)",
        "  overlay_blend.jpg     50/50 blend of camera A and warped B",
        "  matches.jpg           inlier (green) / outlier (red) correspondence lines",
        "  preview_features.jpg  camera A | warped B | checker overlay, side by side",
        "  fit_result.json       match/inlier counts, depth stats, outside-FOV pixels",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()

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
    reg_config = load_config().get("registration", {}) or {}
    output_root = args.output or reg_config.get("output_dir", DEFAULT_REGISTRATION_OUTPUT_DIR)
    output_dir = resolve_path(output_root) / session_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_range = reg_config.get("depth_range", [DEFAULT_DEPTH_MIN, DEFAULT_DEPTH_MAX])
    depth_min, depth_max = float(depth_range[0]), float(depth_range[1])

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

    depths, inliers = triangulate_matches(
        pts_a, pts_b, camera_matrix_a, camera_matrix_b,
        extrinsics.R, extrinsics.T, depth_min, depth_max,
    )
    n_inliers = int(inliers.sum())
    if n_inliers > 0:
        inlier_depths = depths[inliers]
        print(
            f"Triangulated {n_inliers}/{n_matches} matches, "
            f"Z median {np.median(inlier_depths):.4f} m "
            f"[{inlier_depths.min():.4f}, {inlier_depths.max():.4f}]"
        )
    if n_inliers < MIN_INLIERS_TO_TRUST:
        raise SystemExit(
            f"Only {n_inliers} matches triangulated inside {depth_min:.3f}-{depth_max:.3f} m "
            f"(need >= {MIN_INLIERS_TO_TRUST}). Check calibration, camera order, or capture "
            "a more textured scene."
        )

    reject_degenerate = not args.no_reject_degenerate_triangles
    use_border_anchors = not args.no_border_anchors
    use_feather_blend = not args.no_feather_blend
    fill_depth = float(np.median(depths[inliers]))

    mesh_pts_a, mesh_pts_b = pts_a[inliers], pts_b[inliers]
    n_anchors = 0
    if use_border_anchors:
        candidate_anchors_a = border_anchor_points(size_a, BORDER_ANCHOR_SPACING_PX)
        anchors_b, anchors_valid = project_via_plane(
            candidate_anchors_a, camera_matrix_a, camera_matrix_b,
            extrinsics.R, extrinsics.T, fill_depth, size_b,
        )
        anchors_a = candidate_anchors_a[anchors_valid]
        anchors_b = anchors_b[anchors_valid]
        n_anchors = len(anchors_a)
        mesh_pts_a = np.concatenate([mesh_pts_a, anchors_a])
        mesh_pts_b = np.concatenate([mesh_pts_b, anchors_b])

    match_warp, covered, n_rejected_triangles, n_triangles = warp_with_match_field(
        mesh_pts_a, mesh_pts_b, color_b, size_a, reject_degenerate,
    )
    plane_depth = np.full((size_a[1], size_a[0]), fill_depth, dtype=np.float64)
    plane_warp, remap_valid = compose_warped_output(
        plane_depth, color_b, camera_matrix_a, camera_matrix_b,
        extrinsics.R, extrinsics.T, size_a, size_b,
    )
    if use_feather_blend:
        alpha = feather_alpha(covered, FEATHER_WIDTH_PX)[:, :, None]
        warped_color = (
            alpha * match_warp.astype(np.float32) + (1 - alpha) * plane_warp.astype(np.float32)
        ).astype(np.uint8)
    else:
        warped_color = np.where(covered[:, :, None], match_warp, plane_warp)
    n_outside_fov = int((~remap_valid & ~covered).sum())
    rejection_note = (
        f"{n_rejected_triangles}/{n_triangles} degenerate triangles rejected"
        if reject_degenerate else "degenerate triangle rejection disabled"
    )
    anchor_note = f"{n_anchors} border anchors" if use_border_anchors else "border anchors disabled"
    feather_note = f"feathered {FEATHER_WIDTH_PX}px" if use_feather_blend else "hard switch"
    print(
        f"Piecewise-affine warp over {covered.mean() * 100:.1f}% of the frame "
        f"({rejection_note}; {anchor_note}; {feather_note}; "
        f"uncovered pixels use a {fill_depth:.4f} m plane)"
    )

    match_viz = render_match_visualization(color_a, color_b, pts_a, pts_b, inliers)
    overlay_source = match_histogram(warped_color, color_a)
    checker = render_checker(color_a, overlay_source, covered)
    blend = color_a.copy()
    blend[covered] = (
        0.5 * color_a[covered].astype(np.float32) + 0.5 * overlay_source[covered].astype(np.float32)
    ).astype(np.uint8)

    cv2.imwrite(str(output_dir / "warped_features.jpg"), warped_color)
    cv2.imwrite(str(output_dir / "overlay_checker.jpg"), checker)
    cv2.imwrite(str(output_dir / "overlay_blend.jpg"), blend)
    cv2.imwrite(str(output_dir / "matches.jpg"), match_viz)
    save_preview(output_dir / "preview_features.jpg", color_a, warped_color, checker)

    fit_result = {
        "n_matches": n_matches,
        "n_inliers": n_inliers,
        "depth_min_m": float(depths[inliers].min()),
        "depth_median_m": fill_depth,
        "depth_max_m": float(depths[inliers].max()),
        "interpolated_coverage": float(covered.mean()),
        "reject_degenerate_triangles": reject_degenerate,
        "triangles_total": n_triangles,
        "triangles_rejected": n_rejected_triangles,
        "border_anchors_used": use_border_anchors,
        "n_border_anchors": n_anchors,
        "feather_blend_used": use_feather_blend,
        "outside_fov_pixels": n_outside_fov,
        "outside_fov_fraction": n_outside_fov / (size_a[0] * size_a[1]),
    }
    with (output_dir / "fit_result.json").open("w", encoding="utf-8") as f:
        json.dump(fit_result, f, indent=2)
        f.write("\n")

    write_report(
        output_dir / "report_features.txt",
        session_dir, args.camera_a, args.camera_b, extrinsics_path, size_a,
        n_raw_matches, args.min_confidence, n_matches,
        n_inliers, depths, inliers, covered, n_outside_fov,
        reject_degenerate, n_rejected_triangles, n_triangles,
        use_border_anchors, n_anchors, use_feather_blend,
    )

    print(f"\nSaved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
