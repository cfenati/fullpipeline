#!/usr/bin/env python3
"""Register camera B onto camera A via sparse matches + triangulation.

Every Z comes from a correspondence and the calibrated K/R/T -- see
triangulate.py (DLT written out). There is no assumed plane depth. Overlay
is a piecewise-affine warp exact at every triangulated match; pixels
outside the match hull have no correspondence, so they have no Z and stay
unfilled.

    1. Load + undistort the pair.
    2. Match (--matcher disk / loftr / raft / combinations) -> (pts_a, pts_b).
    3. Triangulate with P_a = K_a [I|0], P_b = K_b [R|T]; keep points whose
       Z_a sits in the working-range chirality filter.
    4. Delaunay-interpolate the 2-D correspondences (the projection of that
       3-D mesh) and remap. Degenerate triangles can be masked out.

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
from scipy.interpolate import NearestNDInterpolator
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_cameras import load_config, resolve_path  # noqa: E402
from calibration.stereo import StereoExtrinsics  # noqa: E402
from registration_io import (  # noqa: E402
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_REGISTRATION_OUTPUT_DIR,
    PREVIEW_PANEL_WIDTH,
    default_extrinsics_path,
    downscale_pair,
    labelled,
    load_session_images,
    resize_to_width,
    undistort_pair,
)
from triangulate import (  # noqa: E402
    projection_matrices,
    reproject_via_depth,
    triangulate_dlt,
    triangulate_points as _triangulate_points,
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
MAX_TRIANGLE_DEPTH_RANGE_M = 0.003  # reject Delaunay triangles whose 3 vertices' triangulated
                                    # Z spans more than this many metres. The area/slenderness
                                    # checks above only see the *projected* 2-D triangle, so a
                                    # triangle can pass both (ordinary size, not a sliver) while
                                    # still bridging a real depth discontinuity, if matches
                                    # happen to land densely on both sides of it -- exactly what
                                    # bends a physically straight edge into a curved one in the
                                    # warp. A first guess of 20mm (20% of the config depth_range
                                    # window) turned out to be a no-op on a real hand capture
                                    # (captures/20260818_152114_103039): that scene's whole
                                    # triangulated Z only spans ~30mm end to end, so no triangle
                                    # ever reached 20mm. A sweep on that capture (0 rejected at
                                    # 20mm, 4 at 10mm, 43 at 5mm, 85 at 3mm -- new gaps land at
                                    # finger-valley creases, real discontinuities -- 258 at 2mm
                                    # already speckles flat, non-edge finger surface, 1588 at 1mm
                                    # visibly shreds it) put the knee between 3mm and 2mm; 3mm is
                                    # the current default. Only validated on that one capture --
                                    # revisit on other scenes/rigs before trusting it further.
KEYPOINT_TILE_SIZE_PX = 300  # target tile size for extract_features_tiled, working-
                              # resolution px -- ~300px gives a 4x3 grid at the default
                              # 1164x874 resolution, small enough to force keypoints into
                              # low-texture tiles without shrinking each tile's own budget
                              # to noise.
GEOMETRIC_CONSISTENCY_NEIGHBORS = 8  # how many spatially-nearest matches (in camera-A
                                     # pixel space) each match's displacement is compared
                                     # against in geometric_consistency_mask.
GEOMETRIC_CONSISTENCY_THRESHOLD_PX = 15.0  # reject a match whose displacement deviates
                                            # from its neighbors' median by more than this
                                            # many px, working resolution. Calibrated
                                            # empirically against this rig's observed
                                            # depth-driven displacement variation -- see
                                            # geometric_consistency_mask.
RAFT_GRID_STEP_PX = 12  # match_raft samples dense flow on a grid this many px apart,
                        # working resolution -- dense enough for a useful mesh without
                        # an unreasonable number of triangulate_matches calls.
RAFT_FB_CONSISTENCY_THRESHOLD_PX = 1.5  # match_raft's forward-backward consistency
                                         # error is bimodal, not gradual: correctly-
                                         # tracked points measured ~0.2-0.9px round-trip
                                         # error on this rig, points RAFT lost track of
                                         # (aperture problem on repetitive/low-texture
                                         # regions) jump to 20-200+px with no gray zone
                                         # in between -- this threshold just needs to sit
                                         # in that gap, not be finely tuned.


# --------------------------------------------------------------------------- #
# CLI / config
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register camera B onto camera A via sparse feature matching "
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
             "instead of leaving those pixels unfilled (default: rejected). "
             f"Thresholds: area > {MAX_TRIANGLE_AREA_RATIO}x median, or "
             f"longest_edge^2/(2*area) > {MAX_TRIANGLE_SLENDERNESS}.",
    )
    parser.add_argument(
        "--no-reject-depth-discontinuous-triangles", action="store_true",
        help="Keep Delaunay triangles whose 3 vertices span a large triangulated depth "
             "range in the piecewise-affine warp instead of leaving those pixels unfilled "
             "(default: rejected). Complements --no-reject-degenerate-triangles: that check "
             "only looks at the projected 2-D triangle shape, this looks at actual "
             "triangulated Z, so it catches a triangle that looks like an ordinary size/shape "
             "in camera-A pixels but still bridges a real depth discontinuity -- the usual "
             f"cause of a physically straight edge warping crooked. Threshold: "
             f"{MAX_TRIANGLE_DEPTH_RANGE_M * 1000:.0f}mm across the 3 vertices, tune with "
             "--max-triangle-depth-range.",
    )
    parser.add_argument(
        "--max-triangle-depth-range", type=float, default=MAX_TRIANGLE_DEPTH_RANGE_M,
        help="Depth-discontinuity threshold in metres for "
             "--no-reject-depth-discontinuous-triangles (default: %(default)s).",
    )
    parser.add_argument(
        "--no-depth-extrapolation", action="store_true",
        help="Leave pixels outside the piecewise-affine match hull unfilled instead of "
             "estimating them (default: filled). The default estimates each unfilled "
             "pixel's depth from its nearest triangulated match (NearestNDInterpolator -- "
             "no smoothness/planarity assumption about the scene beyond the hull) and "
             "reprojects through the calibrated K_a/K_b/R/T directly (see "
             "triangulate.reproject_via_depth) -- no feature match needed there at all. "
             "This is a geometry-based depth guess, not a real correspondence, so it's "
             "reported and tinted separately from the matched-hull region in "
             "report_features.txt/fit_result.json and overlay_checker.jpg/preview_features.jpg.",
    )
    parser.add_argument(
        "--match-histogram", action="store_true",
        help="Per-channel histogram-match warped B to camera A (see match_histogram) before "
             "compositing overlay_checker.jpg/overlay_blend.jpg. Opt-in, not default: on "
             "visual inspection the histogram-matched composite looked worse than compositing "
             "the raw warp directly (2026-08-26) -- exposure/gain differences between the two "
             "cameras stay visible without it, but per-channel matching can introduce its own "
             "artifacts (banding, color casts). warped_features.jpg (the raw warp, always "
             "saved) is unaffected either way.",
    )
    parser.add_argument(
        "--no-geometric-consistency-filter", action="store_true",
        help="Keep matches whose displacement disagrees with their spatial neighbors' "
             "(see geometric_consistency_mask) instead of dropping them before "
             "triangulation. Default: filtered, "
             f"threshold {GEOMETRIC_CONSISTENCY_THRESHOLD_PX} px vs. the median of the "
             f"nearest {GEOMETRIC_CONSISTENCY_NEIGHBORS} neighbors.",
    )
    parser.add_argument(
        "--tiled-keypoints", action="store_true",
        help="Extract DISK keypoints per-tile (grid cells, each with its own budget) "
             "instead of one global top-K, so low-texture regions "
             "(palm, background) aren't starved of keypoints by high-texture ones. "
             "Opt-in, not default: measured on held-out ChArUco corners "
             "(check_registration_error.py), tiling raised match-hull coverage but "
             "regressed accuracy (median 0.66->0.77px, p90 1.32->2.22px, "
             "max 3.51->14.20px) by thinning keypoint density on already-textured "
             f"regions -- default tile size if enabled: {KEYPOINT_TILE_SIZE_PX} px.",
    )
    parser.add_argument(
        "--matcher", nargs="+", choices=("disk", "loftr", "raft"), default=["disk"],
        help="Which matcher(s) produce the sparse correspondences the mesh is built "
             "from -- one or more, pooled if more than one (default: %(default)s). "
             "'loftr' is opt-in, not recommended: kornia LoFTR (outdoor-pretrained) "
             "covers ~84%% of the frame vs. DISK+LightGlue's ~45%% and every match "
             "triangulates to a plausible depth, but measured on held-out ChArUco "
             "corners (check_registration_error.py) its point localization is far "
             "worse -- median error 0.66px (disk) vs. 2.97px (loftr) vs. 1.03px "
             "(disk+loftr pooled, which doesn't fully recover disk's precision), "
             "p90/max blowing out to 52.66/104.78px for loftr alone. 'raft' "
             "(torchvision RAFT-large, dense optical flow + forward-backward "
             "consistency filtering, see match_raft) alone is a mixed bag: median "
             "0.62px (competitive with disk) but p90/max 10.46/23.47px -- some "
             "sessions' flow gets fooled into a periodic-but-wrong lock (plausible "
             "on a repetitive ChArUco pattern specifically). 'disk raft' pooled is "
             "the one combination measured better than disk alone on every "
             "aggregate: median 0.56px, p90 1.18px (both improved), max 6.76px "
             "(worse than disk's 3.51px but far better than raft alone) -- disk's "
             "precision anchors exactly the sessions raft's periodic-lock hurt. Not "
             "the default: real extra cost (a second model, ~8s/image-pair on CPU, "
             "the torchvision dependency) for a currently sub-pixel-already metric "
             "to improve further, and 'disk raft' is only validated on this rig's "
             "ChArUco set, not yet on the actual hand-capture use case.",
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


def extract_features_tiled(
    disk: KF.DISK, image_bgr: np.ndarray, device: torch.device,
    max_keypoints: int, tile_size_px: float,
) -> Tuple["KF.DISKFeatures", Tuple[int, int]]:
    """DISK keypoints + descriptors extracted per-tile and merged, instead of
    one global top-max_keypoints budget over the whole image.

    DISK's own top-K selection concentrates on the strongest local response
    (knuckles, nail edges on a hand) and can starve flat regions (palm,
    background) of any keypoints at all even when max_keypoints is generous
    -- that caps how far warp_with_match_field's mesh can ever reach,
    regardless of degenerate-triangle rejection, since there's simply
    nothing to triangulate there. Splitting the
    image into a grid of ~tile_size_px tiles and giving each an equal
    keypoint budget forces spatial spread instead.

    Non-overlapping tiles: a real feature straddling a tile boundary can be
    missed by both tiles, but LightGlue matching doesn't need every feature,
    just enough spread -- not worth the complexity of overlapping tiles with
    duplicate-keypoint suppression for this file's purposes.

    Returns (features, (H, W)) like extract_features -- (H, W) is the full
    image's shape (not any one tile's); keypoints are offset back into
    full-image coordinates before returning, so this is a drop-in return
    value for match_features' hw1/hw2.
    """
    height, width = image_bgr.shape[:2]
    cols = max(1, round(width / tile_size_px))
    rows = max(1, round(height / tile_size_px))
    budget_per_tile = max(1, max_keypoints // (cols * rows))

    keypoints, descriptors, scores = [], [], []
    for row in range(rows):
        y0, y1 = int(round(row * height / rows)), int(round((row + 1) * height / rows))
        for col in range(cols):
            x0, x1 = int(round(col * width / cols)), int(round((col + 1) * width / cols))
            tile_features, _ = extract_features(disk, image_bgr[y0:y1, x0:x1], device, budget_per_tile)
            if tile_features.n == 0:
                continue
            offset = torch.tensor(
                [x0, y0], device=tile_features.keypoints.device, dtype=tile_features.keypoints.dtype,
            )
            keypoints.append(tile_features.keypoints + offset)
            descriptors.append(tile_features.descriptors)
            scores.append(tile_features.detection_scores)

    merged = KF.DISKFeatures(
        torch.cat(keypoints, dim=0), torch.cat(descriptors, dim=0), torch.cat(scores, dim=0),
    )
    return merged, (height, width)


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


def load_loftr_model(device: torch.device) -> KF.LoFTR:
    """Load the pretrained LoFTR matcher once.

    Detector-free, unlike DISK+LightGlue: LoFTR matches an image pair
    directly with no separate keypoint-extraction step, which lets it find
    correspondence in low-texture regions a keypoint detector never fires on
    at all (a mottled background, smooth skin) -- see this file's git history
    for the probe that motivated adding it: on a hand capture, LoFTR found
    2.8x the matches of DISK+LightGlue, covering ~84% of the frame's convex
    hull area vs ~45%, with 100% of matches triangulating to a physically
    plausible depth. Weights auto-download from kornia's model hub on first
    call (needs internet once; cached locally after that).
    """
    return KF.LoFTR(pretrained="outdoor").to(device).eval()


def match_loftr(
    loftr: KF.LoFTR, color_a: np.ndarray, color_b: np.ndarray, device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LoFTR-match two BGR images directly (dense/semi-dense, grayscale input).

    Returns (pts_a, pts_b, scores) for every match kornia returned --
    unfiltered by confidence, matching match_features' convention (the
    caller applies --min-confidence uniformly, whether matches came from
    LoFTR or DISK+LightGlue -- the two scores aren't independently
    calibrated to mean exactly the same thing, but both are nominally
    dual-softmax-style confidence in [0, 1] with higher better, close enough
    for one shared threshold). `scores` is LoFTR's own confidence.
    """
    def to_gray_tensor(image_bgr: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(gray).to(device).float()[None, None] / 255.0

    with torch.no_grad():
        result = loftr({"image0": to_gray_tensor(color_a), "image1": to_gray_tensor(color_b)})
    pts_a = result["keypoints0"].cpu().numpy()
    pts_b = result["keypoints1"].cpu().numpy()
    scores = result["confidence"].cpu().numpy()
    return pts_a, pts_b, scores


def load_raft_model(device: torch.device) -> torch.nn.Module:
    """Load the pretrained RAFT-large optical flow model once.

    Weights auto-download from torchvision's model hub on first call (needs
    internet once; cached locally after that).
    """
    return raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()


def match_raft(
    model: torch.nn.Module, color_a: np.ndarray, color_b: np.ndarray, device: torch.device,
    grid_step: float, fb_threshold_px: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense optical flow (RAFT) sampled on a grid, filtered by forward-backward
    consistency, as a third matching source alongside DISK+LightGlue/LoFTR.

    Unlike either of those, RAFT gives no sparse "confidence" at all -- it's
    a dense per-pixel flow field. This runs flow in both directions (A->B and
    B->A) and, for each grid point, warps forward then back through the
    reverse flow: a correct estimate returns you to (near) where you started,
    a wrong one (RAFT lost track -- the aperture problem, typically on
    repetitive/low-texture regions like this rig's mottled backdrop) does
    not, and does not by a lot (empirically bimodal on this rig: correct
    points round-trip within ~1px, incorrect ones are off by tens to
    hundreds of px -- see RAFT_FB_CONSISTENCY_THRESHOLD_PX). Returns
    (pts_a, pts_b, scores) matching the other matchers' convention: scores
    is 1 - fb_error/fb_threshold_px clipped to [0, 1], so the caller's
    existing --min-confidence gate filters consistently with DISK/LoFTR
    scores instead of needing special-cased handling.
    """
    height, width = color_a.shape[:2]
    pad_h, pad_w = (-height) % 8, (-width) % 8

    def to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).unsqueeze(0)
        return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))

    weights = Raft_Large_Weights.DEFAULT
    img_a, img_b = weights.transforms()(to_tensor(color_a), to_tensor(color_b))
    with torch.no_grad():
        flow_fwd = model(img_a, img_b)[-1][0, :, :height, :width].permute(1, 2, 0).cpu().numpy()
        flow_bwd = model(img_b, img_a)[-1][0, :, :height, :width].permute(1, 2, 0).cpu().numpy()

    rows, cols = np.mgrid[0:height:grid_step, 0:width:grid_step]
    pts_a = np.stack([cols.ravel(), rows.ravel()], axis=1).astype(np.float32)
    forward = flow_fwd[rows, cols].reshape(-1, 2)
    pts_b = pts_a + forward

    map_x = pts_b[:, 0].reshape(1, -1)
    map_y = pts_b[:, 1].reshape(1, -1)
    backward_x = cv2.remap(flow_bwd[:, :, 0], map_x, map_y, interpolation=cv2.INTER_LINEAR).ravel()
    backward_y = cv2.remap(flow_bwd[:, :, 1], map_x, map_y, interpolation=cv2.INTER_LINEAR).ravel()
    recovered_a = pts_b + np.stack([backward_x, backward_y], axis=1)
    fb_error = np.linalg.norm(recovered_a - pts_a, axis=1)

    scores = np.clip(1.0 - fb_error / fb_threshold_px, 0.0, 1.0)
    return pts_a, pts_b, scores


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def geometric_consistency_mask(
    pts_a: np.ndarray, pts_b: np.ndarray, k_neighbors: int, threshold_px: float,
) -> np.ndarray:
    """Flag matches whose displacement (pts_b - pts_a) disagrees with its
    spatial neighbors' displacement in camera A, before triangulation ever
    sees them.

    A correct match on a locally smooth surface moves similarly to its
    nearby matches (parallax varies smoothly with depth, except right at a
    real depth discontinuity); a mismatched point's displacement is
    essentially arbitrary relative to its neighbors. Complements
    degenerate_triangle_mask, which only catches a bad match once it's
    already stretched a Delaunay triangle -- this catches it earlier, before
    it can pull a neighboring good match's triangle out of shape too.

    O(N^2) pairwise distance matrix -- fine at LightGlue's match counts here
    (thousands, not millions); would need a KD-tree at a larger scale.

    Returns a boolean mask aligned with pts_a's rows, True = keep.
    """
    n = len(pts_a)
    k = min(k_neighbors, n - 1)
    if k < 1:
        return np.ones(n, dtype=bool)

    displacement = pts_b - pts_a
    diff = pts_a[:, None, :] - pts_a[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    neighbor_idx = np.argpartition(dist_sq, k - 1, axis=1)[:, :k]
    local_median = np.median(displacement[neighbor_idx], axis=1)
    deviation = np.linalg.norm(displacement - local_median, axis=1)
    return deviation <= threshold_px


def triangulate_matches(
    pts_a: np.ndarray, pts_b: np.ndarray,
    camera_matrix_a: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray,
    depth_min: float, depth_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate matches; inliers are those with Z_a in (depth_min, depth_max).

    See triangulate.py for the DLT. The depth window is a chirality / working-
    range filter, not an assumed plane.
    """
    points_3d = _triangulate_points(pts_a, pts_b, camera_matrix_a, camera_matrix_b, R, T)
    depths = points_3d[:, 2]
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


def depth_discontinuous_triangle_mask(
    triangles: np.ndarray, vertex_depths: np.ndarray, max_depth_range: float,
) -> np.ndarray:
    """Flag Delaunay triangles whose 3 vertices span more real depth (Z, from
    triangulation) than max_depth_range.

    degenerate_triangle_mask only sees the *projected* 2-D triangle in
    camera-A pixel space -- a triangle can be an ordinary size and shape
    there (not huge, not a sliver) while still bridging a real depth
    discontinuity, if matches happen to land densely on both sides of the
    edge causing it. Linear interpolation across that big a Z jump is what
    turns a physically straight edge into a visibly bent one in the warp;
    this catches it directly from the depth these matches already
    triangulated to, rather than via a 2-D shape proxy.

    `vertex_depths` must be aligned with the points array `triangles`
    indexes into (i.e. the same pts_a passed to build_mesh_interpolators).
    Returns a boolean mask aligned with `triangles`' rows, True = reject.
    """
    vertex_z = vertex_depths[triangles]  # (M, 3)
    depth_range = vertex_z.max(axis=1) - vertex_z.min(axis=1)
    return depth_range > max_depth_range


def build_mesh_interpolators(
    pts_a: np.ndarray, pts_b: np.ndarray, reject_degenerate: bool = True,
    depths: np.ndarray | None = None, reject_depth_discontinuous: bool = True,
    max_depth_range: float = MAX_TRIANGLE_DEPTH_RANGE_M,
) -> Tuple["mtri.LinearTriInterpolator", "mtri.LinearTriInterpolator", int, int, int]:
    """Delaunay-triangulate pts_a and build linear interpolators for pts_b as a
    function of it (exact at every vertex), after masking out degenerate
    triangles (see degenerate_triangle_mask) unless reject_degenerate is
    False, and depth-discontinuous triangles (see
    depth_discontinuous_triangle_mask) when depths is given and
    reject_depth_discontinuous is True -- two independent checks, since a
    triangle can fail either without failing the other, so a single
    stretched-out or depth-spanning triangle can't smear a real depth
    discontinuity into a smooth (wrong) blend.

    

    `depths`, when given, must be aligned with pts_a's rows (the same Z
    triangulate_matches produced for these points). Passing None (the
    default) skips the depth check regardless of reject_depth_discontinuous,
    for callers (e.g. check_registration_error.py) that don't have/need it.

    Split out of warp_with_match_field so other callers (e.g. a registration
    accuracy check evaluating the same mesh at specific query points, not a
    full pixel grid) can reuse the exact mesh main() builds instead of
    re-deriving it. Returns (interpolate_x, interpolate_y, n_rejected_shape,
    n_rejected_depth, n_triangles); n_rejected_depth excludes triangles
    already counted in n_rejected_shape. Querying either interpolator
    outside a valid triangle returns a masked value.
    """
    triangulation = mtri.Triangulation(pts_a[:, 0], pts_a[:, 1])
    bad = np.zeros(len(triangulation.triangles), dtype=bool)
    n_rejected_shape = 0
    if reject_degenerate:
        bad_shape = degenerate_triangle_mask(
            pts_a, triangulation.triangles, MAX_TRIANGLE_AREA_RATIO, MAX_TRIANGLE_SLENDERNESS,
        )
        n_rejected_shape = int(bad_shape.sum())
        bad |= bad_shape
    n_rejected_depth = 0
    if reject_depth_discontinuous and depths is not None:
        bad_depth = depth_discontinuous_triangle_mask(
            triangulation.triangles, depths, max_depth_range,
        )
        n_rejected_depth = int((bad_depth & ~bad).sum())
        bad |= bad_depth
    triangulation.set_mask(bad)
    interpolate_x = mtri.LinearTriInterpolator(triangulation, pts_b[:, 0].astype(np.float64))
    interpolate_y = mtri.LinearTriInterpolator(triangulation, pts_b[:, 1].astype(np.float64))
    return interpolate_x, interpolate_y, n_rejected_shape, n_rejected_depth, len(triangulation.triangles)


def warp_with_match_field(
    pts_a: np.ndarray, pts_b: np.ndarray, color_b: np.ndarray, size_a: Tuple[int, int],
    reject_degenerate: bool = True,
    depths: np.ndarray | None = None, reject_depth_discontinuous: bool = True,
    max_depth_range: float = MAX_TRIANGLE_DEPTH_RANGE_M,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """Piecewise-affine warp of camera B onto camera A from sparse matches,
    via build_mesh_interpolators, sampled over the full pixel grid and
    remapped. Returns (warped, covered, n_rejected_shape, n_rejected_depth,
    n_triangles): covered is False outside the convex hull of the matches OR
    inside a rejected triangle -- those pixels have no correspondence, so no
    triangulated Z.
    """
    width, height = size_a
    interpolate_x, interpolate_y, n_rejected_shape, n_rejected_depth, n_triangles = build_mesh_interpolators(
        pts_a, pts_b, reject_degenerate, depths, reject_depth_discontinuous, max_depth_range,
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
    return warped, covered, n_rejected_shape, n_rejected_depth, n_triangles


def extrapolate_uncovered(
    pts_a: np.ndarray, depths: np.ndarray, covered: np.ndarray,
    color_b: np.ndarray, size_a: Tuple[int, int],
    camera_matrix_a: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill pixels outside the piecewise-affine match hull by estimating their
    depth from the nearest triangulated match, then reprojecting through the
    calibrated K_a/K_b/R/T (triangulate.reproject_via_depth) instead of
    leaving them unfilled.

    Nearest-neighbor Z makes no smoothness/planarity assumption about the
    scene beyond the hull -- robust if the real depth there is unpredictable
    (e.g. a hand's silhouette against a background), at the cost of a
    blocky, Voronoi-cell-shaped Z field. RBF/thin-plate-spline extrapolation
    is a smoother but costlier alternative, deferred unless this visibly
    breaks down (see the registration-improvement-ideas backlog).

    Returns (filled, newly_covered): `filled` is color_b remapped over the
    whole grid via the extrapolated correspondence (only meaningful where
    `newly_covered` is True); pixels whose reprojection lands behind camera
    B or outside its frame are excluded from `newly_covered`, i.e. this can
    only add coverage, never claim a pixel that's unrecoverable.
    """
    width, height = size_a
    height_b, width_b = color_b.shape[:2]
    missing_rows, missing_cols = np.nonzero(~covered)
    query_px = np.column_stack([missing_cols, missing_rows]).astype(np.float64)

    nearest_z = NearestNDInterpolator(pts_a, depths)
    z_hat = nearest_z(query_px)
    pixels_b, z_b = reproject_via_depth(query_px, z_hat, camera_matrix_a, camera_matrix_b, R, T)
    in_frame = (
        (z_b > 0)
        & (pixels_b[:, 0] >= 0) & (pixels_b[:, 0] < width_b - 1)
        & (pixels_b[:, 1] >= 0) & (pixels_b[:, 1] < height_b - 1)
    )

    map_x = np.full((height, width), -1.0, dtype=np.float32)
    map_y = np.full((height, width), -1.0, dtype=np.float32)
    map_x[missing_rows[in_frame], missing_cols[in_frame]] = pixels_b[in_frame, 0]
    map_y[missing_rows[in_frame], missing_cols[in_frame]] = pixels_b[in_frame, 1]

    filled = cv2.remap(
        color_b, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    newly_covered = np.zeros((height, width), dtype=bool)
    newly_covered[missing_rows[in_frame], missing_cols[in_frame]] = True
    return filled, newly_covered


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
    extrapolated: np.ndarray | None = None,
) -> np.ndarray:
    """Checkerboard of camera A and warped B, only inside the match hull.

    Outside the hull there is no correspondence and no Z; uncovered pixels
    stay as camera A, dimmed. `extrapolated`, when given, marks pixels
    filled by extrapolate_uncovered's depth guess rather than a real match
    -- those get a warm tint so the less-trustworthy region stays visually
    distinct from the matched/interpolated one instead of reading as
    equally solid coverage.
    """
    height, width = color_a.shape[:2]
    cell = max(32, min(height, width) // n_cells)
    rows, cols = np.ogrid[:height, :width]
    board = ((cols // cell) + (rows // cell)) % 2 == 0
    checker = color_a.copy()
    checker[covered & board] = warped[covered & board]
    checker[~covered] = (color_a[~covered] // 2)
    if extrapolated is not None:
        tint = extrapolated & covered
        checker[tint] = (
            checker[tint].astype(np.float32) * np.array([0.7, 0.7, 1.25])
        ).clip(0, 255).astype(np.uint8)
    return checker


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
        resize_to_width(labelled(color_a, "camera A"), PREVIEW_PANEL_WIDTH),
        resize_to_width(labelled(warped_color, "B warped onto A"), PREVIEW_PANEL_WIDTH),
        resize_to_width(labelled(checker, "overlap (checker)"), PREVIEW_PANEL_WIDTH),
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
    matched_covered: np.ndarray, n_unfilled: int,
    reject_degenerate: bool, n_rejected_triangles: int, n_triangles: int,
    use_tiled_keypoints: bool, matcher: str,
    use_geometric_filter: bool, n_before_geometric_filter: int,
    reject_depth_discontinuous: bool, n_rejected_depth_triangles: int, max_depth_range: float,
    match_histogram_used: bool,
    use_depth_extrapolation: bool, extrapolated: np.ndarray,
) -> None:
    inlier_depths = depths[inliers]
    n_pixels = size_a[0] * size_a[1]
    n_matched_covered = int(matched_covered.sum())
    n_extrapolated = int(extrapolated.sum())
    n_covered_total = n_pixels - n_unfilled

    def pct(count: int, denominator: int) -> str:
        return f"{count / denominator * 100:.1f}%" if denominator else "n/a"

    lines = [
        f"Registration: {camera_b} -> {camera_a} (sparse match + piecewise-affine warp)",
        "=" * 66,
        "",
        f"  session:            {session_dir}",
        f"  extrinsics:         {extrinsics_path}",
        f"  resolution:         {size_a[0]}x{size_a[1]}",
        "",
        "MATCHING",
        "-" * 66,
        f"  matcher:              {matcher}",
        f"  keypoint extraction: {('tiled ~' + str(KEYPOINT_TILE_SIZE_PX) + 'px' if use_tiled_keypoints else 'global top-K') if matcher != 'loftr' else 'n/a (loftr is detector-free)'}",
        f"  raw matches:      {n_raw_matches}",
        f"  above min-confidence {min_confidence}: {n_before_geometric_filter}",
        (
            f"  geometric-consistency filter: kept {n_matches}/{n_before_geometric_filter} "
            f"(threshold {GEOMETRIC_CONSISTENCY_THRESHOLD_PX}px vs. "
            f"{GEOMETRIC_CONSISTENCY_NEIGHBORS}-neighbor median displacement)"
        ) if use_geometric_filter else (
            "  geometric-consistency filter: disabled (--no-geometric-consistency-filter)"
        ),
        "",
        "TRIANGULATION",
        "-" * 66,
        f"  inliers (Z in working range): {n_inliers} / {n_matches} "
        f"({n_inliers / n_matches * 100:.1f}%)",
        f"  triangulated Z (inliers, m): min {inlier_depths.min():.4f}, "
        f"median {np.median(inlier_depths):.4f}, max {inlier_depths.max():.4f}",
        f"  match-hull coverage: {n_matched_covered} ({pct(n_matched_covered, n_pixels)}) "
        "-- uncovered pixels have no correspondence, so no Z",
        (
            f"  degenerate triangles rejected (shape): {n_rejected_triangles} / {n_triangles} "
            f"({pct(n_rejected_triangles, n_triangles)}) -- huge-area or thin-sliver, "
            "excluded from the piecewise warp"
        ) if reject_degenerate else (
            "  degenerate triangle rejection (shape): disabled "
            "(--no-reject-degenerate-triangles)"
        ),
        (
            f"  degenerate triangles rejected (depth discontinuity): "
            f"{n_rejected_depth_triangles} / {n_triangles} "
            f"({pct(n_rejected_depth_triangles, n_triangles)}) -- vertices span > "
            f"{max_depth_range * 1000:.0f}mm of triangulated Z, excluded from the piecewise warp"
        ) if reject_depth_discontinuous else (
            "  degenerate triangle rejection (depth discontinuity): disabled "
            "(--no-reject-depth-discontinuous-triangles)"
        ),
        "",
        "COVERAGE",
        "-" * 66,
        f"  camera A pixels:                            {n_pixels}",
        (
            f"  depth-extrapolated coverage: {n_extrapolated} ({pct(n_extrapolated, n_pixels)}) "
            "-- nearest-neighbor Z + calibrated reprojection, not a real match"
        ) if use_depth_extrapolation else (
            "  depth extrapolation: disabled (--no-depth-extrapolation)"
        ),
        f"  total covered (match-hull + extrapolated):  {n_covered_total} "
        f"({pct(n_covered_total, n_pixels)})",
        f"  unfilled (no correspondence):               {n_unfilled} "
        f"({pct(n_unfilled, n_pixels)})",
        "",
        "OUTPUT FILES",
        "-" * 66,
        "  warped_features.jpg   camera B warped onto camera A (piecewise affine from matches)",
    ] + ([
        "  warped_features_histmatched.jpg  warped B after per-channel histogram match to "
        "camera A (this is what overlay_checker.jpg/overlay_blend.jpg actually composite; "
        "--match-histogram was on for this run)",
    ] if match_histogram_used else [
        "  (--match-histogram was off for this run: overlay_checker.jpg/overlay_blend.jpg "
        "composite the raw warp directly, no warped_features_histmatched.jpg saved)",
    ]) + [
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

    use_tiled_keypoints = args.tiled_keypoints
    use_disk = "disk" in args.matcher
    use_loftr = "loftr" in args.matcher
    use_raft = "raft" in args.matcher
    device = torch.device("cpu")
    native_w, native_h = undistorted_a.shape[1], undistorted_a.shape[0]
    print(f"Registering {args.camera_b} -> {args.camera_a}: native {native_w}x{native_h} "
          f"-> --downscale {args.downscale} -> matching+triangulation+warp all at "
          f"{size_a[0]}x{size_a[1]} (CPU, --matcher {' '.join(args.matcher)}"
          f"{', tiled keypoints' if use_tiled_keypoints and use_disk else ''}); "
          "every saved output image is this working resolution, never native.")

    pts_a_parts, pts_b_parts, scores_parts = [], [], []
    if use_disk:
        disk, matcher = load_models(device)
        if use_tiled_keypoints:
            feats_a, hw_a = extract_features_tiled(
                disk, color_a, device, DEFAULT_MAX_KEYPOINTS, KEYPOINT_TILE_SIZE_PX,
            )
            feats_b, hw_b = extract_features_tiled(
                disk, color_b, device, DEFAULT_MAX_KEYPOINTS, KEYPOINT_TILE_SIZE_PX,
            )
        else:
            feats_a, hw_a = extract_features(disk, color_a, device, DEFAULT_MAX_KEYPOINTS)
            feats_b, hw_b = extract_features(disk, color_b, device, DEFAULT_MAX_KEYPOINTS)
        disk_pts_a, disk_pts_b, disk_scores = match_features(matcher, feats_a, feats_b, hw_a, hw_b)
        print(f"DISK+LightGlue: {len(disk_scores)} raw matches")
        pts_a_parts.append(disk_pts_a)
        pts_b_parts.append(disk_pts_b)
        scores_parts.append(disk_scores)
    if use_loftr:
        loftr = load_loftr_model(device)
        loftr_pts_a, loftr_pts_b, loftr_scores = match_loftr(loftr, color_a, color_b, device)
        print(f"LoFTR: {len(loftr_scores)} raw matches")
        pts_a_parts.append(loftr_pts_a)
        pts_b_parts.append(loftr_pts_b)
        scores_parts.append(loftr_scores)
    if use_raft:
        raft_model = load_raft_model(device)
        raft_pts_a, raft_pts_b, raft_scores = match_raft(
            raft_model, color_a, color_b, device, RAFT_GRID_STEP_PX, RAFT_FB_CONSISTENCY_THRESHOLD_PX,
        )
        print(f"RAFT: {len(raft_scores)} grid samples, "
              f"{int((raft_scores >= args.min_confidence).sum())} forward-backward consistent")
        pts_a_parts.append(raft_pts_a)
        pts_b_parts.append(raft_pts_b)
        scores_parts.append(raft_scores)

    pts_a_all = np.concatenate(pts_a_parts)
    pts_b_all = np.concatenate(pts_b_parts)
    scores_all = np.concatenate(scores_parts)

    keep = scores_all >= args.min_confidence
    pts_a, pts_b = pts_a_all[keep], pts_b_all[keep]
    n_raw_matches, n_matches = len(scores_all), int(keep.sum())
    print(f"Pooled: {n_raw_matches} raw matches, {n_matches} above confidence "
          f"{args.min_confidence}")
    if n_matches < args.min_matches:
        raise SystemExit(
            f"Only {n_matches} matches above --min-confidence {args.min_confidence} "
            f"(need >= {args.min_matches}). Check --camera-a/--camera-b order, exposure "
            "match between cameras, or lower --min-confidence."
        )

    use_geometric_filter = not args.no_geometric_consistency_filter
    n_before_geometric_filter = n_matches
    if use_geometric_filter:
        consistent = geometric_consistency_mask(
            pts_a, pts_b, GEOMETRIC_CONSISTENCY_NEIGHBORS, GEOMETRIC_CONSISTENCY_THRESHOLD_PX,
        )
        pts_a, pts_b = pts_a[consistent], pts_b[consistent]
        n_matches = int(consistent.sum())
        print(f"Geometric-consistency filter: kept {n_matches}/{n_before_geometric_filter} matches")

    depths, inliers = triangulate_matches(
        pts_a, pts_b, camera_matrix_a, camera_matrix_b,
        extrinsics.R, extrinsics.T, depth_min, depth_max,
    )
    n_inliers = int(inliers.sum())
    if n_inliers > 0:
        inlier_depths = depths[inliers]
        first = np.flatnonzero(inliers)[0]
        projection_a, projection_b = projection_matrices(
            camera_matrix_a, camera_matrix_b, extrinsics.R, extrinsics.T,
        )
        x_dlt = triangulate_dlt(pts_a[first], pts_b[first], projection_a, projection_b)
        print(
            f"Triangulated {n_inliers}/{n_matches} matches, "
            f"Z median {np.median(inlier_depths):.4f} m "
            f"[{inlier_depths.min():.4f}, {inlier_depths.max():.4f}]"
        )
        print(
            f"  first inlier DLT: X={x_dlt[0]*1000:.2f}, Y={x_dlt[1]*1000:.2f}, "
            f"Z={x_dlt[2]*1000:.2f} mm  (see triangulate.py)"
        )
    if n_inliers < MIN_INLIERS_TO_TRUST:
        raise SystemExit(
            f"Only {n_inliers} matches triangulated inside {depth_min:.3f}-{depth_max:.3f} m "
            f"(need >= {MIN_INLIERS_TO_TRUST}). Check calibration, camera order, or capture "
            "a more textured scene."
        )

    reject_degenerate = not args.no_reject_degenerate_triangles
    reject_depth_discontinuous = not args.no_reject_depth_discontinuous_triangles
    mesh_pts_a, mesh_pts_b, mesh_depths = pts_a[inliers], pts_b[inliers], depths[inliers]
    match_warp, covered, n_rejected_shape_triangles, n_rejected_depth_triangles, n_triangles = warp_with_match_field(
        mesh_pts_a, mesh_pts_b, color_b, size_a, reject_degenerate,
        mesh_depths, reject_depth_discontinuous, args.max_triangle_depth_range,
    )
    warped_color = match_warp
    matched_covered = covered
    n_rejected_triangles = n_rejected_shape_triangles + n_rejected_depth_triangles
    if reject_depth_discontinuous:
        print(
            f"Depth-discontinuity triangle rejection: {n_rejected_depth_triangles}/{n_triangles} "
            f"additional triangles rejected (vertices span > "
            f"{args.max_triangle_depth_range * 1000:.0f}mm of triangulated Z)"
        )
    depth_median = float(np.median(depths[inliers]))
    print(
        f"Piecewise-affine warp over {matched_covered.mean() * 100:.1f}% of the frame "
        f"from {n_inliers} triangulated matches "
        f"(uncovered pixels have no correspondence, so no Z)"
    )

    use_depth_extrapolation = not args.no_depth_extrapolation
    extrapolated_mask = np.zeros_like(matched_covered)
    if use_depth_extrapolation:
        extrapolated_fill, extrapolated_mask = extrapolate_uncovered(
            mesh_pts_a, mesh_depths, matched_covered, color_b, size_a,
            camera_matrix_a, camera_matrix_b, extrinsics.R, extrinsics.T,
        )
        warped_color[extrapolated_mask] = extrapolated_fill[extrapolated_mask]
        print(
            f"Depth extrapolation (nearest-neighbor Z + calibrated reprojection): filled "
            f"{int(extrapolated_mask.sum())} additional pixels "
            f"({extrapolated_mask.mean() * 100:.1f}% of the frame)"
        )
    covered = matched_covered | extrapolated_mask
    n_unfilled = int((~covered).sum())

    match_viz = render_match_visualization(color_a, color_b, pts_a, pts_b, inliers)
    overlay_source = match_histogram(warped_color, color_a) if args.match_histogram else warped_color
    checker = render_checker(color_a, overlay_source, covered, extrapolated=extrapolated_mask)
    blend = color_a.copy()
    blend[covered] = (
        0.5 * color_a[covered].astype(np.float32) + 0.5 * overlay_source[covered].astype(np.float32)
    ).astype(np.uint8)

    cv2.imwrite(str(output_dir / "warped_features.jpg"), warped_color)
    if args.match_histogram:
        cv2.imwrite(str(output_dir / "warped_features_histmatched.jpg"), overlay_source)
    cv2.imwrite(str(output_dir / "overlay_checker.jpg"), checker)
    cv2.imwrite(str(output_dir / "overlay_blend.jpg"), blend)
    cv2.imwrite(str(output_dir / "matches.jpg"), match_viz)
    save_preview(output_dir / "preview_features.jpg", color_a, warped_color, checker)

    fit_result = {
        "n_matches": n_matches,
        "n_inliers": n_inliers,
        "depth_min_m": float(depths[inliers].min()),
        "depth_median_m": depth_median,
        "depth_max_m": float(depths[inliers].max()),
        "interpolated_coverage": float(matched_covered.mean()),
        "depth_extrapolation_used": use_depth_extrapolation,
        "extrapolated_coverage": float(extrapolated_mask.mean()),
        "total_coverage": float(covered.mean()),
        "reject_degenerate_triangles": reject_degenerate,
        "reject_depth_discontinuous_triangles": reject_depth_discontinuous,
        "max_triangle_depth_range_m": args.max_triangle_depth_range,
        "triangles_total": n_triangles,
        "triangles_rejected": n_rejected_triangles,
        "triangles_rejected_shape": n_rejected_shape_triangles,
        "triangles_rejected_depth_discontinuity": n_rejected_depth_triangles,
        "tiled_keypoints_used": use_tiled_keypoints,
        "matcher": args.matcher,
        "geometric_consistency_filter_used": use_geometric_filter,
        "n_before_geometric_filter": n_before_geometric_filter,
        "match_histogram_used": args.match_histogram,
        "unfilled_pixels": n_unfilled,
        "unfilled_fraction": n_unfilled / (size_a[0] * size_a[1]),
    }
    with (output_dir / "fit_result.json").open("w", encoding="utf-8") as f:
        json.dump(fit_result, f, indent=2)
        f.write("\n")

    write_report(
        output_dir / "report_features.txt",
        session_dir, args.camera_a, args.camera_b, extrinsics_path, size_a,
        n_raw_matches, args.min_confidence, n_matches,
        n_inliers, depths, inliers, matched_covered, n_unfilled,
        reject_degenerate, n_rejected_shape_triangles, n_triangles,
        use_tiled_keypoints, " ".join(args.matcher),
        use_geometric_filter, n_before_geometric_filter,
        reject_depth_discontinuous, n_rejected_depth_triangles, args.max_triangle_depth_range,
        args.match_histogram,
        use_depth_extrapolation, extrapolated_mask,
    )

    print(f"\nSaved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
