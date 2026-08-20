#!/usr/bin/env python3
"""Score register_features.py's warp against real, precisely-known points.

Every comparison of register_features.py's output so far has been eyeballing
a checkerboard/blend overlay of an uncontrolled hand capture -- useful for
spotting gross defects, but it cannot say "the warp is off by N px here" or
let two configurations (e.g. --no-border-anchors vs. default) be ranked by a
number instead of a vibe.

This script reuses the same held-out ChArUco board set cross_validate_stereo.py
already validates calibration against (default captures/cross-validation,
geometric_calibration.cross_validation_captures): board corners are detected
independently in both cameras, so a corner's position in camera B is known
ground truth, not something inferred from the warp being scored. For each
session:
    1. Build the exact same piecewise-affine mesh register_features.py's
       main() would (LightGlue match -> confidence filter -> triangulate ->
       border anchors -> build_mesh_interpolators), reusing its functions
       directly rather than re-deriving the geometry.
    2. Undistort the board corners detected in camera A and B (detection runs
       on the raw distorted JPEG, register_features.py's mesh lives in the
       undistorted+downscaled frame) via cv2.undistortPoints, then scale by
       --downscale to land in that same working-resolution frame.
    3. Evaluate the mesh at each camera-A corner to predict where it should
       land in camera B (falling back to the same calibrated plane
       register_features.py itself falls back to, outside the mesh), and
       compare against the corner's own independently-detected position in B.

Caveat: a flat ChArUco board at calibration distance is not the same subject
as a close-range hand spanning several centimetres of depth -- this measures
warp accuracy on a different (single-plane-friendly) scene, not hand-specific
performance. It is still the only number here that isn't eyeballed.

Usage:
    python check_registration_error.py
    python check_registration_error.py --session captures/cross-validation/<one>
    python check_registration_error.py --no-border-anchors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_cameras import load_config, resolve_path  # noqa: E402
from calibration.opencv_calibrate import DEFAULT_MIN_CORNERS  # noqa: E402
from calibration.stereo import (  # noqa: E402
    DEFAULT_MIN_SHARED_CORNERS,
    StereoExtrinsics,
    StereoObservation,
    collect_session_pairs,
    detect_stereo_observations,
)
from calibration.target_board import TargetBoard  # noqa: E402
from register_pipeline import (  # noqa: E402
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_REGISTRATION_OUTPUT_DIR,
    default_extrinsics_path,
    downscale_pair,
    load_session_images,
    undistort_pair,
)
from register_features import (  # noqa: E402
    DEFAULT_DOWNSCALE,
    DEFAULT_MAX_KEYPOINTS,
    DEFAULT_MIN_CONFIDENCE,
    border_anchor_points,
    build_mesh_interpolators,
    BORDER_ANCHOR_SPACING_PX,
    extract_features,
    extract_features_tiled,
    geometric_consistency_mask,
    GEOMETRIC_CONSISTENCY_NEIGHBORS,
    GEOMETRIC_CONSISTENCY_THRESHOLD_PX,
    KEYPOINT_TILE_SIZE_PX,
    load_loftr_model,
    load_models,
    match_features,
    match_loftr,
    project_via_plane,
    triangulate_matches,
)

DEFAULT_CROSS_VALIDATION_CAPTURES = "captures/cross-validation"
MIN_MESH_POINTS = 4  # fewer real matches than this can't even define a Delaunay mesh


def board_from_extrinsics(extrinsics: StereoExtrinsics, extrinsics_path: Path) -> TargetBoard:
    if not extrinsics.board:
        raise SystemExit(
            f"{extrinsics_path} carries no board info; re-run stereo_calibrate.py "
            "(current version always records it) before scoring against it."
        )
    return TargetBoard.from_dict(extrinsics.board)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score register_features.py's warp against held-out ChArUco corners.",
    )
    parser.add_argument(
        "--captures", nargs="+", default=None,
        help="Directories holding held-out capture sessions (default: "
             "geometric_calibration.cross_validation_captures, else "
             f"{DEFAULT_CROSS_VALIDATION_CAPTURES}).",
    )
    parser.add_argument(
        "--session", default=None,
        help="Restrict to one session folder instead of every session under --captures.",
    )
    parser.add_argument("--camera-a", default="rgb_cam1")
    parser.add_argument("--camera-b", default="rgb_cam2")
    parser.add_argument("--extrinsics", default=None,
                        help="Stereo extrinsics JSON (default: geometric_calibration."
                             "extrinsics_<a>_<b> in config, else "
                             "calibration/results/stereo_<a>_<b>/extrinsics.json).")
    parser.add_argument("--min-corners", type=int, default=DEFAULT_MIN_CORNERS)
    parser.add_argument("--min-shared-corners", type=int, default=DEFAULT_MIN_SHARED_CORNERS)
    parser.add_argument("--downscale", type=float, default=DEFAULT_DOWNSCALE,
                        help="Must match the register_features.py run being scored "
                             "(default: %(default)s).")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--no-border-anchors", action="store_true")
    parser.add_argument("--no-reject-degenerate-triangles", action="store_true")
    parser.add_argument("--no-geometric-consistency-filter", action="store_true")
    parser.add_argument("--tiled-keypoints", action="store_true",
                        help="Opt-in: measured as an accuracy regression, see "
                             "register_features.py --help for details.")
    parser.add_argument("--matcher", choices=("disk", "loftr", "both"), default="disk",
                        help="Must match the register_features.py configuration being "
                             "scored (default: %(default)s).")
    parser.add_argument("--out", "--output", dest="output", default=None,
                        help="Output directory (default: registration.output_dir in "
                             f"config, else {DEFAULT_REGISTRATION_OUTPUT_DIR}) "
                             "/ registration_error.")
    return parser.parse_args()


def undistort_points(points: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    """cv2.undistortPoints with P=camera_matrix, matching undistort_pair's
    cv2.undistort(img, K, dist) convention (no newCameraMatrix -> same K applies
    to the undistorted frame)."""
    undistorted = cv2.undistortPoints(
        points.reshape(-1, 1, 2).astype(np.float64), camera_matrix, distortion, P=camera_matrix,
    )
    return undistorted.reshape(-1, 2)


def evaluate_observation(
    observation: StereoObservation,
    session_dir: Path,
    camera_a: str, camera_b: str,
    extrinsics: StereoExtrinsics,
    disk, lg_matcher, loftr, matcher_choice: str, device: torch.device,
    downscale: float, min_confidence: float,
    depth_min: float, depth_max: float,
    use_border_anchors: bool, reject_degenerate: bool, use_tiled_keypoints: bool,
    use_geometric_filter: bool,
) -> dict:
    """Build register_features.py's mesh for this session and evaluate it at
    this session's independently-detected board corners. Returns a dict with
    per-corner errors (px, working resolution) or a "skipped" reason."""
    image_a, image_b = load_session_images(session_dir, camera_a, camera_b)
    undistorted_a, undistorted_b = undistort_pair(image_a, image_b, extrinsics)
    color_a, color_b, camera_matrix_a, camera_matrix_b = downscale_pair(
        undistorted_a, undistorted_b, extrinsics.camera_matrix_a, extrinsics.camera_matrix_b, downscale,
    )
    size_a = (color_a.shape[1], color_a.shape[0])
    size_b = (color_b.shape[1], color_b.shape[0])

    pts_a_parts, pts_b_parts, scores_parts = [], [], []
    if matcher_choice in ("disk", "both"):
        if use_tiled_keypoints:
            feats_a, hw_a = extract_features_tiled(disk, color_a, device, DEFAULT_MAX_KEYPOINTS, KEYPOINT_TILE_SIZE_PX)
            feats_b, hw_b = extract_features_tiled(disk, color_b, device, DEFAULT_MAX_KEYPOINTS, KEYPOINT_TILE_SIZE_PX)
        else:
            feats_a, hw_a = extract_features(disk, color_a, device, DEFAULT_MAX_KEYPOINTS)
            feats_b, hw_b = extract_features(disk, color_b, device, DEFAULT_MAX_KEYPOINTS)
        disk_pts_a, disk_pts_b, disk_scores = match_features(lg_matcher, feats_a, feats_b, hw_a, hw_b)
        pts_a_parts.append(disk_pts_a)
        pts_b_parts.append(disk_pts_b)
        scores_parts.append(disk_scores)
    if matcher_choice in ("loftr", "both"):
        loftr_pts_a, loftr_pts_b, loftr_scores = match_loftr(loftr, color_a, color_b, device)
        pts_a_parts.append(loftr_pts_a)
        pts_b_parts.append(loftr_pts_b)
        scores_parts.append(loftr_scores)

    pts_a_all = np.concatenate(pts_a_parts)
    pts_b_all = np.concatenate(pts_b_parts)
    scores_all = np.concatenate(scores_parts)
    keep = scores_all >= min_confidence
    pts_a, pts_b = pts_a_all[keep], pts_b_all[keep]
    if use_geometric_filter:
        consistent = geometric_consistency_mask(
            pts_a, pts_b, GEOMETRIC_CONSISTENCY_NEIGHBORS, GEOMETRIC_CONSISTENCY_THRESHOLD_PX,
        )
        pts_a, pts_b = pts_a[consistent], pts_b[consistent]

    depths, inliers = triangulate_matches(
        pts_a, pts_b, camera_matrix_a, camera_matrix_b,
        extrinsics.R, extrinsics.T, depth_min, depth_max,
    )
    n_inliers = int(inliers.sum())
    if n_inliers < MIN_MESH_POINTS:
        return {"label": observation.label, "skipped": f"only {n_inliers} triangulated matches"}

    fill_depth = float(np.median(depths[inliers]))
    mesh_pts_a, mesh_pts_b = pts_a[inliers], pts_b[inliers]
    if use_border_anchors:
        candidate_anchors_a = border_anchor_points(size_a, BORDER_ANCHOR_SPACING_PX)
        anchors_b, anchors_valid = project_via_plane(
            candidate_anchors_a, camera_matrix_a, camera_matrix_b,
            extrinsics.R, extrinsics.T, fill_depth, size_b,
        )
        mesh_pts_a = np.concatenate([mesh_pts_a, candidate_anchors_a[anchors_valid]])
        mesh_pts_b = np.concatenate([mesh_pts_b, anchors_b[anchors_valid]])

    interpolate_x, interpolate_y, n_rejected, n_triangles = build_mesh_interpolators(
        mesh_pts_a, mesh_pts_b, reject_degenerate,
    )

    corner_a = undistort_points(observation.left_points, extrinsics.camera_matrix_a, extrinsics.distortion_a) * downscale
    corner_b_truth = undistort_points(observation.right_points, extrinsics.camera_matrix_b, extrinsics.distortion_b) * downscale

    predicted_x = interpolate_x(corner_a[:, 0], corner_a[:, 1])
    predicted_y = interpolate_y(corner_a[:, 0], corner_a[:, 1])
    mesh_covered = ~np.ma.getmaskarray(predicted_x)
    predicted_mesh = np.ma.filled(np.stack([predicted_x, predicted_y], axis=1), np.nan)

    predicted_plane, plane_valid = project_via_plane(
        corner_a, camera_matrix_a, camera_matrix_b, extrinsics.R, extrinsics.T, fill_depth, size_b,
    )

    predicted = np.where(mesh_covered[:, None], predicted_mesh, predicted_plane)
    valid = mesh_covered | plane_valid
    errors_px = np.linalg.norm(predicted[valid] - corner_b_truth[valid], axis=1)

    return {
        "label": observation.label,
        "n_corners": int(valid.sum()),
        "n_corners_total": len(corner_a),
        "n_mesh_covered": int(mesh_covered.sum()),
        "n_inliers": n_inliers,
        "n_triangles": n_triangles,
        "n_rejected_triangles": n_rejected,
        "errors_px": errors_px,
    }


def main() -> int:
    args = parse_args()
    reg_config = load_config().get("registration", {}) or {}
    geo_config = load_config().get("geometric_calibration", {}) or {}
    depth_range = reg_config.get("depth_range", [DEFAULT_DEPTH_MIN, DEFAULT_DEPTH_MAX])
    depth_min, depth_max = float(depth_range[0]), float(depth_range[1])

    extrinsics_path = resolve_path(
        args.extrinsics or default_extrinsics_path(args.camera_a, args.camera_b)
    )
    if not extrinsics_path.exists():
        raise SystemExit(f"No stereo extrinsics at {extrinsics_path}.")
    extrinsics = StereoExtrinsics.load_json(extrinsics_path)
    board = board_from_extrinsics(extrinsics, extrinsics_path)

    if args.session:
        session_dir = resolve_path(args.session)
        capture_dirs = [session_dir.parent]
    else:
        capture_values = args.captures or geo_config.get("cross_validation_captures") \
            or DEFAULT_CROSS_VALIDATION_CAPTURES
        if isinstance(capture_values, str):
            capture_values = [capture_values]
        capture_dirs = [resolve_path(value) for value in capture_values]

    session_pairs, notes = collect_session_pairs(capture_dirs, args.camera_a, args.camera_b)
    if args.session:
        session_pairs = [pair for pair in session_pairs if pair[0] == session_dir.name]
    session_dirs_by_label = {label: path_a.parent for label, path_a, _ in session_pairs}

    observations, size_a, size_b, skipped = detect_stereo_observations(
        session_pairs, board, min_corners=args.min_corners, min_shared_corners=args.min_shared_corners,
    )
    print(f"Board visible in both cameras for {len(observations)}/{len(session_pairs)} sessions")
    for label, reason in skipped:
        print(f"  discarded {label}: {reason}")
    if not observations:
        raise SystemExit("No usable session had the board visible in both cameras.")

    reject_degenerate = not args.no_reject_degenerate_triangles
    use_border_anchors = not args.no_border_anchors
    use_tiled_keypoints = args.tiled_keypoints
    use_geometric_filter = not args.no_geometric_consistency_filter
    device = torch.device("cpu")
    disk, lg_matcher, loftr = None, None, None
    if args.matcher in ("disk", "both"):
        disk, lg_matcher = load_models(device)
    if args.matcher in ("loftr", "both"):
        loftr = load_loftr_model(device)

    results = []
    for observation in observations:
        session_dir = session_dirs_by_label[observation.label]
        print(f"Evaluating {observation.label} ({len(observation.corner_ids)} shared corners)...")
        result = evaluate_observation(
            observation, session_dir, args.camera_a, args.camera_b, extrinsics,
            disk, lg_matcher, loftr, args.matcher, device, args.downscale, args.min_confidence,
            depth_min, depth_max, use_border_anchors, reject_degenerate, use_tiled_keypoints,
            use_geometric_filter,
        )
        if "skipped" in result:
            print(f"  skipped: {result['skipped']}")
        else:
            errors = result["errors_px"]
            print(f"  {result['n_corners']}/{result['n_corners_total']} corners scored, "
                  f"{result['n_mesh_covered']} via mesh -- "
                  f"median {np.median(errors):.2f} px, max {errors.max():.2f} px")
        results.append(result)

    scored = [r for r in results if "skipped" not in r]
    if not scored:
        raise SystemExit("Every session was skipped; nothing to report.")
    all_errors = np.concatenate([r["errors_px"] for r in scored])
    n_mesh_total = sum(r["n_mesh_covered"] for r in scored)

    output_root = args.output or reg_config.get("output_dir", DEFAULT_REGISTRATION_OUTPUT_DIR)
    output_dir = resolve_path(output_root) / "registration_error"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"Registration accuracy check ({args.camera_b} -> {args.camera_a}, "
        "held-out board corners)",
        "=" * 66,
        "",
        f"  extrinsics:       {extrinsics_path}",
        f"  downscale:        {args.downscale}",
        f"  matcher:          {args.matcher}",
        f"  border anchors:   {'on' if use_border_anchors else 'off (--no-border-anchors)'}",
        f"  degenerate reject: {'on' if reject_degenerate else 'off (--no-reject-degenerate-triangles)'}",
        f"  tiled keypoints:  {'on (--tiled-keypoints)' if use_tiled_keypoints else 'off'}",
        f"  geometric filter: {'on' if use_geometric_filter else 'off (--no-geometric-consistency-filter)'}",
        f"  sessions scored:  {len(scored)}/{len(results)}",
        f"  corners scored:   {len(all_errors)} ({n_mesh_total} via mesh, "
        f"{len(all_errors) - n_mesh_total} via fallback plane)",
        "",
        "ERROR (working-resolution px, camera B)",
        "-" * 66,
        f"  mean / median:    {all_errors.mean():.2f} / {np.median(all_errors):.2f}",
        f"  p90 / max:        {np.percentile(all_errors, 90):.2f} / {all_errors.max():.2f}",
        "",
        "Per-session",
        "-" * 66,
    ]
    for r in results:
        if "skipped" in r:
            lines.append(f"  {r['label']:<24} skipped: {r['skipped']}")
        else:
            e = r["errors_px"]
            lines.append(
                f"  {r['label']:<24} n={r['n_corners']:<4} "
                f"median {np.median(e):>6.2f}px  max {e.max():>6.2f}px"
            )
    lines.append("")
    lines.append(
        "Caveat: this board is captured flat, near calibration distance -- it measures "
        "warp accuracy on a single-plane-friendly scene, not hand-specific performance."
    )
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "downscale": args.downscale,
        "use_border_anchors": use_border_anchors,
        "reject_degenerate_triangles": reject_degenerate,
        "use_tiled_keypoints": use_tiled_keypoints,
        "use_geometric_filter": use_geometric_filter,
        "matcher": args.matcher,
        "n_sessions_scored": len(scored),
        "n_corners_scored": len(all_errors),
        "n_corners_via_mesh": n_mesh_total,
        "mean_px": float(all_errors.mean()),
        "median_px": float(np.median(all_errors)),
        "p90_px": float(np.percentile(all_errors, 90)),
        "max_px": float(all_errors.max()),
        "per_session": [
            {
                "label": r["label"],
                "skipped": r.get("skipped"),
                "n_corners": r.get("n_corners"),
                "median_px": float(np.median(r["errors_px"])) if "errors_px" in r else None,
                "max_px": float(r["errors_px"].max()) if "errors_px" in r else None,
            }
            for r in results
        ],
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"\nOverall: {len(all_errors)} corners, median {np.median(all_errors):.2f} px, "
          f"p90 {np.percentile(all_errors, 90):.2f} px, max {all_errors.max():.2f} px")
    print(f"Saved report.txt / result.json to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
