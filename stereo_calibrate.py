#!/usr/bin/env python3
"""Co-register two cameras: measure their extrinsics and what the pair can do.

Takes the capture sessions in which both cameras saw the ChArUco board, holds
each camera's intrinsics at the values from ``calibrate_cameras.py``, and fits the
rigid transform between them. It then reports the four things that decide whether
the pair is usable at a given working distance: coverage overlap, stereo
baseline, triangulation angle and the depth-dependent registration error.

    python stereo_calibrate.py
    python stereo_calibrate.py --captures captures/calib_rgb1 --reference-depth 0.28
    python stereo_calibrate.py --depth-range 0.15 0.80 --disparity-noise 0.5

Outputs land in ``calibration/results/stereo_<a>_<b>/``: ``extrinsics.json`` for
downstream use, ``report.txt``, two figures, and three images to check the result
by eye.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_cameras import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    default_capture_dir,
    load_config,
    resolve_path,
)
from calibration.opencv_calibrate import (  # noqa: E402
    DEFAULT_MIN_CORNERS,
    DEFAULT_REJECT_SIGMA,
    CameraIntrinsics,
)
from calibration.stereo import (  # noqa: E402
    DEFAULT_BLOCK_GAP_MINUTES,
    DEFAULT_MIN_SHARED_CORNERS,
    calibrate_stereo,
    collect_session_pairs,
    detect_stereo_observations,
    epipolar_residuals,
    group_capture_blocks,
    pose_ensemble,
    pose_scatter,
    rectified_residuals,
    rectify,
    relative_pose_per_view,
    stereo_quality_warnings,
    summarize_capture_blocks,
    suspect_views,
    triangulation_closure,
)
from calibration.stereo_metrics import (  # noqa: E402
    DEFAULT_DEPTH_STEPS,
    DEFAULT_DISPARITY_NOISE_PX,
    DEFAULT_GRID,
    default_depths,
    depth_sweep,
    registration_error,
    registration_uncertainty,
)
from calibration.stereo_report import StereoReport, write_report  # noqa: E402
from calibration.target_board import DEFAULT_BOARD_CONFIG, TargetBoard  # noqa: E402

DEFAULT_RIG_CONFIG = "design/config/rig.yaml"
DEFAULT_AS_BUILT = "design/config/rig_as_built.yaml"
# How far outside the distances the board was actually held the sweep should go.
DEPTH_RANGE_BELOW = 0.5
DEPTH_RANGE_ABOVE = 3.0


def intrinsics_path(config: dict, camera: str, output_root: str) -> str:
    """Where this camera's intrinsics live, preferring an explicit config entry."""
    return config.get(f"intrinsics_{camera}", f"{output_root}/{camera}/intrinsics.json")


def load_intrinsics(path: Path, camera: str) -> CameraIntrinsics:
    if not path.exists():
        raise SystemExit(
            f"No intrinsics for {camera} at {path}.\n"
            f"Run:  python calibrate_cameras.py --camera {camera}"
        )
    intrinsics = CameraIntrinsics.load_json(path)
    # The JSON carries the name it was calibrated under; trust the request.
    intrinsics.name = camera
    return intrinsics


def write_rig_pose(extrinsics, path: Path, rig_path: Optional[Path]) -> Path:
    """Emit the measured pair as a rig-YAML pose block.

    ``design/config/rig.yaml`` holds *intended* poses. This writes the *measured*
    ones in the same spelling, so the coverage and registration studies in
    ``design_rig.py`` can be re-run against the rig that actually exists. When the
    rig file gives camera A a pose, camera B is placed in that same world frame;
    otherwise the world frame is camera A's camera frame.
    """
    R_world_a, t_world_a, anchor = None, None, "camera A's own frame"
    if rig_path is not None and rig_path.exists():
        config = yaml.safe_load(rig_path.read_text(encoding="utf-8")) or {}
        names = [entry.get("name") for entry in config.get("cameras", [])]
        if extrinsics.name_a in names:
            from design import Rig

            camera_a = Rig.from_yaml(rig_path).by_name(extrinsics.name_a)
            R_world_a, t_world_a = camera_a.R, camera_a.t
            anchor = f"{extrinsics.name_a}'s pose in {rig_path.name}"

    R_b, t_b = extrinsics.world_pose_b(R_world_a, t_world_a)
    R_a = np.eye(3) if R_world_a is None else np.asarray(R_world_a)
    t_a = np.zeros(3) if t_world_a is None else np.asarray(t_world_a)

    def block(name: str, R: np.ndarray, t: np.ndarray) -> dict:
        return {
            "name": name,
            "pose": {
                "position": [round(float(v), 6) for v in t],
                "R": [[round(float(v), 9) for v in row] for row in R],
            },
        }

    document = {
        "cameras": [
            block(extrinsics.name_a, R_a, t_a),
            block(extrinsics.name_b, R_b, t_b),
        ]
    }
    header = (
        "# Measured extrinsics, written by stereo_calibrate.py.\n"
        "# T_world_from_camera, metres; the world frame is anchored on "
        f"{anchor}.\n"
        f"# Baseline {extrinsics.baseline_m * 1000:.3f} mm, convergence "
        f"{extrinsics.optical_axis_angle_deg:.3f} deg, from "
        f"{extrinsics.views_used} views.\n"
        "# Paste each pose block over the corresponding `pose:` in "
        "design/config/rig.yaml to\n"
        "# replace the intended geometry with the measured one.\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def update_as_built_yaml(
    extrinsics,
    as_built_path: Path,
    rig_path: Optional[Path],
) -> Optional[Path]:
    """Refresh measured pose blocks in ``rig_as_built.yaml``.

    ``stereo_calibrate`` always writes ``rig_pose.yaml`` next to the results; this
    additionally keeps the design overlay in sync so ``design_rig.py --rig
    design/config/rig_as_built.yaml`` sees the latest measurement without a
    manual paste.
    """
    as_built_path = Path(as_built_path)
    if not as_built_path.exists():
        return None

    existing = yaml.safe_load(as_built_path.read_text(encoding="utf-8")) or {}
    cameras = existing.get("cameras") or []
    if not cameras:
        return None

    R_world_a, t_world_a = None, None
    design = None
    if rig_path is not None and Path(rig_path).exists():
        from design import Rig
        design = Rig.from_yaml(rig_path)
        if extrinsics.name_a in design.names:
            camera_a = design.by_name(extrinsics.name_a)
            R_world_a, t_world_a = camera_a.R, camera_a.t

    R_b, t_b = extrinsics.world_pose_b(R_world_a, t_world_a)
    R_a = np.eye(3) if R_world_a is None else np.asarray(R_world_a)
    t_a = np.zeros(3) if t_world_a is None else np.asarray(t_world_a)

    def pose_block(R: np.ndarray, t: np.ndarray) -> dict:
        return {
            "position": [round(float(v), 6) for v in t],
            "R": [[round(float(v), 9) for v in row] for row in R],
        }

    measured = {
        extrinsics.name_a: pose_block(R_a, t_a),
        extrinsics.name_b: pose_block(R_b, t_b),
    }

    updated = False
    for entry in cameras:
        name = entry.get("name")
        if name in measured:
            entry["pose"] = measured[name]
            updated = True
    if not updated:
        return None

    offset_mm = None
    intended_baseline = None
    if design is not None and extrinsics.name_b in design.names:
        intended = design.by_name(extrinsics.name_b).t
        offset_mm = float(np.linalg.norm(np.asarray(t_b) - intended) * 1000.0)
        if extrinsics.name_a in design.names:
            intended_baseline = float(np.linalg.norm(
                design.by_name(extrinsics.name_b).t - design.by_name(extrinsics.name_a).t
            ) * 1000.0)

    epipolar = getattr(extrinsics, "epipolar_rms_px", None)
    epipolar_line = (f"#   epipolar RMS            {epipolar:.3f} px\n"
                     if epipolar is not None else "")
    header = (
        "# As-built geometry: measured extrinsics instead of design intent.\n"
        "#\n"
        "# This is an overlay. Everything not restated here - intrinsics, target,\n"
        "# analysis settings, the thermal camera - is inherited from rig.yaml, so the two\n"
        "# files cannot drift apart. Run it with:\n"
        "#     python design_rig.py --rig design/config/rig_as_built.yaml info\n"
        "#\n"
        f"# Source: auto-updated by stereo_calibrate.py ({extrinsics.name_a}/"
        f"{extrinsics.name_b})\n"
        f"#   views used              {extrinsics.views_used}\n"
        f"#   stereo RMS              {extrinsics.reprojection_error_px:.3f} px\n"
        f"{epipolar_line}"
        f"#   baseline                {extrinsics.baseline_m * 1000:.2f} mm\n"
        f"#   optical axis angle      {extrinsics.optical_axis_angle_deg:.2f} deg\n"
    )
    if offset_mm is not None and intended_baseline is not None:
        header += (
            "#\n"
            f"#   {extrinsics.name_b} is {offset_mm:.1f} mm from its designed position;\n"
            f"#   baseline is {extrinsics.baseline_m * 1000:.1f} mm rather than "
            f"{intended_baseline:.1f} mm.\n"
        )
    header += (
        "#\n"
        f"# {extrinsics.name_a} is the world-frame anchor (its pose restates the design\n"
        "# intent). Cameras without a measured pair keep their design poses via "
        "extends.\n\n"
    )

    document = {
        "extends": existing.get("extends", "rig.yaml"),
        "name": existing.get("name", "as_built"),
        "cameras": cameras,
    }
    as_built_path.write_text(
        header + yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return as_built_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the extrinsics of a camera pair and evaluate its geometry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="A stereo view is a capture session folder holding an image from both\n"
               "cameras with the board visible in each. capture_pipeline.py writes every\n"
               "camera of a trigger into one folder, so the sessions recorded for the\n"
               "per-camera intrinsics can usually be reused here.",
    )
    parser.add_argument(
        "--captures",
        nargs="+",
        help="Directories holding capture session subfolders "
             "(default: geometric_calibration.stereo_captures, else captures/calib_<a>).",
    )
    parser.add_argument("--camera-a", default="rgb_cam1",
                        help="Reference camera; all geometry is expressed in its frame "
                             "(default: %(default)s).")
    parser.add_argument("--camera-b", default="rgb_cam2",
                        help="Second camera of the pair (default: %(default)s).")
    parser.add_argument("--board",
                        help=f"ChArUco board YAML (default: geometric_calibration.board, "
                             f"else {DEFAULT_BOARD_CONFIG}).")
    parser.add_argument("--intrinsics-a", help="Override the intrinsics JSON for camera A.")
    parser.add_argument("--intrinsics-b", help="Override the intrinsics JSON for camera B.")
    parser.add_argument("--output",
                        help=f"Directory for results (default: geometric_calibration.output_dir, "
                             f"else {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--rig", default=DEFAULT_RIG_CONFIG,
                        help="Rig YAML used to anchor the exported poses in the world "
                             "frame (default: %(default)s).")
    parser.add_argument("--as-built", default=DEFAULT_AS_BUILT,
                        help="As-built overlay YAML to refresh with the measured poses "
                             "(default: %(default)s). Pass empty to skip.")
    parser.add_argument("--no-as-built", action="store_true",
                        help="Do not update the as-built overlay YAML.")

    parser.add_argument("--min-corners", type=int, default=DEFAULT_MIN_CORNERS,
                        help="Minimum corners per image for a view to be considered "
                             "(default: %(default)s).")
    parser.add_argument("--min-shared-corners", type=int, default=DEFAULT_MIN_SHARED_CORNERS,
                        help="Minimum corners seen by both cameras (default: %(default)s).")
    parser.add_argument("--reject-sigma", type=float, default=DEFAULT_REJECT_SIGMA,
                        help="Drop views this many robust sigmas above the median error "
                             "(default: %(default)s).")
    parser.add_argument("--max-view-error", type=float, default=None,
                        help="Absolute per-view limit in px instead of the robust cut-off.")
    parser.add_argument("--keep-all-views", action="store_true",
                        help="Disable outlier rejection.")
    parser.add_argument("--refine-intrinsics", action="store_true",
                        help="Also re-fit the intrinsics. Not recommended: a stereo set "
                             "samples each frame far less well than a dedicated one.")

    parser.add_argument("--exclude", nargs="+", default=[], metavar="SESSION",
                        help="Session names to leave out of the fit.")
    parser.add_argument("--block", type=int, default=None, metavar="N",
                        help="Fit only capture sitting N (1-based), as listed in the "
                             "capture-block table. Use this when sittings disagree.")
    parser.add_argument("--block-gap", type=float, default=DEFAULT_BLOCK_GAP_MINUTES,
                        help="Minutes of idle time that start a new capture sitting "
                             "(default: %(default)s).")
    parser.add_argument("--worst", type=int, default=8,
                        help="How many views to list by epipolar residual, worst first "
                             "(default: %(default)s).")
    parser.add_argument("--flag-sigma", type=float, default=DEFAULT_REJECT_SIGMA,
                        help="Robust sigma for flagging views that disagree with the set "
                             "(default: %(default)s).")
    parser.add_argument("--max-epipolar", type=float, default=None,
                        help="Absolute epipolar limit in px for flagging, instead of the "
                             "robust cut-off.")

    parser.add_argument("--reference-depth", type=float, default=None,
                        help="Distance the depth-agnostic mapping is calibrated at, in "
                             "metres (default: the mean distance the board was held at).")
    parser.add_argument("--depth-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=None,
                        help="Distances to evaluate, in metres (default: derived from the "
                             "distances observed during calibration).")
    parser.add_argument("--depth-steps", type=int, default=DEFAULT_DEPTH_STEPS,
                        help="Number of distances sampled (default: %(default)s).")
    parser.add_argument("--disparity-noise", type=float, default=DEFAULT_DISPARITY_NOISE_PX,
                        help="Matching error in px used for the depth uncertainty "
                             "(default: %(default)s).")
    parser.add_argument("--grid", type=int, default=DEFAULT_GRID,
                        help="Samples per axis on each evaluated plane (default: %(default)s).")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="stereoRectify alpha: 0 crops to valid pixels, 1 keeps all "
                             "(default: %(default)s).")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip the matplotlib figures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config().get("geometric_calibration", {}) or {}

    board_path = args.board or config.get("board", DEFAULT_BOARD_CONFIG)
    output_root = args.output or config.get("output_dir", DEFAULT_OUTPUT_DIR)
    capture_values = (
        args.captures
        or config.get("stereo_captures")
        or [default_capture_dir(args.camera_a)]
    )
    if isinstance(capture_values, str):
        capture_values = [capture_values]
    capture_dirs = [resolve_path(value) for value in capture_values]

    pair_name = f"stereo_{args.camera_a}_{args.camera_b}"
    output_dir = resolve_path(output_root) / pair_name
    output_dir.mkdir(parents=True, exist_ok=True)

    board = TargetBoard.from_yaml(resolve_path(board_path))
    intrinsics_a = load_intrinsics(
        resolve_path(args.intrinsics_a
                     or intrinsics_path(config, args.camera_a, output_root)),
        args.camera_a,
    )
    intrinsics_b = load_intrinsics(
        resolve_path(args.intrinsics_b
                     or intrinsics_path(config, args.camera_b, output_root)),
        args.camera_b,
    )

    session_pairs, notes = collect_session_pairs(
        capture_dirs, args.camera_a, args.camera_b, exclude=args.exclude
    )
    print(f"Co-registering {args.camera_a} -> {args.camera_b} "
          f"from {len(session_pairs)} sessions")
    for capture_dir in capture_dirs:
        print(f"Source: {capture_dir}/*/{{{args.camera_a},{args.camera_b}}}.jpg")
    print(f"Target: {board.squares_x}x{board.squares_y} ChArUco, {board.dictionary}, "
          f"square={board.square_size_m * 1000:.2f} mm, "
          f"legacy_pattern={board.legacy_pattern}")

    discarded_dir = output_dir / "discarded"

    def save_discarded(image, image_path, corners, corner_ids, marker_corners) -> None:
        """Keep frames where the board was not found, to diagnose bad captures."""
        discarded_dir.mkdir(parents=True, exist_ok=True)
        preview = image.copy()
        if marker_corners:
            cv2.aruco.drawDetectedMarkers(preview, tuple(marker_corners))
        if corner_ids is not None and len(corner_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(
                preview,
                np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
                np.asarray(corner_ids, dtype=np.int32).reshape(-1, 1),
                (0, 0, 255),
            )
        cv2.imwrite(str(discarded_dir / f"{image_path.parent.name}_{image_path.stem}.jpg"),
                    preview)

    observations, size_a, size_b, skipped = detect_stereo_observations(
        session_pairs,
        board,
        min_corners=args.min_corners,
        min_shared_corners=args.min_shared_corners,
        on_skip=save_discarded,
    )
    print(f"Both cameras saw the board in {len(observations)}/{len(session_pairs)} sessions "
          f"({args.camera_a} at {size_a[0]}x{size_a[1]}, "
          f"{args.camera_b} at {size_b[0]}x{size_b[1]})")
    for label, reason in skipped:
        print(f"  discarded {label}: {reason}")
    if skipped:
        print(f"  annotated copies written to {discarded_dir}")

    for intrinsics, size in ((intrinsics_a, size_a), (intrinsics_b, size_b)):
        if intrinsics.image_size != size:
            raise SystemExit(
                f"{intrinsics.name} was calibrated at "
                f"{intrinsics.image_size[0]}x{intrinsics.image_size[1]} but these captures "
                f"are {size[0]}x{size[1]}. Rescaling intrinsics across sensor modes changes "
                "the field of view; recalibrate at the capture resolution instead."
            )

    shared = [observation.corner_count for observation in observations]
    print(f"Shared corners per view: {min(shared)} to {max(shared)} "
          f"of {board.total_corners} (median {int(np.median(shared))})")

    all_blocks = group_capture_blocks(observations, gap_minutes=args.block_gap)
    block_summaries = summarize_capture_blocks(
        intrinsics_a,
        intrinsics_b,
        observations,
        board=board,
        gap_minutes=args.block_gap,
    )
    if len(all_blocks) > 1:
        print(f"Capture sittings: {len(all_blocks)} "
              f"(gap > {args.block_gap:g} min between timestamps)")
        for block in block_summaries:
            print(f"  block {block.index}: {block.n_views} views, "
                  f"baseline {block.baseline_mm:.2f} mm, "
                  f"convergence {block.convergence_deg:.2f} deg, "
                  f"epipolar {block.epipolar_rms_px:.3f} px  [{block.span}]")
        for index, block in enumerate(all_blocks, start=1):
            if len(block) < 4:
                print(f"  block {index}: {len(block)} views (too few to fit alone)  "
                      f"[{block[0].label} .. {block[-1].label}]")

    fit_observations = observations
    if args.block is not None:
        if args.block < 1 or args.block > len(all_blocks):
            raise SystemExit(
                f"--block {args.block} is out of range; this set has "
                f"{len(all_blocks)} capture sitting(s)."
            )
        fit_observations = all_blocks[args.block - 1]
        print(f"Fitting only block {args.block} "
              f"({len(fit_observations)} views: {fit_observations[0].label} .. "
              f"{fit_observations[-1].label})")

    extrinsics = calibrate_stereo(
        intrinsics_a,
        intrinsics_b,
        fit_observations,
        board=board,
        fix_intrinsics=not args.refine_intrinsics,
        reject_sigma=args.reject_sigma,
        max_view_error_px=args.max_view_error,
        reject_outliers=not args.keep_all_views,
    )
    if extrinsics.rejected_views:
        print(f"Dropped {len(extrinsics.rejected_views)} view(s) as outliers: "
              + ", ".join(extrinsics.rejected_views))
    extrinsics.discarded_views = dict(skipped)

    kept = [
        observation for observation in fit_observations
        if observation.label in set(extrinsics.source_sessions)
    ]
    views = relative_pose_per_view(extrinsics, kept)
    scatter = pose_scatter(extrinsics, views)
    extrinsics.observed_depth_m = {
        key: scatter[f"depth_{key}_m"] for key in ("min", "mean", "max") if scatter
    }

    observed_min = scatter.get("depth_min_m", 0.2)
    observed_max = scatter.get("depth_max_m", 0.5)
    reference_depth = args.reference_depth or scatter.get("depth_mean_m", 0.3)
    if args.depth_range:
        depth_min, depth_max = args.depth_range
    else:
        depth_min = max(0.02, DEPTH_RANGE_BELOW * observed_min)
        depth_max = DEPTH_RANGE_ABOVE * observed_max
    depths = default_depths(depth_min, depth_max, args.depth_steps)
    print(f"Evaluating {depth_min:.3f}-{depth_max:.3f} m, reference plane at "
          f"{reference_depth:.3f} m")

    epipolar = epipolar_residuals(extrinsics, kept)
    rectification = rectify(extrinsics, alpha=args.alpha, reference_depth=reference_depth)
    rectified = rectified_residuals(extrinsics, rectification, kept)
    closure = triangulation_closure(extrinsics, kept)
    extrinsics.epipolar_rms_px = epipolar["rms_px"]
    extrinsics.rectified_vertical_rms_px = rectified["vertical_rms_px"]
    extrinsics.closure_rms_mm = closure["rms_mm"]
    extrinsics.closure_scale = closure["scale_mean"]

    sweep = depth_sweep(
        extrinsics,
        depths,
        grid=args.grid,
        disparity_noise_px=args.disparity_noise,
    )
    registration = registration_error(extrinsics, reference_depth, depths)
    uncertainty = registration_uncertainty(
        extrinsics, pose_ensemble(extrinsics, views), depths
    )
    warnings = stereo_quality_warnings(
        extrinsics, scatter, epipolar, sweep, closure, rectification,
        blocks=block_summaries,
    )
    suspect = suspect_views(
        views, epipolar, sigma=args.flag_sigma, max_epipolar_px=args.max_epipolar
    )

    report = StereoReport(
        extrinsics=extrinsics,
        observations=kept,
        views=views,
        scatter=scatter,
        epipolar=epipolar,
        rectification=rectification,
        rectified=rectified,
        closure=closure,
        sweep=sweep,
        registration=registration,
        uncertainty=uncertainty,
        warnings=warnings,
        suspect=suspect,
        blocks=block_summaries,
        skipped=skipped,
        notes=notes,
        capture_dirs=capture_dirs,
    )

    extrinsics_file = output_dir / "extrinsics.json"
    extrinsics.save_json(extrinsics_file)
    written = write_report(report, output_dir, figures=not args.no_figures)
    written.append(write_rig_pose(extrinsics, output_dir / "rig_pose.yaml",
                                 resolve_path(args.rig) if args.rig else None))
    if not args.no_as_built and args.as_built:
        as_built = update_as_built_yaml(
            extrinsics,
            resolve_path(args.as_built),
            resolve_path(args.rig) if args.rig else None,
        )
        if as_built is not None:
            written.append(as_built)

    center = extrinsics.center_b_in_a * 1000.0
    at_reference = sweep.at(reference_depth)
    print("")
    print(f"  views used:        {extrinsics.views_used} "
          f"({extrinsics.points_used} shared corners)")
    print(f"  epipolar RMS:      {epipolar['rms_px']:.4f} px "
          f"(p95 {epipolar['p95_px']:.3f}, max {epipolar['max_px']:.3f})")
    floor = extrinsics.intrinsic_rms_px
    if floor is not None:
        print(f"  stereo RMS:        {extrinsics.reprojection_error_px:.4f} px = "
              f"{extrinsics.reprojection_error_px / floor:.1f}x the {floor:.4f} px floor "
              "set by the fixed intrinsics")
    else:
        print(f"  stereo RMS:        {extrinsics.reprojection_error_px:.4f} px")
    print(f"  baseline:          {extrinsics.baseline_m * 1000:.3f} mm"
          + (f" +/- {scatter['baseline_std_mm']:.3f} mm over the views" if scatter else ""))
    print(f"  {args.camera_b} offset:  right {center[0]:+.2f}, down {center[1]:+.2f}, "
          f"forward {center[2]:+.2f} mm")
    print(f"  convergence:       {extrinsics.optical_axis_angle_deg:.3f} deg "
          f"(total rotation {extrinsics.rotation_angle_deg:.3f} deg)")
    print(f"  at {reference_depth:.3f} m:      "
          f"{at_reference['fraction_of_a'] * 100:.1f} % of {args.camera_a}'s view is shared, "
          f"triangulation {at_reference['triangulation_axis_deg']:.2f} deg, "
          f"depth sigma {at_reference['depth_sigma_axis_mm']:.3f} mm")
    near, far = registration.depth_for_error(2.0)
    if np.isfinite(near) and np.isfinite(far):
        print(f"  registration:      a fixed mapping holds 2 px from {near:.4f} to "
              f"{far:.4f} m ({(far - near) * 1000:.1f} mm slab)")
    elif np.isfinite(near):
        print(f"  registration:      a fixed mapping holds 2 px beyond {near:.4f} m")
    print(f"  board rebuilt to:  {closure['rms_mm']:.3f} mm rms "
          f"(scale {closure['scale_mean']:.5f})")

    if warnings:
        print("")
        print("Quality warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    image_paths = {
        observation.label: (observation.left.image_path, observation.right.image_path)
        for observation in kept
    }
    ranked = sorted(
        epipolar["per_view_rms_px"].items(), key=lambda item: item[1], reverse=True
    )
    if args.worst > 0 and ranked:
        print("")
        print(f"Worst {min(args.worst, len(ranked))} views by epipolar residual:")
        for label, value in ranked[:args.worst]:
            print(f"  {value:7.3f} px  {label}"
                  + ("   <- flagged" if label in suspect else ""))
            for image_path in image_paths.get(label, ()):
                print(f"              {image_path}")

    if suspect:
        print("")
        print(f"{len(suspect)} view(s) disagree with the set:")
        for label in sorted(suspect, key=lambda name: -epipolar["per_view_rms_px"].get(name, 0.0)):
            print(f"  {label}: {'; '.join(suspect[label])}")
            for image_path in image_paths.get(label, ()):
                print(f"      {image_path}")
        print("  Leave them out with:  --exclude " + " ".join(sorted(suspect)))
        print("  Or delete those session folders by hand and re-run.")

    print("")
    print(f"Saved extrinsics to {extrinsics_file}")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
