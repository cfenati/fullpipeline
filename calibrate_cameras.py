#!/usr/bin/env python3
"""Estimate RGB camera intrinsics from ChArUco capture sessions.

Example:
    python calibrate_cameras.py --captures captures/calib_rgb1 --camera rgb_cam1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.opencv_calibrate import (  # noqa: E402
    DEFAULT_MIN_CORNERS,
    DEFAULT_REJECT_SIGMA,
    BoardObservation,
    CameraIntrinsics,
    calibrate_intrinsics,
    coverage_image,
    detect_observations,
    quality_warnings,
)
from calibration.target_board import DEFAULT_BOARD_CONFIG, TargetBoard  # noqa: E402

DEFAULT_OUTPUT_DIR = "calibration/results"


def default_capture_dir(camera: str) -> str:
    """Each camera has its own capture session, e.g. rgb_cam2 -> captures/calib_rgb2."""
    return f"captures/calib_{camera.replace('_cam', '')}"


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def collect_image_paths(capture_dir: Path, camera: str) -> Tuple[List[Path], str]:
    """Find <capture_dir>/<session>/<camera>.jpg, or loose images in capture_dir.

    Only images matching the requested camera are ever returned, so other
    cameras captured in the same session are ignored.
    """
    if not capture_dir.exists():
        raise SystemExit(f"Capture directory not found: {capture_dir}")

    for pattern in (f"*/{camera}.jpg", f"{camera}*.jpg"):
        paths = sorted(capture_dir.glob(pattern))
        if paths:
            return paths, pattern

    raise SystemExit(
        f"No '{camera}.jpg' images found under {capture_dir}.\n"
        "Expected one image per capture session subfolder."
    )


def write_report(
    intrinsics: CameraIntrinsics,
    board: TargetBoard,
    observations: List[BoardObservation],
    skipped: List,
    warnings: List[str],
    path: Path,
) -> None:
    horizontal_fov, vertical_fov = intrinsics.field_of_view_deg()
    distortion = intrinsics.distortion.reshape(-1)
    width, height = intrinsics.image_size

    lines = [
        f"Intrinsic calibration: {intrinsics.name}",
        "=" * 46,
        "",
        f"  target:            {board.squares_x}x{board.squares_y} ChArUco, "
        f"{board.dictionary}, legacy_pattern={board.legacy_pattern}",
        f"  square / marker:   {board.square_size_m * 1000:.1f} mm / "
        f"{board.marker_size_m * 1000:.1f} mm",
        f"  resolution:        {width}x{height}",
        f"  model:             {intrinsics.model}",
        "",
        f"  images found:      {len(observations) + len(skipped)}",
        f"  views detected:    {len(observations)}",
        f"  views used:        {intrinsics.views_used}",
        f"  corners used:      {intrinsics.points_used}",
        f"  RMS reprojection:  {intrinsics.reprojection_error_px:.4f} px",
        "",
        f"  fx, fy:            {intrinsics.fx:.3f}, {intrinsics.fy:.3f}",
        f"  cx, cy:            {intrinsics.cx:.3f}, {intrinsics.cy:.3f}",
        f"  FOV (h x v):       {horizontal_fov:.2f} deg x {vertical_fov:.2f} deg",
        "  distortion:        " + ", ".join(f"{value:.6f}" for value in distortion),
    ]

    if warnings:
        lines += ["", "Quality warnings", "-" * 46]
        for warning in warnings:
            lines.append(f"  - {warning}")

    lines += ["", "Per-view reprojection error (px)", "-" * 46]
    for label, error in sorted(
        intrinsics.per_view_errors.items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(f"  {label:<24} {error:.4f}")

    if intrinsics.rejected_views:
        lines += ["", "Rejected as outliers (px)", "-" * 46]
        for label, error in sorted(
            intrinsics.rejected_views.items(), key=lambda item: item[1], reverse=True
        ):
            lines.append(f"  {label:<24} {error:.4f}")

    lines += [
        "",
        "All accepted views, best first",
        "-" * 66,
        f"  {'session':<20}{'corners':>9}{'markers':>9}{'sharpness':>11}{'error px':>10}",
    ]
    for observation in sorted(
        observations, key=lambda item: item.corner_count, reverse=True
    ):
        error = intrinsics.per_view_errors.get(observation.label)
        error_text = f"{error:.3f}" if error is not None else "dropped"
        lines.append(
            f"  {observation.label:<20}{observation.corner_count:>9}"
            f"{observation.marker_count:>9}{observation.sharpness:>11.0f}{error_text:>10}"
        )

    if skipped:
        lines += ["", "Discarded images", "-" * 66]
        for image_path, reason in skipped:
            lines.append(f"  {image_path.parent.name:<20} {reason}")

    lines += ["", f"Source images ({len(intrinsics.source_images)})", "-" * 46]
    lines += [f"  {source}" for source in intrinsics.source_images]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_undistort_preview(
    intrinsics: CameraIntrinsics,
    observations: List[BoardObservation],
    path: Path,
) -> None:
    best = max(observations, key=lambda observation: observation.corner_count)
    image = cv2.imread(str(best.image_path))
    if image is None:
        return

    corrected = intrinsics.undistort(image)
    labelled = []
    for frame, text in ((image, "original"), (corrected, "undistorted")):
        frame = frame.copy()
        cv2.putText(
            frame, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA
        )
        labelled.append(frame)
    cv2.imwrite(str(path), np.hstack(labelled))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate camera intrinsics from ChArUco capture sessions.",
    )
    parser.add_argument(
        "--captures",
        help="Directory holding capture session subfolders (default: captures/calib_<camera>).",
    )
    parser.add_argument(
        "--camera",
        default="rgb_cam1",
        help="Image basename to calibrate inside each session (default: %(default)s).",
    )
    parser.add_argument(
        "--board",
        help=f"ChArUco board YAML (default: geometric_calibration.board, else {DEFAULT_BOARD_CONFIG}).",
    )
    parser.add_argument(
        "--output",
        help=(
            "Directory for calibration results "
            f"(default: geometric_calibration.output_dir, else {DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=DEFAULT_MIN_CORNERS,
        help="Minimum ChArUco corners for a view to be used (default: %(default)s).",
    )
    parser.add_argument(
        "--reject-sigma",
        type=float,
        default=DEFAULT_REJECT_SIGMA,
        help="Drop views this many robust sigmas above the median error (default: %(default)s).",
    )
    parser.add_argument(
        "--max-view-error",
        type=float,
        default=None,
        help="Absolute per-view reprojection limit in px, instead of the robust cut-off.",
    )
    parser.add_argument(
        "--keep-all-views",
        action="store_true",
        help="Disable outlier rejection.",
    )
    parser.add_argument(
        "--rational",
        action="store_true",
        help="Use the 8-parameter rational distortion model instead of 5-parameter.",
    )
    parser.add_argument(
        "--fix-tangential",
        action="store_true",
        help="Force tangential distortion to zero.",
    )
    parser.add_argument(
        "--fix-k3",
        action="store_true",
        help="Force k3 to zero. More stable when k3 is only absorbing corner noise.",
    )
    parser.add_argument(
        "--save-detections",
        action="store_true",
        help="Save per-image annotated detections for visual inspection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calibration_config = load_config().get("geometric_calibration", {}) or {}

    board_path = args.board or calibration_config.get("board", DEFAULT_BOARD_CONFIG)
    output_root = args.output or calibration_config.get("output_dir", DEFAULT_OUTPUT_DIR)

    capture_dir = resolve_path(args.captures or default_capture_dir(args.camera))
    output_dir = resolve_path(output_root) / args.camera
    output_dir.mkdir(parents=True, exist_ok=True)

    board = TargetBoard.from_yaml(resolve_path(board_path))
    image_paths, pattern = collect_image_paths(capture_dir, args.camera)
    print(f"Calibrating {args.camera} from {len(image_paths)} images")
    print(f"Source: {capture_dir}/{pattern}")
    print(
        f"Target: {board.squares_x}x{board.squares_y} ChArUco, {board.dictionary}, "
        f"square={board.square_size_m * 1000:.1f} mm, legacy_pattern={board.legacy_pattern}"
    )

    detections_dir: Optional[Path] = None
    if args.save_detections:
        detections_dir = output_dir / "detections"
        detections_dir.mkdir(parents=True, exist_ok=True)

    def annotate(image: np.ndarray, observation: BoardObservation) -> None:
        if detections_dir is None:
            return
        preview = image.copy()
        cv2.aruco.drawDetectedCornersCharuco(
            preview,
            observation.image_points,
            observation.corner_ids.reshape(-1, 1),
            (0, 0, 255),
        )
        cv2.imwrite(str(detections_dir / f"{observation.label}.jpg"), preview)

    discarded_dir = output_dir / "discarded"

    def save_discarded(
        image: np.ndarray,
        image_path: Path,
        corners: Optional[np.ndarray],
        corner_ids: Optional[np.ndarray],
        marker_corners: Optional[list],
    ) -> None:
        """Keep rejected frames with whatever was found, to diagnose bad captures."""
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
        cv2.imwrite(str(discarded_dir / f"{image_path.parent.name}.jpg"), preview)

    observations, image_size, skipped = detect_observations(
        image_paths,
        board,
        min_corners=args.min_corners,
        on_image=annotate,
        on_skip=save_discarded,
    )
    print(
        f"Detected the board in {len(observations)}/{len(image_paths)} images "
        f"at {image_size[0]}x{image_size[1]}"
    )
    for image_path, reason in skipped:
        print(f"  discarded {image_path.parent.name}: {reason}")
    if skipped:
        print(f"  annotated copies written to {discarded_dir}")

    best = sorted(observations, key=lambda item: item.corner_count, reverse=True)[:5]
    print(f"Best views (of {board.total_corners} corners):")
    for observation in best:
        print(
            f"  {observation.label}  {observation.corner_count} corners, "
            f"{observation.marker_count} markers, sharpness {observation.sharpness:.0f}"
        )

    intrinsics = calibrate_intrinsics(
        args.camera,
        observations,
        image_size,
        board=board,
        rational=args.rational,
        fix_tangential=args.fix_tangential,
        fix_k3=args.fix_k3,
        reject_sigma=args.reject_sigma,
        max_view_error_px=args.max_view_error,
        reject_outliers=not args.keep_all_views,
        source_root=PROJECT_ROOT,
    )
    intrinsics.discarded_views = {
        image_path.parent.name: reason for image_path, reason in skipped
    }
    warnings = quality_warnings(intrinsics, observations)

    intrinsics_path = output_dir / "intrinsics.json"
    intrinsics.save_json(intrinsics_path)
    write_report(intrinsics, board, observations, skipped, warnings, output_dir / "report.txt")
    cv2.imwrite(str(output_dir / "coverage.png"), coverage_image(observations, image_size))
    save_undistort_preview(intrinsics, observations, output_dir / "undistorted.jpg")

    horizontal_fov, vertical_fov = intrinsics.field_of_view_deg()
    print("")
    print(f"  views used:       {intrinsics.views_used} ({intrinsics.points_used} corners)")
    if intrinsics.rejected_views:
        print(f"  outliers dropped: {len(intrinsics.rejected_views)}")
    print(f"  RMS reprojection: {intrinsics.reprojection_error_px:.4f} px")
    print(f"  fx, fy:           {intrinsics.fx:.2f}, {intrinsics.fy:.2f}")
    print(f"  cx, cy:           {intrinsics.cx:.2f}, {intrinsics.cy:.2f}")
    print(f"  FOV:              {horizontal_fov:.1f} deg x {vertical_fov:.1f} deg")
    print(f"  distortion:       {np.array2string(intrinsics.distortion.reshape(-1), precision=4)}")

    if warnings:
        print("")
        print("Quality warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    print("")
    print(f"Saved intrinsics to {intrinsics_path}")
    print(f"Saved report, coverage map and undistort preview to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
