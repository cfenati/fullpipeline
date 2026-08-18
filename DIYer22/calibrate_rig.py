#!/usr/bin/env python3
"""Calibrate rgb_cam1/rgb_cam2 extrinsics with DIYer22's `calibrating` library.

An independent second implementation of stereo_calibrate.py's extrinsics fit:
`calibrating.Stereo` also uses cv2.stereoCalibrate(CALIB_FIX_INTRINSIC), but
with its own ChArUco detection and its own from-scratch intrinsics fit, so a
large disagreement here would point at a bug rather than at the rig.

Requires: pip install boxx  (only new dependency beyond this project's own
requirements.txt; opencv-contrib-python already satisfies calibrating's
cv2.aruco needs).

Usage:
    python3 DIYer22/calibrate_rig.py
    python3 DIYer22/calibrate_rig.py --captures captures/stereo --camera-a rgb_cam1 --camera-b rgb_cam2
"""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "DIYer22" / "calibrating"))

import calibrating  # noqa: E402

# Matches calibration/config/charuco_11x8.yaml.
BOARD_SQUARES_XY = (11, 8)
BOARD_SQUARE_SIZE_MM = 4.0
BOARD_MARKER_SIZE_MM = 2.933
BOARD_ARUCO_DICT = cv2.aruco.DICT_4X4_50


def build_board() -> "calibrating.CharucoBoard":
    board = calibrating.CharucoBoard(
        square_xy=BOARD_SQUARES_XY,
        square_size_mm=BOARD_SQUARE_SIZE_MM,
        marker_size_mm=BOARD_MARKER_SIZE_MM,
        aruco_dict_tag=BOARD_ARUCO_DICT,
    )
    # This project's physical board was printed before OpenCV 4.6 and uses the
    # legacy ChArUco marker layout (calibration/config/charuco_11x8.yaml:
    # legacy_pattern: true). calibrating.CharucoBoard never sets this and
    # defaults to the current-pattern layout, which would silently mismatch
    # detected marker IDs against the wrong board squares.
    board.board.setLegacyPattern(True)
    return board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--captures", default=str(PROJECT_ROOT / "captures" / "stereo_1"),
        help="Directory of session subfolders holding <camera>.jpg for both cameras "
             "(default: %(default)s).",
    )
    parser.add_argument("--camera-a", default="rgb_cam1")
    parser.add_argument("--camera-b", default="rgb_cam2")
    parser.add_argument(
        "--min-shared-corners", type=int, default=8,
        help="Drop a session if the two cameras share fewer corners than this "
             "(default: %(default)s, matching this project's own "
             "DEFAULT_MIN_SHARED_CORNERS in calibration/stereo.py). "
             "cv2.stereoCalibrate's internal per-view PnP needs at least 4; "
             "calibrating.Stereo does not filter for this itself.",
    )
    return parser.parse_args()


def drop_weak_views(caml, camr, min_shared_corners: int) -> None:
    for key in list(caml):
        a_points = caml[key].get("image_points", {})
        b_points = camr.get(key, {}).get("image_points", {})
        shared = len(set(a_points) & set(b_points))
        if 0 < shared < min_shared_corners:
            print(f"  dropping {key}: only {shared} shared corners (min {min_shared_corners})")
            del caml[key]


def main() -> int:
    args = parse_args()
    board = build_board()

    paths_a = sorted(glob(f"{args.captures}/*/{args.camera_a}.jpg"))
    paths_b = sorted(glob(f"{args.captures}/*/{args.camera_b}.jpg"))
    if not paths_a or not paths_b:
        raise SystemExit(
            f"No images under {args.captures}/*/{{{args.camera_a},{args.camera_b}}}.jpg"
        )
    print(f"{args.camera_a}: {len(paths_a)} images, {args.camera_b}: {len(paths_b)} images")

    caml = calibrating.Cam(paths_a, board, name=args.camera_a, save_feature_vis=False)
    camr = calibrating.Cam(paths_b, board, name=args.camera_b, save_feature_vis=False)
    print(caml)
    print(camr)

    drop_weak_views(caml, camr, args.min_shared_corners)

    stereo = calibrating.Stereo(caml, camr)
    baseline_mm = float(np.linalg.norm(stereo.t)) * 1000.0

    print("\n=== DIYer22 stereo fit ===")
    print(f"R:\n{stereo.R}")
    print(f"t (m): {stereo.t.reshape(-1)}")
    print(f"baseline: {baseline_mm:.3f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
