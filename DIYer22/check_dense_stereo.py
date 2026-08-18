#!/usr/bin/env python3
"""Check whether DIYer22's SGBM dense stereo finds any depth on this rig.

Independent confirmation of the finding recorded in register_pipeline.py's
docstring and design/config/rig_next_candidate.yaml: at this rig's close
range (confirmed working distance 0.11-0.21 m) and toe-in baseline
(~50 mm), the convergence angle is too large for rectified dense
correspondence -- SGBM was found to recover depth on almost none of the
frame, which is why register_pipeline.py replaced it with plane sweeping.

This fits calibrating.Stereo's own board-based extrinsics (a second
implementation, independent of this project's calibration/stereo.py) and
runs its own SemiGlobalBlockMatching on a real (non-board) scene pair, so a
near-zero valid fraction here points at the rig geometry, not at a bug in
either codebase.

Usage:
    python3 DIYer22/check_dense_stereo.py --scene captures/20260818_152110_605174
"""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "DIYer22"))
sys.path.insert(0, str(PROJECT_ROOT / "DIYer22" / "calibrating"))

import numpy as np  # noqa: E402

if not hasattr(np, "bool8"):
    # boxx's installed build still references the np.bool8 alias NumPy 2.0
    # removed; shim it rather than patch the vendored library.
    np.bool8 = np.bool_

import boxx  # noqa: E402
import calibrating  # noqa: E402
from calibrate_rig import build_board, drop_weak_views  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--captures", default=str(PROJECT_ROOT / "captures" / "stereo_1"),
        help="Session folders to fit the board-based stereo extrinsics from "
             "(default: %(default)s, matching calibrate_rig.py's default).",
    )
    parser.add_argument("--camera-a", default="rgb_cam1")
    parser.add_argument("--camera-b", default="rgb_cam2")
    parser.add_argument(
        "--min-shared-corners", type=int, default=8,
        help="Matches calibrate_rig.py's own default (default: %(default)s).",
    )
    parser.add_argument(
        "--scene", required=True,
        help="Session folder holding a real (non-board) "
             "<camera-a>.jpg/<camera-b>.jpg pair to run SGBM on.",
    )
    parser.add_argument(
        "--max-depth", type=float, default=0.30,
        help="Depths beyond this are dropped as spurious matches, in metres "
             "(default: %(default)s, just past the rig's confirmed 0.11-0.21 m "
             "working range).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    board = build_board()

    paths_a = sorted(glob(f"{args.captures}/*/{args.camera_a}.jpg"))
    paths_b = sorted(glob(f"{args.captures}/*/{args.camera_b}.jpg"))
    if not paths_a or not paths_b:
        raise SystemExit(
            f"No images under {args.captures}/*/{{{args.camera_a},{args.camera_b}}}.jpg"
        )

    caml = calibrating.Cam(paths_a, board, name=args.camera_a, save_feature_vis=False)
    camr = calibrating.Cam(paths_b, board, name=args.camera_b, save_feature_vis=False)
    drop_weak_views(caml, camr, args.min_shared_corners)

    stereo = calibrating.Stereo(caml, camr)
    print(stereo)

    scene_a = f"{args.scene}/{args.camera_a}.jpg"
    scene_b = f"{args.scene}/{args.camera_b}.jpg"
    img1, img2 = boxx.imread(scene_a), boxx.imread(scene_b)

    stereo.set_stereo_matching(
        calibrating.SemiGlobalBlockMatching({}), max_depth=args.max_depth
    )
    result = stereo.get_depth(img1, img2)
    depth = result["unrectify_depth"]
    valid = depth > 0

    print(f"\nScene: {args.scene}")
    print(f"Valid depth fraction (full frame):        {valid.mean() * 100:.2f}%")
    print(
        f"Valid depth fraction (rectified frame only): "
        f"{(result['rectify_depth'] > 0).mean() * 100:.2f}%"
    )
    if valid.any():
        print(f"Depth range where valid: {depth[valid].min():.3f} - {depth[valid].max():.3f} m")
    else:
        print("No valid depth recovered anywhere in the frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
