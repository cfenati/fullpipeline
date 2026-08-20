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
import cv2  # noqa: E402
from calibrate_rig import build_board, drop_weak_views  # noqa: E402


class TunedSGBM(calibrating.MetaStereoMatching):
    """SemiGlobalBlockMatching with the search window sized to this rig's
    actual close-range disparity, instead of calibrating's own demo-tuned
    default (min/max 2-220 px, sized for a normal-range rig).

    calibrating.Stereo.rectify() downsamples to `max_size` before matching
    and rescales the result back, so min/max here are given in full-resolution
    pixels (matching stereo_calibrate.py report.txt's own "disparity seen on
    the board" line) and converted to the downsampled scale internally.
    """

    def __init__(self, min_disparity_px: float, max_disparity_px: float,
                 max_size: int = 1000, block_size: int = 11):
        self.max_size = max_size
        self.min_disparity_px = min_disparity_px
        self.max_disparity_px = max_disparity_px
        self.block_size = block_size

    def __call__(self, img1, img2):
        resize_ratio = min(self.max_size / max(img1.shape[:2]), 1)
        simg1, simg2 = boxx.resize(img1, resize_ratio), boxx.resize(img2, resize_ratio)

        block_size = self.block_size
        min_disp = int(np.floor(self.min_disparity_px * resize_ratio / 16)) * 16
        max_disp = int(np.ceil(self.max_disparity_px * resize_ratio / 16)) * 16
        num_disp = max(16, max_disp - min_disp)
        stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=5,
            speckleWindowSize=200,
            speckleRange=2,
            disp12MaxDiff=0,
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
        )
        sdisparity = stereo_sgbm.compute(simg1, simg2).astype(np.float32).clip(0)
        sdisparity[sdisparity < min_disp * 16] = 0
        disparity = (
            boxx.resize(sdisparity / 16.0, img1.shape[:2])
            * img1.shape[1] / simg1.shape[1]
        )
        return disparity


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
    parser.add_argument(
        "--stock-sgbm", action="store_true",
        help="Use calibrating's untouched SemiGlobalBlockMatching (2-220 px "
             "search window) instead of TunedSGBM. Left in to reproduce the "
             "first, mismatched-window run for comparison.",
    )
    parser.add_argument(
        "--min-disparity-px", type=float, default=1100,
        help="TunedSGBM's search window floor, full-res px (default: %(default)s, "
             "just under report.txt's board disparity at 0.19 m).",
    )
    parser.add_argument(
        "--max-disparity-px", type=float, default=2150,
        help="TunedSGBM's search window ceiling, full-res px (default: %(default)s, "
             "just over report.txt's board disparity at 0.11 m).",
    )
    parser.add_argument(
        "--out", "--output", dest="output", default=None,
        help="Output directory (default: registration/results/<scene name>/dense_stereo).",
    )
    return parser.parse_args()


def render_depth_visualization(
    rectify_img1: np.ndarray, rectify_img2: np.ndarray, rectify_depth: np.ndarray, max_depth: float,
) -> np.ndarray:
    """rectified left | rectified right | colorized depth (TURBO, near=warm), BGR
    for cv2.imwrite. boxx.imread returns RGB, unlike cv2.imread, so the two
    photo panels need an explicit RGB->BGR swap or colors come out wrong."""
    valid = rectify_depth > 0
    normalized = np.zeros(rectify_depth.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(rectify_depth, 0, max_depth)
        normalized[valid] = np.clip(255 - clipped[valid] / max_depth * 255, 0, 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    depth_vis[~valid] = 0

    def to_bgr(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(
            image, cv2.COLOR_RGB2BGR if image.ndim == 3 else cv2.COLOR_GRAY2BGR
        )

    panels = [to_bgr(rectify_img1), to_bgr(rectify_img2), depth_vis]
    height = max(panel.shape[0] for panel in panels)
    padded = [
        cv2.copyMakeBorder(panel, 0, height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        for panel in panels
    ]
    return np.hstack(padded)


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

    if args.stock_sgbm:
        matcher = calibrating.SemiGlobalBlockMatching({})
    else:
        matcher = TunedSGBM(args.min_disparity_px, args.max_disparity_px)
    stereo.set_stereo_matching(matcher, max_depth=args.max_depth)
    result = stereo.get_depth(img1, img2)
    depth = result["unrectify_depth"]
    valid = depth > 0

    rectified_valid_fraction = (result["rectify_depth"] > 0).mean()
    lines = [
        f"Dense stereo check ({'stock' if args.stock_sgbm else 'tuned'} SGBM)",
        f"Scene: {args.scene}",
        f"Valid depth fraction (full frame):        {valid.mean() * 100:.2f}%",
        f"Valid depth fraction (rectified frame only): {rectified_valid_fraction * 100:.2f}%",
    ]
    if valid.any():
        lines.append(f"Depth range where valid: {depth[valid].min():.3f} - {depth[valid].max():.3f} m")
    else:
        lines.append("No valid depth recovered anywhere in the frame.")
    if not args.stock_sgbm:
        lines.append(
            f"TunedSGBM search window: {args.min_disparity_px}-{args.max_disparity_px} px (full-res)"
        )
    print("\n" + "\n".join(lines))

    output_dir = (
        Path(args.output) if args.output
        else PROJECT_ROOT / "registration" / "results" / Path(args.scene).name / "dense_stereo"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    vis = render_depth_visualization(
        result["rectify_img1"], result["rectify_img2"], result["rectify_depth"], args.max_depth,
    )
    cv2.imwrite(str(output_dir / "depth_vis.jpg"), vis)
    np.save(output_dir / "depth_m.npy", depth)
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved depth_vis.jpg (rectified left | right | colorized depth), "
          f"depth_m.npy, report.txt to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
