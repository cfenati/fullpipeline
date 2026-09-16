#!/usr/bin/env python3
"""Register camera B onto camera A using Fast-FoundationStereo depth.

Rectify, run the network, lift disparity to Z in camera A (Q triangulation,
same formula as triangulate.z_from_disparity), warp B onto A. Invalid
disparity stays NaN -- no constant-depth fill.

Example (run inside the `ffs` conda env, which has the CUDA build of torch):
    python register_foundationstereo.py --session captures/20260818_152114_103039
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FFS_ROOT = PROJECT_ROOT / "Fast-FoundationStereo"
DEFAULT_MODEL = FFS_ROOT / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"

from calibrate_cameras import load_config, resolve_path  # noqa: E402
from calibration.stereo import StereoExtrinsics, rectify, rectify_maps  # noqa: E402
from registration_io import (  # noqa: E402
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_REGISTRATION_OUTPUT_DIR,
    default_extrinsics_path,
    load_session_images,
    save_preview,
    undistort_pair,
    warp_using_z,
)

# Board disparity at full rectified res is 1250-1898 px (report.txt). Scale 0.2
# -> width 931 (<1000) and disparity 250-380, which fits the 23-36-37
# checkpoint's native max_disp of 416. Closer than ~0.11 m overflows it.
DEFAULT_SCALE = 0.2
DEFAULT_MAX_DISP = 416
DEFAULT_VALID_ITERS = 8


def parse_args() -> argparse.Namespace:
    reg_config = load_config().get("registration", {}) or {}
    parser = argparse.ArgumentParser(
        description="Register camera B onto camera A with Fast-FoundationStereo.",
    )
    parser.add_argument(
        "--session", required=True,
        help="Capture session folder holding <camera-a>.jpg and <camera-b>.jpg.",
    )
    parser.add_argument("--camera-a", default="rgb_cam1",
                        help="Left-side / target camera (default: %(default)s).")
    parser.add_argument("--camera-b", default="rgb_cam2",
                        help="Right-side / source camera, warped onto A "
                             "(default: %(default)s).")
    parser.add_argument(
        "--extrinsics", default=None,
        help="Stereo extrinsics JSON (default: geometric_calibration.extrinsics_<a>_<b>).",
    )
    parser.add_argument(
        "--model-dir", default=str(DEFAULT_MODEL),
        help="Fast-FoundationStereo serialized checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--scale", type=float, default=DEFAULT_SCALE,
        help="Resize factor applied to the rectified pair before inference. "
             "0.2 puts width under 1000 and working-range disparity inside "
             f"max_disp={DEFAULT_MAX_DISP} (default: %(default)s).",
    )
    parser.add_argument(
        "--valid-iters", type=int, default=DEFAULT_VALID_ITERS,
        help="Refinement iterations; 4 is faster, 8 is the demo default "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--max-disp", type=int, default=DEFAULT_MAX_DISP,
        help="Maximum disparity at the scaled resolution (default: %(default)s).",
    )
    parser.add_argument(
        "--hiera", type=int, default=0,
        help="Hierarchical inference (0/1). Leave off unless disparity still "
             "overflows --max-disp after scaling.",
    )
    parser.add_argument(
        "--crop", choices=("union", "intersection", "none"), default="union",
        help="Crop the rectified pair before inference so black borders from "
             "the ~18 deg toe-in do not dominate the network. 'intersection' "
             "keeps only pixels seen by both cameras (default: %(default)s).",
    )
    parser.add_argument(
        "--no-zero-disparity", dest="zero_disparity", action="store_false",
        help="Drop cv2.CALIB_ZERO_DISPARITY when rectifying. On this rig's "
             "toe-in that roughly doubles valid_fraction at alpha=1 (measured "
             "~35%% -> ~64%%), at the cost of a nonzero disparity-at-infinity "
             "offset that shifts the expected disparity range (see the "
             "printed expected-disparity line). Default keeps zero_disparity "
             "on, matching the previous, unconditional behaviour.",
    )
    parser.set_defaults(zero_disparity=True)
    parser.add_argument(
        "--prep-only", action="store_true",
        help="Write rectified PNG + K.txt and exit, so you can run "
             "Fast-FoundationStereo/scripts/run_demo.py yourself.",
    )
    parser.add_argument("--out", "--output", dest="output", default=None,
                        help="Output directory (default: registration.output_dir in "
                             f"config, else {DEFAULT_REGISTRATION_OUTPUT_DIR}) "
                             "/ <session>/foundationstereo.")
    return parser.parse_args()


def disparity_from_depth(Q: np.ndarray, depth_m: float) -> float:
    """Invert Rectification.depth_from_disparity: disparity (px) at a given depth.

    Uses the rectified pair's own Q rather than a bare f*B/Z, so it stays
    correct when zero_disparity=False shifts the principal points and makes
    disparity_at_infinity_px nonzero.
    """
    return float((Q[2, 3] / depth_m - Q[3, 3]) / Q[3, 2])


def write_intrinsic_file(path: Path, camera_matrix: np.ndarray, baseline_m: float) -> None:
    """Fast-FoundationStereo K.txt: flattened 3x3, then baseline in metres."""
    flat = " ".join(f"{float(v):.12g}" for v in np.asarray(camera_matrix, dtype=np.float64).reshape(-1))
    path.write_text(f"{flat}\n{float(baseline_m):.12g}\n", encoding="utf-8")


def rectify_session(
    image_a: np.ndarray,
    image_b: np.ndarray,
    extrinsics: StereoExtrinsics,
    alpha: float,
    reference_depth: Optional[float],
    zero_disparity: bool = True,
) -> Tuple[np.ndarray, np.ndarray, object, np.ndarray, np.ndarray]:
    rectification = rectify(extrinsics, alpha=alpha, reference_depth=reference_depth,
                            zero_disparity=zero_disparity)
    map_a = rectify_maps(extrinsics, rectification, "a")
    map_b = rectify_maps(extrinsics, rectification, "b")
    left = cv2.remap(image_a, map_a[0], map_a[1], interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    right = cv2.remap(image_b, map_b[0], map_b[1], interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    src_w, src_h = extrinsics.image_size_a
    valid_left = valid_from_maps(map_a[0], map_a[1], src_w, src_h)
    valid_right = valid_from_maps(map_b[0], map_b[1], extrinsics.image_size_b[0], extrinsics.image_size_b[1])
    return left, right, rectification, valid_left, valid_right


def valid_from_maps(map_x: np.ndarray, map_y: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    return (map_x >= 0) & (map_x <= src_w - 1) & (map_y >= 0) & (map_y <= src_h - 1)


def mask_bbox(mask: np.ndarray, pad: int, bounds: Tuple[int, int]) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise SystemExit("Rectified pair has no valid pixels; check extrinsics / camera order.")
    width, height = bounds
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, width)
    y1 = min(int(ys.max()) + pad + 1, height)
    return x0, y0, x1, y1


def crop_projections(
    P1: np.ndarray, P2: np.ndarray, Q: np.ndarray, x0: int, y0: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift rectified principal points after cropping the left/right images equally."""
    P1 = np.array(P1, dtype=np.float64, copy=True)
    P2 = np.array(P2, dtype=np.float64, copy=True)
    Q = np.array(Q, dtype=np.float64, copy=True)
    P1[0, 2] -= x0
    P1[1, 2] -= y0
    P2[0, 2] -= x0
    P2[1, 2] -= y0
    Q[0, 3] += x0  # -cx_new = -(cx - x0)
    Q[1, 3] += y0
    return P1, P2, Q


def scale_Q(Q: np.ndarray, scale: float) -> np.ndarray:
    """Pixel-unit entries of OpenCV's Q scale with the image; Tx does not."""
    scaled = np.array(Q, dtype=np.float64, copy=True)
    scaled[0, 3] *= scale  # -cx
    scaled[1, 3] *= scale  # -cy
    scaled[2, 3] *= scale  # f
    scaled[3, 3] *= scale  # (cx - cx2) / Tx
    return scaled


def undistorted_to_rectified_maps(
    camera_matrix_a: np.ndarray,
    R1: np.ndarray,
    P1: np.ndarray,
    size_a: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample locations in the rectified left image for every undistorted A pixel."""
    width_a, height_a = size_a
    grid = pixel_grid(width_a, height_a)
    rays = np.linalg.inv(camera_matrix_a) @ grid
    projected = P1[:3, :3] @ (R1 @ rays)
    w = projected[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        map_x = (projected[0] / w).reshape(height_a, width_a)
        map_y = (projected[1] / w).reshape(height_a, width_a)
    valid = w.reshape(height_a, width_a) > 1e-9
    map_x = np.nan_to_num(map_x, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)
    map_y = np.nan_to_num(map_y, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)
    return map_x, map_y, valid


def lift_disparity_to_undistorted_a(
    disparity_scaled: np.ndarray,
    Q: np.ndarray,
    P1: np.ndarray,
    scale: float,
    R1: np.ndarray,
    rectified_size: Tuple[int, int],
    camera_matrix_a: np.ndarray,
    size_a: Tuple[int, int],
    depth_min: float,
    depth_max: float,
    valid_rectified: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lift scaled rectified disparity onto undistorted camera-A Z.

    ``cv2.reprojectImageTo3D`` is the dense form of triangulate.z_from_disparity.
    Pixels without a valid disparity stay NaN.
    """
    Q_scaled = scale_Q(Q, scale)
    disp = np.clip(disparity_scaled.astype(np.float32), 0.0, None)
    xyz_scaled = cv2.reprojectImageTo3D(disp, Q_scaled)
    finite = np.isfinite(xyz_scaled).all(axis=2) & (disp > 0)

    width_r, height_r = rectified_size
    xyz_full = cv2.resize(xyz_scaled, (width_r, height_r), interpolation=cv2.INTER_NEAREST)
    finite_full = cv2.resize(
        finite.astype(np.uint8), (width_r, height_r), interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    if valid_rectified is not None:
        if valid_rectified.shape[:2] != (height_r, width_r):
            raise SystemExit("valid_rectified mask does not match the cropped rectified size")
        finite_full &= valid_rectified.astype(bool)

    points_a = (R1.T @ xyz_full.reshape(-1, 3).T).T.reshape(height_r, width_r, 3)
    z_rect_grid = points_a[:, :, 2]

    map_x, map_y, ray_valid = undistorted_to_rectified_maps(
        camera_matrix_a, R1, P1, size_a,
    )
    in_frame = (
        (map_x >= 0) & (map_x <= width_r - 1)
        & (map_y >= 0) & (map_y <= height_r - 1)
        & ray_valid
    )
    z_a = cv2.remap(
        z_rect_grid.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    sampled_finite = cv2.remap(
        finite_full.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    ) > 0.5

    valid = in_frame & sampled_finite & np.isfinite(z_a) & (z_a >= depth_min) & (z_a <= depth_max)
    depth_map = np.where(valid, z_a.astype(np.float64), np.nan)
    filled_mask = ~valid
    return depth_map, filled_mask


def load_ffs_model(model_path: Path, valid_iters: int, max_disp: int):
    if str(FFS_ROOT) not in sys.path:
        sys.path.insert(0, str(FFS_ROOT))
    import torch

    with open(model_path.parent / "cfg.yaml", "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["valid_iters"] = valid_iters
    cfg["max_disp"] = max_disp

    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()
    return model, cfg


def run_foundationstereo(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    model,
    scale: float,
    valid_iters: int,
    hiera: bool,
) -> np.ndarray:
    if str(FFS_ROOT) not in sys.path:
        sys.path.insert(0, str(FFS_ROOT))
    import torch
    from core.utils.utils import InputPadder
    from Utils import AMP_DTYPE

    if scale != 1.0:
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        left_bgr = cv2.resize(left_bgr, None, fx=scale, fy=scale, interpolation=interpolation)
        right_bgr = cv2.resize(right_bgr, (left_bgr.shape[1], left_bgr.shape[0]),
                               interpolation=interpolation)

    height, width = left_bgr.shape[:2]
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    logging.info(
        "Fast-FoundationStereo forward at %dx%d (1st call can be slow: torch.compile)",
        width, height,
    )
    with torch.autograd.set_grad_enabled(False), torch.amp.autocast(
        "cuda", enabled=True, dtype=AMP_DTYPE,
    ):
        if hiera:
            disp = model.run_hierachical(
                img0, img1, iters=valid_iters, test_mode=True, small_ratio=0.5,
            )
        else:
            disp = model.forward(
                img0, img1, iters=valid_iters, test_mode=True,
                optimize_build_volume="pytorch1",
            )
    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(height, width)
    return np.clip(disp, 0.0, None)


def write_report(
    path: Path,
    session_dir: Path,
    camera_a: str,
    camera_b: str,
    extrinsics_path: Path,
    rectification,
    scale: float,
    valid_iters: int,
    max_disp: int,
    size_a: Tuple[int, int],
    filled_mask: np.ndarray,
    depth_map: np.ndarray,
    remap_valid: np.ndarray,
) -> None:
    n_pixels = size_a[0] * size_a[1]
    n_confident = int((~filled_mask).sum())
    confident = depth_map[~filled_mask]
    lines = [
        f"Registration: {camera_b} -> {camera_a} (Fast-FoundationStereo)",
        "=" * 66,
        "",
        f"  session:            {session_dir}",
        f"  extrinsics:         {extrinsics_path}",
        f"  undistorted size:   {size_a[0]}x{size_a[1]}",
        f"  rectified size:     {rectification.image_size[0]}x{rectification.image_size[1]}",
        f"  rectified focal:    {rectification.focal_px:.2f} px",
        f"  rectified baseline: {rectification.baseline_m * 1000:.3f} mm",
        f"  valid fraction:     {rectification.valid_fraction[0] * 100:.1f}% / "
        f"{rectification.valid_fraction[1] * 100:.1f}%",
        f"  degenerate rectify: {rectification.degenerate}",
        "",
        "INFERENCE",
        "-" * 66,
        f"  scale:              {scale}",
        f"  valid_iters:        {valid_iters}",
        f"  max_disp:           {max_disp}",
        "",
        "COVERAGE",
        "-" * 66,
        f"  camera A pixels:              {n_pixels}",
        f"  FFS-valid depth:              {n_confident} "
        f"({n_confident / n_pixels * 100:.1f}%)",
        f"  no triangulated Z:            {int(filled_mask.sum())}",
        f"  outside shared FOV:           {int((~remap_valid).sum())}",
        "",
        "DEPTH MAP (FFS-valid pixels only)",
        "-" * 66,
    ]
    if n_confident:
        lines.append(
            f"  min / mean / max:   {confident.min():.4f} / "
            f"{confident.mean():.4f} / {confident.max():.4f} m"
        )
    else:
        lines.append("  no valid FFS depths — check left/right order and --scale/--max-disp")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.scale <= 0 or args.scale > 1:
        raise SystemExit(f"--scale must be in (0, 1], got {args.scale}")

    reg_config = load_config().get("registration", {}) or {}
    working_depth = float(reg_config.get("default_depth", 0.168))
    depth_range = reg_config.get("depth_range", [DEFAULT_DEPTH_MIN, DEFAULT_DEPTH_MAX])
    depth_min, depth_max = float(depth_range[0]), float(depth_range[1])
    alpha = float(reg_config.get("rectification_alpha", 1.0))

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
    output_dir = resolve_path(output_root) / session_dir.name / "foundationstereo"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_a, image_b = load_session_images(session_dir, args.camera_a, args.camera_b)
    print(
        f"Rectifying {args.camera_a} (left) / {args.camera_b} (right) "
        f"from JPEG -> PNG, alpha={alpha}"
    )
    left, right, rectification, valid_left, valid_right = rectify_session(
        image_a, image_b, extrinsics, alpha=alpha, reference_depth=working_depth,
        zero_disparity=args.zero_disparity,
    )
    P1 = np.array(rectification.P1, dtype=np.float64, copy=True)
    P2 = np.array(rectification.P2, dtype=np.float64, copy=True)
    Q = np.array(rectification.Q, dtype=np.float64, copy=True)
    crop_box = (0, 0, left.shape[1], left.shape[0])
    if args.crop != "none":
        mask = (valid_left & valid_right) if args.crop == "intersection" else (valid_left | valid_right)
        crop_box = mask_bbox(mask, pad=16, bounds=(left.shape[1], left.shape[0]))
        x0, y0, x1, y1 = crop_box
        left = left[y0:y1, x0:x1]
        right = right[y0:y1, x0:x1]
        valid_left = valid_left[y0:y1, x0:x1]
        valid_right = valid_right[y0:y1, x0:x1]
        P1, P2, Q = crop_projections(P1, P2, Q, x0, y0)
        print(f"Cropped rectified pair to {left.shape[1]}x{left.shape[0]} ({args.crop} bbox)")

    K_rect = np.array(P1[:3, :3], dtype=np.float64)
    cropped_size = (left.shape[1], left.shape[0])

    cv2.imwrite(str(output_dir / "left.png"), left)
    cv2.imwrite(str(output_dir / "right.png"), right)
    write_intrinsic_file(output_dir / "K.txt", K_rect, rectification.baseline_m)

    scaled_w = int(round(left.shape[1] * args.scale))
    scaled_h = int(round(left.shape[0] * args.scale))
    expected_disp = abs(disparity_from_depth(Q, working_depth)) * args.scale
    # Disparity magnitude is monotonic in 1/Z, so the range's near/far edges
    # bracket the worst case across the whole confirmed working range - not
    # just at the single reference depth (matters once zero_disparity=False
    # makes disparity_at_infinity_px nonzero and shifts the whole curve).
    disp_near = disparity_from_depth(Q, depth_min) * args.scale
    disp_far = disparity_from_depth(Q, depth_max) * args.scale
    worst_disp = max(abs(disp_near), abs(disp_far))
    print(
        f"Rectified {rectification.image_size[0]}x{rectification.image_size[1]}, valid "
        f"{rectification.valid_fraction[0] * 100:.1f}% / "
        f"{rectification.valid_fraction[1] * 100:.1f}%, "
        f"focal {rectification.focal_px:.1f} px, "
        f"baseline {rectification.baseline_m * 1000:.2f} mm, "
        f"disparity_at_infinity {rectification.disparity_at_infinity_px:.1f} px"
    )
    print(
        f"Inference size ~{scaled_w}x{scaled_h}; disparity at {working_depth:.3f} m "
        f"≈ {expected_disp:.0f} px, signed range over [{depth_min:.3f}, {depth_max:.3f}] m "
        f"≈ [{disp_near:.0f}, {disp_far:.0f}] px (max_disp={args.max_disp})"
    )
    if disp_near < 0 or disp_far < 0:
        # FFS (like any network trained on the standard convention) only ever
        # predicts disparity >= 0. cv2.CALIB_ZERO_DISPARITY guarantees the
        # zero-disparity plane sits at infinity, so every finite real depth is
        # positive; dropping it (zero_disparity=False) lets OpenCV instead
        # place that plane wherever maximizes FOV overlap, which on this
        # rig's tight convergence lands *closer* than the whole working
        # range - so every real depth needs negative disparity, which the
        # network cannot output. This is a sign flip, not a magnitude
        # overflow: no --scale/--max-disp choice fixes it. Measured effect:
        # 39.6% FFS-valid depth with zero_disparity (default) vs 0.0% without.
        message = (
            "FATAL: signed disparity over the working range is negative "
            f"({disp_near:.0f} to {disp_far:.0f} px) - Fast-FoundationStereo "
            "only predicts disparity >= 0, so no pixel in this depth range is "
            "recoverable. Drop --no-zero-disparity (keep zero_disparity on)."
        )
        if args.prep_only:
            print(message.replace("FATAL", "WARNING", 1))
        else:
            raise SystemExit(message)
    if worst_disp > args.max_disp * 0.95:
        print(
            "WARNING: expected disparity is at/over --max-disp. Lower --scale "
            "or raise --max-disp, otherwise near surfaces will saturate."
        )

    demo_cmd = (
        f"python scripts/run_demo.py --model_dir {args.model_dir} "
        f"--left_file {output_dir / 'left.png'} --right_file {output_dir / 'right.png'} "
        f"--intrinsic_file {output_dir / 'K.txt'} --out_dir {output_dir / 'demo'} "
        f"--remove_invisible 0 --denoise_cloud 0 --scale {args.scale} --get_pc 0 "
        f"--valid_iters {args.valid_iters} --max_disp {args.max_disp} --zfar {depth_max}"
    )
    (output_dir / "run_demo_command.txt").write_text(demo_cmd + "\n", encoding="utf-8")

    if args.prep_only:
        print(f"Wrote rectified PNG + K.txt to {output_dir}")
        print("Run inference from Fast-FoundationStereo/ with:")
        print(f"  {demo_cmd}")
        return 0

    model_path = Path(args.model_dir)
    if not model_path.is_absolute():
        model_path = (FFS_ROOT / args.model_dir).resolve()
        if not model_path.exists():
            model_path = resolve_path(args.model_dir)
    if not model_path.exists():
        raise SystemExit(f"Checkpoint not found: {model_path}")

    model, _cfg = load_ffs_model(model_path, args.valid_iters, args.max_disp)
    disparity = run_foundationstereo(
        left, right, model, scale=args.scale,
        valid_iters=args.valid_iters, hiera=bool(args.hiera),
    )
    np.save(output_dir / "disparity_scaled.npy", disparity)

    vis = cv2.applyColorMap(
        np.clip(disparity / max(disparity.max(), 1e-6) * 255, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    left_scaled = cv2.resize(left, (disparity.shape[1], disparity.shape[0]),
                             interpolation=cv2.INTER_AREA)
    right_scaled = cv2.resize(right, (disparity.shape[1], disparity.shape[0]),
                              interpolation=cv2.INTER_AREA)
    cv2.imwrite(
        str(output_dir / "disp_vis.png"),
        np.concatenate([left_scaled, right_scaled, vis], axis=1),
    )

    undistorted_a, undistorted_b = undistort_pair(image_a, image_b, extrinsics)
    size_a = (undistorted_a.shape[1], undistorted_a.shape[0])
    size_b = (undistorted_b.shape[1], undistorted_b.shape[0])
    depth_map, filled_mask = lift_disparity_to_undistorted_a(
        disparity, Q, P1, args.scale,
        rectification.R1, cropped_size,
        extrinsics.camera_matrix_a, size_a,
        depth_min, depth_max,
        valid_rectified=valid_left & valid_right,
    )
    warped_color, remap_valid = warp_using_z(
        depth_map, undistorted_b,
        extrinsics.camera_matrix_a, extrinsics.camera_matrix_b,
        extrinsics.R, extrinsics.T, size_a, size_b,
    )

    np.save(output_dir / "depth_m.npy", depth_map)
    np.save(output_dir / "filled_mask.npy", filled_mask)
    cv2.imwrite(str(output_dir / "warped.jpg"), warped_color)
    save_preview(
        output_dir / "preview.jpg", undistorted_a, warped_color, depth_map,
        depth_min, depth_max,
    )
    blend = undistorted_a.copy()
    use = ~filled_mask & remap_valid
    blend[use] = (
        0.5 * undistorted_a[use].astype(np.float32)
        + 0.5 * warped_color[use].astype(np.float32)
    ).astype(np.uint8)
    cv2.imwrite(str(output_dir / "overlay_blend.jpg"), blend)

    write_report(
        output_dir / "report.txt",
        session_dir, args.camera_a, args.camera_b, extrinsics_path,
        rectification, args.scale, args.valid_iters, args.max_disp,
        size_a, filled_mask, depth_map, remap_valid,
    )
    summary = {
        "scale": args.scale,
        "valid_iters": args.valid_iters,
        "max_disp": args.max_disp,
        "rectified_size": list(rectification.image_size),
        "valid_fraction": list(rectification.valid_fraction),
        "ffs_valid_fraction": float((~filled_mask).mean()),
        "depth_min_m": float(depth_map[~filled_mask].min()) if (~filled_mask).any() else None,
        "depth_mean_m": float(depth_map[~filled_mask].mean()) if (~filled_mask).any() else None,
        "depth_max_m": float(depth_map[~filled_mask].max()) if (~filled_mask).any() else None,
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    n_valid = int((~filled_mask).sum())
    print(
        f"FFS-valid depth on {n_valid / (size_a[0] * size_a[1]) * 100:.1f}% of camera A "
        f"({n_valid} px)"
    )
    print(f"Saved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
