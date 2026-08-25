"""Shared I/O for registration scripts. No geometry, no assumed depth."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.tri as mtri
import numpy as np

from calibrate_cameras import DEFAULT_OUTPUT_DIR, load_config
from calibration.stereo import StereoExtrinsics

DEFAULT_REGISTRATION_OUTPUT_DIR = "registration/results"
PREVIEW_PANEL_WIDTH = 1000
DEFAULT_DEPTH_MIN = 0.11  # chirality / plausibility filter on triangulated Z, not a plane
DEFAULT_DEPTH_MAX = 0.21


def default_extrinsics_path(camera_a: str, camera_b: str) -> str:
    config = load_config().get("geometric_calibration", {}) or {}
    output_root = config.get("output_dir", DEFAULT_OUTPUT_DIR)
    key = f"extrinsics_{camera_a}_{camera_b}"
    return config.get(key, f"{output_root}/stereo_{camera_a}_{camera_b}/extrinsics.json")


def load_session_images(
    session_dir: Path, camera_a: str, camera_b: str,
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
    image_a: np.ndarray, image_b: np.ndarray, extrinsics: StereoExtrinsics,
) -> Tuple[np.ndarray, np.ndarray]:
    """Undistort with cv2.undistort(img, K, dist) -- no newCameraMatrix.

    Omitting newCameraMatrix reuses the input K, so K loaded from extrinsics.json
    still applies to these images. CameraIntrinsics.undistort() is the wrong
    helper here: it calls getOptimalNewCameraMatrix and drops the new K.
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


def warp_from_correspondences(
    pts_a: np.ndarray, pts_b: np.ndarray, image_b: np.ndarray, size_a: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise-affine warp of B onto A, exact at every correspondence.

    No plane, no assumed depth: barycentric interpolation of the matched
    pixel pairs (the 2-D shadow of the triangulated mesh). Pixels outside
    the match hull are not filled. Returns (warped, covered).
    """
    if len(pts_a) < 3:
        raise SystemExit("Need at least 3 correspondences to build a triangulation.")
    width_a, height_a = size_a
    triangulation = mtri.Triangulation(pts_a[:, 0], pts_a[:, 1])
    interpolate_x = mtri.LinearTriInterpolator(triangulation, pts_b[:, 0].astype(np.float64))
    interpolate_y = mtri.LinearTriInterpolator(triangulation, pts_b[:, 1].astype(np.float64))
    grid_x, grid_y = np.meshgrid(np.arange(width_a), np.arange(height_a))
    sampled_x = interpolate_x(grid_x, grid_y)
    sampled_y = interpolate_y(grid_x, grid_y)
    covered = ~np.ma.getmaskarray(sampled_x)
    map_x = np.ma.filled(sampled_x, -1.0).astype(np.float32)
    map_y = np.ma.filled(sampled_y, -1.0).astype(np.float32)
    warped = cv2.remap(
        image_b, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    return warped, covered


def warp_using_z(
    z_a: np.ndarray, image_b: np.ndarray,
    camera_matrix_a: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray,
    size_a: Tuple[int, int], size_b: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp B onto A given a per-pixel Z in camera A (metres).

    Each A pixel is back-projected along its ray to that Z, transformed with
    R,T, and sampled in B. Z must come from triangulation / disparity -- a
    constant fill is a plane assumption and does not belong here. Pixels with
    non-finite or non-positive Z are left invalid.
    """
    width_a, height_a = size_a
    width_b, height_b = size_b
    u, v = np.meshgrid(
        np.arange(width_a, dtype=np.float64), np.arange(height_a, dtype=np.float64),
    )
    grid = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=0)
    depth = np.asarray(z_a, dtype=np.float64).reshape(1, -1)
    points_a = (np.linalg.inv(camera_matrix_a) @ grid) * depth
    points_b = R @ points_a + np.asarray(T, dtype=np.float64).reshape(3, 1)
    projected = camera_matrix_b @ points_b
    w = projected[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        map_x = (projected[0] / w).reshape(height_a, width_a)
        map_y = (projected[1] / w).reshape(height_a, width_a)
    finite_z = np.isfinite(z_a) & (np.asarray(z_a) > 1e-9)
    valid = (
        finite_z
        & (w.reshape(height_a, width_a) > 1e-9)
        & (map_x >= 0) & (map_x <= width_b - 1)
        & (map_y >= 0) & (map_y <= height_b - 1)
    )
    map_x = np.nan_to_num(map_x, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)
    map_y = np.nan_to_num(map_y, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)
    warped = cv2.remap(
        image_b, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    return warped, valid


def labelled(image: np.ndarray, text: str) -> np.ndarray:
    frame = image.copy()
    cv2.putText(frame, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] <= width:
        return image
    scale = width / image.shape[1]
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def save_preview(
    path: Path, color_a: np.ndarray, warped_color: np.ndarray, depth_map: np.ndarray,
    depth_min: float, depth_max: float,
) -> np.ndarray:
    finite = np.isfinite(depth_map)
    normalized = np.zeros_like(depth_map, dtype=np.float64)
    if finite.any():
        lo, hi = depth_min, depth_max
        normalized[finite] = np.clip((depth_map[finite] - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    depth_colormap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_colormap[~finite] = 0
    panels = [
        resize_to_width(labelled(color_a, "camera A"), PREVIEW_PANEL_WIDTH),
        resize_to_width(labelled(warped_color, "B warped onto A"), PREVIEW_PANEL_WIDTH),
        resize_to_width(labelled(depth_colormap, "triangulated Z"), PREVIEW_PANEL_WIDTH),
    ]
    cv2.imwrite(str(path), np.hstack(panels))
    return depth_colormap
