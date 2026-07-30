"""Pinhole camera model for multi-camera rig design.

Conventions
-----------
World frame : right-handed, metres, +Z up.
Camera frame: OpenCV convention, +X right in the image, +Y down in the image,
              +Z along the optical axis pointing out of the lens.

A pose is stored as ``R`` (3x3) and ``t`` (3,) forming

    T_world_from_camera = [[R, t], [0, 1]]

so ``t`` is the camera centre in world coordinates and the columns of ``R`` are
the camera axes expressed in the world frame.

A pixel ``(u, v)`` at depth ``z`` back-projects to

    p_cam = z * K^-1 @ [u, v, 1]
    p_world = R @ p_cam + t

The four image corners back-projected at the near and far depths give the exact
view frustum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# Intrinsics builders
# --------------------------------------------------------------------------- #
def _principal_point(width: int, height: int) -> Tuple[float, float]:
    """Centre of the sensor in pixel coordinates (pixel centres on integers)."""
    return (width / 2.0 - 0.5, height / 2.0 - 0.5)


def make_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def K_from_fov(width: int, height: int, hfov_deg: float,
               vfov_deg: Optional[float] = None) -> np.ndarray:
    """Intrinsics from field of view. If ``vfov_deg`` is None, square pixels."""
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    fy = fx if vfov_deg is None else (height / 2.0) / np.tan(np.radians(vfov_deg) / 2.0)
    cx, cy = _principal_point(width, height)
    return make_K(fx, fy, cx, cy)


def K_from_sensor(width: int, height: int, focal_length_mm: float,
                  pixel_pitch_um: Optional[float] = None,
                  sensor_width_mm: Optional[float] = None) -> np.ndarray:
    """Intrinsics from a datasheet: focal length plus pixel pitch or sensor width."""
    if pixel_pitch_um is not None:
        pitch_mm = pixel_pitch_um / 1000.0
    elif sensor_width_mm is not None:
        pitch_mm = sensor_width_mm / float(width)
    else:
        raise ValueError("give either pixel_pitch_um or sensor_width_mm")
    f_px = focal_length_mm / pitch_mm
    cx, cy = _principal_point(width, height)
    return make_K(f_px, f_px, cx, cy)


def K_from_calibration(path, resolution: Optional[Sequence[int]] = None
                       ) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
    """Load ``K`` from a ``calibrate_cameras.py`` intrinsics JSON.

    If ``resolution`` differs from the calibrated one the intrinsics are scaled
    by the width ratio, which preserves the horizontal field of view. The
    principal point is re-centred because the crop offset of the other sensor
    mode is unknown; vertical FOV therefore follows the requested aspect ratio.
    """
    data = json.loads(Path(path).read_text())
    K = np.asarray(data["camera_matrix"], dtype=float)
    cal_w, cal_h = (int(v) for v in data["resolution"])
    dist = np.asarray(data.get("distortion", []), dtype=float)

    if resolution is None:
        return K, (cal_w, cal_h), dist

    out_w, out_h = int(resolution[0]), int(resolution[1])
    scale = out_w / float(cal_w)
    cx, cy = _principal_point(out_w, out_h)
    K_out = make_K(K[0, 0] * scale, K[1, 1] * scale, cx, cy)
    return K_out, (out_w, out_h), dist


# --------------------------------------------------------------------------- #
# Pose builders
# --------------------------------------------------------------------------- #
def R_from_azelroll(azimuth_deg: float, elevation_deg: float,
                    roll_deg: float = 0.0) -> np.ndarray:
    """Rotation ``R_world_from_camera`` from intuitive mounting angles.

    azimuth   : heading of the optical axis, measured in the world XY plane from
                +X towards +Y (0 deg looks along +X, 90 deg looks along +Y).
    elevation : tilt of the optical axis out of the XY plane, positive upwards
                (-90 deg looks straight down).
    roll      : rotation of the camera about its own optical axis, turning
                image-right towards image-down. Positive roll therefore rotates
                the camera body clockwise seen from behind, and the scene appears
                to rotate counter-clockwise on screen.

    Beware the singularity at ``elevation = +-90``: the optical axis is then
    vertical, azimuth no longer changes where the camera points, and it becomes
    the same degree of freedom as roll with the opposite sign, i.e.
    ``R(az, -90, roll) == R(0, -90, roll - az)``. This is ordinary gimbal lock in
    any azimuth/elevation/roll parameterisation. For a camera intended to look
    straight down, set ``azimuth`` to 0 and orient the image with ``roll`` alone,
    or use ``look_at`` and avoid the angles entirely.
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    z_cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    # Image-right stays horizontal before roll, and stays well defined even when
    # the camera looks straight up or down.
    x_cam = np.array([np.sin(az), -np.cos(az), 0.0])
    y_cam = np.cross(z_cam, x_cam)

    if roll_deg:
        c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
        x_cam, y_cam = c * x_cam + s * y_cam, -s * x_cam + c * y_cam

    return np.column_stack([x_cam, y_cam, z_cam])


def azelroll_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of :func:`R_from_azelroll`, in degrees.

    Exact wherever the parameterisation is well conditioned. Within the gimbal
    lock at ``elevation = +-90`` the split between azimuth and roll is arbitrary,
    and this returns the canonical choice of ``azimuth = 0`` with the whole
    rotation expressed as roll. So a camera built with ``azimuth: 90`` looking
    straight down reads back as ``azimuth 0, roll -90``; the matrix is identical.
    """
    z_cam = R[:, 2]
    el = np.arcsin(np.clip(z_cam[2], -1.0, 1.0))
    az = np.arctan2(z_cam[1], z_cam[0])
    x0 = np.array([np.sin(az), -np.cos(az), 0.0])
    y0 = np.cross(z_cam, x0)
    roll = np.arctan2(float(R[:, 0] @ y0), float(R[:, 0] @ x0))
    return float(np.degrees(az)), float(np.degrees(el)), float(np.degrees(roll))


def R_from_lookat(position, target, roll_deg: float = 0.0,
                  up=WORLD_UP) -> np.ndarray:
    """Rotation that points the optical axis from ``position`` at ``target``.

    ``roll_deg`` follows the same convention as :func:`R_from_azelroll`. When the
    optical axis is parallel to ``up`` the image-right direction is undefined, so
    it falls back to world +X and the framing is then controlled entirely by
    ``roll_deg``.
    """
    z_cam = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    norm = np.linalg.norm(z_cam)
    if norm < 1e-12:
        raise ValueError("look-at target coincides with the camera centre")
    z_cam /= norm

    x_cam = np.cross(z_cam, np.asarray(up, dtype=float))
    if np.linalg.norm(x_cam) < 1e-6:  # optical axis parallel to world up
        x_cam = np.array([1.0, 0.0, 0.0])
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)

    if roll_deg:
        c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
        x_cam, y_cam = c * x_cam + s * y_cam, -s * x_cam + c * y_cam

    return np.column_stack([x_cam, y_cam, z_cam])


def pose_from_stereo_extrinsics(R_world_from_a, t_world_from_a, R_b_from_a, T_b_from_a
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Place camera B in the world given its extrinsics relative to camera A.

    ``cv2.stereoCalibrate`` returns ``R, T`` that map a point from A's frame into
    B's frame, ``X_b = R_b_from_a @ X_a + T_b_from_a``. That is the opposite
    direction to the ``T_world_from_camera`` this package uses, which is the usual
    reason a measured rig comes out mirrored or on the wrong side. This does the
    conversion once, correctly.

    Returns ``(R, t)`` for camera B, ready to paste into a ``pose.R`` /
    ``pose.position`` block.
    """
    R_wa = np.asarray(R_world_from_a, dtype=float)
    t_wa = np.asarray(t_world_from_a, dtype=float).reshape(3)
    R_ba = np.asarray(R_b_from_a, dtype=float)
    T_ba = np.asarray(T_b_from_a, dtype=float).reshape(3)

    R_wb = R_wa @ R_ba.T
    t_wb = R_wb @ (R_ba @ R_wa.T @ t_wa - T_ba)
    return R_wb, t_wb


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    """A calibrated pinhole camera with a pose and a usable depth range."""

    name: str
    width: int
    height: int
    K: np.ndarray
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    near: float = 0.1
    far: float = 2.0
    modality: str = "rgb"
    color: str = "#3b82f6"
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(0))
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=float).reshape(3, 3)
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=float).reshape(3)
        self.width = int(self.width)
        self.height = int(self.height)

    # -- basic quantities -------------------------------------------------- #
    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def K_inv(self) -> np.ndarray:
        return np.linalg.inv(self.K)

    @property
    def T(self) -> np.ndarray:
        """4x4 ``T_world_from_camera``."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @property
    def optical_axis(self) -> np.ndarray:
        return self.R[:, 2]

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.width / (2.0 * self.fx))))

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.height / (2.0 * self.fy))))

    @property
    def dfov_deg(self) -> float:
        half = np.hypot(self.width / (2.0 * self.fx), self.height / (2.0 * self.fy))
        return float(np.degrees(2.0 * np.arctan(half)))

    @property
    def near_vertical(self) -> bool:
        """True when the optical axis is within 5 deg of straight up or down.

        There, azimuth and roll collapse into one degree of freedom, so the
        reported angles are not a unique description of the pose.
        """
        return abs(self.azimuth_elevation_roll[1]) > 85.0

    @property
    def pixel_pitch_um(self) -> Optional[float]:
        """Detector pixel pitch, if the rig YAML declared a ``sensor`` block."""
        pitch = (self.meta.get("sensor") or {}).get("pixel_pitch_um")
        return None if pitch is None else float(pitch)

    @property
    def focal_length_mm(self) -> Optional[float]:
        """Physical focal length of the lens in mm.

        Taken from a declared ``paraxial_focal_length_mm`` when present, since for
        a non-rectilinear lens ``fx`` is fitted to the published FOV rather than to
        the true on-axis scale. Otherwise derived from ``fx`` and the pixel pitch,
        which for a zoom lens recovers the ring setting the intrinsics belong to -
        the only way to reproduce a calibration later.
        """
        declared = (self.meta.get("sensor") or {}).get("paraxial_focal_length_mm")
        if declared is not None:
            return float(declared)
        pitch = self.pixel_pitch_um
        return None if pitch is None else self.fx * pitch / 1000.0

    @property
    def paraxial_fx(self) -> Optional[float]:
        """On-axis focal length in pixels, from a declared physical focal length.

        For a non-rectilinear lens this differs from ``fx``. Fitting ``fx`` to the
        published FOV gets the frustum boundary right, which is what coverage
        needs, but understates the scale near the optical axis; the physical focal
        length gets the centre right and understates the cone. The two bracket
        reality until a real distortion model is measured.
        """
        f_mm = (self.meta.get("sensor") or {}).get("paraxial_focal_length_mm")
        pitch = self.pixel_pitch_um
        if f_mm is None or pitch is None:
            return None
        return float(f_mm) / (float(pitch) / 1000.0)

    def paraxial_gsd(self, z: float) -> Optional[float]:
        """Ground sampling distance on the optical axis, in metres per pixel."""
        fx = self.paraxial_fx
        return None if fx is None else z / fx

    @property
    def measurement_pixels(self) -> int:
        """Pixels across the smallest *measurable* feature.

        Thermal detectors need several pixels on a target before the reading is
        radiometrically valid; Optris specifies 3x3 for the Xi series. Optical
        cameras default to 1.
        """
        return int((self.meta.get("sensor") or {}).get("measurement_pixels", 1))

    def measurement_spot(self, z: float) -> float:
        """Smallest reliably measurable feature size in metres at depth ``z``."""
        return self.gsd(z)[0] * self.measurement_pixels

    @property
    def azimuth_elevation_roll(self) -> Tuple[float, float, float]:
        return azelroll_from_R(self.R)

    # -- pose mutators ----------------------------------------------------- #
    def set_pose(self, position=None, azimuth_deg=None, elevation_deg=None,
                 roll_deg=None) -> "Camera":
        """Update position and/or mounting angles in place."""
        if position is not None:
            self.t = np.asarray(position, dtype=float).reshape(3)
        if azimuth_deg is not None or elevation_deg is not None or roll_deg is not None:
            az, el, roll = self.azimuth_elevation_roll
            self.R = R_from_azelroll(
                az if azimuth_deg is None else azimuth_deg,
                el if elevation_deg is None else elevation_deg,
                roll if roll_deg is None else roll_deg,
            )
        return self

    def look_at(self, target, roll_deg: float = 0.0) -> "Camera":
        self.R = R_from_lookat(self.t, target, roll_deg)
        return self

    def copy(self) -> "Camera":
        return Camera(
            name=self.name, width=self.width, height=self.height, K=self.K.copy(),
            R=self.R.copy(), t=self.t.copy(), near=self.near, far=self.far,
            modality=self.modality, color=self.color,
            distortion=self.distortion.copy(), meta=dict(self.meta),
        )

    # -- geometry ---------------------------------------------------------- #
    def backproject(self, pixels, z, world: bool = True) -> np.ndarray:
        """Back-project pixels to 3D at depth(s) ``z``.

        ``pixels`` is (N, 2) in pixel units; ``z`` is a scalar or (N,) of depths
        measured along the optical axis. Returns (N, 3).
        """
        uv = np.atleast_2d(np.asarray(pixels, dtype=float))
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        rays = np.column_stack([uv, np.ones(len(uv))]) @ self.K_inv.T
        p_cam = rays * z
        return p_cam @ self.R.T + self.t if world else p_cam

    def project(self, points_world) -> Tuple[np.ndarray, np.ndarray]:
        """Project world points. Returns ``(uv, z)`` with ``z`` the depth."""
        pts = np.atleast_2d(np.asarray(points_world, dtype=float))
        p_cam = (pts - self.t) @ self.R  # R^T @ (p - t)
        z = p_cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = p_cam[:, :2] / z[:, None] * np.array([self.fx, self.fy]) \
                + np.array([self.cx, self.cy])
        return uv, z

    def to_camera_frame(self, points_world) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points_world, dtype=float))
        return (pts - self.t) @ self.R

    def image_corner_pixels(self) -> np.ndarray:
        """The four sensor corners, ordered bottom-left, bottom-right, top-right,
        top-left as seen in the image (v grows downwards)."""
        w, h = self.width, self.height
        return np.array([[-0.5, h - 0.5], [w - 0.5, h - 0.5],
                         [w - 0.5, -0.5], [-0.5, -0.5]], dtype=float)

    def frustum_corners(self, z: float, world: bool = True) -> np.ndarray:
        """The four back-projected image corners at depth ``z``. Shape (4, 3)."""
        return self.backproject(self.image_corner_pixels(), z, world=world)

    def frustum_vertices(self, near: Optional[float] = None,
                         far: Optional[float] = None) -> np.ndarray:
        """Frustum as 8 world points: 4 near corners then 4 far corners."""
        near = self.near if near is None else near
        far = self.far if far is None else far
        return np.vstack([self.frustum_corners(near), self.frustum_corners(far)])

    def footprint(self, z: float) -> Tuple[float, float]:
        """Width and height in metres of the imaged area on a fronto-parallel
        plane at depth ``z``."""
        return (z * self.width / self.fx, z * self.height / self.fy)

    def gsd(self, z: float) -> Tuple[float, float]:
        """Ground sampling distance in metres per pixel at depth ``z``."""
        return (z / self.fx, z / self.fy)

    def sees(self, points_world, margin_px: float = 0.0,
             normals=None, max_incidence_deg: Optional[float] = None
             ) -> np.ndarray:
        """Boolean visibility mask for world points.

        A point is visible when it lies inside the depth range and inside the
        sensor rectangle (shrunk by ``margin_px``). If ``normals`` and
        ``max_incidence_deg`` are given, points whose surface normal is too
        oblique to the viewing ray are also rejected. Occlusion is not modelled.
        """
        uv, z = self.project(points_world)
        ok = (z >= self.near) & (z <= self.far)
        ok &= (uv[:, 0] >= -0.5 + margin_px) & (uv[:, 0] <= self.width - 0.5 - margin_px)
        ok &= (uv[:, 1] >= -0.5 + margin_px) & (uv[:, 1] <= self.height - 0.5 - margin_px)

        if normals is not None and max_incidence_deg is not None:
            pts = np.atleast_2d(np.asarray(points_world, dtype=float))
            n = np.asarray(normals, dtype=float)
            n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            view = self.t - pts
            view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
            cos_inc = np.abs(np.einsum("ij,ij->i", n, view))
            ok &= cos_inc >= np.cos(np.radians(max_incidence_deg))

        return ok

    def incidence_deg(self, points_world, normals) -> np.ndarray:
        """Angle between the surface normal and the viewing direction, degrees."""
        pts = np.atleast_2d(np.asarray(points_world, dtype=float))
        n = np.asarray(normals, dtype=float)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        view = self.t - pts
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
        return np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", n, view)), 0, 1)))

    def summary(self) -> str:
        az, el, roll = self.azimuth_elevation_roll
        gx, _ = self.gsd(self.far)
        f_mm = self.focal_length_mm
        lens = f"f={f_mm:.2f}mm " if f_mm is not None else ""
        return (f"{self.name:<10s} {self.modality:<7s} {self.width}x{self.height} "
                f"({self.megapixels:.1f} MP)  FOV {self.hfov_deg:.1f}x{self.vfov_deg:.1f} deg  "
                f"{lens}f=({self.fx:.0f},{self.fy:.0f})px  "
                f"pos=({self.t[0]:+.3f},{self.t[1]:+.3f},"
                f"{self.t[2]:+.3f})m  az/el/roll=({az:+.1f},{el:+.1f},{roll:+.1f})deg  "
                f"range=[{self.near:.2f},{self.far:.2f}]m  GSD@far={gx * 1000:.3f}mm")
