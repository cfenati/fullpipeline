"""What a measured camera pair can actually do, as a function of distance.

Four questions, all answered from a :class:`~calibration.stereo.StereoExtrinsics`
and therefore from measurements rather than from the intended geometry:

* **coverage overlap** - how much of camera A's field of view camera B also sees,
  on a plane at distance ``z``. Two cameras with a fixed relative pose overlap by
  an amount that depends entirely on ``z``: far away they see almost the same
  scene, close up they barely meet.
* **triangulation angle** - the angle the two rays to a point subtend. Small
  angles make depth ill-conditioned, large angles make matching hard because the
  two views no longer look alike.
* **depth uncertainty** - what a matching error of a fraction of a pixel costs in
  millimetres of range.
* **registration error** - how far a depth-agnostic mapping between the two
  cameras (a homography fitted at one plane) misplaces a point that is not on
  that plane, and separately, how much of the misplacement is due to uncertainty
  in the extrinsics themselves.

Geometry is evaluated in camera A's frame, using the undistorted field of view of
each camera, so distortion is accounted for without ever pushing points through
the forward distortion polynomial - that polynomial is not monotonic outside the
calibrated field, and points far beyond the frame edge would otherwise fold back
inside it and be counted as visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from calibration.stereo import StereoExtrinsics

DEFAULT_GRID = 96
DEFAULT_DEPTH_STEPS = 28
DEFAULT_DISPARITY_NOISE_PX = 0.3


# --------------------------------------------------------------------------- #
# Field of view as a polygon in undistorted normalised coordinates
# --------------------------------------------------------------------------- #
def undistorted_rays(
    pixels: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Pixels to normalised image coordinates ``(x, y)`` at unit depth."""
    return cv2.undistortPoints(
        np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2),
        camera_matrix,
        distortion,
    ).reshape(-1, 2)


def fov_polygon(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    width: int,
    height: int,
    samples_per_edge: int = 32,
) -> np.ndarray:
    """The sensor boundary as a closed polygon in normalised coordinates.

    The sensor spans ``-0.5`` to ``width - 0.5``, matching the convention used by
    ``design.camera.Camera.sees``. Sampling the border rather than only its four
    corners keeps the curvature that distortion puts into the edges.
    """
    t = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
    x_hi, y_hi = width - 0.5, height - 0.5
    x_lo, y_lo = -0.5, -0.5
    border = np.vstack([
        np.column_stack([x_lo + t * (x_hi - x_lo), np.full_like(t, y_lo)]),
        np.column_stack([np.full_like(t, x_hi), y_lo + t * (y_hi - y_lo)]),
        np.column_stack([x_hi - t * (x_hi - x_lo), np.full_like(t, y_hi)]),
        np.column_stack([np.full_like(t, x_lo), y_hi - t * (y_hi - y_lo)]),
    ])
    return undistorted_rays(border, camera_matrix, distortion)


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Crossing-number test, vectorised over the points."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    x1, y1 = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)

    with np.errstate(divide="ignore", invalid="ignore"):
        for xa, ya, xb, yb in zip(x1, y1, x2, y2):
            straddles = (ya > y) != (yb > y)
            crossing = xa + (y - ya) * (xb - xa) / (yb - ya)
            inside ^= straddles & (x < crossing)
    return inside


def _polygons(extrinsics: StereoExtrinsics) -> Tuple[np.ndarray, np.ndarray]:
    return (
        fov_polygon(extrinsics.camera_matrix_a, extrinsics.distortion_a,
                    *extrinsics.image_size_a),
        fov_polygon(extrinsics.camera_matrix_b, extrinsics.distortion_b,
                    *extrinsics.image_size_b),
    )


def _seen_by_b(
    points_a: np.ndarray,
    extrinsics: StereoExtrinsics,
    polygon_b: np.ndarray,
    R: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Which points of A's frame fall inside B, and where they land in B."""
    R = extrinsics.R if R is None else R
    T = extrinsics.T.reshape(3) if T is None else np.asarray(T).reshape(3)
    points_b = points_a @ R.T + T
    depth = points_b[:, 2]
    ahead = depth > 1e-9
    normalised = np.full((len(points_b), 2), np.nan)
    normalised[ahead] = points_b[ahead, :2] / depth[ahead, None]
    visible = np.zeros(len(points_b), dtype=bool)
    visible[ahead] = points_in_polygon(normalised[ahead], polygon_b)
    return visible, points_b


def default_depths(
    depth_min: float,
    depth_max: float,
    steps: int = DEFAULT_DEPTH_STEPS,
) -> np.ndarray:
    """Log-spaced distances: every metric here varies with ``1/z``, not ``z``."""
    return np.geomspace(max(depth_min, 1e-3), max(depth_max, depth_min * 1.01), steps)


def _polygon_area(polygon: np.ndarray) -> float:
    """Shoelace area of a closed polygon."""
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _b_field_on_plane(
    extrinsics: StereoExtrinsics,
    polygon_b: np.ndarray,
    z: float,
) -> Tuple[np.ndarray, bool]:
    """Where camera B's field of view meets a plane at depth ``z`` in A's frame.

    Returns the boundary polygon and whether it closes. A camera looking nearly
    along the plane has border rays that never reach it, and its region on that
    plane is then unbounded - in which case an area, and any IoU built from it,
    would be meaningless.
    """
    directions = np.column_stack([polygon_b, np.ones(len(polygon_b))]) @ extrinsics.R
    center = extrinsics.center_b_in_a
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (z - center[2]) / directions[:, 2]
    reaches = (directions[:, 2] > 1e-9) & (distance > 0)
    points = center + directions * distance[:, None]
    return points[reaches][:, :2], bool(reaches.all())


# --------------------------------------------------------------------------- #
# Coverage overlap, triangulation angle and depth uncertainty vs distance
# --------------------------------------------------------------------------- #
@dataclass
class DepthSweep:
    """Pair geometry sampled on planes fronto-parallel to camera A."""

    depths_m: np.ndarray
    fraction_of_a: np.ndarray  # share of A's field also seen by B
    fraction_of_b: np.ndarray  # share of B's field also seen by A
    iou: np.ndarray
    shared_area_cm2: np.ndarray
    area_a_cm2: np.ndarray
    triangulation_deg: Dict[str, np.ndarray]  # min / mean / max / axis, over the overlap
    depth_sigma_mm: Dict[str, np.ndarray]  # min / mean / max / axis
    disparity_px: np.ndarray  # equivalent disparity of a rectified pair, f * B / z
    gsd_a_mm: np.ndarray
    disparity_noise_px: float
    unbounded_depths: List[float] = field(default_factory=list)

    def at(self, depth: float) -> Dict[str, float]:
        """Every metric at the sampled depth closest to ``depth``."""
        i = int(np.argmin(np.abs(self.depths_m - depth)))
        return {
            "depth_m": float(self.depths_m[i]),
            "fraction_of_a": float(self.fraction_of_a[i]),
            "fraction_of_b": float(self.fraction_of_b[i]),
            "iou": float(self.iou[i]),
            "shared_area_cm2": float(self.shared_area_cm2[i]),
            "area_a_cm2": float(self.area_a_cm2[i]),
            "triangulation_mean_deg": float(self.triangulation_deg["mean"][i]),
            "triangulation_axis_deg": float(self.triangulation_deg["axis"][i]),
            "depth_sigma_mean_mm": float(self.depth_sigma_mm["mean"][i]),
            "depth_sigma_axis_mm": float(self.depth_sigma_mm["axis"][i]),
            "disparity_px": float(self.disparity_px[i]),
            "gsd_a_mm": float(self.gsd_a_mm[i]),
        }

    def best_overlap_depth(self) -> float:
        return float(self.depths_m[int(np.argmax(self.fraction_of_a))])


def depth_sweep(
    extrinsics: StereoExtrinsics,
    depths: np.ndarray,
    grid: int = DEFAULT_GRID,
    disparity_noise_px: float = DEFAULT_DISPARITY_NOISE_PX,
) -> DepthSweep:
    """Overlap, triangulation angle and depth uncertainty at each distance.

    Each metric is evaluated on a plane fronto-parallel to A. Both fields of view
    cut that plane in a polygon whose area follows from its boundary, so the two
    areas and the union are exact; only their intersection is measured by
    sampling, on a grid that covers A's own footprint.

    ``depth_sigma`` is the range uncertainty caused by ``disparity_noise_px`` of
    matching error, computed per point as

        sigma_z = noise * d_b / (f_b * sin(alpha))

    with ``d_b`` the distance from B to the point and ``alpha`` the triangulation
    angle. For a rectified pair this reduces to the familiar
    ``z^2 * noise / (f * baseline)``, but unlike that form it stays valid when the
    cameras converge.
    """
    depths = np.asarray(depths, dtype=float).reshape(-1)
    polygon_a, polygon_b = _polygons(extrinsics)
    width_a, height_a = extrinsics.image_size_a
    fx_a = extrinsics.camera_matrix_a[0, 0]
    focal_b = 0.5 * float(extrinsics.camera_matrix_b[0, 0] + extrinsics.camera_matrix_b[1, 1])
    center_b = extrinsics.center_b_in_a

    axis_ray = undistorted_rays(
        np.array([[(width_a - 1.0) / 2.0, (height_a - 1.0) / 2.0]]),
        extrinsics.camera_matrix_a,
        extrinsics.distortion_a,
    )[0]

    fraction_a, fraction_b, iou = [], [], []
    shared_area, area_a_list = [], []
    triangulation: Dict[str, List[float]] = {k: [] for k in ("min", "mean", "max", "axis")}
    sigma: Dict[str, List[float]] = {k: [] for k in ("min", "mean", "max", "axis")}
    unbounded: List[float] = []

    for z in depths:
        plane_a = polygon_a * z
        area_a = _polygon_area(plane_a)
        plane_b, bounded = _b_field_on_plane(extrinsics, polygon_b, z)
        area_b = _polygon_area(plane_b) if bounded else float("nan")
        if not bounded:
            unbounded.append(float(z))

        # Sample A's footprint only: the intersection can never leave it, and a
        # tight window keeps the resolution high where it is being measured.
        lo, hi = plane_a.min(axis=0), plane_a.max(axis=0)
        xs = np.linspace(lo[0], hi[0], grid)
        ys = np.linspace(lo[1], hi[1], grid)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        points = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.full(grid * grid, z)])

        in_a = points_in_polygon(points[:, :2] / z, polygon_a)
        in_b, _ = _seen_by_b(points, extrinsics, polygon_b)
        both = in_a & in_b

        share_of_a = float(both.sum()) / float(in_a.sum()) if in_a.any() else 0.0
        intersection = share_of_a * area_a
        union = area_a + area_b - intersection

        fraction_a.append(share_of_a)
        fraction_b.append(intersection / area_b if area_b > 0 else float("nan"))
        iou.append(intersection / union if union > 0 else float("nan"))
        shared_area.append(intersection * 1e4)
        area_a_list.append(area_a * 1e4)

        shared_points = points[both]
        if len(shared_points) == 0:
            for key in triangulation:
                triangulation[key].append(float("nan"))
                sigma[key].append(float("nan"))
            continue

        angles = _triangulation_angles(shared_points, center_b)
        distances_b = np.linalg.norm(shared_points - center_b, axis=1)
        sigmas = _depth_sigma(angles, distances_b, focal_b, disparity_noise_px)

        axis_point = np.array([[axis_ray[0] * z, axis_ray[1] * z, z]])
        axis_angle = float(_triangulation_angles(axis_point, center_b)[0])
        axis_sigma = float(
            _depth_sigma(
                np.array([axis_angle]),
                np.linalg.norm(axis_point - center_b, axis=1),
                focal_b,
                disparity_noise_px,
            )[0]
        )

        triangulation["min"].append(float(angles.min()))
        triangulation["mean"].append(float(angles.mean()))
        triangulation["max"].append(float(angles.max()))
        triangulation["axis"].append(axis_angle)
        sigma["min"].append(float(sigmas.min()) * 1000.0)
        sigma["mean"].append(float(sigmas.mean()) * 1000.0)
        sigma["max"].append(float(sigmas.max()) * 1000.0)
        sigma["axis"].append(axis_sigma * 1000.0)

    baseline = extrinsics.baseline_m
    return DepthSweep(
        depths_m=depths,
        fraction_of_a=np.array(fraction_a),
        fraction_of_b=np.array(fraction_b),
        iou=np.array(iou),
        shared_area_cm2=np.array(shared_area),
        area_a_cm2=np.array(area_a_list),
        triangulation_deg={k: np.array(v) for k, v in triangulation.items()},
        depth_sigma_mm={k: np.array(v) for k, v in sigma.items()},
        disparity_px=float(fx_a) * baseline / depths,
        gsd_a_mm=depths / float(fx_a) * 1000.0,
        disparity_noise_px=float(disparity_noise_px),
        unbounded_depths=unbounded,
    )


def _triangulation_angles(points_a: np.ndarray, center_b: np.ndarray) -> np.ndarray:
    """Angle in degrees between the ray from A and the ray from B, per point."""
    ray_a = points_a / np.maximum(np.linalg.norm(points_a, axis=1, keepdims=True), 1e-12)
    to_b = points_a - center_b
    ray_b = to_b / np.maximum(np.linalg.norm(to_b, axis=1, keepdims=True), 1e-12)
    cos = np.clip(np.einsum("ij,ij->i", ray_a, ray_b), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _depth_sigma(
    angles_deg: np.ndarray,
    distances_b: np.ndarray,
    focal_b: float,
    noise_px: float,
) -> np.ndarray:
    """Range uncertainty in metres from a matching error of ``noise_px``."""
    sin_alpha = np.sin(np.radians(angles_deg))
    with np.errstate(divide="ignore", invalid="ignore"):
        return noise_px * distances_b / (focal_b * np.maximum(sin_alpha, 1e-9))


# --------------------------------------------------------------------------- #
# Depth-dependent registration error
# --------------------------------------------------------------------------- #
@dataclass
class RegistrationCurve:
    """Error of a fixed pixel mapping from A to B as the true depth changes."""

    depths_m: np.ndarray
    reference_depth_m: float
    rms_px: np.ndarray
    max_px: np.ndarray
    rms_mm: np.ndarray
    max_mm: np.ndarray
    valid_fraction: np.ndarray
    analytic_px: np.ndarray
    # f_b * baseline, in pixels per dioptre: the slope of the error against
    # inverse depth, and the only quantity the tolerance band depends on.
    parallax_scale_px_m: float = float("nan")
    label: str = "parallax"

    def at(self, depth: float) -> Dict[str, float]:
        i = int(np.argmin(np.abs(self.depths_m - depth)))
        return {
            "depth_m": float(self.depths_m[i]),
            "rms_px": float(self.rms_px[i]),
            "max_px": float(self.max_px[i]),
            "rms_mm": float(self.rms_mm[i]),
            "max_mm": float(self.max_mm[i]),
        }

    def depth_for_error(self, tolerance_px: float) -> Tuple[float, float]:
        """Depth interval in which the mapping stays under ``tolerance_px``.

        This is the usable depth of field of a depth-agnostic overlay: exact on
        the reference plane, degrading in both directions. Solved in closed form
        from ``error = f * B * |1/z - 1/z0|``, because the sampled depths are
        almost never fine enough to resolve a band that can be well under a
        millimetre wide.
        """
        z0, scale = self.reference_depth_m, self.parallax_scale_px_m
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(z0):
            usable = self.rms_px <= tolerance_px
            if not usable.any():
                return (float("nan"), float("nan"))
            return (float(self.depths_m[usable].min()), float(self.depths_m[usable].max()))

        margin = tolerance_px / scale  # in inverse metres
        near_inverse = 1.0 / z0 + margin
        far_inverse = 1.0 / z0 - margin
        near = 1.0 / near_inverse
        far = float("inf") if far_inverse <= 0 else 1.0 / far_inverse
        return (float(near), float(far))


def _frame_grid(
    extrinsics: StereoExtrinsics,
    grid: int,
    margin_fraction: float = 0.02,
) -> np.ndarray:
    """Pixels spanning camera A's frame, as normalised undistorted rays."""
    width, height = extrinsics.image_size_a
    lo, hi = margin_fraction, 1.0 - margin_fraction
    u = np.linspace(lo, hi, grid) * (width - 1.0)
    v = np.linspace(lo, hi, grid) * (height - 1.0)
    pixels = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1).reshape(-1, 2)
    rays = undistorted_rays(pixels, extrinsics.camera_matrix_a, extrinsics.distortion_a)
    return np.column_stack([rays, np.ones(len(rays))])


def _project_into_b(
    points_a: np.ndarray,
    extrinsics: StereoExtrinsics,
    R: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pixels in B for points given in A's frame, distortion included."""
    R = extrinsics.R if R is None else R
    T = extrinsics.T.reshape(3) if T is None else np.asarray(T).reshape(3)
    points_b = points_a @ R.T + T
    pixels, _ = cv2.projectPoints(
        points_b.reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        extrinsics.camera_matrix_b,
        extrinsics.distortion_b,
    )
    return pixels.reshape(-1, 2), points_b[:, 2]


def registration_error(
    extrinsics: StereoExtrinsics,
    reference_depth: float,
    depths: np.ndarray,
    grid: int = 33,
) -> RegistrationCurve:
    """Parallax error of a mapping from A to B that ignores depth.

    Overlaying one camera on the other without a depth map means assuming every
    pixel sits on one plane. A pixel of A is then drawn where the point at
    ``reference_depth`` would appear in B:

        uv_est = project_b(backproject_a(uv, reference_depth))

    while the point actually at depth ``z`` appears at
    ``uv_true = project_b(backproject_a(uv, z))``. The distance between the two is
    the registration error: zero on the reference plane and growing with
    ``|1/z - 1/z0|``, which is why the metric belongs to a depth rather than to
    the calibration.

    ``analytic_px`` is the closed form ``f_b * baseline * |1/z - 1/z0|``, valid
    for a near-rectified pair; comparing it with ``rms_px`` shows how much the
    rotation between the cameras adds.
    """
    depths = np.asarray(depths, dtype=float).reshape(-1)
    if reference_depth not in depths:
        depths = np.unique(np.append(depths, reference_depth))

    rays = _frame_grid(extrinsics, grid)
    _, polygon_b = _polygons(extrinsics)

    reference_points = rays * reference_depth
    pixels_reference, _ = _project_into_b(reference_points, extrinsics)
    visible_reference, _ = _seen_by_b(reference_points, extrinsics, polygon_b)

    rms_px, max_px, rms_mm, max_mm, valid = [], [], [], [], []
    focal_b = float(extrinsics.camera_matrix_b[0, 0])

    for z in depths:
        points = rays * z
        pixels, depth_b = _project_into_b(points, extrinsics)
        visible, _ = _seen_by_b(points, extrinsics, polygon_b)
        usable = visible & visible_reference & (depth_b > 1e-9)
        valid.append(float(usable.mean()))
        if not usable.any():
            rms_px.append(float("nan")), max_px.append(float("nan"))
            rms_mm.append(float("nan")), max_mm.append(float("nan"))
            continue

        error = np.linalg.norm(pixels[usable] - pixels_reference[usable], axis=1)
        # The same angular error means a bigger physical offset further away.
        error_mm = error * depth_b[usable] / focal_b * 1000.0
        rms_px.append(float(np.sqrt(np.mean(error ** 2))))
        max_px.append(float(error.max()))
        rms_mm.append(float(np.sqrt(np.mean(error_mm ** 2))))
        max_mm.append(float(error_mm.max()))

    scale = focal_b * extrinsics.baseline_m
    return RegistrationCurve(
        depths_m=depths,
        reference_depth_m=float(reference_depth),
        rms_px=np.array(rms_px),
        max_px=np.array(max_px),
        rms_mm=np.array(rms_mm),
        max_mm=np.array(max_mm),
        valid_fraction=np.array(valid),
        analytic_px=scale * np.abs(1.0 / depths - 1.0 / reference_depth),
        parallax_scale_px_m=float(scale),
    )


def registration_uncertainty(
    extrinsics: StereoExtrinsics,
    ensemble: Sequence[Tuple[np.ndarray, np.ndarray]],
    depths: np.ndarray,
    grid: int = 25,
) -> Optional[RegistrationCurve]:
    """Registration error caused by uncertainty in the extrinsics themselves.

    Even with a perfect depth map, mapping A onto B can only be as good as the
    measured pose. Each member of ``ensemble`` is a pose the data would support
    (see :func:`calibration.stereo.pose_ensemble`); re-projecting the same points
    with each of them and measuring the spread gives the noise floor of the
    co-registration - the error that no amount of depth information removes.
    """
    if not ensemble:
        return None

    depths = np.asarray(depths, dtype=float).reshape(-1)
    rays = _frame_grid(extrinsics, grid)
    _, polygon_b = _polygons(extrinsics)
    focal_b = float(extrinsics.camera_matrix_b[0, 0])

    rms_px, max_px, rms_mm, max_mm, valid = [], [], [], [], []
    for z in depths:
        points = rays * z
        nominal, depth_b = _project_into_b(points, extrinsics)
        visible, _ = _seen_by_b(points, extrinsics, polygon_b)
        valid.append(float(visible.mean()))
        if not visible.any():
            rms_px.append(float("nan")), max_px.append(float("nan"))
            rms_mm.append(float("nan")), max_mm.append(float("nan"))
            continue

        errors = []
        for R, T in ensemble:
            pixels, _ = _project_into_b(points, extrinsics, R=R, T=T)
            errors.append(np.linalg.norm(pixels[visible] - nominal[visible], axis=1))
        errors = np.concatenate(errors)
        scale_mm = np.tile(depth_b[visible] / focal_b * 1000.0, len(ensemble))
        rms_px.append(float(np.sqrt(np.mean(errors ** 2))))
        max_px.append(float(errors.max()))
        rms_mm.append(float(np.sqrt(np.mean((errors * scale_mm) ** 2))))
        max_mm.append(float((errors * scale_mm).max()))

    return RegistrationCurve(
        depths_m=depths,
        reference_depth_m=float("nan"),
        rms_px=np.array(rms_px),
        max_px=np.array(max_px),
        rms_mm=np.array(rms_mm),
        max_mm=np.array(max_mm),
        valid_fraction=np.array(valid),
        analytic_px=np.zeros(len(depths)),
        label="extrinsic uncertainty",
    )


# --------------------------------------------------------------------------- #
# Where the overlap sits inside the frame
# --------------------------------------------------------------------------- #
def overlap_masks(
    extrinsics: StereoExtrinsics,
    depths: Sequence[float],
    step: int = 4,
) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """Per-depth masks of camera A's frame, True where camera B sees it too.

    The averaged fraction matches :func:`depth_sweep`, but the mask says *where*
    the shared region is, which is what decides whether the subject should be
    moved or a camera re-aimed.
    """
    width, height = extrinsics.image_size_a
    u = np.arange(0, width, step, dtype=float)
    v = np.arange(0, height, step, dtype=float)
    pixels = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1).reshape(-1, 2)
    rays = undistorted_rays(pixels, extrinsics.camera_matrix_a, extrinsics.distortion_a)
    rays = np.column_stack([rays, np.ones(len(rays))])
    _, polygon_b = _polygons(extrinsics)

    masks = []
    for z in depths:
        visible, _ = _seen_by_b(rays * float(z), extrinsics, polygon_b)
        masks.append(visible.reshape(len(u), len(v)).T)
    return masks, (len(v), len(u))


def _clip_to_halfspace(
    polygon: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    """Part of a 3D polygon satisfying ``normal . X + offset >= 0``."""
    if len(polygon) < 3:
        return np.zeros((0, 3))
    distance = polygon @ normal + offset
    inside = distance >= 0.0
    if inside.all():
        return polygon

    kept: List[np.ndarray] = []
    for start in range(len(polygon)):
        end = (start + 1) % len(polygon)
        if inside[start]:
            kept.append(polygon[start])
        if inside[start] != inside[end]:
            t = distance[start] / (distance[start] - distance[end])
            kept.append(polygon[start] + t * (polygon[end] - polygon[start]))
    return np.asarray(kept) if len(kept) >= 3 else np.zeros((0, 3))


def _frustum_halfspaces(
    extrinsics: StereoExtrinsics,
    samples_per_edge: int = 32,
) -> List[Tuple[np.ndarray, float]]:
    """Camera B's field of view as half-spaces in camera A's frame.

    Each edge of B's sensor boundary, together with B's centre, spans a plane;
    B sees a point when the point is on the inner side of every one of them, and
    in front of B. As half-spaces rather than a polygon on some plane, this stays
    finite however the two fields diverge.
    """
    polygon = fov_polygon(
        extrinsics.camera_matrix_b, extrinsics.distortion_b,
        *extrinsics.image_size_b, samples_per_edge=samples_per_edge,
    )
    rays = np.column_stack([polygon, np.ones(len(polygon))])
    interior = rays.mean(axis=0)
    rotation, translation = extrinsics.R, extrinsics.T.reshape(3)

    planes: List[Tuple[np.ndarray, float]] = []
    for ray, next_ray in zip(rays, np.roll(rays, -1, axis=0)):
        normal = np.cross(ray, next_ray)
        if normal @ interior < 0:
            normal = -normal
        # X_b = R X_a + T, so a test on X_b becomes this test on X_a.
        planes.append((rotation.T @ normal, float(normal @ translation)))
    planes.append((rotation.T @ np.array([0.0, 0.0, 1.0]), float(translation[2])))
    return planes


def overlap_outline(
    extrinsics: StereoExtrinsics,
    depth: float,
    samples_per_edge: int = 64,
) -> np.ndarray:
    """Outline, in camera A's pixels, of what both cameras see at one depth.

    Camera A's own footprint on the plane at ``depth`` is clipped against camera
    B's frustum, so the result is bounded and stays inside A's frame - which also
    keeps it inside the range where the distortion model is valid. Unlike the
    boundary of :func:`overlap_masks` it is exact rather than sampled, so it is
    what the figures should draw.
    """
    polygon = fov_polygon(
        extrinsics.camera_matrix_a, extrinsics.distortion_a,
        *extrinsics.image_size_a, samples_per_edge=samples_per_edge,
    )
    footprint = np.column_stack([polygon, np.ones(len(polygon))]) * float(depth)
    for normal, offset in _frustum_halfspaces(extrinsics):
        footprint = _clip_to_halfspace(footprint, normal, offset)
        if len(footprint) < 3:
            return np.zeros((0, 2))

    projected, _ = cv2.projectPoints(
        footprint.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
        extrinsics.camera_matrix_a, extrinsics.distortion_a,
    )
    return projected.reshape(-1, 2)
