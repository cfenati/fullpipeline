"""Quantitative evaluation of a camera rig.

All metrics are geometric and ignore self-occlusion: a point is "seen" when it
falls inside a camera's frustum (and, for surfaces, when the view angle is not
too oblique). That is the right model for choosing mounting positions; add a
mesh ray-caster if you later need true occlusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera import Camera
from .rig import Rig


# --------------------------------------------------------------------------- #
# Visibility
# --------------------------------------------------------------------------- #
def visibility_matrix(cameras: Sequence[Camera], points: np.ndarray,
                      normals: Optional[np.ndarray] = None,
                      max_incidence_deg: Optional[float] = None,
                      margin_px: float = 0.0) -> np.ndarray:
    """Boolean (n_cameras, n_points) matrix of frustum membership."""
    if len(points) == 0:
        return np.zeros((len(cameras), 0), dtype=bool)
    return np.vstack([
        cam.sees(points, margin_px=margin_px, normals=normals,
                 max_incidence_deg=max_incidence_deg)
        for cam in cameras
    ])


@dataclass
class CoverageResult:
    """Outcome of a coverage study over a point set."""

    points: np.ndarray
    visible: np.ndarray                      # (n_cam, n_pts) bool
    names: List[str]
    unit_volume: Optional[float] = None      # m^3 per point, volume studies only
    unit_area: Optional[float] = None        # m^2 per point, surface studies only
    normals: Optional[np.ndarray] = None

    @property
    def n_views(self) -> np.ndarray:
        """Number of cameras seeing each point."""
        return self.visible.sum(axis=0)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def fraction_seen_by_at_least(self, k: int) -> float:
        if self.n_points == 0:
            return 0.0
        return float((self.n_views >= k).mean())

    def fraction_seen_by(self, names: Sequence[str]) -> float:
        """Fraction of points simultaneously inside every listed camera."""
        if self.n_points == 0:
            return 0.0
        idx = [self.names.index(n) for n in names]
        return float(np.all(self.visible[idx], axis=0).mean())

    def measure_at_least(self, k: int) -> float:
        """Volume (m^3) or area (m^2) seen by at least ``k`` cameras."""
        unit = self.unit_volume if self.unit_volume is not None else (self.unit_area or 0.0)
        return float((self.n_views >= k).sum() * unit)

    def per_camera_fraction(self) -> Dict[str, float]:
        if self.n_points == 0:
            return {n: 0.0 for n in self.names}
        return {n: float(v.mean()) for n, v in zip(self.names, self.visible)}

    def pairwise_overlap(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Jaccard index and asymmetric overlap fractions for every camera pair."""
        out: Dict[Tuple[str, str], Dict[str, float]] = {}
        for i, j in combinations(range(len(self.names)), 2):
            a, b = self.visible[i], self.visible[j]
            inter = float(np.count_nonzero(a & b))
            union = float(np.count_nonzero(a | b))
            na, nb = float(a.sum()), float(b.sum())
            out[(self.names[i], self.names[j])] = {
                "iou": inter / union if union else 0.0,
                "fraction_of_first": inter / na if na else 0.0,
                "fraction_of_second": inter / nb if nb else 0.0,
                "n_points": inter,
            }
        return out


def volume_coverage(rig: Rig, pitch: Optional[float] = None,
                    margin_px: Optional[float] = None) -> CoverageResult:
    """Voxelise the target volume and test which cameras contain each voxel."""
    pitch = pitch or rig.analysis.voxel_pitch
    margin = rig.analysis.margin_px if margin_px is None else margin_px
    pts = rig.target.sample_volume(pitch)
    vis = visibility_matrix(rig.cameras, pts, margin_px=margin)
    return CoverageResult(points=pts, visible=vis, names=rig.names,
                          unit_volume=pitch ** 3)


def surface_coverage(rig: Rig, pitch: Optional[float] = None,
                     max_incidence_deg: Optional[float] = None) -> CoverageResult:
    """Sample the target surface and test visibility with a view-angle limit."""
    pitch = pitch or rig.analysis.surface_pitch
    max_inc = (rig.analysis.max_incidence_deg if max_incidence_deg is None
               else max_incidence_deg)
    pts, nrm = rig.target.sample_surface(pitch)
    vis = visibility_matrix(rig.cameras, pts, normals=nrm,
                            max_incidence_deg=max_inc,
                            margin_px=rig.analysis.margin_px)
    return CoverageResult(points=pts, visible=vis, names=rig.names,
                          unit_area=pitch ** 2, normals=nrm)


# --------------------------------------------------------------------------- #
# Resolution / sampling density
# --------------------------------------------------------------------------- #
def resolution_stats(rig: Rig, coverage: CoverageResult) -> Dict[str, Dict[str, float]]:
    """Effective ground sampling distance per camera over the points it sees.

    ``gsd_eff = (z / f) / cos(incidence)`` so foreshortening on oblique surfaces
    is accounted for. Values are millimetres per pixel.
    """
    out: Dict[str, Dict[str, float]] = {}
    for cam, seen in zip(rig.cameras, coverage.visible):
        if not seen.any():
            out[cam.name] = {"n_points": 0}
            continue
        pts = coverage.points[seen]
        _, z = cam.project(pts)
        gsd = z / cam.fx
        if coverage.normals is not None:
            inc = cam.incidence_deg(pts, coverage.normals[seen])
            gsd = gsd / np.maximum(np.cos(np.radians(inc)), 1e-3)
            inc_stats = {"incidence_mean_deg": float(inc.mean()),
                         "incidence_max_deg": float(inc.max())}
        else:
            inc_stats = {}
        out[cam.name] = dict({
            "n_points": int(seen.sum()),
            "gsd_min_mm": float(gsd.min() * 1000),
            "gsd_mean_mm": float(gsd.mean() * 1000),
            "gsd_max_mm": float(gsd.max() * 1000),
            "depth_min_m": float(z.min()),
            "depth_max_m": float(z.max()),
            # A thermal detector needs several pixels on a feature before the
            # reading is radiometrically valid (3x3 for the Optris Xi series).
            "measurement_pixels": cam.measurement_pixels,
            "spot_mean_mm": float(gsd.mean() * 1000 * cam.measurement_pixels),
        }, **inc_stats)
    return out


def pixels_on_target(rig: Rig, coverage: Optional[CoverageResult] = None
                     ) -> Dict[str, float]:
    """Fraction of each sensor filled by the target's bounding sphere."""
    center = rig.target.center
    radius = rig.target.characteristic_size / 2.0
    out = {}
    for cam in rig.cameras:
        z = float(cam.to_camera_frame(center)[0, 2])
        if z <= 0:
            out[cam.name] = 0.0
            continue
        r_px_x = radius / z * cam.fx
        r_px_y = radius / z * cam.fy
        out[cam.name] = float(min(1.0, (np.pi * r_px_x * r_px_y) /
                                  (cam.width * cam.height)))
    return out


# --------------------------------------------------------------------------- #
# Stereo geometry
# --------------------------------------------------------------------------- #
def stereo_pairs(rig: Rig, disparity_noise_px: Optional[float] = None
                 ) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Baseline, convergence and triangulation quality for every camera pair.

    ``depth_sigma_mm`` is the classic stereo depth uncertainty
    ``sigma_z = z^2 * sigma_disparity / (f * B)`` evaluated at the target centre;
    it assumes the pair is roughly rectified, so treat it as an order of
    magnitude for strongly converging pairs.
    """
    noise = (rig.analysis.disparity_noise_px if disparity_noise_px is None
             else disparity_noise_px)
    center = rig.target.center
    out: Dict[Tuple[str, str], Dict[str, float]] = {}

    for a, b in combinations(rig.cameras, 2):
        baseline = float(np.linalg.norm(a.t - b.t))
        cos_axes = float(np.clip(a.optical_axis @ b.optical_axis, -1, 1))
        ray_a, ray_b = center - a.t, center - b.t
        na, nb = np.linalg.norm(ray_a), np.linalg.norm(ray_b)
        tri = float(np.degrees(np.arccos(np.clip(
            (ray_a @ ray_b) / max(na * nb, 1e-12), -1, 1))))
        z = float(np.mean([a.to_camera_frame(center)[0, 2],
                           b.to_camera_frame(center)[0, 2]]))
        f_eff = float(np.mean([a.fx, b.fx]))
        sigma = (z ** 2 * noise / (f_eff * baseline)) if baseline > 1e-9 else np.inf
        out[(a.name, b.name)] = {
            "baseline_mm": baseline * 1000.0,
            "axis_angle_deg": float(np.degrees(np.arccos(cos_axes))),
            "triangulation_angle_deg": tri,
            "distance_a_m": na,
            "distance_b_m": nb,
            "depth_sigma_mm": sigma * 1000.0,
        }
    return out


def triangulation_angles(rig: Rig, coverage: CoverageResult) -> Dict[str, np.ndarray]:
    """Best pairwise triangulation angle at each multi-view point (degrees)."""
    n_pts = coverage.n_points
    best = np.zeros(n_pts)
    for i, j in combinations(range(len(rig.cameras)), 2):
        both = coverage.visible[i] & coverage.visible[j]
        if not both.any():
            continue
        pts = coverage.points[both]
        ra = pts - rig.cameras[i].t
        rb = pts - rig.cameras[j].t
        ra /= np.maximum(np.linalg.norm(ra, axis=1, keepdims=True), 1e-12)
        rb /= np.maximum(np.linalg.norm(rb, axis=1, keepdims=True), 1e-12)
        ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", ra, rb), -1, 1)))
        best[both] = np.maximum(best[both], ang)
    return {"angle_deg": best, "multi_view": coverage.n_views >= 2}


# --------------------------------------------------------------------------- #
# Depth-dependent registration error
# --------------------------------------------------------------------------- #
def registration_error(source: Camera, target_cam: Camera,
                       reference_depth: float,
                       depths: Optional[np.ndarray] = None,
                       n_samples: int = 24) -> Dict[str, np.ndarray]:
    """Parallax error of a depth-agnostic pixel mapping between two cameras.

    Overlaying ``source`` (e.g. thermal) onto ``target_cam`` (e.g. RGB) with a
    single homography is exact only for one plane. For a source pixel ``x``, the
    mapping puts it where the point at ``reference_depth`` would land:

        x_est = project_target(backproject_source(x, reference_depth))

    while the true point at depth ``z`` lands at

        x_true = project_target(backproject_source(x, z)).

    The returned ``rms_px`` / ``max_px`` are in ``target_cam`` pixels; the ``_mm``
    variants convert that back to a physical offset on the subject.
    """
    if depths is None:
        lo = max(source.near, reference_depth * 0.35)
        hi = min(source.far, reference_depth * 2.2)
        if hi <= lo:
            lo, hi = source.near, source.far
        # Include the reference depth so the curve bottoms out exactly at zero.
        depths = np.unique(np.append(np.linspace(lo, hi, n_samples),
                                     reference_depth))
    depths = np.asarray(depths, dtype=float)

    # Sample a grid of source pixels, avoiding the extreme border.
    gu = np.linspace(0.05, 0.95, 9) * source.width - 0.5
    gv = np.linspace(0.05, 0.95, 9) * source.height - 0.5
    uv = np.stack(np.meshgrid(gu, gv, indexing="ij"), axis=-1).reshape(-1, 2)

    ref_world = source.backproject(uv, reference_depth)
    uv_est, _ = target_cam.project(ref_world)

    rms, mx, rms_mm, mx_mm, valid = [], [], [], [], []
    nan = float("nan")
    for z in depths:
        world = source.backproject(uv, z)
        uv_true, z_t = target_cam.project(world)
        ok = z_t > 1e-6
        valid.append(float(np.mean(ok)))
        if not ok.any():  # the whole patch fell behind the target camera
            rms.append(nan), mx.append(nan), rms_mm.append(nan), mx_mm.append(nan)
            continue
        err = np.linalg.norm(uv_true[ok] - uv_est[ok], axis=1)
        # Convert the pixel error to a metric offset at the observed depth.
        err_mm = err * z_t[ok] / target_cam.fx * 1000.0
        rms.append(float(np.sqrt(np.mean(err ** 2))))
        mx.append(float(err.max()))
        rms_mm.append(float(np.sqrt(np.mean(err_mm ** 2))))
        mx_mm.append(float(err_mm.max()))

    return {"depths_m": depths, "rms_px": np.array(rms), "max_px": np.array(mx),
            "rms_mm": np.array(rms_mm), "max_mm": np.array(mx_mm),
            "valid_fraction": np.array(valid),
            "reference_depth_m": float(reference_depth),
            "source": source.name, "target": target_cam.name}


def registration_table(rig: Rig, reference_depth: Optional[float] = None
                       ) -> List[Dict[str, np.ndarray]]:
    """Registration error curves for every camera pair.

    When no reference depth is configured, each pair is calibrated at the mean
    depth of the target centre as seen by the two cameras.
    """
    ref = reference_depth or rig.analysis.registration_depth
    out = []
    for a, b in combinations(rig.cameras, 2):
        z_ref = ref
        if z_ref is None:
            z_ref = float(np.mean([a.to_camera_frame(rig.target.center)[0, 2],
                                   b.to_camera_frame(rig.target.center)[0, 2]]))
        out.append(registration_error(a, b, z_ref))
    return out


# --------------------------------------------------------------------------- #
# Depth sweep of the overlap
# --------------------------------------------------------------------------- #
def overlap_vs_depth(rig: Rig, axis_camera: Optional[Camera] = None,
                     depths: Optional[np.ndarray] = None,
                     pitch: Optional[float] = None,
                     n_depths: int = 30) -> Dict[str, np.ndarray]:
    """Common field of view on planes swept along one camera's optical axis.

    For each depth a plane fronto-parallel to ``axis_camera`` is sampled over
    that camera's footprint; the result is the fraction of its own field of view
    shared with every other camera, plus the shared area in cm^2.
    """
    cam0 = axis_camera or rig.cameras[0]
    if depths is None:
        depths = np.linspace(cam0.near, cam0.far, n_depths)
    depths = np.asarray(depths, dtype=float)
    pitch = pitch or max(rig.analysis.voxel_pitch, 0.004)

    names = rig.names
    frac = {n: [] for n in names}
    area_all, frac_all, area_own = [], [], []

    for z in depths:
        w, h = cam0.footprint(z)
        nu = max(6, int(min(80, w / pitch)))
        nv = max(6, int(min(80, h / pitch)))
        gu = np.linspace(0.0, cam0.width - 1.0, nu)
        gv = np.linspace(0.0, cam0.height - 1.0, nv)
        uv = np.stack(np.meshgrid(gu, gv, indexing="ij"), axis=-1).reshape(-1, 2)
        pts = cam0.backproject(uv, z)
        cell = (w / nu) * (h / nv)

        vis = visibility_matrix(rig.cameras, pts, margin_px=rig.analysis.margin_px)
        for n, v in zip(names, vis):
            frac[n].append(float(v.mean()))
        all_ok = np.all(vis, axis=0)
        frac_all.append(float(all_ok.mean()))
        area_all.append(float(all_ok.sum() * cell) * 1e4)
        area_own.append(float(w * h) * 1e4)

    return {"depths_m": depths, "axis_camera": cam0.name,
            "fraction": {n: np.array(v) for n, v in frac.items()},
            "fraction_all": np.array(frac_all),
            "area_all_cm2": np.array(area_all),
            "area_axis_cm2": np.array(area_own)}


# --------------------------------------------------------------------------- #
# Everything at once
# --------------------------------------------------------------------------- #
@dataclass
class RigReport:
    rig: Rig
    volume: CoverageResult
    surface: CoverageResult
    resolution: Dict[str, Dict[str, float]]
    sensor_fill: Dict[str, float]
    stereo: Dict[Tuple[str, str], Dict[str, float]]
    registration: List[Dict[str, np.ndarray]]
    depth_sweep: Dict[str, np.ndarray]
    triangulation: Dict[str, np.ndarray] = field(default_factory=dict)

    def text(self) -> str:
        return format_report(self)


def evaluate(rig: Rig, quick: bool = False) -> RigReport:
    """Run the full metric suite on a rig."""
    v_pitch = rig.analysis.interactive_voxel_pitch if quick else rig.analysis.voxel_pitch
    s_pitch = (rig.analysis.interactive_voxel_pitch if quick
               else rig.analysis.surface_pitch)
    vol = volume_coverage(rig, pitch=v_pitch)
    surf = surface_coverage(rig, pitch=s_pitch)
    return RigReport(
        rig=rig,
        volume=vol,
        surface=surf,
        resolution=resolution_stats(rig, surf),
        sensor_fill=pixels_on_target(rig),
        stereo=stereo_pairs(rig),
        registration=[] if quick else registration_table(rig),
        depth_sweep={} if quick else overlap_vs_depth(rig),
        triangulation=triangulation_angles(rig, vol),
    )


def format_report(rep: RigReport) -> str:
    rig = rep.rig
    L: List[str] = []
    add = L.append

    add("=" * 78)
    add(f"CAMERA RIG REPORT - {rig.name}")
    add("=" * 78)
    add("")
    add("CAMERAS")
    for cam in rig.cameras:
        add("  " + cam.summary())
    add("")

    add(f"TARGET  {rig.target.kind} centre "
        f"({rig.target.center[0]:.3f}, {rig.target.center[1]:.3f}, "
        f"{rig.target.center[2]:.3f}) m")
    b = rig.target.bounds
    add(f"        extents x[{b[0]:.3f},{b[1]:.3f}] y[{b[2]:.3f},{b[3]:.3f}] "
        f"z[{b[4]:.3f},{b[5]:.3f}] m")
    add("")

    add("VOLUME COVERAGE (target voxels)")
    per_cam = rep.volume.per_camera_fraction()
    for name, f in per_cam.items():
        add(f"  {name:<10s} sees {f * 100:6.2f} % of the target volume")
    for k in range(1, len(rig) + 1):
        add(f"  >= {k} camera(s): {rep.volume.fraction_seen_by_at_least(k) * 100:6.2f} % "
            f"({rep.volume.measure_at_least(k) * 1e6:9.1f} cm^3)")
    add("")

    add("SURFACE COVERAGE (incidence limit "
        f"{rig.analysis.max_incidence_deg:.0f} deg)")
    for name, f in rep.surface.per_camera_fraction().items():
        add(f"  {name:<10s} sees {f * 100:6.2f} % of the target surface")
    for k in range(1, len(rig) + 1):
        add(f"  >= {k} camera(s): {rep.surface.fraction_seen_by_at_least(k) * 100:6.2f} % "
            f"({rep.surface.measure_at_least(k) * 1e4:9.1f} cm^2)")
    blind = rep.surface.n_points - int((rep.surface.n_views >= 1).sum())
    add(f"  blind surface samples: {blind} / {rep.surface.n_points}")
    add("")

    add("PAIRWISE OVERLAP (volume, IoU / share of each)")
    for (a, b_), m in rep.volume.pairwise_overlap().items():
        add(f"  {a:<10s} & {b_:<10s} IoU {m['iou'] * 100:6.2f} %   "
            f"{m['fraction_of_first'] * 100:6.2f} % of {a}   "
            f"{m['fraction_of_second'] * 100:6.2f} % of {b_}")
    add("")

    add("RESOLUTION ON TARGET SURFACE (effective, incidence-corrected)")
    for name, m in rep.resolution.items():
        if not m.get("n_points"):
            add(f"  {name:<10s} no visible surface samples")
            continue
        add(f"  {name:<10s} GSD {m['gsd_min_mm']:.3f} / {m['gsd_mean_mm']:.3f} / "
            f"{m['gsd_max_mm']:.3f} mm/px (min/mean/max)  "
            f"depth {m['depth_min_m']:.3f}-{m['depth_max_m']:.3f} m  "
            f"incidence mean {m.get('incidence_mean_deg', float('nan')):.1f} deg")
        if m.get("measurement_pixels", 1) > 1:
            add(f"  {'':<10s} smallest measurable feature "
                f"{m['spot_mean_mm']:.2f} mm "
                f"({m['measurement_pixels']}x{m['measurement_pixels']} px MFOV)")
    add("")

    add("SENSOR FILL (target bounding disc / sensor area)")
    for name, f in rep.sensor_fill.items():
        add(f"  {name:<10s} {f * 100:6.2f} %")
    add("")

    add("STEREO / PAIR GEOMETRY at target centre")
    for (a, b_), m in rep.stereo.items():
        add(f"  {a:<10s} & {b_:<10s} baseline {m['baseline_mm']:7.1f} mm  "
            f"triangulation {m['triangulation_angle_deg']:5.1f} deg  "
            f"axis angle {m['axis_angle_deg']:5.1f} deg  "
            f"depth sigma {m['depth_sigma_mm']:6.2f} mm")
    tri = rep.triangulation
    if tri and tri["multi_view"].any():
        ang = tri["angle_deg"][tri["multi_view"]]
        add(f"  multi-view voxels: best triangulation angle "
            f"{ang.min():.1f} / {ang.mean():.1f} / {ang.max():.1f} deg (min/mean/max)")
    add("")

    if rep.registration:
        add("DEPTH-DEPENDENT REGISTRATION ERROR")
        add("  (single homography calibrated at the reference depth)")
        for r in rep.registration:
            z0 = r["reference_depth_m"]
            d = r["depths_m"]
            add(f"  {r['source']} -> {r['target']}  reference depth {z0:.3f} m")
            for frac in (0.5, 0.75, 1.25, 1.5):
                z = z0 * frac
                if z < d[0] or z > d[-1]:
                    continue
                i = int(np.argmin(np.abs(d - z)))
                add(f"      at {d[i]:.3f} m ({frac:.0%} of ref): "
                    f"rms {r['rms_px'][i]:8.1f} px / {r['rms_mm'][i]:6.2f} mm   "
                    f"max {r['max_px'][i]:8.1f} px / {r['max_mm'][i]:6.2f} mm")
        add("")

    ds = rep.depth_sweep
    if ds:
        add(f"COMMON FIELD OF VIEW along {ds['axis_camera']}'s optical axis")
        d = ds["depths_m"]
        for i in np.linspace(0, len(d) - 1, min(8, len(d))).astype(int):
            add(f"  z = {d[i]:.3f} m : all cameras share "
                f"{ds['fraction_all'][i] * 100:6.2f} % of {ds['axis_camera']}'s view "
                f"({ds['area_all_cm2'][i]:8.1f} cm^2 of {ds['area_axis_cm2'][i]:8.1f} cm^2)")
        add("")

    add("=" * 78)
    return "\n".join(L)
