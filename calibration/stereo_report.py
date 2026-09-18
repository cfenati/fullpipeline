"""Text report, figures and image previews for a calibrated camera pair."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".matplotlib_cache")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration.stereo import (
    Rectification,
    StereoExtrinsics,
    StereoObservation,
    CaptureBlockSummary,
    WEAK_TRIANGULATION_DEG,
    rectify_maps,
)
from calibration.stereo_metrics import (
    DepthSweep,
    RegistrationCurve,
    overlap_masks,
    overlap_outline,
)

COLOR_A = "#2563eb"
COLOR_B = "#16a34a"
COLOR_SHARED = "#d97706"
COLOR_WARN = "#dc2626"

# Depths highlighted on the overlap map, as fractions of the reference depth.
MAP_DEPTH_FRACTIONS = (0.6, 1.0, 1.6)
# Registration tolerance used to quote a usable depth of field.
REGISTRATION_TOLERANCE_PX = 2.0


@dataclass
class StereoReport:
    """Everything measured about one pair, ready to be written out."""

    extrinsics: StereoExtrinsics
    observations: List[StereoObservation]
    views: List[Dict[str, Any]]
    scatter: Dict[str, float]
    epipolar: Dict[str, Any]
    rectification: Rectification
    rectified: Dict[str, Any]
    closure: Dict[str, Any]
    sweep: DepthSweep
    registration: RegistrationCurve
    uncertainty: Optional[RegistrationCurve] = None
    warnings: List[str] = field(default_factory=list)
    # Kept in the fit, but named so they can be inspected; see
    # :func:`calibration.stereo.suspect_views`. Delete their folders by hand or
    # re-run with --exclude if you want them out of the next fit.
    suspect: Dict[str, List[str]] = field(default_factory=dict)
    blocks: List[CaptureBlockSummary] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    capture_dirs: List[Path] = field(default_factory=list)

    @property
    def reference_depth(self) -> float:
        return self.registration.reference_depth_m

    @property
    def image_paths(self) -> Dict[str, Tuple[Path, Path]]:
        """Both source images of every view, by session name."""
        return {
            observation.label: (observation.left.image_path, observation.right.image_path)
            for observation in self.observations
        }

    def epipolar_rms(self, label: str) -> float:
        return float(self.epipolar["per_view_rms_px"].get(label, float("nan")))


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
def _table_depths(sweep: DepthSweep, rows: int = 9) -> np.ndarray:
    """A readable subset of the sampled depths."""
    return np.unique(np.linspace(0, len(sweep.depths_m) - 1, rows).astype(int))


def _cell(value: float, width: int, decimals: int = 2) -> str:
    """A right-aligned number, or a dash where the metric has no value."""
    if value is None or not np.isfinite(value):
        return f"{'-':>{width}}"
    return f"{value:{width}.{decimals}f}"


def format_report(report: StereoReport) -> str:
    extrinsics = report.extrinsics
    sweep = report.sweep
    board = extrinsics.board or {}
    lines: List[str] = []
    add = lines.append
    rule = "-" * 74

    width_a, height_a = extrinsics.image_size_a
    width_b, height_b = extrinsics.image_size_b

    add(f"Extrinsic calibration: {extrinsics.name_a} -> {extrinsics.name_b}")
    add("=" * 74)
    add("")
    if board:
        add(f"  target:            {board.get('squares_x')}x{board.get('squares_y')} ChArUco, "
            f"{board.get('dictionary')}, legacy_pattern={board.get('legacy_pattern')}")
        add(f"  square / marker:   {float(board.get('square_size_m', 0)) * 1000:.2f} mm / "
            f"{float(board.get('marker_size_m', 0)) * 1000:.2f} mm")
    add(f"  resolution:        {width_a}x{height_a} ({extrinsics.name_a}) / "
        f"{width_b}x{height_b} ({extrinsics.name_b})")
    add(f"  intrinsics:        {'held fixed' if extrinsics.intrinsics_fixed else 're-fitted here'}")
    for capture_dir in report.capture_dirs:
        add(f"  captures:          {capture_dir}")
    add("")
    add(f"  stereo views:      {len(report.observations)}")
    add(f"  views used:        {extrinsics.views_used}")
    add(f"  shared corners:    {extrinsics.points_used}")
    add(f"  epipolar RMS:      {report.epipolar['rms_px']:.4f} px")
    add("")

    add("RELATIVE POSE   X_b = R @ X_a + T")
    add(rule)
    for row in extrinsics.R:
        add("  R    " + "  ".join(f"{value:+10.7f}" for value in row))
    add("  T    " + "  ".join(f"{value * 1000:+10.4f}" for value in extrinsics.T.reshape(3))
        + "   mm")
    add("")
    center = extrinsics.center_b_in_a * 1000.0
    tilt, pan, roll = extrinsics.rotation_xyz_deg
    add(f"  baseline                    {extrinsics.baseline_m * 1000:9.3f} mm")
    add(f"  {extrinsics.name_b} seen from {extrinsics.name_a}:"
        f" right {center[0]:+.2f}, down {center[1]:+.2f}, forward {center[2]:+.2f} mm")
    add(f"  rotation about A's axes     tilt(x) {tilt:+.3f}, pan(y) {pan:+.3f}, "
        f"roll(z) {roll:+.3f} deg")
    add(f"  total rotation              {extrinsics.rotation_angle_deg:9.3f} deg")
    add(f"  optical axes converge by    {extrinsics.optical_axis_angle_deg:9.3f} deg")
    add(f"  baseline vs A's axis        {extrinsics.baseline_axis_angle_deg:9.3f} deg"
        "   (90 = side by side)")
    add("")
    add(f"  Metric scale rests entirely on the printed square size "
        f"({float(board.get('square_size_m', 0)) * 1000:.2f} mm); a 1 % error")
    add("  there is a 1 % error in every distance below.")
    add("")

    add("FIT QUALITY")
    add(rule)
    epipolar = report.epipolar
    add(f"  epipolar rms / p95 / max    {epipolar['rms_px']:8.3f} /{epipolar['p95_px']:7.3f} /"
        f"{epipolar['max_px']:7.3f} px")
    rectified = report.rectified
    add(f"  rectified row misalignment  {rectified['vertical_rms_px']:8.3f} px rms, "
        f"{rectified['vertical_max_px']:.3f} px max")
    closure = report.closure
    add(f"  board reconstructed to      {closure['rms_mm']:8.3f} mm rms, "
        f"{closure['max_mm']:.3f} mm max")
    add(f"  reconstruction scale        {closure['scale_mean']:8.5f} "
        f"+/- {closure['scale_std']:.5f}   (1.0 = consistent)")
    add("")
    add("  Quote the epipolar residual: it excludes board-pose and intrinsics error, so it")
    add("  isolates the pair and applies beyond this one board.")
    add("")
    floor = extrinsics.intrinsic_rms_px
    if floor is not None:
        add(f"  For reference: the fit's own residual is {extrinsics.reprojection_error_px:.3f} px, "
            f"{extrinsics.reprojection_error_px / floor:.1f}x the {floor:.3f} px floor already")
        add("  in these fixed intrinsics. Only that ratio matters -- the fit inherits the")
        add("  intrinsics' error and cannot beat it.")
        add("")

    if report.scatter:
        scatter = report.scatter
        add("RIG RIGIDITY   each view solved on its own")
        add(rule)
        add(f"  baseline over {scatter['n_views']:2d} views      "
            f"{scatter['baseline_mean_mm']:8.3f} +/- {scatter['baseline_std_mm']:.3f} mm"
            f"  ({scatter['baseline_spread'] * 100:.2f} %)")
        add(f"  rotation deviation          {scatter['rotation_deviation_mean_deg']:8.3f} deg mean, "
            f"{scatter['rotation_deviation_max_deg']:.3f} deg max")
        add(f"  translation deviation       {scatter['translation_deviation_rms_mm']:8.3f} mm rms")
        add(f"  board distance              {scatter['depth_min_m']:.3f} / "
            f"{scatter['depth_mean_m']:.3f} / {scatter['depth_max_m']:.3f} m (min/mean/max)")
        add("")
        add("  A rigid pair gives the same baseline every view; scatter here is measurement")
        add("  repeatability, and the only sign of a bumped camera.")
        add("")

    if report.blocks:
        add("CAPTURE BLOCKS   each sitting fitted on its own")
        add(rule)
        add(f"  {'#':>3}{'views':>7}{'baseline':>11}{'conv.':>9}{'epipolar':>10}"
            f"{'depth range':>16}  span")
        add(f"  {'':>3}{'':>7}{'[mm]':>11}{'[deg]':>9}{'[px]':>10}{'[m]':>16}")
        for block in report.blocks:
            add(f"  {block.index:3d}{block.n_views:7d}{block.baseline_mm:11.2f}"
                f"{block.convergence_deg:9.2f}{block.epipolar_rms_px:10.3f}"
                f"  {block.depth_min_m:.3f}-{block.depth_max_m:.3f}  {block.span}")
        if len(report.blocks) >= 2:
            baselines = [block.baseline_mm for block in report.blocks]
            convergences = [block.convergence_deg for block in report.blocks]
            add("")
            add(f"  across sittings: baseline {min(baselines):.2f} to {max(baselines):.2f} mm, "
                f"convergence {min(convergences):.2f} to {max(convergences):.2f} deg.")
            add("  A rigid mount keeps these numbers steady; disagreement means the pose shifted")
            add("  between sittings. Fit one with --block N, or recapture in one session.")
        add("")

    add(f"COVERAGE OVERLAP   on planes fronto-parallel to {extrinsics.name_a}")
    add(rule)
    add(f"  {'depth':>8}{'share of A':>11}{'share of B':>11}{'IoU':>7}"
        f"{'shared':>11}{'A footprint':>13}")
    for i in _table_depths(sweep):
        add(f"  {sweep.depths_m[i]:7.3f}m"
            + _cell(sweep.fraction_of_a[i] * 100, 10, 1) + "%"
            + _cell(sweep.fraction_of_b[i] * 100, 10, 1) + "%"
            + _cell(sweep.iou[i] * 100, 6, 1) + "%"
            + _cell(sweep.shared_area_cm2[i], 8, 1) + "cm2"
            + _cell(sweep.area_a_cm2[i], 10, 1) + "cm2")
    best = sweep.best_overlap_depth()
    add(f"  Overlap peaks at {best:.3f} m. That's a working-distance choice, not a rig")
    add("  property -- a fixed pose overlaps more far away and less up close.")
    if sweep.unbounded_depths:
        add(f"  ({extrinsics.name_b}'s field does not close on the planes at or below "
            f"{max(sweep.unbounded_depths):.3f} m, so its area and the IoU are left blank)")
    add("")

    add("TRIANGULATION GEOMETRY AND DEPTH UNCERTAINTY")
    add(rule)
    add(f"  {'depth':>8}{'angle on axis':>15}{'angle over overlap':>24}"
        f"{'depth sigma':>13}{'disparity':>11}")
    add(f"  {'':>8}{'[deg]':>15}{'min / mean / max':>24}"
        f"{f'@{sweep.disparity_noise_px:.2f}px':>13}{'[px]':>11}")
    for i in _table_depths(sweep):
        add(f"  {sweep.depths_m[i]:7.3f}m"
            + _cell(sweep.triangulation_deg["axis"][i], 15)
            + _cell(sweep.triangulation_deg["min"][i], 8) + " /"
            + _cell(sweep.triangulation_deg["mean"][i], 6) + " /"
            + _cell(sweep.triangulation_deg["max"][i], 6)
            + _cell(sweep.depth_sigma_mm["axis"][i], 11, 3) + "mm"
            + _cell(sweep.disparity_px[i], 11, 0))
    add("")
    add(f"  sigma_z assumes {sweep.disparity_noise_px:.2f} px of matching noise, propagated as")
    add("  noise * distance / (f * sin(angle)) -- the familiar z^2 * noise / (f * baseline)")
    add("  for a rectified pair, but valid for converging cameras too.")
    add("")

    registration = report.registration
    add("DEPTH-DEPENDENT REGISTRATION ERROR")
    add(f"  one fixed pixel mapping {extrinsics.name_a} -> {extrinsics.name_b}, "
        f"calibrated on the plane at {registration.reference_depth_m:.3f} m")
    add(rule)
    add(f"  {'true depth':>12}{'rms':>10}{'max':>10}{'rms':>10}{'max':>10}"
        f"{'closed form':>14}")
    add(f"  {'':>12}{'[px]':>10}{'[px]':>10}{'[mm]':>10}{'[mm]':>10}{'[px]':>14}")
    for i in _table_depths_curve(registration):
        add(f"  {registration.depths_m[i]:11.3f}m"
            + _cell(registration.rms_px[i], 10)
            + _cell(registration.max_px[i], 10)
            + _cell(registration.rms_mm[i], 10, 3)
            + _cell(registration.max_mm[i], 10, 3)
            + _cell(registration.analytic_px[i], 14))
    reference = registration.reference_depth_m
    near, far = registration.depth_for_error(REGISTRATION_TOLERANCE_PX)
    if np.isfinite(near) and np.isfinite(far):
        add(f"  Stays under {REGISTRATION_TOLERANCE_PX:.0f} px only within {near:.4f}-{far:.4f} m "
            f"(-{(reference - near) * 1000:.1f}/+{(far - reference) * 1000:.1f} mm of reference).")
        add(f"  Anything thicker than {(far - near) * 1000:.1f} mm needs a per-pixel depth map.")
    elif np.isfinite(near):
        add(f"  under {REGISTRATION_TOLERANCE_PX:.0f} px for everything beyond {near:.3f} m")
    if report.uncertainty is not None:
        floor = report.uncertainty.at(reference)
        add(f"  Extrinsics-only noise floor: {floor['rms_px']:.2f} px "
            f"({floor['rms_mm']:.3f} mm) at the reference plane -- a depth map removes")
        add("  the parallax error above but not this floor.")
    add("")

    rectification = report.rectification
    add(f"RECTIFICATION   alpha = {rectification.alpha:.2f}")
    add(rule)
    valid_a, valid_b = rectification.valid_fraction
    add(f"  rectified frame             {rectification.image_size[0]}x{rectification.image_size[1]}")
    add(f"  rectified focal             {rectification.focal_px:9.2f} px")
    add(f"  rectified baseline          {rectification.baseline_m * 1000:9.3f} mm")
    add(f"  frame reached by real pixels{valid_a * 100:8.1f} % / {valid_b * 100:.1f} %")
    add(f"  disparity seen on the board {rectified['disparity_min_px']:8.1f} to "
        f"{rectified['disparity_max_px']:.1f} px "
        f"({rectified['depth_min_m']:.3f} to {rectified['depth_max_m']:.3f} m)")
    add("")
    if rectification.rotated:
        add("  Frame is rotated 90 deg: rectification aligns rows to the baseline, which is")
        add("  vertical here.")
    if rectification.degenerate:
        add("  Alpha scaling degenerated (no common rectangle survives): focal length above is")
        add(f"  a plain average of both cameras, and both frames center on the {reference:.3f} m")
        add("  plane instead. Row alignment is unaffected; the wasted frame area is real.")
    add(f"  These depths follow the rectified axis, tilted off {extrinsics.name_a}'s optical")
    add("  axis -- they need not match the board distances above.")
    if rectification.disparity_at_infinity_px:
        add(f"  Zero disparity is the {reference:.3f} m plane, not infinity (infinity sits at "
            f"{rectification.disparity_at_infinity_px:.1f} px) --")
        add("  convert with Q or depth_from_disparity, not f*B/d.")
    if rectified["disparity_reversed"]:
        add(f"  Disparity decreases with range: {extrinsics.name_a} is the second camera along the")
        add("  rectified axis. Swap --camera-a/--camera-b if a block matcher expects positive")
        add("  disparity.")
    add("")

    if report.warnings:
        add("Quality warnings")
        add(rule)
        for warning in report.warnings:
            add(f"  - {warning}")
        add("")

    if report.suspect:
        add("Views that disagree with the set")
        add(rule)
        add("  Kept in the fit for reproducibility from the captures alone. Remove with")
        add("  --exclude <session> ..., or delete the session folder and re-run")
        add("  stereo_calibrate.py.")
        add("")
        for label in sorted(report.suspect, key=report.epipolar_rms, reverse=True):
            add(f"  {label}")
            for reason in report.suspect[label]:
                add(f"      {reason}")
            for image_path in report.image_paths.get(label, ()):
                add(f"      {image_path}")
        add("")

    add("Per-view residuals   (* = flagged above)")
    add(rule)
    add(f"  {'session':<20}{'corners':>9}{'epipolar px':>13}{'stereo px':>11}"
        f"{'baseline mm':>13}{'depth m':>10}")
    by_label = {view["label"]: view for view in report.views}
    for label, epipolar_view in sorted(
        report.epipolar["per_view_rms_px"].items(), key=lambda item: item[1], reverse=True
    ):
        view = by_label.get(label, {})
        error = extrinsics.per_view_errors.get(label, float("nan"))
        mark = "*" if label in report.suspect else " "
        add(f"  {label:<19}{mark}{view.get('corners', 0):>9}{epipolar_view:>13.3f}{error:>11.3f}"
            f"{view.get('baseline_m', float('nan')) * 1000:>13.3f}"
            f"{view.get('depth_a_m', float('nan')):>10.3f}")

    if extrinsics.rejected_views:
        add("")
        add("Rejected as outliers (px)")
        add(rule)
        for label, error in sorted(
            extrinsics.rejected_views.items(), key=lambda item: item[1], reverse=True
        ):
            add(f"  {label:<20}{error:>11.3f}")

    if report.skipped:
        add("")
        add("Sessions without a usable stereo view")
        add(rule)
        for label, reason in report.skipped:
            add(f"  {label:<20} {reason}")

    if report.notes:
        add("")
        add("Capture notes")
        add(rule)
        for note in report.notes:
            add(f"  {note}")

    return "\n".join(lines) + "\n"


def _table_depths_curve(curve: RegistrationCurve, rows: int = 11) -> np.ndarray:
    return np.unique(np.linspace(0, len(curve.depths_m) - 1, rows).astype(int))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _mark_depths(axis, report: StereoReport, legend: bool = False) -> None:
    """Reference depth and the range the board was actually held at."""
    scatter = report.scatter
    if scatter:
        axis.axvspan(scatter["depth_min_m"], scatter["depth_max_m"], color="#e2e8f0",
                     zorder=0, label="board distance in calibration" if legend else None)
    axis.axvline(report.reference_depth, color="k", ls=":", lw=1.3,
                 label=f"reference {report.reference_depth:.3f} m" if legend else None)


def figure_metrics(report: StereoReport, path: Path) -> Path:
    """Overlap, triangulation, depth uncertainty and registration versus distance."""
    sweep = report.sweep
    depths = sweep.depths_m
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 8.4))

    axis = axes[0, 0]
    axis.plot(depths, sweep.fraction_of_a * 100, color=COLOR_A, lw=1.9,
              label=f"seen by {report.extrinsics.name_b} too")
    axis.plot(depths, sweep.fraction_of_b * 100, color=COLOR_B, lw=1.9,
              label=f"seen by {report.extrinsics.name_a} too")
    axis.plot(depths, sweep.iou * 100, color="k", ls="--", lw=1.7, label="IoU")
    _mark_depths(axis, report, legend=True)
    axis.set_ylabel("share of the field of view [%]")
    axis.set_title("Coverage overlap")
    axis.set_ylim(0, 105)

    axis = axes[0, 1]
    axis.plot(depths, sweep.area_a_cm2, color=COLOR_A, lw=1.7,
              label=f"{report.extrinsics.name_a} footprint")
    axis.plot(depths, sweep.shared_area_cm2, color=COLOR_SHARED, lw=2.0, label="shared")
    _mark_depths(axis, report)
    axis.set_ylabel("area [cm$^2$]")
    axis.set_title("Imaged and shared area")
    axis.set_yscale("log")

    axis = axes[0, 2]
    axis.fill_between(depths, sweep.triangulation_deg["min"], sweep.triangulation_deg["max"],
                      color=COLOR_SHARED, alpha=0.22, label="over the overlap")
    axis.plot(depths, sweep.triangulation_deg["axis"], color=COLOR_SHARED, lw=2.0,
              label="on the optical axis")
    axis.axhline(WEAK_TRIANGULATION_DEG, color=COLOR_WARN, ls="--", lw=1.2,
                 label=f"{WEAK_TRIANGULATION_DEG:g} deg, weak depth")
    _mark_depths(axis, report)
    axis.set_ylabel("triangulation angle [deg]")
    axis.set_title("Triangulation geometry")

    axis = axes[1, 0]
    axis.plot(depths, sweep.depth_sigma_mm["axis"], color="#7c3aed", lw=2.0,
              label=f"depth sigma at {sweep.disparity_noise_px:.2f} px match noise")
    axis.plot(depths, sweep.gsd_a_mm, color="#64748b", lw=1.5, ls="-.",
              label=f"{report.extrinsics.name_a} pixel footprint")
    _mark_depths(axis, report)
    axis.set_ylabel("[mm]")
    axis.set_title("Depth uncertainty vs lateral resolution")
    axis.set_yscale("log")

    registration = report.registration
    axis = axes[1, 1]
    axis.plot(registration.depths_m, registration.rms_px, color=COLOR_A, lw=2.0,
              label="parallax, rms")
    axis.plot(registration.depths_m, registration.max_px, color=COLOR_A, lw=1.2, ls="--",
              label="parallax, max")
    axis.plot(registration.depths_m, registration.analytic_px, color="#64748b", lw=1.2,
              ls=":", label="closed form f*B*|1/z - 1/z0|")
    if report.uncertainty is not None:
        axis.plot(report.uncertainty.depths_m, report.uncertainty.rms_px, color=COLOR_WARN,
                  lw=1.8, label="extrinsic uncertainty floor")
    _mark_depths(axis, report)
    axis.set_ylabel("registration error [px]")
    axis.set_title("Depth-dependent registration error")
    axis.set_yscale("log")

    axis = axes[1, 2]
    axis.plot(registration.depths_m, registration.rms_mm, color=COLOR_A, lw=2.0,
              label="parallax, rms")
    axis.plot(registration.depths_m, registration.max_mm, color=COLOR_A, lw=1.2, ls="--",
              label="parallax, max")
    if report.uncertainty is not None:
        axis.plot(report.uncertainty.depths_m, report.uncertainty.rms_mm, color=COLOR_WARN,
                  lw=1.8, label="extrinsic uncertainty floor")
    _mark_depths(axis, report)
    axis.set_ylabel("registration error on the subject [mm]")
    axis.set_title("Registration error, metric")
    axis.set_yscale("log")

    for axis in axes.ravel():
        axis.set_xlabel("distance from "
                        f"{report.extrinsics.name_a} [m]")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=7)

    figure.suptitle(
        f"{report.extrinsics.name_a} + {report.extrinsics.name_b}: measured pair geometry "
        f"(baseline {report.extrinsics.baseline_m * 1000:.1f} mm, "
        f"convergence {report.extrinsics.optical_axis_angle_deg:.2f} deg)",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(figure, path)


def figure_fit_quality(report: StereoReport, path: Path) -> Path:
    """Per-view residuals, residual distributions and the rigidity check."""
    extrinsics = report.extrinsics
    suspect = set(report.suspect)
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    labels = list(extrinsics.per_view_errors)
    order = np.argsort([report.epipolar_rms(label) for label in labels])[::-1]
    labels = [labels[i] for i in order]
    stereo_errors = [extrinsics.per_view_errors[label] for label in labels]
    epipolar_errors = [report.epipolar_rms(label) for label in labels]

    x = np.arange(len(labels))
    axis = axes[0]
    axis.bar(x - 0.2, stereo_errors, width=0.38, color=COLOR_A, label="stereo rms")
    axis.bar(x + 0.2, epipolar_errors, width=0.38, color=COLOR_SHARED, label="epipolar rms")
    axis.set_xticks(x)
    axis.set_xticklabels([label[-6:] for label in labels], rotation=90, fontsize=6)
    # Red tick labels rather than red bars: which of the two residuals triggered
    # the flag is in the report, and recolouring a bar would break the legend.
    for tick, label in zip(axis.get_xticklabels(), labels):
        if label in suspect:
            tick.set_color(COLOR_WARN)
            tick.set_fontweight("bold")
    axis.set_ylabel("[px]")
    axis.set_xlabel("session   (red = flagged)" if suspect else "session")
    axis.set_title("Per-view residuals")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=8)

    axis = axes[1]
    residuals = report.epipolar["residuals_px"]
    if residuals.size:
        axis.hist(residuals, bins=40, color=COLOR_SHARED, alpha=0.75, label="epipolar distance")
    vertical = np.abs(report.rectified["vertical_px"])
    if vertical.size:
        axis.hist(vertical, bins=40, color=COLOR_A, alpha=0.55,
                  label="rectified row offset")
    axis.set_xlabel("residual [px]")
    axis.set_ylabel("corners")
    axis.set_title("Residual distribution over all corners")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=8)

    axis = axes[2]
    if report.views:
        depths = np.array([view["depth_a_m"] for view in report.views])
        baselines = np.array([view["baseline_m"] for view in report.views]) * 1000.0
        flagged = np.array([view["label"] in suspect for view in report.views])
        axis.scatter(depths[~flagged], baselines[~flagged], s=26, color=COLOR_A, zorder=3,
                     label="single view")
        if flagged.any():
            axis.scatter(depths[flagged], baselines[flagged], s=90, marker="X",
                         color=COLOR_WARN, zorder=4, label="flagged")
            for view, depth, baseline in zip(report.views, depths, baselines):
                if view["label"] in suspect:
                    axis.annotate(view["label"][-6:], (depth, baseline), fontsize=6,
                                  color=COLOR_WARN, textcoords="offset points", xytext=(7, 3))
        axis.axhline(extrinsics.baseline_m * 1000.0, color="k", lw=1.6,
                     label=f"fitted {extrinsics.baseline_m * 1000:.2f} mm")
        if report.scatter:
            mean = report.scatter["baseline_mean_mm"]
            std = report.scatter["baseline_std_mm"]
            axis.axhspan(mean - std, mean + std, color="#e2e8f0", zorder=0,
                         label="+/- 1 sigma of the views")
    axis.set_xlabel("board distance [m]")
    axis.set_ylabel("baseline [mm]")
    axis.set_title("Rigidity: baseline from each view alone")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    return _save(figure, path)


# --------------------------------------------------------------------------- #
# Image previews
# --------------------------------------------------------------------------- #
def overlap_map_image(report: StereoReport, path: Path) -> Path:
    """Camera A's frame, shaded where camera B sees the same scene.

    The shaded region is the working area of the pair at the reference depth; the
    outlines show how it slides across the frame as the subject moves.
    """
    extrinsics = report.extrinsics
    width, height = extrinsics.image_size_a
    reference = report.reference_depth
    depths = [reference * fraction for fraction in MAP_DEPTH_FRACTIONS]

    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    colors = [(120, 200, 255), (80, 220, 120), (90, 120, 255)]

    masks, _ = overlap_masks(extrinsics, [reference], step=4)
    reference_mask = cv2.resize(
        masks[0].astype(np.uint8) * 255, (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    shaded = canvas.copy()
    shaded[reference_mask > 0] = (60, 110, 40)
    canvas = cv2.addWeighted(canvas, 0.35, shaded, 0.65, 0)

    for index, depth in enumerate(depths):
        # The outline is computed geometrically rather than traced around the mask,
        # which is sampled every few pixels and would come out as a staircase.
        outline = overlap_outline(extrinsics, depth)
        if len(outline) < 3:
            continue
        cv2.polylines(canvas, [np.round(outline).astype(np.int32)], True,
                      colors[index % len(colors)], 3, cv2.LINE_AA)

    summary = report.sweep.at(reference)
    legend = [f"z = {depth * 1000:.0f} mm" for depth in depths]
    caption = (f"{extrinsics.name_a} frame; shaded = also seen by {extrinsics.name_b} "
               f"at {reference * 1000:.0f} mm ({summary['fraction_of_a'] * 100:.1f} %)")

    # The legend sits on top of the outlines, so darken the strip under it.
    strip_top = height - 44 - 30 * len(legend)
    backdrop = canvas.copy()
    cv2.rectangle(backdrop, (0, strip_top), (width, height), (16, 16, 16), -1)
    canvas = cv2.addWeighted(canvas, 0.25, backdrop, 0.75, 0)

    for index, text in enumerate(legend):
        baseline_y = strip_top + 30 * (index + 1)
        cv2.putText(canvas, text, (16, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    colors[index % len(colors)], 2, cv2.LINE_AA)
    cv2.putText(canvas, caption, (16, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (235, 235, 235), 2, cv2.LINE_AA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    return path


def rectified_preview(report: StereoReport, path: Path, rows: int = 16) -> Optional[Path]:
    """The best stereo view, rectified, with rulers to check row alignment.

    A correctly co-registered pair puts the same board corner on the same line in
    both halves; anything else is visible immediately.
    """
    if not report.observations:
        return None
    best = max(report.observations, key=lambda observation: observation.corner_count)
    image_a = cv2.imread(str(best.left.image_path))
    image_b = cv2.imread(str(best.right.image_path))
    if image_a is None or image_b is None:
        return None

    map_a = rectify_maps(report.extrinsics, report.rectification, "a")
    map_b = rectify_maps(report.extrinsics, report.rectification, "b")
    warped_a = cv2.remap(image_a, *map_a, cv2.INTER_LINEAR)
    warped_b = cv2.remap(image_b, *map_b, cv2.INTER_LINEAR)

    for image, (roi, name) in zip(
        (warped_a, warped_b),
        ((report.rectification.roi_a, report.extrinsics.name_a),
         (report.rectification.roi_b, report.extrinsics.name_b)),
    ):
        if roi[2] > 0 and roi[3] > 0:
            cv2.rectangle(image, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]),
                          (0, 220, 220), 2)
        cv2.putText(image, f"{name} rectified", (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)

    canvas = np.hstack([warped_a, warped_b])
    for row in np.linspace(0, canvas.shape[0] - 1, rows).astype(int):
        cv2.line(canvas, (0, row), (canvas.shape[1] - 1, row), (0, 255, 0), 1)
    cv2.putText(
        canvas,
        f"{best.label}: a corner must sit on the same green line in both halves "
        f"(rms {report.rectified['vertical_rms_px']:.2f} px)",
        (16, canvas.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        cv2.LINE_AA,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    return path


def correspondence_preview(report: StereoReport, path: Path) -> Optional[Path]:
    """Shared corners in both raw frames, joined, to prove the pairing is right.

    What matters is that the lines form one orderly sheaf: the board must be the
    same board, seen twice. They may well cross, and they do whenever one camera
    is rolled with respect to the other. A tangle of lines going in different
    directions instead means corner ids were matched across two different views of
    the target, and everything downstream is meaningless.
    """
    if not report.observations:
        return None
    best = max(report.observations, key=lambda observation: observation.corner_count)
    image_a = cv2.imread(str(best.left.image_path))
    image_b = cv2.imread(str(best.right.image_path))
    if image_a is None or image_b is None:
        return None

    canvas = np.hstack([image_a, image_b])
    offset = image_a.shape[1]
    points_a = best.left_points.reshape(-1, 2)
    points_b = best.right_points.reshape(-1, 2)
    for (ua, va), (ub, vb) in zip(points_a, points_b):
        start = (int(round(ua)), int(round(va)))
        end = (int(round(ub)) + offset, int(round(vb)))
        cv2.line(canvas, start, end, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, start, 3, (0, 0, 255), -1)
        cv2.circle(canvas, end, 3, (0, 0, 255), -1)
    cv2.putText(canvas, f"{report.extrinsics.name_a}", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{report.extrinsics.name_b}", (offset + 16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{best.label}: {best.corner_count} shared corners",
                (16, canvas.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    return path


# --------------------------------------------------------------------------- #
def write_report(report: StereoReport, directory: Path, figures: bool = True) -> List[Path]:
    """Write the text report, the figures and the visual checks."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    paths = [directory / "report.txt"]
    paths[0].write_text(format_report(report), encoding="utf-8")

    if figures:
        paths.append(figure_metrics(report, directory / "pair_geometry.png"))
        paths.append(figure_fit_quality(report, directory / "fit_quality.png"))
    paths.append(overlap_map_image(report, directory / "overlap_map.png"))
    for maker, name in ((rectified_preview, "rectified.jpg"),
                        (correspondence_preview, "correspondences.jpg")):
        out = maker(report, directory / name)
        if out is not None:
            paths.append(out)
    return paths
