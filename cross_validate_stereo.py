#!/usr/bin/env python3
"""Check a fitted stereo pair against a held-out checkerboard pose set.

``stereo_calibrate.py`` grades its own fit on the poses it was fit from, so a
stereo calibration that happens to fit that particular set of board poses
well looks perfect in its own report even when it does not generalise — the
classic overfitting blind spot that in-sample reprojection error cannot see.
This script reuses a *second*, independently captured set of checkerboard
poses (default ``captures/cross-validation``) that never went into the fit,
and checks the frozen extrinsics against it: it triangulates every shared
corner and compares the reconstructed corner-to-corner distances against the
board's printed square size (nothing fitted here to hide a scale bias behind),
and it reports the epipolar and triangulation-closure residuals on these
unseen views for comparison with the numbers in
``calibration/results/stereo_<a>_<b>/report.txt``.

This script is read-only with respect to every existing calibration
artifact — it never re-fits or rewrites ``extrinsics.json``,
``intrinsics.json``, or anything else already under
``calibration/results/<pair>/``. Results land in a new ``cross_validation/``
subfolder next to — never over — the calibration they check.

    python cross_validate_stereo.py
    python cross_validate_stereo.py --in captures/cross-validation --camera-a rgb_cam1 --camera-b rgb_cam2
    python cross_validate_stereo.py --extrinsics calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from calibrate_cameras import load_config, resolve_path  # noqa: E402
from calibration.opencv_calibrate import DEFAULT_MIN_CORNERS  # noqa: E402
from calibration.stereo import (  # noqa: E402
    DEFAULT_MIN_SHARED_CORNERS,
    StereoExtrinsics,
    adjacent_corner_distance_errors,
    collect_session_pairs,
    detect_stereo_observations,
    epipolar_residuals,
    triangulation_closure,
)
from calibration.target_board import TargetBoard  # noqa: E402

DEFAULT_OUTPUT_DIR = "calibration/results"
DEFAULT_CROSS_VALIDATION_CAPTURES = "captures/cross-validation"
# A calibration whose out-of-sample corner-distance error exceeds this is
# reproducing the training board's geometry rather than the true metric
# scale; see adjacent_corner_distance_errors in calibration/stereo.py.
WARN_RELATIVE_ERROR_PCT = 1.0


def _essential_from_pose(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """E = [T]_x @ R, the essential matrix implied by the pose alone.

    Used as a fallback when a loaded extrinsics.json has essential left at
    its all-zero default (e.g. a hand-edited or pre-`essential`-field file):
    a zero E silently makes epipolar_residuals report a perfect 0.0000 px
    residual for every corner, which is exactly the kind of impossibly-good
    number this tool exists to catch.
    """
    t = T.reshape(3)
    skew = np.array([
        [0.0, -t[2], t[1]],
        [t[2], 0.0, -t[0]],
        [-t[1], t[0], 0.0],
    ])
    return skew @ R


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a fitted stereo pair against a held-out checkerboard pose set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Read-only: never re-fits or rewrites extrinsics.json / intrinsics.json.\n"
               "Results land in calibration/results/stereo_<a>_<b>/cross_validation/.",
    )
    parser.add_argument(
        "--in", "--captures", dest="captures", nargs="+", default=None,
        help="Directories holding held-out capture session subfolders "
             "(default: geometric_calibration.cross_validation_captures, else "
             f"{DEFAULT_CROSS_VALIDATION_CAPTURES}).",
    )
    parser.add_argument("--camera-a", default="rgb_cam1",
                        help="Reference camera (default: %(default)s).")
    parser.add_argument("--camera-b", default="rgb_cam2",
                        help="Second camera of the pair (default: %(default)s).")
    parser.add_argument(
        "--extrinsics", default=None,
        help="Extrinsics JSON to validate (default: geometric_calibration."
             "extrinsics_<a>_<b>, else calibration/results/stereo_<a>_<b>/extrinsics.json). "
             "Always read-only.",
    )
    parser.add_argument(
        "--out", "--output", dest="output", default=None,
        help="Calibration results root (default: geometric_calibration.output_dir, else "
             f"{DEFAULT_OUTPUT_DIR}). Output always lands in "
             "<out>/stereo_<a>_<b>/cross_validation/, so this can never collide with "
             "extrinsics.json.",
    )
    parser.add_argument("--min-corners", type=int, default=DEFAULT_MIN_CORNERS,
                        help="Minimum corners per image for a view to be considered "
                             "(default: %(default)s).")
    parser.add_argument("--min-shared-corners", type=int, default=DEFAULT_MIN_SHARED_CORNERS,
                        help="Minimum corners seen by both cameras (default: %(default)s).")
    parser.add_argument("--tolerance-fraction", type=float, default=0.1,
                        help="How close (as a fraction of the square size) a pair's known "
                             "distance must be to one square to count as adjacent "
                             "(default: %(default)s).")
    parser.add_argument("--no-figure", action="store_true", help="Skip the error figure.")
    return parser.parse_args()


def load_extrinsics(path: Path) -> StereoExtrinsics:
    if not path.exists():
        raise SystemExit(
            f"No extrinsics at {path}.\n"
            "Run stereo_calibrate.py first — this script only validates an existing fit, "
            "it does not create one."
        )
    return StereoExtrinsics.load_json(path)


def board_from_extrinsics(extrinsics: StereoExtrinsics, extrinsics_path: Path) -> TargetBoard:
    if not extrinsics.board:
        raise SystemExit(
            f"{extrinsics_path} carries no board info; re-run stereo_calibrate.py "
            "(current version always records it) before cross-validating."
        )
    return TargetBoard.from_dict(extrinsics.board)


def write_report(
    pair_name: str,
    extrinsics_path: Path,
    capture_dirs: List[Path],
    board: TargetBoard,
    n_sessions_total: int,
    skipped: List[Tuple[str, str]],
    notes: List[str],
    epipolar: Dict[str, Any],
    closure: Dict[str, Any],
    corner_errors: Dict[str, Any],
) -> str:
    lines: List[str] = []
    add = lines.append
    rule = "-" * 74

    add(f"Cross-validation: {pair_name}")
    add("=" * 74)
    add("")
    add(f"  calibration checked:  {extrinsics_path}  (read-only, not modified)")
    for capture_dir in capture_dirs:
        add(f"  held-out captures:    {capture_dir}")
    add(f"  sessions usable:      {len(corner_errors['per_view'])}/{n_sessions_total}")
    add(f"  board:                {board.squares_x}x{board.squares_y} ChArUco, "
        f"square {board.square_size_m * 1000:.2f} mm")
    add("")

    add("EPIPOLAR RESIDUAL   (out-of-sample; compare with the fit's own report.txt)")
    add(rule)
    add(f"  rms / p95 / max       {epipolar['rms_px']:.4f} / {epipolar['p95_px']:.3f} / "
        f"{epipolar['max_px']:.3f} px")
    add("")

    add("TRIANGULATION CLOSURE   (rigid fit + free scale onto the known board)")
    add(rule)
    add(f"  rms / max             {closure['rms_mm']:.3f} / {closure['max_mm']:.3f} mm")
    add(f"  scale                 {closure['scale_mean']:.5f} +/- {closure['scale_std']:.5f}"
        "   (1.0 = consistent)")
    add("  On the calibration set this scale is close to 1 almost by construction. On this")
    add("  held-out set it is a genuine check — though the fitted scale here can still")
    add("  absorb a global bias; the next section is the number that cannot.")
    add("")

    add(f"ADJACENT-CORNER DISTANCE vs. KNOWN SQUARE SIZE ({corner_errors['square_size_mm']:.2f} mm)")
    add(rule)
    add(f"  pairs compared        {corner_errors['pairs_total']}")
    add(f"  mean / rms error      {corner_errors['mean_error_mm']:.4f} / "
        f"{corner_errors['rms_error_mm']:.4f} mm  "
        f"({corner_errors['mean_relative_error_pct']:.3f} / "
        f"{corner_errors['rms_relative_error_pct']:.3f} %)")
    add(f"  max abs / std         {corner_errors['max_abs_error_mm']:.4f} / "
        f"{corner_errors['std_error_mm']:.4f} mm")
    add("")
    add("  Nothing here is fitted, so a stereo calibration that reproduces the training")
    add("  board's geometry without the true metric scale shows up here even though its")
    add("  own in-sample reprojection error looked fine.")
    add("")

    if corner_errors["per_view"]:
        add("Per-session breakdown")
        add(rule)
        add(f"  {'session':<24}{'pairs':>7}{'mean mm':>10}{'rms mm':>9}{'max mm':>9}")
        for view in sorted(
            corner_errors["per_view"], key=lambda v: v["rms_error_mm"], reverse=True
        ):
            add(f"  {view['label']:<24}{view['pairs']:>7}{view['mean_error_mm']:>10.4f}"
                f"{view['rms_error_mm']:>9.4f}{view['max_abs_error_mm']:>9.4f}")
        add("")

    if skipped:
        add("Sessions without a usable stereo view")
        add(rule)
        for label, reason in skipped:
            add(f"  {label:<24} {reason}")
        add("")

    if notes:
        add("Capture notes")
        add(rule)
        for note in notes:
            add(f"  {note}")
        add("")

    if (
        corner_errors["pairs_total"] > 0
        and abs(corner_errors["mean_relative_error_pct"]) > WARN_RELATIVE_ERROR_PCT
    ):
        add("Quality warning")
        add(rule)
        add(f"  Mean relative error is {corner_errors['mean_relative_error_pct']:+.2f} %, above "
            f"the {WARN_RELATIVE_ERROR_PCT:.1f} % that a 1 % error in the printed square implies "
            "for the baseline (see the calibration/stereo.py module docstring). The fit may be "
            "overfitting stereo_captures rather than measuring the true rig geometry.")
        add("")

    return "\n".join(lines) + "\n"


def figure_corner_errors(corner_errors: Dict[str, Any], path: Path) -> None:
    per_view = corner_errors["per_view"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    axis = axes[0]
    means = [view["mean_error_mm"] for view in per_view]
    axis.axvline(0.0, color="k", lw=1.0)
    axis.hist(means, bins=min(20, max(5, len(means))), color="#2563eb", alpha=0.8)
    axis.set_xlabel("per-session mean corner-distance error [mm]")
    axis.set_ylabel("sessions")
    axis.set_title("Distribution across held-out sessions")
    axis.grid(alpha=0.3, axis="y")

    axis = axes[1]
    labels = [view["label"][-6:] for view in per_view]
    rms = [view["rms_error_mm"] for view in per_view]
    axis.bar(range(len(labels)), rms, color="#d97706")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=6)
    axis.set_ylabel("rms error [mm]")
    axis.set_title("Per-session rms corner-distance error")
    axis.grid(alpha=0.3, axis="y")

    figure.suptitle(
        f"Cross-validation: adjacent-corner distance vs "
        f"{corner_errors['square_size_mm']:.2f} mm square", fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    config = load_config().get("geometric_calibration", {}) or {}

    output_root = args.output or config.get("output_dir", DEFAULT_OUTPUT_DIR)
    pair_name = f"stereo_{args.camera_a}_{args.camera_b}"

    extrinsics_path = resolve_path(
        args.extrinsics
        or config.get(f"extrinsics_{args.camera_a}_{args.camera_b}")
        or f"{output_root}/{pair_name}/extrinsics.json"
    )
    extrinsics = load_extrinsics(extrinsics_path)
    if extrinsics.name_a != args.camera_a or extrinsics.name_b != args.camera_b:
        print(f"Note: extrinsics file names the pair {extrinsics.name_a}/{extrinsics.name_b}; "
              f"validating it as {args.camera_a}/{args.camera_b}.")
    if not np.any(extrinsics.essential):
        extrinsics.essential = _essential_from_pose(extrinsics.R, extrinsics.T)

    board = board_from_extrinsics(extrinsics, extrinsics_path)

    capture_values = (
        args.captures
        or config.get("cross_validation_captures")
        or DEFAULT_CROSS_VALIDATION_CAPTURES
    )
    if isinstance(capture_values, str):
        capture_values = [capture_values]
    capture_dirs = [resolve_path(value) for value in capture_values]

    print(f"Cross-validating {args.camera_a} -> {args.camera_b} against {extrinsics_path}")
    for capture_dir in capture_dirs:
        print(f"Held-out source: {capture_dir}/*/{{{args.camera_a},{args.camera_b}}}.jpg")

    session_pairs, notes = collect_session_pairs(capture_dirs, args.camera_a, args.camera_b)
    observations, size_a, size_b, skipped = detect_stereo_observations(
        session_pairs, board,
        min_corners=args.min_corners,
        min_shared_corners=args.min_shared_corners,
    )
    print(f"Both cameras saw the board in {len(observations)}/{len(session_pairs)} held-out "
          f"sessions ({args.camera_a} at {size_a[0]}x{size_a[1]}, "
          f"{args.camera_b} at {size_b[0]}x{size_b[1]})")
    for label, reason in skipped:
        print(f"  discarded {label}: {reason}")

    if (size_a, size_b) != (extrinsics.image_size_a, extrinsics.image_size_b):
        raise SystemExit(
            f"Held-out captures are {size_a[0]}x{size_a[1]}/{size_b[0]}x{size_b[1]} but the "
            f"extrinsics were fit at {extrinsics.image_size_a[0]}x{extrinsics.image_size_a[1]}/"
            f"{extrinsics.image_size_b[0]}x{extrinsics.image_size_b[1]}. Capture the held-out "
            "set at the same resolution as the calibration."
        )
    if not observations:
        raise SystemExit(
            "No held-out session had the board visible in both cameras; nothing to validate."
        )

    epipolar = epipolar_residuals(extrinsics, observations)
    closure = triangulation_closure(extrinsics, observations)
    corner_errors = adjacent_corner_distance_errors(
        extrinsics, observations, board, tolerance_fraction=args.tolerance_fraction
    )
    if corner_errors["pairs_total"] == 0:
        print("Warning: no grid-adjacent corner pairs found in any held-out view; "
              "--min-shared-corners may be too low, or the board/resolution do not match.")

    output_dir = resolve_path(output_root) / pair_name / "cross_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_text = write_report(
        pair_name, extrinsics_path, capture_dirs, board,
        n_sessions_total=len(session_pairs), skipped=skipped, notes=notes,
        epipolar=epipolar, closure=closure, corner_errors=corner_errors,
    )
    (output_dir / "report.txt").write_text(report_text, encoding="utf-8")

    payload = {
        "camera_a": args.camera_a,
        "camera_b": args.camera_b,
        "extrinsics_path": str(extrinsics_path),
        "capture_dirs": [str(d) for d in capture_dirs],
        "board": board.to_dict(),
        "sessions_total": len(session_pairs),
        "sessions_used": len(observations),
        "skipped": skipped,
        "notes": notes,
        "epipolar": {k: v for k, v in epipolar.items() if k != "residuals_px"},
        "triangulation_closure": {k: v for k, v in closure.items() if k != "residuals_mm"},
        "adjacent_corner_distance_errors": corner_errors,
    }
    (output_dir / "cross_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    figure_path = None
    if not args.no_figure and corner_errors["per_view"]:
        figure_path = output_dir / "corner_errors.png"
        figure_corner_errors(corner_errors, figure_path)

    print("")
    print(f"  epipolar rms:                 {epipolar['rms_px']:.4f} px")
    print(f"  triangulation closure:        {closure['rms_mm']:.3f} mm rms, "
          f"scale {closure['scale_mean']:.5f}")
    print(f"  adjacent-corner distance:     {corner_errors['mean_error_mm']:+.4f} mm mean, "
          f"{corner_errors['rms_error_mm']:.4f} mm rms "
          f"({corner_errors['rms_relative_error_pct']:.3f} % of "
          f"{corner_errors['square_size_mm']:.2f} mm square)")
    if (
        corner_errors["pairs_total"] > 0
        and abs(corner_errors["mean_relative_error_pct"]) > WARN_RELATIVE_ERROR_PCT
    ):
        print(f"  WARNING: exceeds the {WARN_RELATIVE_ERROR_PCT:.1f} % noticeable-bias threshold")
    print("")
    print(f"Saved cross-validation results to {output_dir}")
    print(f"  {output_dir / 'report.txt'}")
    print(f"  {output_dir / 'cross_validation.json'}")
    if figure_path is not None:
        print(f"  {figure_path}")
    print("")
    print(f"{extrinsics_path} was only read, never written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
