#!/usr/bin/env python3
"""Check this rig's RELATIVE stereo depth (Z-axis) precision.

A flat line-ladder plate validates lateral (X-Y) triangulation accuracy well --
but a flat, fronto-parallel target has ~zero depth variation across it by
construction, so it can never touch the depth axis, which is driven by
different error sources entirely (stereo disparity precision, baseline,
convergence) than in-plane accuracy.

This script does the depth-axis equivalent, using ``calibration/
depth_grid_target.py``'s grid of blocks at known, distinct heights. Method,
per session::

    click at least 3 spread-out reference points on the flat baseplate,
    press n -> triangulate -> fit a plane through them (this is the depth
    datum, NOT any assumed camera-to-plate standoff -- the plate's mounting
    angle to the camera is unknown and is never trusted)
    when prompted (immediately after the reference plane fits, and again
    after every block you finish), type the row,col of the block you're
    about to measure and press Enter -- then click that block's points
    (even just one; repeat clicks average together for a repeatability
    sample, not a new block) and press n: this finalizes the block
    (mean perpendicular distance from its points to the fitted plane) and
    immediately reprompts for the NEXT block's row,col. Some blocks may be
    self-occluded from one or both cameras, that is expected, not a
    failure; see the target's module docstring
    compare cell-to-cell separations against the target's known height
    differences -- this is "relative depth": it never depends on where the
    plate sits relative to the camera, only on differences between points,
    which sidesteps the camera's unknown internal optical-centre offset

Why relative, not absolute. ``measure_points.py``'s ``depth_mm`` is a single
triangulated point's raw Z in camera A's optical frame -- useful, but it is
not comparable to a physical "height" unless the camera's Z axis happens to
be aligned with it, which this rig's 17.66 deg convergence angle guarantees
it is not. Comparing DIFFERENCES between two triangulated points removes that
whole problem, the same way the line ladder's gaps are relative distances,
never an absolute plate position.

Two comparisons are reported per cell, deliberately different quantities:

``pair_measured_mm`` vs ``pair_truth_mm`` (cell-to-cell)
    The headline number, fed to the scale fit. Immune to any constant offset
    in where the fitted plane sits -- only the SEPARATION between cells
    matters, exactly like the line ladder's own pairwise gaps.

``block_measured_mm`` vs the target's ``heights_mm[row][col]`` (plane-to-block)
    Not used for the scale fit (a 0.6 mm block barely moves a multiplicative
    fit), but the most direct check, and the only one of the two that can
    expose a systematic bias in the plane fit itself.

Usage:
    python check_depth_accuracy.py --captures captures/depth_target
    python check_depth_accuracy.py --session captures/depth_target/<timestamp>
    python check_depth_accuracy.py --session captures/depth_target/<timestamp> \\
        --ref 100,200,90,205 --ref 900,200,890,205 \\
        --ref 100,900,90,905 --ref 900,900,890,905 \\
        --cell 0,0,300,400,290,405 --cell 4,4,700,600,690,605   # non-interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_cameras import load_config, resolve_path  # noqa: E402
from calibration.depth_grid_target import (  # noqa: E402
    DEFAULT_DEPTH_GRID_TARGET_CONFIG,
    DepthGridTarget,
)
from calibration.stereo import StereoExtrinsics, collect_session_pairs  # noqa: E402
from measure_points import (  # noqa: E402
    DEFAULT_BLOB_RADIUS_PX,
    DEFAULT_LOUPE_ZOOM,
    DEFAULT_MAX_WINDOW,
    annotate as annotate_points,
    measure_points as triangulate_clicks,
    parse_point,
    run_interactive,
)
from registration_io import default_extrinsics_path, undistort_pair  # noqa: E402

DEFAULT_DEPTH_TARGET_CAPTURES = "captures/depth_target"
DEFAULT_OUTPUT_SUBDIR = "depth_accuracy"


# --------------------------------------------------------------------------- #
# Geometry: one dimension up from a 3D line fit
# --------------------------------------------------------------------------- #

def fit_plane_3d(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """PCA plane fit. Returns centroid, unit normal, and RMS distance to plane.

    One dimension up from a PCA line fit: there the fitted direction is the
    LARGEST singular vector (a line's tangent). Here the normal is the
    SMALLEST singular vector -- the direction of least variance across
    points that are meant to be coplanar.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError(f"need at least 3 points to fit a plane, got {len(points)}")
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    residual = centred @ normal
    return centroid, normal, float(np.sqrt(np.mean(residual ** 2)))


def perpendicular_distance_to_plane(
    point: np.ndarray, plane_centroid: np.ndarray, plane_normal: np.ndarray,
) -> float:
    """Signed perpendicular distance from a point to a fitted plane."""
    offset = np.asarray(point, dtype=np.float64) - np.asarray(plane_centroid, dtype=np.float64)
    return float(offset @ plane_normal)


# --------------------------------------------------------------------------- #
# Scale fit
# --------------------------------------------------------------------------- #

def fit_scale(truth_mm: np.ndarray, measured_mm: np.ndarray) -> Dict[str, Any]:
    """Fit measured = a*true (and measured = a*true + b) across every distance.

    ``a`` is the answer to "am I within 1 %": it is the systematic scale error of
    the whole rig, which a mean absolute error cannot separate from random scatter.
    A significant intercept ``b`` points at a line-localisation bias instead --
    a fixed offset on every length rather than a proportional one.
    """
    truth = np.asarray(truth_mm, dtype=np.float64)
    measured = np.asarray(measured_mm, dtype=np.float64)

    scale = float(truth @ measured / (truth @ truth))
    residual = measured - scale * truth

    design = np.column_stack([truth, np.ones(len(truth))])
    (slope, intercept), *_ = np.linalg.lstsq(design, measured, rcond=None)
    affine_residual = measured - (slope * truth + intercept)

    return {
        "scale": scale,
        "scale_error_pct": (scale - 1.0) * 100.0,
        "residual_rms_mm": float(np.sqrt(np.mean(residual ** 2))),
        "affine_slope": float(slope),
        "affine_slope_error_pct": float((slope - 1.0) * 100.0),
        "affine_intercept_mm": float(intercept),
        "affine_residual_rms_mm": float(np.sqrt(np.mean(affine_residual ** 2))),
    }


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #

def measure_depth_session(
    ref_clicks_a: np.ndarray,
    ref_clicks_b: np.ndarray,
    cell_labels: Sequence[Tuple[int, int]],
    cell_clicks_a: np.ndarray,
    cell_clicks_b: np.ndarray,
    extrinsics: StereoExtrinsics,
    target: DepthGridTarget,
    ref_offsets_px: Optional[np.ndarray] = None,
    cell_offsets_px: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit the baseplate plane from the reference clicks, then measure every
    labeled block's signed perpendicular distance to it.

    ``cell_labels`` pairs 1:1 by index with ``cell_clicks_a``/``cell_clicks_b``
    -- that pairing IS the correspondence, not click order. A label repeated
    across multiple clicks is a repeatability sample of the SAME block, not a
    second block -- clicks are grouped by label before any pairwise or
    scale-fit computation runs, so a pairwise truth of 0 mm (which broke the
    scale fit before this rewrite) can no longer happen from a mislabeled
    repeat. The raw (undeduplicated) per-click labels and measured values are
    also returned, alongside the deduplicated per-block summary, so
    ``aggregate_depth_results`` can pool true click-to-click repeatability
    across sessions.

    ``ref_offsets_px``/``cell_offsets_px``, when provided (the interactive
    path), override the epipolar-offset diagnostic recomputed below from the
    already-snapped clicks with the real pre-snap value captured by
    ``run_interactive``. When ``None`` (the non-interactive ``--ref``/
    ``--cell`` path), the function falls back to its existing
    ``triangulate_clicks``-based computation, which is correct there since
    those clicks are genuinely raw.
    """
    ref_clicks_a = np.asarray(ref_clicks_a, dtype=np.float64).reshape(-1, 2)
    ref_clicks_b = np.asarray(ref_clicks_b, dtype=np.float64).reshape(-1, 2)
    cell_clicks_a = np.asarray(cell_clicks_a, dtype=np.float64).reshape(-1, 2)
    cell_clicks_b = np.asarray(cell_clicks_b, dtype=np.float64).reshape(-1, 2)

    if len(ref_clicks_a) < 3:
        raise ValueError(
            f"need at least 3 reference clicks for a plane fit, got {len(ref_clicks_a)}"
        )
    if not (len(cell_labels) == len(cell_clicks_a) == len(cell_clicks_b)):
        raise ValueError("cell_labels, cell_clicks_a, cell_clicks_b must be the same length")
    if len(cell_labels) < 1:
        raise ValueError(f"need at least 1 labeled block click, got {len(cell_labels)}")

    ref_result = triangulate_clicks(ref_clicks_a, ref_clicks_b, extrinsics)
    reference_points_mm = np.asarray(ref_result["points_mm"], dtype=np.float64)
    centroid, normal, plane_rms_m = fit_plane_3d(reference_points_mm / 1000.0)
    # Orient toward camera A's optical centre (the origin in this frame), so a
    # block protruding toward the camera reads a POSITIVE depth, matching the
    # target's own height_mm sign convention.
    if normal @ (-centroid) < 0.0:
        normal = -normal

    cell_result = triangulate_clicks(cell_clicks_a, cell_clicks_b, extrinsics)
    cell_points_mm = np.asarray(cell_result["points_mm"], dtype=np.float64)
    cell_measured_mm = np.array([
        perpendicular_distance_to_plane(point / 1000.0, centroid, normal) * 1000.0
        for point in cell_points_mm
    ])

    # Group repeat clicks on the same block into one entry: mean measured
    # height, sample count, and repeatability RMS (None for a single sample).
    unique_labels: List[Tuple[int, int]] = []
    block_measured_list: List[float] = []
    block_truth_list: List[float] = []
    block_samples_list: List[int] = []
    block_repeatability_list: List[Any] = []
    for label in cell_labels:
        if label in unique_labels:
            continue
        indices = [index for index, other in enumerate(cell_labels) if other == label]
        samples = cell_measured_mm[indices]
        unique_labels.append(label)
        block_measured_list.append(float(samples.mean()))
        block_truth_list.append(target.height_at(*label))
        block_samples_list.append(int(samples.size))
        block_repeatability_list.append(
            float(np.sqrt(np.mean((samples - samples.mean()) ** 2)))
            if samples.size > 1 else None
        )
    block_measured_mm = np.array(block_measured_list)
    block_truth_mm = np.array(block_truth_list)
    block_samples = np.array(block_samples_list)

    pairs, pair_truth_mm = target.pair_depths_mm(unique_labels)
    pair_measured_mm = block_measured_mm[pairs[:, 1]] - block_measured_mm[pairs[:, 0]]

    # Best-effort order-mismatch warning, at block granularity: all of the
    # target's heights are distinct, so the clicked blocks have a well-defined
    # true rank order to check the measured order against.
    if len(unique_labels) > 1:
        rank = np.argsort(block_truth_mm)
        ranked_measured = block_measured_mm[rank]
        order_mismatch_count = int(np.sum(np.diff(ranked_measured) < 0))
    else:
        order_mismatch_count = 0

    all_points_m = np.concatenate([reference_points_mm, cell_points_mm]) / 1000.0

    return {
        "block_labels": unique_labels,
        "block_truth_mm": block_truth_mm,
        "block_measured_mm": block_measured_mm,
        "block_samples": block_samples,
        "block_repeatability_rms_mm": block_repeatability_list,
        "raw_labels": list(cell_labels),
        "raw_measured_mm": cell_measured_mm,
        "pairs": pairs,
        "pair_truth_mm": pair_truth_mm,
        "pair_measured_mm": pair_measured_mm,
        "plane_rms_mm": plane_rms_m * 1000.0,
        "plane_point_count": len(ref_clicks_a),
        "ref_epipolar_offset_max_px": (
            float(np.max(ref_offsets_px)) if ref_offsets_px is not None
            else float(np.max(ref_result["epipolar_offset_px"]))
        ),
        "cell_epipolar_offset_max_px": (
            float(np.max(cell_offsets_px)) if cell_offsets_px is not None
            else float(np.max(cell_result["epipolar_offset_px"]))
        ),
        "order_mismatch_count": order_mismatch_count,
        "depth_mean_m": float(np.mean(all_points_m[:, 2])),
        "ref_clicks_a": ref_clicks_a,
        "ref_clicks_b": np.asarray(ref_result["clicks_b_snapped"]),
        "ref_points_mm": reference_points_mm,
        "cell_clicks_a": cell_clicks_a,
        "cell_clicks_b": np.asarray(cell_result["clicks_b_snapped"]),
        "cell_points_mm": cell_points_mm,
    }


def aggregate_depth_results(
    results: Sequence[Dict[str, Any]], target: DepthGridTarget,
) -> Dict[str, Any]:
    """Pool per-session pairwise and per-block measurements, then fit scale.

    The per-block table pools every RAW click across every session directly
    (not each session's own per-block mean), so repeatability_rms_mm
    reflects true click-to-click spread -- whether those repeat clicks
    happened within one session or were split across several, whichever
    occurred.
    """
    scored = [result for result in results if "skipped" not in result]

    pair_buckets: Dict[float, List[float]] = {}
    for result in scored:
        for truth, measured in zip(result["pair_truth_mm"], result["pair_measured_mm"]):
            pair_buckets.setdefault(round(float(truth), 6), []).append(float(measured))

    per_pair: List[Dict[str, Any]] = []
    for truth in sorted(pair_buckets):
        values = np.asarray(pair_buckets[truth], dtype=np.float64)
        mean = float(values.mean())
        per_pair.append({
            "truth_mm": truth,
            "samples": int(values.size),
            "measured_mean_mm": mean,
            "error_mm": mean - truth,
        })

    block_buckets: Dict[Tuple[int, int], List[float]] = {}
    for result in scored:
        for label, measured in zip(result["raw_labels"], result["raw_measured_mm"]):
            block_buckets.setdefault(tuple(label), []).append(float(measured))

    per_block: List[Dict[str, Any]] = []
    for label in sorted(block_buckets, key=lambda entry: target.height_at(*entry)):
        truth = target.height_at(*label)
        values = np.asarray(block_buckets[label], dtype=np.float64)
        mean = float(values.mean())
        per_block.append({
            "row": label[0], "col": label[1],
            "truth_mm": truth,
            "samples": int(values.size),
            "measured_mean_mm": mean,
            "repeatability_rms_mm": (
                float(np.sqrt(np.mean((values - mean) ** 2))) if values.size > 1 else None
            ),
            "error_mm": mean - truth,
        })

    block_errors = np.array([entry["error_mm"] for entry in per_block])
    repeatability_values = np.array([
        entry["repeatability_rms_mm"] for entry in per_block
        if entry["repeatability_rms_mm"] is not None
    ])

    has_pairs = any(len(result["pair_truth_mm"]) for result in scored)
    scale_fit = None
    if has_pairs:
        flat_truth = np.concatenate([result["pair_truth_mm"] for result in scored])
        flat_measured = np.concatenate([result["pair_measured_mm"] for result in scored])
        scale_fit = fit_scale(flat_truth, flat_measured)

    per_session = [{
        "label": result["label"],
        "blocks_measured": len(result["block_labels"]),
        "plane_rms_mm": result["plane_rms_mm"],
        "plane_point_count": result["plane_point_count"],
        "order_mismatch_count": result["order_mismatch_count"],
        "depth_mean_m": result["depth_mean_m"],
    } for result in scored]

    return {
        "sessions_scored": len(scored),
        "per_pair": per_pair,
        "per_block": per_block,
        "per_session": per_session,
        "scale_fit": scale_fit,
        "rms_error_mm": float(np.sqrt(np.mean(block_errors ** 2))) if len(block_errors) else float("nan"),
        "max_abs_error_mm": float(np.abs(block_errors).max()) if len(block_errors) else float("nan"),
        "repeatability_rms_mm": (
            float(np.sqrt(np.mean(repeatability_values ** 2))) if len(repeatability_values) else None
        ),
        "plane_rms_mean_mm": float(np.mean([result["plane_rms_mm"] for result in scored])),
        "order_mismatch_total": int(sum(result["order_mismatch_count"] for result in scored)),
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_report(
    summary: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    target: DepthGridTarget,
    context: Dict[str, Any],
    path: Path,
) -> None:
    rule = "=" * 78
    thin = "-" * 78

    fit = summary["scale_fit"]
    if fit is not None:
        headline = (
            f"RESULT: scale error {fit['scale_error_pct']:+.3f} %  "
            f"(residual {fit['residual_rms_mm']:.4f} mm rms)  --  "
            f"{len(summary['per_block'])} block(s) across {summary['sessions_scored']} "
            f"session(s)"
        )
    else:
        headline = (
            f"RESULT: no pairwise data this run (every session measured only one "
            f"distinct block) -- {len(summary['per_block'])} block(s) across "
            f"{summary['sessions_scored']} session(s)"
        )
    if summary["repeatability_rms_mm"] is not None:
        headline += f", repeatability {summary['repeatability_rms_mm']:.4f} mm rms"

    lines = [
        rule,
        "DEPTH-GRID RELATIVE ACCURACY",
        rule,
        headline,
        "",
        f"extrinsics       : {context['extrinsics']}",
        f"target           : {context['target']}",
        f"ground truth     : {target.measured_by}",
        f"uncertainty      : +/- {target.height_uncertainty_mm:.3f} mm per block",
        f"cameras          : {context['camera_a']} (A) / {context['camera_b']} (B)",
        "",
    ]

    skipped = [result for result in results if "skipped" in result]
    if skipped:
        lines.extend([thin, "SKIPPED SESSIONS", thin])
        lines.extend(f"  {result['label']}: {result['skipped']}" for result in skipped)
        lines.append("")

    lines.extend([
        thin,
        "PER BLOCK  (pooled across sessions -- accuracy vs. truth, and repeatability "
        "from repeated clicks/sessions on the same block)",
        thin,
        f"{'truth mm':>9} {'n':>3} {'measured mm':>12} {'repeat. rms mm':>15} {'err mm':>9}",
    ])
    for entry in summary["per_block"]:
        repeat = (
            f"{entry['repeatability_rms_mm']:.4f}"
            if entry["repeatability_rms_mm"] is not None else "--"
        )
        lines.append(
            f"{entry['truth_mm']:>9.3f} {entry['samples']:>3} "
            f"{entry['measured_mean_mm']:>12.3f} {repeat:>15} {entry['error_mm']:>+9.3f}"
        )
    lines.append("")

    if fit is not None:
        lines.extend([
            thin,
            "PAIRWISE SCALE FIT  (block-to-block separations, immune to any offset "
            "in where the reference plane sits)",
            thin,
            f"  measured = a * true          a = {fit['scale']:.5f}  "
            f"-> {fit['scale_error_pct']:+.3f} %",
            f"  residual after scale         {fit['residual_rms_mm']:.4f} mm rms",
            f"  measured = a * true + b      a = {fit['affine_slope']:.5f}  "
            f"({fit['affine_slope_error_pct']:+.3f} %), b = {fit['affine_intercept_mm']:+.4f} mm",
            f"  residual after affine        {fit['affine_residual_rms_mm']:.4f} mm rms",
            "",
            "  a-1 is systematic depth-scale error. A large b points at a fixed",
            "  plane-fit bias rather than a proportional one.",
            "",
        ])

    lines.extend([thin, "PER SESSION", thin])
    for result in results:
        if "skipped" in result:
            continue
        lines.append(
            f"  {result['label']}: {len(result['block_labels'])} block(s), "
            f"plane rms {result['plane_rms_mm']:.4f} mm from {result['plane_point_count']} "
            f"reference points, depth {result['depth_mean_m'] * 1000:.0f} mm"
            + (f", {result['order_mismatch_count']} order mismatch(es)"
               if result["order_mismatch_count"] else "")
        )

    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_cell(text: str) -> Tuple[int, int, float, float, float, float]:
    parts = text.replace(" ", "").split(",")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            f"--cell wants ROW,COL,AX,AY,BX,BY, got '{text}'"
        )
    try:
        row, col = int(parts[0]), int(parts[1])
        ax, ay, bx, by = (float(value) for value in parts[2:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--cell could not parse '{text}': {exc}") from exc
    return row, col, ax, ay, bx, by


class PlaneCollectionSession:
    """Drives batched multi-point plane collection via run_interactive's
    on_point/on_undo/on_advance/on_text_submit hooks.

    Click freely for a plane's worth of points; press n to close the batch
    out. The first batch (index 0) is always the reference -- needs >= 3
    points, no label, fits the datum plane the moment it's closed. Every
    batch after that is a block, identified by row,col typed through the
    in-window text-entry mode (never a fixed count, never fewer than 1
    point); its measured height is the mean of its points' perpendicular
    distances to the reference plane's ALREADY-established normal, not an
    independent per-block plane fit -- every block is part of the same
    rigid printed object as the reference corners, so it shares that one
    normal exactly. See docs/superpowers/specs/2026-08-25-depth-accuracy-
    batch-collection-design.md.
    """

    def __init__(self, target: DepthGridTarget) -> None:
        self.target = target
        self.plane: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.batches: List[Tuple[Optional[Tuple[int, int]], List[int]]] = [(None, [])]
        self.points_mm: Dict[int, np.ndarray] = {}
        # Most recently completed step's one-line summary (reference-plane
        # fit, or a finished block), shown by status_lines() -- overwritten
        # each time on_advance succeeds, never accumulated.
        self._last_result: str = ""

    def on_point(self, index: int, result: Dict[str, Any]) -> None:
        self.points_mm[index] = np.asarray(result["points_mm"][index], dtype=np.float64)
        label, indices = self.batches[-1]
        indices.append(index)

    def on_undo(self, new_count: int) -> None:
        for index in list(self.points_mm):
            if index >= new_count:
                del self.points_mm[index]
        # Drop indices >= new_count from the current batch, back to front; an
        # emptied batch that isn't the reference (index 0) is discarded
        # outright, so the "current batch" pointer falls back to whichever
        # batch actually still owns the removed point (reopening it, even if
        # it was already finalized -- no relabeling needed, the label is
        # still recorded).
        while self.batches:
            label, indices = self.batches[-1]
            indices[:] = [i for i in indices if i < new_count]
            if indices or len(self.batches) == 1:
                break
            self.batches.pop()
        if new_count == 0:
            self.plane = None

    def on_advance(self) -> Optional[str]:
        label, indices = self.batches[-1]
        if label is None:
            if len(indices) < 3:
                print(f"  need at least 3 reference points, have {len(indices)}")
                return None
            points_mm = np.array([self.points_mm[i] for i in indices])
            centroid, normal, rms_m = fit_plane_3d(points_mm / 1000.0)
            if normal @ (-centroid) < 0.0:
                normal = -normal
            self.plane = (centroid, normal)
            self._last_result = (
                f"reference plane fit, rms {rms_m * 1000.0:.4f}mm from {len(indices)} points"
            )
        else:
            if len(indices) < 1:
                print("  click at least one point before pressing n")
                return None
            centroid, normal = self.plane
            values = np.array([
                perpendicular_distance_to_plane(self.points_mm[i] / 1000.0, centroid, normal)
                * 1000.0
                for i in indices
            ])
            mean = float(values.mean())
            truth = self.target.height_at(*label)
            if values.size > 1:
                spread = float(np.sqrt(np.mean((values - mean) ** 2)))
                self._last_result = (
                    f"block {label}: truth {truth:.3f}mm, measured {mean:.3f}mm "
                    f"(delta {mean - truth:+.3f}mm, {values.size} pts, spread {spread:.4f}mm rms)"
                )
            else:
                self._last_result = (
                    f"block {label}: truth {truth:.3f}mm, measured {mean:.3f}mm "
                    f"(delta {mean - truth:+.3f}mm)"
                )
        return "cell row,col > "

    def on_text_submit(self, text: str) -> Optional[str]:
        try:
            row_text, col_text = text.split(",")
            row, col = int(row_text), int(col_text)
            if not (0 <= row < self.target.row_count and 0 <= col < self.target.col_count):
                raise ValueError
        except ValueError:
            return (f"enter as ROW,COL within 0-{self.target.row_count - 1},"
                    f"0-{self.target.col_count - 1}")
        self.batches.append(((row, col), []))
        return None

    def status_lines(self) -> List[str]:
        """The window's whole status strip while this session drives it --
        see docs/superpowers/specs/2026-09-16-depth-accuracy-status-
        simplification-design.md. Always exactly 3 lines, so the keys line
        never shifts position: current task (with the block's ground truth
        height, visible the instant its row,col batch becomes current -- no
        separate print needed), the most recently completed step's result
        (empty until the first one finishes), then the key hints.
        """
        label, indices = self.batches[-1]
        if label is None:
            head = (f"REFERENCE PLANE -- {len(indices)} point(s) clicked (need >= 3), "
                    "press n when done")
        else:
            truth = self.target.height_at(*label)
            head = (f"BLOCK {label} -- ground truth {truth:.3f}mm -- "
                    f"{len(indices)} point(s) clicked, press n to finish")
        return [
            head,
            f"last: {self._last_result}" if self._last_result else "",
            "u undo | n finish batch | q/Esc quit",
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check triangulated depth differences against a measured depth-grid "
                    "target.",
    )
    parser.add_argument(
        "--captures", nargs="+", default=None,
        help="Directories holding depth-grid capture sessions (default: "
             "geometric_calibration.depth_grid_target_captures, else "
             f"{DEFAULT_DEPTH_TARGET_CAPTURES}).",
    )
    parser.add_argument(
        "--session", default=None,
        help="Restrict to one session folder instead of every session under --captures.",
    )
    parser.add_argument("--camera-a", default="rgb_cam1")
    parser.add_argument("--camera-b", default="rgb_cam2")
    parser.add_argument("--extrinsics", default=None,
                        help="Stereo extrinsics JSON (default: geometric_calibration."
                             "extrinsics_<a>_<b> in config, else "
                             "calibration/results/stereo_<a>_<b>/extrinsics.json).")
    parser.add_argument("--target", default=None,
                        help="Depth grid target YAML (default: geometric_calibration."
                             f"depth_grid_target in config, else "
                             f"{DEFAULT_DEPTH_GRID_TARGET_CONFIG}).")
    parser.add_argument("--ref", type=parse_point, action="append", default=None,
                        metavar="AX,AY,BX,BY",
                        help="Non-interactive: one reference-corner click pair, in "
                             "undistorted full-res pixels. Repeat at least 3 times, any "
                             "order among themselves.")
    parser.add_argument("--cell", type=parse_cell, action="append", default=None,
                        metavar="ROW,COL,AX,AY,BX,BY",
                        help="Non-interactive: one block click pair, labeled by its "
                             "(row, col) in the target's heights_mm grid. Repeat for every "
                             "block visible this session (a block may repeat for a "
                             "repeatability sample); occluded blocks are simply omitted, "
                             "not required.")
    parser.add_argument("--zoom", type=int, default=DEFAULT_LOUPE_ZOOM,
                        help="Initial loupe magnification (default: %(default)s).")
    parser.add_argument("--window", type=int, nargs=2, default=list(DEFAULT_MAX_WINDOW),
                        metavar=("W", "H"),
                        help="Window size in pixels (default: %(default)s).")
    parser.add_argument("--blob-radius", type=int, default=DEFAULT_BLOB_RADIUS_PX,
                        help="Search radius when snapping a click to a dot's centroid, "
                             "in full-resolution pixels (default: %(default)s).")
    parser.add_argument("--blob-snap", action="store_true",
                        help="Move each click onto the intensity centroid of the marker "
                             "underneath it. Off by default so a click lands exactly where "
                             "you put it.")
    parser.add_argument("--out", "--output", dest="output", default=None,
                        help="Output directory (default: geometric_calibration.output_dir "
                             f"in config) / {DEFAULT_OUTPUT_SUBDIR}.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    geo_config = load_config().get("geometric_calibration", {}) or {}
    reg_config = load_config().get("registration", {}) or {}
    depth_range = reg_config.get("depth_range", [0.11, 0.21])
    depth_range = (float(depth_range[0]), float(depth_range[1]))

    extrinsics_path = resolve_path(
        args.extrinsics or default_extrinsics_path(args.camera_a, args.camera_b)
    )
    if not extrinsics_path.exists():
        raise SystemExit(f"No stereo extrinsics at {extrinsics_path}.")
    extrinsics = StereoExtrinsics.load_json(extrinsics_path)

    target_path = resolve_path(
        args.target or geo_config.get("depth_grid_target", DEFAULT_DEPTH_GRID_TARGET_CONFIG)
    )
    target = DepthGridTarget.from_yaml(target_path)
    all_heights = [height for row in target.heights_mm for height in row]
    print(f"depth grid: {target.row_count}x{target.col_count}, pitch {target.pitch_mm:.1f} mm, "
          f"heights {min(all_heights):.1f}-{max(all_heights):.1f} mm, "
          "at least 3 reference corners")
    if "NOT verified" in target.measured_by:
        print(f"warning: {target_path.name} still carries unverified ground truth "
              f"({target.measured_by}). Every number below is only as good as that.")

    if args.ref or args.cell:
        if not args.session:
            raise SystemExit("--ref/--cell require --session (they supply coordinates for "
                             "exactly one session).")
        if args.captures:
            raise SystemExit("--ref/--cell cannot be combined with --captures.")
        if not args.ref or len(args.ref) < 3:
            raise SystemExit(
                f"Need at least 3 --ref clicks for a plane fit, got "
                f"{len(args.ref) if args.ref else 0}."
            )
        if not args.cell:
            raise SystemExit("Need at least 1 --cell click to measure a block's depth.")
        for row, col, *_ in args.cell:
            if not (0 <= row < target.row_count and 0 <= col < target.col_count):
                raise SystemExit(
                    f"--cell {row},{col} is out of range for a {target.row_count}x"
                    f"{target.col_count} grid."
                )

    if args.session:
        session_dir = resolve_path(args.session)
        paths = [session_dir / f"{camera}.jpg" for camera in (args.camera_a, args.camera_b)]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            raise SystemExit(f"{session_dir} is missing {', '.join(missing)}.")
        session_pairs, notes = [(session_dir.name, paths[0], paths[1])], []
    else:
        capture_values = args.captures or geo_config.get("depth_grid_target_captures") \
            or DEFAULT_DEPTH_TARGET_CAPTURES
        if isinstance(capture_values, str):
            capture_values = [capture_values]
        capture_dirs = [resolve_path(value) for value in capture_values]
        session_pairs, notes = collect_session_pairs(capture_dirs, args.camera_a, args.camera_b)
    for note in notes:
        print(note)
    if not session_pairs:
        raise SystemExit(f"No {args.camera_a}/{args.camera_b} sessions found.")

    output_dir = resolve_path(
        args.output or geo_config.get("output_dir", "calibration/results")
    ) / DEFAULT_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for label, path_a, path_b in session_pairs:
        raw_a, raw_b = cv2.imread(str(path_a)), cv2.imread(str(path_b))
        if raw_a is None or raw_b is None:
            results.append({"label": label, "skipped": "failed to decode one of the images"})
            print(f"{label}: skipped: failed to decode")
            continue
        image_a, image_b = undistort_pair(raw_a, raw_b, extrinsics)

        if args.ref:
            ref_clicks_a = np.array([[point[0], point[1]] for point in args.ref])
            ref_clicks_b = np.array([[point[2], point[3]] for point in args.ref])
            cell_labels = [(cell[0], cell[1]) for cell in args.cell]
            cell_clicks_a = np.array([[cell[2], cell[3]] for cell in args.cell])
            cell_clicks_b = np.array([[cell[4], cell[5]] for cell in args.cell])
            ref_offsets_px = None
            cell_offsets_px = None
        else:
            print(f"{label}: click reference points on the flat baseplate (at least 3, "
                  "spread out) then press n -- the plane fit and its rms appear in the "
                  "window's status bar, and you'll be prompted for a block's row,col "
                  "right away. Type it, Enter, and its ground truth height appears on "
                  "screen immediately -- double check it before clicking. Click that "
                  "block's points (as many as you like, even just one) and press n: the "
                  "measured-vs-truth result appears in the status bar and it immediately "
                  "prompts for the next block's row,col. Repeat for every block you can "
                  "see. Press q/Esc to finish.")
            session = PlaneCollectionSession(target)
            raw_result = run_interactive(
                image_a, image_b, extrinsics, depth_range,
                max(2, args.zoom), (int(args.window[0]), int(args.window[1])),
                max(3, args.blob_radius) if args.blob_snap else 0,
                float(reg_config.get("default_depth", 0.168)),
                on_point=session.on_point, on_undo=session.on_undo,
                on_advance=session.on_advance, on_text_submit=session.on_text_submit,
                on_status=session.status_lines,
            )
            if not raw_result.get("clicks_a", []):
                results.append({"label": label, "skipped": "no points clicked"})
                print(f"{label}: skipped: no points clicked")
                continue
            labeled_batches = [
                (batch_label, indices) for batch_label, indices in session.batches
                if batch_label is not None and indices
            ]
            if len(session.batches[0][1]) < 3 or not labeled_batches:
                results.append({
                    "label": label,
                    "skipped": "fewer than 3 reference points or no labeled block batches",
                })
                print(f"{label}: skipped: not enough points clicked")
                continue
            ref_indices = session.batches[0][1]
            ref_clicks_a = np.array([raw_result["clicks_a"][i] for i in ref_indices])
            ref_clicks_b = np.array([raw_result["clicks_b_snapped"][i] for i in ref_indices])
            ref_offsets_px = np.array(
                [raw_result["epipolar_offset_px"][i] for i in ref_indices]
            )
            cell_labels = [
                batch_label for batch_label, indices in labeled_batches for _ in indices
            ]
            cell_clicks_a = np.array([
                raw_result["clicks_a"][i] for _, indices in labeled_batches for i in indices
            ])
            cell_clicks_b = np.array([
                raw_result["clicks_b_snapped"][i]
                for _, indices in labeled_batches for i in indices
            ])
            cell_offsets_px = np.array([
                raw_result["epipolar_offset_px"][i]
                for _, indices in labeled_batches for i in indices
            ])

        try:
            measured = measure_depth_session(
                ref_clicks_a, ref_clicks_b, cell_labels, cell_clicks_a, cell_clicks_b,
                extrinsics, target,
                ref_offsets_px=ref_offsets_px, cell_offsets_px=cell_offsets_px,
            )
        except ValueError as exc:
            results.append({"label": label, "skipped": str(exc)})
            print(f"{label}: skipped: {exc}")
            continue

        measured["label"] = label
        results.append(measured)

        combined = {
            "clicks_a": np.vstack([measured["ref_clicks_a"], measured["cell_clicks_a"]]).tolist(),
            "clicks_b_snapped": np.vstack(
                [measured["ref_clicks_b"], measured["cell_clicks_b"]]
            ).tolist(),
            "points_mm": np.vstack([measured["ref_points_mm"], measured["cell_points_mm"]]).tolist(),
        }
        cv2.imwrite(
            str(output_dir / f"{label}_correspondences.jpg"),
            annotate_points(image_a, image_b, combined,
                            (int(args.window[0]), int(args.window[1]))),
        )

        if len(measured["pair_truth_mm"]):
            session_scale = fit_scale(measured["pair_truth_mm"], measured["pair_measured_mm"])
            scale_text = f"scale {session_scale['scale_error_pct']:+.3f} %"
        else:
            scale_text = "scale n/a (1 block)"
        print(f"{label}: {scale_text}, blocks {len(measured['block_labels'])}, "
              f"plane rms {measured['plane_rms_mm']:.4f} mm, "
              f"depth {measured['depth_mean_m'] * 1000:.0f} mm")

    if not any("skipped" not in result for result in results):
        raise SystemExit("No session could be measured; see the skip reasons above.")

    summary = aggregate_depth_results(results, target)
    context = {
        "extrinsics": str(extrinsics_path),
        "target": str(target_path),
        "camera_a": args.camera_a,
        "camera_b": args.camera_b,
    }

    write_report(summary, results, target, context, output_dir / "report.txt")

    payload = {
        "extrinsics": str(extrinsics_path),
        "target": target.to_dict(),
        "target_path": str(target_path),
        "summary": summary,
        "sessions": [
            {"label": result["label"], "skipped": result["skipped"]}
            if "skipped" in result else
            {
                "label": result["label"],
                "block_labels": result["block_labels"],
                "block_truth_mm": result["block_truth_mm"].tolist(),
                "block_measured_mm": result["block_measured_mm"].tolist(),
                "block_samples": result["block_samples"].tolist(),
                "block_repeatability_rms_mm": result["block_repeatability_rms_mm"],
                "pairs": result["pairs"].tolist(),
                "pair_truth_mm": result["pair_truth_mm"].tolist(),
                "pair_measured_mm": result["pair_measured_mm"].tolist(),
                "plane_rms_mm": result["plane_rms_mm"],
                "plane_point_count": result["plane_point_count"],
                "ref_epipolar_offset_max_px": result["ref_epipolar_offset_max_px"],
                "cell_epipolar_offset_max_px": result["cell_epipolar_offset_max_px"],
                "order_mismatch_count": result["order_mismatch_count"],
                "depth_mean_m": result["depth_mean_m"],
            }
            for result in results
        ],
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    fit = summary["scale_fit"]
    print()
    if fit is not None:
        print(f"scale error   {fit['scale_error_pct']:+.3f} %  "
              f"(a = {fit['scale']:.5f}, residual {fit['residual_rms_mm']:.4f} mm rms)")
    else:
        print("scale error   n/a (no pairwise data -- every session measured only one block)")
    print(f"rms error     {summary['rms_error_mm']:.4f} mm  (max {summary['max_abs_error_mm']:.4f} mm)")
    if summary["repeatability_rms_mm"] is not None:
        print(f"repeatability {summary['repeatability_rms_mm']:.4f} mm rms")
    print(f"Saved report.txt / result.json / *_correspondences.jpg to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
