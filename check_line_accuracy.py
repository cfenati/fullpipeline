#!/usr/bin/env python3
"""Check that this rig converts pixels to millimetres correctly.

Every metric length the pipeline reports is scaled by one number: the ChArUco
``square_size_m`` that set the baseline during ``stereo_calibrate.py``. A 1 %
error in that printed square is a 1 % error in every triangulated length.

``cross_validate_stereo.py`` looks like it checks this, but it holds out board
*poses* while reusing the same board and the same nominal square, so a wrong
square cancels exactly and it reports zero error. It grades repeatability, not
scale. This script grades scale, using a separately fabricated line ladder whose
gaps were measured with a different instrument.

Method, per session::

    estimate the background, subtract -> a darkness map of the marks alone
    find the ladder's angle and each line's offset by projection (Radon-style)
    refine each line to the darkness-weighted centre of the mark, per scanline
    sample points along line i in A
    epipolar line in B  ->  intersect with line i in B  ->  correspondence
    triangulate (DLT)   ->  3D points in camera A, metres
    fit a 3D line per ladder line, take perpendicular distances between them

Two details carry most of the accuracy.

The measurement is centre-to-centre, taken as the darkness-weighted centroid of
the mark's profile. Laser kerf, marking width, exposure and blooming all widen a
mark symmetrically, and the centroid of a symmetric profile does not move, so
none of them bias the result.

Distances come from fitting a 3D line per ladder line and taking the
perpendicular offset between the fits. Two points on parallel lines at different
heights are sqrt(gap^2 + dheight^2) apart, not gap apart, so point-to-point
distances between independently chosen samples read high -- by +629 % on a 3 mm
gap in the synthetic check, which is not a subtle effect.

Usage:
    python check_line_accuracy.py --captures captures/line_target
    python check_line_accuracy.py --session captures/line_target/<timestamp>
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
from calibration.line_target import (  # noqa: E402
    DEFAULT_LINE_TARGET_CONFIG,
    LineLadderTarget,
)
from calibration.opencv_calibrate import scaled_px  # noqa: E402
from calibration.stereo import StereoExtrinsics, collect_session_pairs  # noqa: E402
from registration_io import default_extrinsics_path, undistort_pair  # noqa: E402
from triangulate import (  # noqa: E402
    epipolar_lines,
    fundamental_for_undistorted as _fundamental,
    triangulate_points,
)

DEFAULT_LINE_TARGET_CAPTURES = "captures/line_target"
DEFAULT_OUTPUT_SUBDIR = "line_accuracy"

# Pixel budgets are quoted in 1280-wide reference pixels and scaled to the real
# frame with opencv_calibrate.scaled_px, the same way calibration error budgets
# are. Hard-coded pixel counts do not survive the jump to a 4656 px frame.

# Structuring element used to erase the marks and leave the background. Must be
# wider than a mark: a closing takes the local maximum, so any window wider than
# the mark still contains plate. Generous is safe, too small is not.
DEFAULT_BACKGROUND_KERNEL_REF = 40.0
# Half-width of the perpendicular profile whose centroid locates a mark. Must
# comfortably contain the whole mark plus some background on each side.
DEFAULT_REFINE_HALF_WIDTH_REF = 25.0
# Spacing of scanlines along a line during refinement.
DEFAULT_REFINE_STEP_REF = 2.0
# Smoothing applied to the projection histogram before peak picking.
DEFAULT_PEAK_SMOOTHING_REF = 3.0

# How far either side of the nominal orientation to search for the ladder's angle.
DEFAULT_ANGLE_SEARCH_DEG = 12.0
DEFAULT_ANGLE_STEPS = 97

DEFAULT_SAMPLES_PER_LINE = 25
# Below this crossing angle the epipolar/target-line intersection slides along the
# target line under sub-pixel noise. Vertical lines on this near-pure-X baseline
# cross at ~90 deg; anything near 0 means the target is oriented wrong.
DEFAULT_MIN_INTERSECTION_ANGLE_DEG = 20.0

# A pixel counts as part of a mark once it is this fraction of the way from the
# background to the darkest point in the frame.
MARK_THRESHOLD_FRACTION = 0.35
# A scanline contributes a centroid only if it carries this fraction of the
# typical scanline's darkness, which is what keeps a line's ends from dragging
# the fit outward once the mark has run out.
SCANLINE_SIGNAL_FRACTION = 0.4
# Percentile of scanline darkness used as "what a scanline on the mark looks
# like". A median would be a background statistic, since the mark covers only a
# small part of a wide frame.
SCANLINE_REFERENCE_PERCENTILE = 90.0
# MAD multiplier for rejecting centroids that sit off the fitted line.
ROBUST_SIGMA = 3.0
# Detected line offsets must match the target's known gap pattern to within this
# fraction of its narrowest gap, or the session is skipped rather than measured.
PATTERN_TOLERANCE_FRACTION = 0.2
# In an open ladder a peak this strong relative to the strongest counts as a mark.
OPEN_PEAK_FRACTION = 0.25
# An open ladder's gaps must agree with their median to within this fraction,
# which is what proves the visible marks are consecutive.
UNIFORM_TOLERANCE_FRACTION = 0.35
# An extra mark-like peak this strong relative to the weakest accepted one means
# the frame holds something that could be a mark, so which marks are the ladder is
# no longer certain and the session is refused.
EXTRA_PEAK_FRACTION = 0.5
# Hard ceiling on the refinement half-width as a fraction of the narrowest gap
# actually seen, so the window can never reach the neighbouring mark.
REFINE_SPACING_FRACTION = 0.4
# Fraction of each line's usable extent trimmed off both ends before sampling.
EXTENT_MARGIN = 0.05

# Per-mark colours for the correspondence figure. Distinct even on a dark plate.
LINE_COLORS = (
    (40, 180, 40),
    (0, 140, 255),
    (255, 128, 0),
    (255, 0, 200),
    (0, 220, 220),
    (80, 80, 255),
    (180, 255, 0),
    (255, 200, 0),
)
# How many epipolar lines to draw per mark. All sample dots are still shown;
# drawing every epipolar as well would bury the intersections in a stack of
# near-horizontal lines.
EPIPOLAR_DRAW_COUNT = 5


# --------------------------------------------------------------------------- #
# Line geometry helpers
# --------------------------------------------------------------------------- #

def line_from_angle(theta: float, offset: float) -> np.ndarray:
    """Line with unit normal at ``theta`` and perpendicular offset ``offset``."""
    return np.array([np.cos(theta), np.sin(theta), -offset], dtype=np.float64)


def base_point(line_abc: np.ndarray) -> np.ndarray:
    """The point on the line closest to the image origin."""
    a, b, c = line_abc
    return np.array([-a * c, -b * c], dtype=np.float64)


def line_direction(line_abc: np.ndarray) -> np.ndarray:
    a, b, _ = line_abc
    return np.array([-b, a], dtype=np.float64)


def fit_line_abc(points: np.ndarray, orientation: str) -> np.ndarray:
    """Total-least-squares line as (a, b, c) with a^2 + b^2 = 1, so a*x + b*y + c = 0.

    The normal's sign is pinned by the orientation (a > 0 for vertical lines,
    b > 0 for horizontal) so every line in a ladder is described consistently and
    the lines sort left-to-right rather than in an arbitrary order.
    """
    fitted = cv2.fitLine(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
        cv2.DIST_L2, 0, 0.01, 0.01,
    ).ravel()
    vx, vy, x0, y0 = (float(value) for value in fitted)
    a, b = -vy, vx
    c = -(a * x0 + b * y0)
    pin = a if orientation == "vertical" else b
    if pin < 0:
        a, b, c = -a, -b, -c
    return np.array([a, b, c], dtype=np.float64)


def line_image_extent(line_abc: np.ndarray, shape: Tuple[int, int]) -> Tuple[float, float]:
    """Parameter range over which the line stays inside the image."""
    height, width = shape[:2]
    base = base_point(line_abc)
    direction = line_direction(line_abc)

    candidates: List[float] = []
    for axis, bound in ((0, 0.0), (0, width - 1.0), (1, 0.0), (1, height - 1.0)):
        if abs(direction[axis]) > 1e-9:
            candidates.append((bound - base[axis]) / direction[axis])
    if len(candidates) < 2:
        return 0.0, 0.0

    candidates.sort()
    best = (0.0, 0.0)
    for lower, upper in zip(candidates, candidates[1:]):
        point = base + 0.5 * (lower + upper) * direction
        inside = -0.5 <= point[0] <= width - 0.5 and -0.5 <= point[1] <= height - 0.5
        if inside and (upper - lower) > (best[1] - best[0]):
            best = (lower, upper)
    return best


def sample_along_line(
    line_abc: np.ndarray, t_min: float, t_max: float, count: int,
) -> np.ndarray:
    """``count`` evenly spaced points on the line, trimmed in from both ends."""
    margin = EXTENT_MARGIN * (t_max - t_min)
    t = np.linspace(t_min + margin, t_max - margin, count)
    return base_point(line_abc) + t[:, None] * line_direction(line_abc)


def sample_bilinear(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear lookup at sub-pixel coordinates in a float image, edge-clamped."""
    height, width = image.shape[:2]
    x = np.clip(points[:, 0], 0.0, width - 1.001)
    y = np.clip(points[:, 1], 0.0, height - 1.001)
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    fx, fy = x - x0, y - y0
    return (
        image[y0, x0] * (1 - fx) * (1 - fy)
        + image[y0, x0 + 1] * fx * (1 - fy)
        + image[y0 + 1, x0] * (1 - fx) * fy
        + image[y0 + 1, x0 + 1] * fx * fy
    )


# --------------------------------------------------------------------------- #
# Line detection
# --------------------------------------------------------------------------- #

def darkness_map(gray: np.ndarray, orientation: str, kernel_px: float) -> np.ndarray:
    """How far below the local background each pixel sits: bright plate -> 0, mark -> high.

    Background comes from a morphological closing with a structuring element laid
    ACROSS the marks (horizontal for vertical lines). A closing takes the local
    maximum, so a window wider than a mark always contains plate and the marks are
    erased while vignetting and uneven illumination survive. Subtracting leaves the
    marks alone, already flat-fielded.

    Working from this rather than from Canny edges is what makes detection robust:
    every pixel of every mark contributes, instead of depending on whether a Hough
    accumulator happened to link an edge into a long enough segment.
    """
    size = max(3, int(round(kernel_px)) | 1)
    shape = (size, 1) if orientation == "vertical" else (1, size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, shape)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return np.clip(background.astype(np.float32) - gray.astype(np.float32), 0.0, None)


def mark_pixels(darkness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Coordinates and weights of the pixels that belong to a mark."""
    threshold = MARK_THRESHOLD_FRACTION * float(darkness.max())
    if threshold <= 0.0:
        return np.zeros((0, 2)), np.zeros(0)
    rows, cols = np.nonzero(darkness > threshold)
    return (
        np.column_stack([cols, rows]).astype(np.float64),
        darkness[rows, cols].astype(np.float64),
    )


def projection_profile(
    points: np.ndarray, weights: np.ndarray, theta: float, bin_width: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Darkness projected onto the normal at ``theta``: a 1D profile with one peak per mark."""
    normal = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    offsets = points @ normal
    low = float(offsets.min())
    bins = max(2, int(np.ceil((float(offsets.max()) - low) / bin_width)) + 1)
    profile = np.bincount(
        np.clip(((offsets - low) / bin_width).astype(np.intp), 0, bins - 1),
        weights=weights, minlength=bins,
    )
    return profile, low


def estimate_normal_angle(
    points: np.ndarray, weights: np.ndarray, orientation: str,
    search_deg: float, steps: int,
) -> float:
    """Angle of the ladder's normal, found by sharpening the projection.

    Projected along the true line direction every mark collapses to a narrow
    spike; projected at any other angle the marks smear into each other. Summing
    the squared profile measures exactly that concentration, so the maximiser is
    the ladder's angle. This is a one-dimensional Radon transform, and unlike a
    Hough angle it degrades gracefully rather than discretely.
    """
    nominal = 0.0 if orientation == "vertical" else np.pi / 2.0
    candidates = nominal + np.radians(np.linspace(-search_deg, search_deg, steps))
    sharpness = [float(np.sum(projection_profile(points, weights, theta)[0] ** 2))
                 for theta in candidates]
    return float(candidates[int(np.argmax(sharpness))])


def find_line_offsets(
    points: np.ndarray, weights: np.ndarray, theta: float,
    target: LineLadderTarget, smoothing_px: float,
) -> np.ndarray:
    """Perpendicular offset of every mark found, validated against the ladder.

    Fixed ladder (``gaps_mm``): take exactly ``line_count`` peaks and require them
    to be an affine image of ``positions_mm``. Unequal gaps fit only one way, so a
    missed mark breaks the fit and is caught.

    Open ladder (``spacing_mm``): take every mark that stands out and require the
    spacing to be uniform. That check is what proves the run is *consecutive*,
    which is all the measurement needs -- distance from mark i to mark j is
    ``(j - i) * spacing_mm`` regardless of which part of the plate is in frame.
    """
    profile, low = projection_profile(points, weights, theta)

    radius = max(1, int(round(smoothing_px)))
    kernel = cv2.getGaussianKernel(2 * radius + 1, smoothing_px).ravel()
    smoothed = np.convolve(profile, kernel, mode="same")

    interior = smoothed[1:-1]
    is_peak = (interior >= smoothed[:-2]) & (interior > smoothed[2:])
    peaks = np.nonzero(is_peak)[0] + 1
    if not len(peaks):
        raise ValueError("no marks found in the projection profile")

    if target.open_ended:
        offsets = _open_ended_offsets(peaks, smoothed) + low
        _require_uniform(offsets, target)
        return offsets

    expected = target.line_count
    order = peaks[np.argsort(smoothed[peaks])[::-1]]
    span_px = float(peaks.max() - peaks.min())
    minimum_separation = 0.5 * span_px * (min(target.gaps_mm) / target.span_mm())

    chosen: List[int] = []
    for candidate in order:
        if all(abs(candidate - kept) >= minimum_separation for kept in chosen):
            chosen.append(int(candidate))
        if len(chosen) == expected + 1:
            break
    if len(chosen) < expected:
        raise ValueError(
            f"found {len(chosen)} distinguishable marks, expected {expected}"
        )

    # One candidate too many. Taking the strongest `expected` would quietly pick a
    # set, and on a uniform ladder the pattern check below cannot tell a wrong set
    # from a right one, so refuse instead. A scratch, a plate edge or a reflection
    # is far more likely than a genuine extra mark.
    if len(chosen) > expected:
        strengths = smoothed[np.array(chosen)]
        if strengths.min() >= EXTRA_PEAK_FRACTION * np.sort(strengths)[1]:
            raise ValueError(
                f"found at least {expected + 1} mark-like features, expected {expected} "
                "-- something other than the ladder is in frame, or the plate is "
                "clipped and a real mark was replaced by an artefact"
            )
        chosen = chosen[:expected]

    offsets = np.sort(np.array(chosen, dtype=np.float64)) + low

    truth = target.positions_mm()
    design = np.column_stack([truth, np.ones(len(truth))])
    solution, *_ = np.linalg.lstsq(design, offsets, rcond=None)
    residual = offsets - design @ solution
    tolerance = (PATTERN_TOLERANCE_FRACTION * min(target.gaps_mm)
                 * abs(float(solution[0])))
    if np.abs(residual).max() > tolerance:
        raise ValueError(
            f"detected marks do not match the ladder pattern "
            f"(worst deviation {np.abs(residual).max():.1f} px, tolerance {tolerance:.1f} px) "
            "-- a mark was probably missed or something else in the scene was picked up"
        )
    return offsets


def _open_ended_offsets(peaks: np.ndarray, smoothed: np.ndarray) -> np.ndarray:
    """Every mark in an open ladder, with the spacing bootstrapped from the peaks."""
    strong = peaks[smoothed[peaks] > OPEN_PEAK_FRACTION * float(smoothed[peaks].max())]
    if len(strong) < 3:
        raise ValueError(f"found {len(strong)} marks, need at least 3 to measure")

    # Bootstrap the spacing from the peaks themselves, then re-select with a
    # separation floor so a doubled peak on one mark cannot masquerade as two.
    spacing = float(np.median(np.diff(np.sort(strong))))
    order = strong[np.argsort(smoothed[strong])[::-1]]
    chosen: List[int] = []
    for candidate in order:
        if all(abs(candidate - kept) >= 0.5 * spacing for kept in chosen):
            chosen.append(int(candidate))
    return np.sort(np.array(chosen, dtype=np.float64))


def _require_uniform(offsets: np.ndarray, target: LineLadderTarget) -> None:
    """Reject a run of marks that is not evenly spaced.

    Uniformity is the whole warrant for treating the visible marks as consecutive.
    A gap of roughly twice the rest means a mark was missed, and every distance
    past that point would be short by one spacing without looking wrong.
    """
    gaps = np.diff(offsets)
    median = float(np.median(gaps))
    worst = float(np.abs(gaps - median).max())
    if worst > UNIFORM_TOLERANCE_FRACTION * median:
        raise ValueError(
            f"mark spacing is not uniform (worst gap deviates {worst / median * 100:.0f} % "
            f"from the median {median:.1f} px) -- a mark was probably missed, so the "
            "visible marks cannot be assumed consecutive"
        )


def longest_run(mask: np.ndarray) -> np.ndarray:
    """Keep only the longest contiguous True run, since a mark is one stroke."""
    if not mask.any():
        return mask
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    best = int(np.argmax(ends - starts))
    out = np.zeros_like(mask)
    out[starts[best]:ends[best]] = True
    return out


def refine_line(
    darkness: np.ndarray, line_abc: np.ndarray, orientation: str,
    half_width_px: float, step_px: float, mark_floor: float,
) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """Re-fit a line to the darkness-weighted centre of its mark, scanline by scanline.

    Walks along the approximate line and, at each step, takes the centroid of the
    mark's profile across a perpendicular window. That centroid is the mark's
    centre to sub-pixel precision and is immune to mark width, since a symmetric
    profile's centroid does not move when the profile widens.

    Returns the refined line, the parameter range over which the mark actually
    exists, and the mark's mean width in pixels.
    """
    direction = line_direction(line_abc)
    normal = np.array([line_abc[0], line_abc[1]], dtype=np.float64)
    base = base_point(line_abc)

    t_min, t_max = line_image_extent(line_abc, darkness.shape)
    if t_max - t_min < 2.0 * step_px:
        raise ValueError("line falls outside the image")

    steps = np.arange(t_min, t_max, step_px)
    offsets = np.arange(-half_width_px, half_width_px + 1.0)
    centres = base + steps[:, None] * direction
    grid = centres[:, None, :] + offsets[None, :, None] * normal
    profile = sample_bilinear(darkness, grid.reshape(-1, 2)).reshape(len(steps), -1)

    # Only genuinely mark-dark pixels may vote. Background texture -- cloth
    # wrinkles, shadows, the edge of the plate -- otherwise contributes a little
    # darkness everywhere, which drags the centroid and, worse, makes every
    # scanline look like it carries signal.
    profile = np.where(profile > mark_floor, profile, 0.0)

    totals = profile.sum(axis=1)
    if not np.any(totals > 0.0):
        raise ValueError("no darkness along the line")
    # Judge a scanline against the STRONGEST scanlines, not the median of the
    # frame: the mark covers a small part of a wide image, so a median taken over
    # everything is a background statistic and would accept the whole frame.
    reference = float(np.percentile(totals, SCANLINE_REFERENCE_PERCENTILE))
    keep = totals > SCANLINE_SIGNAL_FRACTION * reference
    # A mark is one continuous stroke, so its scanlines are contiguous. Keeping
    # only the longest run discards a neighbouring mark clipped by the window and
    # any isolated speck that happened to clear the threshold -- both of which
    # otherwise sit far from the mark and lever the fitted line off its true angle.
    keep = longest_run(keep)
    if keep.sum() < 2:
        raise ValueError("mark too faint to locate")

    centroid = (profile[keep] @ offsets) / totals[keep]
    located = centres[keep] + centroid[:, None] * normal

    # One robust pass: a few centroids can still be pulled off by a smudge or the
    # edge of the plate, and a least-squares line fit has no defence against them.
    line = fit_line_abc(located, orientation)
    residual = np.abs(np.column_stack([located, np.ones(len(located))]) @ line)
    spread = float(np.median(np.abs(residual - np.median(residual))))
    inlier = residual <= np.median(residual) + ROBUST_SIGMA * 1.4826 * max(spread, 1e-6)
    if inlier.sum() >= max(2, 0.5 * len(located)):
        located, keep_index = located[inlier], np.nonzero(keep)[0][inlier]
    else:
        keep_index = np.nonzero(keep)[0]

    peak = profile[keep_index].max(axis=1)
    width = float(np.mean(totals[keep_index] / np.maximum(peak, 1e-9)))

    return (
        fit_line_abc(located, orientation),
        (float(steps[keep_index].min()), float(steps[keep_index].max())),
        width,
    )


def detect_ladder(
    gray: np.ndarray, target: LineLadderTarget, params: Dict[str, Any],
) -> Tuple[List[np.ndarray], List[Tuple[float, float]], List[float]]:
    """Detect every line of the ladder in one view.

    Returns one line per ladder line (ordered across the frame), the parameter
    extent over which its mark exists, and the mark's width in pixels.
    """
    darkness = darkness_map(gray, target.orientation, params["background_kernel_px"])
    mark_floor = MARK_THRESHOLD_FRACTION * float(darkness.max())
    points, weights = mark_pixels(darkness)
    if len(points) < 100:
        raise ValueError("no marks stand out from the background")

    theta = estimate_normal_angle(
        points, weights, target.orientation,
        params["angle_search_deg"], params["angle_steps"],
    )
    offsets = find_line_offsets(
        points, weights, theta, target, params["peak_smoothing_px"],
    )

    # Never let the refinement window reach the neighbouring mark: its darkness
    # would pull the centroid and bias the very distance being measured. On a
    # tightly spaced uniform ladder the configured half-width can easily exceed
    # half a gap, so clamp to what the marks actually are in this frame.
    spacing_px = float(np.diff(offsets).min())
    half_width = min(params["refine_half_width_px"], REFINE_SPACING_FRACTION * spacing_px)

    lines, extents, widths = [], [], []
    for offset in offsets:
        coarse = line_from_angle(theta, offset)
        if params["refine"]:
            line, extent, width = refine_line(
                darkness, coarse, target.orientation,
                half_width, params["refine_step_px"], mark_floor,
            )
        else:
            line, extent, width = coarse, line_image_extent(coarse, darkness.shape), 0.0
        lines.append(line)
        extents.append(extent)
        widths.append(width)
    return lines, extents, widths


# --------------------------------------------------------------------------- #
# Stereo geometry
# --------------------------------------------------------------------------- #

def fundamental_for_undistorted(extrinsics: StereoExtrinsics) -> np.ndarray:
    """``triangulate.fundamental_for_undistorted`` for this pair's calibration."""
    return _fundamental(
        extrinsics.camera_matrix_a, extrinsics.camera_matrix_b, extrinsics.essential,
    )


def intersect_lines(line_one: np.ndarray, line_two: np.ndarray) -> np.ndarray:
    """Intersection of two homogeneous 2D lines, as a pixel (x, y)."""
    point = np.cross(np.asarray(line_one, dtype=np.float64),
                     np.asarray(line_two, dtype=np.float64))
    if abs(point[2]) < 1e-12:
        raise ValueError("lines are parallel; no intersection")
    return point[:2] / point[2]


def intersection_angles_deg(lines: np.ndarray, line_abc: np.ndarray) -> np.ndarray:
    """Crossing angle between each epipolar line and the target line, in degrees.

    Zero means parallel, which is the degenerate case: the intersection then
    slides arbitrarily far along the target line under sub-pixel noise.
    """
    normals = np.asarray(lines, dtype=np.float64)[:, :2]
    target_normal = np.asarray(line_abc, dtype=np.float64)[:2]
    cosine = np.abs(normals @ target_normal)
    return np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))


def fit_line_3d(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """PCA line fit. Returns centroid, unit direction, and RMS distance to the line."""
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0] / np.linalg.norm(vt[0])
    projected = np.outer(centred @ direction, direction)
    residual = np.linalg.norm(centred - projected, axis=1)
    return centroid, direction, float(np.sqrt(np.mean(residual ** 2)))


def common_direction(directions: Sequence[np.ndarray]) -> np.ndarray:
    """Mean direction of the ladder's lines, with signs aligned first."""
    stacked = np.asarray(directions, dtype=np.float64)
    reference = stacked[0]
    aligned = stacked * np.sign(stacked @ reference)[:, None]
    mean = aligned.mean(axis=0)
    return mean / np.linalg.norm(mean)


def pixel_gap_2d(
    line_one: np.ndarray, line_two: np.ndarray,
    extent_one: Tuple[float, float],
) -> float:
    """Perpendicular image distance between two fitted lines, in pixels.

    Taken at the midpoint of ``line_one``'s mark so a small angle difference
    between the fits does not get levered by measuring at the origin.
    """
    mid = 0.5 * (extent_one[0] + extent_one[1])
    point = base_point(line_one) + mid * line_direction(line_one)
    return float(abs(line_two[0] * point[0] + line_two[1] * point[1] + line_two[2]))


def clip_line_to_image(
    line_abc: np.ndarray, width: int, height: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The segment of a homogeneous line that crosses the image, or None."""
    borders = (
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 1.0 - height]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 1.0 - width]),
    )
    hits: List[np.ndarray] = []
    for border in borders:
        point = np.cross(line_abc, border)
        if abs(point[2]) < 1e-12:
            continue
        pixel = point[:2] / point[2]
        if -1.0 <= pixel[0] <= width and -1.0 <= pixel[1] <= height:
            if all(np.linalg.norm(pixel - kept) > 1.0 for kept in hits):
                hits.append(pixel)
    if len(hits) < 2:
        return None
    best_i, best_j, best_d = 0, 1, 0.0
    for i in range(len(hits)):
        for j in range(i + 1, len(hits)):
            dist = float(np.linalg.norm(hits[i] - hits[j]))
            if dist > best_d:
                best_i, best_j, best_d = i, j, dist
    return hits[best_i], hits[best_j]


def adjacent_distances(
    pairs: np.ndarray, distances: np.ndarray,
) -> List[float]:
    """Distances whose pair is two neighbouring marks, in mark order."""
    by_left = {
        int(i): float(value)
        for (i, j), value in zip(pairs, distances)
        if int(j) == int(i) + 1
    }
    if not by_left:
        return []
    return [by_left[index] for index in range(max(by_left) + 1) if index in by_left]


def perpendicular_distance(
    centroid_one: np.ndarray, centroid_two: np.ndarray, direction: np.ndarray,
) -> float:
    """Distance between two parallel 3D lines: the offset perpendicular to them.

    This is the quantity the ladder's gaps actually describe. A plain
    point-to-point distance between samples at different heights along the two
    lines carries an extra height term and reads high.
    """
    offset = np.asarray(centroid_two) - np.asarray(centroid_one)
    return float(np.linalg.norm(offset - (offset @ direction) * direction))


# --------------------------------------------------------------------------- #
# Per-session measurement
# --------------------------------------------------------------------------- #

def measure_session(
    image_a: np.ndarray,
    image_b: np.ndarray,
    extrinsics: StereoExtrinsics,
    target: LineLadderTarget,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Detect, correspond, triangulate and measure one stereo pair.

    Detection happens on UNDISTORTED images, which is not a convenience. Under
    radial distortion a straight 3D line projects to a curve, so fitting a
    straight line in the raw image fits a chord to an arc and lands the epipolar
    intersection off the true correspondence. Undistorting first makes the marks
    genuinely straight and leaves every detected point already in the ideal
    pinhole frame that the epipolar geometry and the DLT both assume.

    Measured on the synthetic check: detecting in the distorted image gave a
    +0.094 % scale error and 0.040 mm residual on data whose true error is zero;
    detecting after undistortion gave -0.000 % and 0.001 mm.
    """
    clean_a, clean_b = undistort_pair(image_a, image_b, extrinsics)
    gray_a = cv2.cvtColor(clean_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(clean_b, cv2.COLOR_BGR2GRAY)

    lines_a, extents_a, widths_a = detect_ladder(gray_a, target, params)
    lines_b, extents_b, widths_b = detect_ladder(gray_b, target, params)
    if len(lines_a) != len(lines_b):
        raise ValueError(
            f"{len(lines_a)} marks in camera A but {len(lines_b)} in camera B -- the "
            "views must show the same marks to be paired by index; reframe so the same "
            "run of the plate is fully visible in both"
        )
    count = len(lines_a)

    fundamental = fundamental_for_undistorted(extrinsics)
    samples = params["samples_per_line"]

    centroids: List[np.ndarray] = []
    directions: List[np.ndarray] = []
    straightness: List[float] = []
    angles: List[float] = []
    depths: List[float] = []
    annotations_a: List[np.ndarray] = []
    annotations_b: List[np.ndarray] = []
    epipolars_b: List[np.ndarray] = []

    for index, (line_a, extent) in enumerate(zip(lines_a, extents_a)):
        # Already in the ideal pinhole frame: the images were undistorted above.
        pixels_a = sample_along_line(line_a, extent[0], extent[1], samples)
        epipolars = epipolar_lines(pixels_a, fundamental)

        line_b = lines_b[index]
        crossing = intersection_angles_deg(epipolars, line_b)
        keep = crossing >= params["min_intersection_angle_deg"] if params["conditioning_guard"] \
            else np.ones(len(crossing), dtype=bool)
        if keep.sum() < 2:
            raise ValueError(
                f"line {index}: only {int(keep.sum())} samples cross the epipolar lines "
                f"above {params['min_intersection_angle_deg']:.0f} deg "
                f"(median crossing angle {np.median(crossing):.1f} deg) -- "
                "the target is probably oriented along the epipolar direction"
            )
        angles.extend(crossing[keep].tolist())

        kept_epipolar = epipolars[keep]
        pixels_b = np.array([
            intersect_lines(epipolar, line_b) for epipolar in kept_epipolar
        ])
        points_3d = triangulate_points(
            pixels_a[keep], pixels_b,
            extrinsics.camera_matrix_a, extrinsics.camera_matrix_b,
            extrinsics.R, extrinsics.T,
        )
        centroid, direction, residual = fit_line_3d(points_3d)
        centroids.append(centroid)
        directions.append(direction)
        straightness.append(residual * 1000.0)
        depths.extend(points_3d[:, 2].tolist())
        annotations_a.append(pixels_a[keep])
        annotations_b.append(pixels_b)
        epipolars_b.append(kept_epipolar)

    axis = common_direction(directions)
    parallelism = [
        float(np.degrees(np.arccos(np.clip(abs(direction @ axis), 0.0, 1.0))))
        for direction in directions
    ]

    pairs, truth_mm = target.pair_distances_mm(count)
    measured_mm = np.array([
        perpendicular_distance(centroids[i], centroids[j], axis) * 1000.0
        for i, j in pairs
    ])
    pixel_a = np.array([
        pixel_gap_2d(lines_a[i], lines_a[j], extents_a[i]) for i, j in pairs
    ])
    pixel_b = np.array([
        pixel_gap_2d(lines_b[i], lines_b[j], extents_b[i]) for i, j in pairs
    ])

    return {
        "count": count,
        "pairs": pairs,
        "truth_mm": truth_mm,
        "measured_mm": measured_mm,
        "pixel_a": pixel_a,
        "pixel_b": pixel_b,
        "straightness_rms_mm": float(np.mean(straightness)),
        "parallelism_max_deg": float(np.max(parallelism)),
        "intersection_angle_median_deg": float(np.median(angles)),
        "depth_mean_m": float(np.mean(depths)),
        "depth_min_m": float(np.min(depths)),
        "depth_max_m": float(np.max(depths)),
        "line_width_px_a": float(np.mean(widths_a)),
        "line_width_px_b": float(np.mean(widths_b)),
        "lines_a": lines_a,
        "lines_b": lines_b,
        "samples_a": annotations_a,
        "samples_b": annotations_b,
        "epipolars_b": epipolars_b,
        "extents_a": extents_a,
        "extents_b": extents_b,
        # Undistorted, because that is the frame the lines and samples live in.
        "image_a": clean_a,
        "image_b": clean_b,
    }


# --------------------------------------------------------------------------- #
# Aggregation
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


def aggregate_results(results: Sequence[Dict[str, Any]], target: LineLadderTarget) -> Dict[str, Any]:
    """Pool per-session measurements by true length, then fit the overall scale.

    Grouping by the true distance rather than by mark index is what lets sessions
    that caught different runs of an open-ended plate be pooled at all: two
    captures showing 14 and 16 marks still both contain 5 mm, 10 mm, 15 mm ...
    lengths, and those are the same physical quantity however the marks happened
    to be numbered in each frame.
    """
    scored = [result for result in results if "skipped" not in result]

    buckets: Dict[float, List[float]] = {}
    for result in scored:
        for truth, measured in zip(result["truth_mm"], result["measured_mm"]):
            buckets.setdefault(round(float(truth), 6), []).append(float(measured))

    per_pair: List[Dict[str, Any]] = []
    for truth in sorted(buckets):
        values = np.asarray(buckets[truth], dtype=np.float64)
        mean = float(values.mean())
        per_pair.append({
            "truth_mm": truth,
            "samples": int(values.size),
            "measured_mean_mm": mean,
            "error_mm": mean - truth,
            "error_pct": (mean - truth) / truth * 100.0,
        })

    errors = np.array([entry["error_mm"] for entry in per_pair])
    relative = np.array([entry["error_pct"] for entry in per_pair])

    flat_truth = np.concatenate([r["truth_mm"] for r in scored])
    flat_measured = np.concatenate([r["measured_mm"] for r in scored])

    per_session = [{
        "label": result["label"],
        "marks": result["count"],
        "depth_mean_m": result["depth_mean_m"],
        "straightness_rms_mm": result["straightness_rms_mm"],
        "parallelism_max_deg": result["parallelism_max_deg"],
        "intersection_angle_median_deg": result["intersection_angle_median_deg"],
        "line_width_px_a": result["line_width_px_a"],
        "line_width_px_b": result["line_width_px_b"],
        "scale": fit_scale(result["truth_mm"], result["measured_mm"])["scale"],
    } for result in scored]

    return {
        "sessions_scored": len(scored),
        "per_pair": per_pair,
        "per_session": per_session,
        "scale_fit": fit_scale(flat_truth, flat_measured),
        "mean_error_mm": float(errors.mean()),
        "rms_error_mm": float(np.sqrt(np.mean(errors ** 2))),
        "max_abs_error_mm": float(np.abs(errors).max()),
        "mean_abs_relative_pct": float(np.abs(relative).mean()),
        "max_abs_relative_pct": float(np.abs(relative).max()),
        "straightness_rms_mm": float(np.mean([r["straightness_rms_mm"] for r in scored])),
        "intersection_angle_median_deg": float(
            np.median([r["intersection_angle_median_deg"] for r in scored])
        ),
        "depth_mean_m": float(np.mean([r["depth_mean_m"] for r in scored])),
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def _line_color(index: int) -> Tuple[int, int, int]:
    return LINE_COLORS[index % len(LINE_COLORS)]


def _mark_image_top(
    line: np.ndarray, extent: Tuple[float, float],
) -> np.ndarray:
    """The endpoint of the mark that sits higher in the image (smaller y)."""
    direction = line_direction(line)
    base = base_point(line)
    start = base + extent[0] * direction
    end = base + extent[1] * direction
    return start if start[1] < end[1] else end


def _draw_mark(
    canvas: np.ndarray, line: np.ndarray, points: np.ndarray,
    extent: Tuple[float, float], color: Tuple[int, int, int],
    thickness: int, label: str,
) -> None:
    direction = line_direction(line)
    base = base_point(line)
    start = (base + extent[0] * direction).astype(int)
    end = (base + extent[1] * direction).astype(int)
    cv2.line(canvas, tuple(start), tuple(end), color, thickness)
    radius = max(2, 2 * thickness)
    for point in points:
        cv2.circle(canvas, tuple(np.round(point).astype(int)), radius, color, -1)
    if label and len(points):
        text_at = tuple(np.round(_mark_image_top(line, extent)).astype(int))
        cv2.putText(canvas, label, (text_at[0] + 4 * thickness, text_at[1] - 4 * thickness),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thickness, (0, 0, 255),
                    max(1, thickness), cv2.LINE_AA)


def _draw_gap_labels(
    canvas: np.ndarray, lines: Sequence[np.ndarray],
    extents: Sequence[Tuple[float, float]], gaps_mm: Sequence[float],
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45 * thickness
    weight = max(1, thickness)
    for index, gap in enumerate(gaps_mm):
        if index + 1 >= len(lines):
            break
        left = _mark_image_top(lines[index], extents[index])
        right = _mark_image_top(lines[index + 1], extents[index + 1])
        mid = (left + right) / 2.0
        text = f"{gap:.2f} mm"
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, weight)
        x = int(round(mid[0] - text_w / 2.0))
        # Tight neighbouring gaps (3 mm is ~160 px here) would overlap if every
        # label sat on the same row, so odd gaps drop one line.
        y = int(round(min(left[1], right[1]) - 6 * thickness
                       - (index % 2) * (text_h + 8 * thickness)))
        y = max(text_h + 8, y)
        cv2.rectangle(
            canvas,
            (x - 4, y - text_h - 4),
            (x + text_w + 4, y + baseline + 4),
            (0, 0, 0), -1,
        )
        cv2.putText(canvas, text, (x, y), font, scale, (0, 0, 255),
                    weight, cv2.LINE_AA)


def draw_annotation(
    image: np.ndarray, lines: Sequence[np.ndarray], samples: Sequence[np.ndarray],
    extents: Sequence[Tuple[float, float]],
    adjacent_mm: Optional[Sequence[float]] = None,
    camera_name: str = "",
) -> np.ndarray:
    """Detected lines, sample points, and neighbouring triangulated gaps.

    Each line is drawn only over the extent where its mark was actually found, not
    as an infinite line, so the picture shows what was measured rather than where
    a fitted line happens to go. A line drawn across the whole frame hides exactly
    the failure worth seeing: samples wandering off the mark onto the background.

    Gap labels are the stereo (3D) millimetre distances, the same numbers as the
    report -- not this camera's pixel spacing.
    """
    canvas = image.copy()
    width = canvas.shape[1]
    thickness = max(1, int(round(width / 1280.0)))
    if camera_name:
        cv2.putText(canvas, camera_name, (8 * thickness, 24 * thickness),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * thickness, (0, 0, 255),
                    max(1, thickness), cv2.LINE_AA)
    for index, (line, points, extent) in enumerate(zip(lines, samples, extents)):
        _draw_mark(canvas, line, points, extent, _line_color(index),
                   thickness, str(index))
    if adjacent_mm:
        _draw_gap_labels(canvas, lines, extents, adjacent_mm, thickness)
    return canvas


def draw_correspondences(
    image_a: np.ndarray, image_b: np.ndarray,
    lines_a: Sequence[np.ndarray], lines_b: Sequence[np.ndarray],
    samples_a: Sequence[np.ndarray], samples_b: Sequence[np.ndarray],
    extents_a: Sequence[Tuple[float, float]], extents_b: Sequence[Tuple[float, float]],
    epipolars_b: Sequence[np.ndarray],
    adjacent_mm: Optional[Sequence[float]] = None,
    camera_a: str = "rgb_cam1", camera_b: str = "rgb_cam2",
) -> np.ndarray:
    """Side-by-side view of how a sample in A is paired with a point in B.

    For each kept sample on mark i in A the epipolar line is drawn in B, and the
    correspondence is the intersection of that line with mark i in B. Matching
    dots are the same colour; a subset of the epipolar lines is drawn so the
    crossing is visible without stacking every sample's line on top of the rest.
    """
    left = image_a.copy()
    right = image_b.copy()
    thickness = max(1, int(round(left.shape[1] / 1280.0)))
    height_b, width_b = right.shape[:2]

    cv2.putText(left, camera_a, (8 * thickness, 24 * thickness),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * thickness, (0, 0, 255),
                max(1, thickness), cv2.LINE_AA)
    cv2.putText(right, camera_b, (8 * thickness, 24 * thickness),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * thickness, (0, 0, 255),
                max(1, thickness), cv2.LINE_AA)

    for index, (line_a, line_b, pts_a, pts_b, extent_a, extent_b, epipolar) in enumerate(
        zip(lines_a, lines_b, samples_a, samples_b, extents_a, extents_b, epipolars_b)
    ):
        color = _line_color(index)
        _draw_mark(left, line_a, pts_a, extent_a, color, thickness, str(index))
        _draw_mark(right, line_b, pts_b, extent_b, color, thickness, str(index))
        n_pts = len(pts_a)
        if n_pts == 0:
            continue
        pick = np.unique(np.linspace(0, n_pts - 1, min(EPIPOLAR_DRAW_COUNT, n_pts)).astype(int))
        for sample_index in pick:
            clipped = clip_line_to_image(epipolar[sample_index], width_b, height_b)
            if clipped is not None:
                start = tuple(np.round(clipped[0]).astype(int))
                end = tuple(np.round(clipped[1]).astype(int))
                cv2.line(right, start, end, color, max(1, thickness // 2), cv2.LINE_AA)

    if adjacent_mm:
        _draw_gap_labels(left, lines_a, extents_a, adjacent_mm, thickness)
        _draw_gap_labels(right, lines_b, extents_b, adjacent_mm, thickness)

    height = max(left.shape[0], right.shape[0])
    gutter = 16
    canvas = np.zeros((height, left.shape[1] + gutter + right.shape[1], 3), dtype=np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1] + gutter:] = right
    shift = np.array([left.shape[1] + gutter, 0], dtype=np.int32)

    for index, (pts_a, pts_b) in enumerate(zip(samples_a, samples_b)):
        color = _line_color(index)
        n_pts = len(pts_a)
        if n_pts == 0:
            continue
        pick = np.unique(np.linspace(0, n_pts - 1, min(EPIPOLAR_DRAW_COUNT, n_pts)).astype(int))
        for sample_index in pick:
            start = tuple(np.round(pts_a[sample_index]).astype(int))
            end = tuple(np.round(pts_b[sample_index] + shift).astype(int))
            cv2.line(canvas, start, end, color, max(1, thickness // 2), cv2.LINE_AA)
    return canvas


def write_report(
    summary: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    target: LineLadderTarget,
    params: Dict[str, Any],
    context: Dict[str, Any],
    path: Path,
) -> None:
    rule = "=" * 78
    thin = "-" * 78
    lines = [
        rule,
        "LINE-LADDER METRIC ACCURACY",
        rule,
        f"extrinsics       : {context['extrinsics']}",
        f"target           : {context['target']}",
        f"ground truth     : {target.measured_by}",
        f"ladder           : {context['ladder']}",
        f"uncertainty      : +/- {target.gaps_uncertainty_mm:.3f} mm on the span",
        f"orientation      : {target.orientation}",
        f"samples per line : {params['samples_per_line']}",
        f"sub-pixel refine : {'on' if params['refine'] else 'OFF'}",
        f"conditioning     : {'on' if params['conditioning_guard'] else 'OFF'} "
        f"(min {params['min_intersection_angle_deg']:.0f} deg)",
        f"sessions scored  : {summary['sessions_scored']}",
        f"cameras          : {context['camera_a']} (A) / {context['camera_b']} (B)",
        "",
    ]

    skipped = [result for result in results if "skipped" in result]
    if skipped:
        lines.extend([thin, "SKIPPED SESSIONS", thin])
        lines.extend(f"  {result['label']}: {result['skipped']}" for result in skipped)
        lines.append("")

    camera_a = context["camera_a"]
    camera_b = context["camera_b"]
    col_a = f"{camera_a} px"
    col_b = f"{camera_b} px"
    pair_header = (
        f"{'pair':<6} {'true mm':>8} {'stereo mm':>10} "
        f"{col_a:>12} {col_b:>12} {'err mm':>9} {'err %':>8}"
    )

    for result in results:
        if "skipped" in result:
            continue
        lines.extend([
            thin,
            f"SESSION {result['label']}  --  stereo mm is the 3D measurement; "
            f"{camera_a}/{camera_b} px are that camera's image-plane gap",
            thin,
            pair_header,
        ])
        pairs = np.asarray(result["pairs"])
        order = np.lexsort((pairs[:, 0], pairs[:, 1] - pairs[:, 0]))
        for index in order:
            left, right = (int(value) for value in pairs[index])
            truth = float(result["truth_mm"][index])
            stereo = float(result["measured_mm"][index])
            error = stereo - truth
            lines.append(
                f"{f'{left}-{right}':<6} {truth:>8.3f} {stereo:>10.3f} "
                f"{result['pixel_a'][index]:>12.1f} {result['pixel_b'][index]:>12.1f} "
                f"{error:>+9.3f} {(error / truth) * 100:>+8.3f}"
            )
        lines.append("")

    fit = summary["scale_fit"]
    lines.extend([
        thin,
        "SCALE FIT  (the headline number)",
        thin,
        f"  measured = a * true          a = {fit['scale']:.5f}  "
        f"-> {fit['scale_error_pct']:+.3f} %",
        f"  residual after scale         {fit['residual_rms_mm']:.4f} mm rms",
        f"  measured = a * true + b      a = {fit['affine_slope']:.5f}  "
        f"({fit['affine_slope_error_pct']:+.3f} %), b = {fit['affine_intercept_mm']:+.4f} mm",
        f"  residual after affine        {fit['affine_residual_rms_mm']:.4f} mm rms",
        "",
        "  a-1 is systematic scale error (e.g. a mis-measured ChArUco square).",
        "  A large b is a line-localisation bias: a fixed offset, not a scale error.",
        "",
        thin,
        "OVERALL",
        thin,
        f"  mean error            {summary['mean_error_mm']:+.4f} mm",
        f"  rms error             {summary['rms_error_mm']:.4f} mm",
        f"  max abs error         {summary['max_abs_error_mm']:.4f} mm",
        f"  mean abs relative     {summary['mean_abs_relative_pct']:.3f} %",
        f"  max abs relative      {summary['max_abs_relative_pct']:.3f} %",
        "",
        thin,
        "MEASUREMENT QUALITY",
        thin,
        f"  3D line straightness  {summary['straightness_rms_mm']:.4f} mm rms "
        "(point-level noise within a session)",
        f"  epipolar crossing     {summary['intersection_angle_median_deg']:.1f} deg median "
        "(90 is ideal, low is degenerate)",
        f"  mean working depth    {summary['depth_mean_m'] * 1000:.1f} mm",
        "",
        thin,
        "PER SESSION",
        thin,
        f"{'session':<24} {'marks':>6} {'depth mm':>9} {'scale %':>9} "
        f"{camera_a+' w':>10} {camera_b+' w':>10} {'cross deg':>10}",
    ])
    for entry in summary["per_session"]:
        lines.append(
            f"{entry['label'][:24]:<24} {entry['marks']:>6} "
            f"{entry['depth_mean_m'] * 1000:>9.1f} "
            f"{(entry['scale'] - 1) * 100:>+9.3f} "
            f"{entry['line_width_px_a']:>10.1f} {entry['line_width_px_b']:>10.1f} "
            f"{entry['intersection_angle_median_deg']:>10.1f}"
        )

    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check triangulated millimetre distances against a measured line ladder.",
    )
    parser.add_argument(
        "--captures", nargs="+", default=None,
        help="Directories holding line-target capture sessions (default: "
             "geometric_calibration.line_target_captures, else "
             f"{DEFAULT_LINE_TARGET_CAPTURES}).",
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
                        help="Line ladder YAML (default: geometric_calibration.line_target "
                             f"in config, else {DEFAULT_LINE_TARGET_CONFIG}).")
    parser.add_argument("--background-kernel", type=float,
                        default=DEFAULT_BACKGROUND_KERNEL_REF,
                        help="Width of the structuring element that erases the marks to "
                             "leave the background, in 1280-wide reference pixels, scaled "
                             "to the real frame. Must exceed a mark's width "
                             "(default: %(default)s).")
    parser.add_argument("--refine-half-width", type=float,
                        default=DEFAULT_REFINE_HALF_WIDTH_REF,
                        help="Half-width of the perpendicular profile whose centroid "
                             "locates a mark, reference px (default: %(default)s).")
    parser.add_argument("--refine-step", type=float, default=DEFAULT_REFINE_STEP_REF,
                        help="Scanline spacing along a line, reference px "
                             "(default: %(default)s).")
    parser.add_argument("--peak-smoothing", type=float, default=DEFAULT_PEAK_SMOOTHING_REF,
                        help="Smoothing of the projection profile before peak picking, "
                             "reference px (default: %(default)s).")
    parser.add_argument("--angle-search-deg", type=float, default=DEFAULT_ANGLE_SEARCH_DEG,
                        help="Half-range searched either side of the nominal orientation "
                             "(default: %(default)s).")
    parser.add_argument("--angle-steps", type=int, default=DEFAULT_ANGLE_STEPS)
    parser.add_argument("--samples-per-line", type=int, default=DEFAULT_SAMPLES_PER_LINE)
    parser.add_argument("--min-intersection-angle-deg", type=float,
                        default=DEFAULT_MIN_INTERSECTION_ANGLE_DEG,
                        help="Reject samples whose epipolar line crosses the target line "
                             "below this angle (default: %(default)s).")
    parser.add_argument("--no-conditioning-guard", action="store_true",
                        help="Keep samples regardless of epipolar crossing angle.")
    parser.add_argument("--no-refine", action="store_true",
                        help="Use the coarse projection peak as each line, skipping the "
                             "per-scanline darkness-centroid refinement.")
    parser.add_argument("--out", "--output", dest="output", default=None,
                        help="Output directory (default: geometric_calibration.output_dir "
                             f"in config) / {DEFAULT_OUTPUT_SUBDIR}.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    geo_config = load_config().get("geometric_calibration", {}) or {}

    extrinsics_path = resolve_path(
        args.extrinsics or default_extrinsics_path(args.camera_a, args.camera_b)
    )
    if not extrinsics_path.exists():
        raise SystemExit(f"No stereo extrinsics at {extrinsics_path}.")
    extrinsics = StereoExtrinsics.load_json(extrinsics_path)

    target_path = resolve_path(
        args.target or geo_config.get("line_target", DEFAULT_LINE_TARGET_CONFIG)
    )
    target = LineLadderTarget.from_yaml(target_path)

    if target.open_ended:
        ladder = (f"uniform {target.spacing_mm:g} mm spacing, count taken from the image "
                  f"({target.orientation})")
    else:
        ladder = (f"{target.line_count} marks, gaps {list(target.gaps_mm)} mm, "
                  f"span {target.span_mm():.2f} mm ({target.orientation})")
    print(f"ladder: {ladder}")

    # Refuse the degenerate orientation before spending time on detection. A mark
    # running parallel to the epipolar direction carries no depth information at
    # all -- the aperture problem -- so no amount of detection quality can rescue
    # it, and "found N marks" would be a misleading way to report that.
    centre = np.array([[extrinsics.image_size_a[0] / 2.0,
                        extrinsics.image_size_a[1] / 2.0]])
    epipolar = epipolar_lines(centre, fundamental_for_undistorted(extrinsics))
    mark_normal = np.array([1.0, 0.0]) if target.orientation == "vertical" \
        else np.array([0.0, 1.0])
    crossing = float(intersection_angles_deg(epipolar, np.append(mark_normal, 0.0))[0])
    if crossing < args.min_intersection_angle_deg:
        other = "horizontal" if target.orientation == "vertical" else "vertical"
        raise SystemExit(
            f"This rig cannot measure a {target.orientation} ladder: its marks cross the "
            f"epipolar lines at only {crossing:.1f} deg.\n"
            "A mark parallel to the stereo baseline gives no depth at all (the aperture "
            "problem) -- there is nothing along it to match between the two views.\n"
            f"Rotate the target 90 deg so the marks run {other} in the image, and set "
            f"orientation: {other} in {target_path.name}."
        )
    print(f"marks cross the epipolar lines at {crossing:.1f} deg (90 is ideal)")

    if not target.open_ended and not target.strictly_increasing:
        print(f"note: {target_path.name} has non-increasing gaps, so the ladder looks "
              "the same shifted by one mark. A mark lost off the frame edge cannot be "
              "detected from the pattern -- frame the whole plate with clear margin in "
              "BOTH cameras. Sessions where the mark count is wrong are skipped.")
    if "NOT verified" in target.measured_by:
        print(f"warning: {target_path.name} still carries unverified ground truth "
              f"({target.measured_by}). Every number below is only as good as that.")

    if args.session:
        # Look only in the named folder. Scanning its parent would report every
        # unrelated sibling session as "no rgb_cam1.jpg", which buries the one
        # message that matters.
        session_dir = resolve_path(args.session)
        capture_dirs = [session_dir]
    else:
        capture_values = args.captures or geo_config.get("line_target_captures") \
            or DEFAULT_LINE_TARGET_CAPTURES
        if isinstance(capture_values, str):
            capture_values = [capture_values]
        capture_dirs = [resolve_path(value) for value in capture_values]

    if args.session:
        paths = [session_dir / f"{camera}.jpg" for camera in (args.camera_a, args.camera_b)]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            raise SystemExit(f"{session_dir} is missing {', '.join(missing)}.")
        session_pairs, notes = [(session_dir.name, paths[0], paths[1])], []
    else:
        session_pairs, notes = collect_session_pairs(
            capture_dirs, args.camera_a, args.camera_b,
        )
    for note in notes:
        print(note)
    if not session_pairs:
        raise SystemExit(f"No {args.camera_a}/{args.camera_b} sessions under "
                         f"{', '.join(str(path) for path in capture_dirs)}.")

    image_size = extrinsics.image_size_a
    params = {
        "background_kernel_px": scaled_px(args.background_kernel, image_size),
        "refine_half_width_px": scaled_px(args.refine_half_width, image_size),
        "refine_step_px": scaled_px(args.refine_step, image_size),
        "peak_smoothing_px": scaled_px(args.peak_smoothing, image_size),
        "angle_search_deg": args.angle_search_deg,
        "angle_steps": args.angle_steps,
        "samples_per_line": args.samples_per_line,
        "min_intersection_angle_deg": args.min_intersection_angle_deg,
        "conditioning_guard": not args.no_conditioning_guard,
        "refine": not args.no_refine,
    }

    output_dir = resolve_path(
        args.output or geo_config.get("output_dir", "calibration/results")
    ) / DEFAULT_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for label, path_a, path_b in session_pairs:
        image_a = cv2.imread(str(path_a))
        image_b = cv2.imread(str(path_b))
        if image_a is None or image_b is None:
            results.append({"label": label, "skipped": "failed to decode one of the images"})
            print(f"{label}: skipped: failed to decode")
            continue
        try:
            measured = measure_session(image_a, image_b, extrinsics, target, params)
        except ValueError as exc:
            results.append({"label": label, "skipped": str(exc)})
            print(f"{label}: skipped: {exc}")
            continue

        measured["label"] = label
        results.append(measured)
        gaps_mm = adjacent_distances(measured["pairs"], measured["measured_mm"])
        cv2.imwrite(
            str(output_dir / f"{label}_{args.camera_a}_lines.jpg"),
            draw_annotation(
                measured["image_a"], measured["lines_a"],
                measured["samples_a"], measured["extents_a"],
                adjacent_mm=gaps_mm, camera_name=args.camera_a,
            ),
        )
        cv2.imwrite(
            str(output_dir / f"{label}_{args.camera_b}_lines.jpg"),
            draw_annotation(
                measured["image_b"], measured["lines_b"],
                measured["samples_b"], measured["extents_b"],
                adjacent_mm=gaps_mm, camera_name=args.camera_b,
            ),
        )
        cv2.imwrite(
            str(output_dir / f"{label}_correspondences.jpg"),
            draw_correspondences(
                measured["image_a"], measured["image_b"],
                measured["lines_a"], measured["lines_b"],
                measured["samples_a"], measured["samples_b"],
                measured["extents_a"], measured["extents_b"],
                measured["epipolars_b"],
                adjacent_mm=gaps_mm,
                camera_a=args.camera_a, camera_b=args.camera_b,
            ),
        )
        session_scale = fit_scale(measured["truth_mm"], measured["measured_mm"])
        print(f"{label}: scale {session_scale['scale_error_pct']:+.3f} %, "
              f"depth {measured['depth_mean_m'] * 1000:.0f} mm, "
              f"crossing {measured['intersection_angle_median_deg']:.0f} deg")

    if not any("skipped" not in result for result in results):
        raise SystemExit("No session could be measured; see the skip reasons above.")

    summary = aggregate_results(results, target)
    context = {
        "extrinsics": str(extrinsics_path),
        "target": str(target_path),
        "ladder": ladder,
        "camera_a": args.camera_a,
        "camera_b": args.camera_b,
    }

    write_report(summary, results, target, params, context, output_dir / "report.txt")

    payload = {
        "extrinsics": str(extrinsics_path),
        "target": target.to_dict(),
        "target_path": str(target_path),
        "configuration": params,
        "summary": summary,
        "sessions": [
            {"label": result["label"], "skipped": result["skipped"]}
            if "skipped" in result else
            {
                "label": result["label"],
                "measured_mm": result["measured_mm"].tolist(),
                "pixel_a": result["pixel_a"].tolist(),
                "pixel_b": result["pixel_b"].tolist(),
                "truth_mm": result["truth_mm"].tolist(),
                "pairs": result["pairs"].tolist(),
                "depth_mean_m": result["depth_mean_m"],
                "straightness_rms_mm": result["straightness_rms_mm"],
                "parallelism_max_deg": result["parallelism_max_deg"],
                "intersection_angle_median_deg": result["intersection_angle_median_deg"],
                "line_width_px_a": result["line_width_px_a"],
                "line_width_px_b": result["line_width_px_b"],
            }
            for result in results
        ],
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    stale_figure = output_dir / "line_accuracy.png"
    if stale_figure.exists():
        stale_figure.unlink()

    fit = summary["scale_fit"]
    print()
    print(f"scale error   {fit['scale_error_pct']:+.3f} %  "
          f"(a = {fit['scale']:.5f}, residual {fit['residual_rms_mm']:.4f} mm rms)")
    print(f"mean abs err  {summary['mean_abs_relative_pct']:.3f} %  "
          f"({summary['rms_error_mm']:.4f} mm rms, max {summary['max_abs_error_mm']:.4f} mm)")
    print(f"Saved report.txt / result.json / *_lines.jpg / *_correspondences.jpg to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
