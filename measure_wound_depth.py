#!/usr/bin/env python3
"""Measure a real specimen's local depth against its own immediate surroundings.

Built for a case check_depth_accuracy.py's plane-fit machinery cannot cover: a
real specimen (e.g. a wound) has no known ground-truth height and, unlike a
rigid calibration plate, no guarantee of being flat even locally. So there is
no single global reference plane here. Instead, click points along the rim
(intact surface immediately surrounding the area of interest) and, for every
point you then measure, only the k NEAREST rim points (by 3D distance, not
click order) are used to fit that point's own small local plane -- different
points, especially across a curved region, get different local planes drawn
from whichever rim points are actually close to them.

Method, per session::

    click points along the rim (at least --neighbors of them, spread around
    the area of interest, close to it -- not distant surface), press n to
    lock the rim in
    click every point you want measured (any number, no further n needed) --
    each is immediately triangulated, its k nearest rim points found, a local
    plane fit through just those, and its signed distance to that local plane
    reported as this point's depth. A point recessed relative to its
    immediate surroundings reads a POSITIVE depth (opposite sign convention
    from check_depth_accuracy.py, which reports a block PROTRUDING toward the
    camera as positive -- here the specimen is expected to be a depression,
    and "depth in mm below the surface" reading positive is the intuitive
    convention for that)

Reported depths print live as you click, using whatever rim points exist at
that moment -- a convenient running estimate, not the final answer. The
authoritative numbers saved to report.txt/result.json are always recomputed
once, at the end of the session, from the FINAL rim and point sets (after any
undos) -- see measure_wound_session. This also means undoing a rim point
after the rim is already "locked" is always safe: it only changes what that
one final recomputation uses, it can never leave a half-updated number on
screen.

Usage:
    python measure_wound_depth.py --session captures/<timestamp>
    python measure_wound_depth.py --session captures/<timestamp> \\
        --rim 100,200,90,205 --rim 900,200,890,205 --rim 100,900,90,905 \\
        --rim 900,900,890,905 --rim 500,150,490,155 \\
        --wound 500,500,490,505   # non-interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_cameras import load_config, resolve_path  # noqa: E402
from calibration.stereo import StereoExtrinsics  # noqa: E402
from check_depth_accuracy import fit_plane_3d, perpendicular_distance_to_plane  # noqa: E402
from measure_points import (  # noqa: E402
    DEFAULT_BLOB_RADIUS_PX,
    DEFAULT_LOUPE_ZOOM,
    DEFAULT_MAX_WINDOW,
    Panel,
    draw_marker,
    measure_points as triangulate_clicks,
    parse_point,
    run_interactive,
)
from registration_io import default_extrinsics_path, undistort_pair  # noqa: E402

DEFAULT_OUTPUT_SUBDIR = "wound_depth"
DEFAULT_NEIGHBORS = 5


# --------------------------------------------------------------------------- #
# Geometry: local plane from a point's nearest rim neighbors
# --------------------------------------------------------------------------- #

def nearest_rim_plane(
    point_mm: np.ndarray, rim_points_mm: np.ndarray, neighbors: int,
) -> Dict[str, float]:
    """Fit a plane through a point's ``neighbors`` nearest rim points (3D mm
    distance) and return that point's signed depth below it.

    Every measured point gets its OWN local plane from whichever rim points
    are actually close to it -- this is what tolerates a curved surface
    without assuming one global plane fits everywhere, unlike
    check_depth_accuracy.py's single reference plane (valid there only
    because a calibration plate is rigid and flat by construction).
    """
    point_mm = np.asarray(point_mm, dtype=np.float64)
    rim_points_mm = np.asarray(rim_points_mm, dtype=np.float64).reshape(-1, 3)
    if len(rim_points_mm) < neighbors:
        raise ValueError(
            f"need at least {neighbors} rim points, got {len(rim_points_mm)}"
        )
    distances_mm = np.linalg.norm(rim_points_mm - point_mm, axis=1)
    nearest = np.argsort(distances_mm)[:neighbors]
    centroid, normal, rms_m = fit_plane_3d(rim_points_mm[nearest] / 1000.0)
    if normal @ (-centroid) < 0.0:
        normal = -normal
    signed_mm = perpendicular_distance_to_plane(
        point_mm / 1000.0, centroid, normal,
    ) * 1000.0
    return {
        # protruding toward the camera is positive under
        # perpendicular_distance_to_plane's convention -- flipped here since
        # a specimen of interest is expected to be a RECESSION relative to
        # its rim, and "depth below the surrounding surface" reading
        # positive is the intuitive convention for that.
        "depth_mm": -signed_mm,
        "plane_rms_mm": rms_m * 1000.0,
        "neighbor_count": int(neighbors),
        "farthest_neighbor_mm": float(distances_mm[nearest].max()),
    }


def measure_wound_session(
    rim_clicks_a: np.ndarray, rim_clicks_b: np.ndarray,
    point_clicks_a: np.ndarray, point_clicks_b: np.ndarray,
    extrinsics: StereoExtrinsics, neighbors: int,
    rim_offsets_px: Optional[np.ndarray] = None,
    point_offsets_px: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Triangulate the rim and the measured points, then compute every
    measured point's depth against its own local rim-derived plane.

    This is the single authoritative computation -- both the interactive
    session (recomputing once at the end, from the FINAL rim/point sets
    after any undos) and the non-interactive --rim/--wound path call this
    same function, the way check_depth_accuracy.measure_depth_session is
    shared between its own two entry points.

    ``rim_offsets_px``/``point_offsets_px``, when given (the interactive
    path), override the epipolar-offset diagnostic with the real pre-snap
    value captured live by run_interactive; when None (the non-interactive
    path), it's recomputed from the already-snapped clicks, which is correct
    there since those clicks are genuinely raw.
    """
    rim_clicks_a = np.asarray(rim_clicks_a, dtype=np.float64).reshape(-1, 2)
    rim_clicks_b = np.asarray(rim_clicks_b, dtype=np.float64).reshape(-1, 2)
    point_clicks_a = np.asarray(point_clicks_a, dtype=np.float64).reshape(-1, 2)
    point_clicks_b = np.asarray(point_clicks_b, dtype=np.float64).reshape(-1, 2)

    if len(rim_clicks_a) < neighbors:
        raise ValueError(
            f"need at least {neighbors} rim clicks, got {len(rim_clicks_a)}"
        )
    if len(point_clicks_a) < 1:
        raise ValueError(f"need at least 1 measured point, got {len(point_clicks_a)}")

    rim_result = triangulate_clicks(rim_clicks_a, rim_clicks_b, extrinsics)
    rim_points_mm = np.asarray(rim_result["points_mm"], dtype=np.float64)

    point_result = triangulate_clicks(point_clicks_a, point_clicks_b, extrinsics)
    points_mm = np.asarray(point_result["points_mm"], dtype=np.float64)

    measurements: List[Dict[str, Any]] = []
    for index, point_mm in enumerate(points_mm):
        local = nearest_rim_plane(point_mm, rim_points_mm, neighbors)
        local["index"] = index
        local["epipolar_offset_px"] = float(
            point_offsets_px[index] if point_offsets_px is not None
            else point_result["epipolar_offset_px"][index]
        )
        measurements.append(local)

    return {
        "neighbors": neighbors,
        "rim_clicks_a": rim_clicks_a,
        "rim_clicks_b": np.asarray(rim_result["clicks_b_snapped"]),
        "rim_points_mm": rim_points_mm,
        "rim_epipolar_offset_max_px": float(
            np.max(rim_offsets_px) if rim_offsets_px is not None
            else np.max(rim_result["epipolar_offset_px"])
        ),
        "point_clicks_a": point_clicks_a,
        "point_clicks_b": np.asarray(point_result["clicks_b_snapped"]),
        "points_mm": points_mm,
        "measurements": measurements,
    }
