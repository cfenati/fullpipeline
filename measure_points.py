#!/usr/bin/env python3
"""Measure a real-world length between two points you pick in a stereo pair.

No target model, no detector, no assumed depth. You click a feature, the geometry
does the rest.

Why two clicks per point. A pixel is a ray, not a place: on its own it could be
5 mm away or 5 m away. A length only exists once a point is seen by both cameras
and the two rays are intersected, which is ``triangulate.py``. So each 3D point
costs one click in camera A and one in camera B.

Why the second click is easy. Given the click in A, the calibration says the match
in B lies on one line -- the epipolar line -- and nowhere else. The tool draws it
and projects your click onto it, so error perpendicular to that line is discarded
rather than believed. Only your position ALONG the line still matters. The
reported ``off`` value is how far your click had to move; a large one means you
clicked the wrong feature, and it is the main thing worth watching.

Both images are undistorted first, so the epipolar lines are straight and every
click is already in the ideal pinhole frame the triangulation assumes.

Usage:
    python measure_points.py --session captures/<timestamp>
    python measure_points.py --session captures/<timestamp> \\
        --point 1820,1140,1500,1150 --point 1820,1400,1500,1410   # non-interactive

Interactive keys:
    left click   place the point (alternates camera A -> camera B)
    arrow keys   nudge the last click by one full-resolution pixel
    u            undo the last click        r  clear everything
    + / -        loupe zoom                 s  save report and annotated image
    q / Esc      quit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_cameras import load_config, resolve_path  # noqa: E402
from touch_controls import BAR_HEIGHT as TOUCH_BAR_HEIGHT, TouchController  # noqa: E402
from calibration.stereo import StereoExtrinsics  # noqa: E402
from registration_io import (  # noqa: E402
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_REFERENCE_DEPTH,
    default_extrinsics_path,
    undistort_pair,
)
from triangulate import (  # noqa: E402
    distance_to_line,
    epipolar_lines,
    fundamental_for_undistorted,
    snap_to_line,
    triangulate_points,
)

DEFAULT_OUTPUT_SUBDIR = "point_measurements"
# The window must fit on a normal screen. An oversized window ends up partly
# offscreen, never takes keyboard focus, and then no key -- including q -- is
# ever delivered to waitKey, which looks like the program hanging.
DEFAULT_MAX_WINDOW = (1600, 900)
DEFAULT_LOUPE_ZOOM = 8
LOUPE_SIZE = 240
STATUS_HEIGHT = 150
# Radius of the window searched when snapping a click onto a dot's centroid.
DEFAULT_BLOB_RADIUS_PX = 18

COLOR_A = (219, 99, 37)
COLOR_B = (74, 163, 22)
COLOR_EPIPOLAR = (9, 134, 217)
COLOR_PENDING = (60, 60, 220)
COLOR_LABEL = (40, 220, 255)


def snap_to_blob(gray: np.ndarray, point: Sequence[float], radius: int) -> np.ndarray:
    """Move a click onto the intensity centroid of the dot it landed on.

    A hand click is good to a couple of pixels; the centroid of a dot's whole
    profile is good to a fraction of one, and it is unbiased by how big the dot
    is or how it was exposed. Polarity is read from the click itself, so a dark
    dot on white and a bright dot on black both work.

    Returns the click unchanged when there is nothing dot-like there, so a click
    on a line, a corner, or plain texture is never silently dragged somewhere else.
    """
    if radius <= 0:
        return np.asarray(point, dtype=np.float64)
    x, y = int(round(point[0])), int(round(point[1]))
    height, width = gray.shape[:2]
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float64)
    if patch.size < 9:
        return np.asarray(point, dtype=np.float64)

    background = float(np.median(patch))
    centre = float(gray[y, x])
    weight = (background - patch) if centre < background else (patch - background)
    weight = np.clip(weight, 0.0, None)
    # Keep only the strong half, so the centroid follows the dot rather than the
    # gradual falloff around it.
    weight = np.where(weight > 0.5 * weight.max(), weight, 0.0)
    total = weight.sum()
    if total <= 1e-9 or weight.max() < 5.0:
        return np.asarray(point, dtype=np.float64)

    ys, xs = np.nonzero(weight)
    values = weight[ys, xs]
    return np.array([
        x0 + float((xs * values).sum() / total),
        y0 + float((ys * values).sum() / total),
    ])


class Panel:
    """One image with its own zoom and pan, mapping screen pixels to image pixels."""

    def __init__(self, image: np.ndarray, size: Tuple[int, int]) -> None:
        self.image = image
        self.width, self.height = size
        self.fit()

    def fit(self) -> None:
        self.scale = min(self.width / self.image.shape[1],
                         self.height / self.image.shape[0])
        self.offset = np.array([
            (self.image.shape[1] - self.width / self.scale) / 2.0,
            (self.image.shape[0] - self.height / self.scale) / 2.0,
        ])

    def to_image(self, screen: Sequence[float]) -> np.ndarray:
        return np.array([screen[0] / self.scale + self.offset[0],
                         screen[1] / self.scale + self.offset[1]])

    def to_screen(self, image_point: Sequence[float]) -> Tuple[int, int]:
        return (int(round((image_point[0] - self.offset[0]) * self.scale)),
                int(round((image_point[1] - self.offset[1]) * self.scale)))

    def zoom_at(self, screen: Sequence[float], factor: float) -> None:
        """Zoom about the cursor, so the feature under it stays put."""
        anchor = self.to_image(screen)
        self.scale = float(np.clip(self.scale * factor, 0.02, 20.0))
        self.offset = anchor - np.array(screen, dtype=np.float64) / self.scale

    def pan(self, delta: Sequence[float]) -> None:
        self.offset = self.offset - np.array(delta, dtype=np.float64) / self.scale

    def render(self) -> np.ndarray:
        """Crop-and-scale the visible region, padding wherever the image runs out."""
        canvas = np.zeros((self.height, self.width, 3), np.uint8)
        x0 = int(np.floor(self.offset[0]))
        y0 = int(np.floor(self.offset[1]))
        x1 = int(np.ceil(self.offset[0] + self.width / self.scale))
        y1 = int(np.ceil(self.offset[1] + self.height / self.scale))
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(self.image.shape[1], x1), min(self.image.shape[0], y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return canvas
        crop = self.image[cy0:cy1, cx0:cx1]
        scaled = cv2.resize(
            crop, None, fx=self.scale, fy=self.scale,
            interpolation=cv2.INTER_NEAREST if self.scale > 1.5 else cv2.INTER_AREA,
        )
        dx = int(round((cx0 - self.offset[0]) * self.scale))
        dy = int(round((cy0 - self.offset[1]) * self.scale))
        sx0, sy0 = max(0, dx), max(0, dy)
        sx1 = min(self.width, dx + scaled.shape[1])
        sy1 = min(self.height, dy + scaled.shape[0])
        if sx1 > sx0 and sy1 > sy0:
            canvas[sy0:sy1, sx0:sx1] = scaled[sy0 - dy:sy1 - dy, sx0 - dx:sx1 - dx]
        return canvas


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #

def measure_points(
    clicks_a: np.ndarray, clicks_b: np.ndarray, extrinsics: StereoExtrinsics,
) -> Dict[str, Any]:
    """Snap the B clicks to their epipolar lines, then triangulate every point.

    Both click arrays are full-resolution pixels in the UNDISTORTED images.
    """
    clicks_a = np.asarray(clicks_a, dtype=np.float64).reshape(-1, 2)
    clicks_b = np.asarray(clicks_b, dtype=np.float64).reshape(-1, 2)

    fundamental = fundamental_for_undistorted(
        extrinsics.camera_matrix_a, extrinsics.camera_matrix_b, extrinsics.essential,
    )
    lines = epipolar_lines(clicks_a, fundamental)
    snapped = snap_to_line(clicks_b, lines)
    # How far each click had to move to reach its epipolar line. This is the one
    # number that says whether the two clicks were the same physical feature.
    offsets = np.abs(distance_to_line(clicks_b, lines))

    points_3d = triangulate_points(
        clicks_a, snapped,
        extrinsics.camera_matrix_a, extrinsics.camera_matrix_b,
        extrinsics.R, extrinsics.T,
    )

    pairs: List[Dict[str, Any]] = []
    for i in range(len(points_3d)):
        for j in range(i + 1, len(points_3d)):
            pairs.append({
                "from": i,
                "to": j,
                "distance_mm": float(np.linalg.norm(points_3d[j] - points_3d[i]) * 1000.0),
            })

    return {
        "clicks_a": clicks_a.tolist(),
        "clicks_b_raw": clicks_b.tolist(),
        "clicks_b_snapped": snapped.tolist(),
        "epipolar_offset_px": offsets.tolist(),
        "points_mm": (points_3d * 1000.0).tolist(),
        "depth_mm": (points_3d[:, 2] * 1000.0).tolist(),
        "pairs": pairs,
    }


def format_result(result: Dict[str, Any], depth_range: Tuple[float, float]) -> List[str]:
    """Human-readable summary, with the checks that say whether to trust it."""
    lines = ["", f"{'pt':>3} {'depth mm':>9} {'off px':>8}   X / Y / Z  (mm)"]
    for index, (point, depth, offset) in enumerate(zip(
        result["points_mm"], result["depth_mm"], result["epipolar_offset_px"],
    )):
        flag = ""
        if not depth_range[0] * 1000 <= depth <= depth_range[1] * 1000:
            flag = "  <- outside the rig's working range"
        elif offset > 5.0:
            flag = "  <- click was far off the epipolar line"
        lines.append(
            f"{index:>3} {depth:>9.2f} {offset:>8.2f}   "
            f"{point[0]:8.2f} {point[1]:8.2f} {point[2]:8.2f}{flag}"
        )

    lines.append("")
    for pair in result["pairs"]:
        lines.append(f"  point {pair['from']} -> {pair['to']} : "
                     f"{pair['distance_mm']:.3f} mm")
    return lines


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def draw_marker(canvas: np.ndarray, panel: "Panel", point: Sequence[float],
                colour, index: int, origin: int = 0) -> None:
    """A thin crosshair with a gap at its centre, so the exact pixel stays visible.

    A filled marker or a circle hides the very pixel you are trying to place, which
    is the one that decides the measurement.
    """
    x, y = panel.to_screen(point)
    x += origin
    if not (origin - 40 <= x <= origin + panel.width + 40 and -40 <= y <= panel.height + 40):
        return
    arm, gap = 18, 4
    cv2.line(canvas, (x - arm, y), (x - gap, y), colour, 1, cv2.LINE_AA)
    cv2.line(canvas, (x + gap, y), (x + arm, y), colour, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y - arm), (x, y - gap), colour, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y + gap), (x, y + arm), colour, 1, cv2.LINE_AA)
    cv2.putText(canvas, str(index), (x + arm + 3, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def link_view(panel_a: "Panel", panel_b: "Panel",
              extrinsics: StereoExtrinsics, depth_m: float) -> None:
    """Point camera B's view at whatever camera A is looking at, at B's own zoom.

    A ONE-TIME snap, called only at session start and when `l` toggles link
    ON -- NOT on every zoom/pan of A. Panel_a and panel_b are otherwise fully
    independent: zooming or panning one never moves or rescales the other.
    That was tried (auto-following on every zoom, and separately also
    matching B's scale to A's) and reverted both times: recentring accuracy
    is bounded by how close `depth_m` is to the TRUE depth of whatever A is
    looking at, and a depth error produces a roughly CONSTANT pixel shift in
    B regardless of zoom (on this rig's geometry, ~14 px per mm of depth
    error at the image centre) -- so a view that keeps re-deriving itself
    from a stale depth guess on every zoom/pan just looks like unwanted
    coupling, and forcing B's zoom to follow made it worse (B's shrinking
    field of view eventually can't contain that fixed error at all). This is
    especially visible on this specific target (adjacent blocks span a real
    height range), where depth_hint is routinely stale-by-tens-of-mm for
    whatever the user has just panned to. Panel B's zoom is always left
    untouched here regardless -- only the one-time offset is set.

    That depth is a VIEW aid only -- it moves the window, never a measurement.
    Every reported length still comes from triangulating two real clicks, so a
    wrong depth here costs you a slightly off-centre view and nothing else.
    """
    centre_a = panel_a.offset + np.array(
        [panel_a.width, panel_a.height], dtype=np.float64) / (2.0 * panel_a.scale)
    ray = np.linalg.inv(extrinsics.camera_matrix_a) @ np.array(
        [centre_a[0], centre_a[1], 1.0])
    in_b = extrinsics.R @ (ray * depth_m) + np.asarray(extrinsics.T).reshape(3)
    if in_b[2] <= 1e-6:
        return
    projected = extrinsics.camera_matrix_b @ in_b
    centre_b = projected[:2] / projected[2]

    panel_b.offset = centre_b - np.array(
        [panel_b.width, panel_b.height], dtype=np.float64) / (2.0 * panel_b.scale)


def centre_on_ray(panel_b: "Panel", point_a: np.ndarray,
                  extrinsics: StereoExtrinsics, depth_m: float) -> None:
    """Aim camera B's view at where a specific point of A is expected to appear.

    Recentres only -- panel_b keeps its own zoom, independent of panel_a's.
    """
    ray = np.linalg.inv(extrinsics.camera_matrix_a) @ np.array(
        [point_a[0], point_a[1], 1.0])
    in_b = extrinsics.R @ (ray * depth_m) + np.asarray(extrinsics.T).reshape(3)
    if in_b[2] <= 1e-6:
        return
    projected = extrinsics.camera_matrix_b @ in_b
    centre = projected[:2] / projected[2]
    panel_b.offset = centre - np.array(
        [panel_b.width, panel_b.height], dtype=np.float64) / (2.0 * panel_b.scale)


def draw_epipolar(canvas: np.ndarray, panel: "Panel", line: np.ndarray,
                  origin: int) -> None:
    """Draw an (a, b, c) line across whatever part of the panel is visible."""
    a, b, c = line
    left = panel.offset[0]
    right = panel.offset[0] + panel.width / panel.scale
    top = panel.offset[1]
    bottom = panel.offset[1] + panel.height / panel.scale
    if abs(b) > abs(a):
        ends = [(left, -(a * left + c) / b), (right, -(a * right + c) / b)]
    else:
        ends = [(-(b * top + c) / a, top), (-(b * bottom + c) / a, bottom)]
    points = []
    for point in ends:
        x, y = panel.to_screen(point)
        points.append((x + origin, y))
    cv2.line(canvas, points[0], points[1], COLOR_EPIPOLAR, 1, cv2.LINE_AA)


def loupe(image: np.ndarray, centre: Sequence[float], zoom: int) -> np.ndarray:
    """Magnified crop around a full-resolution coordinate, with a crosshair."""
    half = max(4, LOUPE_SIZE // (2 * zoom))
    x, y = int(round(centre[0])), int(round(centre[1]))
    height, width = image.shape[:2]
    x0 = int(np.clip(x - half, 0, max(0, width - 2 * half)))
    y0 = int(np.clip(y - half, 0, max(0, height - 2 * half)))
    crop = image[y0:y0 + 2 * half, x0:x0 + 2 * half]
    if crop.size == 0:
        return np.zeros((LOUPE_SIZE, LOUPE_SIZE, 3), np.uint8)
    view = cv2.resize(crop, (LOUPE_SIZE, LOUPE_SIZE), interpolation=cv2.INTER_NEAREST)
    mid = LOUPE_SIZE // 2
    cv2.line(view, (mid, 0), (mid, LOUPE_SIZE), (0, 0, 255), 1)
    cv2.line(view, (0, mid), (LOUPE_SIZE, mid), (0, 0, 255), 1)
    cv2.rectangle(view, (0, 0), (LOUPE_SIZE - 1, LOUPE_SIZE - 1), (200, 200, 200), 1)
    return view


def draw_measurements(canvas: np.ndarray, panel: "Panel", points: Sequence[Sequence[float]],
                      distances: Sequence[float], origin: int) -> None:
    """Join consecutive points and write the millimetres on the segment itself.

    The number belongs next to the thing it measures. Keeping it only in a status
    bar means you cannot tell which pair produced which figure once there are more
    than two points.
    """
    for index in range(1, len(points)):
        x0, y0 = panel.to_screen(points[index - 1])
        x1, y1 = panel.to_screen(points[index])
        cv2.line(canvas, (x0 + origin, y0), (x1 + origin, y1), COLOR_LABEL, 1, cv2.LINE_AA)
        label = f"{distances[index - 1]:.3f} mm"
        mid = ((x0 + x1) // 2 + origin, (y0 + y1) // 2 - 8)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(canvas, (mid[0] - 3, mid[1] - th - 3),
                      (mid[0] + tw + 3, mid[1] + 4), (20, 20, 20), -1)
        cv2.putText(canvas, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    COLOR_LABEL, 2, cv2.LINE_AA)


def render(state: Dict[str, Any]) -> np.ndarray:
    panel_a, panel_b = state["panel_a"], state["panel_b"]
    view_a, view_b = panel_a.render(), panel_b.render()
    canvas = np.hstack([view_a, view_b])
    split = panel_a.width
    # A and B are two independently-zoomed/panned views abutted in one window;
    # without a visible seam the boundary between them is easy to miss.
    cv2.line(canvas, (split, 0), (split, canvas.shape[0]), (140, 140, 140), 1, cv2.LINE_AA)

    consecutive = state["consecutive"]
    if len(state["clicks_a"]) >= 2:
        draw_measurements(canvas, panel_a, state["clicks_a"], consecutive, 0)
        draw_measurements(canvas, panel_b, state["clicks_b"], consecutive, split)

    for index, point in enumerate(state["clicks_a"]):
        draw_marker(canvas, panel_a, point, COLOR_A, index)
    for index, point in enumerate(state["clicks_b"]):
        draw_marker(canvas, panel_b, point, COLOR_B, index, split)

    if state["pending_a"] is not None:
        draw_marker(canvas, panel_a, state["pending_a"], COLOR_PENDING,
                    len(state["clicks_a"]))
        draw_epipolar(canvas, panel_b, state["pending_line"], split)

    cursor = state["cursor"]
    if cursor is not None:
        on_b = cursor[0] >= split
        panel = panel_b if on_b else panel_a
        full = panel.to_image((cursor[0] - (split if on_b else 0), cursor[1]))
        view = loupe(panel.image, full, state["zoom"])
        canvas[0:LOUPE_SIZE, canvas.shape[1] - LOUPE_SIZE:] = view

    status = np.full((STATUS_HEIGHT, canvas.shape[1], 3), 28, np.uint8)
    for row, text in enumerate(state["status"][:5]):
        cv2.putText(status, text, (12, 24 + row * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([canvas, status])


# --------------------------------------------------------------------------- #
# Interactive session
# --------------------------------------------------------------------------- #

def build_status_lines(
    state: Dict[str, Any], on_status: Optional[Callable[[], List[str]]],
    touch: bool = False,
) -> List[str]:
    """Build the window's bottom status-strip lines for one frame.

    Extracted from run_interactive's main loop so it's testable with a
    synthetic state dict, without a real window -- same reasoning as
    handle_interactive_key. Text-entry mode always shows the same 2-line
    prompt/buffer display regardless of on_status: that's generic
    input-mechanism chrome a caller's on_status has no reason to reimplement.
    Outside text-entry mode, on_status (when supplied) replaces the default
    4-line generic display wholesale -- a caller with its own session
    semantics (e.g. check_depth_accuracy.py's PlaneCollectionSession) owns
    the whole status strip instead of measure_points.py's own point-by-point
    narration, which is meaningless once clicks no longer describe a single
    running length measurement.
    """
    if state["text_mode"]:
        return [
            f"{state['text_prompt']}{state['text_buffer']}_",
            "type digits and , then Enter to confirm, Esc to cancel",
        ]
    if on_status is not None and not touch:
        return on_status()
    placed = len(state["clicks_a"])
    if touch:
        head = (f"point {placed}: tap the SAME feature in camera B, then Place"
                if state["pending_a"] is not None
                else f"point {placed}: tap a feature in camera A, then Place")
        if on_status is not None:
            return [head] + on_status()
    else:
        head = (f"point {placed}: click the SAME feature in camera B, near the blue line"
                if state["pending_a"] is not None
                else f"point {placed}: click a feature in camera A")
    recent = "   ".join(f"{i}->{i+1}: {d:.3f}mm"
                        for i, d in enumerate(state["consecutive"]))[-140:]
    if touch:
        return [
            head,
            f"measurements: {recent}" if recent else "measurements: (need two points)",
            "drag = pan   Zoom +/- = zoom   arrows nudge the crosshair   Place = record the point",
            "Undo | Reset | Lock rim (wound depth) | Finish = save and show the result",
        ]
    return [
        head,
        f"measurements: {recent}" if recent else "measurements: (need two points)",
        ("wheel = zoom (fully independent)   right-drag = pan (fully independent)   "
         f"0 = fit both   l = aim B at each new A-click "
         f"{'ON' if state['linked'] else 'OFF'}"),
        "u undo | r reset | n advance/label | q or Esc = finish and print the report",
    ]

def handle_interactive_key(
    state: Dict[str, Any], key: int, extrinsics: StereoExtrinsics,
    on_undo: Optional[Callable[[int], None]],
    on_advance: Optional[Callable[[], Optional[str]]],
    on_text_submit: Optional[Callable[[str], Optional[str]]],
    recompute: Callable[[], None],
) -> bool:
    """Handle one already-``& 0xFF``-masked key code. Returns True if the
    session should end (q/Esc outside text-entry mode).

    Extracted from run_interactive's main loop so the key-handling state
    machine -- including the text-entry mode -- is testable with a synthetic
    state dict and key codes, without a real window.
    """
    if state["text_mode"]:
        if key in (13, 10):
            if on_text_submit is not None:
                error = on_text_submit(state["text_buffer"])
                if error is None:
                    state["text_mode"] = False
                    state["text_prompt"] = ""
                    state["text_buffer"] = ""
                else:
                    print(f"  {error}")
                    state["text_buffer"] = ""
        elif key == 27:
            state["text_mode"] = False
            state["text_prompt"] = ""
            state["text_buffer"] = ""
        elif key in (8, 127):
            state["text_buffer"] = state["text_buffer"][:-1]
        elif 48 <= key <= 57 or key == ord(","):
            state["text_buffer"] += chr(key)
        return False

    if key in (ord("q"), ord("Q"), 27):
        return True
    if key == ord("u"):
        if state["pending_a"] is not None:
            state["pending_a"] = None
        elif state["clicks_b"]:
            state["clicks_a"].pop(); state["clicks_b"].pop()
            state["click_offsets_px"].pop()
            recompute()
            if on_undo is not None:
                on_undo(len(state["clicks_a"]))
    elif key == ord("r"):
        state["clicks_a"].clear(); state["clicks_b"].clear()
        state["click_offsets_px"].clear()
        state["pending_a"] = None; recompute()
        if on_undo is not None:
            on_undo(0)
    elif key == ord("n"):
        if state["pending_a"] is not None:
            print("  finish or cancel the current click (click in camera B, or press u) before pressing n")
        elif on_advance is not None:
            prompt = on_advance()
            if prompt is not None:
                state["text_mode"] = True
                state["text_prompt"] = prompt
                state["text_buffer"] = ""
    elif key == ord("0"):
        state["panel_a"].fit(); state["panel_b"].fit()
    elif key == ord("l"):
        state["linked"] = not state["linked"]
        if state["linked"]:
            link_view(state["panel_a"], state["panel_b"], extrinsics, state["depth_hint"])
    elif key in (ord("+"), ord("=")):
        state["zoom"] = min(32, state["zoom"] * 2)
    elif key == ord("-"):
        state["zoom"] = max(2, state["zoom"] // 2)
    return False

def run_interactive(image_a: np.ndarray, image_b: np.ndarray,
                    extrinsics: StereoExtrinsics, depth_range: Tuple[float, float],
                    zoom: int, max_window: Tuple[int, int], blob_radius: int,
                    depth_m: float,
                    on_point: Optional[Callable[[int, Dict[str, Any]], None]] = None,
                    on_undo: Optional[Callable[[int], None]] = None,
                    on_advance: Optional[Callable[[], Optional[str]]] = None,
                    on_text_submit: Optional[Callable[[str], Optional[str]]] = None,
                    on_status: Optional[Callable[[], List[str]]] = None,
                    touch: bool = False,
                    ) -> Dict[str, Any]:
    fundamental = fundamental_for_undistorted(
        extrinsics.camera_matrix_a, extrinsics.camera_matrix_b, extrinsics.essential,
    )
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    # Touch mode adds a button bar under the status strip; the requested window size stays
    # the total, so the canvas is still exactly 1:1 with screen pixels.
    bar_height = TOUCH_BAR_HEIGHT if touch else 0
    panel_size = (max(320, max_window[0] // 2),
                  max(320, max_window[1] - STATUS_HEIGHT - bar_height))

    state: Dict[str, Any] = {
        "panel_a": Panel(image_a, panel_size), "panel_b": Panel(image_b, panel_size),
        "zoom": zoom, "clicks_a": [], "clicks_b": [], "pending_a": None,
        "pending_line": None, "cursor": None, "status": [], "consecutive": [],
        "click_offsets_px": [],
        # Modal text entry (row,col labels etc.), driven entirely through this
        # same key-handling loop -- NEVER through input(), which would block
        # cv2's event pump and freeze the window until it gets force-killed.
        # This is not a hypothetical: it's exactly what happened running the
        # previous (input()-based) design on real hardware.
        "text_mode": False, "text_prompt": "", "text_buffer": "",
        "drag": None, "linked": True,
        # Best guess at how far away the subject is, used ONLY to aim camera B's
        # view. Starts at the configured nominal and is replaced by the depth of
        # the last point actually triangulated, which is far better: at high zoom
        # B shows only ~100 px, while a 40 mm depth error shifts the match by
        # hundreds, putting the feature off-screen.
        "depth_hint": depth_m,
    }
    link_view(state["panel_a"], state["panel_b"], extrinsics, state["depth_hint"])

    touch_controller: Optional[TouchController] = (
        TouchController(state, STATUS_HEIGHT, show_lock=on_advance is not None)
        if touch else None)

    def recompute() -> None:
        state["consecutive"] = []
        # >= 1, not >= 2: measure_points() triangulates a single point fine (its
        # "pairs" list is just empty until a 2nd point exists), and on_mouse reads
        # state["result"] unconditionally after every click, including the first,
        # to refresh depth_hint - guarding on 2 here left that read a KeyError on
        # every session's first point.
        if len(state["clicks_b"]) >= 1:
            result = measure_points(
                np.array(state["clicks_a"]), np.array(state["clicks_b"]), extrinsics,
            )
            result["epipolar_offset_px"] = list(state["click_offsets_px"])
            points = np.array(result["points_mm"])
            state["consecutive"] = [
                float(np.linalg.norm(points[i] - points[i - 1]))
                for i in range(1, len(points))
            ]
            state["result"] = result

    def on_mouse(event, x, y, flags, _param):
        if state["text_mode"]:
            return
        state["cursor"] = (x, y)
        split = state["panel_a"].width
        on_b = x >= split
        panel = state["panel_b"] if on_b else state["panel_a"]
        local = (x - (split if on_b else 0), y)

        if event == cv2.EVENT_MOUSEWHEEL:
            # Zooming one panel must never move the other -- panel_a and
            # panel_b are fully independent here, on purpose (this was tried
            # both ways: auto-following on zoom/pan looked like coupling and
            # broke down badly at high zoom, see link_view's docstring).
            panel.zoom_at(local, 1.25 if flags > 0 else 0.8)
            return
        if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_MBUTTONDOWN:
            state["drag"] = (x, y, panel)
            return
        if event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            state["drag"] = None
            return
        if state["drag"] is not None and event == cv2.EVENT_MOUSEMOVE:
            px, py, target = state["drag"]
            target.pan((x - px, y - py))
            state["drag"] = (x, y, target)
            return
        if event != cv2.EVENT_LBUTTONDOWN or y >= panel.height:
            return

        full = panel.to_image(local)
        if state["pending_a"] is None:
            if not on_b:
                snapped = snap_to_blob(gray_a, full, blob_radius)
                state["pending_a"] = snapped.tolist()
                state["pending_line"] = epipolar_lines(snapped[None, :], fundamental)[0]
                if state["linked"]:
                    # Put the match on screen in B before it is asked for.
                    centre_on_ray(state["panel_b"], snapped,
                                  extrinsics, state["depth_hint"])
        elif on_b:
            # Centroid first (the dot's true centre), then the epipolar line
            # (which discards whatever click error is left across the line) --
            # capture the distance to that line HERE, before it's discarded.
            blob = snap_to_blob(gray_b, full, blob_radius)
            raw_offset = float(np.abs(
                distance_to_line(blob[None, :], state["pending_line"][None, :])[0]
            ))
            snapped = snap_to_line(blob[None, :], state["pending_line"][None, :])[0]
            state["clicks_a"].append(state["pending_a"])
            state["clicks_b"].append(snapped.tolist())
            state["click_offsets_px"].append(raw_offset)
            state["pending_a"] = None
            recompute()
            state["depth_hint"] = float(state["result"]["depth_mm"][-1]) / 1000.0
            if state["consecutive"] and on_status is None:
                index = len(state["clicks_a"]) - 1
                print(f"  point {index - 1} -> {index} : "
                      f"{state['consecutive'][-1]:.3f} mm    "
                      f"depth {state['result']['depth_mm'][-1]:.1f} mm, "
                      f"click offset {raw_offset:.1f} px",
                      flush=True)
            if on_point is not None:
                on_point(len(state["clicks_a"]) - 1, state["result"])

    window = "measure_points"
    # AUTOSIZE, with the canvas built to fit the requested window size. A resizable
    # window scales the image it shows, and then the mouse coordinates no longer
    # match canvas pixels -- clicks land a few pixels from where you aimed, which is
    # fatal when a pixel is 0.019 mm. Fixing the canvas keeps the mapping exactly 1:1.
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(
        window, on_mouse if touch_controller is None else touch_controller.wrap(on_mouse))

    while True:
        state["status"] = build_status_lines(state, on_status, touch=touch)
        if touch_controller is not None:
            touch_controller.sync()
        frame = render(state)
        if touch_controller is not None:
            frame = touch_controller.compose(frame)
        cv2.imshow(window, frame)
        key = cv2.waitKey(20)
        # Closing the window with its X button must end the session too, or the
        # loop would spin forever on an invisible window. Some OpenCV/Qt builds
        # raise "NULL guiReceiver" here instead of returning <1 once the window
        # is gone -- both mean the same thing: stop.
        try:
            still_open = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            still_open = False
        if not still_open:
            break
        if key == -1 and touch_controller is not None:
            key = touch_controller.pop_key()
        if key == -1:
            continue
        key &= 0xFF
        if handle_interactive_key(
            state, key, extrinsics, on_undo, on_advance, on_text_submit, recompute,
        ):
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if len(state["clicks_b"]) < 2:
        return {}
    result = measure_points(
        np.array(state["clicks_a"]), np.array(state["clicks_b"]), extrinsics,
    )
    result["epipolar_offset_px"] = list(state["click_offsets_px"])
    return result


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def annotate(image_a: np.ndarray, image_b: np.ndarray, result: Dict[str, Any],
             max_window: Tuple[int, int]) -> np.ndarray:
    """Saved picture of exactly what was clicked, with the millimetres on it."""
    panel_size = (max(320, max_window[0] // 2), max(320, max_window[1]))
    panel_a, panel_b = Panel(image_a, panel_size), Panel(image_b, panel_size)
    canvas = np.hstack([panel_a.render(), panel_b.render()])
    split = panel_a.width

    points = np.array(result["points_mm"])
    consecutive = [float(np.linalg.norm(points[i] - points[i - 1]))
                   for i in range(1, len(points))]
    if len(points) >= 2:
        draw_measurements(canvas, panel_a, result["clicks_a"], consecutive, 0)
        draw_measurements(canvas, panel_b, result["clicks_b_snapped"], consecutive, split)
    for index, point in enumerate(result["clicks_a"]):
        draw_marker(canvas, panel_a, point, COLOR_A, index)
    for index, point in enumerate(result["clicks_b_snapped"]):
        draw_marker(canvas, panel_b, point, COLOR_B, index, split)
    return canvas


def save(result: Dict[str, Any], image_a: np.ndarray, image_b: np.ndarray,
         output_dir: Path, depth_range: Tuple[float, float]) -> None:
    """Write the numbers and an annotated picture of exactly what was clicked."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # The images travel with the result for annotation but must not be serialised.
    payload = {key: value for key, value in result.items()
               if not isinstance(value, np.ndarray)}
    with (output_dir / "measurements.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    cv2.imwrite(str(output_dir / "measured.jpg"),
                annotate(image_a, image_b, result, DEFAULT_MAX_WINDOW))
    report = ["POINT-TO-POINT MEASUREMENT"] + format_result(result, depth_range)
    (output_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved measurements.json / report.txt / measured.jpg to {output_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_point(text: str) -> Tuple[float, float, float, float]:
    parts = [float(value) for value in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--point wants AX,AY,BX,BY in undistorted full-res pixels, got '{text}'"
        )
    return tuple(parts)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure a real-world length between points picked in a stereo pair.",
    )
    parser.add_argument("--session", required=True,
                        help="Capture folder holding <camera-a>.jpg and <camera-b>.jpg.")
    parser.add_argument("--camera-a", default="rgb_cam1")
    parser.add_argument("--camera-b", default="rgb_cam2")
    parser.add_argument("--extrinsics", default=None,
                        help="Stereo extrinsics JSON (default: geometric_calibration."
                             "extrinsics_<a>_<b> in config, else "
                             "calibration/results/stereo_<a>_<b>/extrinsics.json).")
    parser.add_argument("--point", type=parse_point, action="append", default=None,
                        metavar="AX,AY,BX,BY",
                        help="Non-interactive: one point per flag, as its pixel in "
                             "camera A and its (approximate) pixel in camera B. The B "
                             "half is snapped to the epipolar line. Repeat for more.")
    parser.add_argument("--zoom", type=int, default=DEFAULT_LOUPE_ZOOM,
                        help="Initial loupe magnification (default: %(default)s).")
    parser.add_argument("--window", type=int, nargs=2, default=list(DEFAULT_MAX_WINDOW),
                        metavar=("W", "H"),
                        help="Window size in pixels. Keep it inside your screen: a "
                             "window larger than the display sits partly offscreen and "
                             "never receives key presses (default: %(default)s).")
    parser.add_argument("--blob-radius", type=int, default=DEFAULT_BLOB_RADIUS_PX,
                        help="Search radius when snapping a click to a dot's centroid, "
                             "in full-resolution pixels (default: %(default)s).")
    parser.add_argument("--blob-snap", action="store_true",
                        help="Move each click onto the intensity centroid of the dot "
                             "underneath it. Off by default so a click lands exactly "
                             "where you put it; worth turning on for a dot grid, where "
                             "the centroid beats the steadiest hand.")
    parser.add_argument("--out", "--output", dest="output", default=None,
                        help="Output directory (default: registration.output_dir in "
                             f"config) / <session> / {DEFAULT_OUTPUT_SUBDIR}.")
    parser.add_argument("--touch", action="store_true",
                        help="Touchscreen mode: on-screen buttons (place, undo, zoom, finish) and "
                             "tap-to-aim instead of keyboard keys and the mouse wheel.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    reg_config = config.get("registration", {}) or {}
    depth_range = reg_config.get("depth_range", [DEFAULT_DEPTH_MIN, DEFAULT_DEPTH_MAX])
    depth_range = (float(depth_range[0]), float(depth_range[1]))

    extrinsics_path = resolve_path(
        args.extrinsics or default_extrinsics_path(args.camera_a, args.camera_b)
    )
    if not extrinsics_path.exists():
        raise SystemExit(f"No stereo extrinsics at {extrinsics_path}.")
    extrinsics = StereoExtrinsics.load_json(extrinsics_path)

    session_dir = resolve_path(args.session)
    paths = [session_dir / f"{camera}.jpg" for camera in (args.camera_a, args.camera_b)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"{session_dir} is missing {', '.join(missing)}.")
    raw_a, raw_b = cv2.imread(str(paths[0])), cv2.imread(str(paths[1]))
    if raw_a is None or raw_b is None:
        raise SystemExit(f"Failed to decode images in {session_dir}.")

    # Undistort up front: the epipolar lines are then straight, and every click is
    # already in the ideal pinhole frame the triangulation assumes.
    image_a, image_b = undistort_pair(raw_a, raw_b, extrinsics)

    output_dir = resolve_path(
        args.output or reg_config.get("output_dir", "registration/results")
    ) / session_dir.name / DEFAULT_OUTPUT_SUBDIR

    if args.point:
        clicks_a = np.array([[p[0], p[1]] for p in args.point], dtype=np.float64)
        clicks_b = np.array([[p[2], p[3]] for p in args.point], dtype=np.float64)
        if len(clicks_a) < 2:
            raise SystemExit("Two points are needed for a length; pass --point twice.")
        result = measure_points(clicks_a, clicks_b, extrinsics)
        print("\n".join(format_result(result, depth_range)))
        save(result, image_a, image_b, output_dir, depth_range)
        return 0

    print(f"{session_dir.name}: click a feature in camera A, then the same feature in "
          "camera B. Each measurement is printed here as you make it; press q or Esc "
          "(or close the window) to finish and get the report.")
    result = run_interactive(
        image_a, image_b, extrinsics, depth_range, max(2, args.zoom),
        (int(args.window[0]), int(args.window[1])),
        max(3, args.blob_radius) if args.blob_snap else 0,
        float(reg_config.get("default_depth", DEFAULT_REFERENCE_DEPTH)),
        touch=args.touch,
    )
    if not result:
        print("Nothing measured.")
        return 0
    print("\n".join(format_result(result, depth_range)))
    save(result, image_a, image_b, output_dir, depth_range)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
