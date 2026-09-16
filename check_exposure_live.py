#!/usr/bin/env python3
"""Live side-by-side RGB camera preview with a brightness/color readout,
for physically matching aperture/focus rings between rgb_cam1 and rgb_cam2."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cameras.rgb_camera import RGBCamera, controls_for
from check_color import RegionStats, _region_stats

WINDOW_NAME = "Live exposure/vignetting readout (q to quit)"
OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_COLOR = (0, 255, 0)
PANEL_BG = (20, 20, 20)
TEXT_COLOR = (0, 255, 0)
PANEL_ROW_HEIGHT = 18
SUMMARY_HEIGHT = 32
REGION_MARGIN_FRACTION = 0.12


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def sample_regions(
    frame: np.ndarray,
    margin_fraction: float = REGION_MARGIN_FRACTION,
) -> list[tuple[RegionStats, tuple[int, int, int, int]]]:
    """Center + 4 corners, as (stats, (x, y, w, h)) boxes in frame coordinates."""
    height, width = frame.shape[:2]
    margin = max(4, int(min(height, width) * margin_fraction))
    center_y, center_x = height // 2, width // 2

    boxes = {
        "center": (center_x - margin, center_y - margin, 2 * margin, 2 * margin),
        "top_left": (0, 0, margin, margin),
        "top_right": (width - margin, 0, margin, margin),
        "bottom_left": (0, height - margin, margin, margin),
        "bottom_right": (width - margin, height - margin, margin, margin),
    }

    results = []
    for name, (x, y, w, h) in boxes.items():
        region = frame[y : y + h, x : x + w]
        results.append((_region_stats(name, region), (x, y, w, h)))
    return results


def _draw_boxes(
    frame: np.ndarray,
    regions: list[tuple[RegionStats, tuple[int, int, int, int]]],
) -> np.ndarray:
    annotated = frame.copy()
    for _stats, (x, y, w, h) in regions:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), BOX_COLOR, 2)
    return annotated


def _stats_panel(
    regions: list[tuple[RegionStats, tuple[int, int, int, int]]],
    width: int,
) -> np.ndarray:
    height = PANEL_ROW_HEIGHT * len(regions) + 8
    panel = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    for index, (stats, _box) in enumerate(regions):
        text = (
            f"{stats.name:<12} gray={stats.gray_mean:6.1f} "
            f"R={stats.r_mean:6.1f} G={stats.g_mean:6.1f} B={stats.b_mean:6.1f} "
            f"R/G={stats.rg_ratio:.3f} B/G={stats.bg_ratio:.3f}"
        )
        y = 14 + index * PANEL_ROW_HEIGHT
        cv2.putText(panel, text, (6, y), OVERLAY_FONT, 0.42, TEXT_COLOR, 1, cv2.LINE_AA)
    return panel


def _summary_bar(center1: RegionStats, center2: RegionStats, width: int) -> np.ndarray:
    bar = np.full((SUMMARY_HEIGHT, width, 3), PANEL_BG, dtype=np.uint8)
    delta = center1.gray_mean - center2.gray_mean
    text = (
        f"cam1 center gray={center1.gray_mean:6.1f}   "
        f"cam2 center gray={center2.gray_mean:6.1f}   "
        f"delta={delta:+6.1f}"
    )
    cv2.putText(bar, text, (8, 22), OVERLAY_FONT, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)
    return bar


def build_display(
    frame1: np.ndarray,
    frame2: np.ndarray,
    regions1: list[tuple[RegionStats, tuple[int, int, int, int]]],
    regions2: list[tuple[RegionStats, tuple[int, int, int, int]]],
) -> np.ndarray:
    annotated1 = _draw_boxes(frame1, regions1)
    annotated2 = _draw_boxes(frame2, regions2)

    panel1 = _stats_panel(regions1, annotated1.shape[1])
    panel2 = _stats_panel(regions2, annotated2.shape[1])

    top = np.hstack([annotated1, annotated2])
    bottom = np.hstack([panel1, panel2])

    center1 = next(stats for stats, _box in regions1 if stats.name == "center")
    center2 = next(stats for stats, _box in regions2 if stats.name == "center")
    summary = _summary_bar(center1, center2, top.shape[1])

    return np.vstack([summary, top, bottom])


def open_camera(
    rgb_config: dict,
    device: str,
    name: str,
    key: str,
    width: int,
    height: int,
    fps: int,
) -> RGBCamera:
    camera = RGBCamera(
        device=device,
        name=name,
        width=width,
        height=height,
        fps=fps,
        controls=controls_for(rgb_config, key),
    )
    camera.open()
    return camera


def run(width: int, height: int, fps: int, interval: float) -> int:
    config = load_config()
    rgb_config = config["rgb"]

    cam1 = None
    cam2 = None
    try:
        try:
            cam1 = open_camera(rgb_config, rgb_config["cam1"], "rgb_cam1", "cam1", width, height, fps)
            cam2 = open_camera(rgb_config, rgb_config["cam2"], "rgb_cam2", "cam2", width, height, fps)
        except RuntimeError as error:
            print(f"Could not open cameras: {error}")
            return 1

        print("Live readout running. Press q in the preview window to quit.")

        regions1 = None
        regions2 = None
        last_update = 0.0

        while True:
            frame1, frame2 = RGBCamera.grab_pair(cam1, cam2)
            if frame1 is None or frame2 is None:
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            now = time.monotonic()
            if regions1 is None or now - last_update >= interval:
                regions1 = sample_regions(frame1)
                regions2 = sample_regions(frame2)
                last_update = now

            display = build_display(frame1, frame2, regions1, regions2)
            cv2.imshow(WINDOW_NAME, display)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            # Some OpenCV/Qt builds raise "NULL guiReceiver" here instead of
            # returning <1 once the window is closed -- both mean: stop.
            try:
                still_open = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
            except cv2.error:
                still_open = False
            if not still_open:
                break

        return 0
    finally:
        if cam1 is not None:
            cam1.release()
        if cam2 is not None:
            cam2.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live side-by-side preview of both RGB cameras with a brightness/color "
            "readout, for physically matching aperture/focus rings by eye."
        ),
    )
    parser.add_argument("--width", type=int, default=1280, help="Capture width for both cameras (default: 1280).")
    parser.add_argument("--height", type=int, default=960, help="Capture height for both cameras (default: 960).")
    parser.add_argument("--fps", type=int, default=30, help="Requested capture FPS (default: 30).")
    parser.add_argument(
        "--interval",
        type=float,
        default=10,
        help="Seconds between numeric-readout refreshes; the video itself updates every frame.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args.width, args.height, args.fps, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
