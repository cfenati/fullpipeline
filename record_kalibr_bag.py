#!/usr/bin/env python3
"""Record a ROS1 bag with one image topic per RGB camera, for Kalibr calibration.

Grabs frames directly from both RGB cameras (the same RGBCamera class used by
capture_pipeline.py) and writes them straight into a .bag file using the
rosbags library (pure-Python ROS1 bag reader/writer, no ROS1 install
required - see requirements.txt).

Shows a live preview window per camera (downscaled to rgb.preview_max_width,
same as capture_pipeline.py's GUI stream). Recording runs until interrupted
with Ctrl+C, or by pressing 'q'/Esc in a preview window.

Example:
    python3 record_kalibr_bag.py --output captures/kalibr/rgb_run1.bag
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml
from rosbags.rosbag1 import Writer, WriterError
from rosbags.typesys import Stores, get_typestore

from cameras.rgb_camera import RGBCamera, controls_for

TYPESTORE = get_typestore(Stores.ROS1_NOETIC)
Image = TYPESTORE.types["sensor_msgs/msg/Image"]
Header = TYPESTORE.types["std_msgs/msg/Header"]
Time = TYPESTORE.types["builtin_interfaces/msg/Time"]

PROJECT_ROOT = Path(__file__).resolve().parent
PROGRESS_INTERVAL_FRAMES = 30
DROPPED_FRAME_RETRY_DELAY_S = 0.05
MAX_CONSECUTIVE_GRAB_FAILURES = 50
RECOVER_SETTLE_S = 2.0

TOPIC_CAM1 = "/cam0/image_raw"
TOPIC_CAM2 = "/cam1/image_raw"
FRAME_ID_CAM1 = "cam0"
FRAME_ID_CAM2 = "cam1"


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def resolve_output_path(output: Optional[str]) -> Path:
    """Resolve --output to a bag path, defaulting to captures/kalibr/rgb_<timestamp>.bag.

    A relative path is rooted under captures/kalibr/ (matching this repo's
    git-ignored data-hygiene convention) unless it already starts with
    "captures". A path with no suffix gets ".bag" appended.
    """
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = PROJECT_ROOT / "captures" / "kalibr" / f"rgb_{timestamp}.bag"
    else:
        path = Path(output).expanduser()
        if path.is_absolute():
            path = path.resolve()
        elif path.parts and path.parts[0] == "captures":
            path = (PROJECT_ROOT / path).resolve()
        else:
            path = (PROJECT_ROOT / "captures" / "kalibr" / path).resolve()

    if not path.suffix:
        path = path.with_suffix(".bag")
    return path


def open_cameras(rgb_config: dict) -> Tuple[RGBCamera, RGBCamera]:
    controls1 = controls_for(rgb_config, "cam1")
    controls2 = controls_for(rgb_config, "cam2")
    if not controls1 and not controls2:
        print(
            "Warning: config.yaml has no rgb.controls configured - recording "
            "with each camera's auto-exposure/auto-white-balance defaults, "
            "which can drift between cameras and hurt calibration quality."
        )
    cam1 = RGBCamera(
        device=rgb_config["cam1"],
        name="RGB Camera 1",
        width=rgb_config["width"],
        height=rgb_config["height"],
        fps=rgb_config["fps"],
        controls=controls1,
    )
    cam2 = RGBCamera(
        device=rgb_config["cam2"],
        name="RGB Camera 2",
        width=rgb_config["width"],
        height=rgb_config["height"],
        fps=rgb_config["fps"],
        controls=controls2,
    )
    cam1.open()
    time.sleep(0.3)
    cam2.open()
    return cam1, cam2


def _print_rate_estimate(cam1: RGBCamera, cam2: RGBCamera) -> None:
    pair_bytes = (cam1.width * cam1.height + cam2.width * cam2.height) * 3
    fps = min(cam1.fps, cam2.fps) or 0
    pair_mb = pair_bytes / (1024 * 1024)
    rate_mb_s = pair_mb * fps
    print(
        f"Estimated ~{pair_mb:.0f} MB per frame pair at ~{fps:.1f} fps "
        f"(~{rate_mb_s:.0f} MB/s). Lower rgb.width/height/fps in config.yaml "
        "first if that's impractical for this recording."
    )


def preview_frame(frame: np.ndarray, max_width: Optional[int]) -> np.ndarray:
    """Downscale a frame for the live preview only; the bag keeps full resolution."""
    if frame is None or not max_width:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    return cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def image_message(frame: np.ndarray, frame_id: str, stamp: Time) -> Image:
    """Build a sensor_msgs/Image from a BGR frame."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel BGR frame, got shape {frame.shape}")
    height, width = frame.shape[:2]
    return Image(
        header=Header(seq=0, stamp=stamp, frame_id=frame_id),
        height=height,
        width=width,
        encoding="bgr8",
        is_bigendian=0,
        step=width * 3,
        data=np.ascontiguousarray(frame).reshape(-1),
    )


def stamp_now() -> Time:
    """A builtin_interfaces/Time built from the wall clock.

    Sampled after both frames are retrieved, so it trails actual exposure by
    the MJPG decode time of both frames - fine for camera-to-camera sync
    (both topics share this one stamp), but would matter for camera-IMU sync.
    """
    now = time.time()
    sec = int(now)
    nanosec = int((now - sec) * 1e9)
    return Time(sec=sec, nanosec=nanosec)


def record(
    cam1: RGBCamera,
    cam2: RGBCamera,
    bag_path: Path,
    preview_max_width: Optional[int] = None,
) -> int:
    """Grab frame pairs, show a live preview, and write them to bag_path until interrupted.

    Returns the number of frame pairs written.
    """
    bag_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    consecutive_failures = 0
    start = time.time()
    with Writer(bag_path) as writer:
        conn1 = writer.add_connection(TOPIC_CAM1, Image.__msgtype__, typestore=TYPESTORE)
        conn2 = writer.add_connection(TOPIC_CAM2, Image.__msgtype__, typestore=TYPESTORE)

        print(f"Recording to {bag_path} - press Ctrl+C or 'q' in a preview window to stop")
        try:
            while True:
                frame1, frame2 = RGBCamera.grab_pair(cam1, cam2)
                if frame1 is None or frame2 is None:
                    consecutive_failures += 1
                    if consecutive_failures in (1, 10, 25):
                        print("RGB grab failed, attempting camera recovery...")
                        if RGBCamera.recover_pair(cam1, cam2, RECOVER_SETTLE_S):
                            consecutive_failures = 0
                            continue
                    if consecutive_failures >= MAX_CONSECUTIVE_GRAB_FAILURES:
                        print("Failed to grab from one or both RGB cameras, stopping")
                        break
                    time.sleep(DROPPED_FRAME_RETRY_DELAY_S)
                    continue

                consecutive_failures = 0
                stamp = stamp_now()
                timestamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
                msg1 = image_message(frame1, FRAME_ID_CAM1, stamp)
                msg2 = image_message(frame2, FRAME_ID_CAM2, stamp)
                writer.write(conn1, timestamp_ns, bytes(TYPESTORE.serialize_ros1(msg1, Image.__msgtype__)))
                writer.write(conn2, timestamp_ns, bytes(TYPESTORE.serialize_ros1(msg2, Image.__msgtype__)))
                frame_count += 1

                cv2.imshow(cam1.name, preview_frame(frame1, preview_max_width))
                cv2.imshow(cam2.name, preview_frame(frame2, preview_max_width))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("Stopping ('q' pressed)")
                    break

                if frame_count % PROGRESS_INTERVAL_FRAMES == 0:
                    elapsed = time.time() - start
                    size_mb = bag_path.stat().st_size / (1024 * 1024)
                    print(f"{frame_count} pairs, {elapsed:.1f}s elapsed, {size_mb:.1f} MB")
        except KeyboardInterrupt:
            print("\nStopping (Ctrl+C)")
        finally:
            cv2.destroyAllWindows()

    return frame_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record a ROS1 bag with one image topic per RGB camera, for Kalibr."
    )
    parser.add_argument(
        "--output",
        help="Bag file path. Defaults to captures/kalibr/rgb_<timestamp>.bag",
    )
    args = parser.parse_args()

    bag_path = resolve_output_path(args.output)

    try:
        config = load_config()
        cam1, cam2 = open_cameras(config["rgb"])
        _print_rate_estimate(cam1, cam2)
        try:
            frame_count = record(
                cam1, cam2, bag_path, config["rgb"].get("preview_max_width")
            )
        finally:
            cam1.release()
            cam2.release()
    except (RuntimeError, WriterError) as error:
        print(f"Error: {error}")
        return 1

    print(f"Wrote {frame_count} frame pairs to {bag_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
