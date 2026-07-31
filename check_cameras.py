#!/usr/bin/env python3
"""Quick health check for all cameras used by capture_pipeline.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import yaml

from cameras.blackfly_camera import BlackflyCamera
from cameras.rgb_camera import RGBCamera
from cameras.thermal_camera import ThermalCamera


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["thermal"]["config_xml"] = (PROJECT_ROOT / config["thermal"]["config_xml"]).resolve()
    return config


def check_v4l(device: str, name: str) -> bool:
    print(f"\n[{name}] {device}")
    if not Path(device).exists():
        print("  FAIL: device node does not exist")
        return False

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("  FAIL: cannot open with OpenCV/V4L2")
        return False

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("  FAIL: opened but could not read a frame")
        return False

    print(f"  OK: frame shape {frame.shape}")
    return True


def check_blackfly(config: dict) -> bool:
    blackfly_config = config.get("blackfly", {})
    if not blackfly_config.get("enabled", False):
        print("\n[FLIR Blackfly] disabled in config.yaml")
        return True

    print("\n[FLIR Blackfly] Spinnaker/PySpin")
    cam = BlackflyCamera(
        camera_index=int(blackfly_config.get("camera_index", 0)),
        serial=blackfly_config.get("serial"),
        timeout_ms=int(blackfly_config.get("timeout_ms", 1000)),
        gain_auto=bool(blackfly_config.get("gain_auto", True)),
        gain=blackfly_config.get("gain"),
        max_fps=blackfly_config.get("max_fps"),
        preview_max_width=blackfly_config.get("preview_max_width", 1280),
        stream_newest_only=bool(blackfly_config.get("stream_newest_only", True)),
    )
    try:
        cam.open()
        frame = cam.grab()
    except Exception as error:
        print(f"  FAIL: {error}")
        return False
    finally:
        cam.release()

    if frame is None:
        print("  FAIL: camera opened but returned no frame (incomplete/timeout)")
        return False

    print(f"  OK: frame shape {frame.shape}, mean pixel {frame.mean():.1f}")
    return True


def check_thermal(config: dict) -> bool:
    print("\n[Thermal] libirimager")
    thermal = ThermalCamera(config_xml=config["thermal"]["config_xml"])
    try:
        thermal.open()
        frame = thermal.grab()
    except Exception as error:
        print(f"  FAIL: {error}")
        return False
    finally:
        thermal.release()

    if frame is None:
        print("  FAIL: no thermal frame")
        return False

    print(
        "  OK: "
        f"thermal={frame.thermal_raw.shape}, "
        f"palette={frame.palette_bgr.shape}, "
        f"mean={frame.mean_temp_c:.2f} C"
    )
    return True


def print_usb_summary() -> None:
    print("\n[USB topology]")
    try:
        output = subprocess.check_output(["lsusb", "-t"], text=True)
    except Exception as error:
        print(f"  Could not run lsusb -t: {error}")
        return
    print(output.rstrip())


def main() -> int:
    config = load_config()
    rgb = config["rgb"]

    print("Camera health check")
    print("=" * 40)
    print_usb_summary()

    ok1 = check_v4l(rgb["cam1"], "RGB Camera 1")
    ok2 = check_v4l(rgb["cam2"], "RGB Camera 2")
    ok3 = check_thermal(config)
    ok4 = check_blackfly(config)

    print("\nSummary")
    print(f"  RGB 1:    {'OK' if ok1 else 'FAIL'}")
    print(f"  RGB 2:    {'OK' if ok2 else 'FAIL'}")
    print(f"  Thermal:  {'OK' if ok3 else 'FAIL'}")
    print(f"  Blackfly: {'OK' if ok4 else 'FAIL'}")

    if not (ok1 and ok2 and ok3 and ok4):
        print(
            "\nTips:"
            "\n  - RGB 'No such device' usually means USB bandwidth/hub overload."
            "\n  - Spread cameras across different USB controllers if possible."
            "\n  - Close SpinView/other apps before running the pipeline."
            "\n  - Blackfly uses Spinnaker, not /dev/video* — test with SpinView or this script."
        )
        return 1

    print("\nAll configured cameras responded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
