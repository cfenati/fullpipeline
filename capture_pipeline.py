#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import tty
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from cameras.blackfly_camera import BlackflyCamera
from cameras.rgb_camera import RGBCamera, controls_for
from cameras.thermal_camera import ThermalCamera, ThermalFrame
from gui import CaptureGUI
from color_correction import RGBColorCorrector, load_rgb_color_corrector
from sync_metrics import (
    GrabTimings,
    SyncSummary,
    format_timings,
    grab_all_timed,
)

PROJECT_ROOT = Path(__file__).resolve().parent
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]
MAX_CONSECUTIVE_GRAB_FAILURES = 50


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    config["output_dir"] = (PROJECT_ROOT / config["output_dir"]).resolve()
    config["thermal"]["config_xml"] = (
        PROJECT_ROOT / config["thermal"]["config_xml"]
    ).resolve()
    return config


def preview_frame(
    frame: Optional[np.ndarray],
    max_width: Optional[int],
) -> Optional[np.ndarray]:
    """Downscale a frame for the GUI only; capture/save paths keep full resolution."""
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


def resolve_capture_output_dir(base_output_dir: Path, output: Optional[str]) -> Path:
    """Resolve --output to a directory under captures (or an absolute path)."""
    if not output:
        return base_output_dir

    path = Path(output).expanduser()
    if path.is_absolute():
        return path.resolve()

    if path.parts and path.parts[0] == "captures":
        return (PROJECT_ROOT / path).resolve()

    return (base_output_dir / path).resolve()


class TerminalInput:
    """Read single keypresses from the terminal (non-blocking)."""

    def __init__(self) -> None:
        self._enabled = sys.stdin.isatty()
        self._old_settings = None

    def __enter__(self) -> "TerminalInput":
        if self._enabled:
            self._old_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._enabled and self._old_settings is not None:
            termios.tcsetattr(
                sys.stdin.fileno(),
                termios.TCSADRAIN,
                self._old_settings,
            )

    def poll_key(self) -> int:
        if not self._enabled:
            return 0

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return 0

        char = sys.stdin.read(1)
        if not char:
            return 0

        return ord(char)


def warmup_cameras(
    thermal: ThermalCamera,
    warmup_seconds: float,
    cam1: Optional[RGBCamera] = None,
    cam2: Optional[RGBCamera] = None,
    gui: Optional[CaptureGUI] = None,
    warmup_rgb: bool = False,
) -> None:
    if warmup_seconds <= 0:
        return

    if gui is None:
        print(f"Warming up thermal camera for {warmup_seconds:.0f}s...")

    deadline = time.time() + warmup_seconds
    frame_count = 0

    while time.time() < deadline:
        thermal.grab()
        if warmup_rgb and cam1 is not None and cam2 is not None:
            RGBCamera.grab_pair(cam1, cam2)

        frame_count += 1
        remaining = max(0.0, deadline - time.time())

        if gui is not None:
            gui.set_warmup_progress(remaining, frame_count)
        else:
            print(
                f"\r  {remaining:.1f}s remaining ({frame_count} frames)",
                end="",
                flush=True,
            )

        time.sleep(0.1)

    if gui is None:
        print("\nWarmup complete.")
    else:
        gui.set_status("Warmup complete.")


def get_blackfly_config(config: dict) -> Optional[dict]:
    blackfly_config = config.get("blackfly", {})
    if not blackfly_config.get("enabled", False):
        return None
    return blackfly_config


def open_blackfly(
    config: dict, force_manual_gain: bool = False
) -> Optional[BlackflyCamera]:
    blackfly_config = get_blackfly_config(config)
    if blackfly_config is None:
        return None

    gain_auto = (
        False if force_manual_gain else bool(blackfly_config.get("gain_auto", False))
    )
    blackfly = BlackflyCamera(
        name=blackfly_config.get("name", "FLIR Blackfly"),
        camera_index=int(blackfly_config.get("camera_index", 0)),
        serial=blackfly_config.get("serial"),
        timeout_ms=int(blackfly_config.get("timeout_ms", 1000)),
        gain_auto=gain_auto,
        gain=blackfly_config.get("gain"),
        max_fps=blackfly_config.get("max_fps"),
        preview_max_width=blackfly_config.get("preview_max_width", 1280),
        stream_newest_only=bool(blackfly_config.get("stream_newest_only", True)),
    )
    blackfly.open()
    return blackfly


def gain_sweep_values(gain_sweep_config: dict) -> list[float]:
    """Inclusive start..stop steps (dB) from a blackfly.gain_sweep config block."""
    start = float(gain_sweep_config.get("start", 20))
    stop = float(gain_sweep_config.get("stop", 40))
    step = float(gain_sweep_config.get("step", 5))
    if step <= 0:
        raise ValueError("blackfly.gain_sweep.step must be > 0")

    values = []
    value = start
    while value <= stop + 1e-9:
        values.append(round(value, 2))
        value += step
    return values


def get_stability_config(config: dict) -> dict:
    defaults = {
        "preview_interval_ms": 66,
        "recover_rgb": True,
        "recover_settle_s": 2.0,
        "warmup_rgb": False,
        "open_rgb_before_thermal": True,
        "parallel_grab": True,
    }
    return {**defaults, **config.get("stability", {})}


def open_cameras(
    config: dict,
) -> tuple[RGBCamera, RGBCamera, ThermalCamera, Optional[BlackflyCamera]]:
    rgb_config = config["rgb"]
    stability = get_stability_config(config)
    cam1 = RGBCamera(
        device=rgb_config["cam1"],
        name="RGB Camera 1",
        width=rgb_config["width"],
        height=rgb_config["height"],
        fps=rgb_config["fps"],
        controls=controls_for(rgb_config, "cam1"),
    )
    cam2 = RGBCamera(
        device=rgb_config["cam2"],
        name="RGB Camera 2",
        width=rgb_config["width"],
        height=rgb_config["height"],
        fps=rgb_config["fps"],
        controls=controls_for(rgb_config, "cam2"),
    )
    thermal = ThermalCamera(config_xml=config["thermal"]["config_xml"])

    if stability["open_rgb_before_thermal"]:
        cam1.open()
        time.sleep(0.3)
        cam2.open()
        time.sleep(0.3)
        thermal.open()
    else:
        thermal.open()
        cam1.open()
        cam2.open()

    blackfly = open_blackfly(config)
    if blackfly is not None:
        time.sleep(0.5)
    return cam1, cam2, thermal, blackfly


def save_capture(
    output_dir: Path,
    timestamp: str,
    frame1,
    frame2,
    thermal_frame: Optional[ThermalFrame],
    cam1: RGBCamera,
    cam2: RGBCamera,
    thermal: ThermalCamera,
    sync_timings: Optional[GrabTimings] = None,
    blackfly_frame: Optional[np.ndarray] = None,
    blackfly: Optional[BlackflyCamera] = None,
    color_corrector: Optional[RGBColorCorrector] = None,
) -> Path:
    session_dir = output_dir / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    rgb1_path = session_dir / "rgb_cam1.jpg"
    rgb2_path = session_dir / "rgb_cam2.jpg"
    blackfly_path = session_dir / "flir_blackfly.jpg"
    palette_path = session_dir / "thermal_palette.png"
    thermal_temp_tiff_path = session_dir / "thermal_temperature.tiff"
    thermal_temp_npy_path = session_dir / "thermal_temperature.npy"
    metadata_path = session_dir / "metadata.json"

    save_frame1 = frame1
    save_frame2 = frame2
    if color_corrector is not None:
        save_frame1 = color_corrector.correct_cam1(frame1)
        save_frame2 = color_corrector.correct_cam2(frame2)

    cv2.imwrite(str(rgb1_path), save_frame1, JPEG_PARAMS)
    cv2.imwrite(str(rgb2_path), save_frame2, JPEG_PARAMS)
    if blackfly_frame is not None:
        cv2.imwrite(str(blackfly_path), blackfly_frame, JPEG_PARAMS)

    thermal_metadata = None
    mean_temp_c = None
    if thermal_frame is not None:
        cv2.imwrite(str(palette_path), thermal_frame.palette_bgr)

        temp_centidegrees = ThermalCamera.celsius_to_int16_centidegrees(
            thermal_frame.temperature_c
        )
        cv2.imwrite(str(thermal_temp_tiff_path), temp_centidegrees)
        np.save(thermal_temp_npy_path, thermal_frame.temperature_c)

        thermal_metadata = ThermalCamera.metadata_to_dict(thermal_frame.metadata)
        mean_temp_c = thermal_frame.mean_temp_c

    metadata = {
        "timestamp": timestamp,
        "capture_time_iso": datetime.now(timezone.utc).isoformat(),
        "rgb": {
            "cam1": cam1.info(),
            "cam2": cam2.info(),
            "color_correction": (
                color_corrector.info() if color_corrector is not None else None
            ),
        },
        "thermal": {
            **thermal.info(),
            "mean_temp_c": mean_temp_c,
            "frame_metadata": thermal_metadata,
            "temperature_encoding": {
                "tiff_dtype": "int16",
                "unit": "centidegrees_celsius",
                "decode": "temperature_c = pixel_value / 100.0",
            },
        },
        "files": {
            "rgb_cam1": rgb1_path.name,
            "rgb_cam2": rgb2_path.name,
            "flir_blackfly": blackfly_path.name if blackfly_frame is not None else None,
            "thermal_palette": palette_path.name if thermal_frame else None,
            "thermal_temperature_tiff": (
                thermal_temp_tiff_path.name if thermal_frame else None
            ),
            "thermal_temperature_npy": (
                thermal_temp_npy_path.name if thermal_frame else None
            ),
        },
    }

    if blackfly is not None:
        metadata["blackfly"] = blackfly.info()

    if sync_timings is not None:
        metadata["sync"] = sync_timings.to_dict(thermal_frame)

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"Saved capture to {session_dir}")
    return session_dir


def grab_all(
    cam1: RGBCamera,
    cam2: RGBCamera,
    thermal: ThermalCamera,
    blackfly: Optional[BlackflyCamera] = None,
    parallel: bool = True,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[ThermalFrame],
    Optional[np.ndarray],
    GrabTimings,
]:
    frame1, frame2, thermal_frame, blackfly_frame, timings = grab_all_timed(
        cam1, cam2, thermal, blackfly, parallel=parallel
    )
    return frame1, frame2, thermal_frame, blackfly_frame, timings


def sync_test(
    config_path: Path,
    samples: int,
    report_path: Optional[Path] = None,
    print_each: bool = False,
) -> int:
    print(f"Running sync test ({samples} samples)...")
    config = load_config(config_path)
    cam1 = cam2 = thermal = blackfly = None
    timings_list: list[GrabTimings] = []

    try:
        cam1, cam2, thermal, blackfly = open_cameras(config)
        warmup_cameras(
            thermal,
            config["thermal"].get("warmup_seconds", 8),
            cam1,
            cam2,
        )

        for index in range(samples):
            frame1, frame2, thermal_frame, blackfly_frame, timings = grab_all(
                cam1, cam2, thermal, blackfly
            )
            if frame1 is None or frame2 is None or thermal_frame is None:
                print(f"  sample {index + 1}/{samples}: grab failed, skipped")
                continue
            if blackfly is not None and blackfly_frame is None:
                print(f"  sample {index + 1}/{samples}: blackfly grab failed, skipped")
                continue

            timings_list.append(timings)
            if print_each:
                print(f"  sample {index + 1}/{samples}: {format_timings(timings, thermal_frame)}")

        if not timings_list:
            print("Sync test failed: no successful grabs")
            return 1

        summary = SyncSummary.from_timings(timings_list)
        print(summary.format_report())

        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as report_file:
                json.dump(summary.to_dict(), report_file, indent=2)
            print(f"Report saved to {report_path}")

        return 0
    except Exception as error:
        print(f"Sync test failed: {error}")
        return 1
    finally:
        if cam1 is not None:
            cam1.release()
        if cam2 is not None:
            cam2.release()
        if thermal is not None:
            thermal.release()
        if blackfly is not None:
            blackfly.release()


def smoke_test(config_path: Path) -> int:
    camera_count = "4-camera" if get_blackfly_config(load_config(config_path)) else "3-camera"
    print(f"Running {camera_count} smoke test...")
    config = load_config(config_path)
    cam1 = cam2 = thermal = blackfly = None

    try:
        cam1, cam2, thermal, blackfly = open_cameras(config)
        warmup_cameras(
            thermal,
            config["thermal"].get("warmup_seconds", 8),
            cam1,
            cam2,
        )
        frame1, frame2, thermal_frame, blackfly_frame, _timings = grab_all(
            cam1, cam2, thermal, blackfly
        )

        if frame1 is None or frame2 is None:
            print("Smoke test failed: could not grab RGB frames")
            return 1

        if thermal_frame is None:
            print("Smoke test failed: could not grab thermal frame")
            return 1

        if blackfly is not None and blackfly_frame is None:
            print("Smoke test failed: could not grab FLIR Blackfly frame")
            return 1

        message = (
            "Smoke test passed: "
            f"RGB1={frame1.shape}, RGB2={frame2.shape}, "
            f"thermal={thermal_frame.thermal_raw.shape}, "
            f"mean_temp={thermal_frame.mean_temp_c:.2f} C"
        )
        if blackfly_frame is not None:
            message += f", blackfly={blackfly_frame.shape}"
        print(message)
        return 0
    except Exception as error:
        print(f"Smoke test failed: {error}")
        return 1
    finally:
        if cam1 is not None:
            cam1.release()
        if cam2 is not None:
            cam2.release()
        if thermal is not None:
            thermal.release()
        if blackfly is not None:
            blackfly.release()


def run_pipeline(
    config_path: Path,
    show_preview: bool = True,
    sync_metrics: bool = False,
    sync_print: bool = False,
    output: Optional[str] = None,
) -> int:
    config = load_config(config_path)
    output_dir = resolve_capture_output_dir(config["output_dir"], output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Capture output: {output_dir}")
    stability = get_stability_config(config)

    cam1 = cam2 = thermal = blackfly = None
    gui: Optional[CaptureGUI] = None
    blackfly_enabled = get_blackfly_config(config) is not None
    color_corrector = load_rgb_color_corrector(config, PROJECT_ROOT)

    try:
        if show_preview:
            window_size = config.get("gui", {}).get("window_size", "1400x920")
            gui = CaptureGUI(
                output_dir=output_dir,
                window_size=window_size,
            )
            gui.set_blackfly_enabled(blackfly_enabled)
            gui.set_status("Opening cameras...")

        cam1, cam2, thermal, blackfly = open_cameras(config)
        warmup_seconds = config["thermal"].get("warmup_seconds", 8)
        warmup_cameras(
            thermal,
            warmup_seconds,
            cam1,
            cam2,
            gui=gui,
            warmup_rgb=stability["warmup_rgb"],
        )

        if gui is not None:
            gui.set_status("Ready — Save button, S in terminal, or Ctrl+S")
        else:
            print("Pipeline ready.")
            print("Keys: s = save | q = quit (in this terminal)")

        terminal_input: Optional[TerminalInput] = None
        if sys.stdin.isatty():
            terminal_input = TerminalInput()
            terminal_input.__enter__()

        consecutive_grab_failures = 0
        blackfly_miss_count = 0
        preview_interval_s = stability["preview_interval_ms"] / 1000.0
        recover_rgb = stability["recover_rgb"]
        recover_settle_s = stability["recover_settle_s"]
        parallel_grab = stability["parallel_grab"]

        try:
            while True:
                loop_start_s = time.perf_counter()
                if gui is not None and not gui.is_open():
                    print("GUI closed.")
                    break

                frame1, frame2, thermal_frame, blackfly_frame, grab_timings = grab_all(
                    cam1,
                    cam2,
                    thermal,
                    blackfly,
                    parallel=parallel_grab,
                )

                if gui is not None:
                    preview_max_width = config["rgb"].get("preview_max_width")
                    preview_frame1 = preview_frame(frame1, preview_max_width)
                    preview_frame2 = preview_frame(frame2, preview_max_width)
                    if color_corrector is not None:
                        preview_frame1, preview_frame2 = color_corrector.preview_frames(
                            preview_frame1, preview_frame2
                        )
                    gui.update_frames(
                        preview_frame1,
                        preview_frame2,
                        thermal_frame,
                        blackfly_frame,
                    )

                if blackfly is not None and blackfly_frame is None:
                    blackfly_miss_count += 1
                    if blackfly_miss_count in (1, 10, 30):
                        print(
                            "FLIR Blackfly grab failed "
                            f"({blackfly_miss_count} times) — "
                            "check lens cap, lighting, or if another app holds the camera"
                        )
                    if gui is not None:
                        gui.set_status("FLIR grab failed — other cameras still live")
                else:
                    blackfly_miss_count = 0

                if frame1 is None or frame2 is None:
                    consecutive_grab_failures += 1
                    if recover_rgb and consecutive_grab_failures in (1, 10, 25):
                        print("RGB grab failed, attempting camera recovery...")
                        if RGBCamera.recover_pair(cam1, cam2, recover_settle_s):
                            consecutive_grab_failures = 0
                            if gui is not None:
                                gui.set_status("RGB cameras recovered")
                            time.sleep(0.2)
                            continue
                    if consecutive_grab_failures >= MAX_CONSECUTIVE_GRAB_FAILURES:
                        print("Failed to grab from one or both RGB cameras")
                        break
                    if gui is not None:
                        gui.set_status("RGB grab failed — retrying")
                    time.sleep(0.1)
                    continue

                consecutive_grab_failures = 0

                if gui is not None:
                    should_capture = gui.consume_save_request()
                    should_quit = gui.should_quit()
                else:
                    should_capture = False
                    should_quit = False

                if terminal_input is not None:
                    key = terminal_input.poll_key()
                    if key == ord("s"):
                        should_capture = True
                    if key == ord("q"):
                        should_quit = True

                if should_capture:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_blackfly_frame = blackfly_frame
                    if (
                        blackfly is not None
                        and blackfly.preview_max_width
                        and blackfly_frame is not None
                    ):
                        full_res_frame = blackfly.grab(full_resolution=True)
                        if full_res_frame is not None:
                            save_blackfly_frame = full_res_frame
                    session_dir = save_capture(
                        output_dir,
                        timestamp,
                        frame1,
                        frame2,
                        thermal_frame,
                        cam1,
                        cam2,
                        thermal,
                        sync_timings=grab_timings if sync_metrics else None,
                        blackfly_frame=save_blackfly_frame,
                        blackfly=blackfly,
                        color_corrector=color_corrector,
                    )
                    if sync_print:
                        print(format_timings(grab_timings, thermal_frame))
                    if gui is not None:
                        gui.note_capture_saved(session_dir)

                if should_quit:
                    break

                if preview_interval_s > 0:
                    elapsed_s = time.perf_counter() - loop_start_s
                    remaining_s = preview_interval_s - elapsed_s
                    if remaining_s > 0:
                        time.sleep(remaining_s)
        finally:
            if terminal_input is not None:
                terminal_input.__exit__(None, None, None)

        return 0
    except Exception as error:
        print(f"Pipeline error: {error}")
        return 1
    finally:
        if gui is not None:
            gui.close()
        if cam1 is not None:
            cam1.release()
        if cam2 is not None:
            cam2.release()
        if thermal is not None:
            thermal.release()
        if blackfly is not None:
            blackfly.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture synchronized frames from two RGB cameras, "
            "one thermal camera, and an optional FLIR Blackfly."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Open all cameras, grab one frame, and exit",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable live preview windows",
    )
    parser.add_argument(
        "--sync-test",
        action="store_true",
        help="Run a timing benchmark and print sync statistics",
    )
    parser.add_argument(
        "--sync-samples",
        type=int,
        default=30,
        help="Number of grabs for --sync-test (default: 30)",
    )
    parser.add_argument(
        "--sync-report",
        type=Path,
        default=None,
        help="Optional JSON output path for --sync-test summary",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Subfolder under captures/ for saved sessions "
            "(e.g. --output experiment1 -> captures/experiment1/<timestamp>/). "
            "Absolute paths are also accepted."
        ),
    )
    parser.add_argument(
        "--sync-metrics",
        action="store_true",
        help="Store per-capture grab timings in metadata.json",
    )
    parser.add_argument(
        "--sync-print",
        action="store_true",
        help="Print grab timings on each save (or each sample with --sync-test)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.smoke_test:
        return smoke_test(args.config)

    if args.sync_test:
        return sync_test(
            config_path=args.config,
            samples=args.sync_samples,
            report_path=args.sync_report,
            print_each=args.sync_print,
        )

    sync_metrics = args.sync_metrics
    sync_print = args.sync_print
    if sync_print and not sync_metrics and not args.sync_test:
        sync_metrics = True

    return run_pipeline(
        config_path=args.config,
        show_preview=not args.no_preview,
        sync_metrics=sync_metrics,
        sync_print=sync_print,
        output=args.output,
    )


if __name__ == "__main__":
    sys.exit(main())
