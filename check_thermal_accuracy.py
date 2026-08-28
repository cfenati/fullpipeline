#!/usr/bin/env python3
"""Compare the Optris thermal camera's reported temperature for a hot
plate against an independent ground truth (a handheld temperature gun),
logged over time to also check for calibration drift within a session."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cameras.thermal_camera import ThermalCamera  # noqa: E402

WINDOW_NAME = "Thermal accuracy check (r = record gun reading, q = quit)"
DEFAULT_SCALE = 4
DEFAULT_INTERVAL_S = 10.0
DEFAULT_OUT_DIR = PROJECT_ROOT / "thermal_reports"
OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0, 255, 0)
ROI_COLOR = (0, 255, 0)

LOG_CSV_FIELDS = ["timestamp_iso", "elapsed_s", "roi_mean_c", "roi_max_c"]
CHECKPOINT_CSV_FIELDS = [
    "timestamp_iso", "elapsed_s", "camera_mean_c", "camera_max_c",
    "gun_c", "offset_mean_c", "offset_max_c",
]


def load_thermal_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    thermal = config["thermal"]
    thermal["config_xml"] = (PROJECT_ROOT / thermal["config_xml"]).resolve()
    return thermal


def parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError(f"--roi expects X,Y,W,H, got: {value!r}")
    x, y, w, h = (int(p) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError(f"--roi width/height must be positive, got: {value!r}")
    return x, y, w, h


def scale_roi_to_raw(
    box: tuple[int, int, int, int],
    scale: int,
    raw_width: int,
    raw_height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    raw_x = max(0, min(raw_width - 1, x // scale))
    raw_y = max(0, min(raw_height - 1, y // scale))
    raw_w = max(1, min(raw_width - raw_x, w // scale))
    raw_h = max(1, min(raw_height - raw_y, h // scale))
    return raw_x, raw_y, raw_w, raw_h


def scale_roi_to_display(roi: tuple[int, int, int, int], scale: int) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    return x * scale, y * scale, w * scale, h * scale


def roi_stats(temperature_c: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = roi
    region = temperature_c[y : y + h, x : x + w]
    return float(region.mean()), float(region.max())


def compute_offsets(camera_mean_c: float, camera_max_c: float, gun_c: float) -> tuple[float, float]:
    return camera_mean_c - gun_c, camera_max_c - gun_c


def handle_key(
    state: dict,
    key: int,
    roi_mean_c: float,
    roi_max_c: float,
    elapsed_s: float,
) -> Optional[dict]:
    """Handle one already-`& 0xFF`-masked key code for the live loop's modal
    gun-reading text entry, mutating `state` in place.

    `roi_mean_c`/`roi_max_c`/`elapsed_s` are this frame's camera reading.
    The instant 'r' is pressed they're frozen into `state["pending"]` --
    typing the gun's value can take a few seconds, during which the camera
    keeps grabbing frames, so the checkpoint must use the reading from the
    moment the user decided to take it, not whatever the live frame is once
    they finish typing.

    Returns a checkpoint dict (elapsed_s, camera_mean_c, camera_max_c,
    gun_c) once Enter submits a parseable number, else None. On an
    unparseable buffer, stays in text-entry mode (clearing the buffer) so
    the user can just retype without pressing 'r' again. Sets
    state["quit"] = True on 'q' outside text-entry mode; 'q' is swallowed
    (not a quit request) while text-entry mode is active.

    Extracted as a pure function -- no camera, no cv2 window -- so this
    state machine is testable with synthetic state/key inputs. Mirrors
    measure_points.py's handle_interactive_key, written after a prior
    input()-based design froze the cv2 window on real hardware (see
    measure_points.py:525-529): input() blocks on terminal stdin, which
    requires the terminal (not the cv2 window) to have OS focus and stops
    cv2's event loop from being pumped while it waits.
    """
    if state["text_mode"]:
        if key in (13, 10):
            text = state["text_buffer"]
            try:
                gun_c = float(text)
            except ValueError:
                print(f"Could not parse {text!r} as a number -- try again.")
                state["text_buffer"] = ""
                return None
            state["text_mode"] = False
            state["text_buffer"] = ""
            pending = state["pending"]
            return {
                "elapsed_s": pending["elapsed_s"],
                "camera_mean_c": pending["roi_mean_c"],
                "camera_max_c": pending["roi_max_c"],
                "gun_c": gun_c,
            }
        if key == 27:
            state["text_mode"] = False
            state["text_buffer"] = ""
        elif key in (8, 127):
            state["text_buffer"] = state["text_buffer"][:-1]
        elif 48 <= key <= 57 or key == ord("."):
            state["text_buffer"] += chr(key)
        return None

    if key == ord("q"):
        state["quit"] = True
    elif key == ord("r"):
        state["text_mode"] = True
        state["text_buffer"] = ""
        state["pending"] = {
            "elapsed_s": elapsed_s,
            "roi_mean_c": roi_mean_c,
            "roi_max_c": roi_max_c,
        }
    return None


def colorize_temperature(temperature_c: np.ndarray, scale: int) -> np.ndarray:
    t_min = float(temperature_c.min())
    t_max = float(temperature_c.max())
    span = max(t_max - t_min, 1e-6)
    normalized = ((temperature_c - t_min) / span * 255.0).astype(np.uint8)
    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    height, width = colorized.shape[:2]
    return cv2.resize(
        colorized, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST
    )


def draw_overlay(
    display: np.ndarray,
    roi_box_scaled: tuple[int, int, int, int],
    roi_mean_c: float,
    roi_max_c: float,
    elapsed_s: float,
    last_offset_mean_c: Optional[float],
    text_entry: Optional[str] = None,
) -> np.ndarray:
    annotated = display.copy()
    x, y, w, h = roi_box_scaled
    cv2.rectangle(annotated, (x, y), (x + w, y + h), ROI_COLOR, 2)

    offset_text = "n/a" if last_offset_mean_c is None else f"{last_offset_mean_c:+.2f}C"
    second_line = text_entry if text_entry is not None else "[r] record gun reading   [q] quit"
    lines = [
        f"t={elapsed_s:6.1f}s  ROI mean={roi_mean_c:5.1f}C max={roi_max_c:5.1f}C  last offset={offset_text}",
        second_line,
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            annotated, line, (8, 20 + index * 20), OVERLAY_FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA
        )
    return annotated


def write_log_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_checkpoints_csv(checkpoints: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHECKPOINT_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(checkpoints)


def build_summary_text(checkpoints: list[dict]) -> str:
    if not checkpoints:
        return "No gun readings were recorded (press 'r' during the session to record one).\n"

    lines = [f"{len(checkpoints)} checkpoint(s) recorded.", ""]
    first = checkpoints[0]
    lines.append(
        f"First checkpoint: t={first['elapsed_s']:.1f}s  "
        f"camera={first['camera_mean_c']:.2f}C  gun={first['gun_c']:.2f}C  "
        f"offset={first['offset_mean_c']:+.2f}C"
    )

    if len(checkpoints) == 1:
        lines.append("")
        lines.append("Only one checkpoint recorded -- accuracy assessed, drift not assessable.")
        return "\n".join(lines) + "\n"

    last = checkpoints[-1]
    lines.append(
        f"Last checkpoint:  t={last['elapsed_s']:.1f}s  "
        f"camera={last['camera_mean_c']:.2f}C  gun={last['gun_c']:.2f}C  "
        f"offset={last['offset_mean_c']:+.2f}C"
    )

    drift_c = last["offset_mean_c"] - first["offset_mean_c"]
    elapsed_span_s = last["elapsed_s"] - first["elapsed_s"]
    lines.append("")
    lines.append(f"Drift (offset_mean_c, last - first): {drift_c:+.2f}C over {elapsed_span_s:.1f}s")
    if elapsed_span_s > 0:
        drift_rate_c_per_hour = drift_c / elapsed_span_s * 3600.0
        lines.append(f"Drift rate: {drift_rate_c_per_hour:+.2f} C/hour")

    lines.append("")
    lines.append("All checkpoints:")
    for checkpoint in checkpoints:
        lines.append(
            f"  t={checkpoint['elapsed_s']:7.1f}s  "
            f"camera={checkpoint['camera_mean_c']:.2f}C  gun={checkpoint['gun_c']:.2f}C  "
            f"offset={checkpoint['offset_mean_c']:+.2f}C"
        )

    return "\n".join(lines) + "\n"


def write_summary(checkpoints: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_summary_text(checkpoints), encoding="utf-8")


def write_plot(log_rows: list[dict], checkpoints: list[dict], path: Path) -> None:
    if not checkpoints:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    times = [row["elapsed_s"] for row in log_rows]
    means = [row["roi_mean_c"] for row in log_rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times, means, color="tab:orange", label="ROI mean temp (camera)")

    checkpoint_times = [c["elapsed_s"] for c in checkpoints]
    checkpoint_means = [c["camera_mean_c"] for c in checkpoints]
    ax.scatter(checkpoint_times, checkpoint_means, color="tab:blue", zorder=3, label="gun checkpoint")
    for checkpoint in checkpoints:
        ax.annotate(
            f"{checkpoint['offset_mean_c']:+.2f}C",
            (checkpoint["elapsed_s"], checkpoint["camera_mean_c"]),
            textcoords="offset points",
            xytext=(6, 6),
        )

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Thermal camera accuracy / drift check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def open_thermal_camera(thermal_config: dict) -> ThermalCamera:
    camera = ThermalCamera(config_xml=thermal_config["config_xml"])
    camera.open()
    return camera


def select_roi_interactive(camera: ThermalCamera, scale: int) -> tuple[int, int, int, int]:
    frame = None
    while frame is None:
        frame = camera.grab()

    display = colorize_temperature(frame.temperature_c, scale)
    box = cv2.selectROI(WINDOW_NAME, display, showCrosshair=True)
    cv2.destroyWindow(WINDOW_NAME)
    if box[2] <= 0 or box[3] <= 0:
        raise RuntimeError("No ROI selected -- drag a box over the hot plate, or pass --roi.")

    raw_height, raw_width = frame.temperature_c.shape
    return scale_roi_to_raw(box, scale, raw_width, raw_height)


def run(
    roi: Optional[tuple[int, int, int, int]],
    interval: float,
    duration: Optional[float],
    scale: int,
    out_dir: Path,
) -> int:
    thermal_config = load_thermal_config()
    camera = None
    try:
        try:
            camera = open_thermal_camera(thermal_config)
        except (RuntimeError, FileNotFoundError) as error:
            print(f"Could not open thermal camera: {error}")
            return 1

        warmup_s = float(thermal_config.get("warmup_seconds", 1))
        warmup_end = time.monotonic() + warmup_s
        while time.monotonic() < warmup_end:
            camera.grab()

        if roi is None:
            roi = select_roi_interactive(camera, scale)
        print(f"ROI (raw thermal pixels): {roi}")
        print("Live readout running. Press 'r' to record a gun reading, 'q' to quit.")

        log_rows: list[dict] = []
        checkpoints: list[dict] = []
        last_offset_mean_c: Optional[float] = None

        key_state = {"text_mode": False, "text_buffer": "", "pending": None, "quit": False}

        start_time = time.monotonic()
        last_log_elapsed_s = -interval  # force a sample on the first iteration

        while True:
            frame = camera.grab()
            if frame is None:
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            elapsed_s = time.monotonic() - start_time
            roi_mean_c, roi_max_c = roi_stats(frame.temperature_c, roi)

            if elapsed_s - last_log_elapsed_s >= interval:
                log_rows.append(
                    {
                        "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
                        "elapsed_s": round(elapsed_s, 3),
                        "roi_mean_c": round(roi_mean_c, 3),
                        "roi_max_c": round(roi_max_c, 3),
                    }
                )
                last_log_elapsed_s = elapsed_s

            text_entry = None
            if key_state["text_mode"]:
                text_entry = (
                    f"Gun reading (C): {key_state['text_buffer']}_   "
                    "[Enter=submit, Esc=cancel]"
                )

            display = colorize_temperature(frame.temperature_c, scale)
            display = draw_overlay(
                display,
                scale_roi_to_display(roi, scale),
                roi_mean_c,
                roi_max_c,
                elapsed_s,
                last_offset_mean_c,
                text_entry=text_entry,
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            result = handle_key(key_state, key, roi_mean_c, roi_max_c, elapsed_s)
            if result is not None:
                gun_c = result["gun_c"]
                offset_mean_c, offset_max_c = compute_offsets(
                    result["camera_mean_c"], result["camera_max_c"], gun_c
                )
                checkpoints.append(
                    {
                        "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
                        "elapsed_s": round(result["elapsed_s"], 3),
                        "camera_mean_c": round(result["camera_mean_c"], 3),
                        "camera_max_c": round(result["camera_max_c"], 3),
                        "gun_c": gun_c,
                        "offset_mean_c": round(offset_mean_c, 3),
                        "offset_max_c": round(offset_max_c, 3),
                    }
                )
                last_offset_mean_c = offset_mean_c
                print(
                    f"[record] t=+{result['elapsed_s']:.1f}s camera={result['camera_mean_c']:.2f}C "
                    f"gun={gun_c:.2f}C offset={offset_mean_c:+.2f}C"
                )

            if key_state["quit"]:
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if duration is not None and elapsed_s >= duration:
                break

        session_dir = out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        write_log_csv(log_rows, session_dir / "log.csv")
        write_checkpoints_csv(checkpoints, session_dir / "checkpoints.csv")
        write_summary(checkpoints, session_dir / "summary.txt")
        write_plot(log_rows, checkpoints, session_dir / "plot.png")
        print(f"\nWrote session output to {session_dir}")
        print(build_summary_text(checkpoints))

        return 0
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the thermal camera's reported temperature for a hot plate "
            "against a handheld temperature gun, logged over time to check drift."
        ),
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Fixed ROI as X,Y,W,H in raw thermal-pixel coordinates. Skips interactive selection.",
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL_S,
        help=f"Seconds between logged samples (default: {DEFAULT_INTERVAL_S}).",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Stop automatically after this many seconds (default: run until q/Ctrl+C).",
    )
    parser.add_argument(
        "--scale", type=int, default=DEFAULT_SCALE,
        help=f"Upscale factor for the live display (default: {DEFAULT_SCALE}).",
    )
    parser.add_argument(
        "--out", type=str, default=str(DEFAULT_OUT_DIR),
        help="Base output directory (default: thermal_reports/).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roi = parse_roi(args.roi) if args.roi else None
    out_dir = Path(args.out).expanduser().resolve()
    return run(roi, args.interval, args.duration, args.scale, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
