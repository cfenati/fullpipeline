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
PLOT_SMOOTHING_SPAN_S = 60.0
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


def smooth_series(values: list[float], window_samples: int) -> list[float]:
    if window_samples <= 1 or len(values) < 2:
        return list(values)
    half = window_samples // 2
    n = len(values)
    smoothed = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def compute_offset_stats(checkpoints: list[dict], accuracy_abs_c: float, accuracy_pct: float) -> dict:
    offsets = [c["offset_mean_c"] for c in checkpoints]
    n = len(offsets)
    mean_offset_c = sum(offsets) / n
    std_offset_c = (sum((o - mean_offset_c) ** 2 for o in offsets) / n) ** 0.5

    per_checkpoint = []
    worst = None
    for checkpoint in checkpoints:
        tolerance_c = max(accuracy_abs_c, abs(checkpoint["gun_c"]) * accuracy_pct / 100.0)
        within_spec = abs(checkpoint["offset_mean_c"]) <= tolerance_c
        entry = {"checkpoint": checkpoint, "tolerance_c": tolerance_c, "within_spec": within_spec}
        per_checkpoint.append(entry)
        if not within_spec and (
            worst is None or abs(checkpoint["offset_mean_c"]) > abs(worst["checkpoint"]["offset_mean_c"])
        ):
            worst = entry

    stats = {
        "n": n,
        "mean_offset_c": mean_offset_c,
        "std_offset_c": std_offset_c,
        "min_offset_c": min(offsets),
        "max_offset_c": max(offsets),
        "max_abs_offset_c": max(abs(o) for o in offsets),
        "per_checkpoint": per_checkpoint,
        "all_within_spec": worst is None,
        "worst": worst,
    }

    if n >= 2:
        elapsed = [c["elapsed_s"] for c in checkpoints]
        slope_c_per_s, _intercept = np.polyfit(elapsed, offsets, 1)
        slope_c_per_hour = slope_c_per_s * 3600.0
        session_span_s = elapsed[-1] - elapsed[0]
        predicted_drift_c = slope_c_per_hour * (session_span_s / 3600.0)
        stats.update(
            {
                "slope_c_per_hour": slope_c_per_hour,
                "predicted_drift_c": predicted_drift_c,
                "session_span_s": session_span_s,
                "drift_within_noise": abs(predicted_drift_c) < std_offset_c,
            }
        )

    return stats


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
) -> np.ndarray:
    annotated = display.copy()
    x, y, w, h = roi_box_scaled
    cv2.rectangle(annotated, (x, y), (x + w, y + h), ROI_COLOR, 2)

    offset_text = "n/a" if last_offset_mean_c is None else f"{last_offset_mean_c:+.2f}C"
    lines = [
        f"t={elapsed_s:6.1f}s  ROI mean={roi_mean_c:5.1f}C max={roi_max_c:5.1f}C  last offset={offset_text}",
        "[r] record gun reading   [q] quit",
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


def build_summary_text(checkpoints: list[dict], accuracy_abs_c: float, accuracy_pct: float) -> str:
    if not checkpoints:
        return "No gun readings were recorded (press 'r' during the session to record one).\n"

    stats = compute_offset_stats(checkpoints, accuracy_abs_c, accuracy_pct)
    spec_line = f"Spec (Optris datasheet): +/-{accuracy_abs_c:.2f}C or +/-{accuracy_pct:.1f}% of reading"

    if stats["n"] == 1:
        checkpoint = checkpoints[0]
        entry = stats["per_checkpoint"][0]
        verdict = "PASS" if entry["within_spec"] else "FAIL"
        lines = [
            "1 checkpoint recorded.",
            "",
            spec_line,
            "",
            f"t={checkpoint['elapsed_s']:.1f}s  camera={checkpoint['camera_mean_c']:.2f}C  "
            f"gun={checkpoint['gun_c']:.2f}C  offset={checkpoint['offset_mean_c']:+.2f}C  "
            f"(tolerance +/-{entry['tolerance_c']:.2f}C: {verdict})",
            "",
            "Only one checkpoint recorded -- accuracy assessed, drift not assessable.",
        ]
        return "\n".join(lines) + "\n"

    span_s = stats["session_span_s"]
    span_txt = f"{int(span_s // 60)}m {span_s % 60:.0f}s"

    lines = [f"Thermal accuracy check -- {stats['n']} checkpoint(s) over {span_txt}", ""]
    lines.append(spec_line)
    lines.append("")
    lines.append("Offset (camera - gun):")
    lines.append(f"  mean   {stats['mean_offset_c']:+.2f} C")
    lines.append(f"  std     {stats['std_offset_c']:.2f} C")
    lines.append(f"  range  {stats['min_offset_c']:+.2f} C .. {stats['max_offset_c']:+.2f} C")
    if stats["all_within_spec"]:
        lines.append(
            f"  -> PASS: all {stats['n']} checkpoint(s) within spec "
            f"(worst case {stats['max_abs_offset_c']:.2f}C)"
        )
    else:
        worst_checkpoint = stats["worst"]["checkpoint"]
        lines.append(
            f"  -> FAIL: checkpoint at t={worst_checkpoint['elapsed_s']:.1f}s "
            f"offset={worst_checkpoint['offset_mean_c']:+.2f}C exceeds tolerance "
            f"+/-{stats['worst']['tolerance_c']:.2f}C"
        )

    lines.append("")
    lines.append("Drift (linear fit across all checkpoints):")
    lines.append(f"  slope  {stats['slope_c_per_hour']:+.2f} C/hour")
    if stats["drift_within_noise"]:
        lines.append(
            f"  -> not distinguishable from noise (predicted drift over session: "
            f"{stats['predicted_drift_c']:+.2f}C, smaller than offset std {stats['std_offset_c']:.2f}C)"
        )
    else:
        lines.append(
            f"  -> possible real trend -- treat cautiously, only one session recorded "
            f"(predicted drift over session: {stats['predicted_drift_c']:+.2f}C)"
        )

    lines.append("")
    lines.append("All checkpoints:")
    for entry in stats["per_checkpoint"]:
        checkpoint = entry["checkpoint"]
        marker = "" if entry["within_spec"] else "  <-- FAIL"
        lines.append(
            f"  t={checkpoint['elapsed_s']:7.1f}s  "
            f"camera={checkpoint['camera_mean_c']:.2f}C  gun={checkpoint['gun_c']:.2f}C  "
            f"offset={checkpoint['offset_mean_c']:+.2f}C{marker}"
        )

    return "\n".join(lines) + "\n"


def write_summary(checkpoints: list[dict], accuracy_abs_c: float, accuracy_pct: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_summary_text(checkpoints, accuracy_abs_c, accuracy_pct), encoding="utf-8")


def write_plot(
    log_rows: list[dict],
    checkpoints: list[dict],
    accuracy_abs_c: float,
    accuracy_pct: float,
    path: Path,
) -> None:
    if not checkpoints:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    times = [row["elapsed_s"] for row in log_rows]
    means = [row["roi_mean_c"] for row in log_rows]

    checkpoint_times = [c["elapsed_s"] for c in checkpoints]
    checkpoint_gun = [c["gun_c"] for c in checkpoints]
    checkpoint_offset = [c["offset_mean_c"] for c in checkpoints]

    fig, (ax_temp, ax_offset) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]}
    )

    if len(times) >= 2:
        gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
        median_dt_s = sorted(gaps)[len(gaps) // 2] if gaps else DEFAULT_INTERVAL_S
    else:
        median_dt_s = DEFAULT_INTERVAL_S
    window_samples = max(1, round(PLOT_SMOOTHING_SPAN_S / median_dt_s))
    smoothed_means = smooth_series(means, window_samples)

    ax_temp.plot(times, means, color="tab:orange", alpha=0.25, linewidth=0.8, label="ROI mean temp (raw)")
    ax_temp.plot(
        times, smoothed_means, color="tab:orange", linewidth=1.8,
        label=f"ROI mean temp ({int(PLOT_SMOOTHING_SPAN_S)}s smoothed)",
    )
    ax_temp.scatter(checkpoint_times, checkpoint_gun, color="tab:blue", zorder=3, label="gun reading")
    if len(means) >= 5:
        low, high = np.percentile(means, [1, 99])
        margin = max(high - low, 0.2) * 0.25
        ax_temp.set_ylim(low - margin, high + margin)
    ax_temp.set_ylabel("Temperature (C)")
    ax_temp.set_title("Thermal camera accuracy / drift check")
    ax_temp.legend(loc="best")

    ax_offset.scatter(checkpoint_times, checkpoint_offset, color="tab:blue", zorder=3, label="offset (camera - gun)")
    ax_offset.axhline(0.0, color="gray", linestyle="--", linewidth=1, label="zero (gun reference)")

    stats = compute_offset_stats(checkpoints, accuracy_abs_c, accuracy_pct)
    mean_offset_c = stats["mean_offset_c"]
    std_offset_c = stats["std_offset_c"]
    ax_offset.axhline(
        mean_offset_c, color="tab:red", linewidth=1, label=f"mean offset {mean_offset_c:+.2f}C"
    )
    ax_offset.axhspan(mean_offset_c - std_offset_c, mean_offset_c + std_offset_c, color="tab:red", alpha=0.12)

    verdict = "PASS" if stats["all_within_spec"] else "FAIL"
    stats_text = (
        f"n={stats['n']}  mean={mean_offset_c:+.2f}C  std={std_offset_c:.2f}C\n"
        f"spec +/-{accuracy_abs_c:.2f}C: {verdict}"
    )
    ax_offset.text(
        0.02, 0.95, stats_text, transform=ax_offset.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax_offset.set_xlabel("Elapsed time (s)")
    ax_offset.set_ylabel("Offset (C)")
    ax_offset.legend(loc="lower right", fontsize=8)

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
    accuracy_abs_c = float(thermal_config["accuracy_abs_c"])
    accuracy_pct = float(thermal_config["accuracy_pct"])
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

            display = colorize_temperature(frame.temperature_c, scale)
            display = draw_overlay(
                display,
                scale_roi_to_display(roi, scale),
                roi_mean_c,
                roi_max_c,
                elapsed_s,
                last_offset_mean_c,
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                gun_text = input("Gun reading (C): ").strip()
                try:
                    gun_c = float(gun_text)
                except ValueError:
                    print(f"Could not parse {gun_text!r} as a number -- reading not recorded.")
                else:
                    offset_mean_c, offset_max_c = compute_offsets(roi_mean_c, roi_max_c, gun_c)
                    checkpoints.append(
                        {
                            "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
                            "elapsed_s": round(elapsed_s, 3),
                            "camera_mean_c": round(roi_mean_c, 3),
                            "camera_max_c": round(roi_max_c, 3),
                            "gun_c": gun_c,
                            "offset_mean_c": round(offset_mean_c, 3),
                            "offset_max_c": round(offset_max_c, 3),
                        }
                    )
                    last_offset_mean_c = offset_mean_c
                    print(
                        f"[record] t=+{elapsed_s:.1f}s camera={roi_mean_c:.2f}C "
                        f"gun={gun_c:.2f}C offset={offset_mean_c:+.2f}C"
                    )

            # Some OpenCV/Qt builds raise "NULL guiReceiver" here instead of
            # returning <1 once the window is closed -- both mean: stop.
            try:
                still_open = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
            except cv2.error:
                still_open = False
            if not still_open:
                break
            if duration is not None and elapsed_s >= duration:
                break

        session_dir = out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        write_log_csv(log_rows, session_dir / "log.csv")
        write_checkpoints_csv(checkpoints, session_dir / "checkpoints.csv")
        write_summary(checkpoints, accuracy_abs_c, accuracy_pct, session_dir / "summary.txt")
        write_plot(log_rows, checkpoints, accuracy_abs_c, accuracy_pct, session_dir / "plot.png")
        print(f"\nWrote session output to {session_dir}")
        print(build_summary_text(checkpoints, accuracy_abs_c, accuracy_pct))

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
