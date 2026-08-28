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
