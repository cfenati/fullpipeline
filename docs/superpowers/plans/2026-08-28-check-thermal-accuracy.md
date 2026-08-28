# check_thermal_accuracy.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `check_thermal_accuracy.py`, a live diagnostic that compares
the Optris thermal camera's reported temperature for a hot plate against a
handheld temperature gun, logging over time so a session also catches
calibration drift.

**Architecture:** Single new script, thermal-camera-only (no RGB, no
Blackfly). It self-colorizes `ThermalCamera`'s `temperature_c` array (never
the vendor `palette_bgr`, which is a different, stride-padded resolution)
so a user-dragged ROI box maps back to raw array indices with no ambiguity.
A live OpenCV loop logs the ROI's mean/max temperature on an interval timer
and lets the user record timestamped gun-reading "checkpoints" by pressing
a key; on exit it writes a CSV time series, a checkpoints CSV, a text
summary (first-vs-last offset, drift, drift rate), and an optional
matplotlib plot.

**Tech Stack:** Python 3.9, `opencv-contrib-python`, `numpy`, `PyYAML`,
`matplotlib>=3.7` (all already in `requirements.txt`); this repo's
`cameras.thermal_camera.ThermalCamera`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-28-check-thermal-accuracy-design.md`
  — read it first for the "why" behind every decision below.
- No test suite or linter in this repo (per `CLAUDE.md`) — pure-logic
  pieces (Tasks 1–2) are verified with throwaway `python3 -c` heredoc
  scripts against synthetic data, same precedent as
  `docs/superpowers/plans/2026-08-24-check-depth-accuracy.md`. Not
  committed.
- `from __future__ import annotations` + type hints on every signature,
  matching every other script in this repo.
- Import **only** `cameras.thermal_camera.ThermalCamera` for camera access
  — no `cameras.rgb_camera`, no `cameras.blackfly_camera`. This is a
  thermal-only tool, unlike `check_exposure_live.py`/`capture_pipeline.py`.
- Do not modify `cameras/thermal_camera.py` — `ThermalCamera` is imported
  and used unmodified (no emissivity/radiation-parameter wrapper this
  iteration; see spec's "Emissivity / transmissivity" section for why).
- `offset_mean_c` (not `offset_max_c`) is the headline number that drives
  the drift calculation in `summary.txt` — `offset_max_c` is a diagnostic
  field only, never used in arithmetic.
- The thermal sensor's raw buffer (382×288 on the camera used to write
  this plan) is genuinely different from the palette buffer (384×288) —
  never use `frame.palette_bgr` for anything ROI-related; always colorize
  `frame.temperature_c` directly.
- Everything that touches the physical camera or an OpenCV GUI window
  (Task 3) can only be smoke-tested, not fully exercised — a human
  pressing `r` and dragging a ROI box cannot be scripted. Say this
  explicitly in that task's verification rather than claiming full
  end-to-end coverage.

---

### Task 1: ROI math + colorized display (pure functions, no camera)

**Files:**
- Create: `check_thermal_accuracy.py`

**Interfaces:**
- Produces: `parse_roi(value: str) -> tuple[int, int, int, int]`;
  `scale_roi_to_raw(box: tuple[int,int,int,int], scale: int, raw_width: int, raw_height: int) -> tuple[int,int,int,int]`;
  `scale_roi_to_display(roi: tuple[int,int,int,int], scale: int) -> tuple[int,int,int,int]`;
  `roi_stats(temperature_c: np.ndarray, roi: tuple[int,int,int,int]) -> tuple[float, float]`
  (returns `(mean_c, max_c)`); `compute_offsets(camera_mean_c: float, camera_max_c: float, gun_c: float) -> tuple[float, float]`
  (returns `(offset_mean_c, offset_max_c)`, each `camera - gun`);
  `colorize_temperature(temperature_c: np.ndarray, scale: int) -> np.ndarray`
  (BGR `uint8`, shape `(H*scale, W*scale, 3)`); `draw_overlay(display: np.ndarray, roi_box_scaled: tuple[int,int,int,int], roi_mean_c: float, roi_max_c: float, elapsed_s: float, last_offset_mean_c: Optional[float]) -> np.ndarray`.

- [ ] **Step 1: Write the file header and pure ROI/offset math**

```python
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
```

- [ ] **Step 2: Write the colorized display + overlay functions (append to the same file)**

```python
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
```

- [ ] **Step 3: Verify the pure functions with a throwaway heredoc (no camera, no display needed)**

Run:
```bash
python3 - <<'EOF'
import numpy as np
import sys
sys.path.insert(0, "/home/cfenati/projects/MasterThesis/FullPipeline")
from check_thermal_accuracy import (
    parse_roi, scale_roi_to_raw, scale_roi_to_display, roi_stats,
    compute_offsets, colorize_temperature, draw_overlay,
)

assert parse_roi("10,20,30,40") == (10, 20, 30, 40)
try:
    parse_roi("1,2,3")
    assert False, "expected ValueError"
except ValueError:
    pass

# selectROI box in scale=4 display coords -> raw coords
assert scale_roi_to_raw((40, 80, 120, 160), scale=4, raw_width=382, raw_height=288) == (10, 20, 30, 40)
# clip to raw bounds when the box runs past the edge (clips both w and h here)
assert scale_roi_to_raw((1500, 1140, 40, 40), scale=4, raw_width=382, raw_height=288) == (375, 285, 7, 3)

assert scale_roi_to_display((10, 20, 30, 40), scale=4) == (40, 80, 120, 160)

temperature_c = np.full((288, 382), 20.0, dtype=np.float32)
temperature_c[20:60, 10:40] = 60.0  # hot region inside the ROI
mean_c, max_c = roi_stats(temperature_c, (10, 20, 30, 40))
assert mean_c == 60.0 and max_c == 60.0, (mean_c, max_c)

offset_mean_c, offset_max_c = compute_offsets(60.0, 62.0, 58.0)
assert offset_mean_c == 2.0 and offset_max_c == 4.0

display = colorize_temperature(temperature_c, scale=4)
assert display.shape == (288 * 4, 382 * 4, 3) and display.dtype == np.uint8

annotated = draw_overlay(display, (40, 80, 120, 160), 60.0, 62.0, 12.3, 2.0)
assert annotated.shape == display.shape

print("Task 1 checks passed.")
EOF
```
Expected: `Task 1 checks passed.` with no assertion errors.

- [ ] **Step 4: Commit**

```bash
git add check_thermal_accuracy.py
git commit -m "Add ROI math and colorized display helpers for check_thermal_accuracy.py"
```

---

### Task 2: Output writers (CSV / summary / plot)

**Files:**
- Modify: `check_thermal_accuracy.py` (append)

**Interfaces:**
- Consumes: `LOG_CSV_FIELDS`, `CHECKPOINT_CSV_FIELDS` (Task 1).
- Produces: `write_log_csv(rows: list[dict], path: Path) -> None`;
  `write_checkpoints_csv(checkpoints: list[dict], path: Path) -> None`;
  `build_summary_text(checkpoints: list[dict]) -> str`;
  `write_summary(checkpoints: list[dict], path: Path) -> None`;
  `write_plot(log_rows: list[dict], checkpoints: list[dict], path: Path) -> None`
  (no-op if `checkpoints` is empty — per spec, plot is only written when
  there's at least one checkpoint to annotate).
- Row/checkpoint dicts use exactly the keys in `LOG_CSV_FIELDS` /
  `CHECKPOINT_CSV_FIELDS` — this is what Task 3's main loop must produce.

- [ ] **Step 1: Write the CSV writers**

```python
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
```

- [ ] **Step 2: Write the summary text + writer**

```python
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
```

- [ ] **Step 3: Write the plot writer**

```python
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
```

- [ ] **Step 4: Verify with a throwaway heredoc against synthetic data**

Run:
```bash
python3 - <<'EOF'
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, "/home/cfenati/projects/MasterThesis/FullPipeline")
from check_thermal_accuracy import (
    write_log_csv, write_checkpoints_csv, build_summary_text, write_summary, write_plot,
)

log_rows = [
    {"timestamp_iso": "2026-08-28T10:00:00", "elapsed_s": 0.0, "roi_mean_c": 60.0, "roi_max_c": 61.0},
    {"timestamp_iso": "2026-08-28T10:05:00", "elapsed_s": 300.0, "roi_mean_c": 61.5, "roi_max_c": 62.0},
]
checkpoints = [
    {"timestamp_iso": "2026-08-28T10:00:00", "elapsed_s": 0.0, "camera_mean_c": 60.0,
     "camera_max_c": 61.0, "gun_c": 58.0, "offset_mean_c": 2.0, "offset_max_c": 3.0},
    {"timestamp_iso": "2026-08-28T10:05:00", "elapsed_s": 300.0, "camera_mean_c": 61.5,
     "camera_max_c": 62.0, "gun_c": 58.5, "offset_mean_c": 3.0, "offset_max_c": 3.5},
]

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    write_log_csv(log_rows, tmp_path / "log.csv")
    write_checkpoints_csv(checkpoints, tmp_path / "checkpoints.csv")
    write_summary(checkpoints, tmp_path / "summary.txt")
    write_plot(log_rows, checkpoints, tmp_path / "plot.png")

    assert (tmp_path / "log.csv").read_text().count("\n") == 3  # header + 2 rows
    assert (tmp_path / "checkpoints.csv").read_text().count("\n") == 3
    summary = (tmp_path / "summary.txt").read_text()
    assert "Drift (offset_mean_c, last - first): +1.00C over 300.0s" in summary
    assert "Drift rate: +12.00 C/hour" in summary
    assert (tmp_path / "plot.png").exists()

    # zero-checkpoint edge case: no plot, summary says so explicitly
    write_plot(log_rows, [], tmp_path / "no_plot.png")
    assert not (tmp_path / "no_plot.png").exists()
    assert "No gun readings were recorded" in build_summary_text([])

    # single-checkpoint edge case: accuracy only, no drift claim
    single_summary = build_summary_text(checkpoints[:1])
    assert "drift not assessable" in single_summary
    assert "Drift (offset_mean_c" not in single_summary

print("Task 2 checks passed.")
EOF
```
Expected: `Task 2 checks passed.` with no assertion errors.

- [ ] **Step 5: Commit**

```bash
git add check_thermal_accuracy.py
git commit -m "Add CSV/summary/plot output writers for check_thermal_accuracy.py"
```

---

### Task 3: Camera integration, live loop, CLI

**Files:**
- Modify: `check_thermal_accuracy.py` (append + entry point)

**Interfaces:**
- Consumes: everything from Tasks 1–2 (`load_thermal_config`,
  `colorize_temperature`, `draw_overlay`, `scale_roi_to_raw`,
  `scale_roi_to_display`, `roi_stats`, `compute_offsets`,
  `write_log_csv`, `write_checkpoints_csv`, `write_summary`, `write_plot`,
  `build_summary_text`, `WINDOW_NAME`, `DEFAULT_SCALE`,
  `DEFAULT_INTERVAL_S`, `DEFAULT_OUT_DIR`); `ThermalCamera` from
  `cameras.thermal_camera` (`.open()`, `.grab() -> Optional[ThermalFrame]`
  with `.temperature_c: np.ndarray`, `.release()`).
- Produces: `open_thermal_camera(thermal_config: dict) -> ThermalCamera`;
  `select_roi_interactive(camera: ThermalCamera, scale: int) -> tuple[int,int,int,int]`;
  `run(roi: Optional[tuple[int,int,int,int]], interval: float, duration: Optional[float], scale: int, out_dir: Path) -> int`;
  `parse_args() -> argparse.Namespace`; `main() -> int`.

- [ ] **Step 1: Write camera open + interactive ROI selection**

```python
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
```

- [ ] **Step 2: Write the main loop**

```python
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
```

- [ ] **Step 3: Write the CLI and entry point**

```python
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
```

- [ ] **Step 4: Import check**

Run: `python3 -c "import check_thermal_accuracy"` (from
`/home/cfenati/projects/MasterThesis/FullPipeline`)
Expected: no output, exit code 0.

- [ ] **Step 5: Live smoke test against the connected camera (non-interactive path only)**

The thermal camera is connected right now, so run a short, fully
non-interactive pass using `--roi` (skipping the interactive drag) and
`--duration` to confirm the camera opens, the loop runs, and files land
correctly — including the zero-checkpoint edge case, since no keypress is
simulated:

```bash
cd /home/cfenati/projects/MasterThesis/FullPipeline
python3 check_thermal_accuracy.py --roi 100,100,50,50 --interval 2 --duration 6 \
    --out /tmp/claude-1000/thermal_smoke_test
```

Expected:
- Prints `ROI (raw thermal pixels): (100, 100, 50, 50)` and the running
  instructions.
- Exits 0 after ~6s with `Wrote session output to
  /tmp/claude-1000/thermal_smoke_test/<timestamp>/`.
- `log.csv` has a header plus ~3 rows (one every ~2s) with plausible
  `roi_mean_c`/`roi_max_c` values (double-digit °C, not raw counts in the
  thousands).
- `checkpoints.csv` has only a header row (no keys were pressed).
- `summary.txt` reads `No gun readings were recorded...`.
- `plot.png` does **not** exist (no checkpoints).

This only proves the camera-open/logging/duration-stop/file-writing path.
It does **not** exercise interactive ROI dragging or pressing `r` to
record a gun reading — those need a human at the keyboard with the OpenCV
window focused, which cannot be scripted here. Note this explicitly rather
than claiming full coverage; a real accuracy/drift session against the
actual hot plate and gun is the real end-to-end test, and only the user
can run it.

- [ ] **Step 6: Commit**

```bash
git add check_thermal_accuracy.py
git commit -m "Add camera loop and CLI for check_thermal_accuracy.py"
```

---

### Task 4: Data hygiene + docs wrap-up

**Files:**
- Modify: `.gitignore`
- Modify: `docs/superpowers/specs/2026-08-28-check-thermal-accuracy-design.md`

**Interfaces:** None (no code).

- [ ] **Step 1: Add the new output directory to `.gitignore`**

Add a line under the existing generated-data section (near `color_reports/`):

```
# Thermal accuracy/drift check output
thermal_reports/
```

- [ ] **Step 2: Update the spec's status line**

In `docs/superpowers/specs/2026-08-28-check-thermal-accuracy-design.md`,
change:

```
Status: Draft — not yet implemented. The data-format claims below are
empirically verified against the connected camera; the script itself is not
yet built.
```

to:

```
Status: **Implemented** (code verified with synthetic data and a
non-interactive smoke test against the connected camera; interactive ROI
selection and a real hot-plate/gun session are still untested — see this
plan's Task 3, Step 5).
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore docs/superpowers/specs/2026-08-28-check-thermal-accuracy-design.md
git commit -m "Ignore thermal_reports/ output and mark thermal-accuracy spec implemented"
```

## Verification summary

Tasks 1–2 are fully verified headlessly with synthetic data (no camera, no
display needed). Task 3's camera-open/warmup/logging/duration-stop/
file-writing path is verified live against the currently-connected
thermal camera. Interactive ROI dragging (`cv2.selectROI`) and the `r`-key
gun-reading prompt are **not** exercised by any automated step in this
plan — they require a human operator at the keyboard and are the first
thing to try manually once this plan is complete.

## Not done here (see spec's "Out of scope")

Emissivity/transmissivity/ambient-temperature control; automatic hot-plate
detection; multiple ROIs per session; live plotting during the run;
cross-session drift aggregation; a real hot-plate + temperature-gun
session (hardware/consumable-dependent, must be run by the user).
