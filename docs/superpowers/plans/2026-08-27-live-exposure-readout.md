# Live Exposure/Vignetting Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `check_exposure_live.py`, a live side-by-side preview of both RGB cameras with a 5-region (center + 4 corners) brightness/color readout overlay, so the user can physically match the two lenses' aperture/focus rings while watching numbers update.

**Architecture:** Single standalone script. Opens both `RGBCamera`s (existing `cameras/rgb_camera.py`) at a reduced resolution using `config.yaml`'s existing, unmodified controls. A loop grabs synchronized pairs via `RGBCamera.grab_pair`, draws fixed-position region boxes on every frame (smooth video), and recomputes the numeric overlay (reusing `RegionStats`/`_region_stats` from `check_color.py`) on a throttle so digits stay readable. Displayed via a single `cv2.imshow` window; `q` or closing the window exits.

**Tech Stack:** Python 3.9, OpenCV (`cv2`), NumPy, PyYAML — all already used elsewhere in this repo. No new dependencies.

## Global Constraints

- `from __future__ import annotations` + type hints on function signatures (repo convention).
- Config loaded via a local `load_config()` reading `config.yaml`, paths resolved against `PROJECT_ROOT = Path(__file__).resolve().parent` (repo convention).
- Script is an argparse CLI with `--help` documenting flags (repo convention).
- No test suite/linter exists in this repo. Verification is via import/syntax checks and hardware-free logic checks with synthetic data — per `CLAUDE.md`, camera-hardware behavior cannot be exercised without the physical rig, and that must be stated explicitly rather than claimed as working.
- Camera controls come from `controls_for(rgb_config, "cam1"/"cam2")` in `config.yaml`, applied completely unmodified — no `auto_exposure`/`gain` overrides of any kind (confirmed with user: rings are adjusted physically, camera settings stay fixed).
- No file output, no config.yaml writes, no flat-field generation — diagnostic-only tool.

---

### Task 1: `check_exposure_live.py`

**Files:**
- Create: `check_exposure_live.py`

**Interfaces:**
- Consumes: `cameras.rgb_camera.RGBCamera` (`__init__(device, name, width, height, fps, controls)`, `.open()`, `.grab_pair(cam1, cam2) -> (Optional[np.ndarray], Optional[np.ndarray])` staticmethod, `.release()`), `cameras.rgb_camera.controls_for(rgb_config: dict, key: str) -> dict`, `check_color.RegionStats` (dataclass with `.name`, `.b_mean`, `.g_mean`, `.r_mean`, `.gray_mean`, `.rg_ratio`, `.bg_ratio`), `check_color._region_stats(name: str, region: np.ndarray) -> RegionStats`.
- Produces: `sample_regions(frame: np.ndarray, margin_fraction: float = 0.12) -> list[tuple[RegionStats, tuple[int, int, int, int]]]`, `build_display(frame1, frame2, regions1, regions2) -> np.ndarray`, `run(width: int, height: int, fps: int, interval: float) -> int`, `main() -> int`. Nothing downstream depends on these (single-task plan) but names are fixed here for the verification steps below.

- [ ] **Step 1: Write the script**

```python
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
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
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
        default=0.25,
        help="Seconds between numeric-readout refreshes; the video itself updates every frame (default: 0.25).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args.width, args.height, args.fps, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify imports and syntax**

Run: `python -c "import check_exposure_live"`
Expected: no output, exit code 0. (Confirms `check_color`/`cameras.rgb_camera` imports resolve and there are no syntax errors — does not touch hardware.)

- [ ] **Step 3: Verify region sampling and display compositing with synthetic frames**

Run:
```bash
python -c "
import numpy as np
from check_exposure_live import sample_regions, build_display

frame1 = np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8)
frame2 = np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8)
regions1 = sample_regions(frame1)
regions2 = sample_regions(frame2)
assert [name for name, _ in [(s.name, b) for s, b in regions1]] == ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
display = build_display(frame1, frame2, regions1, regions2)
assert display.dtype == np.uint8
assert display.shape[1] == frame1.shape[1] + frame2.shape[1]
assert display.shape[0] > frame1.shape[0]
print('OK', display.shape)
"
```
Expected: prints `OK` followed by the composed image shape, exit code 0. This exercises the entire non-hardware code path (region math, box drawing, panel/summary rendering) without a camera.

- [ ] **Step 4: Verify the CLI surface**

Run: `python check_exposure_live.py --help`
Expected: argparse help text listing `--width`, `--height`, `--fps`, `--interval` with the defaults above, exit code 0.

- [ ] **Step 5: Note hardware-dependent behavior explicitly**

`run()` (camera open, the `grab_pair` loop, and the `cv2.imshow` window) talks to physical cameras and cannot be exercised in this environment — per `CLAUDE.md`, state this explicitly rather than claim it was verified. The command to hand the user for their own hardware test is:

```bash
python check_exposure_live.py
```

- [ ] **Step 6: Commit**

Only if the user asks for a commit at this point — this repo's convention (and standing instructions) is not to commit without an explicit request. If asked:

```bash
git add check_exposure_live.py docs/superpowers/specs/2026-08-27-live-exposure-readout-design.md docs/superpowers/plans/2026-08-27-live-exposure-readout.md
git commit -m "$(cat <<'EOF'
Add live exposure/vignetting readout tool for matching camera aperture rings

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
