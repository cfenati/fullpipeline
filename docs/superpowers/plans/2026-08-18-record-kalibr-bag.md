# record_kalibr_bag.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `record_kalibr_bag.py`, a new entry-point script that records a ROS1 `.bag` file with one image topic per RGB camera, formatted for the Kalibr calibration toolbox.

**Architecture:** Single new top-level script. It opens both RGB cameras directly via the existing `cameras/rgb_camera.py` `RGBCamera` class (same as `capture_pipeline.py`), grabs synchronized frame pairs, and writes them straight into a `.bag` file using the offline `rosbag` Python API — no `roscore`, no ROS driver nodes. ROS imports (`rospy`, `rosbag`, `sensor_msgs`) are guarded (`try`/`except ImportError`, following the existing `PySpin` pattern in `cameras/blackfly_camera.py`) so the config/path-resolution logic stays importable and testable in this dev environment even though ROS1 itself isn't installed here.

**Tech Stack:** Python 3.9, `rospy`/`rosbag`/`sensor_msgs` (ROS1, external to this repo's `requirements.txt` — must be sourced from a ROS1 install), `numpy`, `PyYAML`, this repo's `cameras.rgb_camera.RGBCamera`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-18-record-kalibr-bag-design.md` (read this first).
- `from __future__ import annotations` + type hints on function signatures (project convention).
- No test suite or linter exists in this repo (per `CLAUDE.md`) — do not add pytest or a `tests/` directory. Verify via direct `python3 -c` execution of hardware/ROS-free code paths, `python3 -m py_compile` for syntax, and explicit manual read-through for anything that needs real cameras or a sourced ROS1 environment. State plainly when something could not be executed.
- Topics: `rgb.cam1` → `/cam0/image_raw` (frame_id `cam0`), `rgb.cam2` → `/cam1/image_raw` (frame_id `cam1`) — Kalibr's own default naming.
- Message format: `sensor_msgs/Image`, `encoding="bgr8"`, built by hand (no `cv_bridge`).
- Default bag path: `captures/kalibr/rgb_<timestamp>.bag` (under the git-ignored `captures/` dir); override via `--output`.
- Recording runs until Ctrl+C — no duration/frame-count flags.
- `config.yaml` schema is unchanged — reuse the existing `rgb` section (`cam1`, `cam2`, `width`, `height`, `fps`, `controls`).

---

### Task 1: Script skeleton — guarded ROS import, config/path helpers, CLI

**Files:**
- Create: `record_kalibr_bag.py`

**Interfaces:**
- Produces: `PROJECT_ROOT: Path`, `load_config() -> dict`, `resolve_output_path(output: Optional[str]) -> Path`, `_require_ros() -> None` (raises `RuntimeError` if ROS1 isn't importable), module-level `rospy`/`rosbag`/`Image` (each `None` if ROS1 isn't installed).

- [ ] **Step 1: Write `record_kalibr_bag.py` with imports, guard, and helpers**

```python
#!/usr/bin/env python3
"""Record a ROS1 bag with one image topic per RGB camera, for Kalibr calibration.

Grabs frames directly from both RGB cameras (the same RGBCamera class used by
capture_pipeline.py) and writes them straight into a .bag file via the
offline rosbag API - no roscore or ROS driver nodes required. Requires a
ROS1 Python environment (rospy, rosbag, sensor_msgs) on sys.path, e.g.:

    source /opt/ros/noetic/setup.bash
    python3 record_kalibr_bag.py

Recording runs until interrupted with Ctrl+C.

Example:
    python3 record_kalibr_bag.py --output captures/kalibr/rgb_run1.bag
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import yaml

from cameras.rgb_camera import RGBCamera, controls_for

try:
    import rosbag
    import rospy
    from sensor_msgs.msg import Image
except ImportError:
    rosbag = None
    rospy = None
    Image = None

PROJECT_ROOT = Path(__file__).resolve().parent
PROGRESS_INTERVAL_FRAMES = 30
DROPPED_FRAME_RETRY_DELAY_S = 0.05

TOPIC_CAM1 = "/cam0/image_raw"
TOPIC_CAM2 = "/cam1/image_raw"
FRAME_ID_CAM1 = "cam0"
FRAME_ID_CAM2 = "cam1"


def _require_ros() -> None:
    if rospy is None or rosbag is None or Image is None:
        raise RuntimeError(
            "rospy/rosbag/sensor_msgs are not importable. Source your ROS1 "
            "environment first, e.g. `source /opt/ros/noetic/setup.bash`."
        )


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def resolve_output_path(output: Optional[str]) -> Path:
    """Resolve --output to a bag path, defaulting to captures/kalibr/rgb_<timestamp>.bag."""
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "captures" / "kalibr" / f"rgb_{timestamp}.bag"

    path = Path(output).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


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
    _require_ros()

    print(f"Resolved bag output path: {bag_path}")
    print("ROS1 environment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify `load_config()` and `resolve_output_path()` directly (no ROS/hardware needed)**

Run:

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
import record_kalibr_bag as m

config = m.load_config()
assert 'rgb' in config
assert config['rgb']['cam1'] and config['rgb']['cam2']
print('load_config OK:', sorted(config['rgb'].keys()))

default_path = m.resolve_output_path(None)
assert default_path.parent == m.PROJECT_ROOT / 'captures' / 'kalibr'
assert default_path.suffix == '.bag'

relative_path = m.resolve_output_path('captures/kalibr/custom.bag')
assert relative_path == (m.PROJECT_ROOT / 'captures' / 'kalibr' / 'custom.bag').resolve()

absolute_path = m.resolve_output_path('/tmp/custom.bag')
assert absolute_path == Path('/tmp/custom.bag').resolve()

print('resolve_output_path OK')
"
```

Expected: prints `load_config OK: [...]` and `resolve_output_path OK`, no assertion errors. This exercises real code paths — no ROS or camera hardware involved, so this genuinely passes or fails (not a hardware-gated skip).

- [ ] **Step 3: Verify `--help` and the ROS guard**

Run: `python3 record_kalibr_bag.py --help`
Expected: argparse usage text, exit code 0 (never reaches `_require_ros()`).

Run: `python3 record_kalibr_bag.py --output /tmp/probe.bag`
Expected (on this machine, where ROS1 is not installed): prints `Resolved bag output path: /tmp/probe.bag`, then raises `RuntimeError: rospy/rosbag/sensor_msgs are not importable...`. This is the correct, intended behavior here — it confirms the guard fires with a clear message instead of a cryptic `AttributeError` on `None`. On a machine with ROS1 sourced, this same command would instead print `ROS1 environment OK` and return 0.

- [ ] **Step 4: Commit**

```bash
git add record_kalibr_bag.py
git commit -m "$(cat <<'EOF'
Add record_kalibr_bag.py skeleton: config loading, output path resolution, guarded ROS1 import

Config/path logic is real and verified directly; camera capture and
bag writing land in a follow-up commit since they need ROS1 + hardware
this dev environment doesn't have.
EOF
)"
```

---

### Task 2: Camera lifecycle, ROS message construction, and recording loop

**Files:**
- Modify: `record_kalibr_bag.py` (add functions; replace `main()`'s body from Task 1)

**Interfaces:**
- Consumes: `PROJECT_ROOT`, `load_config()`, `resolve_output_path()`, `_require_ros()`, `TOPIC_CAM1`/`TOPIC_CAM2`, `FRAME_ID_CAM1`/`FRAME_ID_CAM2`, `PROGRESS_INTERVAL_FRAMES`, `DROPPED_FRAME_RETRY_DELAY_S` from Task 1. `RGBCamera` / `controls_for` from `cameras/rgb_camera.py` (`RGBCamera(device, name, width, height, fps, controls)`, `.open()`, `.release()`, staticmethod `.grab_pair(cam1, cam2) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]`).
- Produces: `open_cameras(rgb_config: dict) -> Tuple[RGBCamera, RGBCamera]`, `image_message(frame: np.ndarray, frame_id: str, stamp: Any) -> Any`, `stamp_now() -> Any`, `record(cam1: RGBCamera, cam2: RGBCamera, bag_path: Path) -> int`.

- [ ] **Step 1: Add camera lifecycle, message-building, and recording loop functions**

Insert these functions after `resolve_output_path()` (before `main()`):

```python
def open_cameras(rgb_config: dict) -> Tuple[RGBCamera, RGBCamera]:
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
    cam1.open()
    time.sleep(0.3)
    cam2.open()
    return cam1, cam2


def image_message(frame: np.ndarray, frame_id: str, stamp: Any) -> Any:
    """Build a sensor_msgs/Image from a BGR frame without cv_bridge."""
    height, width = frame.shape[:2]
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = height
    msg.width = width
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = width * 3
    msg.data = np.ascontiguousarray(frame).tobytes()
    return msg


def stamp_now() -> Any:
    """A rospy.Time built from the wall clock, without needing an initialized node."""
    now = time.time()
    secs = int(now)
    nsecs = int((now - secs) * 1e9)
    return rospy.Time(secs, nsecs)


def record(cam1: RGBCamera, cam2: RGBCamera, bag_path: Path) -> int:
    """Grab frame pairs and write them to bag_path until interrupted. Returns frame count."""
    bag_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    start = time.time()
    bag = rosbag.Bag(str(bag_path), "w")
    try:
        print(f"Recording to {bag_path} - press Ctrl+C to stop")
        while True:
            frame1, frame2 = RGBCamera.grab_pair(cam1, cam2)
            if frame1 is None or frame2 is None:
                print("Warning: dropped a frame pair, retrying")
                time.sleep(DROPPED_FRAME_RETRY_DELAY_S)
                continue

            stamp = stamp_now()
            bag.write(TOPIC_CAM1, image_message(frame1, FRAME_ID_CAM1, stamp), t=stamp)
            bag.write(TOPIC_CAM2, image_message(frame2, FRAME_ID_CAM2, stamp), t=stamp)
            frame_count += 1

            if frame_count % PROGRESS_INTERVAL_FRAMES == 0:
                elapsed = time.time() - start
                size_mb = bag_path.stat().st_size / (1024 * 1024)
                print(f"{frame_count} pairs, {elapsed:.1f}s elapsed, {size_mb:.1f} MB")
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C)")
    finally:
        bag.close()

    return frame_count
```

- [ ] **Step 2: Replace `main()`'s body to wire in camera opening and recording**

Replace the existing `main()` (from Task 1) with:

```python
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
    _require_ros()

    config = load_config()
    cam1, cam2 = open_cameras(config["rgb"])
    try:
        frame_count = record(cam1, cam2, bag_path)
    finally:
        cam1.release()
        cam2.release()

    print(f"Wrote {frame_count} frame pairs to {bag_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Remove the old `if __name__ == "__main__":` block from Task 1's version so there's exactly one at the end of the file.

- [ ] **Step 3: Syntax-check the full file**

Run: `python3 -m py_compile record_kalibr_bag.py`
Expected: no output, exit code 0.

- [ ] **Step 4: Re-run Task 1's config/path verification as a regression check**

Run the same `python3 -c "..."` snippet from Task 1 Step 2 again.
Expected: identical output (`load_config OK: [...]`, `resolve_output_path OK`) — confirms adding the camera/ROS code didn't change the still-testable code paths.

Also run: `python3 record_kalibr_bag.py --help`
Expected: usage text, exit code 0.

- [ ] **Step 5: State what could not be verified here**

This dev environment has no ROS1 install and no attached camera hardware, so `open_cameras()`, `image_message()`, `stamp_now()`, `record()`, and the full `main()` recording flow cannot be executed or import-tested here (`import record_kalibr_bag` still succeeds because of the Task 1 guard, but calling `record()` would immediately fail on `rosbag.Bag(...)` since `rosbag is None`). Confirm this explicitly rather than claiming the recording path works — say so in the task report. Verification for this task is: successful `py_compile`, the regression check above, and a manual read-through of `record()`/`image_message()`/`open_cameras()` against `docs/superpowers/specs/2026-08-18-record-kalibr-bag-design.md`'s "Message format" and "Loop & lifecycle" sections, confirming topic names, frame_id values, encoding, and cleanup order (release both cameras, close the bag) all match.

- [ ] **Step 6: Commit**

```bash
git add record_kalibr_bag.py
git commit -m "$(cat <<'EOF'
Add camera capture and ROS bag recording loop to record_kalibr_bag.py

Completes the script: opens both RGB cameras, grabs synced frame
pairs, and writes them to /cam0/image_raw and /cam1/image_raw as
sensor_msgs/Image until Ctrl+C. Needs a sourced ROS1 environment and
attached cameras to run - not executable end-to-end in this dev
environment, verified via py_compile and read-through against the
design spec instead.
EOF
)"
```
