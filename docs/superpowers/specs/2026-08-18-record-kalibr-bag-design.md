# Design: `record_kalibr_bag.py`

**Date:** 2026-08-18
**Status:** Approved

## Purpose

A new standalone entry-point script that records a ROS1 `.bag` file containing
two synchronized image topics, one per RGB camera, formatted for consumption
by the [Kalibr](https://github.com/ethz-asl/kalibr) camera calibration
toolbox. This is separate from and does not replace this repo's own
`calibrate_cameras.py` / `stereo_calibrate.py`, which operate on saved JPEGs.

## Approach

Grab frames directly from hardware (reusing `cameras/rgb_camera.py`'s
`RGBCamera`, the same class `capture_pipeline.py` uses) and write them
straight into the bag via the offline `rosbag` Python API. No `roscore` and no
ROS driver nodes involved — the script itself is the only thing that needs a
ROS1 Python environment available.

## Requirements / environment

Needs ROS1's Python packages (`rospy`, `rosbag`, `sensor_msgs`) importable —
normally via `source /opt/ros/<distro>/setup.bash` before running. These are
not installed in this dev environment, so — consistent with this repo's other
hardware/vendor-SDK-dependent scripts (Optris `libirimager`, FLIR `PySpin`) —
this script cannot be executed or import-tested here. Validation is limited to
a static/syntax read-through; this will be stated explicitly rather than
claiming it runs.

## Config & CLI

Loads `config.yaml` the same way `capture_pipeline.py` does: `rgb.cam1` /
`rgb.cam2` device paths, `width` / `height` / `fps`, `controls`.

New CLI flag:

- `--output PATH` — bag file path. Default: `captures/kalibr/rgb_<timestamp>.bag`
  (under the already git-ignored `captures/`, per this repo's data-hygiene
  convention).

Recording runs until interrupted with Ctrl+C (no duration/frame-count flags).

## Topic mapping

| Camera (config.yaml) | ROS topic        |
|-----------------------|-------------------|
| `rgb.cam1`             | `/cam0/image_raw` |
| `rgb.cam2`             | `/cam1/image_raw` |

These match Kalibr's own `bagcreator` default naming, so
`kalibr_calibrate_cameras` needs no extra `--topics` flag.

## Message format

`sensor_msgs/Image`, constructed by hand (no `cv_bridge`, which is often
broken outside a fully ROS-sourced Python environment):

- `encoding = "bgr8"`
- `height`, `width` from the captured frame's shape
- `step = width * 3`
- `data = frame.tobytes()`
- `header.frame_id = "cam0"` / `"cam1"`
- `header.stamp` built from `time.time()` via `rospy.Time(secs, nsecs)`
  (avoids `rospy.Time.now()`, which can require an initialized node/roscore)

## Loop & lifecycle

1. Open both cameras (same pattern as `capture_pipeline.py`'s `open_cameras`).
2. Loop: `RGBCamera.grab_pair()` → build both `Image` messages under the same
   timestamp (keeps them synced as a pair) → `bag.write()` each to its topic.
3. Print periodic progress: frame count, elapsed time, approx. bag size.
4. On `KeyboardInterrupt`, or if `grab_pair` fails, clean up in a `finally`:
   release both cameras and close the bag so it isn't left corrupted.

## Out of scope

- No `rosbag record` wrapping of externally-published topics (frames come
  straight from hardware).
- No duration/frame-count stop conditions — Ctrl+C only.
- No `CompressedImage` variant — raw `bgr8` only.
- No changes to `config.yaml` schema — reuses the existing `rgb` section.
