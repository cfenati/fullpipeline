# Design: `record_kalibr_bag.py`

> **Retired.** `record_kalibr_bag.py` was deleted in `ff8c992` ("retire line-accuracy/Kalibr tooling"); the script no longer exists. Kept as a design record only.

**Date:** 2026-08-18
**Status:** Approved, then retired (see note above)

## Purpose

A new standalone entry-point script that records two mp4 from my cameras containing
two synchronized image topics, one per RGB camera, formatted for consumption
by the [Kalibr](https://github.com/ethz-asl/kalibr) camera calibration
toolbox. This is separate from and does not replace this repo's own
`calibrate_cameras.py` / `stereo_calibrate.py`, which operate on saved JPEGs.

## Approach

Grab frames directly from hardware (reusing `cameras/rgb_camera.py`'s
`RGBCamera`, the same class `capture_pipeline.py` uses) and write them
straight into a mp4 format. 

## Config & CLI

Loads `config.yaml` the same way `capture_pipeline.py` does: `rgb.cam1` /
`rgb.cam2` device paths, `width` / `height` / `fps`, `controls`.

New CLI flag:

- `--output PATH` — bag file path. Default: `captures/kalibr/rgb_<timestamp>.bag`
  (under the already git-ignored `captures/`, per this repo's data-hygiene
  convention).

Recording runs until interrupted with Ctrl+C (no duration/frame-count flags).


## Loop & lifecycle

1. Open both cameras (same pattern as `capture_pipeline.py`'s `open_cameras`).
2. Loop: `RGBCamera.grab_pair()` → build both `Image` messages under the same
   timestamp (keeps them synced as a pair) → `bag.write()` each to its topic.
3. Print periodic progress: frame count, elapsed time, approx. bag size.
4. On `KeyboardInterrupt`, or if `grab_pair` fails, clean up in a `finally`:
   release both cameras and close the bag so it isn't left corrupted.

## Out of scope
save two mp4 files from cameras rgb1 and 2 that are syncronized and that will next feed the kalibr pipeline