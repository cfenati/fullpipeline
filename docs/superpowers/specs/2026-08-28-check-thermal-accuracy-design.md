# Design: check_thermal_accuracy.py

Date: 2026-08-28

Status: **Implemented** (code verified with synthetic data and a
non-interactive smoke test against the connected camera; interactive ROI
selection and a real hot-plate/gun session are still untested — see this
plan's Task 3, Step 5).

## Purpose

Validate that the Optris thermal camera reports the right absolute
temperature for a physical hot plate, against an independent ground truth
(a handheld temperature gun), and check whether that accuracy holds or
drifts over the course of a session. No existing `check_*` script in this
repo touches thermal radiometric accuracy — `check_depth_accuracy.py` and
`check_line_accuracy.py` validate the RGB stereo pair's *geometric*
accuracy, which is an unrelated error budget (triangulation, not
radiometry).

## Data format (confirmed empirically before designing the rest of this)

`ThermalCamera.grab()` (`cameras/thermal_camera.py`) returns a `ThermalFrame`
with two independent buffers:

- `thermal_raw`: `uint16`, **not temperature** — confirmed live on the
  connected camera (382×288, values 1225–1755 pointed at a mixed
  room-temperature/warm-object scene).
- `temperature_c`: the wrapper's own decode, `thermal_raw / 10.0 - 100.0`
  (`ThermalCamera.raw_to_celsius`), applied automatically inside `grab()`.
  Confirmed sane against the same live frames: 22.5–75.5°C, stable across 5
  consecutive grabs (±0.1°C). This script reuses `temperature_c` as-is and
  does not re-derive or second-guess this decode.
- `palette_bgr`: a **separately-sized** buffer (384×288 here, vs. thermal's
  382×288 — an SDK stride-padding difference, not a scale/crop the wrapper
  corrects for). No code in this repo maps a `palette_bgr` pixel back to a
  `temperature_c` index, and this script does not attempt it — see
  "Display / ROI" below for how it avoids needing to.

This confirms the wrapper's existing Celsius decode is trustworthy; the
open question this script exists to answer is whether the *camera's
calibration* (not this repo's decode math) matches an independent
reference over time.

## Method

1. Open only `ThermalCamera` (no RGB, no Blackfly) via
   `config.yaml`'s `thermal.config_xml`, same pattern as
   `capture_pipeline.py`'s `open_cameras` but thermal-only.
2. **ROI selection.** Grab one frame, build a self-colorized preview by
   normalizing `temperature_c` and applying `cv2.applyColorMap`, upscaled
   `--scale`x (default 4) for visibility (raw sensor res is only 382×288).
   `cv2.selectROI` on that image to drag a box over the hot plate; divide
   the returned box by `--scale` to get raw `temperature_c` indices.
   Because the ROI is selected on an image *we* rendered directly from
   `temperature_c`, the mapping back is exact — no stride/palette ambiguity.
   `--roi X,Y,W,H` (raw thermal-pixel coords) skips interactive selection
   for a repeat session against the same physical setup.
3. **Live loop.** Grab continuously; redraw the colorized/upscaled view
   with the ROI box and an overlay (elapsed time, current ROI mean/max °C,
   last recorded offset) every frame. Every `--interval` seconds (default
   10 seconds) append `(timestamp, elapsed_s, roi_mean_c, roi_max_c)` to an
   in-memory log — decoupling display smoothness from log density, same
   split `check_exposure_live.py` uses for its numeric readout.
4. **Recording a gun reading.** Press `r` at any time: the loop pauses,
   prompts in the terminal for the gun's °C value (`input()`), and appends
   a checkpoint: `(timestamp, elapsed_s, camera_mean_c, camera_max_c,
   gun_c, offset_mean_c, offset_max_c)` where `offset = camera − gun`.
   Printed immediately to the terminal as confirmation. You'd press this
   once right away (accuracy) and again later in the same run (drift).
5. **Ending.** `q` / window close, or `--duration` seconds elapsed if
   given. `ThermalCamera.release()` in a `finally` block.
6. **Output.** Write `log.csv`, `checkpoints.csv`, `summary.txt`, and (if
   matplotlib is available and ≥1 checkpoint exists) `plot.png` to
   `thermal_reports/<timestamp>/`.

## Emissivity / transmissivity — deliberately out of scope

The SDK exposes `evo_irimager_set_radiation_parameters(emissivity,
transmissivity, tAmbient)` (`/usr/include/libirimager/direct_binding.h`),
but `ThermalCamera` doesn't wrap it, and this script doesn't add that
wrapper. Getting emissivity right requires either a materials lookup table
for the plate's actual surface finish or an empirical calibration against a
trusted contact reference — neither is in hand right now, so it's deferred
rather than guessed at.

Practical consequence for reading this script's output: a **large, stable**
camera-vs-gun offset that does not change over the session is more
consistent with an emissivity/material mismatch than with camera drift or
error. A **small offset that grows over time** is the actual drift signal
this script is built to catch. If results come back looking like the
former, the follow-up is wrapping `evo_irimager_set_radiation_parameters`
on `ThermalCamera` and adding `--emissivity`/`--transmissivity`/`--ambient`
flags here — not re-litigating this script's own logic.

## Interfaces

- New: `check_thermal_accuracy.py` — imports only
  `cameras.thermal_camera.ThermalCamera`; no RGB (`cameras/rgb_camera.py`)
  or Blackfly import, unlike `check_exposure_live.py`/`capture_pipeline.py`.
- Reused unmodified: `ThermalCamera.open/grab/release`, `ThermalCamera.
  raw_to_celsius` (applied inside `grab()`, not called directly), the
  `load_config()` + `config["thermal"]["config_xml"]` path-resolution
  pattern used by `check_cameras.py`/`capture_pipeline.py`.

## CLI surface

```
python check_thermal_accuracy.py [--roi X,Y,W,H] [--interval 10.0] \
    [--duration SECONDS] [--scale 4] [--out thermal_reports]
```

No `--width`/`--height`/`--fps` (thermal sensor resolution isn't
configurable the way RGB capture resolution is).

## Output

`thermal_reports/<YYYYMMDD_HHMMSS>/`:

- `log.csv` — full time series: `timestamp_iso, elapsed_s, roi_mean_c,
  roi_max_c`.
- `checkpoints.csv` — one row per gun reading: `timestamp_iso, elapsed_s,
  camera_mean_c, camera_max_c, gun_c, offset_mean_c, offset_max_c`.
- `summary.txt` — offset at first vs. last checkpoint; drift (°C) and
  drift rate (°C/hour) if ≥2 checkpoints exist; an explicit note if only
  one checkpoint was recorded (accuracy assessed, drift not assessable
  from a single point). **`offset_mean_c` is the headline number** driving
  drift — it's less sensitive to single-pixel noise than a max. `offset_
  max_c` is reported alongside as a diagnostic only (e.g. it would flag a
  hand-drawn ROI that's drifted off the plate onto a cooler edge, which a
  mean could mask); it does not drive the drift calculation.
- `plot.png` (matplotlib, only if ≥1 checkpoint) — `roi_mean_c` vs.
  elapsed time, with checkpoints marked and annotated with their offset.
- `thermal_reports/` added to `.gitignore`, matching the existing
  generated-data convention (`captures/`, `color_reports/`,
  `calibration/results/`, etc.).

Comparing accuracy across days (e.g. "did it drift since last week?") is a
manual diff of two sessions' `summary.txt`/`checkpoints.csv` — no
cross-session aggregation tool is built, to keep this script's own scope
small.

## Verified

Empirically confirmed live against the connected camera this session (see
"Data format" above): buffer shapes/dtypes, the raw→Celsius decode's
sanity, and that `ThermalCamera.open()` works over USB regardless of which
physical port it's plugged into (auto-detected by the SDK, no hardcoded
device path in `config.yaml` or `ThermalCamera`).

The ROI-selection / logging / checkpoint / output flow itself is not yet
built or run — per this repo's hardware-dependency convention, it needs a
live run against the actual hot plate and gun once implemented, which
can't be simulated the way `design_rig.py`'s geometry-only paths can.

## Out of scope

Emissivity/transmissivity/ambient-temperature control (see above);
automatic hot-plate detection (manual ROI drag only); multiple ROIs per
session; live plotting during the run (plot is generated once, after the
run ends); cross-session drift aggregation.
