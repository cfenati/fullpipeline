# FullPipeline

Multi-camera acquisition and calibration rig: two ELP-USB16MP01 16 MP RGB cameras,
one Optris Xi 400 LT thermal camera, and an optional FLIR Blackfly (Spinnaker).

Each stage of the workflow has exactly one command-line entry point:

| Stage | Script | What it produces |
| --- | --- | --- |
| Check the hardware | `check_cameras.py` | pass/fail per camera |
| Capture | `capture_pipeline.py` | `captures/<session>/<timestamp>/` |
| Color uniformity | `check_color.py` | uniformity plots + flat-field gain maps |
| Intrinsics | `calibrate_cameras.py` | `calibration/results/<camera>/intrinsics.json` |
| Clean up a calibration set | `prune_calibration.py` | deletes the worst capture sessions |
| Extrinsics | `stereo_calibrate.py` | `calibration/results/stereo_<a>_<b>/extrinsics.json` (+ updates `rig_as_built.yaml`) |
| Evaluate / optimize the rig | `design_rig.py` | coverage / stereo / lens studies on the measured geometry |

Everything reads defaults from `config.yaml`, so most commands run with no
arguments at all. Command-line flags always win over the config file.

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Two dependencies are not on PyPI and must be installed from the vendor:

- **Optris thermal**: `libirimager` / the Optris SDK, plus a camera XML in
  `OptrixThermalCamera/config/` (default `generic.xml`).
- **FLIR Blackfly**: Spinnaker + the `PySpin` Python bindings. On Ubuntu 24.04,
  Spinnaker needs the Ubuntu 22.04 FFmpeg libraries; `run_blackfly.sh` sets
  `LD_LIBRARY_PATH` for you.

If you only want the geometry study in `design_rig.py`, `numpy`, `matplotlib`,
`pyvista`, `opencv-python` and `PyYAML` are enough — no camera drivers needed.

---

## Typical order of operations

Calibrate first (intrinsics → extrinsics), then evaluate or optimize the
measured rig. `design_rig.py` is last: it consumes `rig_as_built.yaml` written
by stereo calibration.

```bash
python check_cameras.py                                        # 1. hardware alive?
python capture_pipeline.py --output calib_rgb1                 # 2. shoot the board
python calibrate_cameras.py --camera rgb_cam1                  # 3. intrinsics
python prune_calibration.py --camera rgb_cam1 --count 5        # 4. drop bad views (dry run)
python prune_calibration.py --camera rgb_cam1 --count 5 --apply
python calibrate_cameras.py --camera rgb_cam1                  # 5. re-fit on the clean set
python calibrate_cameras.py --camera rgb_cam2                  # 6. same for the other camera
python stereo_calibrate.py --camera-a rgb_cam1 --camera-b rgb_cam2   # 7. extrinsics (+ rig_as_built.yaml)
python design_rig.py --rig design/config/rig_as_built.yaml report     # 8. evaluate measured geometry
python design_rig.py --rig design/config/rig_as_built.yaml optimize   # 9. refine baseline / height / aim
```

---

## 1. `check_cameras.py` — hardware health check

Opens every camera named in `config.yaml`, grabs one frame from each, and prints
the USB topology. No arguments; the only input is `config.yaml`.

```bash
python check_cameras.py
```

Exit code `0` means all configured cameras responded, `1` means at least one
failed. The Blackfly is skipped when `blackfly.enabled: false`.

---

## 2. `capture_pipeline.py` — synchronized capture

Grabs both RGB cameras, the thermal camera and the Blackfly as close together as
the drivers allow, then writes one folder per trigger.

```bash
python capture_pipeline.py                                  # GUI preview, save with the button / S / Ctrl+S
python capture_pipeline.py --smoke-test                     # open everything, grab one frame, exit
python capture_pipeline.py --capture-once                    # preview, press s once, exit
python capture_pipeline.py --output calib_rgb1               # -> captures/calib_rgb1/<timestamp>/
python capture_pipeline.py --interval 5 --output timelapse   # headless, one capture every 5 s
python capture_pipeline.py --sync-test --sync-samples 50 --sync-report sync.json
python capture_pipeline.py --sync-metrics --sync-print       # record grab timings in metadata.json
```

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | path | `config.yaml` | Pipeline config to load. |
| `--output` | str | `output_dir` from config | Subfolder under `captures/` (absolute paths accepted). |
| `--smoke-test` | flag | off | Open all cameras, grab one frame, report shapes, exit. |
| `--capture-once` | flag | off | Preview until `s`, save once, exit. |
| `--interval` | float | none | Auto-capture every N seconds; implies `--no-preview`. |
| `--no-preview` | flag | off | Run headless, no GUI window. |
| `--sync-test` | flag | off | Timing benchmark instead of capturing. |
| `--sync-samples` | int | `30` | Number of grabs for `--sync-test`. |
| `--sync-report` | path | none | Write the `--sync-test` summary as JSON. |
| `--sync-metrics` | flag | off | Store per-capture grab timings in `metadata.json`. |
| `--sync-print` | flag | off | Print grab timings on every save (implies `--sync-metrics`). |

**Controls while running:** `s` saves and `q` quits, either from the GUI or by
typing in the terminal that launched the process.

**One capture session on disk:**

```
captures/<name>/20260728_151706/
├── rgb_cam1.jpg               # color-corrected if rgb.color_correction.enabled
├── rgb_cam2.jpg
├── flir_blackfly.jpg          # full resolution, only if the Blackfly is enabled
├── thermal_palette.png        # false-color preview
├── thermal_temperature.tiff   # int16 centidegrees C -> divide by 100
├── thermal_temperature.npy    # float32 degrees C
└── metadata.json              # camera settings, thermal metadata, sync timings
```

Because every camera of a trigger lands in the same folder, one capture set works
both as an intrinsics set and as a stereo set.

---

## 3. `check_color.py` — RGB uniformity and flat-field maps

Measures vignetting, border tint and channel balance, and can produce the gain
maps that `capture_pipeline.py` applies on the fly. Shoot a uniformly lit white
sheet filling the frame, then:

```bash
python check_color.py --live                                       # grab fresh frames from both cameras
python check_color.py --session captures/color_calibration/20260724_095512
python check_color.py --image white_ref.jpg --save-flat-field      # write the reusable gain maps
python check_color.py --image a.jpg --image b.jpg --no-correct     # analyze only, no correction
python check_color.py --v4l2                                       # dump the V4L2 controls
```

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--image` | path (repeatable) | none | Analyze specific image files. |
| `--session` | dir | none | Session folder holding `rgb_cam1.jpg` / `rgb_cam2.jpg`. |
| `--live` | flag | off | Grab a fresh frame from both RGB cameras via `config.yaml`. |
| `--output` | dir | `color_reports/` | Where plots, corrected images and `.npz` maps go. |
| `--border-fraction` | float | `0.12` | Border band thickness as a fraction of `min(w, h)`. |
| `--blur-sigma` | float | `0.08` | Gaussian sigma (fraction of `min(w, h)`) for the flat-field fit. |
| `--correct` / `--no-correct` | flag | on | Apply flat-field correction and save `*_corrected.jpg`. |
| `--save-flat-field` | flag | off | Also save per-channel gains as `*_flat_field.npz`. |
| `--show` | flag | off | Open the matplotlib windows interactively. |
| `--v4l2` | flag | off | Print the relevant V4L2 controls for both cameras. |

To put the result into production, copy the `.npz` files into
`color_calibration/` and point `rgb.color_correction.flat_field_cam1/cam2` at
them in `config.yaml`.

---

## 4. `calibrate_cameras.py` — intrinsics from ChArUco captures

Detects the ChArUco board in every session of a capture directory and fits
`fx, fy, cx, cy` plus distortion for one camera.

```bash
python calibrate_cameras.py --camera rgb_cam1                     # uses captures/calib_rgb1
python calibrate_cameras.py --captures captures/calib_rgb1_2 --camera rgb_cam1
python calibrate_cameras.py --camera rgb_cam2 --fix-k3            # steadier when k3 only absorbs noise
python calibrate_cameras.py --camera rgb_cam1 --rational          # 8-parameter distortion model
python calibrate_cameras.py --camera rgb_cam1 --max-view-error 1.5 --save-detections
python calibrate_cameras.py --camera rgb_cam1 --keep-all-views    # no outlier rejection
```

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--camera` | str | `rgb_cam1` | Image basename inside each session, and the output folder name. |
| `--captures` | dir | `captures/calib_<cam>` | Directory of session subfolders (`rgb_cam1` -> `captures/calib_rgb1`). |
| `--board` | yaml | `geometric_calibration.board` | ChArUco description. |
| `--output` | dir | `geometric_calibration.output_dir` | Results root; a `<camera>/` subfolder is created. |
| `--min-corners` | int | `12` | Minimum ChArUco corners for a view to count. |
| `--reject-sigma` | float | `3.0` | Drop views this many robust sigmas above the median error. |
| `--max-view-error` | float | none | Absolute per-view limit in px, replacing the robust cut-off. |
| `--keep-all-views` | flag | off | Disable outlier rejection entirely. |
| `--rational` | flag | off | 8-parameter rational distortion instead of 5-parameter. |
| `--fix-tangential` | flag | off | Force `p1 = p2 = 0`. |
| `--fix-k3` | flag | off | Force `k3 = 0`. |
| `--save-detections` | flag | off | Write an annotated copy of every accepted view. |

**Inputs:** `<captures>/<session>/<camera>.jpg` (loose `<camera>*.jpg` files in
the capture directory also work), plus the board YAML.

**Outputs** in `calibration/results/<camera>/`: `intrinsics.json`, `report.txt`
(per-view errors, rejected views, quality warnings), `coverage.png`,
`undistorted.jpg`, and `discarded/` with annotated copies of every frame where
detection failed.

The board is described by `calibration/config/charuco_11x8.yaml` — 11x8 squares,
`DICT_4X4_50`, `legacy_pattern: true` for boards generated before OpenCV 4.6. The
printable target itself is `calibration/checkerboard_pdf.pdf`. Intrinsics are
scale invariant, so `square_size_m` only matters for poses and baselines; measure
your actual print if you care about those.

---

## 5. `prune_calibration.py` — remove the worst capture sessions

Reads the per-view errors that `calibrate_cameras.py` recorded and deletes the
session folders that hurt the fit most. **Dry run by default** — nothing is
deleted without `--apply`.

```bash
python prune_calibration.py --camera rgb_cam1 --count 5                  # show what would go
python prune_calibration.py --camera rgb_cam1 --count 5 --apply          # actually delete
python prune_calibration.py --camera rgb_cam1 --max-error 2.0 --apply    # everything above 2 px
python prune_calibration.py --camera rgb_cam2 --count 3 --keep-undetected --apply
```

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--camera` | str | `rgb_cam1` | Whose calibration report to read. |
| `--count` | int | none | Delete this many worst sessions. |
| `--max-error` | float | none | Delete every session above this reprojection error in px. |
| `--captures` | dir | `captures/calib_<cam>` | Capture directory to prune. |
| `--results` | dir | `calibration/results` | Where `intrinsics.json` lives. |
| `--keep-undetected` | flag | off | Ignore sessions where the board was never found. |
| `--apply` | flag | off | Perform the deletion. |
| `--force` | flag | off | Allow leaving fewer than 8 sessions. |
| `--allow-stale` | flag | off | Prune from a report whose sessions are already partly deleted. |

At least one of `--count` or `--max-error` is required. Two guard rails will stop
you: the tool refuses to leave fewer than 8 sessions (override with `--force`),
and refuses to work from a stale report (re-run `calibrate_cameras.py` first, or
override with `--allow-stale`). Deleting a session removes **every** camera's
image for that timestamp, not just the one you named, so prune and re-calibrate
each camera in turn.

---

## 6. `stereo_calibrate.py` — extrinsics for a camera pair

Takes the sessions where both cameras saw the board, holds each camera's
intrinsics fixed, and fits the rigid transform between them. It then reports what
decides whether the pair is usable at a working distance: coverage overlap,
baseline, triangulation angle, and depth-dependent registration error.

```bash
python stereo_calibrate.py                                              # rgb_cam1 -> rgb_cam2, config defaults
python stereo_calibrate.py --captures captures/calib_rgb1 --reference-depth 0.28
python stereo_calibrate.py --depth-range 0.15 0.80 --disparity-noise 0.5
python stereo_calibrate.py --camera-a rgb_cam1 --camera-b rgb_cam2 --no-figures
```

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--camera-a` | str | `rgb_cam1` | Reference camera; all geometry is in its frame. |
| `--camera-b` | str | `rgb_cam2` | Second camera of the pair. |
| `--captures` | dir(s) | `geometric_calibration.stereo_captures` | One or more session directories. |
| `--board` | yaml | `geometric_calibration.board` | ChArUco description. |
| `--intrinsics-a` / `--intrinsics-b` | json | from config / results dir | Override either intrinsics file. |
| `--output` | dir | `geometric_calibration.output_dir` | Results root. |
| `--rig` | yaml | `design/config/rig.yaml` | Rig used to anchor the exported poses in the world frame. |
| `--min-corners` | int | `12` | Minimum corners per image. |
| `--min-shared-corners` | int | `8` | Minimum corners seen by *both* cameras. |
| `--reject-sigma` | float | `3.0` | Robust outlier cut-off in sigmas. |
| `--max-view-error` | float | none | Absolute per-view limit in px instead. |
| `--keep-all-views` | flag | off | Disable outlier rejection. |
| `--refine-intrinsics` | flag | off | Re-fit intrinsics too (not recommended). |
| `--reference-depth` | float (m) | mean observed board distance | Depth the fixed mapping is calibrated at. |
| `--depth-range` | 2 floats (m) | derived from the captures | Distances to evaluate. |
| `--depth-steps` | int | `28` | How many distances to sample. |
| `--disparity-noise` | float (px) | `0.3` | Matching error used for depth uncertainty. |
| `--grid` | int | `96` | Samples per axis on each evaluated plane. |
| `--alpha` | float | `0.0` | `stereoRectify` alpha: 0 crops to valid pixels, 1 keeps all. |
| `--no-figures` | flag | off | Skip the matplotlib figures. |

**Inputs:** sessions containing *both* `<camera-a>.jpg` and `<camera-b>.jpg`,
plus both `intrinsics.json` files. Both cameras must have been calibrated at the
capture resolution — the script refuses to rescale intrinsics across sensor
modes.

**Outputs** in `calibration/results/stereo_<a>_<b>/`: `extrinsics.json`,
`report.txt`, `rig_pose.yaml` (measured poses in rig-YAML spelling, ready to
paste over `design/config/rig.yaml`), the figures `fit_quality.png`,
`overlap_map.png`, `pair_geometry.png`, and `correspondences.jpg` /
`rectified.jpg` for checking by eye.

---

## 7. `design_rig.py` — evaluate and optimize the measured rig

Run this **after** intrinsics and stereo calibration. Pure geometry, no hardware:
it reads a rig YAML (normally `design/config/rig_as_built.yaml`, auto-updated by
`stereo_calibrate.py`) and answers what the cameras cover, what a layout change
would do, and where to move things. Every subcommand accepts the global flags
`--rig <yaml>` (default `design/config/rig.yaml`), `--out <dir>` (default
`design/out`), `--voxel <m>` and `--style frustum|pyramid`.

```bash
python design_rig.py --rig design/config/rig_as_built.yaml info
python design_rig.py --rig design/config/rig_as_built.yaml optics
python design_rig.py --rig design/config/rig_as_built.yaml view
python design_rig.py --rig design/config/rig_as_built.yaml report --render
python design_rig.py --rig design/config/rig_as_built.yaml optimize
python design_rig.py --rig design/config/rig_as_built.yaml export
```

| Subcommand | Key flags |
| --- | --- |
| `info` | — |
| `optics` | — |
| `view` | keys inside the viewer: `r` report, `e` export, `c` coverage cloud, `t` reset, `q` quit |
| `plot` | `-o/--output`, `--save`, `--multiview`, `--coverage`, `--common`, `--min-views N` |
| `report` | `--render`, `--coverage`, `--common` |
| `export` | `--no-stl` |
| `lens` | `--camera thermal`, `--focal 5 7.7 12`, `--pitch 17`, `--from-focal` |
| `optimize` (`stereo`) | `--cameras A B`, `--baseline MIN MAX` (mm), `--height MIN MAX` (m), `--elevation MIN MAX` (deg), `--aim look-at\|elevation\|hold`, `--triangulation MIN MAX`, `--steps N`, `--write PATH` |
| `sweep` | positional `param start stop`, then `--steps`, `--camera`, `--look-at-target` |

More examples:

```bash
python design_rig.py --rig design/config/rig_as_built.yaml optimize \
    --baseline 40 120 --height 0.25 0.32 --steps 7 \
    --write design/config/rig_optimized.yaml
python design_rig.py --rig design/config/rig_as_built.yaml optimize \
    --aim elevation --baseline 51.3 51.3 --height 0.276 0.276 \
    --elevation -90 -60
python design_rig.py lens --camera rgb_cam1 --focal 5 6.5 8 12
python design_rig.py sweep z 0.25 0.40 --steps 10 --camera thermal
```

`sweep` varies one degree of freedom of one camera: `x`, `y`, `z` and `far` are
in metres, `azimuth`, `elevation` and `roll` in degrees.

`design/config/rig.yaml` is the *intended* geometry (useful for what-if studies
before hardware exists). After calibration, prefer
`design/config/rig_as_built.yaml`: stereo writes measured RGB poses into it, and
`optimize` / `report` / `view` should be run against that file.

---

## 8. Blackfly-only viewer

A standalone live view with manual gain control, useful for setting exposure
without starting the whole pipeline:

```bash
./run_blackfly.sh          # sets LD_LIBRARY_PATH for Spinnaker on Ubuntu 24.04
python view_blackfly.py    # only if Spinnaker's libraries are already on the path
```

Keys: `q` quit, `]` gain up, `[` gain down, `s` save a frame to `OutputFLIR/`.

---

## `config.yaml` reference

| Key | Meaning |
| --- | --- |
| `output_dir` | Root for captures (`captures`). |
| `rgb.cam1` / `rgb.cam2` | V4L2 device paths. Use `/dev/v4l/by-path/...` so the mapping survives a reboot. |
| `rgb.width` / `height` / `fps` | Capture mode. **Calibrate at the resolution you capture at.** |
| `rgb.color_correction` | Flat-field correction: `enabled`, `strength`, `max_gain`/`min_gain`, `neutralize_white`, `preview_corrected`, `on_the_fly_fallback`, and the two `flat_field_*` paths. |
| `geometric_calibration.board` | ChArUco YAML used by both calibration scripts. |
| `geometric_calibration.output_dir` | Where calibration results are written. |
| `geometric_calibration.intrinsics_rgb_cam1/2` | Explicit intrinsics paths for `stereo_calibrate.py`. |
| `geometric_calibration.stereo_captures` | Default session directories for `stereo_calibrate.py`. |
| `thermal.config_xml` | Optris camera XML; `warmup_seconds` before the first grab. |
| `blackfly` | `enabled`, `camera_index`/`serial`, `timeout_ms`, `max_fps`, `preview_max_width`, `gain_auto`/`gain`. |
| `stability` | Preview interval, parallel grabbing, RGB recovery after a failed grab, open order. |
| `gui.window_size` | Preview window geometry, e.g. `"1400x920"`. |

---

## Repository layout

```
capture_pipeline.py      capture entry point
gui.py                   Tk preview window used by the pipeline
sync_metrics.py          grab-timing instrumentation (module, no CLI)
color_correction.py      flat-field correction used at capture time (module, no CLI)
cameras/                 RGBCamera, ThermalCamera, BlackflyCamera drivers
calibration/             ChArUco board, intrinsics, stereo, report writers + the printable target
design/                  rig model: cameras, optics, coverage analysis, export, 3D viewer
captures/                capture sessions (git-ignored)
calibration/results/     intrinsics and extrinsics per camera / pair
color_calibration/       flat-field maps in production use
color_reports/           check_color.py output
OptrixThermalCamera/     Optris SDK samples and camera XML
```

---

## Troubleshooting

- **RGB "No such device"** during capture: usually USB bandwidth. Spread the
  cameras across different USB controllers and check `lsusb -t` in
  `check_cameras.py`'s output. `stability.recover_rgb` reopens the pair
  automatically after repeated failures.
- **Blackfly grabs fail**: check the lens cap and lighting, and make sure
  SpinView or another app is not holding the camera.
- **Board not detected**: look at `calibration/results/<camera>/discarded/` —
  the annotated frames show whether any markers were found at all. A wrong
  `legacy_pattern` setting is the usual cause of near-total failure.
- **Stereo says "calibrated at WxH but these captures are WxH"**: recalibrate the
  intrinsics at the capture resolution instead of rescaling.
- **`prune_calibration.py` refuses to run**: the report is older than the capture
  set. Re-run `calibrate_cameras.py`, then prune again.
