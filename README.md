# FullPipeline

Multi-camera acquisition and calibration: two ELP 16 MP RGB cameras, one Optris
Xi 400 LT thermal, optional FLIR Blackfly. Defaults live in `config.yaml`; CLI
flags override them. Run `python <script>.py --help` for full options.


| Stage               | Script                 | Output                                                        |
| ------------------- | ---------------------- | ------------------------------------------------------------- |
| Hardware check      | `check_cameras.py`     | pass/fail per camera                                          |
| Capture             | `capture_pipeline.py`  | `captures/<session>/<timestamp>/`                             |
| Color uniformity    | `check_color.py`       | plots + flat-field `.npz` maps                                |
| Intrinsics          | `calibrate_cameras.py` | `calibration/results/<cam>/intrinsics.json`                   |
| Prune bad views     | `prune_calibration.py` | deletes worst capture sessions                                |
| Extrinsics          | `stereo_calibrate.py`  | `calibration/results/stereo_<a>_<b>/` (+ `rig_as_built.yaml`) |
| Cross-validation    | `cross_validate_stereo.py` | `calibration/results/stereo_<a>_<b>/cross_validation/` |
| Depth accuracy      | `check_depth_accuracy.py` | `calibration/results/depth_accuracy/` (relative depth error vs. a measured depth-grid target) |
| Rig eval / optimize | `design_rig.py`        | coverage studies on measured geometry                         |
| Registration (sparse)| `register_features.py` | matches → DLT triangulation → piecewise-affine warp |




## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Vendor SDKs (not on PyPI): Optris `libirimager` + XML in `OptrixThermalCamera/config/`;
FLIR Spinnaker + `PySpin`. Geometry-only work (`design_rig.py`) needs just
`numpy`, `matplotlib`, `pyvista`, `opencv-python`, `PyYAML`.

## Workflow

```bash
python check_cameras.py
python capture_pipeline.py --output calib_rgb1          # shoot ChArUco board
python calibrate_cameras.py --camera rgb_cam1
python prune_calibration.py --camera rgb_cam1 --count 5 # dry run; add --apply to delete
python calibrate_cameras.py --camera rgb_cam1           # re-fit
python calibrate_cameras.py --camera rgb_cam2
python stereo_calibrate.py --camera-a rgb_cam1 --camera-b rgb_cam2
python design_rig.py --rig design/config/rig_as_built.yaml report
python register_features.py --session captures/<session>
```

Calibrate at the same resolution you capture. Prefer
`design/config/rig_as_built.yaml` (written by stereo) over the intended
`design/config/rig.yaml`.

## Capture

```bash
python capture_pipeline.py                         # GUI; S / Ctrl+S save, Q quit
python capture_pipeline.py --output calib_rgb1     # -> captures/calib_rgb1/<ts>/
python capture_pipeline.py --smoke-test            # open, grab once, exit
python capture_pipeline.py --no-preview            # headless; type s / q
python capture_pipeline.py --flir-gain-sweep       # sweep FLIR gain, save one frame per step
```

Each trigger writes one folder with `rgb_cam1.jpg`, `rgb_cam2.jpg`, thermal
palette/temperature, optional Blackfly, and `metadata.json`. The same sessions
serve both intrinsics and stereo.

`--flir-gain-sweep` is FLIR-only: instead of one Blackfly frame at a fixed
gain, it saves one frame per gain step (range set by `blackfly.gain_sweep` in
config.yaml) into `captures/flir_gain_sweep/<ts>/`, for picking an operating
gain before committing to `blackfly.gain`.

## Color correction

Flat-field correction removes spatial vignetting / border tint from the RGB
cameras. It equalises the frame *spatially*; it does not re-balance the global
white point, so a uniform colour cast from the lighting survives correction by
design (see the note at the end).

### 1. Lock camera controls

`rgb.controls` in `config.yaml` is applied via `v4l2-ctl` every time a camera
opens, because driver state resets on replug/reboot. Exposure and white
balance must be fixed, or captures drift away from the white reference the
gain maps were built from.

```yaml
rgb.controls:
  common:  {white_balance_automatic: 0, white_balance_temperature: 5000,
            auto_exposure: 3, gain: 0}
```

AWB is held off even though the camera default is on: left on, the two cameras
white-balance independently per frame and drift out of colour agreement with
each other. Exposure is left on Aperture Priority (`auto_exposure: 3`) so
captures are always well exposed.

`auto_exposure: 1` (Manual) is the alternative, and it buys radiometric
reproducibility plus brightness-matched cameras — on this rig cam2's optics
collect ~37% less light, so at equal exposure it lands near gray 104 versus
cam1's 164, and a per-camera `exposure_time_absolute` fixes that. The catch is
that a manual exposure must be chosen for the *subject*: exposing for a white
reference underexposes real captures by ~44%. To re-derive values, run
`check_color.py --live --no-correct`, compare the `center` gray of both
cameras, and scale by `(target_gray / current_gray) ** (1 / 0.68)` — the
sensor response is non-linear (gray ∝ exposure^0.68 measured here).

Note `white_balance_temperature` moves the red/blue axis only. A green cast
(R/G *and* B/G both low) comes from the lamp's spectrum and cannot be fixed
in-camera — see step 3.

### 2. Capture a white reference

Fill both cameras with a flat white/gray sheet, at your real subject
distance, under the lighting you actually capture with. Illumination falloff
is distance-dependent, so the reference must match the working geometry.

### 3. Build flat-field maps and inspect

Only write when white sheet is under cameras

```bash
python check_color.py --live --save-flat-field
```

That writes under `color_reports/`:


| File                                 | Meaning                                |
| ------------------------------------ | -------------------------------------- |
| `live_capture/rgb_cam*_live.jpg`     | raw white-frame capture                |
| `rgb_cam*_live_uniformity.png`       | raw diagnostics (always uncorrected)   |
| `rgb_cam*_live_corrected_s1.00.jpg`  | corrected preview                      |
| `rgb_cam*_live_correction_s1.00.png` | before/after R/G and B/G maps          |
| `rgb_cam*_live_flat_field.npz`       | per-channel gain maps for the pipeline |


On the correction plot, “after” R/G and B/G should look **flat** — that is what
this step fixes. They stay near the raw centre value (~0.86 here) rather than
reaching 1.0, because flat-field only removes *variation across the frame*, not
the overall white point. `*_uniformity.png` only ever shows the raw frame, so it
will not improve; judge the result on the correction comparison.

### 4. Install maps for `capture_pipeline.py`

`config.yaml` already points at these names:

```yaml
rgb.color_correction:
  enabled: true
  flat_field_cam1: color_calibration/rgb_cam1_flat_field.npz
  flat_field_cam2: color_calibration/rgb_cam2_flat_field.npz
```

Copy the new maps over those paths:

```bash
cp color_reports/rgb_cam1_live_flat_field.npz color_calibration/rgb_cam1_flat_field.npz
cp color_reports/rgb_cam2_live_flat_field.npz color_calibration/rgb_cam2_flat_field.npz
```

The `.npz` files store only the gain maps. How strongly they are applied lives
in `config.yaml` (`strength: 1.0`, `max_gain: 1.6`, `min_gain: 0.85`) — the
corners need ~1.6x, so lowering either leaves the edges visibly dark.

Then:

```bash
python capture_pipeline.py
```

Preview and saved RGB frames use the loaded maps when
`rgb.color_correction.enabled` is true.

### Re-check later

```bash
python check_color.py --live                 # apply maps from config
python check_color.py --live --no-correct    # raw diagnostics only
python check_color.py --image color_reports/live_capture/rgb_cam2_live.jpg
```

Re-run steps 1–4 whenever lighting or camera settings change.

### On the residual green cast

The raw centre sits near R/G = B/G = 0.86, i.e. green is strong against *both*
red and blue. That is the lamp's spectrum, not a camera setting:
`white_balance_temperature` only moves the red/blue axis, so no camera control
can remove it, and flat-field deliberately does not touch the global white
point. Software attempts to force those ratios to 1.0 were tried and removed —
they look artificial and, because they are keyed to a centre patch, they get
worse whenever the target moves. The real fix is a high-CRI (≥90) light source.

## Guided live capture (recommended)

```bash
python calibrate_live.py --camera-a rgb_cam1 --camera-b rgb_cam2
```

Auto-captures every `--interval` seconds (default 3s) whenever the ChArUco
board is visible with enough shared corners in both cameras, and re-fits
cam1/cam2 intrinsics *and* stereo extrinsics live after every capture, so you
see RMS, views used, coverage and a ready/not-ready card for each of the
three fits as you go, plus a live count of views auto-excluded for having no
board or a high residual. Nothing auto-stops — keep capturing until every
card reads READY, then press `f` (Finish & Fit) to run `calibrate_cameras.py`
and `stereo_calibrate.py` for you.

Results land in a fresh `calibration/results/live_<timestamp>/` folder, not
the canonical `calibration/results/` or `design/config/rig_as_built.yaml` —
nothing existing is overwritten. Once you're happy with a result, promote it
by re-running the offline commands below with their default output. Sessions
are captured under the same `captures/stereo` the manual workflow uses
(`geometric_calibration.intrinsics_captures`/`stereo_captures` in
`config.yaml`), so switching between the two, or pruning afterward, needs no
migration step. `python calibrate_live.py --no-preview` runs headless
(`f`/`q` in the terminal).

## Calibration

Board: `calibration/config/charuco_11x8.yaml` (printable PDF next to it).
`legacy_pattern: true` for boards made before OpenCV 4.6.

```bash
python calibrate_cameras.py --camera rgb_cam1
python prune_calibration.py --camera rgb_cam1 --count 5 --apply   # dry-run without --apply
python stereo_calibrate.py --camera-a rgb_cam1 --camera-b rgb_cam2
```

Prune deletes whole timestamp folders (all cameras), refuses fewer than 8
sessions (`--force` overrides), and needs a fresh report (re-calibrate or
`--allow-stale`). Stereo holds intrinsics fixed and writes `extrinsics.json`,
figures, and measured poses into `rig_as_built.yaml`.

```bash
python cross_validate_stereo.py --camera-a rgb_cam1 --camera-b rgb_cam2
```

Checks the fitted extrinsics against a second, independently captured
checkerboard pose set (`captures/cross-validation` by default) instead of the
set they were fit from. Triangulates held-out corners through the frozen
`R`/`T`/intrinsics and compares the reconstructed corner-to-corner distances
to the board's known square size — catching overfitting to `stereo_captures`
that in-sample reprojection/epipolar error can't see, since that error is
graded on the very poses that set the scale. Read-only: never rewrites
`extrinsics.json`; results land in a separate `cross_validation/` subfolder.

```bash
python check_depth_accuracy.py --session captures/depth_target/<timestamp> \
    --ref AX,AY,BX,BY ... --cell ROW,COL,AX,AY,BX,BY ...
```

A flat, fronto-parallel plate has ~zero depth variation across it by
construction, so it cannot validate the Z-axis at all. `check_depth_accuracy.py`
targets that axis instead, using a
[depth-grid target](calibration/config/depth_grid_target.yaml) — a 5×5 grid of
blocks at known, distinct heights (0.6–30 mm) on one baseplate, plus 4 flush
corner fiducials.

It clicks (or is given, non-interactively) the corner fiducials and fits the
plane through them — never assuming the plate sits perpendicular to the
camera — then triangulates every visible grid cell and takes its **signed
perpendicular distance to that plane**. Only relative depth is graded:
cell-to-cell separations against their known differences (the headline scale
fit) and each cell's plane-to-cell distance against its own known height (a
direct check that isolates plane-fit bias). Some cells may be self-occluded
from one or both cameras in a given capture — that's expected, not a failure;
correspondence is by `(row, col)` label, not click order or count.

See
[docs/superpowers/specs/2026-08-24-check-depth-accuracy-design.md](docs/superpowers/specs/2026-08-24-check-depth-accuracy-design.md)
for the full method, the FOV/occlusion constraints that shaped the physical
grid's layout, and synthetic verification against this rig's real calibrated
geometry (the physical target is not yet fabricated).

## Rig design

```bash
python design_rig.py --rig design/config/rig_as_built.yaml report --render
python design_rig.py --rig design/config/rig_as_built.yaml optimize
python design_rig.py --rig design/config/rig_as_built.yaml view
python design_rig.py --rig design/config/rig_as_built.yaml sweep z 0.25 0.40
```

Subcommands: `info`, `optics`, `view`, `plot`, `report`, `optimize`, `sweep`.

## Registration

Z is never assumed. A pixel is a ray; two corresponding pixels plus `K`/`R`/`T`
intersect in one 3-D point. That intersection is `triangulate.py` (DLT written
out). Overlay scripts warp only where a correspondence exists; everything else
stays unfilled.

```bash
python register_features.py --session captures/hand
```

`register_features.py` matches sparse features (LightGlue by default) and
triangulates those matches the same way. Warp is piecewise-affine over the
match hull. Needs `torch`/`kornia`; CPU-only.

`check_registration_error.py` scores the feature-mesh warp against held-out
board corners that land inside the match hull.

```bash
python register_features.py --session captures/hand --downscale 0.5
```

Outputs under `registration/results/<session>/`.

## Config highlights


| Key                       | Notes                                   |
| ------------------------- | --------------------------------------- |
| `rgb.cam1` / `cam2`       | Prefer `/dev/v4l/by-path/...`           |
| `rgb.width` / `height`    | Must match calibration resolution       |
| `rgb.color_correction`    | Flat-field enable + map paths           |
| `geometric_calibration.*` | Board, results dir, stereo + cross-validation capture dirs |
| `thermal` / `blackfly`    | Device enablement and settings          |
| `blackfly.gain_sweep`     | dB range/step (`start`/`stop`/`step`) for `--flir-gain-sweep` |




## Layout

```
cameras/  capture_pipeline.py  gui.py     drivers + capture
calibration/  calibrate_*.py  stereo_*.py  prune_*.py
design/  design_rig.py                     geometry / coverage
triangulate.py                             DLT: pixel pair + K,R,T → (X,Y,Z)
register_features.py                       matches → triangulate → piecewise-affine warp
color_correction.py  check_color.py        flat-field
captures/  calibration/results/            data (mostly git-ignored)
```

