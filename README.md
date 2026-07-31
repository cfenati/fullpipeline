# FullPipeline

Multi-camera acquisition and calibration: two ELP 16 MP RGB cameras, one Optris
Xi 400 LT thermal, optional FLIR Blackfly. Defaults live in `config.yaml`; CLI
flags override them. Run `python <script>.py --help` for full options.

| Stage | Script | Output |
| --- | --- | --- |
| Hardware check | `check_cameras.py` | pass/fail per camera |
| Capture | `capture_pipeline.py` | `captures/<session>/<timestamp>/` |
| Color uniformity | `check_color.py` | plots + flat-field `.npz` maps |
| Intrinsics | `calibrate_cameras.py` | `calibration/results/<cam>/intrinsics.json` |
| Prune bad views | `prune_calibration.py` | deletes worst capture sessions |
| Extrinsics | `stereo_calibrate.py` | `calibration/results/stereo_<a>_<b>/` (+ `rig_as_built.yaml`) |
| Rig eval / optimize | `design_rig.py` | coverage studies on measured geometry |
| Registration | `register_pipeline.py` | depth-aware warp between cameras |

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
python register_pipeline.py --session captures/<session> --compare
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
```

Each trigger writes one folder with `rgb_cam1.jpg`, `rgb_cam2.jpg`, thermal
palette/temperature, optional Blackfly, and `metadata.json`. The same sessions
serve both intrinsics and stereo.

## Color

Shoot a full-frame white sheet, then:

```bash
python check_color.py --live --save-flat-field
```

Copy the `.npz` maps into `color_calibration/` and point
`rgb.color_correction.flat_field_cam*` at them in `config.yaml`.

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

## Rig design

```bash
python design_rig.py --rig design/config/rig_as_built.yaml report --render
python design_rig.py --rig design/config/rig_as_built.yaml optimize
python design_rig.py --rig design/config/rig_as_built.yaml view
python design_rig.py --rig design/config/rig_as_built.yaml sweep z 0.25 0.40
```

Subcommands: `info`, `optics`, `view`, `plot`, `report`, `optimize`, `sweep`.

## Registration

Warps source → target using calibrated extrinsics (RGB–RGB) or rig poses
(thermal). Depth from stereo SGBM or a fixed plane.

```bash
python register_pipeline.py --session captures/hand --compare
python register_pipeline.py --session captures/hand --source thermal --mode plane --depth 0.28
```

Outputs under `registration/results/<session>/`.

## Config highlights

| Key | Notes |
| --- | --- |
| `rgb.cam1` / `cam2` | Prefer `/dev/v4l/by-path/...` |
| `rgb.width` / `height` | Must match calibration resolution |
| `rgb.color_correction` | Flat-field enable + map paths |
| `geometric_calibration.*` | Board, results dir, stereo capture dirs |
| `thermal` / `blackfly` | Device enablement and settings |

## Layout

```
cameras/  capture_pipeline.py  gui.py     drivers + capture
calibration/  calibrate_*.py  stereo_*.py  prune_*.py
design/  design_rig.py                     geometry / coverage
registration/  register_pipeline.py        depth-aware warp
color_correction.py  check_color.py        flat-field
captures/  calibration/results/            data (mostly git-ignored)
```

## Troubleshooting

- **RGB "No such device"**: USB bandwidth — spread cameras across controllers (`lsusb -t`).
- **Board not detected**: check `calibration/results/<cam>/discarded/`; usually wrong `legacy_pattern`.
- **Stereo resolution mismatch**: recalibrate intrinsics at capture resolution (no rescale).
- **Prune refuses**: re-run `calibrate_cameras.py` so the report matches the capture set.
- **Blackfly fails**: lens cap / lighting, or another app (SpinView) holding the device.
