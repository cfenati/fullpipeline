# FullPipeline

Multi-camera acquisition/calibration/registration pipeline (2x RGB, Optris thermal,
optional FLIR Blackfly) for a master's thesis rig. **[README.md](README.md) is the
canonical workflow doc** — stage table, CLI commands, config keys, color-correction
procedure. Read it before making changes; don't duplicate its content here.

## Running things

Most entry-point scripts (`capture_pipeline.py`, `check_cameras.py`,
`calibrate_cameras.py`, `stereo_calibrate.py`, `calibrate_live.py`) talk to physical cameras via
`v4l2-ctl` / vendor SDKs and **will not run without the hardware attached**.
Don't assume a script failure means broken code — check whether it's a missing-device
error first. Geometry-only work (`design_rig.py`, `design/`) has no hardware
dependency and runs anywhere with `numpy`/`pyvista`/`opencv-python`.

Python 3.9. No venv is committed — `python3 -m venv .venv && pip install -r
requirements.txt` per the README. Vendor SDKs (Optris `libirimager`, FLIR
`PySpin`) are not on PyPI and must already be installed on the target machine.

## Verifying changes

There is no test suite and no linter config in this repo. To validate a change:
- Prefer a syntax/import check (`python -c "import module"`) or running the
  hardware-free paths (`design_rig.py`) directly.
- For camera/capture code, changes can't be exercised without the rig; say so
  explicitly rather than claiming a change works when it wasn't run.
- `check_color.py --image <file>` can validate color-correction logic against a
  saved JPEG without live cameras.

## Conventions observed in the code

- `from __future__ import annotations` + type hints on function signatures.
- Config loaded once from `config.yaml` via `load_config()`, paths resolved
  against `PROJECT_ROOT = Path(__file__).resolve().parent`.
- Scripts are argparse CLIs (`--help` documents flags); keep new scripts
  consistent with that pattern rather than adding a different config style.
- Comments in `config.yaml` explain non-obvious hardware/physics choices (e.g.
  why AWB is forced off, why aperture-priority exposure is default) — treat
  these as load-bearing documentation, not clutter, when editing config.

## Data hygiene

`captures/`, `calibration/results/`, `color_calibration/`, `color_reports/`,
`design/out/`, and `registration/results/` are all git-ignored (see
[.gitignore](.gitignore)) — they're large generated/measured data, not source.
Don't add them to commits even if `git add -A` picks them up.
