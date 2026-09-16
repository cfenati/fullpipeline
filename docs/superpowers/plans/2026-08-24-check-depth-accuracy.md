# check_depth_accuracy.py Implementation Plan

> **Status: implemented in a single session (2026-08-24).** Unlike the line-
> ladder plan, this was executed directly rather than dispatched task-by-task
> to a fresh agent, so every step below is a record of what was actually
> built and verified, not a prospective plan. See
> `docs/superpowers/specs/2026-08-24-check-depth-accuracy-design.md` for the
> full design rationale — read that first.

**Goal:** Add `check_depth_accuracy.py`, validating this rig's RELATIVE
stereo depth (Z-axis) precision using a 3D-printed depth-grid target, the
axis `check_line_accuracy.py`'s flat line ladder cannot touch.

**Architecture:** `calibration/depth_grid_target.py:DepthGridTarget` (mirrors
`LineLadderTarget`) describes a grid of blocks at known heights plus flush
corner reference fiducials. `check_depth_accuracy.py` triangulates the
reference clicks, fits a plane through them (never assuming the baseplate's
mounting angle), and measures each clicked cell's signed perpendicular
distance to that plane — reusing `measure_points.py`'s click UI and
triangulation, and `check_line_accuracy.py`'s `fit_scale`, unmodified.

**Tech Stack:** Python 3.9, `opencv-contrib-python`, `numpy`, `PyYAML`
(already in `requirements.txt`); this repo's `calibration.stereo.
StereoExtrinsics`/`collect_session_pairs`, `registration_io.
default_extrinsics_path`/`undistort_pair`, `measure_points.py`'s click
pipeline, `check_line_accuracy.fit_scale`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-24-check-depth-accuracy-design.md`.
- No test suite or linter in this repo (per `CLAUDE.md`) — verified with
  throwaway `python3 -c` scripts against synthetic data and this rig's real
  calibrated extrinsics, run during implementation and not committed, same
  precedent as `docs/superpowers/plans/2026-08-19-check-line-accuracy.md`.
- `from __future__ import annotations` + type hints on every signature.
- RGB cameras only, defaults `rgb_cam1`/`rgb_cam2`.
- Depth is relative-only: every reported quantity is a difference between
  two triangulated points, never a single point's absolute Z.
- Correspondence is keyed by `(row, col)` grid label, never by click order
  or a fixed count — partial visibility (occluded cells) is the expected
  case, not an error.
- Do not modify `measure_points.py` or `check_line_accuracy.py` — both are
  imported from directly.

---

### Task 0: Synthetic occlusion check (before committing to fabrication)

- [x] Loaded the rig's real calibrated extrinsics
  (`calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json`), built
  the candidate grid geometry (50×50 mm, 10 mm pitch, heights per the
  original 5-column layout) at the planned standoff, and ran a ray-box
  intersection test for both cameras' optical centres against every other
  block's bounding box.
- [x] Verified robustness: swept base standoff 170–185 mm and lateral offset
  ±5 mm in both axes — 0/25 blocked in every case tested.
- [x] Recorded the final arrangement directly in
  `calibration/config/depth_grid_target.yaml`'s `heights_mm`, with a comment
  explaining why it's mirrored and warning not to reorder without rerunning
  the check. --> I will probably chage the height of the cell as i re design it, so this is not definitive anyways and you sould not worry about mirroring or anything else

### Task 1: `DepthGridTarget` dataclass + config

**Files:** `calibration/depth_grid_target.py` (new),
`calibration/config/depth_grid_target.yaml` (new), `config.yaml` (edited).

- [x] Wrote `DepthGridTarget`: `heights_mm` (row-major tuple of tuples),
  `pitch_mm`, `reference_corner_count`, `height_uncertainty_mm`,
  `measured_by`; validation (rectangular grid, non-negative heights,
  positive pitch, `reference_corner_count >= 3`, non-negative uncertainty);
  `height_at(row, col)`; `pair_depths_mm(cells)` (mirrors
  `LineLadderTarget.pair_distances_mm()`, generalized to a session-dependent
  cell subset); `from_dict`/`from_yaml`/`to_dict`.
- [x] Wrote `depth_grid_target.yaml` with the Task-0-verified arrangement,
  `measured_by: "nominal (CAD) -- NOT verified"`, and mounting-guidance
  comments (standoff, FOV/occlusion rationale).
- [x] Added `geometric_calibration.depth_grid_target` /
  `depth_grid_target_captures` to `config.yaml`, beside the existing
  `line_target` pair.
- [x] Verified: round-trip (`from_dict`/`to_dict`), every validation error,
  and `pair_depths_mm()` against a hand-computed table for a partial cell
  subset — all pass (`python3 -c` heredoc, not committed).

### Task 2: Plane-fit geometry

**File:** `check_depth_accuracy.py` (new).

- [x] Wrote `fit_plane_3d` (PCA/SVD; normal = smallest singular vector)
- [x] Verified against a deliberately tilted synthetic plane (normal
  `[0.12, -0.35, -0.928]`, not axis-aligned): normal recovered to 1.7e-16,
  RMS 1.1e-17; point-to-plane offset recovered to 1e-12 m.

### Task 3: Core measurement + CLI

**File:** `check_depth_accuracy.py`.

- [x] `measure_depth_session`: triangulates reference and cell clicks
  separately via `measure_points.measure_points` (imported, unmodified),
  fits the plane, orients its normal toward camera A's optical centre,
  computes `cell_measured_mm` and `pair_measured_mm`/`pair_truth_mm`, and a
  best-effort order-mismatch warning (all 25 target heights are distinct, so
  clicked cells have a well-defined true rank order to check against).
- [x] `aggregate_depth_results`/`write_report`: written fresh rather than
  reusing `check_line_accuracy.py`'s versions (incompatible per-session
  shapes — see spec's Interfaces section) but mirroring their structure
  (bucket-by-truth pairwise table, pooled scale fit, per-session quality
  table), plus a per-cell table for the direct plane-to-cell check.
- [x] CLI: `--captures`/`--session`, `--camera-a`/`--camera-b`,
  `--extrinsics`, `--target`, `--ref AX,AY,BX,BY` (repeatable ×
  `reference_corner_count`), `--cell ROW,COL,AX,AY,BX,BY` (repeatable,
  0–25), click-UI passthrough flags reused directly from `measure_points.py`
  (`DEFAULT_LOUPE_ZOOM`, `DEFAULT_MAX_WINDOW`, `DEFAULT_BLOB_RADIUS_PX`),
  `--out`. Interactive mode reuses `run_interactive` unmodified and prompts
  for `(row, col)` labels once the click session ends, in click order (no
  hook exists inside `run_interactive` for per-click metadata, and it is not
  modified to add one).
- [x] Verified full pipeline against the rig's real calibrated `K_a, K_b, R,
  T`: a tilted synthetic reference plane plus 7 grid cells (partial subset)
  recovered every depth to 5.0e-14 mm and every pairwise separation to
  6.2e-14 mm. Injected +1.0 mm error on one cell recovered as exactly
  +1.000000 mm, others unchanged. End-to-end CLI subprocess run (dummy
  images, real extrinsics/target, `--ref`/`--cell` flags, partial 3-of-25
  cell set) wrote correct `report.txt`/`result.json`/annotated JPEG.

### Task 4: Companion docs

- [x] `docs/superpowers/specs/2026-08-24-check-depth-accuracy-design.md`
  (Purpose, Method, decisions, Interfaces, Reporting, physical grid,
  Verified, Out of scope).
- [x] This plan file, recorded as a completed log rather than a prospective
  task list.

## Verification summary

All verification was synthetic `python3 -c` heredoc runs against this rig's
actual calibrated geometry (no physical target exists yet) — see the spec
doc's "Verified" section for exact numbers. `python3 -c "import
calibration.depth_grid_target"` and `python3 -c "import
check_depth_accuracy"` both succeed.

## Not done here (see spec's "Out of scope")

Fabricating and calipering the physical grid; running the script against a
real capture; dense FoundationStereo depth validation; absolute depth
accuracy; automatic block detection.
