# Design: check_depth_accuracy.py

Date: 2026-08-24
Status: **Implemented** (code verified synthetically; physical target not yet
fabricated)

Companion to `docs/superpowers/plans/2026-08-19-check-line-accuracy.md` and
`docs/superpowers/specs/2026-08-19-check-line-accuracy-design.md`, which this
document assumes as background — read those first if the line ladder's own
method and provenance conventions are unfamiliar.

## Purpose

`check_line_accuracy.py` validates this rig's **lateral (X-Y)** triangulation
accuracy: it grades the scale that converts pixels to millimetres. But its
own target — a flat, fronto-parallel line ladder — has ~zero depth variation
across it by construction, so it can never touch the **depth (Z) axis**,
which is governed by a different error budget entirely (stereo disparity
precision, baseline, convergence angle) than in-plane accuracy. Nothing in
this repo checked depth against an independent ground truth before this
script.

The gap was found directly, not theoretically: clicking two points at
different physical heights with `measure_points.py` and reading its
`depth_mm` field returns 182.9 mm — the raw Z-coordinate of ONE triangulated
point in camera A's optical frame, not a difference between the two points,
and not aligned with any physical "height" axis on the clicked object (this
rig's cameras converge at 17.66°, so no physical axis is guaranteed aligned
with camera A's optical axis). `check_depth_accuracy.py` exists to measure
what that click actually could not: a real depth *difference*, against an
independently fabricated target whose heights are known.

## Method

Per session:

1. **Reference clicks.** Click the 4 flush (zero-standoff) corner fiducials
   on the depth-grid target's baseplate, in undistorted pixel coordinates,
   in both cameras.
2. **Cell clicks.** Click every grid cell visible in both cameras. Some cells
   may be self-occluded from one or both cameras (see "The physical grid")
   — a session that measures 15 of 25 cells is a normal, valid result, not a
   partial failure. Correspondence is by `(row, col)` label, never by click
   order or count.
3. **Triangulation.** `measure_points.measure_points` (imported directly,
   unmodified) on the reference clicks and, separately, on the cell clicks —
   `triangulate.triangulate_points` (DLT), giving 3D points in camera A's
   frame, metres.
4. **Plane fit.** PCA/SVD through the 4 triangulated reference points
   (`fit_plane_3d`) — the smallest singular vector is the plane's normal.
   This plane, not any assumed camera-to-baseplate standoff, is the datum
   every cell's depth is measured against.
5. **Depth.** Signed perpendicular distance from each triangulated cell point
   to the fitted plane (`perpendicular_distance_to_plane`), oriented so a
   cell protruding toward the camera reads positive.
6. **Comparison**, two ways (see "The decisions that carry the accuracy"):
   cell-to-cell separations against the target's known pairwise height
   differences (the headline, scale-fit quantity), and each cell's
   plane-to-cell distance against its own known height (a direct check that
   isolates plane-fit bias, not used for the scale fit).

## The decisions that carry the accuracy

**Fit the reference plane; never assume the baseplate's orientation.** The
target is a handheld/tripod-mounted object with no guarantee its face is
perpendicular to camera A's optical axis. A naive raw-Z difference between
two triangulated points is contaminated by exactly this unknown tilt,
proportional to their lateral separation — the same failure mode
`check_line_accuracy.py`'s perpendicular-line-fit avoids for lateral
distances. Fitting the plane from points **on the object itself** removes
the dependency on any assumed mounting angle.

**Two comparisons, not one, because they isolate different failure modes.**
Cell-to-cell (pairwise) separations are immune to a constant offset in where
the fitted plane sits — this is why it drives the scale fit, mirroring the
line ladder's own use of pairwise gaps over absolute mark position.
Plane-to-cell distances are not: they are the only quantity that can expose
a systematic bias in the plane fit itself (e.g. if the reference points
happen to be clicked with a correlated error). Reporting both, rather than
picking one, is deliberate.

**Relative depth only — the camera's own optical-centre offset is never
touched.** `measure_points.py`'s raw `depth_mm` is only meaningful as an
absolute distance if you trust where camera A's optical centre physically
sits, which is not independently known. Every quantity this script reports
is a *difference* between two triangulated points, so that unknown constant
cancels — the same reasoning that led the line ladder to use gaps, not
absolute plate position.

**Orient the plane normal explicitly, don't trust the SVD's sign.** `SVD`
returns a normal up to sign, which is meaningless on its own — a "positive
height" only means something once the normal is pinned to point toward the
camera. `if normal @ (-centroid) < 0: normal = -normal` fixes this using the
one fact that's always true regardless of target placement: the camera's
optical centre is the coordinate origin in this frame.

**A cheap, non-fatal order-mismatch warning, not a hard safety net.** Unlike
the line ladder's self-checking, strictly-increasing gap pattern (which
makes a lost mark structurally detectable), there is no automatic pattern
check here — correspondence is manual, by grid label. All 25 of the target's
heights are distinct, so ranking the *clicked* cells by known height and
checking the *measured* order agrees catches an obviously mislabeled click,
but this is explicitly a diagnostic, not a guarantee (see "Out of scope").

## Interfaces

- `calibration/depth_grid_target.py` — `DepthGridTarget` frozen dataclass
  mirroring `LineLadderTarget`'s conventions: `heights_mm` (row-major grid),
  `pitch_mm`, `reference_corner_count`, `height_uncertainty_mm`,
  `measured_by`; `height_at(row, col)`; `pair_depths_mm(cells)` — all
  `C(n,2)` pairs for whichever cells were actually clicked this session
  (mirrors `LineLadderTarget.pair_distances_mm()`, generalized from a fixed
  ladder length to a session-dependent subset); `from_dict`/`from_yaml`/
  `to_dict`.
- `calibration/config/depth_grid_target.yaml` — the grid's measured geometry
  and the occlusion-verified arrangement (see "The physical grid").
- `config.yaml` — `geometric_calibration.depth_grid_target` and
  `depth_grid_target_captures`, alongside the existing `line_target` pair.
- `check_depth_accuracy.py` — argparse CLI following `check_line_accuracy.py`'s
  conventions (`--captures`/`--session`, `--camera-a`/`--camera-b`,
  `--extrinsics`, `--target`, `--out`). Non-interactive correspondence via
  `--ref AX,AY,BX,BY` (repeatable, exactly `reference_corner_count`) and
  `--cell ROW,COL,AX,AY,BX,BY` (repeatable, 0–25); interactive mode reuses
  `measure_points.run_interactive` unmodified and prompts for `(row, col)`
  labels once the click session ends, in click order.
- Reused directly and unmodified: `check_line_accuracy.fit_scale`;
  `measure_points.measure_points`, `run_interactive`, `annotate`,
  `parse_point`, `DEFAULT_LOUPE_ZOOM`, `DEFAULT_MAX_WINDOW`,
  `DEFAULT_BLOB_RADIUS_PX`; `calibration.stereo.StereoExtrinsics`,
  `collect_session_pairs`; `registration_io.default_extrinsics_path`,
  `undistort_pair`. `aggregate_results`/`write_report` were **not** reused
  from `check_line_accuracy.py` — both are tightly coupled to line-specific
  per-session shapes with no clean analog here — and are reimplemented as
  `aggregate_depth_results`/`write_report` with depth-specific fields
  (plane RMS, per-cell table) in their place.

## Reporting

`report.txt`, `result.json`, one annotated correspondence JPEG per session
(via `measure_points.annotate`, reused as-is). The headline is the pairwise
fit `measured = a · true` over cell-to-cell separations: `a − 1` is the
systematic depth-scale error. A companion `a · true + b` fit distinguishes a
fixed plane-fit bias (large `b`) from a proportional scale error. The
per-cell table (plane-to-cell, not used in the scale fit) is reported
separately, since it answers a different question — whether the plane fit
itself is biased — that the pairwise numbers cannot see.

## The physical grid

Confirmed target: a single 3D-printed baseplate, **50×50 mm footprint**, a
**5×5 grid of 10×10 mm blocks** tiling edge-to-edge, heights **0.6–30 mm**.
Not yet fabricated at time of writing — two geometric constraints, both
derived from this rig's actual calibrated numbers rather than assumed, were
resolved before finalizing the design:

**FOV.** The rig's vertical FOV (23.92°) is tighter than horizontal
(31.46°) — the binding constraint. At the tallest block's tip (closest point
to camera, hence largest angular size), a 50 mm object needs ~11.3° of
vertical half-angle against a ~12.0° limit: workable, but with almost no
margin. This is why the design adds only small (~4×4 mm) corner pads for the
plane-fit reference points, not a full-perimeter border, and why the
recommended mounting standoff (tallest tip ~148–150 mm, base ~178–180 mm)
sits inboard of the depth range's edges rather than pushed to 130 mm.

**Occlusion.** The original column order (shortest 0.6–1.4 mm column through
tallest 10–30 mm column, left to right) leaves one cell shadowed from camera
B: a ray-box intersection test using the rig's *actual* calibrated `K`/`R`/`T`
(baseline 49.93 mm, convergence 17.66°) found the 9.0 mm block occluded by
its 30 mm neighbour from camera B's oblique viewing angle. Mirroring the
column order — tallest column moved to the edge nearer camera A's side,
away from camera B's more oblique rays — clears all 25 cells from both
cameras, verified robust across a sweep of ±5 mm lateral offset and
170–185 mm standoff (0/25 blocked in every case tested). The config
(`calibration/config/depth_grid_target.yaml`) records this arrangement with
an explicit warning not to reorder columns without rerunning the check.

**Ground truth provenance.** As with the line ladder, `heights_mm` in
`depth_grid_target.yaml` are currently CAD nominal, not calipered —
`measured_by` says so explicitly, and the script warns on every run until
it's updated post-fabrication.

## Verified

Synthetic only — no physical target exists yet. All checks used this rig's
**actual** calibrated extrinsics
(`calibration/results/stereo_rgb_cam1_rgb_cam2/extrinsics.json`), not
idealized geometry:

- `DepthGridTarget` round-trip, validation errors (non-rectangular grid,
  negative heights, non-positive pitch, `reference_corner_count < 3`,
  negative uncertainty), and `pair_depths_mm()` against a hand-computed
  table — all pass.
- `fit_plane_3d`/`perpendicular_distance_to_plane` against a deliberately
  tilted synthetic plane (normal `[0.12, -0.35, -0.928]`, not axis-aligned):
  normal recovered to 1.7e-16, RMS 1.1e-17, point-to-plane offset recovered
  to 1e-12 m.
- Full pipeline against the real calibrated `K_a, K_b, R, T`: a tilted
  reference plane plus 7 grid cells (a deliberately partial subset,
  exercising the variable-N path) projected through the real cameras and
  fed through `measure_depth_session` recovered every cell's depth to
  **5.0e-14 mm** and every pairwise separation to **6.2e-14 mm** — floating-
  point noise, confirming the geometry, not an approximation.
- Injected-error check: perturbing one synthetic cell's height by +1.0 mm
  (without touching the target's own truth) is recovered as **+1.000000 mm**
  on that cell alone, all others unchanged — the pipeline is sensitive to
  real error, not silently blind to it.
- End-to-end CLI subprocess check: dummy same-size images, real extrinsics
  and target YAML, `--ref`/`--cell` flags with a deliberately partial
  3-of-25 cell set — `report.txt`, `result.json`, and the annotated JPEG are
  all written, and the recovered per-cell errors match ground truth to
  <1e-13 mm.

## Out of scope

Dense FoundationStereo / disparity-map depth validation (sparse click-based
triangulation only, this iteration); absolute depth accuracy (distance from
camera) — only relative depth-difference (ΔZ) is graded; automatic
block/fiducial detection (deliberately manual and click-based; no Radon- or
Hough-style detector); a hard click-order/labeling safety net beyond the
best-effort monotonicity warning (a swapped or mislabeled click can still
produce a plausible but wrong report); thermal/FLIR support; live capture;
non-flat/non-rigid baseplate; multiple baseplate poses combined within one
session.
