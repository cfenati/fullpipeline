# Design: measure_wound_depth.py -- relative depth of a real specimen (wound) against its own local surroundings

Date: 2026-09-15
Status: **Implemented** (2026-09-15, `ab1d534`..`7199e3b`)

## Purpose

`check_depth_accuracy.py`'s plane-fit machinery (`fit_plane_3d`,
`perpendicular_distance_to_plane`) was built to validate the rig's depth
accuracy against a rigid, flat calibration plate: one global reference plane,
fit from corner clicks anywhere on the plate, is valid everywhere on it
because the whole object is planar by construction.

That assumption breaks for a real specimen. Trying to reuse
`check_depth_accuracy.py` directly to measure how deep a wound sits below the
surrounding skin surface produces nonsense, for two independent reasons:

1. Every measured "block" is looked up against `DepthGridTarget.height_at
   (row, col)` -- a known CAD height for a specific calibration block. A
   wound has no such ground truth; the row,col labeling and the pairwise
   scale-fit built on top of it don't apply at all.
2. More fundamentally: skin is not guaranteed flat, even locally. A single
   plane fit through points spread across the skin around a wound would bake
   in whatever body-surface curvature exists as if it were wound depth --
   exactly the error this rig's whole "relative, not absolute" design
   philosophy (see `check_depth_accuracy.py`'s module docstring) exists to
   avoid.

This design covers a new, standalone script that keeps the parts of the
existing tools that generalize (stereo click -> triangulate -> plane fit ->
signed perpendicular distance, and the click/undo/epipolar-snap machinery in
`measure_points.run_interactive`) and drops the parts that don't
(`DepthGridTarget`, ground-truth lookup, pairwise scale fit, one-global-plane
assumption).

## Method

### 1. Local reference surface, not one global plane

Rather than fitting one plane through all reference clicks, the user clicks
points along the wound's margin (intact skin immediately surrounding the
wound, not distant skin) -- the "rim." For each wound point measured, only
the **k nearest rim points** (by 3D Euclidean distance in mm, using their
already-triangulated coordinates) are used to fit that wound point's own
local plane. Different wound points, especially on either side of a curved
region, get different local planes drawn from whichever rim points are
actually close to them -- curvature across the whole rim is never assumed to
be one flat surface.

`k` is a CLI flag, `--neighbors` (default 5, minimum enforced at 3 -- a plane
needs at least 3 non-collinear points). The user is expected to click more
than `k` rim points spread around the margin; if the rim pool size equals
exactly `k`, every wound point resolves to the same single local plane
(harmless, but degrades to the old one-global-plane behavior and is worth
noting in the module docstring as a usage tip, not a hard error).

`fit_plane_3d` and `perpendicular_distance_to_plane` are imported from
`check_depth_accuracy.py` unchanged, not duplicated.

### 2. Two-phase session, driven by the existing hook system

Reuses `measure_points.run_interactive`'s `on_point`/`on_undo`/`on_advance`
hooks (the same ones `check_depth_accuracy.PlaneCollectionSession` uses) via
a new `RimAndWoundSession` class. No `on_text_submit` hook is needed --
wound points are never labeled or grouped, only auto-numbered.

**Phase 1 -- rim collection** (`self.rim_closed = False`): every completed
click pair is a rim point. `on_point` records its triangulated mm coordinates
and prints `rim point N recorded`. Pressing **n** invokes `on_advance`: if
fewer than `--neighbors` points have been clicked, it refuses ("need at least
N rim points, have M") and stays in phase 1. Otherwise it sets
`rim_closed = True`, prints a confirmation (`rim locked: N points`), and
returns `None` -- no text-entry prompt, since there is nothing to type.

**Phase 2 -- wound points** (`rim_closed = True`): every subsequent
completed click pair is, immediately and without pressing **n** again, a
wound-point measurement:
1. Compute 3D (mm) Euclidean distance from this point to every rim point.
2. Take the `k` nearest.
3. `fit_plane_3d` through those `k` points -> local centroid, normal, RMS.
4. Orient the normal toward camera A's optical centre (same convention as
   `check_depth_accuracy.py`: `if normal @ (-centroid) < 0: normal = -normal`).
5. `perpendicular_distance_to_plane(wound_point, centroid, normal)` gives a
   signed distance where a point *protruding toward the camera* is positive
   (the existing rig-wide convention). A wound is a **recession**, so the
   sign is flipped before reporting: `depth_mm = -signed_distance_mm`. This
   flip is called out explicitly in the module docstring so the convention
   difference from `check_depth_accuracy.py` (which never flips) is not
   silently lost on a future reader.
6. Print immediately: `wound point I: depth D.DDD mm (local plane rms
   R.RRRR mm from k rim points, farthest F.F mm)`. The farthest-neighbor
   distance is a cheap diagnostic -- if it's large relative to the wound's
   own size, the "local" plane isn't very local and the user should click
   more rim points nearer that side of the wound.

A second **n** press after `rim_closed` is already `True` is a harmless
no-op: print "rim already closed -- every click is measured immediately" and
return `None`.

### 3. Undo

`u` still removes exactly one most-recent completed click, same as today.
`on_undo(new_count)`:
- Before the rim closes: pops the point from the rim list, same pattern as
  `PlaneCollectionSession`.
- After the rim closes: only allowed to remove wound points. If `new_count`
  would cut into the closed rim pool (i.e., `new_count` < the rim point
  count), refuse with a printed message ("can't undo into the closed rim
  batch") and leave state unchanged. Rationale: wound-point depths already
  printed are a function of the rim pool at the time they were computed;
  silently reopening the rim and letting it shrink would leave stale,
  already-reported numbers with no indication they no longer match the
  current rim.

### 4. Non-interactive mode

`--rim AX,AY,BX,BY` (repeatable, at least `--neighbors` times) and `--wound
AX,AY,BX,BY` (repeatable, at least once), mirroring the `--point` /
`--ref`/`--cell` pattern in `measure_points.py` / `check_depth_accuracy.py`.
Runs the identical k-NN-plus-local-plane math without a window. Per
CLAUDE.md's verification guidance (no test suite; hardware-dependent scripts
are validated via non-interactive flags with known coordinates), this is the
path used to check the geometry logic against synthetic points forward-
projected through a real saved extrinsics file, the same technique
`check_depth_accuracy.py`'s own verification used.

### 5. Output

Same convention and location style as `measure_points.py`: written to
`registration/results/<session>/wound_depth/` (using
`registration.output_dir` from config, matching `measure_points.py`, since
this is a one-off measurement tool, not a repeatable-target accuracy check
like `check_depth_accuracy.py`'s `calibration/results/.../depth_accuracy`).

- `report.txt` -- one row per wound point: index, depth_mm, local plane RMS,
  neighbor count, farthest-neighbor distance, epipolar click offset (the
  existing per-click diagnostic already returned by `measure_points`).
- `result.json` -- same fields, machine-readable, plus the rim point
  coordinates and their mm positions (so a session can be audited or
  reprocessed later without re-clicking).
- An annotated JPEG (`annotate`, adapted from `measure_points.annotate`):
  rim points drawn in one color, wound points in another, each wound point's
  depth labeled next to its marker.

### 6. Error handling

- Rim closed with fewer than `--neighbors` points: refused at `on_advance`
  time (interactive), or a `SystemExit` at argument-validation time
  (non-interactive, mirroring `check_depth_accuracy.py`'s `--ref`/`--cell`
  upfront checks).
- Session ends (`q`/Esc) with `rim_closed` still `False`, or `True` but zero
  wound points clicked: print "no wound points measured" and skip writing
  any output files, rather than crashing on an empty result -- same pattern
  as `check_depth_accuracy.py`'s "skipped" sessions.
- Undo into a closed rim batch: refused, see Method 3.

## Interfaces

New:
- `measure_wound_depth.py` -- new file. CLI flags: `--session` (required),
  `--camera-a`/`--camera-b` (defaults `rgb_cam1`/`rgb_cam2`, matching
  siblings), `--extrinsics`, `--neighbors` (default 5), `--rim` (repeatable),
  `--wound` (repeatable), `--zoom`/`--window`/`--blob-radius`/`--blob-snap`
  (reused verbatim from `measure_points.py`'s argument shapes), `--out`.
- `RimAndWoundSession` class (in the new file), implementing
  `on_point`/`on_undo`/`on_advance` for `run_interactive`.
- `nearest_rim_plane(wound_point_mm, rim_points_mm, k)` -- small helper:
  returns the `k`-nearest subset and its `fit_plane_3d` result. Kept as a
  standalone function (not inlined in the session class) so the
  non-interactive path can call it directly with `--rim`/`--wound`-supplied
  points, exactly as `measure_depth_session` is a standalone function
  `check_depth_accuracy.py`'s interactive and non-interactive paths both
  call.

Reused unchanged (imported, not duplicated):
- `measure_points.run_interactive`, `handle_interactive_key`,
  `measure_points` (the triangulation function), `parse_point`,
  `DEFAULT_BLOB_RADIUS_PX`, `DEFAULT_LOUPE_ZOOM`, `DEFAULT_MAX_WINDOW`.
- `check_depth_accuracy.fit_plane_3d`,
  `check_depth_accuracy.perpendicular_distance_to_plane`.
- `calibrate_cameras.load_config`, `calibrate_cameras.resolve_path`.
- `calibration.stereo.StereoExtrinsics`.
- `registration_io.default_extrinsics_path`, `registration_io.undistort_pair`.

Unchanged: everything in `check_depth_accuracy.py`, `measure_points.py`,
`calibration/depth_grid_target.py` -- this is purely additive.

## Out of scope

- Any change to `check_depth_accuracy.py`'s grid/target/scale-fit behavior.
- Repeat-click averaging or named labels for wound points (explicitly
  rejected during design -- every click is its own independent measurement
  at its own location, unlike the calibration grid's repeated-block-click
  averaging).
- A global smooth-surface (e.g. quadratic) fit or Delaunay-triangulated
  interpolation across the rim -- considered and explicitly not chosen in
  favor of the simpler per-point-nearest-k-neighbors local plane.
- Automatic wound-margin detection, wound area/volume estimation, or any
  clinical measurement beyond point depth -- this script answers "how deep
  is this point below its immediate surroundings," nothing more.
- Any change to units, coordinate frames, or the rig's depth-range checks
  beyond what `measure_points.measure_points` already returns.

## Verification plan

No test suite exists in this repo (per CLAUDE.md). Verification follows the
same pattern as `check_depth_accuracy.py`:
1. `python -c "import measure_wound_depth"` -- syntax/import check.
2. `nearest_rim_plane` and `RimAndWoundSession`'s hook logic verified with
   synthetic in-memory points (no window needed) -- e.g. rim points on a
   known tilted plane, a wound point at a known offset from it, asserting
   the recovered depth matches the known offset within floating-point
   tolerance, and that a curved rim (two locally-flat patches at different
   tilts) recovers each wound point's own local depth rather than being
   biased by the other patch.
3. End-to-end CLI subprocess check via `--rim`/`--wound`, using a real saved
   extrinsics file and synthetic pixel coordinates forward-projected from
   known 3D points (the same technique used to verify
   `check_depth_accuracy.py` without hardware).
4. The actual click-through-a-window interactive path, and any real wound
   photograph, cannot be exercised by script -- a real hardware session on
   an actual specimen remains the user's own next step, same caveat as every
   other interactive tool in this repo.
