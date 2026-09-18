# Design: check_line_accuracy.py

> **Retired.** `check_line_accuracy.py` and `calibration/line_target.py` were deleted in `ff8c992`. Relative accuracy is now checked by `check_depth_accuracy.py` (see `2026-08-24-check-depth-accuracy-design.md`, which still refers to this tool's method). Kept as a design record only.

Date: 2026-08-19 (drafted), 2026-08-21 (rewritten to match the implementation)
Status: **Implemented, then retired** (see note above)

Supersedes the 2026-08-19 draft of this file. The accompanying plan,
`docs/superpowers/plans/2026-08-19-check-line-accuracy.md`, was **not** followed:
its detection design proved unusable in practice (see "What changed and why"),
so it is stale and should not be executed. This document describes what exists.

## Purpose

Every metric length this rig reports is scaled by one number: `square_size_m:
0.004` in `calibration/config/charuco_11x8.yaml`. `calibration/stereo.py:16-22`
already spells out the consequence — a 1 % error in that printed square is a 1 %
error in every triangulated length in the thesis.

Nothing in the repo could catch it. `cross_validate_stereo.py` holds out board
*poses* but reuses the same board with the same nominal square, so a wrong
square cancels exactly and it reports zero error; `register_board.py` shares the
blind spot. Both grade **repeatability**; neither grades **scale**.

`check_line_accuracy.py` grades scale against an independently fabricated plate
of parallel marks whose centre-to-centre gaps were measured with a different
instrument. Target bar: ~1 % relative error, characterised rather than merely
passed.

## Method

Per session, on an undistorted image pair:

1. **Background and darkness.** Morphological closing with a structuring element
   laid across the marks erases them and leaves the illumination; subtracting
   gives a flat-fielded darkness map of the marks alone.
2. **Angle and offsets.** Project the darkness onto a candidate normal and
   maximise the projection's concentration — a 1D Radon transform — to get the
   ladder's angle, then pick one peak per mark. The peaks are validated against
   the target's known gap pattern by least squares; a session whose peaks are not
   an affine image of `positions_mm` is **skipped, not measured**.
3. **Sub-pixel refinement.** Walk each line and take the darkness-weighted
   centroid of the mark's profile across a perpendicular window, then refit.
4. **Correspondence.** For each sample on line *i* in A, intersect its epipolar
   line with line *i* in B. Samples whose crossing angle is too shallow are
   rejected (`--min-intersection-angle-deg`).
5. **Triangulation.** `triangulate.triangulate_points` (DLT), giving 3D points in
   camera A, metres.
6. **Distance.** Fit a 3D line per mark, take the **perpendicular distance
   between the fits**, for all line pairs.

## The four decisions that carry the accuracy

**Undistort the images before detecting, not the points afterwards.** Under
radial distortion a straight 3D line projects to a *curve*. Fitting a straight
line to it in the raw image fits a chord to an arc and lands the epipolar
intersection off the true correspondence. Measured on synthetic data whose true
error is exactly zero: detecting in the distorted image gave +0.094 % scale error
and 0.040 mm residual; detecting after `registration_io.undistort_pair` gave
-0.000 % and 0.0006 mm. This was the single largest error source found.

**Perpendicular distance between 3D line fits, not point-to-point.** Two points
on parallel lines at different heights are `sqrt(gap² + Δheight²)` apart. The
2026-08-19 draft asserted that height matching "isn't required because a flat
rigid target's spacing doesn't depend on where along the lines you measure" —
that is false for a Euclidean point distance. Measured on the synthetic ladder,
naive point-to-point read **+629 %** on a 3 mm gap and +11 % on a 45 mm gap.

**Centre-to-centre via the darkness centroid.** Laser kerf, marking width,
exposure and blooming all widen a mark symmetrically, and a symmetric profile's
centroid does not move. Verified: mark widths of 0.25, 0.5 and 1.0 mm give
identical results (residual 0.0396 / 0.0396 / 0.0398 mm).

**Projection-based detection, not Hough.** See below.

## What changed and why (vs. the 2026-08-19 plan)

- **Hough was replaced.** The plan specified Canny → `HoughLinesP` → cluster by
  midpoint coordinate → `cv2.fitLine`. Two failures: clustering a *tilted* line
  by its raw midpoint x is wrong, since the midpoint drifts by
  `extent · tan(tilt)` — 175 px across a 2000 px line at only 5°, far beyond any
  sane cluster distance, and the capture protocol deliberately includes tilted
  poses. And Hough under-detected badly on clean images: 9 segments recovered
  from 12 real edges, causing silent mis-indexing (a +69 % scale error in one
  run). The projection detector uses every mark pixel and succeeded on mark
  widths from 0.25 to 2.0 mm where Hough failed on 3 of 4.
- **No fundamental-matrix bug.** The draft claimed `extrinsics.fundamental` was
  wrong for undistorted points and had to be recomputed from `E`. **This was
  incorrect** — `cv2.stereoCalibrate` already returns `inv(K_b)ᵀ·E·inv(K_a)`,
  verified identical to 1.2e-15 relative on this rig's `extrinsics.json`. The
  code still recomputes it, for independence from an optional stored field, but
  it is not a correction and must not be described as one.
- **`register_pipeline.py` is gone**; `default_extrinsics_path` comes from
  `registration_io.py`.
- **`triangulate.py` is reused** rather than reimplemented.
- **All pairwise distances**, not only adjacent gaps: six marks give fifteen
  lengths spanning 3-45 mm from one capture, and a real lever arm for the fit.
- **Output** goes to `calibration/results/line_accuracy/` (already git-ignored,
  and beside `cross_validation/`, its methodological sibling) rather than a new
  top-level `line_accuracy_reports/`.

## Interfaces

- `calibration/line_target.py` — `LineLadderTarget` frozen dataclass following
  `target_board.TargetBoard`: `gaps_mm`, `orientation`, `line_width_mm`,
  `gaps_uncertainty_mm`, `measured_by`; properties `line_count`, `positions_mm`,
  `span_mm`, `strictly_increasing`; `pair_distances_mm()`.
- `calibration/config/line_target.yaml` — the plate's measured geometry.
- `config.yaml` — `geometric_calibration.line_target` and `line_target_captures`.
- `check_line_accuracy.py` — argparse CLI following `check_registration_error.py`;
  `--no-refine` and `--no-conditioning-guard` escape hatches, both defaulting on,
  with the configuration recorded in `result.json`.

## Reporting

`report.txt`, `result.json`, one annotated JPEG per camera per session, and
`line_accuracy.png`. The headline is the fit `measured = a · true`: `a − 1` is
the systematic scale error, which a mean absolute error cannot separate from
random scatter. A companion `a · true + b` fit distinguishes a localisation bias
(large `b`) from a scale error. Within-session point noise (3D line straightness)
is reported separately from across-session repeatability, rather than pooled.

## The physical plate

Three constraints matter more than anything in the code:

- **Span ≤ 45 mm.** Camera A sees ~87 × 66 mm at the 0.155 m mean working depth
  and the stereo overlap is smaller (17.6° vergence).
- **Surface-mark, do not cut through.** At ±8.8° vergence a through-slot in a
  `t` mm plate shows its front and back edges differently to each camera:
  `t · tan(8.8°)`, or 0.31 mm at `t = 2 mm`. Marked anodised aluminium is ideal;
  foil ≤ 0.2 mm if cutting through is unavoidable.
- **Measure the as-marked plate.** `gaps_mm` is the accuracy ceiling of the whole
  experiment. Laser positioning of ~0.05 mm over 45 mm is 0.11 %, enough to
  resolve a 0.2-0.5 % calibration scale error — but only if verified rather than
  assumed. `measured_by` records which.

**The plate in use (2026-08-21): 10 marks, uniformly 5 mm apart, ~20 mm long,
span 45 mm, giving 45 pairwise distances from 5 to 45 mm.** Spacing is nominal
and user-stated; there is no CAD and it has not been verified with calipers, so
`measured_by` still carries "NOT verified" and the script warns on every run.

Uniform spacing costs a safeguard. With unequal gaps the detected marks fit the
known pattern only one way, so a mark lost off the frame edge breaks the affine
fit and is caught. With equal gaps every subset of the ladder is still a valid
ladder, the affine check cannot see a missed mark, and if the two cameras lose
*different* marks then A's line i is matched against B's line i+1 and the result
is wrong without looking wrong. Two mitigations:

- **Count, not pattern.** The detector refuses any session that does not yield
  exactly `line_count` marks, and also refuses one where an extra mark-like peak
  is at least `EXTRA_PEAK_FRACTION` as strong as the weakest accepted mark. A
  scratch, plate edge or reflection is far likelier than a genuine extra mark.
  Verified by planting a spurious 11th mark: the session is skipped, not measured.
- **Procedural.** Frame the whole plate with clear margin in *both* cameras.

At 5 mm spacing the refinement window can exceed half a gap, which would let a
neighbouring mark's darkness pull the centroid and bias the measured distance, so
it is clamped to `REFINE_SPACING_FRACTION` of the narrowest spacing seen in frame.

Ground truth must come from the **span**, not a single gap: 0.05 mm calipers on
one 5 mm gap is 1 %, the same size as the error being hunted, whereas 0.05 mm
across the full 45 mm is 0.11 %.

## Verified

Synthetic, no hardware needed. With a 15° rotated `R` and non-zero distortion the
pipeline recovers 3D points to 7.6 nm and all fifteen distances to 5.2e-6 %. End
to end through the real CLI on four rendered poses spanning 130-175 mm with
tilt: **scale error +0.000 %, residual 0.0006 mm rms, max 0.0005 mm** — roughly
100× below the plate's own 0.05 mm fabrication uncertainty, so the measurement is
limited by the artefact, not by this code. An injected +1 % scale is recovered as
+1.0000 %.

Re-verified on the actual 10-mark uniform 5 mm plate geometry (same four poses,
20 mm marks): **scale error -0.025 %, residual 0.0177 mm rms, max 0.0240 mm**.
Worse than the unequal 6-mark case because the marks sit three times closer
together, but still ~40× inside the 1 % bar and below the plate's own uncertainty.

## Out of scope

Thermal / FLIR support; two-orientation grids; live capture; monocular operation;
non-flat or non-rigid targets.
