# Design: fix check_depth_accuracy.py's interactive labeling and diagnostics

Date: 2026-08-25
Status: Proposed

Companion to `docs/superpowers/specs/2026-08-24-check-depth-accuracy-design.md`,
which this document assumes as background and does not repeat (the depth-grid
target's physical layout, the plane-fit method, the pairwise-vs-plane-to-cell
distinction). Read that first.

## Purpose

`check_depth_accuracy.py` was implemented and synthetically verified on
2026-08-24, but blocked on a physical target. The target has since been
fabricated and the tool has now been run on real captures
(`captures/depth/20260824_09*`, results in
`calibration/results/depth_accuracy/`) -- and every session's result is
useless: `report.txt` shows the same `(row, col)` label recorded 2-4 times per
session (e.g. `(1,2)` four times with different measured values), "errors" of
3-7 mm against a target with a 0.05 mm uncertainty budget, and a pairwise
scale fit of `a = nan`.

Investigating those real results (not a fresh guess) found two distinct,
concrete bugs, confirmed against the actual code:

1. **Label capture is blind and post-hoc.** The interactive session collects
   every click first; only after the window closes does the script ask the
   user to type `row,col` for each non-reference click, identified purely by
   its position in click order, with nothing on screen linking a label to
   what was actually clicked. This produced the duplicate labels seen in the
   real sessions.
2. **The epipolar-offset diagnostic -- the one number meant to catch a bad
   correspondence -- always reads 0.0 in interactive sessions.**
   `run_interactive`'s `on_mouse` stores the click in B only *after* snapping
   it onto the epipolar line (`state["clicks_b"].append(snapped.tolist())`);
   the raw pre-snap click is discarded. Every later recomputation of
   `epipolar_offset_px` (both the live print and the final report) runs on
   this already-snapped point, so the reported "distance from the epipolar
   line" is trivially ~0 regardless of how far off the real click was. This
   bug is not depth-specific -- `measure_points.py`'s own interactive report
   has the identical defect -- but it is what left the user with **no visor
   into whether their real problem (self-reported: imprecise correspondence-
   finding between the two cameras) was actually occurring.**

A third bug is a direct consequence of (1): `DepthGridTarget.pair_depths_mm`
generates a truth-difference for every pair of clicked cells, including pairs
that are actually the same mislabeled cell twice (truth = 0). `fit_scale`
computes `scale = truth @ measured / (truth @ truth)` -- when every pair in a
session shares the same (wrong) label, `truth @ truth` is exactly 0, hence
`a = nan`.

The click-and-triangulate mechanics themselves (`measure_points.py`'s
two-panel click / epipolar-line-assist / loupe UI) are confirmed **not** the
problem -- they're reused unmodified for ordinary length measurement and work
fine there. This design touches only the depth-grid-specific labeling and
reporting layer, plus the one shared bug in `run_interactive` that both tools
happen to inherit.

## Method

### 1. Label at click time, using the number printed on the block

Today: click everything, close the window, then type a label for every click
in order, from memory. New: `run_interactive` gains two small optional hooks
(default `None`, so `measure_points.py`'s own call site is unaffected):

- `on_point(index, result)` -- called right after a click pair (A + B) is
  completed and `recompute()` has run. `result` is the same dict
  `recompute()` already builds (`points_mm`, `depth_mm`, raw
  `epipolar_offset_px` -- see bug 2 below).
- `on_undo(new_count)` -- called whenever `u` or `r` changes the completed
  point count, so a caller keeping its own per-point bookkeeping (labels)
  can stay in sync by truncating to `new_count`.

`check_depth_accuracy.py` uses these to drive a stateful callback:

- Points `0 .. reference_corner_count - 1`: no label needed (the corners are
  interchangeable plane-fit inputs). Print a running count
  (`"corner 2/4 recorded"`). The instant the last corner lands, fit the
  reference plane immediately and print its RMS -- so a bad set of corner
  clicks is caught before the user spends ten minutes clicking blocks against
  a bad datum, instead of finding out at the very end.
- Every point after that: prompt in the terminal, *while the block just
  clicked is still fresh*, for the height number engraved on it (e.g.
  `"engraved height (mm) > 3.6"`). Reverse-look-up `(row, col)` from that
  value against the target (see "Interfaces"); re-prompt with the list of
  valid heights on a miss. Immediately triangulate against the already-fit
  plane and print the comparison for that one block:
  `"3.6mm: measured 3.52mm (delta -0.08mm)"`.
- A label matching one already used earlier in the session is recognized as
  a repeat, not a new cell, and reported as a repeatability sample:
  `"3.6mm, sample 2/2: measured 3.60mm (spread so far: 0.057mm rms)"`.

Reading a number off the physical block removes the one thing that was
actually hard to get right by eye -- the grid's `(row, col)` indexing depends
on a mounting/mirroring convention (see the 2026-08-24 design's "Occlusion"
section) that has no reason to match how the block looks sitting on a bench.
The engraved number has no such ambiguity.

Nothing about clicking, panning, zooming, or the loupe changes.

### 2. Fix the epipolar-offset diagnostic (in `measure_points.run_interactive`)

Capture the offset at the moment it's still knowable -- after any blob-snap,
before the projection onto the epipolar line that discards it:

```python
blob = snap_to_blob(gray_b, full, blob_radius)
raw_offset = float(np.abs(blob[0] * line[0] + blob[1] * line[1] + line[2]))
snapped = snap_to_line(blob[None, :], line[None, :])[0]
...
state["click_offsets_px"].append(raw_offset)
```

`state["click_offsets_px"]` (one real number per completed point, in click
order) overrides the trivial recomputed `epipolar_offset_px` in the dict
`run_interactive` returns, and is what both the existing live print and the
new `on_point` callback see. This fixes the diagnostic for
`measure_points.py`'s own interactive sessions too, not just the depth tool's.

### 3. Aggregate same-label clicks before building the pairwise table

`measure_depth_session` groups completed clicks by label first. Each unique
`(row, col)` becomes one entry: `cell_measured_mm` (mean of its samples),
`cell_samples` (count), `cell_repeatability_rms_mm` (RMS about the mean,
`None` when `cell_samples == 1`). `pair_depths_mm` then runs over this
deduplicated set, exactly as the 2026-08-24 design already intended --
"all heights in this target are distinct" was already a stated invariant,
just not one the old click-order-labeled flow could actually guarantee. This
makes the `truth @ truth == 0` failure structurally unreachable: a pairwise
truth of 0 mm can now only happen if the target itself had a duplicate
height, which is now rejected at load time (see "Interfaces").

This also settles a case the old code treated as an error: a session that
visits only **one** distinct cell, clicked repeatedly (a pure repeatability
probe, in the spirit of the user's own "many points on one plane" idea, just
scoped to one labeled block instead of a separate mode). `pair_depths_mm` on
a one-cell set simply produces zero pairs -- no error -- so such a session is
now valid and scored: it contributes its `cell_repeatability_rms_mm` and
plane-to-cell numbers, but nothing to the pairwise scale fit (which needs
>= 2 distinct cells to have any pairs at all). The old
`ValueError("need at least 2 cells...")` and the interactive skip threshold
(`total_clicks < n_ref + 2`) both relax accordingly: a session is now
acceptable once it has the `reference_corner_count` corners plus **one**
labeled cell click, not two.

## Reporting

`report.txt` gains a one-line headline above everything else:

```
RESULT: pairwise scale error +0.42 %  (residual 0.031 mm rms)  --
        6 cell(s) across 3 session(s), repeatability 0.028 mm rms
```

so the answer to "is this within budget" doesn't require reading the tables.
The per-cell table gains `samples` and `repeatability mm` columns (blank when
`samples == 1`). Everything else -- the pairwise scale fit, the per-cell
plane-to-cell table, the measurement-quality section, the per-session table --
keeps its current shape; only the inputs feeding it change (deduplicated
cells instead of raw clicks).

No chart this round -- raised and discussed: the "measured" values plotted
would still be genuine relative-depth quantities (plane-to-cell and
cell-to-cell differences, never a raw absolute Z; see the 2026-08-24 design's
"Two comparisons, not one" section), so it wasn't a correctness concern, but
it's lower priority than the two real bugs and can be added later once real
sessions confirm the fix.

## Interfaces

Changed:

- `measure_points.run_interactive` -- adds optional `on_point` and `on_undo`
  keyword parameters (default `None`, backward compatible); fixes
  `epipolar_offset_px` in its return value to reflect the real pre-snap
  click, not a recomputation on an already-snapped point.
- `calibration.depth_grid_target.DepthGridTarget` -- adds
  `cell_at_height(height_mm, tol=1e-6) -> Tuple[int, int]` (reverse lookup,
  raises with the sorted list of valid heights on no match);
  `__post_init__` now rejects a grid with any duplicate height (this was
  already an assumed invariant -- now it's enforced, since the label scheme
  depends on it).
- `check_depth_accuracy.py`:
  - Interactive flow rewritten around the new hooks, replacing
    `_prompt_cell_labels` (deleted).
  - `measure_depth_session` groups clicks by label before building the
    pairwise table (see Method 3); its return dict gains `cell_samples` and
    `cell_repeatability_rms_mm` per cell.
  - `--cell` CLI flag changes from `ROW,COL,AX,AY,BX,BY` to
    `HEIGHT,AX,AY,BX,BY`, so scripted and interactive sessions identify a
    cell the same way. `--ref` is unchanged (corners carry no label).
  - `write_report` adds the headline line and the two new per-cell columns.

Unchanged: everything about `--ref`, the plane fit itself
(`fit_plane_3d`/`perpendicular_distance_to_plane`), the annotated-JPEG output,
`result.json`'s overall shape (extended, not restructured), and every other
CLI flag.

## Out of scope

Everything the 2026-08-24 design already scoped out still applies (dense
FoundationStereo depth, absolute depth accuracy, automatic block detection,
thermal/FLIR, live capture, multi-pose sessions). Additionally, for this
pass: a geometry-guided overlay that projects expected cell positions onto
the live image so no label is ever typed (discussed with the user as a
larger alternative; deferred unless the engraved-number flow above turns out
not to be reliable enough in practice); a chart alongside `report.txt`
(discussed above, deferred not rejected).

## Verification plan

No physical-target-in-hand session is available to me directly, so
verification is: (a) extend the existing synthetic checks from the
2026-08-24 design to cover the new grouping/aggregation path -- specifically,
a synthetic session with deliberately repeated labels must no longer produce
`nan` in the scale fit, and must produce the correct `cell_repeatability_rms_mm`
from injected per-click noise; (b) a unit-level check of the epipolar-offset
fix using a deliberately off-line synthetic B click, confirming the returned
`epipolar_offset_px` matches the known perpendicular distance instead of ~0;
(c) an end-to-end CLI subprocess check via the new `HEIGHT,AX,AY,BX,BY`
`--cell` syntax, mirroring the one already in the 2026-08-24 design. Actually
re-clicking the real `captures/depth/20260824_*` photos correctly is a manual
step for the user (it requires the interactive tool itself to get accurate
pixel coordinates) -- not something this plan can substitute a script for.
