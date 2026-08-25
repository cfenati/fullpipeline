# Design: fix check_depth_accuracy.py's GUI freeze, switch to batched multi-point plane collection

Date: 2026-08-25
Status: Proposed

Companion to `docs/superpowers/specs/2026-08-25-depth-accuracy-interactive-fix-design.md`
(the prior redesign this one supersedes in part) and
`docs/superpowers/specs/2026-08-24-check-depth-accuracy-design.md` (background:
physical grid, plane-fit method, why the reference plane comes first). Read
those first.

## Purpose

The prior redesign's `LabelingSession` calls Python's blocking `input()` from
inside `on_mouse` -- a callback OpenCV invokes during `cv2.waitKey()`. OpenCV's
window only processes OS events (repaint, close, move) while `cv2.waitKey()` is
running; the moment `input()` blocks on the terminal, the window stops
responding entirely. This was tried on real hardware for the first time this
session and reproduced exactly as the architecture predicts: the window
freezes and gets force-killed. This is a known OpenCV anti-pattern (blocking
I/O inside a GUI callback), not a one-off bug -- no amount of patching the
*existing* per-click-prompt design fixes it; the prompt itself has to move out
of the callback entirely.

Separately, real use exposed a second, independent problem with the
single-point-per-block model: a lone click's triangulated Z is one noisy
sample. The user wants what they originally proposed back at the start of this
work -- multiple points per plane, both for the reference and for each block
-- because averaging (or, better, using the already-well-constrained reference
normal) is more stable than trusting one point.

Both problems share one fix: replace "prompt after every click" with "click
freely, press a key when a batch is done," where the key press is handled
inside the same `cv2.waitKey()` loop that's already pumping window events (no
blocking terminal I/O, ever), and each batch is a plane's worth of points
instead of one point per block.

## Method

### 1. In-window modal text entry, replacing `input()`

`run_interactive` gains a small built-in text-entry mode: pressing a key
starts it (see Method 2), each subsequent key event is either a digit, `,`,
Backspace, Enter, or Esc, and the accumulated text renders live in the status
bar. Enter hands the finished string to a callback; Esc cancels back to normal
click mode. This is generic -- `run_interactive` has no idea the text means
"row,col" -- so it stays reusable the way `on_point`/`on_undo` already are.

No blocking call exists anywhere in the interactive loop after this change.
The window keeps processing `cv2.waitKey()` continuously regardless of
whether the user is clicking or typing.

### 2. Two hooks replace the current per-click prompt

- `on_advance() -> Optional[str]` -- fires on **n**. The driver closes out the
  currently-open batch. If it doesn't have enough points yet (the reference
  batch needs >= 3 for a plane fit; a block batch has no hard minimum, even 1
  point is meaningful -- see Method 4), it prints why and returns `None`,
  staying in click mode. Otherwise it finalizes the batch (fits the plane, if
  this was the reference; computes and prints the mean + spread, if this was
  a block) and returns a prompt string (e.g. `"cell row,col > "`), which
  `run_interactive` uses to enter text-entry mode.
- `on_text_submit(text: str) -> Optional[str]` -- fires on Enter during
  text-entry mode. The driver parses `row,col`, validates it's in range for
  the target's grid. Returns `None` to accept (back to click mode, ready for
  the next batch); returns an error string to redisplay and stay in
  text-entry mode for another attempt.

`on_point`/`on_undo` (from the prior redesign) are kept, but `on_point`
changes from "prompt for a label" to "silently accumulate this click's index
into the currently-open batch" (a lightweight `print` is fine -- prints don't
block).

### 3. Session shape

No prompt is needed to start: the tool begins in the reference batch
implicitly, matching today's instructional print ("click the reference
corners first"). Click freely; press **n**. If fewer than 3 points were
clicked, it says so and keeps waiting. Once accepted, the plane fits
immediately (matches the prior redesign's live plane-RMS feedback), and the
tool prompts for the first block's `row,col` via the new text-entry mode.
Type it, Enter. Click that block's points -- any number, including just one.
Press **n** -- the block's mean distance and spread (if more than one point)
print immediately, and the tool prompts for the next block's `row,col`.
Repeat. **q**/Esc ends the session, same as today.

### 4. Block point count: minimum 1, not 0, by design

`on_advance` requires `>= 1` point for a block batch (reject with "click at
least one point before pressing n" if attempted at 0) and `>= 3` for the
reference batch. A block only needs as few as 1 point (not more) precisely
because of a decision already made
in `docs/superpowers/specs/2026-08-25-depth-accuracy-interactive-fix-design.md`
("Why the reference plane comes first"): every block-top is part of the SAME
rigid printed object as the reference corners, so it shares the reference
plane's normal exactly -- there is no independent tilt to solve for. A
block's "distance" is the mean of its points' individual perpendicular
projections onto that already-established normal, not a fresh 3-point plane
fit with its own normal. Averaging more points reduces noise (this is the
"more stable" the user asked for) but there's nothing structurally requiring
3; 1 point degrades gracefully to exactly today's single-click measurement,
just without a forced prompt after it.

Fitting an independent plane per block (the alternative considered and
rejected) would need >= 3 well-separated points crammed onto a 10x10mm face
at ~180mm standoff -- both harder to achieve than on the 50x50mm reference
footprint, and numerically fragile even when achieved (a plane fit from
closely-spaced points is dominated by noise in exactly the direction that
matters). Reusing the reference's already-well-constrained normal sidesteps
this entirely.

### 5. Undo

`u` still does exactly one thing at the `run_interactive` level: remove the
single most recent completed click. The driver's `on_undo(new_count)` tracks
which batch that click belonged to and shrinks it. If the removed click was
the last point of an already-*finalized* batch (the user pressed `n`, then
`u`), that batch reopens as the current one -- no re-entry of its `row,col`
needed, since the label is still recorded; the user just resumes clicking or
presses `n` again once satisfied.

One sequencing edge case: if the user has already confirmed the *next*
batch's `row,col` (text-entry accepted, a new empty batch is now current)
before clicking anything for it, and then presses `u`, there's nothing in
the current batch to remove -- the removal falls through to the previous
(finalized) batch's last point, exactly as above. The empty pending batch
and its already-confirmed label are simply discarded when the current-batch
pointer moves back; nothing downstream ever sees a zero-point batch.

### 6. `--cell` reverts to `ROW,COL,AX,AY,BX,BY`

The prior redesign's principle -- interactive and scripted sessions identify
a block the same way -- still holds; only which identifier that is has
changed back. `cell_at_height` and the tolerance-aware uniqueness check in
`DepthGridTarget.__post_init__` (added to support it) have no remaining
caller once this lands and are deleted, not kept around unused.

## What does not change

`measure_depth_session`, `aggregate_depth_results`, `write_report`, and the
report format are untouched. `measure_depth_session` already accepts
`cell_labels`/`cell_clicks_a`/`cell_clicks_b` as parallel per-CLICK arrays
(one entry per click, repeats allowed, grouped by label internally) -- a
batch is just a different way of producing those same parallel arrays (every
click in a batch repeats that batch's label), not a different data shape.
The per-block repeatability pooling added in the final review of the prior
redesign (raw per-click pooling across sessions) applies unchanged.

## Interfaces

Changed:
- `measure_points.run_interactive` -- adds `on_advance`/`on_text_submit`
  hooks and the built-in text-entry mode; `on_point`'s existing contract is
  unchanged (still fires once per completed click, in order), only what the
  *caller* (the driver) does with it changes. `on_undo`'s contract is
  unchanged.
- `check_depth_accuracy.py`:
  - `LabelingSession` rewritten around batches instead of per-click labels,
    renamed to `PlaneCollectionSession` to reflect this.
  - `main()`'s interactive branch wired to the new hooks; expands each
    finalized batch into parallel per-click arrays before calling
    `measure_depth_session` exactly as before.
  - `--cell` CLI flag and its parsing revert to `ROW,COL,AX,AY,BX,BY`.
  - Module docstring's usage example reverts to match.
- `calibration/depth_grid_target.py` -- `cell_at_height` and the
  tolerance-aware duplicate-height check in `__post_init__` are deleted
  (no remaining caller). `height_at` and `pair_depths_mm` are unchanged.

Unchanged: `measure_depth_session`, `aggregate_depth_results`, `write_report`,
`fit_plane_3d`, `perpendicular_distance_to_plane`, everything about
`--ref`'s CLI shape, the annotated-JPEG output, `result.json`'s shape.

## Out of scope

Everything the prior two designs already scoped out (dense FoundationStereo,
absolute depth accuracy, automatic block detection, thermal/FLIR, live
capture, multi-pose sessions, a geometry-guided overlay, a report chart).
Additionally: no change to how the reference batch's minimum (3) is chosen
or validated beyond "enough for a plane fit" -- not tied to the target's
`reference_corner_count` field as an exact count any more, since the whole
point of this redesign is that the count is no longer fixed up front.

## Verification plan

Same constraint as before: the actual click-through-a-window path cannot be
exercised by script, and the whole point of this design is a bug that only
manifested on real hardware -- the previous redesign's synthetic/CLI-driven
verification could not have caught it, and this one won't be able to fully
prove it out either. What CAN be verified by script: (a) the modal text-entry
state machine in `run_interactive` (digit/backspace/enter/esc handling,
building up a string) via direct calls with synthetic key codes, no real
window needed for the state-machine logic itself; (b) the batch-driver logic
(`on_point`/`on_advance`/`on_text_submit`/`on_undo`) via direct calls with
synthetic arguments, mirroring how the prior `LabelingSession` was verified;
(c) an end-to-end CLI subprocess check via the reverted `--ref`/`--cell
ROW,COL,...` non-interactive path, confirming `measure_depth_session` still
produces correct results when fed a batch's worth of same-labeled clicks.
Trying a real session on the actual rig remains the user's own next step
after this lands, same as it was before -- this is explicitly not something
this plan can substitute a script for.
