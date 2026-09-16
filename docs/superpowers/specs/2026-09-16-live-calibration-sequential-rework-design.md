# Sequential live calibration rework — design

## Problem

`calibrate_live.py`/`calibration/live_capture.py`/`live_gui.py` (built
2026-09-15, unverified on real cameras per memory) has three problems that
only became apparent thinking through actual use:

1. **Startup stall / cross-session pollution.** `LiveCalibrationSession.__init__`
   calls `seed_from_existing()`, which re-detects the ChArUco board in every
   session folder already sitting in `captures_dir` (`calibration/live_capture.py:182-212`),
   synchronously, before the GUI even opens. Since `captures_dir` defaults to
   a long-lived shared folder (`captures/stereo`), this both re-scans
   everything from every past run (slow, grows without bound) and folds
   unrelated old sessions into this run's fit — the opposite of what a fresh
   calibration attempt should do.
2. **Freeze on every capture.** `LiveCalibrationSession.tick()`
   (`calibration/live_capture.py:308-371`) does two full-resolution
   `cv2.imwrite()` calls and two full-resolution `detect_board_in_frame()`
   calls synchronously on the main/GUI thread before returning. Nothing pumps
   the Tk event loop during that stretch, so the GUI visibly freezes at the
   instant of every capture. (Live re-*fitting* was already moved to a
   background thread per the comment at `live_capture.py:376-388` — that part
   works; it's the per-capture save+detect step that still blocks.)
3. **Overlapping, not sequential, capture.** `tick()` always saves and gates
   on *both* cameras together via a shared-corner stereo gate
   (`check_gate()`/`pair_observations_from_boards`), so cam1 intrinsics,
   cam2 intrinsics, and the stereo pair are all being filled in from the same
   mixed pile of views at once. The user wants dedicated cam1 views, then
   dedicated cam2 views, then dedicated stereo views — each phase precise on
   its own terms before moving on — plus a way to see which parts of the
   frame still need coverage instead of discovering gaps only after a bad fit
   forces reshooting ~50% of a phase.

## Scope

Rework the same three files (`calibration/live_capture.py`,
`calibrate_live.py`, `live_gui.py`) into a **three-phase sequential** tool:
cam1 intrinsics → cam2 intrinsics → stereo pair. Simplicity and capture speed
are explicit priorities — as few images as will reliably fit, not as many as
can be gathered. `calibration/opencv_calibrate.py`, `calibration/stereo.py`,
`calibrate_cameras.py`, `stereo_calibrate.py`, and `prune_calibration.py` are
all reused unchanged; this is a rework of the live-driving loop and GUI, not
of the fitting math.

(Side note, not in scope: `calibration/calibrate_cameras.py` is a byte-identical,
untracked, never-committed duplicate of the real root-level
`calibrate_cameras.py` that `calibrate_live.py` already imports from —
looks like a stray copy from earlier work. Flagging it; not touching it here.)

## Phase state machine

Three phases, driven in order, each with its own target view count:

| Phase | Camera(s) gated | Target views | Purpose |
|---|---|---|---|
| 1 | camera-a only | 20 (`--mono-views`) | cam1 intrinsics |
| 2 | camera-b only | 20 (`--mono-views`) | cam2 intrinsics |
| 3 | both, shared-corner gate | 15 (`--stereo-views`) | stereo extrinsics |

Both `RGBCamera`s open once at startup and stay open for the whole run — no
per-phase re-open/close — but during phases 1 and 2 only the relevant
camera's frame is graded against the gate and saved; the other camera is
simply not grabbed that tick (`RGBCamera.grab()`, not `grab_pair()`). Phase 3
reuses today's shared-corner `check_gate()`/`pair_observations_from_boards`
gate unchanged.

Target counts are **minimums, not caps**: reaching the target auto-triggers
that phase's fit (below), but capture can continue past it (top-up flow).

## Per-run captures directory (fixes stall + cross-session pollution)

Each run creates its own subfolder, `<captures_dir>/live_<run_timestamp>/`,
and only ever reads/writes inside it — mirroring the existing
`live_<timestamp>` naming already used for `--out`. `seed_from_existing()` is
deleted outright: nothing is re-detected from disk at startup, so startup is
instant regardless of how many old sessions exist elsewhere under
`captures_dir`, and a run never mixes in captures from a different run. Losing
resume-into-an-old-folder is an intentional trade for a fast, uncomplicated
first version; if resuming a specific run is wanted later, pass that run's
`live_<timestamp>` folder itself as `--in`.

## Per-capture work: async detection, coverage map, no live full-fit

On each auto-capture:

1. Save the frame(s) synchronously — fast, was never the freeze source.
2. Hand the frame to a background thread that runs `detect_board_in_frame()`
   (corner extraction only) and appends the `BoardObservation` to that
   phase's in-memory list. Nothing about a full `calibrate_intrinsics()` /
   `calibrate_stereo()` solve happens per capture.

This removes the synchronous full-res detection from the main thread (fixing
the freeze) and, as a side effect, gives a running observation list the GUI
can turn into a live coverage thumbnail every loop tick via the existing
`coverage_image()` helper — cheap enough to redraw continuously, so gaps
(e.g. "nothing in the left 15% of frame") are visible *while shooting*, not
only after a fit comes back with a warning.

## Phase-end fit: reuse the offline scripts, not a new solver

When a phase's target is reached (or "Refit now" is clicked):

1. Shell out to `calibrate_cameras.py` (phases 1-2) or `stereo_calibrate.py`
   (phase 3) into this run's `output_root`, exactly like today's
   `run_finish()` already does per-camera — guaranteeing the result is
   byte-identical to the offline path.
2. Load the written `intrinsics.json`/`extrinsics.json`
   (`CameraIntrinsics.load_json()` / `StereoExtrinsics.load_json()`).
3. Compute `quality_warnings()` / `stereo_quality_warnings()` in-process using
   the observation list already accumulated for the coverage map — no
   refitting, just the same warning computation `live_capture.py` already
   does today, now run once per phase instead of once per tick.

If warnings come back, the phase-result panel offers, in order of
preference:

- **Capture N more** (`--top-up`, default 5) — resumes auto-capture for the
  phase, *keeping* the existing views, then re-runs step 1-3 automatically
  once the new target is hit.
- **Refit now** — re-run step 1-3 immediately on the current view set (e.g.
  after manually filling a coverage gap the map pointed out).
- **Discard phase & restart** — clears this phase's captured sessions and
  observation list, back to 0. Manual fallback, not the default outcome; kept
  for when the sane fix really is starting over (e.g. the board itself was
  reconfigured).

A phase can also be advanced past without being warning-free — "ready" is
informational (matches the existing card language: "works anytime, ready or
not"), not a hard gate.

## Phase 3 completion

Once the stereo phase's fit is accepted, run the existing `review_and_prune()`
unchanged (bad-session detection + optional pruning), then print the same
promote-to-canonical instructions `run_finish()` prints today. No separate
"Finish" step is needed — each phase already wrote its result incrementally.

## GUI (`live_gui.py`)

- Header: `Phase 1/3: cam1 intrinsics — 7/20 captured`.
- One preview panel for phases 1-2 (just that phase's camera), two
  side-by-side for phase 3 — reusing the existing board-overlay/flash-on-capture
  annotation code.
- Coverage thumbnail always visible next to the preview (not gated behind
  reaching the target), updating as async detections land.
- On target reached: a result panel with RMS/views/warnings and
  Continue / Capture N more / Refit now / Discard & restart buttons.
- Status bar: pause/resume, phase capture count, last-save timestamp, open
  captures folder — same as today.
- Final screen after phase 3 + prune: run summary instead of a "Continue"
  button.

## CLI surface

```
python calibrate_live.py [--camera-a rgb_cam1] [--camera-b rgb_cam2]
                          [--mono-views 20] [--stereo-views 15] [--top-up 5]
                          [--interval 3.0] [--min-corners N] [--min-shared-corners N]
                          [--board ...] [--in ...] [--out ...] [--no-preview]
```

Dropped from today's CLI: `--recalibrate-every`, `--reject-sigma`,
`--min-views-before-rejection` — outlier rejection now happens entirely
inside the offline scripts being shelled out to, not reimplemented in the
live tool.

## File impact

- `calibration/live_capture.py` — largely rewritten: phase state machine,
  per-phase observation accumulation via background-thread detection, subprocess-based
  phase-end fit + in-process warning computation, top-up/discard controls.
  `seed_from_existing`, `MIN_VIEWS_BEFORE_REJECTION` gating, and the
  per-tick `_recalibrate_worker` full-solve path are removed.
- `calibrate_live.py` — main loop restructured around phases (open cameras
  once, drive gate+capture per active phase, call into phase-fit/top-up/discard,
  final `review_and_prune()` call unchanged).
- `live_gui.py` — reworked layout per above; keeps `fit_to_box` reuse from
  `gui.py` and the existing flash/overlay drawing code.
- Unchanged: `calibration/opencv_calibrate.py`, `calibration/stereo.py`,
  `calibrate_cameras.py`, `stereo_calibrate.py`, `prune_calibration.py`.

## Out of scope / deferred

- Resuming a specific prior `live_<timestamp>` run's in-progress phase state
  (possible by pointing `--in` at that folder, but no dedicated "resume" UX).
- Per-block plane-fit style coverage scoring — this is a 2D scatter/heatmap
  of raw corner hits, same as the existing `coverage_image()`, not a
  quantitative "% of frame covered" metric.
- Any change to the fitting math, outlier-rejection thresholds, or warning
  thresholds inside `opencv_calibrate.py`/`stereo.py` — this rework only
  changes when/how those existing functions are invoked.
