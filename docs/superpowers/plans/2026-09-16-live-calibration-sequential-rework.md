# Sequential Live Calibration Rework Implementation Plan

> **Status: implemented in a single session (2026-09-16).** Executed
> directly rather than dispatched task-by-task to a fresh agent, so every
> step below is a record of what was actually built and verified, not a
> prospective plan. See
> `docs/superpowers/specs/2026-09-16-live-calibration-sequential-rework-design.md`
> for the full design rationale — read that first.

**Goal:** Replace `calibrate_live.py`'s overlapping, freeze-on-capture,
seed-everything capture loop with a three-phase sequential tool (cam1
intrinsics → cam2 intrinsics → stereo pair), each phase fit once at its
target view count via the existing offline scripts, with a live coverage map
so gaps get filled during the shoot instead of forcing a reshoot afterward.

**Architecture:** `calibration/live_capture.py:SequentialLiveCalibrationSession`
owns a `Phase` state machine and per-phase observation dicts. Per-capture
work is a fast synchronous save plus a background-thread
`detect_board_in_frame()` call whose result is queued and drained by the
main thread's `poll()` — no per-capture fit, so nothing blocks the GUI
thread. Reaching a phase's target (or a manual "Refit now") shells out to
`calibrate_cameras.py`/`stereo_calibrate.py` exactly as the old
`run_finish()` did, then computes `quality_warnings()`/
`stereo_quality_warnings()` in-process from the same observations used for
the coverage map. `calibrate_live.py` drives the phase loop against real
cameras; `live_gui.py` renders one phase at a time.

**Tech Stack:** Python 3.9, OpenCV (`cv2.aruco`), NumPy, Tkinter/PIL — all
already used elsewhere in this repo. No new dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-16-live-calibration-sequential-rework-design.md`.
- No test suite or linter in this repo (per `CLAUDE.md`) — verified with
  throwaway `python3` scripts (fixture-based logic checks, a real synthetic-
  ChArUco end-to-end run, and a real Tk smoke test since this environment
  has a display), run during implementation and not committed.
- `from __future__ import annotations` + type hints on every signature.
- `calibration/opencv_calibrate.py`, `calibration/stereo.py`,
  `calibrate_cameras.py`, `stereo_calibrate.py`, `prune_calibration.py` are
  all reused unchanged — this rework only changes when/how they're invoked,
  never their fitting math or warning thresholds.
- Per-capture work never runs a `calibrateCamera()`/stereo solve — only
  `detect_board_in_frame()` (corner extraction). The fit runs once per phase.
- Each run captures into its own fresh `<captures_dir>/live_<run_timestamp>/`
  subfolder — nothing is read from, or mixed in from, a previous run.

---

### Task 1: `calibration/live_capture.py` — phase state machine

**Files:** `calibration/live_capture.py` (full rewrite).

- [x] `Phase` enum (`CAM_A`, `CAM_B`, `STEREO`) and `PHASE_ORDER` tuple.
- [x] `GateStatus` dataclass kept as-is (open/reason/left/right).
- [x] `PhaseFitResult` dataclass: `phase`, `ready`, `view_count`, `warnings`,
  `error`, `intrinsics` (mono only), `extrinsics`/`scatter` (stereo only),
  `coverage`, `edge_gaps`.
- [x] `SequentialLiveCalibrationSession(board, board_path, captures_dir,
  output_root, camera_a, camera_b, min_corners, min_shared_corners,
  min_spread, mono_target=20, stereo_target=15, top_up=5)`. No
  `seed_from_existing()` — startup does zero disk scanning, fixing both the
  slow-startup and cross-session-pollution complaints.
- [x] `check_gate(preview_a, preview_b)`: single-camera gate for `CAM_A`/
  `CAM_B` phases (`_check_gate_mono`, ignores the other camera's frame
  entirely), reuses the existing shared-corner `pair_observations_from_boards`
  gate unchanged for `STEREO`.
- [x] `tick(frame_a, frame_b)`: saves only the frame(s) the current phase
  needs (mono phase 1/2 sessions hold just one camera's `.jpg`, so the
  existing `*/​<camera>.jpg` glob in `calibrate_cameras.py`/
  `collect_session_pairs` naturally excludes them from the other fits — no
  per-phase subdirectories needed), writes `metadata.json`, then hands the
  frame to a background `threading.Thread` (`_detect_worker`) and returns
  the session label immediately.
- [x] `_detect_worker`/`poll()`: the worker only calls
  `detect_board_in_frame()` and pushes an `(epoch, closure)` tuple onto a
  `queue.Queue`; `poll()` (main thread only) drains it and applies the
  closure, so all dict mutation happens on one thread with no locks needed.
  `epoch` (bumped by `request_discard()`/`advance()`) makes `poll()` drop a
  stale in-flight result from a just-discarded/just-advanced phase instead
  of corrupting the new phase's fresh dicts.
- [x] `current_coverage()`/`edge_gaps()`: reuse `coverage_image()` and (new)
  a same-threshold-as-`quality_warnings()` `EDGE_MARGIN_TOLERANCE = 0.12`
  border-gap check over the current phase's accumulated observations — the
  "show coverage to know which borders are missing" ask, live during the
  shoot rather than only in the post-fit warning text.
- [x] `maybe_start_fit()`/`request_refit()`/`_start_fit()`: async (own
  background thread) subprocess-based fit, deduplicated via
  `_last_fit_trigger_count` so reaching a target that was already fit
  doesn't loop-refit forever; `request_refit()` bypasses that dedup.
- [x] `_run_mono_fit()`/`_run_stereo_fit()`: shell out to
  `calibrate_cameras.py --camera <cam> --in <captures_dir> --board
  <board_path> --out <output_root>` / `stereo_calibrate.py --camera-a ...
  --intrinsics-a/-b ... --no-as-built` (identical flags to the old
  `run_finish()`), then `CameraIntrinsics.load_json()`/
  `StereoExtrinsics.load_json()` and `quality_warnings()`/
  `stereo_quality_warnings()` against the observations already accumulated
  for the coverage map.
- [x] `request_top_up(extra)`: raises `_phase_targets[phase]`, which is the
  *only* gating condition for auto-capture in `calibrate_live.py`'s main
  loop — no separate "resume" flag needed.
- [x] `request_discard()`: `shutil.rmtree`s this phase's session folders,
  clears its observation dicts, bumps the epoch.
- [x] `advance()`: refuses (`False`) until `self.phase in self._results`
  (a fit has run at least once, ready or not — matches the design's "ready
  is informational, not a hard gate"); resets the next phase's dicts.

**Verification (all hardware-free, real code paths, no camera):**
- `python3 -c "import calibration.live_capture"` — OK.
- Real `detect_board_in_frame()` against a `cv2.aruco.CharucoBoard.
  generateImage()` render — detected 70/70 corners.
- Fixture-based logic suite (`detect_board_in_frame`/`subprocess.call`
  monkeypatched, hand-built `BoardObservation`/`CameraIntrinsics` fixtures):
  mono gate+capture only saves that camera's file; mono gate ignores the
  other camera; coverage/edge-gap math against a center-only sample reports
  all 4 edges as gaps; fit-trigger dedup (`maybe_start_fit()` fires once at
  target, not again at the same count, fires again after top-up reaches the
  new target); discard clears state, deletes session dirs on disk, and a
  simulated stale in-flight result from before the discard is dropped by
  `poll()`; `advance()` refuses before any fit and resets the next phase.
  All 6 checks passed.
- **Real end-to-end run** (no mocking at all): rendered the actual
  configured board (11×8 ChArUco) at 6 random synthetic perspective warps,
  ran them through `check_gate()`/`tick()`/`poll()` for real, then a real
  `calibrate_cameras.py` subprocess call. It found the board in 6/6 images,
  wrote a real `intrinsics.json`, and the in-process `quality_warnings()`
  call reproduced the exact same 4 warnings the subprocess's own stdout
  reported — confirming the "byte-identical to the offline path" property
  the design called for. (RMS/distortion numbers themselves were nonsense,
  as expected from crude 2D perspective warps rather than real 3D poses —
  this run exercises the plumbing, not calibration accuracy.)
- Freeze check: `tick()` on a synthetic 4656×3496 frame (this rig's real
  capture resolution) returned in 76 ms — confirms detection no longer runs
  synchronously in the capture path.

### Task 2: `calibrate_live.py` — phase-driven main loop

**Files:** `calibrate_live.py` (full rewrite).

- [x] `open_rgb_cameras()` unchanged — both cameras open once for the whole
  run, no per-phase reopen.
- [x] `review_and_prune()` kept verbatim (unchanged) — it only reads
  `output_root`/`captures_dir` after the fact, agnostic to how they were
  produced.
- [x] `parse_args()`: added `--mono-views` (20), `--stereo-views` (15),
  `--top-up` (5); dropped `--reject-sigma`, `--recalibrate-every`,
  `--min-views-before-rejection` (outlier rejection now lives entirely
  inside the offline scripts, never reimplemented here). `--in`/`--captures`
  is now documented as the *parent* directory — the run writes into its own
  `live_<timestamp>` subfolder inside it.
- [x] `main()`: builds `captures_dir = resolve_path(args.captures) /
  f"live_{run_timestamp}"` and constructs one
  `SequentialLiveCalibrationSession`. Loop grabs only the camera(s) the
  current phase needs (`RGBCamera.grab_pair` for `STEREO`, a single
  `.grab()` for a mono phase — same per-camera `.recover()` fallback,
  generalized from the old pair-only recovery), calls `check_gate()` +
  `poll()` every iteration, gates capture on `view_count() < target()` (the
  single condition that also makes top-up "just work" — raising the target
  makes this true again with no separate pause/resume plumbing), and calls
  `maybe_start_fit()` every iteration.
- [x] GUI/terminal action wiring: `consume_refit_request()`/
  `consume_top_up_request()`/`consume_continue_request()`/
  `consume_discard_request()` (GUI) or `r`/`u`/`c`/`d` keys (headless),
  alongside the existing pause (space) and quit (`q`).
- [x] Loop exits via `session.is_done()` (all 3 phases fit and advanced);
  final message + `review_and_prune()` call only then.

**Verification:**
- `python3 -c "import calibrate_live"` — OK.
- `python3 calibrate_live.py --help` — lists all flags above with correct
  defaults.
- Camera-open, the grab loop, and the live GUI/terminal loop all talk to
  physical hardware and were **not** run end-to-end — this cannot be
  exercised without the RGB cameras attached, per `CLAUDE.md`. Stating this
  explicitly rather than claiming it works.

### Task 3: `live_gui.py` — one-phase-at-a-time layout

**Files:** `live_gui.py` (full rewrite).

- [x] Header shows `Phase N/3: <camera(s)> <intrinsics|stereo pair> —
  X/Y captured`.
- [x] Both preview panels always exist (avoids dynamic grid churn); during a
  mono phase the inactive one shows a "not used this phase" placeholder
  instead of a stale frame.
- [x] Single result card: ready/not-ready, views/RMS(/baseline for stereo),
  coverage-gap line, warnings, coverage thumbnail — replaces the old
  cam1/cam2/stereo 3-card grid, since only one phase is live at a time.
- [x] Toolbar: Refit Now (r), Capture More (u), Continue (c), Discard &
  Restart (d), Pause/Resume (space), Open Captures Folder, Quit (q) — each
  a `request_*()`/`consume_*_request()` pair matching the old
  `request_finish()`/`consume_finish_request()` convention.
- [x] `update(session, preview_a, preview_b, gate)` is the one call the main
  loop makes per iteration; it derives which panels/labels to show from
  `session.phase` and `session.phase_camera_label()` rather than the caller
  branching on phase itself.

**Verification (real Tk, this environment has a display — `DISPLAY=:0`):**
- `python3 -c "import live_gui"` — OK.
- Live smoke test: constructed a real `LiveCaptureGUI`, drove a real
  `SequentialLiveCalibrationSession` through phase 1 with mocked
  detection/subprocess (same technique as Task 1's fixture suite), called
  `update()` at each stage, and asserted: mono-phase placeholder shows on
  the inactive panel and disappears on the active one; capture-count and
  last-save labels update; refit/top-up/continue request-consume pairs each
  fire exactly once; after a real fit landed, the ready/stats labels
  populated; after `advance()` the header read "Phase 2/3" and the
  placeholder swapped to the other panel. All assertions passed
  (`LIVE_GUI_SMOKE_TEST_OK`); window closed cleanly after.

## Verification summary

All logic verified hardware-free per `CLAUDE.md` (no test suite in this
repo): import/syntax checks on all three files, a fixture-based state-
machine suite, one fully-real end-to-end run (synthetic ChArUco renders →
real detection → real `calibrate_cameras.py` subprocess → real
`quality_warnings()`), and a real Tk GUI smoke test (a display was
available in this environment). Everything that requires the physical RGB
cameras — `calibrate_live.py`'s actual camera-open/grab loop and the GUI
driven from live frames — was **not** run and is explicitly unverified on
real hardware, consistent with how `calibrate_live.py`/`live_capture.py`/
`live_gui.py` were originally built (2026-09-15, per memory) and every other
hardware-dependent script in this repo.

## Not done here (see spec's "Out of scope")

Resuming a specific prior `live_<timestamp>` run's in-progress phase state;
a quantitative "% of frame covered" coverage metric (only the existing
scatter/heatmap); any change to fitting math or warning thresholds inside
`opencv_calibrate.py`/`stereo.py`.
