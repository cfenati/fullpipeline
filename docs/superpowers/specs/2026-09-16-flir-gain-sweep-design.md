# FLIR Blackfly gain sweep mode — design

## Problem

`capture_pipeline.py` saves one Blackfly frame per trigger, at whatever gain
`config.yaml`'s `blackfly.gain` specifies. The right gain for this rig hasn't
been picked yet, so a single fixed value can't be validated — we need a way
to compare several gain settings on the same scene without repeatedly
editing config and re-running the pipeline.

The camera's real `Gain` node range (queried live via PySpin against the
attached BFS-U3-32S4M) is **0–47.99 dB**. A 20–40 dB sweep sits comfortably
inside that range on both ends, so it's a reasonable starting sweep — no
value gets silently clamped to a duplicate.

## Scope

Opt-in only. Normal capture sessions (`python capture_pipeline.py`) are
unaffected — they keep saving exactly one Blackfly frame at the configured
gain, alongside RGB/thermal as today. The sweep is a separate mode entered
via a new flag, mirroring how `--smoke-test` and `--sync-test` already work
as alternate top-level modes dispatched in `main()` before `run_pipeline`.

Sweep mode is FLIR-only: it does not open or grab from RGB/thermal. It's a
tuning pass for one camera, not a synchronized multi-camera capture.

## CLI & config

New flag:

```bash
python capture_pipeline.py --flir-gain-sweep
```

Requires `blackfly.enabled: true` in config; errors with a clear message if
Blackfly isn't enabled.

New config block under `blackfly:` in `config.yaml`:

```yaml
blackfly:
  ...
  gain_sweep:
    start: 20
    stop: 40
    step: 5   # dB (Spinnaker Gain node range on this camera: 0-48 dB)
```

Editable without touching code, since the right values aren't settled yet.

`--output` behaves as it does everywhere else in this script (subfolder
under `captures/`, or absolute path). When omitted in sweep mode, it
defaults to `flir_gain_sweep` instead of the pipeline's normal root
`output_dir`, so tuning shots don't land mixed in with real experiment
sessions:

```
captures/flir_gain_sweep/<timestamp>/
```

`--no-preview` (existing flag) also suppresses the live preview window in
sweep mode, for headless runs.

## `BlackflyCamera.set_gain()`

Today, gain is only ever set once, in `open()`. Sweeping requires changing
it mid-session without reopening the camera (reopening is slow and resets
other camera state). Add:

```python
def set_gain(self, gain: float) -> float:
    """Set Gain (dB) on an already-open camera. Disables GainAuto on first
    use. Returns the value actually applied (clamped to the node's min/max)."""
```

- Disables `GainAuto` the first time it's called — a sweep is inherently
  manual-gain, and leaving auto-gain on would fight it.
- Clamps to the node's real `GetMin()`/`GetMax()`, same pattern as the
  existing clamping logic in `open()`.
- Updates `self.gain` so `info()` (used in saved metadata) stays accurate
  for whichever gain was last applied.

Sweep mode always opens the Blackfly with `gain_auto=False`, regardless of
`config.yaml`'s `blackfly.gain_auto` value — that setting is for normal
capture sessions only.

## Sweep loop / UX

New function `flir_gain_sweep_mode()` in `capture_pipeline.py`, parallel in
structure to `smoke_test()` / `sync_test()`:

1. Load config, resolve output dir (default `flir_gain_sweep` as above).
2. Open only the Blackfly camera (`open_blackfly()`, forcing
   `gain_auto=False`).
3. If preview is enabled: a single cv2 window streaming the live Blackfly
   preview (same pattern as `check_exposure_live.py`), so the operator can
   frame the shot before triggering. `TerminalInput` (already in this file)
   handles `s`/`q` keys without needing Enter — works identically whether or
   not the preview window is shown, so `--no-preview` degrades gracefully to
   headless terminal-only control.
4. On `s`:
   - `timestamp = time.strftime(...)`, create `<output_dir>/<timestamp>/`.
   - For each gain value in the configured sweep (`start`, `start+step`, ...,
     up to and including `stop`):
     - `blackfly.set_gain(value)`.
     - Grab and discard one frame (lets exposure settle after the gain
       change before the frame that gets saved).
     - Grab a full-resolution frame (`blackfly.grab(full_resolution=True)`)
       and save it as `flir_gain_<NN>.jpg` (NN = integer dB).
     - If a grab fails at a given gain: print a warning, skip that file, and
       continue the sweep rather than aborting it.
   - Write `metadata.json` in the same folder: timestamp, capture_time_iso,
     requested vs. applied gain per file, `blackfly.info()` snapshot.
5. On `q`: quit and release the camera.

Repeated `s` presses produce additional timestamped sweep folders within the
same run, so the operator can re-trigger for a different scene/distance
without losing earlier sweeps or restarting the script.

## Out of scope

- No change to normal single-shot Blackfly capture in `run_pipeline()`.
- No GUI (Tkinter) integration — sweep mode is terminal + optional cv2
  window only, consistent with other tuning utilities in this repo
  (`check_exposure_live.py`).
- No automatic "best gain" selection/scoring — this is a manual comparison
  tool; picking the gain from the saved images is a human step.
