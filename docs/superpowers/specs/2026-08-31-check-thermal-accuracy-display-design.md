# Design: check_thermal_accuracy.py — clearer accuracy/drift display

Date: 2026-08-31
Status: **Implemented** (verified against real session data below; not yet
exercised through a live camera run, which is unaffected by this change)

## Purpose

`check_thermal_accuracy.py` logs the Optris thermal camera's ROI-mean
temperature over time and lets an operator record handheld-gun checkpoints
(`r` key) to compare against. It already writes `log.csv`, `checkpoints.csv`,
`summary.txt`, and `plot.png` per session. The first real session
(`thermal_reports/20260828_160834/`, 42 checkpoints over ~90 minutes) showed
both outputs are hard to read:

- `plot.png` stamps all 42 offset values as text directly on a noisy line —
  the labels overlap into an unreadable block, and the plot shows the
  camera's own trace with dots at the *camera's* value, never the gun's
  actual reading, so there's no visual way to see the two instruments
  tracking (or not tracking) each other.
- `summary.txt`'s only aggregate figure is a "drift" computed from the
  *first and last* checkpoint only: `+0.07C over 5372.4s` → `+0.05 C/hour`.
  Computed against this session's actual scatter (offset std dev 0.21°C
  across all 42 points), a two-point difference this size is well within
  noise — the summary reports it as if it were a measured trend.
- Neither output states whether the session is within the camera's own
  accuracy spec, so "is this OK?" has no answer without pulling the CSV into
  a notebook by hand (done once, manually, to write this spec — see
  "Findings" below).

## Findings from the existing session

Computed directly from `thermal_reports/20260828_160834/checkpoints.csv`
(n=42): mean offset (camera − gun) **+0.02°C**, std dev **0.21°C**, range
**−0.34°C to +0.46°C**. Gun readings cluster 23.0–23.3°C (0.1°C device
resolution).

Optris datasheet spec (provided by the user, not previously in this repo):
accuracy **±2°C or ±2% of reading**, whichever is larger (2% of 23°C ≈
0.46°C, so the binding tolerance here is ±2°C); NETD (sensitivity) **50 mK =
0.05°C**; warm-up **10 minutes**.

Conclusions this design's outputs need to surface automatically:
- **No systematic shift** — mean offset 0.02°C is indistinguishable from
  zero given the noise.
- **Comfortably within spec** — worst single checkpoint (0.46°C) is far
  inside the ±2°C tolerance.
- **No real drift** — the naive first/last delta (+0.07°C) is smaller than
  the point-to-point noise (0.21°C std), so it should not be reported as a
  trend without a proper fit across all points.

Not addressed by this design (see "Out of scope"): `config.yaml`'s
`thermal.warmup_seconds: 1` undershoots the datasheet's 10-minute figure;
this session's 0.21°C scatter vs. the 0.05°C NETD spec is a different
physical quantity (ROI-mean/gun repeatability vs. single-pixel sensitivity)
and isn't compared in this design.

## Design

### `config.yaml`

Add the datasheet spec next to the existing `thermal:` block, following this
repo's convention of keeping hardware/physics constants in config with an
explanatory comment (per `CLAUDE.md`), not hardcoded in the script:

```yaml
thermal:
  config_xml: OptrixThermalCamera/config/generic.xml
  warmup_seconds: 1
  accuracy_abs_c: 2.0   # Optris datasheet: +/-2C or +/-2% of reading, whichever is larger
  accuracy_pct: 2.0
```

Per-reading tolerance: `max(accuracy_abs_c, gun_c * accuracy_pct / 100)`.
`load_thermal_config()` already returns this whole dict; `run()` passes
`accuracy_abs_c`/`accuracy_pct` through to `write_summary`/`write_plot`
alongside the existing arguments (no new CLI flags — this is a hardware
spec, not a per-run choice).

### Stats computation

A new pure function, `compute_offset_stats(checkpoints, accuracy_abs_c,
accuracy_pct) -> dict`, used by both `build_summary_text` and `write_plot`
so the two outputs can never disagree:

- `n`, `mean_offset_c`, `std_offset_c` (population std, matching this file's
  existing style), `min_offset_c`, `max_offset_c`, `max_abs_offset_c`
- per-checkpoint `tolerance_c` (from that checkpoint's own `gun_c`) and
  `within_spec: bool`; overall `all_within_spec = all(...)`
- drift: `numpy.polyfit(elapsed_s_array, offset_mean_c_array, 1)` across
  *all* checkpoints → `slope_c_per_hour`; `predicted_drift_c =
  slope_c_per_hour * session_hours`; `drift_within_noise =
  abs(predicted_drift_c) < std_offset_c`
- Only defined when `len(checkpoints) >= 2` (mirrors the existing
  single-checkpoint special case in `build_summary_text`, which stays as-is:
  "accuracy assessed, drift not assessable").

### `summary.txt`

`build_summary_text` restructured to lead with the aggregate verdict, then
keep the full per-checkpoint list (added previously per commit
`042760f`) as the appendix:

```
Thermal accuracy check -- 42 checkpoint(s) over 89m 32s

Spec (Optris datasheet): +/-2.00C or +/-2.0% of reading

Offset (camera - gun):
  mean   +0.02 C
  std     0.21 C
  range  -0.34 C .. +0.46 C
  -> PASS: all 42 checkpoint(s) within spec (worst case 0.46C, tolerance up to 2.00C)

Drift (linear fit across all checkpoints):
  slope  +0.05 C/hour
  -> not distinguishable from noise (predicted drift over session: 0.07C, smaller than offset std 0.21C)

All checkpoints:
  t=   24.3s  camera=22.86C  gun=23.10C  offset=-0.24C
  ...
```

If `all_within_spec` is `False`, the verdict line reads `FAIL` and names the
worst offending checkpoint (timestamp + offset), and that checkpoint's line
in the appendix list gets a trailing `<-- FAIL` marker. If the drift fit
*isn't* within noise, the verdict reads "possible real trend -- treat
cautiously, only one session recorded" instead of "not distinguishable from
noise".

### `plot.png`

Two stacked panels sharing the x-axis (`plt.subplots(2, 1, sharex=True)`):

- **Top (temperature)**: camera ROI-mean line as today, plus gun checkpoints
  plotted at their **actual `gun_c` value** (not `camera_mean_c`) as a
  separate marker series — so the two instruments' traces are both visible
  and a real divergence would be visually obvious. No per-point text labels.
  Y-limits computed from a percentile-based range (e.g. 1st–99th percentile
  of the plotted line, padded) rather than raw min/max, so a single
  transient frame (this session's first log row is a 21.97°C cold-start
  outlier before the sensor settles, per `log.csv`) doesn't stretch the
  whole axis.
- **Bottom (offset)**: `camera_mean_c - gun_c` scatter per checkpoint vs.
  elapsed time, a dashed zero-reference line, a solid mean-offset line with
  a shaded ±1σ band, and a text box (upper corner) giving `n`, mean±std, and
  the PASS/FAIL verdict. Spec tolerance lines are not drawn on this panel —
  at ±2°C they'd sit far outside the ~±0.5°C data range and compress the
  interesting variation to a flat line; the pass/fail verdict is stated in
  the text box instead.

## Interfaces

Functions touched in `check_thermal_accuracy.py`: `load_thermal_config`
(returns the two new keys, no signature change), `build_summary_text`,
`write_summary`, `write_plot` (each gains `accuracy_abs_c: float,
accuracy_pct: float` parameters), `run` (reads the two config values once
and threads them through). New function: `compute_offset_stats`. No changes
to `LOG_CSV_FIELDS`, `CHECKPOINT_CSV_FIELDS`, the capture loop, ROI
selection, or CLI flags — `log.csv`/`checkpoints.csv` schemas are unchanged.

## Verified

No hardware needed — `compute_offset_stats`, `build_summary_text`, and
`write_plot` are pure functions over already-recorded data. Verified by
loading the real captured session (`thermal_reports/20260828_160834/`,
42 checkpoints, genuinely noisy data — not synthetic) via `csv.DictReader`
and calling the new functions directly:

- Regenerated `summary.txt` matches the hand-computed "Findings" numbers
  exactly: mean +0.02C, std 0.21C, range −0.34C..+0.46C, all 42 checkpoints
  PASS against the ±2.00C spec.
- The linear-fit drift came out **−0.02 C/hour** (predicted drift over the
  session: −0.03C, within the 0.21C noise floor) — opposite in *sign* from
  the old first/last two-point estimate (+0.05 C/hour). Both are noise, but
  the sign flip is direct evidence the old metric wasn't measuring a real
  trend, confirming the reason for replacing it.
- `plot.png` regenerated as a two-panel figure: the offset panel visually
  reads as scatter around zero with the PASS stats box, in contrast to the
  original single-panel plot's 42 overlapping text labels.
- Edge cases exercised directly: zero checkpoints (unchanged early-return
  message) and a single checkpoint (stats/plot render with std=0.00C, no
  drift section, no crash from the n<2 guard on the regression fields).

## Out of scope

`config.yaml`'s `warmup_seconds: 1` vs. the datasheet's 10-minute warm-up
(flagged during design, left for a separate change); comparing offset noise
against the NETD (0.05°C) sensitivity spec (different physical quantity,
would need its own design pass); changing the live capture loop, ROI
selection, or the live overlay; `offset_max_c`/`camera_max_c` handling in
the plot (only `offset_mean_c` is analyzed/plotted; the `*_max_c` CSV fields
are unchanged and unused by this design); CSV schemas; a `--tolerance-c`/
`--tolerance-pct` CLI override (spec lives only in `config.yaml`, per user
choice).
