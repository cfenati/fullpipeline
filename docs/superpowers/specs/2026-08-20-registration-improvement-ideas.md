# Registration improvement ideas (backlog)

**Date:** 2026-08-20
**Status:** Backlog -- not designed, not approved, not scheduled. Not a spec;
recorded here so ideas surfaced while triaging `register_features.py` aren't
lost. Promote an item to a real design doc before implementing it.

## Context

As of this date, `register_features.py` does LightGlue sparse matching +
piecewise-affine warp (mesh from real matches + synthetic border anchors,
degenerate-triangle rejection, feathered blend into a calibrated fallback
plane -- see its own docstring for the current algorithm). On the held-out
ChArUco board set (`captures/cross-validation`, scored by
`check_registration_error.py`), the warp itself is accurate to a median 0.66 px
/ p90 1.32 px / max 3.51 px across 357 corners -- the core geometry is sound.
Remaining hand-registration error is a *coverage* problem (how much of the
frame the match hull reaches, and how well it handles real depth
discontinuities within the hull), not a warp-math bug.

`check_registration_error.py`'s caveat: every held-out corner landed inside
the match hull (0 used the fallback plane) because a flat board fills the
frame densely and evenly -- this metric currently can't discriminate
border-anchor/feathering benefit, since those only matter where the hull
runs out. A more representative held-out target (something with real depth
variation, or deliberately sparse texture near the frame edge) would be
needed to score that.

## Ideas, roughly cheapest/highest-leverage first

1. **Tiled/grid-quota keypoint extraction, status: tried, negative.**
   Implemented as `extract_features_tiled` / `--tiled-keypoints` (opt-in, not
   default). Coverage on a hand capture did increase (35.5% -> 38.5% hull
   coverage on `captures/20260818_152114_103039`), matching the hypothesis
   visually -- but `check_registration_error.py` on the held-out ChArUco set
   caught what eyeballing missed: accuracy regressed (median 0.66 -> 0.77px,
   p90 1.32 -> 2.22px, **max 3.51 -> 14.20px**, some corners lost mesh
   coverage entirely). Splitting a fixed keypoint budget across tiles thins
   density on already-textured regions (the board itself, or a hand's
   knuckles/creases) more than it helps low-texture ones -- net negative on
   a scene that's mostly texture-rich to begin with. Kept as an opt-in flag
   rather than removed, in case a genuinely texture-sparse subject reverses
   the trade-off, but don't default to it without new evidence.

2. **Geometric-consistency pre-filtering.** Reject matches whose local
   displacement is an outlier vs. its neighbors *before* triangulating --
   catches bad matches before they can form a bad triangle, complementing
   the post-hoc degenerate-triangle rejection already in place.

3. **Ensemble/second matcher.** Pool LightGlue+DISK matches with a second
   independent matcher (ORB/SIFT, or a dense matcher like LoFTR) to extend
   coverage into regions DISK undersamples.

4. **RAFT-style dense optical flow (flagged as a specific future option).**
   Two-view flow gives a genuinely dense correspondence field directly -- no
   rectification needed (unlike FoundationStereo, which loses ~65% of the
   frame to this rig's 18 deg toe-in per `register_foundationstereo.py`'s
   docstring), no sparse-keypoints-then-mesh step (unlike LightGlue). Forward-
   backward consistency + the existing triangulation-based depth-range filter
   could reject bad flow. Sits structurally between `register_features.py`
   and `register_foundationstereo.py`; GPU is available on this machine.

5. **Classical dense stereo (SGBM), status: tried, inconclusive-to-negative.**
   `DIYer22/check_dense_stereo.py`'s `TunedSGBM` fixed the disparity search
   window to this rig's actual close-range window (1100-2150 px full-res,
   vs. the library default 2-220 px). Result on `captures/triangulate`: still
   only 8.07% valid depth (vs. 27-32% for `register_pipeline.py`'s own
   ZNCC plane-sweep). `depth_vis.jpg` shows valid pixels concentrated on
   finger-silhouette edges, almost none on the finger's own interior surface
   -- consistent with the script's original docstring conclusion (this rig's
   convergence angle is the limiter, not window tuning) rather than refuting
   it. Not recommended as a further investment absent a new idea for why it
   would behave differently.

6. **Wider physical baseline.** `design/config/rig_baseline_widened.yaml`
   (92mm vs. as-built ~50mm) is already in the repo, untracked. A wider
   baseline increases parallax/depth sensitivity, shrinking how much any
   local region's real depth variation matters relative to baseline -- helps
   every software approach simultaneously (SGBM, piecewise-affine, flow,
   FoundationStereo). Worth checking via `design_rig.py`'s coverage tooling
   before committing to a physical rebuild.

7. **Tighter stereo calibration.** Current fit: stereo RMS ~2.07 px
   (`design/config/rig_as_built.yaml`, auto-updated by `stereo_calibrate.py`).
   Calibration error is a systematic floor under every registration
   approach's accuracy; worth investigating whether more/better-distributed
   checkerboard views would tighten it.

8. **A more discriminating accuracy target.** To actually measure the
   border-anchor/feathering benefit (see caveat above), capture a held-out
   target with real depth variation near the frame edge -- not just another
   flat board capture.

## Explicitly not pursued now

- Finishing a real `register_foundationstereo.py` inference run (dense DL,
  GPU) -- deferred by user choice in favor of iterating on
  `register_features.py` first (2026-08-20). The script is complete and a
  rectified pair was already prepped once via `--prep-only`.
