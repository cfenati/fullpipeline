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

2. **Geometric-consistency pre-filtering, status: tried, neutral.**
   Implemented as `geometric_consistency_mask` / `--no-geometric-consistency-filter`
   (default: on). Rejects a match whose displacement (pts_b - pts_a) deviates
   from its 8 nearest neighbors' median by more than 15px, before
   triangulation -- catches a bad match before it can form a bad triangle,
   complementing degenerate-triangle rejection (which only catches one after
   the fact). Measured effect: only 7/2782 matches rejected on a hand
   capture (`captures/20260818_152114_103039`), and on the held-out ChArUco
   set `check_registration_error.py` reports byte-identical numbers with the
   filter on or off (357 corners, median 0.66px, max 3.51px either way) --
   this rig's LightGlue matches are apparently already too clean on both
   available test scenes for this filter to have much to catch. Kept
   default-on (sound rationale, no measured downside, negligible cost), but
   it hasn't been proven to help either -- would need a messier/higher-
   outlier-rate test scene to actually discriminate its value.

3. **Ensemble/second matcher (LoFTR), status: tried, negative alone.**
   Implemented as `load_loftr_model`/`match_loftr` + `--matcher disk|loftr|raft`
   (space-separated to pool multiple, default: `disk`). A cheap probe first
   (not the full pipeline) confirmed LoFTR
   (kornia, outdoor-pretrained) genuinely covers new ground: 2.8x the raw
   matches of DISK+LightGlue, ~84% of the frame's convex hull vs. ~45%, and
   100% of matches triangulated to a physically plausible depth on a hand
   capture -- strong enough evidence to justify building the full
   integration, unlike the weaker case for tiled keypoints.

   But `check_registration_error.py` tells a different story: median point
   error 0.66px (disk) vs. **2.97px (loftr)** vs. 1.03px (both pooled), p90
   1.32 vs. 52.66 vs. 3.58px, max 3.51 vs. 104.78 vs. 104.73px. LoFTR's
   coverage is real but its sub-pixel localization is far worse than
   DISK+LightGlue's -- plausible in hindsight, since the outdoor-pretrained
   checkpoint is a coarse-to-fine dense matcher tuned for broad scene
   correspondence under wide baseline/viewpoint change, not precise
   checkerboard-corner-grade localization. Pooling ("both") only partially
   recovers disk's precision and its max error stays almost as bad as
   loftr-alone, meaning even one imprecise pooled vertex can still corrupt a
   nearby mesh triangle. Third confirmed case this session (after tiled
   keypoints) where visually-plausible extra coverage did not survive
   contact with ground truth -- **don't trust coverage % as a proxy for
   accuracy; always check `check_registration_error.py`.**

   Kept available (`--matcher loftr`/`both`) rather than removed, in case a
   future scene has so little DISK-matchable texture that LoFTR's worse
   precision is still better than no coverage at all -- not because either
   option is currently recommended. Untried refinement if this is revisited:
   the "indoor" LoFTR checkpoint instead of "outdoor" (close-range hand/
   object shots may be a better match for that pretraining domain), or using
   LoFTR only to seed matches in regions with no nearby DISK match instead of
   naively pooling everywhere.

4. **RAFT-style dense optical flow, status: tried, positive when pooled with
   disk.** Implemented as `load_raft_model`/`match_raft` + `--matcher raft`
   (torchvision RAFT-large, no new heavy dependency beyond `torchvision`
   itself). A probe first caught its own bug (naive preprocessing skipped
   torchvision's expected uint8->float normalization, producing near-random
   flow) -- worth remembering: verify a probe's *inputs* look sane before
   trusting a discouraging result. Once fixed, forward-backward consistency
   error on this rig turned out cleanly bimodal, not gradual: correctly-
   tracked points round-trip within ~0.2-0.9px, lost-tracking points (the
   aperture problem, on repetitive/low-texture regions) jump to 20-200+px
   with no gray zone -- so a single threshold (1.5px) separates them well.

   `raft` alone: median 0.62px (competitive with disk's 0.66px) but p90
   10.46px / max 23.47px -- a handful of sessions get fooled into a
   periodic-but-self-consistent wrong lock, plausibly specific to the
   ChArUco board's repetitive pattern (a textbook adversarial case for
   forward-backward consistency checks) rather than a hand's aperiodic
   texture. **`disk raft` pooled is the one combination measured better
   than disk alone on every aggregate this session**: median 0.56px, p90
   1.18px (both improved over disk alone), max 6.76px (worse than disk's
   3.51px but far better than raft alone, and every session's median lands
   under 1px -- disk's precision specifically rescues the sessions raft's
   periodic-lock hurt). Verified on the hand capture too (not just ChArUco):
   62.8% hull coverage, 16.2s runtime, visually clean.

   Not made default: real added cost (a second model, ~8s extra per image
   pair on CPU, a new `torchvision` dependency -- added to requirements.txt
   regardless) to improve an already-sub-pixel metric further, and only
   validated by the point-accuracy numbers on ChArUco corners, not yet
   confirmed as a win on the actual hand-capture use case by any measure
   beyond visual inspection. Worth deliberately turning on
   (`--matcher disk raft`) if squeezing out that last fraction of a pixel
   matters, or revisiting as the default if it holds up on more scenes.

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

9. **Depth-aware triangle rejection, status: implemented and threshold-tuned
   2026-08-26 on one real capture, not yet cross-checked on others.**
   Implemented as `depth_discontinuous_triangle_mask` /
   `--no-reject-depth-discontinuous-triangles` (default: on, threshold
   `--max-triangle-depth-range`, default 3mm). Motivated by the user
   observing non-straight edges in registered output on scenes with large
   height/depth variation: the existing `degenerate_triangle_mask` only looks
   at the *projected* 2-D triangle in camera-A pixel space (area,
   slenderness), so a triangle can be an ordinary size/shape there while
   still bridging a real depth discontinuity, if matches land densely on
   both sides of it -- linear interpolation across that Z jump is what bends
   a physically straight edge. This new check reuses the Z already
   triangulated per-match (no extra compute) and rejects a triangle directly
   when its 3 vertices span more than the threshold in real depth,
   independent of the shape check.

   Threshold was NOT left at the initial guess: a first default of 20mm
   (20% of the config `depth_range` window) turned out to be a no-op on
   `captures/20260818_152114_103039` (real hand capture) -- that scene's
   whole triangulated Z only spans ~30mm end to end, so no triangle ever
   reached 20mm (0/5535 rejected). A sweep on that capture, checked visually
   via `preview_features.jpg` at each point: 10mm->4 rejected, 5mm->43,
   3mm->85 (new gaps land at finger-valley creases -- real discontinuities
   -- with no visible damage to smooth finger surface), 2mm->258 (starts
   speckling flat non-edge finger surface -- triangulation noise, not a real
   discontinuity), 1mm->1588 (28.7% of all triangles, visibly shreds the
   surface). Shipped default: 3mm, the last clean point before the knee.

   Caveat: only validated on that one capture, and not run through
   `check_registration_error.py`'s ChArUco set -- that board is flat, so it
   can never exercise this check at all (same blind spot already on record
   for border-anchor/feathering in the Context section above). Revisit the
   3mm default if a differently-shaped scene (larger real depth range, or a
   much flatter one) suggests a different number.

10. **Idea, not implemented, deferred by user 2026-08-26: edge maps as a
    keypoint-placement signal, not a correspondence source.** Distinct from
    the LoFTR-seeding idea in #3 -- rather than adding another matcher's
    correspondences near edges, use classical edge detection (e.g. Canny) on
    camera A as a *prior* that biases where DISK (or a SuperPoint-style
    detector) places/keeps keypoints, so the mesh gets denser vertices
    flanking a depth discontinuity without trusting the edge pixels
    themselves as matches (an edge pixel's appearance differs between the
    two cameras' viewpoints, so directly matching a synthetic edge point
    would be unreliable -- the edge map would only *guide sampling*, the
    correspondence would still come from real DISK/LightGlue matches just
    outside it). Complements idea #9: denser vertices near the discontinuity
    shrinks how far any surviving triangle has to bridge across it, on top
    of rejecting the ones that still do.

## Explicitly not pursued now

- Finishing a real `register_foundationstereo.py` inference run (dense DL,
  GPU) -- deferred by user choice in favor of iterating on
  `register_features.py` first (2026-08-20). The script is complete and a
  rectified pair was already prepped once via `--prep-only`.
