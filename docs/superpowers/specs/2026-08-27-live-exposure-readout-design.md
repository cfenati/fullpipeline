# Live exposure/vignetting readout — design

## Problem

Color correction was built from a white-board reference, but the reference
capture had brightness/vignetting mismatches between `rgb_cam1` and
`rgb_cam2` that weren't caught until later (see README's note that cam2
collects ~37% less light than cam1). Fixing this at the source means
physically matching the two lenses' aperture and focus rings before
recapturing the white reference — which requires watching live brightness
numbers from both cameras while turning the rings by hand.

`check_color.py` already does deep one-shot analysis (9-region breakdown,
flat-field maps, matplotlib figures) but captures once and exits. There's no
live, continuous view of the cameras for physically adjusting them.

## Scope

A new standalone script, `check_exposure_live.py`, for interactively
adjusting the RGB cameras' physical aperture/focus rings while watching a
live preview and brightness/color readout. Diagnostic tool only — no file
output, no config.yaml changes, no flat-field generation.

## Camera controls

Uses `controls_for(rgb_config, "cam1"/"cam2")` from `config.yaml` completely
unmodified — same call `check_color.py` and `capture_pipeline.py` make. No
control overrides of any kind (auto_exposure stays at whatever config.yaml
says, currently `3` / Aperture Priority; gain stays at its configured value).
The user adjusts the physical rings; camera settings are not touched by this
tool.

## Resolution

Opens both `RGBCamera`s at a reduced resolution instead of config.yaml's full
capture resolution (4656x3496, which caps at ~10fps per the README). Default
1280x960 (matches the sensor's native ~4:3 aspect, well under the
2320x1744/30fps ceiling mentioned in config.yaml's comments) for a responsive
loop. Overridable via `--width`/`--height`/`--fps`.

## Display

Single OpenCV window (`cv2.imshow`), not the Tkinter `CaptureGUI` from
`gui.py` — that class is purpose-built for the capture workflow (save
button, status bar, thermal/Blackfly panels) and doesn't fit a two-camera
diagnostic loop.

- cam1 and cam2 frames shown side-by-side in one window.
- 5 regions sampled per frame: center + 4 corners (top_left, top_right,
  bottom_left, bottom_right) — the classic vignetting-check set. Reuses
  `RegionStats` / `_region_stats` from `check_color.py` for the per-channel
  mean/ratio math rather than reimplementing it. (The full 9-region
  edge+corner breakdown from `check_color.py` is that script's one-shot
  report; this tool favors a smaller, glanceable set since regions are drawn
  as boxes on a live image rather than printed as a table.)
- Each region is drawn as a rectangle on its camera's frame; a small overlay
  panel near each box shows that region's B/G/R, gray, R/G, B/G.
- A summary line at the top of the window shows the cam1-vs-cam2 center-gray
  difference — the brightness-matching number relevant to the ~37% light gap.

## Refresh behavior

- The displayed image updates every grabbed frame (smooth video, so focus
  is visually judgeable while turning the ring).
- The numeric overlay (region stats, summary line) recomputes on a throttle,
  `--interval` seconds (default 0.25s), so digits stay readable instead of
  flickering every frame.
- Grabbing uses `RGBCamera.grab_pair(cam1, cam2)` (existing synchronized-pair
  helper) each loop iteration.

## Loop control / errors

- Runs until `q` is pressed in the OpenCV window or the window is closed
  (`cv2.waitKey`), matching `capture_pipeline.py`'s key convention.
- Cameras release in a `finally` block.
- Camera open failure: clear error message and exit, same style as
  `check_cameras.py` (no traceback).
- A failed `grab_pair` tick is skipped (loop continues) rather than crashing,
  consistent with `RGBCamera`'s existing retry-oriented design.

## CLI surface

```
python check_exposure_live.py [--width 1280] [--height 960] [--fps 30] [--interval 0.25]
```

No file output. No `--session`/`--image` modes (live-only, unlike
`check_color.py`). No config.yaml writes.

## Out of scope / deferred

- No manual exposure lock or auto-detected exposure value — explicitly
  rejected; auto_exposure stays in whatever mode config.yaml sets.
- No headless/plain-terminal fallback mode — GUI is the primary and only
  display mode for this tool.
- No integration with `check_color.py`'s flat-field save path; that remains
  the follow-up step once the rings are physically matched.
