# Touchscreen menu for the Jetson — design

Status: **Approved design, not yet implemented** (2026-09-25).

## Problem

Every stage of the pipeline is a separate argparse CLI that has to be typed
into a terminal. The rig will be operated on the Jetson's touchscreen by a
non-technical operator: no keyboard, no shell, and paths like
`--session captures/<timestamp>` can't be typed.

Two existing tool families also can't be driven by touch at all:

- `measure_points.py` / `measure_wound_depth.py` (via `run_interactive`) are
  OpenCV HighGUI windows. Points are placed with the mouse, but zoom is the
  wheel, pan is right/middle-drag, and every other action (undo, reset, lock
  rim, finish) is a keyboard key. A touchscreen delivers only left-click.
- `gui.py` / `live_gui.py` are Tkinter but sized for a desktop (8 buttons in
  one row in `live_gui.py`, keyboard hints in the labels).

## Scope

A home-screen launcher plus a `--touch` mode for the tools it launches. Every
script stays independently runnable from the CLI with unchanged behaviour when
`--touch` is not passed (one exception: the capture-folder collision fix
below).

Chosen approach: **subprocess launcher**. The menu spawns each existing script
as a child process; it never imports camera SDKs, a crashing stage cannot take
the menu down, and only one stage runs at a time (so two stages can never
fight over `/dev/video*`). Rejected: merging all stages into one Tk process
(each of `capture_pipeline.py`, `calibrate_live.py` and the measure tools owns
its own main loop; merging is invasive and would break the CLI entry points),
and porting the measure tools to Tkinter (~1400 lines of validated,
cv2-native geometry/UI code for a cosmetic gain).

## Files

| File | Role |
|---|---|
| `menu.py` (new) | Tkinter screens only: home, session picker, name keyboard, running, error, result. Entry point; `--install-shortcut`. |
| `menu_logic.py` (new) | Pure logic, no display needed: session discovery, command building, name sanitizing, failure-text mapping, newest-report lookup. |
| `touch_controls.py` (new) | Touch translation layer for the cv2 measure tools, plus the shared confirm/debounce helpers and `apply_touch_style()` for the Tk screens. |
| `gui.py`, `live_gui.py` | `--touch` styling / layout. |
| `measure_points.py`, `measure_wound_depth.py` | `--touch` flag; three hooks in `run_interactive`. |
| `capture_pipeline.py`, `calibrate_live.py` | `--touch` flag passed through to their GUI; capture folder collision fix. |
| `tests/` (new) | stdlib `unittest` for the pure logic. |
| `.gitignore` | add `logs/`. |
| `README.md` | short "Touchscreen menu" section (README is the canonical workflow doc). |

## Home screen

Six large tiles in a 2×3 grid plus a large Quit button. The window is
maximized, not fullscreen/kiosk: it is launched from a desktop icon, the
desktop stays reachable, and the Nautilus window from "Output folder" needs to
appear normally. It adapts to the screen size the Jetson reports
(`winfo_screenwidth/height`) and is designed for at least 1280×720.

| Tile | Command run (all with cwd = project root, `sys.executable`) |
|---|---|
| Take images | `capture_pipeline.py --touch [--output <name>]` |
| Calibrate cameras | `calibrate_live.py --touch` |
| Measure wound depth | `measure_wound_depth.py --touch --session <dir> --window W H` |
| Measure length | `measure_points.py --touch --session <dir> --window W H` |
| Register images | `register_features.py --session <dir>` |
| Output folder | `xdg-open captures/` (same as the existing `gui.py` button) |

`register_features.py` opens no window (no `imshow`/Tk in the file), so it has
no `--touch` flag; it only gets the Running screen.

## Running a stage

- The child's stdout+stderr go to `logs/menu/<timestamp>_<stage>.log`
  (git-ignored).
- The menu stays on a **"Running <stage>…" screen** with elapsed time and a big
  **Stop** button (kills the child, returns home). It does *not* hide itself:
  if the child takes seconds to open cameras, the operator would otherwise see
  an empty desktop with no way back. The child's own window opens on top.
- Exit 0 → home screen; for the three analysis tiles → **Result screen**.
- Non-zero exit → plain-language message plus "Show details" (last ~20 log
  lines) and "Retry". A small table maps known failure text to friendly
  wording (e.g. missing device → "Cameras not found — check USB and power").

## Session picker (depth, length, register)

- Big **"Latest capture"** button for the common case: the session with the
  newest directory mtime across all non-hidden folders (mtime, not name,
  because named and unnamed captures don't share a naming scheme).
- Below it a folder list; inside a folder, sessions newest first with
  thumbnails.
- A session is a directory containing `rgb_cam1.jpg` and `rgb_cam2.jpg`, at
  most two levels under `captures/` (`captures/<ts>/` from an unnamed capture,
  `captures/<name>/<ts>/` from a named one).
- Hidden from the picker: the folders named by
  `geometric_calibration.intrinsics_captures`, `.stereo_captures` and
  `.cross_validation_captures` (`captures/stereo` alone holds 385 sessions).
  Read from existing config, no new key.

## Naming a capture

On-screen keyboard: a–z, 0–9, `_`, `-`, backspace, clear; input is sanitized
to `[A-Za-z0-9_-]`. Existing folder names (`wound_a`, …) appear as one-tap
chips so typing is rare. "Skip" omits `--output`, leaving the current
timestamp-only behaviour.

## Result screen

The measure tools print their numbers to stdout, which the menu does not show,
so the operator would never see a wound depth. After a successful stage the
menu shows the newest report (`report.txt`, or `report_features.txt` from
registration) and the image next to it (`annotated.jpg` from wound depth,
`measured.jpg` from measure_points, `preview_features.jpg` from registration) written
since the stage started, searched recursively under `registration.output_dir`
from `config.yaml` (default `registration/results/`; the default of all three
analysis tools). This screen
is an addition beyond the four things originally asked for and can be
dropped without affecting anything else.

## Touch layer for the measure tools (`touch_controls.py`)

A **translation layer, not a rewrite**: finger gestures become the mouse/key
events `run_interactive` already understands, so the geometry code is
untouched. `run_interactive` gets three small hooks, inactive without
`--touch`: (1) wrap the mouse callback; (2) draw the button bar and crosshair
onto the frame; (3) read queued button presses as key codes into the existing
`handle_interactive_key` (a pure function of a key code). `run_interactive` is
shared with `check_depth_accuracy.py`, which must keep working unchanged.

| Finger | Effect |
|---|---|
| Drag on an image | Pans that panel (calls the panel's own `pan`). |
| Tap on an image (< ~12 px movement) | Tentative crosshair + loupe at that spot. Nothing recorded. |
| ◀ ▶ ▲ ▼ | Nudge the crosshair one screen pixel. |
| ✔ Place | Synthetic left-click at the crosshair: existing blob-snap, epipolar snap and recording run as today. |
| ✖ Cancel | Clear the crosshair. |

Tap-then-confirm rather than tap-places because a finger covers the feature
and there is no hover.

Button bar (~120 px — two rows of 60 px — below the status strip; its height is subtracted from
`--window` so the canvas stays 1:1 with screen pixels — the tool's own comment
warns that scaling makes clicks land off target):

| Button | Sends | Notes |
|---|---|---|
| Undo | `u` | |
| Reset | `r` | second tap to confirm |
| Zoom + / − | the panel's own `zoom_at`, 1.25× / 0.8× | acts on the last-touched panel |
| Fit | `0` | |
| Lock rim | `n` | wound-depth mode only |
| Finish | `q` | second tap to confirm |

"Second tap to confirm" = the button relabels to "Tap again to confirm" for
3 s; otherwise one stray touch ends the session or wipes every point.

Status strip: in touch mode the keyboard hints ("wheel = zoom…") are replaced.
`measure_wound_depth.py` passes the existing `on_status` hook (already used by
`check_depth_accuracy.py`) in touch mode to show "Rim: 4 of 5 needed",
"Rim locked", and "Point 2: depth 3.4 mm (live estimate)", because it
currently reports these only via `print` and a too-early Lock rim tap would
look dead.

## Touch mode for capture and live calibration

- `apply_touch_style()` in `touch_controls.py` (so `menu.py` need not import
  `gui.py` and its camera imports): larger fonts and button padding, window
  maximized. `gui.py` and `live_gui.py` import it from there.
- **Capture** (`capture_pipeline.py --touch`): one very large **Take photo**
  button plus Done. "Open Output Folder" hidden (the menu has that tile).
  S/Q shortcuts still work.
- **Live calibration** (`calibrate_live.py --touch`): the 8 toolbar buttons
  wrap into two rows, "(r)/(u)…" hints dropped, "Open Captures Folder"
  hidden. **Discard Worst** and **Discard & Restart** use the same
  second-tap confirmation.

## Capture-folder collision fix (approved)

`save_capture` uses `mkdir(exist_ok=True)` (`capture_pipeline.py:278`) with a
timestamp accurate only to the second (line 674). Two captures within one
second write into the same folder, and the second overwrites the first's
files; the on-screen counter still increments. On a touchscreen a
double-tap makes this likely.

1. **(Approved.)** When the timestamped folder already exists, add a `_2`,
   `_3`, … suffix — the scheme `calibrate_live.py` already uses
   (`LiveCalibrationSession._new_session_dir`, `calibration/live_capture.py:392`).
   This changes CLI behaviour too; it goes in its own commit.
2. In touch mode only, ignore Take-photo taps for ~1 s after the previous save
   *finishes* (not after the tap: `save_capture` blocks the Tk loop for
   seconds, so a queued second tap is only delivered once the save ends).
   (Presented alongside fix 1; not separately confirmed — see review note.)

Both derived from reading the code; not run on hardware.

## Install

`python menu.py --install-shortcut` writes `~/Desktop/FullPipeline.desktop`
with absolute paths to the interpreter, `menu.py` and the project root. Some
GNOME versions need "Allow Launching" clicked once on first use.

## Verification

- `unittest` for the pure logic: session discovery (temp dirs), command
  building, name sanitizing, failure-text mapping, newest-report lookup,
  gesture classifier (tap vs drag), button hit-testing, crosshair
  nudge/confirm state, confirm-timeout state.
- Import/syntax check of every touched file; Tk screenshots under Xvfb if it
  is available on the dev machine.
- **Not exercised without the hardware:** any camera code path, and real
  touchscreen input. This will be stated in the final summary rather than
  claimed as working.

## Assumptions to check on the Jetson

- The touchscreen delivers press, move and release to an OpenCV window as
  emulated mouse events (depends on the OpenCV GUI backend).
- A child window opens on top of the maximized menu under the Jetson's window
  manager.
- Screen resolution is at least 1280×720 (the layout adapts, but this is the
  design floor).
- `.desktop` launching works after the one-time "Allow Launching".

## Out of scope

Camera health-check tile; an in-app results browser beyond the Result screen;
kiosk autostart on boot; the manual multi-step calibration chain
(`calibrate_cameras` → `prune_calibration` → `stereo_calibrate`); the accuracy
checks; making `register_features.py` report progress.
