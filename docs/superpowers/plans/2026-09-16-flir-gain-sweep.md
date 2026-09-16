# FLIR Gain Sweep Capture Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--flir-gain-sweep` mode to `capture_pipeline.py` that saves a folder of Blackfly frames across a configurable dB gain range (default 20–40, step 5) instead of one frame at a fixed gain, so the right gain can be picked visually before it's locked into `config.yaml`.

**Architecture:** A new `BlackflyCamera.set_gain()` method allows changing gain on an already-open camera. A new `capture_gain_sweep()` function in `capture_pipeline.py` drives a per-trigger sweep (set gain → discard-settle → full-res grab → save) and writes a `metadata.json` alongside the images. A new `flir_gain_sweep_mode()` function wires this into a small interactive loop (live cv2 preview + terminal-key fallback), dispatched from `main()` exactly like the existing `--smoke-test`/`--sync-test` alternate modes. Normal capture (`run_pipeline()`) is untouched.

**Tech Stack:** Python 3.9, PySpin (FLIR Spinnaker SDK), OpenCV (`cv2`), PyYAML.

## Global Constraints

- Python 3.9, `from __future__ import annotations` + type hints on all new function signatures (existing repo convention).
- No new imports needed in either file touched — `cv2`, `sys`, `time`, `json`, `datetime`, `Path`, `Optional` are already imported in `capture_pipeline.py`; `cv2`/`np`/`PySpin` already imported in `cameras/blackfly_camera.py`.
- No test suite/linter exists in this repo (per `CLAUDE.md`). Verification below uses standalone scripts run with `python3`, not pytest. Camera-touching scripts require the Blackfly attached — this session confirmed a Blackfly S BFS-U3-32S4M is attached via USB (`lsusb`) and its `Gain` node range is **0.0–47.99 dB** (queried live via PySpin). Every hardware-dependent verification step below should actually be run against it; if the camera is not attached when a task is executed, say so explicitly instead of claiming the step passed.
- `config.yaml` comments are load-bearing documentation (repo convention) — the new `gain_sweep` block must keep an explanatory comment, not just bare keys.
- `captures/` (and thus `captures/flir_gain_sweep/`) is git-ignored — verification output there or under the scratchpad directory must never be `git add`-ed.
- Config keys added: `blackfly.gain_sweep.{start,stop,step}` (dB, inclusive range, default `20`/`40`/`5`).

---

### Task 1: `BlackflyCamera.set_gain()`

**Files:**
- Modify: `cameras/blackfly_camera.py:248-250` (insert a new method between `grab()`, which ends at line 248, and `release()`, which starts at line 250)
- Test: ad hoc script at `<scratchpad>/verify_set_gain.py` (not committed — this repo has no test suite; see Global Constraints)

**Interfaces:**
- Consumes: nothing new — uses `self._cam`, `self.is_open()`, `_require_pyspin()`, all already defined in this class.
- Produces: `BlackflyCamera.set_gain(gain: float) -> float`, used by Task 4's `capture_gain_sweep()`.

- [ ] **Step 1: Write the verification script**

Create `<scratchpad>/verify_set_gain.py` (use your actual scratchpad directory path):

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/cfenati/projects/MasterThesis/FullPipeline")
sys.path.insert(0, str(PROJECT_ROOT))

from cameras.blackfly_camera import BlackflyCamera

blackfly = BlackflyCamera(gain_auto=True, gain=None)
blackfly.open()
try:
    assert blackfly.gain_auto is True, "expected gain_auto True before set_gain"

    applied = blackfly.set_gain(25.0)
    print("applied for 25.0:", applied)
    assert 24.9 <= applied <= 25.1, applied
    assert blackfly.gain_auto is False, "set_gain must disable gain_auto"
    assert blackfly.info()["gain"] == applied

    clamped_high = blackfly.set_gain(1000.0)
    print("applied for 1000.0 (out of range):", clamped_high)
    assert 40.0 < clamped_high <= 48.0, clamped_high

    clamped_low = blackfly.set_gain(-10.0)
    print("applied for -10.0 (out of range):", clamped_low)
    assert clamped_low == 0.0, clamped_low

    print("PASS")
finally:
    blackfly.release()
```

- [ ] **Step 2: Run it to confirm it fails (method doesn't exist yet)**

Run: `python3 <scratchpad>/verify_set_gain.py`
Expected: `AttributeError: 'BlackflyCamera' object has no attribute 'set_gain'`

- [ ] **Step 3: Implement `set_gain()`**

Insert into `cameras/blackfly_camera.py` between the end of `grab()` (line 248) and `def release(self)` (line 250):

```python
    def set_gain(self, gain: float) -> float:
        """Set Gain (dB) on an already-open camera, disabling GainAuto first.
        Returns the value actually applied, clamped to the node's min/max."""
        if not self.is_open():
            raise RuntimeError(f"{self.name} is not open")

        pyspin = _require_pyspin()

        try:
            if self._cam.GainAuto.GetAccessMode() == pyspin.RW:
                self._cam.GainAuto.SetValue(pyspin.GainAuto_Off)
        except Exception:
            pass

        nodemap = self._cam.GetNodeMap()
        gain_node = pyspin.CFloatPtr(nodemap.GetNode("Gain"))
        if not (pyspin.IsAvailable(gain_node) and pyspin.IsWritable(gain_node)):
            raise RuntimeError(f"{self.name}: Gain node is not writable")

        clamped = max(gain_node.GetMin(), min(float(gain), gain_node.GetMax()))
        gain_node.SetValue(clamped)
        self.gain_auto = False
        self.gain = clamped
        return clamped
```

- [ ] **Step 4: Run the verification script again**

Run: `python3 <scratchpad>/verify_set_gain.py`
Expected: prints the three applied values (≈25.0, ≈47.99, 0.0) then `PASS`. This exercises the real attached camera — if `BlackflyCamera().open()` raises because no Blackfly is attached, note that explicitly rather than claiming this step passed.

- [ ] **Step 5: Commit**

```bash
git add cameras/blackfly_camera.py
git commit -m "Add BlackflyCamera.set_gain() for changing gain without reopening"
```

---

### Task 2: `config.yaml` — `blackfly.gain_sweep` block

**Files:**
- Modify: `config.yaml:127-128` (end of the `blackfly:` block)

**Interfaces:**
- Consumes: nothing.
- Produces: `config["blackfly"]["gain_sweep"]` dict with keys `start`, `stop`, `step` (all numbers), consumed by Task 3's `gain_sweep_values()`.

- [ ] **Step 1: Edit the config**

In `config.yaml`, the `blackfly:` block currently ends with:

```yaml
  gain_auto: true
  gain: null
```

Change it to:

```yaml
  gain_auto: true
  gain: null
  # Used only by `capture_pipeline.py --flir-gain-sweep`, which always runs
  # in manual gain regardless of gain_auto above. Range is dB (Spinnaker's
  # Gain node); on this rig's BFS-U3-32S4M it spans 0-48 dB, so 20-40 leaves
  # headroom on both ends while the right operating gain is still unknown.
  gain_sweep:
    start: 20
    stop: 40
    step: 5
```

- [ ] **Step 2: Verify it parses correctly**

Run:

```bash
python3 -c "
import yaml
config = yaml.safe_load(open('config.yaml'))
print(config['blackfly']['gain_sweep'])
"
```

Expected: `{'start': 20, 'stop': 40, 'step': 5}`

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "Add blackfly.gain_sweep config block for --flir-gain-sweep"
```

---

### Task 3: `gain_sweep_values()` helper + `open_blackfly(force_manual_gain=...)`

**Files:**
- Modify: `capture_pipeline.py:165-183` (`open_blackfly`) — add a `force_manual_gain` parameter
- Modify: `capture_pipeline.py:184` (insert new function `gain_sweep_values` right after `open_blackfly`, before `def get_stability_config`)
- Test: ad hoc script at `<scratchpad>/verify_gain_sweep_helpers.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `gain_sweep_values(gain_sweep_config: dict) -> list[float]`, used by Task 4 and Task 5.
  - `open_blackfly(config: dict, force_manual_gain: bool = False) -> Optional[BlackflyCamera]` — existing signature gains one optional, backward-compatible parameter. The sole existing caller, `open_cameras()` at `capture_pipeline.py:216`, calls `open_blackfly(config)` with no change in behavior. Used by Task 5.

- [ ] **Step 1: Write the verification script**

Create `<scratchpad>/verify_gain_sweep_helpers.py`:

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/cfenati/projects/MasterThesis/FullPipeline")
sys.path.insert(0, str(PROJECT_ROOT))

from capture_pipeline import gain_sweep_values, load_config, open_blackfly

values = gain_sweep_values({"start": 20, "stop": 40, "step": 5})
print("default sweep:", values)
assert values == [20.0, 25.0, 30.0, 35.0, 40.0], values

values_uneven = gain_sweep_values({"start": 20, "stop": 40, "step": 7})
print("uneven sweep:", values_uneven)
assert values_uneven == [20.0, 27.0, 34.0], values_uneven

config = load_config(PROJECT_ROOT / "config.yaml")
blackfly = open_blackfly(config, force_manual_gain=True)
assert blackfly is not None, "blackfly.enabled must be true in config.yaml for this check"
assert blackfly.gain_auto is False, "force_manual_gain=True must force gain_auto off"
blackfly.release()

print("PASS")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python3 <scratchpad>/verify_gain_sweep_helpers.py`
Expected: `ImportError: cannot import name 'gain_sweep_values'` (or a `TypeError` on `force_manual_gain` if imports are adjusted manually to test partially) — either way, fails before implementation.

- [ ] **Step 3: Implement both changes**

Replace `open_blackfly` at `capture_pipeline.py:165-182`:

```python
def open_blackfly(
    config: dict, force_manual_gain: bool = False
) -> Optional[BlackflyCamera]:
    blackfly_config = get_blackfly_config(config)
    if blackfly_config is None:
        return None

    gain_auto = (
        False if force_manual_gain else bool(blackfly_config.get("gain_auto", False))
    )
    blackfly = BlackflyCamera(
        name=blackfly_config.get("name", "FLIR Blackfly"),
        camera_index=int(blackfly_config.get("camera_index", 0)),
        serial=blackfly_config.get("serial"),
        timeout_ms=int(blackfly_config.get("timeout_ms", 1000)),
        gain_auto=gain_auto,
        gain=blackfly_config.get("gain"),
        max_fps=blackfly_config.get("max_fps"),
        preview_max_width=blackfly_config.get("preview_max_width", 1280),
        stream_newest_only=bool(blackfly_config.get("stream_newest_only", True)),
    )
    blackfly.open()
    return blackfly


def gain_sweep_values(gain_sweep_config: dict) -> list[float]:
    """Inclusive start..stop steps (dB) from a blackfly.gain_sweep config block."""
    start = float(gain_sweep_config.get("start", 20))
    stop = float(gain_sweep_config.get("stop", 40))
    step = float(gain_sweep_config.get("step", 5))
    if step <= 0:
        raise ValueError("blackfly.gain_sweep.step must be > 0")

    values = []
    value = start
    while value <= stop + 1e-9:
        values.append(round(value, 2))
        value += step
    return values
```

(This replaces the old 18-line `open_blackfly` body and inserts `gain_sweep_values` immediately after it, ahead of `def get_stability_config`.)

- [ ] **Step 4: Run the verification script again**

Run: `python3 <scratchpad>/verify_gain_sweep_helpers.py`
Expected: prints both lists, then `PASS`. The `open_blackfly` part exercises the real attached camera — note explicitly if it's not attached rather than claiming success.

- [ ] **Step 5: Commit**

```bash
git add capture_pipeline.py
git commit -m "Add gain_sweep_values() helper and open_blackfly(force_manual_gain=)"
```

---

### Task 4: `capture_gain_sweep()`

**Files:**
- Modify: `capture_pipeline.py:332-334` (insert new function between the end of `save_capture()`, line 332, and `def grab_all(`, line 334)
- Test: ad hoc script at `<scratchpad>/verify_capture_gain_sweep.py`

**Interfaces:**
- Consumes: `BlackflyCamera.set_gain()` (Task 1), `BlackflyCamera.grab(full_resolution: bool = False)` (existing), `BlackflyCamera.info()` (existing), `JPEG_PARAMS` (existing module constant).
- Produces: `capture_gain_sweep(blackfly: BlackflyCamera, output_dir: Path, timestamp: str, gain_values: list[float]) -> Path`, used by Task 5's `flir_gain_sweep_mode()`.

- [ ] **Step 1: Write the verification script**

Create `<scratchpad>/verify_capture_gain_sweep.py`:

```python
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/cfenati/projects/MasterThesis/FullPipeline")
sys.path.insert(0, str(PROJECT_ROOT))

from capture_pipeline import (
    capture_gain_sweep,
    gain_sweep_values,
    load_config,
    open_blackfly,
)

output_dir = Path("<scratchpad>/gain_sweep_verify")
if output_dir.exists():
    shutil.rmtree(output_dir)

config = load_config(PROJECT_ROOT / "config.yaml")
blackfly = open_blackfly(config, force_manual_gain=True)
assert blackfly is not None, "blackfly.enabled must be true in config.yaml for this check"

gain_values = gain_sweep_values(config["blackfly"]["gain_sweep"])
timestamp = time.strftime("%Y%m%d_%H%M%S")
session_dir = capture_gain_sweep(blackfly, output_dir, timestamp, gain_values)
blackfly.release()

print("session_dir:", session_dir)
jpgs = sorted(p.name for p in session_dir.glob("*.jpg"))
print("jpgs:", jpgs)
assert jpgs == [
    "flir_gain_20.jpg",
    "flir_gain_25.jpg",
    "flir_gain_30.jpg",
    "flir_gain_35.jpg",
    "flir_gain_40.jpg",
], jpgs

for name in jpgs:
    size = (session_dir / name).stat().st_size
    assert size > 0, f"{name} is empty"

metadata = json.loads((session_dir / "metadata.json").read_text())
assert len(metadata["files"]) == 5, metadata["files"]
for entry in metadata["files"]:
    assert 19.9 <= entry["applied_gain_db"] <= 40.1, entry
assert metadata["blackfly"]["gain_auto"] is False

print("PASS")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python3 <scratchpad>/verify_capture_gain_sweep.py`
Expected: `ImportError: cannot import name 'capture_gain_sweep'`

- [ ] **Step 3: Implement `capture_gain_sweep()`**

Insert into `capture_pipeline.py` between the end of `save_capture()` (line 332, `return session_dir`) and `def grab_all(` (line 334):

```python
def capture_gain_sweep(
    blackfly: BlackflyCamera,
    output_dir: Path,
    timestamp: str,
    gain_values: list[float],
) -> Path:
    session_dir = output_dir / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict] = []
    for requested_gain in gain_values:
        applied_gain = blackfly.set_gain(requested_gain)
        blackfly.grab()  # discard one frame so the new gain settles before the saved frame
        frame = blackfly.grab(full_resolution=True)
        if frame is None:
            print(f"  gain {requested_gain:.1f} dB: grab failed, skipped")
            continue

        filename = f"flir_gain_{int(round(applied_gain))}.jpg"
        cv2.imwrite(str(session_dir / filename), frame, JPEG_PARAMS)
        files.append(
            {
                "file": filename,
                "requested_gain_db": requested_gain,
                "applied_gain_db": applied_gain,
            }
        )
        print(f"  gain {applied_gain:.1f} dB -> {filename}")

    metadata = {
        "timestamp": timestamp,
        "capture_time_iso": datetime.now(timezone.utc).isoformat(),
        "blackfly": blackfly.info(),
        "files": files,
    }
    with (session_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return session_dir
```

- [ ] **Step 4: Run the verification script again**

Run: `python3 <scratchpad>/verify_capture_gain_sweep.py`
Expected: prints the session dir, five `gain ... dB -> flir_gain_NN.jpg` lines, the jpg list, then `PASS`. This exercises the real attached camera — note explicitly if it's not attached rather than claiming success.

- [ ] **Step 5: Commit**

```bash
git add capture_pipeline.py
git commit -m "Add capture_gain_sweep() to save a per-gain FLIR image folder"
```

---

### Task 5: `--flir-gain-sweep` CLI flag and interactive loop

**Files:**
- Modify: `capture_pipeline.py:33` (add a window-name constant after `MAX_CONSECUTIVE_GRAB_FAILURES`)
- Modify: `capture_pipeline.py` — insert new function `flir_gain_sweep_mode()` immediately before `def parse_args() -> argparse.Namespace:` (currently line 663)
- Modify: `capture_pipeline.py:676-680` (`parse_args`, add new argument after `--smoke-test`)
- Modify: `capture_pipeline.py:728-729` (`main`, add dispatch after the `--smoke-test` branch)
- Test: ad hoc script at `<scratchpad>/verify_flir_gain_sweep_e2e.py`

**Interfaces:**
- Consumes: `gain_sweep_values()`, `open_blackfly(force_manual_gain=True)` (Task 3), `capture_gain_sweep()` (Task 4), `get_blackfly_config()`, `resolve_capture_output_dir()`, `TerminalInput` (all existing).
- Produces: `flir_gain_sweep_mode(config_path: Path, show_preview: bool = True, output: Optional[str] = None) -> int`, wired into `main()`.

- [ ] **Step 1: Add the window-name constant**

In `capture_pipeline.py`, change lines 31-33 from:

```python
PROJECT_ROOT = Path(__file__).resolve().parent
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]
MAX_CONSECUTIVE_GRAB_FAILURES = 50
```

to:

```python
PROJECT_ROOT = Path(__file__).resolve().parent
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]
MAX_CONSECUTIVE_GRAB_FAILURES = 50
GAIN_SWEEP_WINDOW_NAME = "FLIR gain sweep preview (s = capture sweep, q = quit)"
```

- [ ] **Step 2: Add `flir_gain_sweep_mode()`**

Insert immediately before `def parse_args() -> argparse.Namespace:`:

```python
def flir_gain_sweep_mode(
    config_path: Path,
    show_preview: bool = True,
    output: Optional[str] = None,
) -> int:
    config = load_config(config_path)
    if get_blackfly_config(config) is None:
        print("FLIR gain sweep requires blackfly.enabled: true in config.yaml")
        return 1

    output_dir = resolve_capture_output_dir(
        config["output_dir"], output or "flir_gain_sweep"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gain_values = gain_sweep_values(config["blackfly"].get("gain_sweep", {}))
    print(f"FLIR gain sweep output: {output_dir}")
    print(f"Gain steps (dB): {gain_values}")

    blackfly: Optional[BlackflyCamera] = None
    terminal_input: Optional[TerminalInput] = None
    try:
        blackfly = open_blackfly(config, force_manual_gain=True)

        if show_preview:
            cv2.namedWindow(GAIN_SWEEP_WINDOW_NAME)
        if sys.stdin.isatty():
            terminal_input = TerminalInput()
            terminal_input.__enter__()

        print("Ready — s = capture sweep | q = quit")

        while True:
            frame = blackfly.grab()
            should_capture = False
            should_quit = False

            if show_preview:
                if frame is not None:
                    cv2.imshow(GAIN_SWEEP_WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    should_capture = True
                if key == ord("q"):
                    should_quit = True
                try:
                    still_open = (
                        cv2.getWindowProperty(
                            GAIN_SWEEP_WINDOW_NAME, cv2.WND_PROP_VISIBLE
                        )
                        >= 1
                    )
                except cv2.error:
                    still_open = False
                if not still_open:
                    should_quit = True

            if terminal_input is not None:
                key = terminal_input.poll_key()
                if key == ord("s"):
                    should_capture = True
                if key == ord("q"):
                    should_quit = True

            if should_capture:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                session_dir = capture_gain_sweep(
                    blackfly, output_dir, timestamp, gain_values
                )
                print(f"Saved gain sweep to {session_dir}")

            if should_quit:
                break

        return 0
    except Exception as error:
        print(f"FLIR gain sweep failed: {error}")
        return 1
    finally:
        if terminal_input is not None:
            terminal_input.__exit__(None, None, None)
        if blackfly is not None:
            blackfly.release()
        if show_preview:
            cv2.destroyAllWindows()
```

- [ ] **Step 3: Add the CLI flag**

In `parse_args()`, after the `--smoke-test` argument (currently `capture_pipeline.py:676-680`):

```python
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Open all cameras, grab one frame, and exit",
    )
    parser.add_argument(
        "--flir-gain-sweep",
        action="store_true",
        help=(
            "Save a folder of FLIR Blackfly frames swept across "
            "blackfly.gain_sweep in config.yaml, instead of running the "
            "normal capture pipeline"
        ),
    )
```

- [ ] **Step 4: Wire the dispatch in `main()`**

In `main()`, after the `--smoke-test` branch (currently `capture_pipeline.py:728-729`):

```python
    if args.smoke_test:
        return smoke_test(args.config)

    if args.flir_gain_sweep:
        return flir_gain_sweep_mode(
            config_path=args.config,
            show_preview=not args.no_preview,
            output=args.output,
        )

    if args.sync_test:
```

- [ ] **Step 5: Write the end-to-end verification script**

Create `<scratchpad>/verify_flir_gain_sweep_e2e.py`. This drives the real interactive loop over a pseudo-terminal (the loop's key handling needs a real tty, which piped stdin doesn't provide) and requires the Blackfly attached:

```python
import os
import pty
import shutil
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/cfenati/projects/MasterThesis/FullPipeline")
output_dir = Path("<scratchpad>/gain_sweep_e2e")
if output_dir.exists():
    shutil.rmtree(output_dir)

master_fd, slave_fd = pty.openpty()
proc = subprocess.Popen(
    [
        "python3",
        "capture_pipeline.py",
        "--flir-gain-sweep",
        "--no-preview",
        "--output",
        str(output_dir),
    ],
    cwd=PROJECT_ROOT,
    stdin=slave_fd,
    stdout=slave_fd,
    stderr=slave_fd,
    close_fds=True,
)
os.close(slave_fd)

time.sleep(2)  # let the Blackfly open
os.write(master_fd, b"s")
time.sleep(4)  # 5 gain steps: discard grab + full-res grab + save, each
os.write(master_fd, b"q")
proc.wait(timeout=15)

output_bytes = b""
try:
    while True:
        chunk = os.read(master_fd, 65536)
        if not chunk:
            break
        output_bytes += chunk
except OSError:
    pass
print(output_bytes.decode(errors="replace"))

sessions = sorted(output_dir.iterdir())
assert len(sessions) == 1, sessions
jpgs = sorted(p.name for p in sessions[0].glob("*.jpg"))
assert jpgs == [
    "flir_gain_20.jpg",
    "flir_gain_25.jpg",
    "flir_gain_30.jpg",
    "flir_gain_35.jpg",
    "flir_gain_40.jpg",
], jpgs
assert (sessions[0] / "metadata.json").exists()
print("PASS")
```

- [ ] **Step 6: Run the end-to-end verification script**

Run: `python3 <scratchpad>/verify_flir_gain_sweep_e2e.py`
Expected: printed subprocess output showing "FLIR gain sweep output: ...", "Gain steps (dB): [20.0, 25.0, 30.0, 35.0, 40.0]", "Ready — s = capture sweep | q = quit", the five `gain ... dB -> flir_gain_NN.jpg` lines, "Saved gain sweep to ...", then `PASS` from the script itself. If the Blackfly is not attached when this step runs, say so explicitly rather than claiming it passed — the earlier `open_blackfly()` call will raise and the subprocess will exit early, which the assertions on `sessions` will catch as a failure.

- [ ] **Step 7: Manual sanity check (preview window)**

Since the pty test above only exercises the `--no-preview` path, separately confirm the live preview window works by running, in a real terminal (not piped):

```bash
python3 capture_pipeline.py --flir-gain-sweep
```

Confirm a window titled "FLIR gain sweep preview (s = capture sweep, q = quit)" shows a live Blackfly feed, pressing `s` prints the five save lines and a new folder appears under `captures/flir_gain_sweep/`, and pressing `q` (or closing the window) exits cleanly. This step needs a human at a real display/terminal — record the outcome rather than skipping it silently.

- [ ] **Step 8: Commit**

```bash
git add capture_pipeline.py
git commit -m "Wire --flir-gain-sweep CLI flag and interactive sweep loop"
```

---

## Self-Review Notes

- **Spec coverage:** CLI flag + config block (Task 2, 5), `set_gain()` (Task 1), FLIR-only sweep loop with live preview + terminal fallback (Task 5), per-trigger sweep folder + `flir_gain_<NN>.jpg` naming + `metadata.json` (Task 4), default `flir_gain_sweep` output folder (Task 5), grab-and-discard settling frame (Task 4), skip-and-warn on a failed grab mid-sweep (Task 4), `--no-preview` support (Task 5), forced manual gain regardless of `config.yaml`'s `gain_auto` (Task 1 + Task 3) — all covered. Out-of-scope items from the spec (no change to `run_pipeline()`, no Tkinter GUI integration, no automatic gain scoring) are respected — no task touches `run_pipeline()` or `gui.py`.
- **Type consistency:** `gain_sweep_values(gain_sweep_config: dict) -> list[float]` (Task 3) is called identically in Tasks 4 and 5. `capture_gain_sweep(blackfly: BlackflyCamera, output_dir: Path, timestamp: str, gain_values: list[float]) -> Path` (Task 4) matches its call site in Task 5. `open_blackfly(config: dict, force_manual_gain: bool = False)` (Task 3) matches its call in Task 5.
- **Placeholder scan:** no TBD/TODO; every step has runnable code. `<scratchpad>` is a literal placeholder for the executor's own scratchpad path (per environment instructions), not a plan gap.
