"""Display-free logic behind menu.py: which stages exist, how to launch them, where captures
and results live. Kept free of Tk and of any camera import so it is unit-testable anywhere.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CAMERA_FILES = ("rgb_cam1.jpg", "rgb_cam2.jpg")
UNNAMED_FOLDER = ""
MAX_NAME_LEN = 40
SCREEN_MARGIN_W = 40
SCREEN_MARGIN_H = 140  # top bar + window title bar + slack
MIN_WINDOW = (960, 540)
_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    script: str
    touch: bool = False           # script accepts --touch
    needs_session: bool = False   # script needs --session <dir>
    window: bool = False          # script accepts --window W H
    naming: bool = False          # operator may name the capture folder
    result: bool = False          # show the Result screen afterwards


STAGES: Dict[str, Stage] = {
    "capture": Stage("capture", "Take images", "capture_pipeline.py", touch=True, naming=True),
    "calibrate": Stage("calibrate", "Calibration", "calibrate_live.py", touch=True),
    "depth": Stage("depth", "Measure wound depth", "measure_wound_depth.py",
                   touch=True, needs_session=True, window=True, result=True),
    "length": Stage("length", "Measure length", "measure_points.py",
                    touch=True, needs_session=True, window=True, result=True),
    "register": Stage("register", "Register images", "register_features.py",
                      needs_session=True, result=True),
}
STAGE_ORDER: List[str] = ["capture", "calibrate", "depth", "length", "register"]


def build_command(stage: Stage, python: str, project_root: Path,
                  session: Optional[Path] = None, name: Optional[str] = None,
                  window: Optional[Tuple[int, int]] = None) -> List[str]:
    command = [python, str(project_root / stage.script)]
    if stage.touch:
        command.append("--touch")
    if stage.needs_session:
        if session is None:
            raise ValueError(f"{stage.title} needs a capture session")
        command += ["--session", str(session)]
    if stage.window and window is not None:
        command += ["--window", str(window[0]), str(window[1])]
    if stage.naming and name:
        command += ["--output", name]
    return command


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_menu_config(project_root: Path) -> Dict[str, Any]:
    with (project_root / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def captures_dir_from(config: Dict[str, Any], project_root: Path) -> Path:
    """``output_dir`` may point at removable storage that isn't mounted right now; fall back
    to the project's own captures/ folder rather than blocking (see storage_problem())."""
    configured = (project_root / config.get("output_dir", "captures")).resolve()
    if configured.exists() or configured.parent.is_dir():
        return configured
    return (project_root / "captures").resolve()


def results_dir_from(config: Dict[str, Any], project_root: Path) -> Path:
    registration = config.get("registration") or {}
    return (project_root / registration.get("output_dir", "registration/results")).resolve()


def hidden_capture_dirs(config: Dict[str, Any], project_root: Path) -> List[Path]:
    """Calibration capture folders (hundreds of sessions) that the measure picker must hide."""
    geo = config.get("geometric_calibration") or {}
    names = [geo.get("intrinsics_captures"), geo.get("cross_validation_captures")]
    names += list(geo.get("stereo_captures") or [])
    return [(project_root / name).resolve() for name in names if name]


def storage_problem(captures_dir: Path) -> Optional[str]:
    """A plain-language message if the captures folder cannot be used, else None.

    ``output_dir`` may point at removable storage (e.g. a USB drive at /mnt/usbdrive). When
    that is not mounted neither the folder nor its parent exists, and creating it with
    ``mkdir(parents=True)`` could silently write to the internal disk instead.
    """
    if captures_dir.exists() or captures_dir.parent.is_dir():
        return None
    return (f"The captures folder {captures_dir} was not found. If captures are saved to a "
            "USB drive, check that it is plugged in, then try again.")


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Session:
    path: Path
    folder: str   # UNNAMED_FOLDER for captures/<timestamp>/, else the folder name
    name: str     # the timestamp directory name
    mtime: float

    @property
    def display(self) -> str:
        try:
            stamp = datetime.strptime(self.name[:15], "%Y%m%d_%H%M%S").strftime("%d %b %Y  %H:%M:%S")
        except ValueError:
            stamp = self.name
        return f"{self.folder} / {stamp}" if self.folder else stamp


def _is_session(directory: Path, cameras: Sequence[str]) -> bool:
    return all((directory / camera).is_file() for camera in cameras)


def find_sessions(captures_dir: Path, hidden: Sequence[Path] = (),
                  cameras: Sequence[str] = CAMERA_FILES) -> List[Session]:
    """Capture sessions at most two levels under ``captures_dir``, newest first.

    ``captures/<timestamp>/`` (unnamed) and ``captures/<name>/<timestamp>/`` (named) both
    count. ``hidden`` folders are skipped without being listed.
    """
    if not captures_dir.is_dir():
        return []
    skip = {Path(path).resolve() for path in hidden}
    sessions: List[Session] = []
    for child in captures_dir.iterdir():
        if not child.is_dir() or child.resolve() in skip:
            continue
        if _is_session(child, cameras):
            sessions.append(Session(child, UNNAMED_FOLDER, child.name, child.stat().st_mtime))
            continue
        for sub in child.iterdir():
            if sub.is_dir() and _is_session(sub, cameras):
                sessions.append(Session(sub, child.name, sub.name, sub.stat().st_mtime))
    sessions.sort(key=lambda s: (s.mtime, s.name), reverse=True)
    return sessions


def latest_session(sessions: Sequence[Session]) -> Optional[Session]:
    return max(sessions, key=lambda s: (s.mtime, s.name)) if sessions else None


def group_by_folder(sessions: Sequence[Session]) -> List[Tuple[str, List[Session]]]:
    ordered = sorted(sessions, key=lambda s: (s.mtime, s.name), reverse=True)
    groups: Dict[str, List[Session]] = {}
    for session in ordered:
        groups.setdefault(session.folder, []).append(session)
    return sorted(groups.items(), key=lambda item: item[1][0].mtime, reverse=True)


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
class Page(NamedTuple):
    items: List[Any]
    index: int
    count: int


def paginate(items: Sequence[Any], page: int, size: int) -> Page:
    count = max(1, -(-len(items) // size))
    index = min(max(page, 0), count - 1)
    return Page(list(items[index * size:(index + 1) * size]), index, count)


def sanitize_session_name(text: str) -> str:
    return _NAME_UNSAFE.sub("", text)[:MAX_NAME_LEN]


def window_for_screen(width: int, height: int) -> Tuple[int, int]:
    """A cv2 window size that fits inside the screen (an oversized one never gets input)."""
    return (max(MIN_WINDOW[0], width - SCREEN_MARGIN_W),
            max(MIN_WINDOW[1], height - SCREEN_MARGIN_H))


# --------------------------------------------------------------------------- #
# Running a stage
# --------------------------------------------------------------------------- #
class StageRun:
    """One stage as a subprocess: output to a log file, polled (never blocked on) by the UI.

    The child runs in its own process group so Stop reaches everything it spawned. Stop
    sends SIGINT first (Python ``finally`` blocks release the cameras), then escalates to
    SIGTERM and SIGKILL if the child ignores it.
    """

    def __init__(self, command: List[str], log_path: Path, cwd: Path,
                 term_after_s: float = 5.0, kill_after_s: float = 8.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._command = command
        self._log_path = log_path
        self._cwd = cwd
        self._term_after_s = term_after_s
        self._kill_after_s = kill_after_s
        self._clock = clock
        self._proc: Optional[subprocess.Popen] = None
        self._t0 = 0.0
        self._stop_at: Optional[float] = None
        self._termed = False
        self._killed = False
        self.started_at = 0.0

    @property
    def stopped(self) -> bool:
        return self._stop_at is not None

    def start(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("wb") as log:
            self._proc = subprocess.Popen(
                self._command, cwd=str(self._cwd), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        self._t0 = self._clock()
        self.started_at = time.time()

    def _signal(self, sig: int) -> None:
        if self._proc is None:
            return
        try:
            os.killpg(self._proc.pid, sig)  # pgid == pid because of start_new_session
        except ProcessLookupError:
            pass

    def stop(self) -> None:
        if self._stop_at is None:
            self._stop_at = self._clock()
            self._signal(signal.SIGINT)

    def poll(self) -> Optional[int]:
        """Exit code, or None while running. Also escalates a Stop the child is ignoring."""
        if self._proc is None:
            return None
        code = self._proc.poll()
        if code is None and self._stop_at is not None:
            waited = self._clock() - self._stop_at
            if waited >= self._kill_after_s and not self._killed:
                self._killed = True
                self._signal(signal.SIGKILL)
            elif waited >= self._term_after_s and not self._termed:
                self._termed = True
                self._signal(signal.SIGTERM)
        return code

    def elapsed(self) -> float:
        return self._clock() - self._t0

    def log_text(self) -> str:
        try:
            return self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


# --------------------------------------------------------------------------- #
# Failure text and results
# --------------------------------------------------------------------------- #
GENERIC_FAILURE = "Something went wrong. Tap Show details for the technical message."

# (substring found in the log, plain-language message); first match wins. The substrings are
# the real messages raised in cameras/*.py, capture_pipeline.py and the measure tools.
_FAILURE_HINTS: Tuple[Tuple[str, str], ...] = (
    ("Cannot open rgb_cam", "An RGB camera could not be opened. Check both USB cables, then try again."),
    ("evo_irimager_usb_init failed", "The thermal camera could not be started. Check its USB cable and power, then try again."),
    ("Thermal config not found", "The thermal camera settings file is missing. Ask whoever maintains the rig."),
    ("No FLIR/Spinnaker cameras found", "The FLIR camera was not found. Check its USB cable, then try again."),
    ("Failed to grab from one or both RGB cameras", "The cameras stopped sending images. Check the USB cables, then try again."),
    ("No stereo extrinsics", "This rig has not been calibrated yet. Run Calibration first."),
    ("is missing rgb_cam", "That capture is missing a camera image. Pick a different capture."),
    ("Failed to decode images", "That capture's images could not be read. Pick a different capture."),
)


def friendly_error(log_text: str) -> str:
    for needle, message in _FAILURE_HINTS:
        if needle in log_text:
            return message
    return GENERIC_FAILURE


def tail_lines(text: str, count: int = 20) -> str:
    return "\n".join(text.rstrip("\n").splitlines()[-count:])


REPORT_GLOB = "report*.txt"
REPORT_NAMES = ("report.txt", "report_features.txt")
RESULT_IMAGE_NAMES = ("annotated.jpg", "measured.jpg", "preview_features.jpg")


@dataclass(frozen=True)
class Report:
    report_path: Path
    image_path: Optional[Path]


def find_newest_report(results_dir: Path, since: float) -> Optional[Report]:
    """The newest report written at or after ``since`` (epoch s), with the image beside it."""
    if not results_dir.is_dir():
        return None
    newest: Optional[Tuple[float, Path]] = None
    for path in results_dir.rglob(REPORT_GLOB):
        if path.name not in REPORT_NAMES or not path.is_file():
            continue
        mtime = path.stat().st_mtime
        if mtime >= since - 1.0 and (newest is None or mtime > newest[0]):
            newest = (mtime, path)
    if newest is None:
        return None
    folder = newest[1].parent
    image = next((folder / name for name in RESULT_IMAGE_NAMES if (folder / name).is_file()), None)
    return Report(newest[1], image)


# --------------------------------------------------------------------------- #
# Desktop shortcut
# --------------------------------------------------------------------------- #
def desktop_entry_text(python: str, menu_script: Path, project_root: Path) -> str:
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        "Name=FullPipeline",
        "Comment=Capture, calibrate and measure",
        f'Exec="{python}" "{menu_script}"',
        f"Path={project_root}",
        "Terminal=false",
        "Categories=Utility;",
        "",
    ])


def install_shortcut(python: str, menu_script: Path, project_root: Path,
                     home: Optional[Path] = None) -> List[Path]:
    home = home or Path.home()
    text = desktop_entry_text(python, menu_script, project_root)
    written: List[Path] = []
    for directory in (home / "Desktop", home / ".local" / "share" / "applications"):
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "FullPipeline.desktop"
        target.write_text(text, encoding="utf-8")
        target.chmod(0o755)
        written.append(target)
    return written
