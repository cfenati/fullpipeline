"""Display-free logic behind menu.py: which stages exist, how to launch them, where captures
and results live. Kept free of Tk and of any camera import so it is unit-testable anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

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
    "calibrate": Stage("calibrate", "Calibrate cameras", "calibrate_live.py", touch=True),
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
    return (project_root / config.get("output_dir", "captures")).resolve()


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
