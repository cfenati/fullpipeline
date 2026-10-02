"""Sequential live calibration: cam1 intrinsics -> cam2 intrinsics -> stereo pair.

No Tkinter or camera imports here on purpose - this module is the state machine
behind ``calibrate_live.py``'s GUI, and is exercised on its own (fed frames from
disk instead of a camera) in the hardware-free verification path.

Each phase captures its own dedicated views (single-camera gate for the two
intrinsics phases, shared-corner gate for the stereo phase) and is fit exactly
once, when its target view count is reached - never per capture. Per-capture
work is limited to saving the frame and a background-thread corner detection
that only feeds the coverage map; the expensive calibrateCamera()/stereo solve
runs by shelling out to the existing offline calibrate_cameras.py/
stereo_calibrate.py CLIs, so results are byte-identical to the offline path.
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from calibration.opencv_calibrate import (
    DEFAULT_MIN_CORNERS,
    DEFAULT_MIN_SPREAD,
    BoardObservation,
    CameraIntrinsics,
    coverage_bounds,
    coverage_image,
    detect_board_in_frame,
    quality_warnings,
)
from calibration.stereo import (
    DEFAULT_MIN_SHARED_CORNERS,
    StereoExtrinsics,
    StereoObservation,
    pair_observations_from_boards,
    pose_scatter,
    relative_pose_per_view,
    stereo_quality_warnings,
)
from calibration.target_board import TargetBoard

PROJECT_ROOT = Path(__file__).resolve().parent.parent

JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]

# Matches quality_warnings()'s own "board never reached this edge" tolerance
# (opencv_calibrate.py's unsampled-edge check) - not re-exported as a shared
# constant there, so duplicated here rather than reaching into a private
# threshold. Keep the two in sync if that check's margin ever changes.
EDGE_MARGIN_TOLERANCE = 0.12

# A mono fit can be "ready" (quality_warnings() finds nothing outright wrong)
# and still be too imprecise a base for a good stereo fit - ready's own RMS
# bar is looser than this. Both are on top of, not instead of, "ready".
DEFAULT_MAX_MONO_RMS_FOR_STEREO = 1.5
DEFAULT_MIN_COVERAGE_FRACTION = 0.5

# A view with fewer than this fraction of the board's corners is usually the
# board partially cropped by the frame edge, not a clean full view. A run of
# these is the leading cause of a distortion coefficient (esp. k3) blowing up
# despite a fine RMS - worth surfacing while still shooting, not just in
# report.txt after the fit.
THIN_VIEW_CORNER_FRACTION = 0.5

# Two of stereo_quality_warnings()'s checks (single-view baseline spread and
# single-view rotation deviation) are a rig-rigidity diagnostic, not a
# registration-accuracy one - hidden by default per user feedback ("the
# important part is that the registration is correct"), matched on a stable
# substring since the message text itself lives in calibration/stereo.py and
# isn't touched here.
_RIGIDITY_WARNING_MARKERS = ("a rigid pair should agree", "disagrees with the fitted rotation")

# stereo_quality_warnings()'s narrow-working-distance-range check assumes the
# rig is meant to generalize across a range of distances - not true for a
# fixed close-range setup imaging one small, nearby subject, where varying
# distance by 2x isn't physically possible and isn't the point. Hidden by
# default per user feedback; independent toggle from rigidity since the two
# are unrelated concerns.
_DEPTH_RANGE_WARNING_MARKERS = ("vary the working distance",)


class Phase(Enum):
    CAM_A = "cam_a"
    CAM_B = "cam_b"
    STEREO = "stereo"


PHASE_ORDER: Tuple[Phase, ...] = (Phase.CAM_A, Phase.CAM_B, Phase.STEREO)


@dataclass
class CaptureQualitySummary:
    """Live spread of corner counts across the current phase's captures so far."""

    count: int
    min_corners: int
    avg_corners: float
    max_corners: int
    total_corners: int
    thin_count: int
    thin_threshold: int


@dataclass
class GateStatus:
    """Whether the board is currently usable for an auto-capture, and why."""

    open: bool
    reason: str
    left: Optional[BoardObservation] = None
    right: Optional[BoardObservation] = None


@dataclass
class PhaseFitResult:
    """Outcome of running the once-per-phase offline fit."""

    phase: Phase
    ready: bool
    view_count: int
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    intrinsics: Optional[CameraIntrinsics] = None  # CAM_A / CAM_B only
    extrinsics: Optional[StereoExtrinsics] = None  # STEREO only
    scatter: Dict[str, float] = field(default_factory=dict)  # STEREO only
    coverage: Optional[np.ndarray] = None
    coverage_fraction: float = 0.0
    edge_gaps: List[str] = field(default_factory=list)
    # Views the fit actually used, worst reprojection error first - capped to
    # the 10 worst so a "discard the worst N" action has something concrete
    # to act on without needing a separate report.txt lookup.
    worst_views: List[Tuple[str, float]] = field(default_factory=list)


def _coverage_fraction(observations: List[BoardObservation], image_size: Tuple[int, int]) -> float:
    """Convex-hull area of every corner seen this phase, as a fraction of the frame.

    Same idea as opencv_calibrate.py's private per-view ``_corner_spread``,
    aggregated across every observation instead of one image - a phase-level
    "how much of the frame did this actually sample" number, not a per-view
    gate.
    """
    if not observations:
        return 0.0
    points = np.vstack([obs.image_points.reshape(-1, 2) for obs in observations]).astype(np.float32)
    if len(points) < 3:
        return 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(points)))
    width, height = image_size
    return hull_area / float(width * height)


class SequentialLiveCalibrationSession:
    """Drives cam1 -> cam2 -> stereo capture, fitting each phase once at target.

    ``captures_dir`` is written in exactly the layout
    ``collect_image_paths()``/``collect_session_pairs()`` already expect
    (``<captures_dir>/<timestamp>/<camera>.jpg`` + ``metadata.json``), so a
    mono-phase session simply omits the other camera's file and an offline
    fit over the same ``captures_dir`` naturally only picks up the sessions
    relevant to it - no per-phase subdirectories needed.
    """

    def __init__(
        self,
        board: TargetBoard,
        board_path: Path,
        captures_dir: Path,
        output_root: Path,
        camera_a: str = "rgb_cam1",
        camera_b: str = "rgb_cam2",
        min_corners: int = DEFAULT_MIN_CORNERS,
        min_shared_corners: int = DEFAULT_MIN_SHARED_CORNERS,
        min_spread: float = DEFAULT_MIN_SPREAD,
        mono_target: int = 20,
        stereo_target: int = 15,
        top_up: int = 5,
        max_mono_rms_for_stereo: Optional[float] = DEFAULT_MAX_MONO_RMS_FOR_STEREO,
        min_coverage_fraction: Optional[float] = DEFAULT_MIN_COVERAGE_FRACTION,
        show_rigidity_warnings: bool = False,
        show_depth_range_warning: bool = False,
    ) -> None:
        self.board = board
        self.board_path = Path(board_path)
        self.captures_dir = Path(captures_dir)
        self.output_root = Path(output_root)
        self.camera_a = camera_a
        self.camera_b = camera_b
        self.min_corners = min_corners
        self.min_shared_corners = min_shared_corners
        self.min_spread = min_spread
        self.top_up = max(1, top_up)
        # None disables that particular check entirely (--no-mono-quality-gate).
        self.max_mono_rms_for_stereo = max_mono_rms_for_stereo
        self.min_coverage_fraction = min_coverage_fraction
        self.show_rigidity_warnings = show_rigidity_warnings
        self.show_depth_range_warning = show_depth_range_warning

        self._detector = board.create_detector()
        self._object_points = board.chessboard_corners()
        self._image_size_a: Optional[Tuple[int, int]] = None
        self._image_size_b: Optional[Tuple[int, int]] = None

        self._phase_index = 0
        self._phase_targets: Dict[Phase, int] = {
            Phase.CAM_A: mono_target,
            Phase.CAM_B: mono_target,
            Phase.STEREO: stereo_target,
        }
        self._results: Dict[Phase, PhaseFitResult] = {}

        self._observations_a: Dict[str, BoardObservation] = {}
        self._observations_b: Dict[str, BoardObservation] = {}
        self._stereo_observations: Dict[str, StereoObservation] = {}
        self._discarded_a: Dict[str, str] = {}
        self._discarded_b: Dict[str, str] = {}
        self._discarded_stereo: Dict[str, str] = {}
        self._session_labels: List[str] = []

        # Bumped on discard/advance so a detection result from a capture made
        # just before the boundary - still in flight on a background thread -
        # is dropped by poll() instead of landing in the wrong phase's dicts.
        self._epoch = 0
        self._pending: "queue.Queue[Tuple[int, Callable[[], None]]]" = queue.Queue()

        self._fit_lock = threading.Lock()
        self._fitting = False
        self._last_fit_trigger_count: int = -1

        self.captures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Read-only state
    # ------------------------------------------------------------------ #
    @property
    def phase(self) -> Phase:
        index = min(self._phase_index, len(PHASE_ORDER) - 1)
        return PHASE_ORDER[index]

    def phase_camera_label(self) -> str:
        if self.phase is Phase.CAM_A:
            return self.camera_a
        if self.phase is Phase.CAM_B:
            return self.camera_b
        return f"{self.camera_a} + {self.camera_b}"

    def target(self) -> int:
        return self._phase_targets[self.phase]

    def view_count(self) -> int:
        if self.phase is Phase.CAM_A:
            return len(self._observations_a)
        if self.phase is Phase.CAM_B:
            return len(self._observations_b)
        return len(self._stereo_observations)

    def is_fitting(self) -> bool:
        return self._fitting

    def last_fit(self) -> Optional[PhaseFitResult]:
        return self._results.get(self.phase)

    def is_done(self) -> bool:
        return self._phase_index >= len(PHASE_ORDER)

    def _current_observations_for_coverage(
        self,
    ) -> Tuple[List[BoardObservation], Optional[Tuple[int, int]]]:
        if self.phase is Phase.CAM_A:
            return list(self._observations_a.values()), self._image_size_a
        if self.phase is Phase.CAM_B:
            return list(self._observations_b.values()), self._image_size_b
        return [obs.left for obs in self._stereo_observations.values()], self._image_size_a

    def current_coverage(self) -> Optional[np.ndarray]:
        observations, image_size = self._current_observations_for_coverage()
        if not observations or image_size is None:
            return None
        return coverage_image(observations, image_size)

    def current_coverage_fraction(self) -> float:
        """Live version of the fraction used to gate entry into the stereo phase."""
        observations, image_size = self._current_observations_for_coverage()
        if not observations or image_size is None:
            return 0.0
        return _coverage_fraction(observations, image_size)

    def edge_gaps(self) -> List[str]:
        """Which frame border(s) the board hasn't reached yet, for live guidance."""
        observations, image_size = self._current_observations_for_coverage()
        if not observations or image_size is None:
            return []
        min_x, max_x, min_y, max_y = coverage_bounds(observations)
        width, height = image_size
        margins = {
            "left": min_x / width,
            "right": (width - max_x) / width,
            "top": min_y / height,
            "bottom": (height - max_y) / height,
        }
        return [edge for edge, margin in margins.items() if margin > EDGE_MARGIN_TOLERANCE]

    def capture_quality_summary(self) -> Optional[CaptureQualitySummary]:
        """Live corner-count spread for the current phase - updates every capture.

        Shown during capture, not only in the post-fit result, so a run of
        thin/partial-board views is visible while there's still a chance to
        correct it instead of discovering it in report.txt afterward.
        """
        observations, _ = self._current_observations_for_coverage()
        if not observations:
            return None
        counts = [obs.corner_count for obs in observations]
        total = self.board.total_corners
        threshold = int(total * THIN_VIEW_CORNER_FRACTION)
        return CaptureQualitySummary(
            count=len(counts),
            min_corners=min(counts),
            avg_corners=sum(counts) / len(counts),
            max_corners=max(counts),
            total_corners=total,
            thin_count=sum(1 for c in counts if c < threshold),
            thin_threshold=threshold,
        )

    # ------------------------------------------------------------------ #
    # Live loop: gate + capture
    # ------------------------------------------------------------------ #
    def check_gate(
        self, preview_a: Optional[np.ndarray], preview_b: Optional[np.ndarray]
    ) -> GateStatus:
        """Cheap per-frame check: is the board usable for a capture right now?

        Runs on already-downscaled preview frames (never touches the observation
        dicts or triggers a fit) so the overlay/status loop stays responsive
        between captures. Only the camera(s) relevant to the current phase are
        graded - a mono phase ignores whatever the other camera sees.
        """
        if self.phase is Phase.CAM_A:
            return self._check_gate_mono(preview_a, is_camera_a=True)
        if self.phase is Phase.CAM_B:
            return self._check_gate_mono(preview_b, is_camera_a=False)

        raw_a = detect_board_in_frame(
            preview_a, self._detector, self._object_points, self.min_corners, self.min_spread
        )
        raw_b = detect_board_in_frame(
            preview_b, self._detector, self._object_points, self.min_corners, self.min_spread
        )
        left = self._wrap(raw_a, Path("<live>") / "preview_a") if raw_a.ok else None
        right = self._wrap(raw_b, Path("<live>") / "preview_b") if raw_b.ok else None
        observation, reason = pair_observations_from_boards(
            "<live>", left, right, self._object_points, self.min_shared_corners
        )
        return GateStatus(open=observation is not None, reason=reason or "ok", left=left, right=right)

    def _check_gate_mono(self, preview: Optional[np.ndarray], is_camera_a: bool) -> GateStatus:
        raw = detect_board_in_frame(
            preview, self._detector, self._object_points, self.min_corners, self.min_spread
        )
        if not raw.ok:
            return GateStatus(open=False, reason=raw.reason)
        observation = self._wrap(raw, Path("<live>") / ("preview_a" if is_camera_a else "preview_b"))
        if is_camera_a:
            return GateStatus(open=True, reason="ok", left=observation)
        return GateStatus(open=True, reason="ok", right=observation)

    @staticmethod
    def _wrap(raw, image_path: Path) -> BoardObservation:
        return BoardObservation(
            image_path=image_path,
            corner_ids=raw.corner_ids,
            image_points=raw.image_points,
            object_points=raw.object_points,
            marker_count=raw.marker_count,
            sharpness=raw.sharpness,
        )

    def _new_session_dir(self) -> Tuple[str, Path]:
        """A fresh, collision-safe ``<captures_dir>/<timestamp>[_N]`` folder."""
        base = time.strftime("%Y%m%d_%H%M%S")
        label = base
        suffix = 1
        while (self.captures_dir / label).exists():
            suffix += 1
            label = f"{base}_{suffix}"
        session_dir = self.captures_dir / label
        session_dir.mkdir(parents=True)
        return label, session_dir

    def _note_image_size(self, image: np.ndarray, is_camera_a: bool) -> None:
        height, width = image.shape[:2]
        if is_camera_a and self._image_size_a is None:
            self._image_size_a = (width, height)
        elif not is_camera_a and self._image_size_b is None:
            self._image_size_b = (width, height)

    def tick(self, frame_a: Optional[np.ndarray], frame_b: Optional[np.ndarray]) -> str:
        """Save a gated capture and queue background detection for it.

        Only the frame(s) relevant to the current phase are saved - a mono
        phase writes just that camera's file, so the shared session layout
        naturally excludes those sessions from the other camera's/stereo's fit.
        Returns the new session's label immediately; detection (and therefore
        this view showing up in view_count()/current_coverage()) lands
        asynchronously once the background thread finishes and poll() drains it.
        """
        phase = self.phase
        label, session_dir = self._new_session_dir()
        self._session_labels.append(label)

        path_a: Optional[Path] = None
        path_b: Optional[Path] = None
        files: Dict[str, str] = {}

        if phase in (Phase.CAM_A, Phase.STEREO) and frame_a is not None:
            self._note_image_size(frame_a, is_camera_a=True)
            path_a = session_dir / f"{self.camera_a}.jpg"
            cv2.imwrite(str(path_a), frame_a, JPEG_PARAMS)
            files[self.camera_a] = path_a.name

        if phase in (Phase.CAM_B, Phase.STEREO) and frame_b is not None:
            self._note_image_size(frame_b, is_camera_a=False)
            path_b = session_dir / f"{self.camera_b}.jpg"
            cv2.imwrite(str(path_b), frame_b, JPEG_PARAMS)
            files[self.camera_b] = path_b.name

        (session_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "timestamp": label,
                    "capture_time_iso": datetime.now(timezone.utc).isoformat(),
                    "source": "calibrate_live.py",
                    "phase": phase.value,
                    "files": files,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        epoch = self._epoch
        thread = threading.Thread(
            target=self._detect_worker,
            args=(phase, epoch, label, frame_a, path_a, frame_b, path_b),
            daemon=True,
        )
        thread.start()
        return label

    def _detect_worker(
        self,
        phase: Phase,
        epoch: int,
        label: str,
        frame_a: Optional[np.ndarray],
        path_a: Optional[Path],
        frame_b: Optional[np.ndarray],
        path_b: Optional[Path],
    ) -> None:
        """Background-thread corner extraction only - no fit runs here.

        Runs off the main/GUI thread so the freeze that used to happen on
        every capture (full-resolution detection blocking the event loop)
        cannot happen: this thread only ever pushes a closure onto
        ``self._pending`` for the main thread to apply via poll().
        """
        if phase is Phase.CAM_A:
            raw = detect_board_in_frame(
                frame_a, self._detector, self._object_points, self.min_corners, self.min_spread
            )
            observation = self._wrap(raw, path_a) if raw.ok else None
            reason = None if raw.ok else raw.reason
            self._pending.put((epoch, lambda: self._apply_mono(True, label, observation, reason)))
            return

        if phase is Phase.CAM_B:
            raw = detect_board_in_frame(
                frame_b, self._detector, self._object_points, self.min_corners, self.min_spread
            )
            observation = self._wrap(raw, path_b) if raw.ok else None
            reason = None if raw.ok else raw.reason
            self._pending.put((epoch, lambda: self._apply_mono(False, label, observation, reason)))
            return

        raw_a = detect_board_in_frame(
            frame_a, self._detector, self._object_points, self.min_corners, self.min_spread
        )
        raw_b = detect_board_in_frame(
            frame_b, self._detector, self._object_points, self.min_corners, self.min_spread
        )
        left = self._wrap(raw_a, path_a) if raw_a.ok else None
        right = self._wrap(raw_b, path_b) if raw_b.ok else None
        reason_a = None if raw_a.ok else raw_a.reason
        reason_b = None if raw_b.ok else raw_b.reason
        self._pending.put((epoch, lambda: self._apply_stereo(label, left, right, reason_a, reason_b)))

    def _apply_mono(
        self, is_camera_a: bool, label: str, observation: Optional[BoardObservation], reason: Optional[str]
    ) -> None:
        observations = self._observations_a if is_camera_a else self._observations_b
        discarded = self._discarded_a if is_camera_a else self._discarded_b
        if observation is not None:
            observations[label] = observation
        elif reason is not None:
            discarded[label] = reason

    def _apply_stereo(
        self,
        label: str,
        left: Optional[BoardObservation],
        right: Optional[BoardObservation],
        reason_a: Optional[str],
        reason_b: Optional[str],
    ) -> None:
        if left is not None:
            self._observations_a[label] = left
        elif reason_a is not None:
            self._discarded_a[label] = reason_a
        if right is not None:
            self._observations_b[label] = right
        elif reason_b is not None:
            self._discarded_b[label] = reason_b

        observation, reason = pair_observations_from_boards(
            label, left, right, self._object_points, self.min_shared_corners
        )
        if observation is not None:
            self._stereo_observations[label] = observation
        elif reason is not None:
            self._discarded_stereo[label] = reason

    def poll(self) -> None:
        """Drain finished background detections into this phase's state.

        Call once per main-loop iteration. All dict mutation happens here, on
        the calling (main) thread only - background threads never touch
        ``self._observations_*``/``self._discarded_*`` directly, so no lock is
        needed around them.
        """
        while True:
            try:
                epoch, apply = self._pending.get_nowait()
            except queue.Empty:
                break
            if epoch == self._epoch:
                apply()

    # ------------------------------------------------------------------ #
    # Phase-end fit
    # ------------------------------------------------------------------ #
    def maybe_start_fit(self) -> bool:
        """Auto-trigger the once-per-phase fit when target is first reached."""
        if self._fitting or self.view_count() < self.target():
            return False
        if self.view_count() == self._last_fit_trigger_count:
            return False
        return self._start_fit()

    def request_refit(self) -> bool:
        """Manual re-run of the current phase's fit on the views captured so far."""
        if self._fitting or self.view_count() == 0:
            return False
        return self._start_fit()

    def _start_fit(self) -> bool:
        with self._fit_lock:
            if self._fitting:
                return False
            self._fitting = True
        self._last_fit_trigger_count = self.view_count()

        phase = self.phase
        if phase in (Phase.CAM_A, Phase.CAM_B):
            camera = self.camera_a if phase is Phase.CAM_A else self.camera_b
            observations = dict(self._observations_a if phase is Phase.CAM_A else self._observations_b)
            args: Tuple = (phase, camera, observations)
            target = self._fit_mono_worker
        else:
            observations = dict(self._stereo_observations)
            args = (phase, observations)
            target = self._fit_stereo_worker

        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
        return True

    def _fit_mono_worker(
        self, phase: Phase, camera: str, observations: Dict[str, BoardObservation]
    ) -> None:
        try:
            self._results[phase] = self._run_mono_fit(camera, observations)
        finally:
            with self._fit_lock:
                self._fitting = False

    def _run_mono_fit(self, camera: str, observations: Dict[str, BoardObservation]) -> PhaseFitResult:
        return_code = subprocess.call([
            sys.executable, str(PROJECT_ROOT / "calibrate_cameras.py"),
            "--camera", camera, "--in", str(self.captures_dir),
            "--board", str(self.board_path), "--out", str(self.output_root),
        ])

        intrinsics_path = self.output_root / camera / "intrinsics.json"
        if not intrinsics_path.exists():
            return PhaseFitResult(
                phase=self.phase, ready=False, view_count=len(observations),
                error="fit failed - no intrinsics.json written; see the console output above",
            )

        intrinsics = CameraIntrinsics.load_json(intrinsics_path)
        warnings = quality_warnings(intrinsics, list(observations.values()))
        # calibrate_cameras.py can exit non-zero after already writing
        # intrinsics.json (e.g. a later reporting step failing) - the numeric
        # fit is still trustworthy, so it's surfaced as a warning rather than
        # discarded outright, but silently hiding the anomaly would be worse.
        if return_code != 0:
            warnings = [
                f"calibrate_cameras.py exited with code {return_code} - intrinsics.json was written, "
                "but report.txt/coverage/undistort-preview may be stale or missing; check the console "
                "output above.",
                *warnings,
            ]
        image_size = self._image_size_a if camera == self.camera_a else self._image_size_b
        observation_list = list(observations.values())
        coverage = coverage_image(observation_list, image_size) if image_size else None
        coverage_fraction = _coverage_fraction(observation_list, image_size) if image_size else 0.0
        worst_views = sorted(intrinsics.per_view_errors.items(), key=lambda kv: kv[1], reverse=True)[:10]
        return PhaseFitResult(
            phase=self.phase, ready=not warnings, view_count=intrinsics.views_used,
            warnings=warnings, intrinsics=intrinsics, coverage=coverage,
            coverage_fraction=coverage_fraction, edge_gaps=self.edge_gaps(),
            worst_views=worst_views,
        )

    def _fit_stereo_worker(self, phase: Phase, observations: Dict[str, StereoObservation]) -> None:
        try:
            self._results[phase] = self._run_stereo_fit(observations)
        finally:
            with self._fit_lock:
                self._fitting = False

    def _run_stereo_fit(self, observations: Dict[str, StereoObservation]) -> PhaseFitResult:
        intrinsics_a_path = self.output_root / self.camera_a / "intrinsics.json"
        intrinsics_b_path = self.output_root / self.camera_b / "intrinsics.json"

        # intrinsics_path() in stereo_calibrate.py prefers config.yaml's fixed
        # intrinsics_rgb_cam1/rgb_cam2 keys over --out unless --intrinsics-a/-b
        # are given explicitly - without these it would silently reuse the OLD
        # canonical intrinsics instead of what phases 1-2 just wrote.
        #
        # --rig "" (empty, not omitted) disables write_rig_pose()'s separate
        # canonical-rig pose lookup (design/config/rig.yaml, by default).
        # That lookup ignores --intrinsics-a/-b entirely and builds every
        # camera listed in the rig config from its own fixed intrinsics path
        # - on a machine with no prior calibration at all, that path doesn't
        # exist yet and stereo_calibrate.py crashes there (after already
        # saving extrinsics.json). A live/exploratory run has no canonical
        # pose to anchor against anyway, so this is the correct choice, not
        # just a crash workaround - confirmed against a real capture that
        # reproduced the crash without this flag and succeeded with it.
        return_code = subprocess.call([
            sys.executable, str(PROJECT_ROOT / "stereo_calibrate.py"),
            "--camera-a", self.camera_a, "--camera-b", self.camera_b,
            "--in", str(self.captures_dir), "--board", str(self.board_path),
            "--out", str(self.output_root),
            "--intrinsics-a", str(intrinsics_a_path), "--intrinsics-b", str(intrinsics_b_path),
            "--no-as-built", "--rig", "",
        ])

        extrinsics_path = self.output_root / f"stereo_{self.camera_a}_{self.camera_b}" / "extrinsics.json"
        if not extrinsics_path.exists():
            return PhaseFitResult(
                phase=self.phase, ready=False, view_count=len(observations),
                error="fit failed - no extrinsics.json written; see the console output above",
            )

        extrinsics = StereoExtrinsics.load_json(extrinsics_path)
        views = relative_pose_per_view(extrinsics, list(observations.values()))
        scatter = pose_scatter(extrinsics, views)
        warnings = stereo_quality_warnings(extrinsics, scatter)
        hidden_markers: Tuple[str, ...] = ()
        if not self.show_rigidity_warnings:
            hidden_markers += _RIGIDITY_WARNING_MARKERS
        if not self.show_depth_range_warning:
            hidden_markers += _DEPTH_RANGE_WARNING_MARKERS
        if hidden_markers:
            warnings = [
                warning for warning in warnings
                if not any(marker in warning for marker in hidden_markers)
            ]
        if return_code != 0:
            warnings = [
                f"stereo_calibrate.py exited with code {return_code} - extrinsics.json was written, "
                "but report.txt/rig_pose.yaml/figures may be stale or missing; check the console "
                "output above.",
                *warnings,
            ]
        left_observations = [obs.left for obs in observations.values()]
        coverage = (
            coverage_image(left_observations, self._image_size_a) if self._image_size_a else None
        )
        coverage_fraction = (
            _coverage_fraction(left_observations, self._image_size_a) if self._image_size_a else 0.0
        )
        worst_views = sorted(extrinsics.per_view_errors.items(), key=lambda kv: kv[1], reverse=True)[:10]
        return PhaseFitResult(
            phase=self.phase, ready=not warnings, view_count=extrinsics.views_used,
            warnings=warnings, extrinsics=extrinsics, scatter=scatter, coverage=coverage,
            coverage_fraction=coverage_fraction, edge_gaps=self.edge_gaps(),
            worst_views=worst_views,
        )

    # ------------------------------------------------------------------ #
    # Phase controls
    # ------------------------------------------------------------------ #
    def request_top_up(self, extra: Optional[int] = None) -> None:
        """Raise the current phase's target so capture resumes without losing views."""
        self._phase_targets[self.phase] += max(1, extra if extra is not None else self.top_up)

    def request_discard(self) -> None:
        """Clear the current phase's captured sessions and start it over."""
        self._epoch += 1
        for label in self._session_labels:
            session_dir = self.captures_dir / label
            if session_dir.exists():
                shutil.rmtree(session_dir)
        self._session_labels.clear()
        self._observations_a.clear()
        self._discarded_a.clear()
        self._observations_b.clear()
        self._discarded_b.clear()
        self._stereo_observations.clear()
        self._discarded_stereo.clear()
        self._results.pop(self.phase, None)
        self._last_fit_trigger_count = -1

    def discard_views(self, labels: List[str]) -> int:
        """Remove specific captured views from the current phase, keeping the rest.

        For pruning the worst offenders named by ``last_fit().worst_views``
        after a not-ready fit, without reshooting the whole phase like
        ``request_discard()`` does. The target drops by the same amount
        removed, landing exactly back on the new (lower) view_count rather
        than leaving room below it - so neither capture nor an auto-fit
        resumes on its own; both wait for an explicit ``request_top_up()``
        ("Capture More") or ``request_refit()`` ("Refit now").
        """
        to_remove = [label for label in labels if label in self._session_labels]
        if not to_remove:
            return 0

        self._epoch += 1
        for label in to_remove:
            session_dir = self.captures_dir / label
            if session_dir.exists():
                shutil.rmtree(session_dir)
            self._session_labels.remove(label)
            self._observations_a.pop(label, None)
            self._discarded_a.pop(label, None)
            self._observations_b.pop(label, None)
            self._discarded_b.pop(label, None)
            self._stereo_observations.pop(label, None)
            self._discarded_stereo.pop(label, None)

        self._phase_targets[self.phase] -= len(to_remove)
        self._results.pop(self.phase, None)
        # Target was just lowered to match the new view_count exactly, which
        # would otherwise read as "target reached" and auto-refit the same
        # reduced set immediately - pre-arm the dedup at the current count so
        # maybe_start_fit() stays quiet until request_top_up() (or a manual
        # request_refit(), which bypasses this dedup on purpose) moves it.
        self._last_fit_trigger_count = self.view_count()
        return len(to_remove)

    def can_advance(self) -> Tuple[bool, str]:
        """Whether Continue is allowed right now, and why not if it isn't.

        Beyond "a fit has run", entering the stereo phase additionally needs
        both mono fits to clear ``max_mono_rms_for_stereo``/
        ``min_coverage_fraction`` - "ready" alone (quality_warnings() found
        nothing wrong) is a looser bar than a stereo fit actually needs, and
        a stereo phase built on an imprecise mono fit fails in a way that
        looks like a stereo-phase problem but isn't one.

        Finishing the stereo phase (the whole session) is gated the same way:
        a not-ready stereo fit - quality warnings, or stereo_calibrate.py
        itself exiting non-zero - must not be able to reach is_done() and get
        auto-promoted over the canonical calibration by promote_to_canonical().
        """
        if self.phase not in self._results:
            return False, "Run a fit first (Refit now) before continuing."
        if self.phase is Phase.STEREO:
            result = self._results[Phase.STEREO]
            if not result.ready:
                return False, "Stereo fit not ready - " + "; ".join(result.warnings)
            return True, ""
        if self.phase is not Phase.CAM_B:
            return True, ""

        problems: List[str] = []
        for phase, label in ((Phase.CAM_A, self.camera_a), (Phase.CAM_B, self.camera_b)):
            result = self._results.get(phase)
            if result is None or result.intrinsics is None:
                problems.append(f"{label}: no successful fit")
                continue
            rms = result.intrinsics.reprojection_error_px
            if self.max_mono_rms_for_stereo is not None and rms > self.max_mono_rms_for_stereo:
                problems.append(
                    f"{label} RMS {rms:.2f}px > {self.max_mono_rms_for_stereo:.2f}px"
                )
            if (
                self.min_coverage_fraction is not None
                and result.coverage_fraction < self.min_coverage_fraction
            ):
                problems.append(
                    f"{label} coverage {result.coverage_fraction * 100:.0f}% < "
                    f"{self.min_coverage_fraction * 100:.0f}%"
                )
        if problems:
            return False, "Not precise enough for stereo yet - " + "; ".join(problems)
        return True, ""

    def advance(self) -> bool:
        """Move to the next phase. False if can_advance() refuses."""
        if not self.can_advance()[0]:
            return False
        self._epoch += 1
        self._phase_index += 1
        if not self.is_done():
            self._session_labels.clear()
            self._observations_a.clear()
            self._discarded_a.clear()
            self._observations_b.clear()
            self._discarded_b.clear()
            self._stereo_observations.clear()
            self._discarded_stereo.clear()
            self._last_fit_trigger_count = -1
        return True
