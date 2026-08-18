"""Extrinsic calibration of a camera pair from shared ChArUco views.

Conventions
-----------
The pair is stored exactly as ``cv2.stereoCalibrate`` returns it: a rotation
``R`` and a translation ``T`` that take a point from camera A's frame into
camera B's frame,

    X_b = R @ X_a + T.

Camera B's optical centre in A's frame is therefore ``C_b = -R.T @ T`` and the
baseline is ``|T|``. Note that ``T_world_from_camera`` used by the ``design``
package points the other way; convert with
:func:`design.camera.pose_from_stereo_extrinsics` rather than by hand.

Scale
-----
The baseline is metric only because the board is metric. Every length here
inherits the fractional error of ``square_size_m`` in the board YAML: a 1 %
error in the printed square is a 1 % error in the baseline, and in every
distance triangulated from it. Angles, rectification and pixel residuals are
unaffected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from calibration.opencv_calibrate import (
    DEFAULT_MIN_CORNERS,
    DEFAULT_REJECT_SIGMA,
    BoardObservation,
    CameraIntrinsics,
    detect_observations,
    outlier_threshold,
    scaled_px,
)
from calibration.target_board import TargetBoard

# A pose needs 3 non-collinear correspondences; below this a view contributes
# noise rather than geometry.
DEFAULT_MIN_SHARED_CORNERS = 8
MIN_PAIRS = 4
MIN_PAIRS_AFTER_REJECTION = 6
# Never discard more than this fraction in one fit. Beyond that the set is
# disagreeing with itself, and the answer is to inspect the capture blocks, not
# to keep trimming until a coherent subset appears by chance.
MAX_REJECT_FRACTION = 0.25

GOOD_EPIPOLAR_RMS_PX = 0.5
# How far above the residual of its own fixed intrinsics a stereo fit may sit
# before the excess is the pair's fault rather than the cameras'.
STEREO_RMS_FLOOR_RATIO = 1.5
# Fallback for the same check when the intrinsics carry no recorded residual.
GOOD_STEREO_RMS_PX = 0.5
# A view contributing far fewer corners than the rest still clears the shared
# corner minimum, but over so small a patch that its own pose - and therefore
# the baseline it implies - is guesswork.
WEAK_CORNER_FRACTION = 0.3
# Never flag a single-view baseline that agrees with the set this well, however
# tight the rest of the distribution happens to be.
BASELINE_FLAG_FLOOR = 0.01
# Relative spread of the per-view baseline above which the two cameras were
# probably not rigidly fixed to each other for the whole capture.
RIGID_BASELINE_TOLERANCE = 0.02
RIGID_ROTATION_TOLERANCE_DEG = 0.5
# Below this triangulation angle the depth of a match is poorly constrained.
WEAK_TRIANGULATION_DEG = 5.0
# A gap longer than this between consecutive session timestamps starts a new
# capture block (a separate sitting of the board).
DEFAULT_BLOCK_GAP_MINUTES = 2.0
# When two capture blocks disagree by more than this, one rigid R,T cannot
# describe the whole set - the mount moved, or warmed up, between sittings.
BLOCK_BASELINE_TOLERANCE_MM = 1.0
BLOCK_CONVERGENCE_TOLERANCE_DEG = 0.5


# --------------------------------------------------------------------------- #
# Pairing and detection
# --------------------------------------------------------------------------- #
@dataclass
class StereoObservation:
    """One capture session in which both cameras saw the same board corners."""

    label: str
    left: BoardObservation
    right: BoardObservation
    corner_ids: np.ndarray
    object_points: np.ndarray  # (N, 1, 3) board coordinates in metres
    left_points: np.ndarray  # (N, 1, 2) pixels in camera A
    right_points: np.ndarray  # (N, 1, 2) pixels in camera B

    @property
    def corner_count(self) -> int:
        return int(len(self.corner_ids))


def collect_session_pairs(
    capture_dirs: Sequence[Path],
    camera_a: str,
    camera_b: str,
    exclude: Sequence[str] = (),
) -> Tuple[List[Tuple[str, Path, Path]], List[str]]:
    """Find sessions holding an image from both cameras.

    ``capture_pipeline.py`` writes every camera of a trigger into one timestamped
    folder, so a stereo view is simply a folder containing both files. Having a
    file from each camera is necessary but not sufficient: a session recorded to
    cover one camera's own frame usually holds an image from the other camera in
    which the board is partly or wholly outside the field. Those are dropped
    later, by the detector, on the evidence of the image itself.

    Session names are timestamps and so unique within a capture root, but the
    same session can appear under several roots if it was copied there. The first
    root providing a name wins and the duplicate is reported rather than fitted
    twice.

    ``exclude`` drops sessions by name, which is how a view found to disagree
    with the rest of the set is kept out of the fit without touching the
    captures.

    Returns the ``(label, path_a, path_b)`` triples and a list of notes about
    folders that were not usable.
    """
    pairs: Dict[str, Tuple[str, Path, Path]] = {}
    notes: List[str] = []
    excluded = set(exclude)

    for capture_dir in capture_dirs:
        if not capture_dir.exists():
            raise SystemExit(f"Capture directory not found: {capture_dir}")
        sessions = sorted(path for path in capture_dir.iterdir() if path.is_dir())
        if not sessions:
            notes.append(f"{capture_dir}: no session subfolders")
            continue
        for session in sessions:
            image_a = session / f"{camera_a}.jpg"
            image_b = session / f"{camera_b}.jpg"
            if session.name in excluded:
                notes.append(f"{session.name}: excluded on request")
                continue
            if not image_a.exists() or not image_b.exists():
                missing = camera_a if not image_a.exists() else camera_b
                notes.append(f"{session.name}: no {missing}.jpg")
                continue
            if session.name in pairs:
                notes.append(f"{session.name}: duplicate of {pairs[session.name][1].parent}")
                continue
            pairs[session.name] = (session.name, image_a, image_b)

    if not pairs:
        raise SystemExit(
            f"No session folder under {', '.join(str(d) for d in capture_dirs)} "
            f"contains both {camera_a}.jpg and {camera_b}.jpg.\n"
            "Extrinsics need the board visible in both cameras at the same instant."
        )
    return [pairs[label] for label in sorted(pairs)], notes


def detect_stereo_observations(
    session_pairs: Sequence[Tuple[str, Path, Path]],
    board: TargetBoard,
    min_corners: int = DEFAULT_MIN_CORNERS,
    min_shared_corners: int = DEFAULT_MIN_SHARED_CORNERS,
    on_skip: Optional[Any] = None,
) -> Tuple[
    List[StereoObservation],
    Tuple[int, int],
    Tuple[int, int],
    List[Tuple[str, str]],
]:
    """Detect the board in both images of every session and keep shared corners.

    Only corners identified in *both* images can constrain the relative pose, so
    each camera is detected independently and the two id sets are intersected.

    Returns the usable stereo observations, the image size of each camera, and
    ``(label, reason)`` pairs for sessions that were dropped.
    """
    observations_a, size_a, skipped_a = detect_observations(
        [path_a for _, path_a, _ in session_pairs],
        board,
        min_corners=min_corners,
        on_skip=on_skip,
    )
    observations_b, size_b, skipped_b = detect_observations(
        [path_b for _, _, path_b in session_pairs],
        board,
        min_corners=min_corners,
        on_skip=on_skip,
    )

    by_label_a = {observation.label: observation for observation in observations_a}
    by_label_b = {observation.label: observation for observation in observations_b}
    reasons_a = {path.parent.name: reason for path, reason in skipped_a}
    reasons_b = {path.parent.name: reason for path, reason in skipped_b}

    all_object_points = board.chessboard_corners()
    stereo: List[StereoObservation] = []
    skipped: List[Tuple[str, str]] = []

    for label, _, _ in session_pairs:
        left = by_label_a.get(label)
        right = by_label_b.get(label)
        if left is None or right is None:
            missing = []
            if left is None:
                missing.append(f"A: {reasons_a.get(label, 'not detected')}")
            if right is None:
                missing.append(f"B: {reasons_b.get(label, 'not detected')}")
            skipped.append((label, "; ".join(missing)))
            continue

        shared = np.intersect1d(left.corner_ids, right.corner_ids)
        if len(shared) < min_shared_corners:
            skipped.append(
                (
                    label,
                    f"only {len(shared)} corners seen by both cameras "
                    f"(min {min_shared_corners}); "
                    f"A saw {left.corner_count}, B saw {right.corner_count}",
                )
            )
            continue

        index_a = {int(cid): i for i, cid in enumerate(left.corner_ids)}
        index_b = {int(cid): i for i, cid in enumerate(right.corner_ids)}
        rows_a = [index_a[int(cid)] for cid in shared]
        rows_b = [index_b[int(cid)] for cid in shared]

        stereo.append(
            StereoObservation(
                label=label,
                left=left,
                right=right,
                corner_ids=shared.astype(np.int32),
                object_points=all_object_points[shared].reshape(-1, 1, 3).astype(np.float32),
                left_points=left.image_points.reshape(-1, 2)[rows_a]
                .reshape(-1, 1, 2)
                .astype(np.float32),
                right_points=right.image_points.reshape(-1, 2)[rows_b]
                .reshape(-1, 1, 2)
                .astype(np.float32),
            )
        )

    return stereo, size_a, size_b, skipped


# --------------------------------------------------------------------------- #
# Extrinsics
# --------------------------------------------------------------------------- #
@dataclass
class StereoExtrinsics:
    """Measured relative pose of camera B with respect to camera A."""

    name_a: str
    name_b: str
    image_size_a: Tuple[int, int]
    image_size_b: Tuple[int, int]
    camera_matrix_a: np.ndarray
    distortion_a: np.ndarray
    camera_matrix_b: np.ndarray
    distortion_b: np.ndarray
    R: np.ndarray
    T: np.ndarray
    essential: np.ndarray
    fundamental: np.ndarray
    reprojection_error_px: float
    views_used: int
    points_used: int
    intrinsics_fixed: bool = True
    # RMS each camera's own intrinsic calibration reached, carried over so the
    # stereo RMS can be read against the floor it inherits.
    intrinsic_rms_a_px: Optional[float] = None
    intrinsic_rms_b_px: Optional[float] = None
    per_view_errors: Dict[str, float] = field(default_factory=dict)
    rejected_views: Dict[str, float] = field(default_factory=dict)
    discarded_views: Dict[str, str] = field(default_factory=dict)
    board: Optional[Dict[str, Any]] = None
    source_sessions: List[str] = field(default_factory=list)
    # Filled in by the diagnostics below; all optional so the core fit can be
    # saved on its own.
    epipolar_rms_px: Optional[float] = None
    rectified_vertical_rms_px: Optional[float] = None
    closure_rms_mm: Optional[float] = None
    closure_scale: Optional[float] = None
    observed_depth_m: Dict[str, float] = field(default_factory=dict)

    # -- derived quantities ------------------------------------------------- #
    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.T))

    @property
    def intrinsic_rms_px(self) -> Optional[float]:
        """The residual the two fixed intrinsics already carry, combined.

        A stereo fit with ``CALIB_FIX_INTRINSIC`` reprojects through camera
        matrices it is not allowed to touch, so its RMS cannot fall below the one
        those matrices already had on their own calibration set. This is the floor
        the stereo RMS should be read against; an absolute threshold instead
        grades the intrinsics twice and the pair not at all.
        """
        values = [
            value for value in (self.intrinsic_rms_a_px, self.intrinsic_rms_b_px)
            if value is not None and np.isfinite(value) and value > 0
        ]
        if not values:
            return None
        return float(np.sqrt(np.mean(np.square(values))))

    @property
    def center_b_in_a(self) -> np.ndarray:
        """Camera B's optical centre expressed in camera A's frame, metres."""
        return -self.R.T @ self.T.reshape(3)

    @property
    def rotation_angle_deg(self) -> float:
        """Total rotation between the two camera frames."""
        cos = (float(np.trace(self.R)) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    @property
    def rotation_xyz_deg(self) -> Tuple[float, float, float]:
        """``R`` as rotations about A's x (tilt), y (pan) and z (roll) axes."""
        angles = cv2.RQDecomp3x3(self.R)[0]
        return (float(angles[0]), float(angles[1]), float(angles[2]))

    @property
    def optical_axis_angle_deg(self) -> float:
        """Angle between the two optical axes: how strongly the pair toes in."""
        return float(np.degrees(np.arccos(np.clip(float(self.R[2, 2]), -1.0, 1.0))))

    @property
    def baseline_axis_angle_deg(self) -> float:
        """Angle between the baseline and camera A's optical axis.

        90 degrees is a textbook side-by-side pair. Smaller values mean one camera
        sits partly *in front of* the other, which is what makes a pair hard to
        rectify and, in the limit of 0 degrees, impossible to triangulate with.
        """
        center = self.center_b_in_a
        norm = float(np.linalg.norm(center))
        if norm < 1e-12:
            return float("nan")
        return float(np.degrees(np.arccos(np.clip(center[2] / norm, -1.0, 1.0))))

    @property
    def T_b_from_a(self) -> np.ndarray:
        """4x4 rigid transform taking A's frame to B's frame."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.R
        matrix[:3, 3] = self.T.reshape(3)
        return matrix

    def world_pose_b(
        self,
        R_world_from_a: Optional[np.ndarray] = None,
        t_world_from_a: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Camera B's ``T_world_from_camera`` given camera A's.

        Defaults to A at the world origin, i.e. the world frame *is* A's camera
        frame, which is the natural frame for a pair measured on its own.
        """
        from design.camera import pose_from_stereo_extrinsics

        R_wa = np.eye(3) if R_world_from_a is None else np.asarray(R_world_from_a, float)
        t_wa = np.zeros(3) if t_world_from_a is None else np.asarray(t_world_from_a, float)
        return pose_from_stereo_extrinsics(R_wa, t_wa, self.R, self.T)

    def intrinsics(self, which: str = "a") -> CameraIntrinsics:
        """Repackage one camera's intrinsics, e.g. to undistort with them."""
        if which.lower() not in {"a", "b"}:
            raise ValueError("which must be 'a' or 'b'")
        is_a = which.lower() == "a"
        return CameraIntrinsics(
            name=self.name_a if is_a else self.name_b,
            model="pinhole",
            image_size=self.image_size_a if is_a else self.image_size_b,
            camera_matrix=self.camera_matrix_a if is_a else self.camera_matrix_b,
            distortion=self.distortion_a if is_a else self.distortion_b,
            reprojection_error_px=self.reprojection_error_px,
            views_used=self.views_used,
            points_used=self.points_used,
        )

    # -- io ----------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        tilt, pan, roll = self.rotation_xyz_deg
        return {
            "camera_a": self.name_a,
            "camera_b": self.name_b,
            "convention": "X_b = R @ X_a + T, T in metres, OpenCV camera frames",
            "resolution_a": [int(v) for v in self.image_size_a],
            "resolution_b": [int(v) for v in self.image_size_b],
            "camera_matrix_a": self.camera_matrix_a.tolist(),
            "distortion_a": self.distortion_a.reshape(-1).tolist(),
            "camera_matrix_b": self.camera_matrix_b.tolist(),
            "distortion_b": self.distortion_b.reshape(-1).tolist(),
            "R": self.R.tolist(),
            "T": self.T.reshape(-1).tolist(),
            "T_b_from_a": self.T_b_from_a.tolist(),
            "essential": self.essential.tolist(),
            "fundamental": self.fundamental.tolist(),
            "baseline_m": self.baseline_m,
            "baseline_mm": self.baseline_m * 1000.0,
            "center_b_in_a_mm": (self.center_b_in_a * 1000.0).tolist(),
            "rotation_angle_deg": self.rotation_angle_deg,
            "rotation_xyz_deg": {"tilt_x": tilt, "pan_y": pan, "roll_z": roll},
            "optical_axis_angle_deg": self.optical_axis_angle_deg,
            "baseline_axis_angle_deg": self.baseline_axis_angle_deg,
            "reprojection_error_px": self.reprojection_error_px,
            "intrinsic_rms_a_px": self.intrinsic_rms_a_px,
            "intrinsic_rms_b_px": self.intrinsic_rms_b_px,
            "epipolar_rms_px": self.epipolar_rms_px,
            "rectified_vertical_rms_px": self.rectified_vertical_rms_px,
            "closure_rms_mm": self.closure_rms_mm,
            "closure_scale": self.closure_scale,
            "observed_depth_m": self.observed_depth_m,
            "views_used": self.views_used,
            "points_used": self.points_used,
            "intrinsics_fixed": self.intrinsics_fixed,
            "board": self.board,
            "source_sessions": self.source_sessions,
            "per_view_errors_px": self.per_view_errors,
            "rejected_views_px": self.rejected_views,
            "discarded_views": self.discarded_views,
        }

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as json_file:
            json.dump(self.to_dict(), json_file, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "StereoExtrinsics":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Extrinsics file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))

        width_a, height_a = data["resolution_a"]
        width_b, height_b = data["resolution_b"]
        return cls(
            name_a=data["camera_a"],
            name_b=data["camera_b"],
            image_size_a=(int(width_a), int(height_a)),
            image_size_b=(int(width_b), int(height_b)),
            camera_matrix_a=np.asarray(data["camera_matrix_a"], dtype=np.float64),
            distortion_a=np.asarray(data["distortion_a"], dtype=np.float64).reshape(1, -1),
            camera_matrix_b=np.asarray(data["camera_matrix_b"], dtype=np.float64),
            distortion_b=np.asarray(data["distortion_b"], dtype=np.float64).reshape(1, -1),
            R=np.asarray(data["R"], dtype=np.float64),
            T=np.asarray(data["T"], dtype=np.float64).reshape(3, 1),
            essential=np.asarray(data.get("essential", np.zeros((3, 3))), dtype=np.float64),
            fundamental=np.asarray(data.get("fundamental", np.zeros((3, 3))), dtype=np.float64),
            reprojection_error_px=float(data.get("reprojection_error_px", float("nan"))),
            views_used=int(data.get("views_used", 0)),
            points_used=int(data.get("points_used", 0)),
            intrinsics_fixed=bool(data.get("intrinsics_fixed", True)),
            intrinsic_rms_a_px=data.get("intrinsic_rms_a_px"),
            intrinsic_rms_b_px=data.get("intrinsic_rms_b_px"),
            per_view_errors=data.get("per_view_errors_px", {}),
            rejected_views=data.get("rejected_views_px", {}),
            discarded_views=data.get("discarded_views", {}),
            board=data.get("board"),
            source_sessions=data.get("source_sessions", []),
            epipolar_rms_px=data.get("epipolar_rms_px"),
            rectified_vertical_rms_px=data.get("rectified_vertical_rms_px"),
            closure_rms_mm=data.get("closure_rms_mm"),
            closure_scale=data.get("closure_scale"),
            observed_depth_m=data.get("observed_depth_m", {}),
        )


def _distortion_flags(distortion_a: np.ndarray, distortion_b: np.ndarray) -> int:
    """Match the flags to the distortion vectors the intrinsics were fitted with."""
    longest = max(distortion_a.size, distortion_b.size)
    if longest > 8:
        return cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
    if longest > 5:
        return cv2.CALIB_RATIONAL_MODEL
    return 0


def _run_stereo_calibration(
    observations: Sequence[StereoObservation],
    camera_matrix_a: np.ndarray,
    distortion_a: np.ndarray,
    camera_matrix_b: np.ndarray,
    distortion_b: np.ndarray,
    image_size: Tuple[int, int],
    flags: int,
) -> Dict[str, Any]:
    object_points = [observation.object_points for observation in observations]
    points_a = [observation.left_points for observation in observations]
    points_b = [observation.right_points for observation in observations]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)

    result = cv2.stereoCalibrateExtended(
        object_points,
        points_a,
        points_b,
        camera_matrix_a.copy(),
        distortion_a.copy(),
        camera_matrix_b.copy(),
        distortion_b.copy(),
        image_size,
        np.eye(3),
        np.zeros((3, 1)),
        flags=flags,
        criteria=criteria,
    )
    # OpenCV >= 5 also returns the per-view board poses; older builds stop at
    # the per-view errors.
    rms, K_a, d_a, K_b, d_b, R, T, essential, fundamental = result[:9]
    per_view = np.asarray(result[-1], dtype=float).reshape(len(observations), -1)

    return {
        "rms": float(rms),
        "camera_matrix_a": np.asarray(K_a, dtype=np.float64),
        "distortion_a": np.asarray(d_a, dtype=np.float64),
        "camera_matrix_b": np.asarray(K_b, dtype=np.float64),
        "distortion_b": np.asarray(d_b, dtype=np.float64),
        "R": np.asarray(R, dtype=np.float64),
        "T": np.asarray(T, dtype=np.float64).reshape(3, 1),
        "essential": np.asarray(essential, dtype=np.float64),
        "fundamental": np.asarray(fundamental, dtype=np.float64),
        # One number per view: the RMS over both cameras.
        "per_view": np.sqrt(np.mean(per_view ** 2, axis=1)),
    }


def calibrate_stereo(
    intrinsics_a: CameraIntrinsics,
    intrinsics_b: CameraIntrinsics,
    observations: Sequence[StereoObservation],
    board: Optional[TargetBoard] = None,
    fix_intrinsics: bool = True,
    reject_sigma: float = DEFAULT_REJECT_SIGMA,
    max_view_error_px: Optional[float] = None,
    reject_outliers: bool = True,
) -> StereoExtrinsics:
    """Fit the relative pose of two cameras, dropping views that do not agree.

    With ``fix_intrinsics`` the per-camera intrinsics are held at the values
    measured by ``calibrate_cameras.py`` and only ``R`` and ``T`` are estimated.
    That is the recommended mode: a stereo set covers each frame far less well
    than a dedicated intrinsic set, so re-fitting focal length and distortion
    here trades a well-determined quantity for a poorly determined one.

    The rejection threshold is taken from the first fit of the full set and then
    held fixed. Recomputing it after every drop makes the cut-off cascade: each
    removal tightens the distribution, which lowers the threshold, which
    condemns the next view, until a coherent subset appears by chance rather
    than by evidence.
    """
    if len(observations) < MIN_PAIRS:
        raise RuntimeError(
            f"{intrinsics_a.name}/{intrinsics_b.name}: need at least {MIN_PAIRS} "
            f"stereo views, found {len(observations)}"
        )

    distortion_a = intrinsics_a.distortion.reshape(1, -1)
    distortion_b = intrinsics_b.distortion.reshape(1, -1)
    flags = _distortion_flags(distortion_a, distortion_b)
    flags |= cv2.CALIB_FIX_INTRINSIC if fix_intrinsics else cv2.CALIB_USE_INTRINSIC_GUESS

    kept = list(observations)
    rejected: Dict[str, float] = {}
    threshold: Optional[float] = None
    max_rejects = max(
        0,
        min(
            len(kept) - MIN_PAIRS_AFTER_REJECTION,
            int(len(kept) * MAX_REJECT_FRACTION),
        ),
    )

    while True:
        fit = _run_stereo_calibration(
            kept,
            intrinsics_a.camera_matrix,
            distortion_a,
            intrinsics_b.camera_matrix,
            distortion_b,
            intrinsics_a.image_size,
            flags,
        )
        if not reject_outliers:
            break

        per_view = fit["per_view"]
        if threshold is None:
            threshold = outlier_threshold(
                per_view,
                reject_sigma,
                max_view_error_px,
                image_size=intrinsics_a.image_size,
            )
        worst = int(np.argmax(per_view))
        if (
            per_view[worst] <= threshold
            or len(kept) - 1 < MIN_PAIRS_AFTER_REJECTION
            or len(rejected) >= max_rejects
        ):
            break

        # One view at a time: a single bad view pulls the fit far enough to make
        # healthy views look like outliers too. The threshold itself stays put.
        rejected[kept[worst].label] = float(per_view[worst])
        kept.pop(worst)

    return StereoExtrinsics(
        name_a=intrinsics_a.name,
        name_b=intrinsics_b.name,
        image_size_a=intrinsics_a.image_size,
        image_size_b=intrinsics_b.image_size,
        camera_matrix_a=fit["camera_matrix_a"],
        distortion_a=fit["distortion_a"],
        camera_matrix_b=fit["camera_matrix_b"],
        distortion_b=fit["distortion_b"],
        R=fit["R"],
        T=fit["T"],
        essential=fit["essential"],
        fundamental=fit["fundamental"],
        reprojection_error_px=fit["rms"],
        views_used=len(kept),
        points_used=sum(observation.corner_count for observation in kept),
        intrinsics_fixed=fix_intrinsics,
        intrinsic_rms_a_px=intrinsics_a.reprojection_error_px,
        intrinsic_rms_b_px=intrinsics_b.reprojection_error_px,
        per_view_errors={
            observation.label: float(error)
            for observation, error in zip(kept, fit["per_view"])
        },
        rejected_views=rejected,
        board=board.to_dict() if board is not None else None,
        source_sessions=[observation.label for observation in kept],
    )


# --------------------------------------------------------------------------- #
# Per-view relative pose: is the pair actually rigid?
# --------------------------------------------------------------------------- #
def _solve_board_pose(
    observation: BoardObservation,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Pose of the board in one camera, plus its reprojection RMS in pixels."""
    object_points = observation.object_points.reshape(-1, 1, 3).astype(np.float64)
    image_points = observation.image_points.reshape(-1, 1, 2).astype(np.float64)
    if len(object_points) < 4:
        return None

    # The ChArUco target is planar, so use the planar-specific solver and refine
    # it; the generic iterative solver can settle on the mirrored pose.
    flags = getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, distortion, flags=flags
        )
    except cv2.error:
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        return None

    rvec, tvec = cv2.solvePnPRefineLM(
        object_points, image_points, camera_matrix, distortion, rvec, tvec
    )
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
    return cv2.Rodrigues(rvec)[0], tvec.reshape(3), rms


def relative_pose_per_view(
    extrinsics: StereoExtrinsics,
    observations: Sequence[StereoObservation],
) -> List[Dict[str, Any]]:
    """Estimate the pair geometry independently from each view.

    Each view alone determines the board's pose in both cameras and therefore
    the relative pose of the pair. The scatter of these single-view estimates is
    the honest repeatability of the measurement, and the only way to notice that
    the rig moved (or was bumped) partway through the capture: a rigid pair gives
    the same baseline in every view, whatever the board was doing.
    """
    out: List[Dict[str, Any]] = []
    for observation in observations:
        pose_a = _solve_board_pose(
            observation.left, extrinsics.camera_matrix_a, extrinsics.distortion_a
        )
        pose_b = _solve_board_pose(
            observation.right, extrinsics.camera_matrix_b, extrinsics.distortion_b
        )
        if pose_a is None or pose_b is None:
            continue

        R_a, t_a, rms_a = pose_a
        R_b, t_b, rms_b = pose_b
        R_ba = R_b @ R_a.T
        T_ba = t_b - R_ba @ t_a

        centroid = observation.object_points.reshape(-1, 3).mean(axis=0)
        depth_a = float((R_a @ centroid + t_a)[2])
        depth_b = float((R_b @ centroid + t_b)[2])

        cos = (float(np.trace(R_ba)) - 1.0) / 2.0
        out.append(
            {
                "label": observation.label,
                "R": R_ba,
                "T": T_ba,
                "baseline_m": float(np.linalg.norm(T_ba)),
                "rotation_angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
                "depth_a_m": depth_a,
                "depth_b_m": depth_b,
                "pnp_rms_a_px": rms_a,
                "pnp_rms_b_px": rms_b,
                "corners": observation.corner_count,
            }
        )
    return out


def pose_scatter(
    extrinsics: StereoExtrinsics,
    views: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """Spread of the single-view pair estimates around the fitted extrinsics."""
    if not views:
        return {}

    baselines = np.array([view["baseline_m"] for view in views])
    angles, translations = [], []
    for view in views:
        delta_R = view["R"] @ extrinsics.R.T
        angles.append(float(np.degrees(np.linalg.norm(cv2.Rodrigues(delta_R)[0]))))
        translations.append(view["T"].reshape(3) - extrinsics.T.reshape(3))
    translations = np.asarray(translations)
    depths = np.array([view["depth_a_m"] for view in views])

    return {
        "n_views": len(views),
        "baseline_mean_mm": float(baselines.mean() * 1000.0),
        "baseline_std_mm": float(baselines.std(ddof=1) * 1000.0) if len(views) > 1 else 0.0,
        "baseline_spread": float(baselines.std(ddof=1) / baselines.mean()) if len(views) > 1 else 0.0,
        "rotation_deviation_mean_deg": float(np.mean(angles)),
        "rotation_deviation_max_deg": float(np.max(angles)),
        "translation_deviation_rms_mm": float(
            np.sqrt(np.mean(np.sum(translations ** 2, axis=1))) * 1000.0
        ),
        "depth_min_m": float(depths.min()),
        "depth_mean_m": float(depths.mean()),
        "depth_max_m": float(depths.max()),
    }


def pose_ensemble(
    extrinsics: StereoExtrinsics,
    views: Sequence[Dict[str, Any]],
    scale: Optional[float] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Plausible alternatives to the fitted pose, for error propagation.

    The single-view estimates scatter by the repeatability of one measurement.
    The fit averages ``n`` of them, so the deviations are shrunk by
    ``1/sqrt(n)`` to represent the uncertainty of the fitted pose itself. Feed
    the result to :func:`calibration.stereo_metrics.registration_uncertainty` to
    see what that uncertainty costs when mapping one camera onto the other.
    """
    if len(views) < 2:
        return []
    shrink = float(1.0 / np.sqrt(len(views))) if scale is None else float(scale)

    ensemble: List[Tuple[np.ndarray, np.ndarray]] = []
    for view in views:
        delta = cv2.Rodrigues(view["R"] @ extrinsics.R.T)[0].reshape(3) * shrink
        R = cv2.Rodrigues(delta)[0] @ extrinsics.R
        T = extrinsics.T.reshape(3) + (view["T"].reshape(3) - extrinsics.T.reshape(3)) * shrink
        ensemble.append((R, T))
    return ensemble


# --------------------------------------------------------------------------- #
# Residuals: epipolar geometry, rectification, triangulation closure
# --------------------------------------------------------------------------- #
def _undistorted_pixels(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Pixels moved to where an ideal pinhole with the same K would see them."""
    return cv2.undistortPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        camera_matrix,
        distortion,
        P=camera_matrix,
    ).reshape(-1, 2)


def epipolar_residuals(
    extrinsics: StereoExtrinsics,
    observations: Sequence[StereoObservation],
) -> Dict[str, Any]:
    """Symmetric epipolar distance of every shared corner, in pixels.

    This is the residual that matters for stereo matching: it says how far off a
    horizontal search along the epipolar line would be. Unlike the calibration
    RMS it does not depend on the board pose, so it stays meaningful when applied
    to any later image pair.
    """
    K_a, K_b = extrinsics.camera_matrix_a, extrinsics.camera_matrix_b
    # F for ideal pinholes, so it can be applied to undistorted pixels.
    fundamental = np.linalg.inv(K_b).T @ extrinsics.essential @ np.linalg.inv(K_a)

    per_view: Dict[str, float] = {}
    residuals: List[np.ndarray] = []
    for observation in observations:
        points_a = _undistorted_pixels(observation.left_points, K_a, extrinsics.distortion_a)
        points_b = _undistorted_pixels(observation.right_points, K_b, extrinsics.distortion_b)
        homogeneous_a = np.column_stack([points_a, np.ones(len(points_a))])
        homogeneous_b = np.column_stack([points_b, np.ones(len(points_b))])

        line_b = homogeneous_a @ fundamental.T  # epipolar line in B for each A point
        line_a = homogeneous_b @ fundamental  # and the other way round
        algebraic = np.abs(np.einsum("ij,ij->i", homogeneous_b, line_b))
        norm_b = np.hypot(line_b[:, 0], line_b[:, 1])
        norm_a = np.hypot(line_a[:, 0], line_a[:, 1])
        distance = 0.5 * algebraic * (1.0 / np.maximum(norm_b, 1e-12)
                                      + 1.0 / np.maximum(norm_a, 1e-12))
        residuals.append(distance)
        per_view[observation.label] = float(np.sqrt(np.mean(distance ** 2)))

    all_residuals = np.concatenate(residuals) if residuals else np.zeros(0)
    return {
        "residuals_px": all_residuals,
        "per_view_rms_px": per_view,
        "rms_px": float(np.sqrt(np.mean(all_residuals ** 2))) if all_residuals.size else float("nan"),
        "max_px": float(all_residuals.max()) if all_residuals.size else float("nan"),
        "p95_px": float(np.percentile(all_residuals, 95)) if all_residuals.size else float("nan"),
    }


@dataclass
class Rectification:
    """Output of ``cv2.stereoRectify`` plus what it means for this pair."""

    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    roi_a: Tuple[int, int, int, int]
    roi_b: Tuple[int, int, int, int]
    image_size: Tuple[int, int]
    alpha: float
    # True when the baseline is closer to vertical than horizontal, so the
    # rectified frames are the originals turned on their side.
    rotated: bool = False
    # True when OpenCV's own scaling had to be replaced, see :func:`rectify`.
    degenerate: bool = False
    # Share of each rectified frame that real pixels reach.
    valid_fraction: Tuple[float, float] = (float("nan"), float("nan"))
    # True when camera B's sensor is relabelled by a clean 180-degree in-plane
    # turn before rectifying, see :func:`_prefer_flipped_b`.
    b_flipped: bool = False
    @property
    def focal_px(self) -> float:
        return float(self.P1[0, 0])

    @property
    def baseline_m(self) -> float:
        """Baseline of the rectified pair, from P2's translation term."""
        return float(abs(self.P2[0, 3] / self.P2[0, 0])) if self.P2[0, 0] else 0.0

    @property
    def disparity_at_infinity_px(self) -> float:
        """Disparity of an infinitely distant point.

        Zero in the textbook setup, where the two principal points coincide. When
        they are offset so that zero disparity falls on the working plane instead,
        this is how far that moved it.
        """
        return float(self.P1[0, 2] - self.P2[0, 2])

    def depth_from_disparity(self, disparity_px) -> np.ndarray:
        """Range in metres from a horizontal offset, via ``Q``."""
        disparity = np.asarray(disparity_px, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.Q[2, 3] / (self.Q[3, 2] * disparity + self.Q[3, 3])


def _flip_camera_180(
    camera_matrix: np.ndarray, distortion: np.ndarray, image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Camera model as seen through a 180-degree in-plane relabelling of the sensor.

    Turning a camera's own axes 180 degrees about its optical axis is just a
    choice of labelling - it does not change what the lens does - so the
    principal point mirrors to the opposite corner and the tangential
    distortion terms (the only ones that are not rotationally symmetric) flip
    sign; radial terms are unchanged.
    """
    width, height = image_size
    camera_matrix = camera_matrix.copy()
    camera_matrix[0, 2] = width - 1 - camera_matrix[0, 2]
    camera_matrix[1, 2] = height - 1 - camera_matrix[1, 2]
    distortion = distortion.copy()
    flat = distortion.reshape(-1)
    if flat.size >= 4:
        flat[2] *= -1.0  # p1
        flat[3] *= -1.0  # p2
    return camera_matrix, distortion


def _prefer_flipped_b(
    R: np.ndarray,
    T: np.ndarray,
    camera_matrix_b: np.ndarray,
    distortion_b: np.ndarray,
    image_size_b: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Prefer whichever labelling of camera B's axes keeps the pair's rotation small.

    ``cv2.stereoCalibrate`` reports ``R`` for however B's sensor happens to be
    read out. When B is mounted rolled close to 180 degrees relative to A -
    common when a module is flipped to route its cable the other way - that
    roll rides on top of the true toe-in and is large enough to break
    ``cv2.stereoRectify``'s own scaling (see :func:`rectify`). Relabelling B's
    axes by a clean 180-degree turn is free (no physical camera changes), so
    take whichever labelling minimises the total relative rotation; a normal,
    already-close-to-parallel pair is untouched because flipping it would only
    make the rotation larger.
    """
    flip = np.diag([-1.0, -1.0, 1.0])

    def angle(matrix: np.ndarray) -> float:
        cos = (float(np.trace(matrix)) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    if angle(flip @ R) >= angle(R):
        return R, T, camera_matrix_b, distortion_b, False

    camera_matrix_b, distortion_b = _flip_camera_180(camera_matrix_b, distortion_b, image_size_b)
    return flip @ R, flip @ T, camera_matrix_b, distortion_b, True


def rectify(
    extrinsics: StereoExtrinsics,
    alpha: float = 0.0,
    reference_depth: Optional[float] = None,
) -> Rectification:
    """Row-align the pair so a match differs only in its horizontal coordinate.

    ``alpha = 0`` keeps only pixels valid in both frames; ``alpha = 1`` keeps
    everything and leaves black borders. A strongly converging pair loses a lot of
    frame here, which is itself a useful verdict on the geometry.

    ``cv2.stereoRectify`` always puts the rectified horizontal axis along the
    baseline. For a pair mounted one above the other that turns the images on
    their side, so the output size is transposed to match - otherwise the content
    does not fit and ``alpha`` compensates with an absurd focal length.

    Rectifying also swings each optical axis until it is perpendicular to the
    baseline. When the baseline is far from perpendicular to begin with (see
    :attr:`StereoExtrinsics.baseline_axis_angle_deg`) that swing is large, part of
    each field of view ends up at or behind infinity, and OpenCV's own scaling
    returns a negative focal length or an empty common rectangle. In that case the
    rotations are still correct, so they are kept and only the new camera matrix is
    replaced by a neutral one: the mean focal length of the two cameras, centred.
    The pair is then still row-aligned, and ``valid_fraction`` reports how little
    of the frame survived.

    In that fallback, ``reference_depth`` additionally offsets the two principal
    points so zero disparity falls on the plane at that distance rather than at
    infinity. Without it a close pair is unreadable: a 50 mm baseline at 130 mm
    puts nearly a whole frame width of disparity between the two views, so the
    subject leaves one of them entirely.
    """
    width, height = extrinsics.image_size_a
    R_b, T_b, camera_matrix_b, distortion_b, b_flipped = _prefer_flipped_b(
        extrinsics.R, extrinsics.T.reshape(3),
        extrinsics.camera_matrix_b, extrinsics.distortion_b, extrinsics.image_size_b,
    )
    rotated = abs(T_b[1]) > abs(T_b[0])
    output_size = (height, width) if rotated else (width, height)

    R1, R2, P1, P2, Q, roi_a, roi_b = cv2.stereoRectify(
        extrinsics.camera_matrix_a,
        extrinsics.distortion_a,
        camera_matrix_b,
        distortion_b,
        (width, height),
        R_b,
        T_b.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
        newImageSize=output_size,
    )
    R1, R2, P1, P2, Q = (np.asarray(x) for x in (R1, R2, P1, P2, Q))

    # R1/R2 keep the pair row-aligned for any R_b/T_b (R1 == R2 @ R_b always
    # holds), but cv2 is free to pick either rectified axis for the baseline -
    # it is not always horizontal. compute_stereo_depth's SGBM only ever
    # searches rows, so a baseline it placed on the vertical axis has to be
    # turned back onto the horizontal one; any common rotation of R1 and R2
    # preserves row-alignment, so this costs nothing. cv2's own P1/P2/Q were
    # built for the unturned R1/R2 and no longer apply, so they get rebuilt
    # the same way the degenerate fallback below does.
    baseline_dir = R2 @ T_b
    turned = abs(baseline_dir[1]) > abs(baseline_dir[0])
    if turned:
        turn = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        R1, R2 = turn @ R1, turn @ R2
        output_size = (output_size[1], output_size[0])
        rotated = not rotated

    out_width, out_height = output_size
    degenerate = turned or not (
        P1[0, 0] > 0
        and 0.0 <= P1[0, 2] <= out_width
        and 0.0 <= P1[1, 2] <= out_height
    )
    if degenerate:
        P1, P2, Q = _neutral_projections(
            extrinsics, R1, R2, output_size, reference_depth,
            R_b=R_b, T_b=T_b, camera_matrix_b=camera_matrix_b,
        )
        roi_a = roi_b = (0, 0, 0, 0)

    rectification = Rectification(
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        roi_a=tuple(int(v) for v in roi_a), roi_b=tuple(int(v) for v in roi_b),
        image_size=output_size, alpha=float(alpha), rotated=rotated,
        degenerate=degenerate, b_flipped=b_flipped,
    )
    rectification.valid_fraction = _valid_fraction(extrinsics, rectification)
    return rectification


def _neutral_projections(
    extrinsics: StereoExtrinsics,
    R1: np.ndarray,
    R2: np.ndarray,
    output_size: Tuple[int, int],
    reference_depth: Optional[float] = None,
    *,
    R_b: Optional[np.ndarray] = None,
    T_b: Optional[np.ndarray] = None,
    camera_matrix_b: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rectified projections with a plain focal length, aimed at the working plane.

    The rectifying rotations are taken as given: they are what makes the pair
    row-aligned and OpenCV gets them right even when its scaling fails. Only the
    focal length and the two principal points are chosen here.

    ``R_b``/``T_b``/``camera_matrix_b`` are the (possibly 180-degree relabelled,
    see :func:`_prefer_flipped_b`) values actually used to reach ``R1``/``R2``;
    they default to the raw extrinsics for callers that never flip B.
    """
    if R_b is None:
        R_b = extrinsics.R
    if T_b is None:
        T_b = extrinsics.T.reshape(3)
    if camera_matrix_b is None:
        camera_matrix_b = extrinsics.camera_matrix_b

    width, height = output_size
    focal = 0.25 * float(
        extrinsics.camera_matrix_a[0, 0] + extrinsics.camera_matrix_a[1, 1]
        + camera_matrix_b[0, 0] + camera_matrix_b[1, 1]
    )
    center_x, center_y = (width - 1.0) / 2.0, (height - 1.0) / 2.0

    # After rectification both cameras share one orientation and camera B is
    # displaced from camera A along the rectified x axis by this much.
    tx = float((R2 @ T_b)[0])

    cx_a = cx_b = center_x
    cy = center_y
    if reference_depth:
        # Point both frames at the centre of the working plane. Rectifying swings
        # each optical axis onto the baseline's perpendicular, which alone moves
        # the subject far off centre, and at close range the baseline adds most of
        # a frame width of disparity on top - so without this the subject leaves
        # at least one of the two frames.
        point = np.array([0.0, 0.0, float(reference_depth)])
        in_a = R1 @ point
        in_b = R2 @ (R_b @ point + T_b)
        cx_a = center_x - focal * in_a[0] / in_a[2]
        cx_b = center_x - focal * in_b[0] / in_b[2]
        # in_b is in_a shifted along x, so one shared value aligns both rows.
        cy = center_y - focal * in_a[1] / in_a[2]

    P1 = np.array([[focal, 0.0, cx_a, 0.0], [0.0, focal, cy, 0.0],
                   [0.0, 0.0, 1.0, 0.0]])
    P2 = np.array([[focal, 0.0, cx_b, focal * tx], [0.0, focal, cy, 0.0],
                   [0.0, 0.0, 1.0, 0.0]])
    Q = np.array([
        [1.0, 0.0, 0.0, -cx_a],
        [0.0, 1.0, 0.0, -cy],
        [0.0, 0.0, 0.0, focal],
        [0.0, 0.0, -1.0 / tx, (cx_a - cx_b) / tx],
    ])
    return P1, P2, Q


def _valid_fraction(
    extrinsics: StereoExtrinsics,
    rectification: "Rectification",
) -> Tuple[float, float]:
    """Share of each rectified frame that real source pixels reach."""
    width, height = extrinsics.image_size_a
    fractions = []
    for which in ("a", "b"):
        map_x, map_y = rectify_maps(extrinsics, rectification, which)
        inside = (
            (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
        )
        fractions.append(float(inside.mean()))
    return (fractions[0], fractions[1])


def rectify_maps(
    extrinsics: StereoExtrinsics,
    rectification: Rectification,
    which: str = "a",
) -> Tuple[np.ndarray, np.ndarray]:
    """Remap tables that undistort and row-align one camera of the pair."""
    is_a = which.lower() == "a"
    if is_a:
        camera_matrix, distortion = extrinsics.camera_matrix_a, extrinsics.distortion_a
    else:
        camera_matrix, distortion = extrinsics.camera_matrix_b, extrinsics.distortion_b
        if rectification.b_flipped:
            camera_matrix, distortion = _flip_camera_180(
                camera_matrix, distortion, extrinsics.image_size_b,
            )
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        rectification.R1 if is_a else rectification.R2,
        rectification.P1 if is_a else rectification.P2,
        rectification.image_size,
        cv2.CV_32FC1,
    )
    if not is_a and rectification.b_flipped:
        # The maps above sample a 180-degree-relabelled B; undo that so they
        # index straight into the raw, unrotated source image instead of
        # requiring callers to rotate it first.
        width, height = extrinsics.image_size_b
        map_x = (width - 1) - map_x
        map_y = (height - 1) - map_y
    return map_x, map_y


def rectified_residuals(
    extrinsics: StereoExtrinsics,
    rectification: Rectification,
    observations: Sequence[StereoObservation],
) -> Dict[str, Any]:
    """Vertical disagreement of known correspondences after rectification.

    In a correctly rectified pair the same board corner sits on the same image
    row in both frames, so the vertical residual is a direct, interpretable
    measure of the extrinsics: it is the number of rows a block matcher would
    have to search to find its correspondence.
    """
    vertical: List[np.ndarray] = []
    disparity: List[np.ndarray] = []
    depths: List[np.ndarray] = []

    camera_matrix_b, distortion_b = extrinsics.camera_matrix_b, extrinsics.distortion_b
    if rectification.b_flipped:
        camera_matrix_b, distortion_b = _flip_camera_180(
            camera_matrix_b, distortion_b, extrinsics.image_size_b,
        )
    width_b, height_b = extrinsics.image_size_b

    for observation in observations:
        points_a = cv2.undistortPoints(
            observation.left_points.astype(np.float64),
            extrinsics.camera_matrix_a, extrinsics.distortion_a,
            R=rectification.R1, P=rectification.P1,
        ).reshape(-1, 2)
        right_points = observation.right_points.astype(np.float64)
        if rectification.b_flipped:
            # R2/P2 were built for the 180-degree-relabelled B; raw corner
            # pixels need the same relabelling before undistorting with them.
            right_points = right_points.copy()
            right_points[..., 0] = (width_b - 1) - right_points[..., 0]
            right_points[..., 1] = (height_b - 1) - right_points[..., 1]
        points_b = cv2.undistortPoints(
            right_points,
            camera_matrix_b, distortion_b,
            R=rectification.R2, P=rectification.P2,
        ).reshape(-1, 2)
        vertical.append(points_a[:, 1] - points_b[:, 1])
        d = points_a[:, 0] - points_b[:, 0]
        disparity.append(d)
        depths.append(rectification.depth_from_disparity(d))

    vertical = np.concatenate(vertical) if vertical else np.zeros(0)
    disparity = np.concatenate(disparity) if disparity else np.zeros(0)
    depths = np.concatenate(depths) if depths else np.zeros(0)
    finite = np.isfinite(depths) & (depths > 0)

    return {
        "vertical_px": vertical,
        "vertical_rms_px": float(np.sqrt(np.mean(vertical ** 2))) if vertical.size else float("nan"),
        "vertical_max_px": float(np.abs(vertical).max()) if vertical.size else float("nan"),
        "disparity_px": disparity,
        "disparity_min_px": float(disparity.min()) if disparity.size else float("nan"),
        "disparity_max_px": float(disparity.max()) if disparity.size else float("nan"),
        # Measured from infinity, not from zero: the two principal points may be
        # offset. Negative means A sits on the far side of the rectified axis, so
        # A is playing the right-hand camera.
        "disparity_reversed": bool(
            disparity.size
            and np.median(disparity) < rectification.disparity_at_infinity_px
        ),
        "depth_min_m": float(depths[finite].min()) if finite.any() else float("nan"),
        "depth_max_m": float(depths[finite].max()) if finite.any() else float("nan"),
    }


def _fit_rigid(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares rigid transform taking ``source`` onto ``target``.

    Also returns the uniform scale that would fit best, reported separately so
    that a scale error shows up in the residual instead of being absorbed.
    """
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    a = source - source_center
    b = target - target_center
    U, S, Vt = np.linalg.svd(a.T @ b)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    scale = float(S[0] + S[1] + d * S[2]) / float(np.sum(a ** 2))
    return R, target_center - R @ source_center, scale


def _triangulate_observation(
    observation: StereoObservation,
    projection_a: np.ndarray,
    projection_b: np.ndarray,
    camera_matrix_a: np.ndarray,
    distortion_a: np.ndarray,
    camera_matrix_b: np.ndarray,
    distortion_b: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Triangulated 3D corners and their known board-space coordinates.

    Points whose triangulated homogeneous weight is too small to trust are
    dropped. Returns ``None`` when fewer than 4 points survive — too few to
    say anything about the view.
    """
    points_a = _undistorted_pixels(observation.left_points, camera_matrix_a, distortion_a)
    points_b = _undistorted_pixels(observation.right_points, camera_matrix_b, distortion_b)
    homogeneous = cv2.triangulatePoints(projection_a, projection_b, points_a.T, points_b.T)
    w = homogeneous[3]
    valid = np.abs(w) > 1e-12
    if valid.sum() < 4:
        return None
    points_3d = (homogeneous[:3, valid] / w[valid]).T
    board_points = observation.object_points.reshape(-1, 3)[valid]
    return points_3d, board_points


def triangulation_closure(
    extrinsics: StereoExtrinsics,
    observations: Sequence[StereoObservation],
) -> Dict[str, Any]:
    """Reconstruct the board from the pair and compare it with the real board.

    The corners are triangulated from the two views and fitted back onto the
    known planar grid. The residual is the 3D accuracy the pair actually
    achieves at the distance the board was held. ``scale`` is the factor by which
    the reconstruction is too large; it is close to 1 by construction, since the
    same board set the baseline, so read it as a consistency check rather than an
    independent verification of the printed square size.
    """
    K_a, K_b = extrinsics.camera_matrix_a, extrinsics.camera_matrix_b
    projection_a = K_a @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_b = K_b @ np.hstack([extrinsics.R, extrinsics.T.reshape(3, 1)])

    per_view: List[Dict[str, float]] = []
    residuals: List[np.ndarray] = []

    for observation in observations:
        result = _triangulate_observation(
            observation, projection_a, projection_b,
            K_a, extrinsics.distortion_a, K_b, extrinsics.distortion_b,
        )
        if result is None:
            continue
        points_3d, board_points = result

        R, t, scale = _fit_rigid(board_points, points_3d)
        fitted = board_points @ R.T + t
        error = np.linalg.norm(points_3d - fitted, axis=1)
        residuals.append(error)
        per_view.append(
            {
                "label": observation.label,
                "depth_m": float(points_3d[:, 2].mean()),
                "rms_mm": float(np.sqrt(np.mean(error ** 2)) * 1000.0),
                "max_mm": float(error.max() * 1000.0),
                "scale": scale,
                "points": int(len(points_3d)),
            }
        )

    all_residuals = np.concatenate(residuals) if residuals else np.zeros(0)
    scales = np.array([view["scale"] for view in per_view]) if per_view else np.zeros(0)
    return {
        "per_view": per_view,
        "residuals_mm": all_residuals * 1000.0,
        "rms_mm": float(np.sqrt(np.mean(all_residuals ** 2)) * 1000.0) if all_residuals.size else float("nan"),
        "max_mm": float(all_residuals.max() * 1000.0) if all_residuals.size else float("nan"),
        "scale_mean": float(scales.mean()) if scales.size else float("nan"),
        "scale_std": float(scales.std(ddof=1)) if scales.size > 1 else 0.0,
    }


def adjacent_corner_distance_errors(
    extrinsics: StereoExtrinsics,
    observations: Sequence[StereoObservation],
    board: TargetBoard,
    tolerance_fraction: float = 0.1,
) -> Dict[str, Any]:
    """Triangulated corner-to-corner distance vs. the board's printed square size.

    :func:`triangulation_closure` fits a rigid transform, plus a free scale,
    onto the known board and reports what is left over — a systematic scale
    error is exactly what that fitted scale is built to absorb, so it barely
    moves the residual. This instead compares the triangulated distance
    between every pair of corners one square apart directly against
    ``board.square_size_m``, with nothing fitted that could hide a scale bias.

    Run on views that were never part of the stereo fit (see
    ``cross_validate_stereo.py``), this is a genuine out-of-sample check of
    the metric scale baked into ``extrinsics.T`` — the one thing in-sample
    reprojection error cannot see, because it is graded on the same poses
    that set the scale in the first place.

    ``tolerance_fraction`` selects pairs whose *known* board-space distance is
    within that fraction of one square size; a diagonal pair differs by ~41%
    and a two-square pair by 100%, so the default of 0.1 cleanly picks out
    only true grid-adjacent (horizontal or vertical) pairs.
    """
    K_a, K_b = extrinsics.camera_matrix_a, extrinsics.camera_matrix_b
    projection_a = K_a @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_b = K_b @ np.hstack([extrinsics.R, extrinsics.T.reshape(3, 1)])
    tolerance_m = tolerance_fraction * board.square_size_m

    per_view: List[Dict[str, Any]] = []
    all_errors_mm: List[np.ndarray] = []

    for observation in observations:
        result = _triangulate_observation(
            observation, projection_a, projection_b,
            K_a, extrinsics.distortion_a, K_b, extrinsics.distortion_b,
        )
        if result is None:
            continue
        points_3d, board_points = result
        if len(points_3d) < 2:
            continue

        known = np.linalg.norm(
            board_points[:, None, :] - board_points[None, :, :], axis=-1
        )
        measured = np.linalg.norm(
            points_3d[:, None, :] - points_3d[None, :, :], axis=-1
        )
        rows, cols = np.triu_indices(len(points_3d), k=1)
        known_pairs = known[rows, cols]
        measured_pairs = measured[rows, cols]

        adjacent = np.abs(known_pairs - board.square_size_m) < tolerance_m
        if not adjacent.any():
            continue

        errors_mm = (measured_pairs[adjacent] - known_pairs[adjacent]) * 1000.0
        all_errors_mm.append(errors_mm)
        per_view.append(
            {
                "label": observation.label,
                "pairs": int(adjacent.sum()),
                "mean_error_mm": float(errors_mm.mean()),
                "rms_error_mm": float(np.sqrt(np.mean(errors_mm ** 2))),
                "max_abs_error_mm": float(np.abs(errors_mm).max()),
            }
        )

    all_errors = np.concatenate(all_errors_mm) if all_errors_mm else np.zeros(0)
    square_mm = board.square_size_m * 1000.0
    return {
        "square_size_mm": square_mm,
        "tolerance_mm": tolerance_m * 1000.0,
        "per_view": per_view,
        "pairs_total": int(all_errors.size),
        "mean_error_mm": float(all_errors.mean()) if all_errors.size else float("nan"),
        "rms_error_mm": float(np.sqrt(np.mean(all_errors ** 2))) if all_errors.size else float("nan"),
        "max_abs_error_mm": float(np.abs(all_errors).max()) if all_errors.size else float("nan"),
        "std_error_mm": float(all_errors.std(ddof=1)) if all_errors.size > 1 else 0.0,
        "mean_relative_error_pct": (
            float(all_errors.mean() / square_mm * 100.0) if all_errors.size else float("nan")
        ),
        "rms_relative_error_pct": (
            float(np.sqrt(np.mean(all_errors ** 2)) / square_mm * 100.0)
            if all_errors.size else float("nan")
        ),
    }


# --------------------------------------------------------------------------- #
# Naming and removing the views that spoil a set
# --------------------------------------------------------------------------- #
def _median_absolute_outliers(
    values: np.ndarray,
    sigma: float,
    floor: float,
) -> Tuple[np.ndarray, float, float]:
    """Two-sided robust outlier test around the median.

    The median absolute deviation rather than the standard deviation, because a
    single bad view inflates the standard deviation by enough to hide inside it.
    """
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    deviation = float(np.median(np.abs(values - median)))
    threshold = max(sigma * 1.4826 * deviation, floor)
    return np.abs(values - median) > threshold, median, threshold


def suspect_views(
    views: Sequence[Dict[str, Any]],
    epipolar: Optional[Dict[str, Any]] = None,
    sigma: float = DEFAULT_REJECT_SIGMA,
    max_epipolar_px: Optional[float] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, List[str]]:
    """Views that survived the fit but disagree with the rest of the set.

    Outlier rejection inside :func:`calibrate_stereo` only removes views the
    least-squares fit cannot accommodate at all. A view can clear that bar and
    still be the reason a metric looks poor, so it is worth naming separately on
    three independent counts:

    * an epipolar residual above the robust cut-off for this set - the pose this
      view implies is not the pose the set agreed on;
    * a single-view baseline far from the median - the same complaint in metres,
      and the one that shows up as a stray point in the rigidity scatter;
    * far fewer shared corners than the rest - not an error yet, but the reason
      one usually follows.

    Returns ``{label: [reason, ...]}``; a view can be flagged on more than one
    count. Nothing is removed here, so the fit stays reproducible from the
    captures alone.
    """
    flags: Dict[str, List[str]] = {}

    def flag(label: str, reason: str) -> None:
        flags.setdefault(label, []).append(reason)

    if epipolar:
        per_view = {
            label: value for label, value in epipolar.get("per_view_rms_px", {}).items()
            if np.isfinite(value)
        }
        if len(per_view) >= MIN_PAIRS:
            labels = list(per_view)
            values = np.array([per_view[label] for label in labels])
            threshold = outlier_threshold(
                values,
                sigma,
                max_epipolar_px,
                image_size=image_size,
            )
            for label, value in zip(labels, values):
                if value > threshold:
                    flag(label, f"epipolar {value:.3f} px, over the {threshold:.3f} px "
                                "cut-off for this set")

    if len(views) >= MIN_PAIRS:
        baselines = np.array([view["baseline_m"] for view in views]) * 1000.0
        outlying, median, threshold = _median_absolute_outliers(
            baselines, sigma, BASELINE_FLAG_FLOOR * float(np.median(baselines))
        )
        for view, value, is_outlying in zip(views, baselines, outlying):
            if is_outlying:
                flag(view["label"], f"baseline {value:.2f} mm on its own, {value - median:+.2f} mm "
                                    f"from the {median:.2f} mm median (cut-off "
                                    f"+/-{threshold:.2f} mm)")

        counts = np.array([view.get("corners", 0) for view in views], dtype=float)
        median_count = float(np.median(counts))
        for view, count in zip(views, counts):
            if count < WEAK_CORNER_FRACTION * median_count:
                flag(view["label"], f"only {int(count)} shared corners against a median of "
                                    f"{int(median_count)}")

    return flags


# --------------------------------------------------------------------------- #
# Capture blocks: sittings separated by time
# --------------------------------------------------------------------------- #
_SESSION_STAMP = re.compile(r"(?:^|_)(\d{8})_(\d{6})(?:$|_)")


def session_minutes(label: str) -> Optional[float]:
    """Minutes since midnight from a ``YYYYMMDD_HHMMSS`` session name, if any."""
    match = _SESSION_STAMP.search(label)
    if match is None:
        return None
    stamp = match.group(2)
    return int(stamp[:2]) * 60 + int(stamp[2:4]) + int(stamp[4:]) / 60.0


def group_capture_blocks(
    observations: Sequence[StereoObservation],
    gap_minutes: float = DEFAULT_BLOCK_GAP_MINUTES,
) -> List[List[StereoObservation]]:
    """Split views into sittings separated by gaps longer than ``gap_minutes``.

    Session labels are timestamps from ``capture_pipeline.py``. A gap of a few
    minutes between consecutive sessions is a new sitting of the board - and,
    on a mount that is not perfectly rigid, a new relative pose of the pair.
    """
    if not observations:
        return []

    def sort_key(observation: StereoObservation) -> Tuple[float, str]:
        minutes = session_minutes(observation.label)
        return (minutes if minutes is not None else float("inf"), observation.label)

    ordered = sorted(observations, key=sort_key)
    blocks: List[List[StereoObservation]] = [[ordered[0]]]
    for previous, observation in zip(ordered, ordered[1:]):
        earlier = session_minutes(previous.label)
        later = session_minutes(observation.label)
        if (
            earlier is not None
            and later is not None
            and later - earlier > gap_minutes
        ):
            blocks.append([])
        blocks[-1].append(observation)
    return blocks


@dataclass
class CaptureBlockSummary:
    """Independent fit of one capture sitting."""

    index: int
    labels: List[str]
    n_views: int
    baseline_mm: float
    convergence_deg: float
    epipolar_rms_px: float
    stereo_rms_px: float
    baseline_std_mm: float
    depth_min_m: float
    depth_max_m: float

    @property
    def span(self) -> str:
        if not self.labels:
            return "-"
        if len(self.labels) == 1:
            return self.labels[0]
        return f"{self.labels[0]} .. {self.labels[-1]}"


def summarize_capture_blocks(
    intrinsics_a: CameraIntrinsics,
    intrinsics_b: CameraIntrinsics,
    observations: Sequence[StereoObservation],
    board: Optional[TargetBoard] = None,
    gap_minutes: float = DEFAULT_BLOCK_GAP_MINUTES,
) -> List[CaptureBlockSummary]:
    """Fit each capture sitting on its own, with no outlier rejection.

    Rejection is disabled on purpose: the point is to see what each sitting
    itself measures, not what a cascade would keep of it. Blocks with fewer than
    :data:`MIN_PAIRS` views are omitted.
    """
    summaries: List[CaptureBlockSummary] = []
    for index, block in enumerate(group_capture_blocks(observations, gap_minutes), start=1):
        if len(block) < MIN_PAIRS:
            continue
        extrinsics = calibrate_stereo(
            intrinsics_a,
            intrinsics_b,
            block,
            board=board,
            reject_outliers=False,
        )
        views = relative_pose_per_view(extrinsics, block)
        scatter = pose_scatter(extrinsics, views)
        epipolar = epipolar_residuals(extrinsics, block)
        summaries.append(
            CaptureBlockSummary(
                index=index,
                labels=[observation.label for observation in block],
                n_views=len(block),
                baseline_mm=extrinsics.baseline_m * 1000.0,
                convergence_deg=extrinsics.optical_axis_angle_deg,
                epipolar_rms_px=float(epipolar["rms_px"]),
                stereo_rms_px=extrinsics.reprojection_error_px,
                baseline_std_mm=float(scatter.get("baseline_std_mm", float("nan"))),
                depth_min_m=float(scatter.get("depth_min_m", float("nan"))),
                depth_max_m=float(scatter.get("depth_max_m", float("nan"))),
            )
        )
    return summaries


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #
def stereo_quality_warnings(
    extrinsics: StereoExtrinsics,
    scatter: Dict[str, float],
    epipolar: Optional[Dict[str, Any]] = None,
    sweep: Optional[Any] = None,
    closure: Optional[Dict[str, Any]] = None,
    rectification: Optional[Rectification] = None,
    blocks: Optional[Sequence[CaptureBlockSummary]] = None,
) -> List[str]:
    """Flag results that are numerically valid but should not be trusted."""
    warnings: List[str] = []

    if rectification is not None and rectification.degenerate:
        warnings.append(
            f"The pair does not rectify: the baseline is only "
            f"{extrinsics.baseline_axis_angle_deg:.0f} deg from {extrinsics.name_a}'s optical "
            f"axis (90 deg would be side by side) and the cameras toe in by "
            f"{extrinsics.optical_axis_angle_deg:.0f} deg, so aligning the rows throws most of "
            "both frames away. Triangulate matched points directly instead of running a "
            "rectified block matcher."
        )

    floor = extrinsics.intrinsic_rms_px
    if floor is not None:
        if extrinsics.reprojection_error_px > STEREO_RMS_FLOOR_RATIO * floor:
            warnings.append(
                f"Stereo RMS is {extrinsics.reprojection_error_px:.2f} px, "
                f"{extrinsics.reprojection_error_px / floor:.1f}x the {floor:.2f} px these "
                "intrinsics already carried on their own calibration set. Only the excess is "
                "about the pair, and it says the fixed intrinsics do not describe the part of "
                "the frame these views land in - recalibrate each camera over the region the "
                "stereo captures actually use."
            )
    else:
        stereo_limit = scaled_px(GOOD_STEREO_RMS_PX, extrinsics.image_size_a)
        if extrinsics.reprojection_error_px > stereo_limit:
            warnings.append(
                f"Stereo RMS is {extrinsics.reprojection_error_px:.2f} px (expected below "
                f"{stereo_limit:.1f} px at {extrinsics.image_size_a[0]}x"
                f"{extrinsics.image_size_a[1]}). Either the corner detections are soft or the "
                "intrinsics do not fit these images; re-check the per-camera calibration first."
            )

    if epipolar:
        epi_limit = scaled_px(GOOD_EPIPOLAR_RMS_PX, extrinsics.image_size_a)
        if epipolar["rms_px"] > epi_limit:
            warnings.append(
                f"Epipolar RMS is {epipolar['rms_px']:.2f} px (expected below "
                f"{epi_limit:.1f} px at this resolution), so a matcher searching one "
                "row along the epipolar line will miss correspondences."
            )

    n_input = extrinsics.views_used + len(extrinsics.rejected_views)
    if extrinsics.rejected_views and n_input > 0:
        fraction = len(extrinsics.rejected_views) / n_input
        if fraction >= MAX_REJECT_FRACTION * 0.8:
            warnings.append(
                f"Rejected {len(extrinsics.rejected_views)} of {n_input} views "
                f"({fraction * 100:.0f} %). That much disagreement usually means the capture "
                "spans more than one relative pose of the pair; check the capture-block "
                "table and fit one sitting with --block N."
            )

    if extrinsics.views_used < 10:
        warnings.append(
            f"Only {extrinsics.views_used} stereo views were used; 15+ views spread over "
            "depth and across both frames make the pose far more stable."
        )

    spread = scatter.get("baseline_spread")
    if spread is not None and spread > RIGID_BASELINE_TOLERANCE:
        warnings.append(
            f"The single-view baseline varies by {spread * 100:.1f} % "
            f"({scatter['baseline_std_mm']:.1f} mm about {scatter['baseline_mean_mm']:.1f} mm). "
            "A rigid pair should agree to well under 1 %, so either a camera moved during "
            "the capture or the views are too weak to constrain the pose."
        )

    rotation_deviation = scatter.get("rotation_deviation_max_deg")
    if rotation_deviation is not None and rotation_deviation > RIGID_ROTATION_TOLERANCE_DEG:
        warnings.append(
            f"One view disagrees with the fitted rotation by {rotation_deviation:.2f} deg; "
            "check that view for motion blur or a mis-decoded board."
        )

    depth_min = scatter.get("depth_min_m")
    depth_max = scatter.get("depth_max_m")
    if depth_min and depth_max and depth_max / max(depth_min, 1e-9) < 1.3:
        warnings.append(
            f"The board only ever sat between {depth_min:.3f} and {depth_max:.3f} m. "
            "Extrinsics fitted at a single distance extrapolate poorly; vary the working "
            "distance by at least a factor of two."
        )

    if blocks and len(blocks) >= 2:
        baselines = [block.baseline_mm for block in blocks]
        convergences = [block.convergence_deg for block in blocks]
        baseline_span = max(baselines) - min(baselines)
        convergence_span = max(convergences) - min(convergences)
        if (
            baseline_span > BLOCK_BASELINE_TOLERANCE_MM
            or convergence_span > BLOCK_CONVERGENCE_TOLERANCE_DEG
        ):
            warnings.append(
                f"The {len(blocks)} capture sittings disagree: baseline spans "
                f"{baseline_span:.2f} mm and convergence spans {convergence_span:.2f} deg. "
                "A rigid mount should not move that much between sittings; the pooled fit "
                "is a compromise. Prefer --block N on the cleanest sitting, or recapture "
                "the pair in one continuous session."
            )

    if sweep is not None and getattr(sweep, "unbounded_depths", None):
        depths = sweep.unbounded_depths
        warnings.append(
            f"At {len(depths)} of the evaluated distances (from {min(depths):.3f} m) part of "
            f"{extrinsics.name_b}'s field never reaches the plane, so its area and the IoU "
            "are undefined there. The share of camera A's own view is still exact."
        )

    if sweep is not None:
        angles = sweep.triangulation_deg.get("axis")
        if angles is not None and np.isfinite(angles).any():
            widest = float(np.nanmax(angles))
            if widest > 45.0:
                warnings.append(
                    f"The triangulation angle reaches {widest:.0f} deg over the evaluated "
                    "range. Beyond roughly 30 deg the two views of a surface stop looking "
                    "alike, so dense matching fails long before the geometry does."
                )

    if closure and np.isfinite(closure.get("scale_mean", float("nan"))):
        offset = abs(closure["scale_mean"] - 1.0)
        if offset > 0.02:
            warnings.append(
                f"Triangulating the board reproduces it {offset * 100:.1f} % off scale, "
                "which should not happen when the same board set the baseline. Suspect "
                "mixed-up cameras or a board config that does not match the print."
            )

    return warnings
