"""ChArUco detection and pinhole intrinsic calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from calibration.target_board import TargetBoard

DEFAULT_MIN_CORNERS = 12
DEFAULT_MIN_SPREAD = 0.01
DEFAULT_REJECT_SIGMA = 3.0
MIN_VIEWS = 4
MIN_VIEWS_AFTER_REJECTION = 8

# A view with sub-pixel error is never a blunder, however far it sits from the
# median. Rejecting the upper tail of a tight, healthy distribution biases the
# focal length, so only gross outliers are worth removing.
REJECT_FLOOR_PX = 1.0

# Above this RMS the corner detections are dominated by blur / compression noise
# rather than by the lens model.
GOOD_RMS_PX = 0.5
# A principal point further than this (fraction of the frame) from the centre is
# almost always a fitting artefact rather than a real lens decentring.
PRINCIPAL_POINT_TOLERANCE = 0.1


@dataclass
class BoardObservation:
    """Interior ChArUco corners found in a single image."""

    image_path: Path
    corner_ids: np.ndarray
    image_points: np.ndarray
    object_points: np.ndarray
    marker_count: int
    sharpness: float = 0.0

    @property
    def corner_count(self) -> int:
        return int(len(self.corner_ids))

    @property
    def label(self) -> str:
        return self.image_path.parent.name or self.image_path.stem


@dataclass
class CameraIntrinsics:
    name: str
    model: str
    image_size: Tuple[int, int]
    camera_matrix: np.ndarray
    distortion: np.ndarray
    reprojection_error_px: float
    views_used: int
    points_used: int
    per_view_errors: Dict[str, float] = field(default_factory=dict)
    rejected_views: Dict[str, float] = field(default_factory=dict)
    board: Optional[Dict[str, Any]] = None
    source_images: List[str] = field(default_factory=list)
    # Sessions where the board could not be detected at all, keyed by reason.
    discarded_views: Dict[str, str] = field(default_factory=dict)

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    def field_of_view_deg(self) -> Tuple[float, float]:
        width, height = self.image_size
        horizontal = 2.0 * np.degrees(np.arctan2(width / 2.0, self.fx))
        vertical = 2.0 * np.degrees(np.arctan2(height / 2.0, self.fy))
        return float(horizontal), float(vertical)

    def undistort(self, image: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.distortion,
            self.image_size,
            alpha,
            self.image_size,
        )
        return cv2.undistort(image, self.camera_matrix, self.distortion, None, new_matrix)

    def to_dict(self) -> Dict[str, Any]:
        horizontal_fov, vertical_fov = self.field_of_view_deg()
        return {
            "name": self.name,
            "model": self.model,
            "resolution": [int(self.image_size[0]), int(self.image_size[1])],
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion": self.distortion.reshape(-1).tolist(),
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "fov_deg": {"horizontal": horizontal_fov, "vertical": vertical_fov},
            "reprojection_error_px": self.reprojection_error_px,
            "views_used": self.views_used,
            "points_used": self.points_used,
            "board": self.board,
            "source_images": self.source_images,
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
    def load_json(cls, path: Path) -> "CameraIntrinsics":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Intrinsics file not found: {path}")

        with path.open("r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        width, height = data["resolution"]
        return cls(
            name=data["name"],
            model=data.get("model", "pinhole"),
            image_size=(int(width), int(height)),
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            distortion=np.asarray(data["distortion"], dtype=np.float64).reshape(1, -1),
            reprojection_error_px=float(data.get("reprojection_error_px", float("nan"))),
            views_used=int(data.get("views_used", 0)),
            points_used=int(data.get("points_used", 0)),
            per_view_errors=data.get("per_view_errors_px", {}),
            rejected_views=data.get("rejected_views_px", {}),
            board=data.get("board"),
            source_images=data.get("source_images", []),
            discarded_views=data.get("discarded_views", {}),
        )


def _corner_spread(image_points: np.ndarray, image_size: Tuple[int, int]) -> float:
    """Convex-hull area of the detected corners as a fraction of the frame."""
    points = image_points.reshape(-1, 2).astype(np.float32)
    if len(points) < 3:
        return 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(points)))
    return hull_area / float(image_size[0] * image_size[1])


def _retry_without_duplicate_markers(
    detector: Any,
    gray: np.ndarray,
    marker_corners: Optional[Any],
    marker_ids: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Recover a frame where one marker was decoded twice.

    A single duplicated id makes the board ambiguous and OpenCV then returns no
    chessboard corners at all, even when the frame is otherwise perfect.
    """
    if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
        return None, None

    ids = np.asarray(marker_ids).reshape(-1)
    seen: set = set()
    unique_indices = []
    for index, marker_id in enumerate(ids.tolist()):
        if marker_id not in seen:
            seen.add(marker_id)
            unique_indices.append(index)

    if len(unique_indices) == len(ids):
        return None, None

    deduplicated_corners = tuple(
        np.asarray(marker_corners[index], dtype=np.float32).reshape(1, 4, 2)
        for index in unique_indices
    )
    deduplicated_ids = np.asarray(
        [ids[index] for index in unique_indices], dtype=np.int32
    ).reshape(-1, 1)

    try:
        result = detector.detectBoard(gray, None, None, deduplicated_corners, deduplicated_ids)
    except cv2.error:
        return None, None
    return result[0], result[1]


def detect_observations(
    image_paths: Sequence[Path],
    board: TargetBoard,
    min_corners: int = DEFAULT_MIN_CORNERS,
    min_spread: float = DEFAULT_MIN_SPREAD,
    on_image: Optional[Any] = None,
    on_skip: Optional[Any] = None,
) -> Tuple[List[BoardObservation], Tuple[int, int], List[Tuple[Path, str]]]:
    """Detect ChArUco corners in every image.

    Returns the usable observations, the common image size, and (path, reason)
    pairs for images that were skipped.
    """
    detector = board.create_detector()
    all_object_points = board.chessboard_corners()

    observations: List[BoardObservation] = []
    skipped: List[Tuple[Path, str]] = []
    image_size: Optional[Tuple[int, int]] = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append((image_path, "unreadable"))
            continue

        height, width = image.shape[:2]
        if image_size is None:
            image_size = (width, height)
        elif (width, height) != image_size:
            skipped.append((image_path, f"resolution {width}x{height} != {image_size[0]}x{image_size[1]}"))
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, corner_ids, marker_corners, marker_ids = detector.detectBoard(gray)

        if corner_ids is None or len(corner_ids) == 0:
            corners, corner_ids = _retry_without_duplicate_markers(
                detector, gray, marker_corners, marker_ids
            )

        if corner_ids is None or len(corner_ids) < min_corners:
            found = 0 if corner_ids is None else len(corner_ids)
            markers = 0 if marker_ids is None else len(marker_ids)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            skipped.append(
                (
                    image_path,
                    f"only {found} of {board.total_corners} corners (min {min_corners}); "
                    f"{markers} markers seen; sharpness {sharpness:.0f}",
                )
            )
            if on_skip is not None:
                on_skip(image, image_path, corners, corner_ids, marker_corners)
            continue

        spread = _corner_spread(corners, image_size)
        if spread < min_spread:
            skipped.append((image_path, f"corners cover {spread * 100:.1f}% of frame"))
            if on_skip is not None:
                on_skip(image, image_path, corners, corner_ids, marker_corners)
            continue

        flat_ids = np.asarray(corner_ids, dtype=np.int32).reshape(-1)
        observation = BoardObservation(
            image_path=image_path,
            corner_ids=flat_ids,
            image_points=np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
            object_points=all_object_points[flat_ids].reshape(-1, 1, 3),
            marker_count=0 if marker_ids is None else int(len(marker_ids)),
            sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        )
        observations.append(observation)

        if on_image is not None:
            on_image(image, observation)

    if image_size is None:
        raise RuntimeError("No readable images were provided")

    return observations, image_size, skipped


def _run_calibration(
    observations: Sequence[BoardObservation],
    image_size: Tuple[int, int],
    flags: int,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    object_points = [observation.object_points for observation in observations]
    image_points = [observation.image_points for observation in observations]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    rms, camera_matrix, distortion, _, _, _, _, per_view_errors = cv2.calibrateCameraExtended(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=flags,
        criteria=criteria,
    )
    return float(rms), camera_matrix, distortion, np.asarray(per_view_errors).reshape(-1)


def _relative_path(path: Path, root: Optional[Path]) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def outlier_threshold(
    per_view_errors: np.ndarray,
    reject_sigma: float,
    max_view_error_px: Optional[float],
) -> float:
    """Robust cut-off relative to the noise floor of this particular dataset.

    An absolute threshold is meaningless here: soft, compressed frames have a
    ~1 px noise floor, so a fixed limit would discard most of a healthy set.
    Shared with the stereo fit, which faces the same distribution of errors.
    """
    if max_view_error_px is not None:
        return max_view_error_px
    median = float(np.median(per_view_errors))
    deviation = float(np.median(np.abs(per_view_errors - median)))
    return max(median + reject_sigma * 1.4826 * deviation, REJECT_FLOOR_PX)


def calibrate_intrinsics(
    name: str,
    observations: Sequence[BoardObservation],
    image_size: Tuple[int, int],
    board: Optional[TargetBoard] = None,
    rational: bool = False,
    fix_tangential: bool = False,
    fix_k3: bool = False,
    reject_sigma: float = DEFAULT_REJECT_SIGMA,
    max_view_error_px: Optional[float] = None,
    reject_outliers: bool = True,
    source_root: Optional[Path] = None,
) -> CameraIntrinsics:
    """Calibrate a pinhole camera, dropping views whose reprojection error is too high."""
    if len(observations) < MIN_VIEWS:
        raise RuntimeError(
            f"{name}: need at least {MIN_VIEWS} usable views, found {len(observations)}"
        )

    flags = 0
    if rational:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if fix_tangential:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if fix_k3:
        flags |= cv2.CALIB_FIX_K3

    kept = list(observations)
    rejected: Dict[str, float] = {}

    while True:
        rms, camera_matrix, distortion, per_view_errors = _run_calibration(
            kept, image_size, flags
        )
        if not reject_outliers:
            break

        threshold = outlier_threshold(per_view_errors, reject_sigma, max_view_error_px)
        worst = int(np.argmax(per_view_errors))
        if (
            per_view_errors[worst] <= threshold
            or len(kept) - 1 < MIN_VIEWS_AFTER_REJECTION
        ):
            break

        # Drop one view at a time: a single bad view distorts the fit enough to
        # make several healthy views look like outliers.
        rejected[kept[worst].label] = float(per_view_errors[worst])
        kept.pop(worst)

    if rational:
        model = "pinhole-rational"
    elif fix_k3:
        model = "pinhole-k1k2"
    else:
        model = "pinhole"

    return CameraIntrinsics(
        name=name,
        model=model,
        image_size=image_size,
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion=np.asarray(distortion, dtype=np.float64),
        reprojection_error_px=rms,
        views_used=len(kept),
        points_used=sum(observation.corner_count for observation in kept),
        per_view_errors={
            observation.label: float(error)
            for observation, error in zip(kept, per_view_errors)
        },
        rejected_views=rejected,
        board=board.to_dict() if board is not None else None,
        source_images=[_relative_path(o.image_path, source_root) for o in kept],
    )


def coverage_bounds(
    observations: Sequence[BoardObservation],
) -> Tuple[float, float, float, float]:
    points = np.vstack([observation.image_points.reshape(-1, 2) for observation in observations])
    return (
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 1].min()),
        float(points[:, 1].max()),
    )


def quality_warnings(
    intrinsics: CameraIntrinsics,
    observations: Sequence[BoardObservation],
) -> List[str]:
    """Flag results that are numerically valid but not trustworthy."""
    warnings: List[str] = []
    width, height = intrinsics.image_size

    if intrinsics.reprojection_error_px > GOOD_RMS_PX:
        warnings.append(
            f"RMS reprojection error is {intrinsics.reprojection_error_px:.2f} px "
            f"(expected below {GOOD_RMS_PX:.1f} px). The corner detections are limited by "
            "focus, lighting or JPEG compression, not by the lens model."
        )

    offset_x = abs(intrinsics.cx - width / 2.0) / width
    offset_y = abs(intrinsics.cy - height / 2.0) / height
    if max(offset_x, offset_y) > PRINCIPAL_POINT_TOLERANCE:
        warnings.append(
            f"Principal point ({intrinsics.cx:.0f}, {intrinsics.cy:.0f}) is far from the frame "
            f"centre ({width / 2:.0f}, {height / 2:.0f}). With a planar target this usually means "
            "systematic corner bias is being absorbed into cx/cy rather than a decentred lens."
        )

    if intrinsics.views_used < 15:
        warnings.append(
            f"Only {intrinsics.views_used} views were used; 20+ well spread views give a "
            "more stable distortion estimate."
        )

    min_x, max_x, min_y, max_y = coverage_bounds(observations)
    margins = {
        "left": min_x / width,
        "right": (width - max_x) / width,
        "top": min_y / height,
        "bottom": (height - max_y) / height,
    }
    unsampled = [edge for edge, margin in margins.items() if margin > 0.12]
    if unsampled:
        warnings.append(
            f"The board never reached the {', '.join(unsampled)} edge(s) of the frame, "
            "so distortion is extrapolated there."
        )

    return warnings


def coverage_image(
    observations: Sequence[BoardObservation],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """Scatter of every detected corner, to judge how well the frame was sampled."""
    width, height = image_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for observation in observations:
        for point in observation.image_points.reshape(-1, 2):
            cv2.circle(canvas, (int(point[0]), int(point[1])), 3, (0, 200, 255), -1)
    return canvas
