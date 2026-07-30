"""Geometric camera calibration from ChArUco captures.

``opencv_calibrate`` measures one camera at a time (intrinsics); ``stereo``
measures the rigid transform between two of them (extrinsics) and
``stereo_metrics`` turns that transform into overlap, triangulation and
registration numbers as a function of working distance.
"""

from calibration.target_board import TargetBoard
from calibration.opencv_calibrate import (
    BoardObservation,
    CameraIntrinsics,
    calibrate_intrinsics,
    detect_observations,
)
from calibration.stereo import (
    Rectification,
    StereoExtrinsics,
    StereoObservation,
    calibrate_stereo,
    collect_session_pairs,
    detect_stereo_observations,
    epipolar_residuals,
    rectify,
    relative_pose_per_view,
    triangulation_closure,
)
from calibration.stereo_metrics import (
    DepthSweep,
    RegistrationCurve,
    depth_sweep,
    overlap_masks,
    overlap_outline,
    registration_error,
    registration_uncertainty,
)

__all__ = [
    "BoardObservation",
    "CameraIntrinsics",
    "DepthSweep",
    "Rectification",
    "RegistrationCurve",
    "StereoExtrinsics",
    "StereoObservation",
    "TargetBoard",
    "calibrate_intrinsics",
    "calibrate_stereo",
    "collect_session_pairs",
    "depth_sweep",
    "detect_observations",
    "detect_stereo_observations",
    "epipolar_residuals",
    "overlap_masks",
    "overlap_outline",
    "rectify",
    "registration_error",
    "registration_uncertainty",
    "relative_pose_per_view",
    "triangulation_closure",
]
