"""Stereo triangulation: pixel pair + K,R,T -> (X, Y, Z) in camera A, metres.

Nothing here assumes a plane or a known depth. A single pixel is a ray.
Two corresponding pixels (the same physical point seen by both cameras)
intersect in one 3D point. That intersection is this file.

Convention (same as calibration/stereo.py and extrinsics.json)::

    X_b = R @ X_a + T          T in metres
    x_a ~ K_a @ X_a            (homogeneous, after undistortion)
    x_b ~ K_b @ X_b

So the two camera matrices in the linear triangulation are::

    P_a = K_a @ [I | 0]
    P_b = K_b @ [R | T]

Each correspondence ``x = (u, v)`` contributes two independent rows of
``x × (P X) = 0``. Stacking both cameras gives a 4×4 homogeneous system
whose null-space is X. That is Hartley & Zisserman's DLT; OpenCV's
``cv2.triangulatePoints`` is the same SVD, batched.

Read ``triangulate_dlt`` for one point written out. ``triangulate_points``
is the batched call every registration script should use.

Rectified special case (after stereoRectify, matches share a row)::

    Z = Q[2,3] / (Q[3,2] * d + Q[3,3])     d = x_left - x_right

which is the same 3D point, just with the cameras already row-aligned so
the DLT collapses to ``Z = f B / (d - d_infinity)``.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def projection_matrices(
    camera_matrix_a: np.ndarray,
    camera_matrix_b: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """P_a = K_a [I|0], P_b = K_b [R|T]. X is in camera A's frame, metres."""
    projection_a = camera_matrix_a @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_b = camera_matrix_b @ np.hstack([R, np.asarray(T, dtype=np.float64).reshape(3, 1)])
    return projection_a, projection_b


def undistort_points(
    points: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray,
) -> np.ndarray:
    """Move measured pixels to where an ideal pinhole with the same K would see them.

    ``P=camera_matrix`` keeps the output in pixels (not normalised coordinates),
    matching ``cv2.undistort(image, K, dist)`` with no newCameraMatrix.
    """
    undistorted = cv2.undistortPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        camera_matrix, distortion, P=camera_matrix,
    )
    return undistorted.reshape(-1, 2)


def dlt_design_matrix(
    x_a: np.ndarray, x_b: np.ndarray,
    projection_a: np.ndarray, projection_b: np.ndarray,
) -> np.ndarray:
    """4×4 matrix of ``x × P X = 0`` for one correspondence.

    Row 0-1 from camera A, row 2-3 from camera B. SVD's last right-singular
    vector is the homogeneous X that (approximately) lies in the null space.
    """
    ua, va = float(x_a[0]), float(x_a[1])
    ub, vb = float(x_b[0]), float(x_b[1])
    return np.vstack([
        ua * projection_a[2] - projection_a[0],
        va * projection_a[2] - projection_a[1],
        ub * projection_b[2] - projection_b[0],
        vb * projection_b[2] - projection_b[1],
    ])


def triangulate_dlt(
    x_a: np.ndarray, x_b: np.ndarray,
    projection_a: np.ndarray, projection_b: np.ndarray,
) -> np.ndarray:
    """One correspondence -> (X, Y, Z) in camera A, metres.

    For a pixel (u, v) and projection P, ``x × P X = 0`` expands to two rows::

        u * P[2] - P[0]
        v * P[2] - P[1]

    Stack the four rows (two cameras), SVD, take the last right-singular
    vector, dehomogenise. Chirality (X.Z > 0, in front of both cameras) is
    the caller's filter, not this function's.
    """
    design = dlt_design_matrix(x_a, x_b, projection_a, projection_b)
    _, _, vt = np.linalg.svd(design)
    homogeneous = vt[-1]
    return homogeneous[:3] / homogeneous[3]


def triangulate_points(
    pts_a: np.ndarray, pts_b: np.ndarray,
    camera_matrix_a: np.ndarray, camera_matrix_b: np.ndarray,
    R: np.ndarray, T: np.ndarray,
) -> np.ndarray:
    """N correspondences (undistorted pixels) -> (N, 3) points in camera A, metres.

    Same P_a / P_b as ``triangulate_dlt``. ``cv2.triangulatePoints`` is this
    SVD, vectorised over the N pairs.
    """
    projection_a, projection_b = projection_matrices(
        camera_matrix_a, camera_matrix_b, R, T,
    )
    homogeneous = cv2.triangulatePoints(
        projection_a, projection_b,
        np.asarray(pts_a, dtype=np.float64).reshape(-1, 2).T,
        np.asarray(pts_b, dtype=np.float64).reshape(-1, 2).T,
    )
    homogeneous /= homogeneous[3]
    return homogeneous[:3].T


def fundamental_for_undistorted(
    camera_matrix_a: np.ndarray, camera_matrix_b: np.ndarray, essential: np.ndarray,
) -> np.ndarray:
    """F for ideal pinholes, so it can be applied to undistorted pixels.

    ``F = K_b^-T E K_a^-1``. Numerically identical to what ``cv2.stereoCalibrate``
    stores (verified at 1e-15 relative on this rig), but derived here so the
    pinhole assumption is explicit at the point of use and nothing depends on an
    optional saved field.
    """
    inv_a = np.linalg.inv(camera_matrix_a)
    inv_b = np.linalg.inv(camera_matrix_b)
    return inv_b.T @ essential @ inv_a


def epipolar_lines(points_a: np.ndarray, fundamental: np.ndarray) -> np.ndarray:
    """Epipolar line in B for each undistorted point in A, as (a, b, c) with a^2+b^2=1.

    The match for a point in A can lie anywhere along this line in B and nowhere
    else. That is one whole degree of freedom removed for free, which is why a
    click in B only has to be roughly right: projecting it onto this line is
    strictly better than trusting it.
    """
    points = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    lines = homogeneous @ np.asarray(fundamental, dtype=np.float64).T
    norm = np.hypot(lines[:, 0], lines[:, 1])
    return lines / np.maximum(norm, 1e-12)[:, None]


def snap_to_line(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Closest point on each line: drops the click error perpendicular to it."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 3)
    offset = np.einsum("ij,ij->i", np.column_stack([points, np.ones(len(points))]), lines)
    return points - offset[:, None] * lines[:, :2]


def z_from_disparity(disparity_px, Q: np.ndarray) -> np.ndarray:
    """Rectified triangulation: Z in metres from horizontal disparity and Q.

    ``Q`` is ``cv2.stereoRectify``'s 4×4; this is the same 3D point as
    ``triangulate_points`` once the pair has been row-aligned.
    """
    disparity = np.asarray(disparity_px, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return Q[2, 3] / (Q[3, 2] * disparity + Q[3, 3])
