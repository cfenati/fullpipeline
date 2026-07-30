"""Export optimised camera poses for CAD, downstream calibration and reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import numpy as np

from .camera import Camera
from .rig import Rig


def quaternion_from_R(R: np.ndarray) -> np.ndarray:
    """Unit quaternion ``(w, x, y, z)`` from a rotation matrix."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = np.zeros(3)
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = q
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def euler_zyx_deg(R: np.ndarray) -> np.ndarray:
    """Intrinsic Z-Y-X (yaw, pitch, roll) Euler angles in degrees.

    This is the convention most CAD packages expect for a rotated placement.
    """
    sy = -R[2, 0]
    if abs(sy) < 1.0 - 1e-9:
        pitch = np.arcsin(sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        pitch = np.pi / 2 * np.sign(sy)
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    return np.degrees([yaw, pitch, roll])


def camera_record(cam: Camera) -> dict:
    az, el, roll = cam.azimuth_elevation_roll
    yaw, pitch, roll_zyx = euler_zyx_deg(cam.R)
    return {
        "name": cam.name,
        "modality": cam.modality,
        "resolution": [cam.width, cam.height],
        "megapixels": round(cam.megapixels, 3),
        "K": cam.K.tolist(),
        "fx_px": cam.fx, "fy_px": cam.fy, "cx_px": cam.cx, "cy_px": cam.cy,
        "hfov_deg": cam.hfov_deg, "vfov_deg": cam.vfov_deg, "dfov_deg": cam.dfov_deg,
        "depth_range_m": [cam.near, cam.far],
        "position_m": cam.t.tolist(),
        "position_mm": (cam.t * 1000.0).tolist(),
        "R_world_from_camera": cam.R.tolist(),
        "T_world_from_camera": cam.T.tolist(),
        "quaternion_wxyz": quaternion_from_R(cam.R).tolist(),
        "mount_angles_deg": {"azimuth": az, "elevation": el, "roll": roll},
        "euler_zyx_deg": {"yaw": yaw, "pitch": pitch, "roll": roll_zyx},
        "optical_axis": cam.optical_axis.tolist(),
        "footprint_at_far_m": list(cam.footprint(cam.far)),
        "gsd_at_far_mm": [v * 1000.0 for v in cam.gsd(cam.far)],
        "distortion": cam.distortion.tolist(),
    }


def export_poses_json(rig: Rig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rig": rig.name,
        "units": {"length": "m", "angle": "deg"},
        "conventions": {
            "world": "right-handed, +Z up",
            "camera": "OpenCV: +X image right, +Y image down, +Z optical axis",
            "pose": "T_world_from_camera; translation is the camera centre in world",
        },
        "target": {
            "type": rig.target.kind,
            "center_m": rig.target.center.tolist(),
            "bounds_m": rig.target.bounds.tolist(),
        },
        "cameras": [camera_record(c) for c in rig.cameras],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


_CSV_FIELDS = ["name", "modality", "x_mm", "y_mm", "z_mm", "azimuth_deg",
               "elevation_deg", "roll_deg", "yaw_zyx_deg", "pitch_zyx_deg",
               "roll_zyx_deg", "qw", "qx", "qy", "qz", "hfov_deg", "vfov_deg",
               "near_m", "far_m", "axis_x", "axis_y", "axis_z"]


def export_poses_csv(rig: Rig, path) -> Path:
    """Flat table in millimetres and degrees, ready to paste into a CAD sketch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for cam in rig.cameras:
            az, el, roll = cam.azimuth_elevation_roll
            yaw, pitch, roll_zyx = euler_zyx_deg(cam.R)
            q = quaternion_from_R(cam.R)
            ax = cam.optical_axis
            writer.writerow({
                "name": cam.name, "modality": cam.modality,
                "x_mm": round(cam.t[0] * 1000, 3),
                "y_mm": round(cam.t[1] * 1000, 3),
                "z_mm": round(cam.t[2] * 1000, 3),
                "azimuth_deg": round(az, 3), "elevation_deg": round(el, 3),
                "roll_deg": round(roll, 3),
                "yaw_zyx_deg": round(yaw, 3), "pitch_zyx_deg": round(pitch, 3),
                "roll_zyx_deg": round(roll_zyx, 3),
                "qw": round(q[0], 6), "qx": round(q[1], 6),
                "qy": round(q[2], 6), "qz": round(q[3], 6),
                "hfov_deg": round(cam.hfov_deg, 3),
                "vfov_deg": round(cam.vfov_deg, 3),
                "near_m": round(cam.near, 4), "far_m": round(cam.far, 4),
                "axis_x": round(ax[0], 6), "axis_y": round(ax[1], 6),
                "axis_z": round(ax[2], 6),
            })
    return path


def export_frustum_meshes(rig: Rig, directory, scale_to_mm: bool = True
                          ) -> List[Path]:
    """One STL per frustum plus a merged STL, for import as CAD reference bodies."""
    from .visualize import frustum_solid

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scale = 1000.0 if scale_to_mm else 1.0
    written: List[Path] = []
    meshes = []

    for cam in rig.cameras:
        mesh = frustum_solid(cam).triangulate()
        mesh.points = mesh.points * scale
        out = directory / f"frustum_{cam.name}.stl"
        mesh.save(out)
        written.append(out)
        meshes.append(mesh)

    if meshes:
        merged = meshes[0].copy()
        for m in meshes[1:]:
            merged = merged.merge(m)
        out = directory / "frustums_all.stl"
        merged.save(out)
        written.append(out)
    return written


def export_all(rig: Rig, directory, stl: bool = True) -> List[Path]:
    """Write poses (JSON + CSV), the reloadable rig YAML and frustum STLs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [
        export_poses_json(rig, directory / "camera_poses.json"),
        export_poses_csv(rig, directory / "camera_poses.csv"),
        rig.save_yaml(directory / "rig_current.yaml"),
    ]
    if stl:
        paths += export_frustum_meshes(rig, directory / "cad")
    return paths
