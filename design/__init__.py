"""Multi-camera system design: frustum modelling, coverage analysis, 3D viewer.

Typical use::

    from design import Rig, evaluate, run_viewer

    rig = Rig.from_yaml("design/config/rig.yaml")
    print(evaluate(rig).text())
    run_viewer(rig)
"""

from .analysis import (CoverageResult, RigReport, evaluate, overlap_vs_depth,
                       registration_error, registration_table, resolution_stats,
                       stereo_pairs, surface_coverage, triangulation_angles,
                       visibility_matrix, volume_coverage)
from .camera import (Camera, K_from_calibration, K_from_fov, K_from_sensor,
                     R_from_azelroll, R_from_lookat, azelroll_from_R)
from .export import export_all, export_frustum_meshes, export_poses_csv, export_poses_json
from .rig import AnalysisSettings, Rig, build_camera
from .target import Target

__all__ = [
    "AnalysisSettings", "Camera", "CoverageResult", "Rig", "RigReport", "Target",
    "K_from_calibration", "K_from_fov", "K_from_sensor",
    "R_from_azelroll", "R_from_lookat", "azelroll_from_R",
    "build_camera", "evaluate", "export_all", "export_frustum_meshes",
    "export_poses_csv", "export_poses_json", "overlap_vs_depth",
    "registration_error", "registration_table", "resolution_stats",
    "stereo_pairs", "surface_coverage", "triangulation_angles",
    "visibility_matrix", "volume_coverage",
]


def run_viewer(rig, **kwargs):
    """Open the interactive viewer (imports PyVista lazily)."""
    from .visualize import run_viewer as _run
    return _run(rig, **kwargs)


def plot_rig(rig, **kwargs):
    """Render a static figure (imports PyVista lazily)."""
    from .visualize import plot_rig as _plot
    return _plot(rig, **kwargs)
