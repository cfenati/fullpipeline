"""3D visualisation of camera frustums with PyVista.

Two entry points:

``plot_rig``   renders a static multi-view figure (good for the thesis / reports).
``run_viewer`` opens an interactive window with sliders for every degree of
               freedom, live coverage metrics and export shortcuts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv

from .analysis import volume_coverage
from .camera import Camera
from .rig import Rig

# Quad faces of a frustum given 4 near corners (0-3) then 4 far corners (4-7).
_FRUSTUM_FACES = np.hstack([
    [4, 0, 1, 2, 3],
    [4, 7, 6, 5, 4],
    [4, 0, 1, 5, 4],
    [4, 1, 2, 6, 5],
    [4, 2, 3, 7, 6],
    [4, 3, 0, 4, 7],
])

_PYRAMID_FACES = np.hstack([
    [3, 0, 1, 2], [3, 0, 2, 3],          # far quad split into two triangles
    [3, 4, 0, 1], [3, 4, 1, 2], [3, 4, 2, 3], [3, 4, 3, 0],
])


# --------------------------------------------------------------------------- #
# Geometry builders
# --------------------------------------------------------------------------- #
def frustum_solid(cam: Camera, near: Optional[float] = None,
                  far: Optional[float] = None) -> pv.PolyData:
    """Closed frustum between the near and far planes."""
    verts = cam.frustum_vertices(near, far)
    return pv.PolyData(verts, _FRUSTUM_FACES)


def frustum_pyramid(cam: Camera, far: Optional[float] = None) -> pv.PolyData:
    """Cone of vision from the pinhole to the far plane."""
    far_corners = cam.frustum_corners(cam.far if far is None else far)
    verts = np.vstack([far_corners, cam.t])
    return pv.PolyData(verts, _PYRAMID_FACES)


def frustum_edges(cam: Camera, near: Optional[float] = None,
                  far: Optional[float] = None) -> pv.PolyData:
    """Wireframe: the near and far rectangles plus the four rays from the pinhole."""
    near = cam.near if near is None else near
    far = cam.far if far is None else far
    n = cam.frustum_corners(near)
    f = cam.frustum_corners(far)
    pts = np.vstack([n, f, cam.t.reshape(1, 3)])
    lines = []
    for i in range(4):
        lines += [2, i, (i + 1) % 4]              # near rectangle
        lines += [2, 4 + i, 4 + (i + 1) % 4]      # far rectangle
        lines += [2, 8, 4 + i]                    # pinhole to far corner
    return pv.PolyData(pts, lines=np.hstack(lines))


def image_plane(cam: Camera, depth: float) -> pv.PolyData:
    """The imaged rectangle at a given depth, as a single quad."""
    c = cam.frustum_corners(depth)
    return pv.PolyData(c, np.hstack([[4, 0, 1, 2, 3]]))


def camera_body(cam: Camera, length: float = 0.05,
                width: float = 0.030) -> pv.PolyData:
    """A schematic housing behind the pinhole, oriented with the optical axis."""
    hw = width / 2.0
    local = np.array([
        [-hw, -hw, -length], [hw, -hw, -length], [hw, hw, -length], [-hw, hw, -length],
        [-hw, -hw, 0.0], [hw, -hw, 0.0], [hw, hw, 0.0], [-hw, hw, 0.0],
    ])
    pts = local @ cam.R.T + cam.t
    return pv.PolyData(pts, _FRUSTUM_FACES)


def camera_axes(cam: Camera, scale: float = 0.05) -> List[Tuple[pv.PolyData, str]]:
    """Short X/Y/Z arrows showing the camera frame (red/green/blue)."""
    out = []
    for i, color in enumerate(("red", "green", "blue")):
        arrow = pv.Arrow(start=cam.t, direction=cam.R[:, i], scale=scale,
                         tip_length=0.3, tip_radius=0.08, shaft_radius=0.025)
        out.append((arrow, color))
    return out


def optical_axis_line(cam: Camera) -> pv.PolyData:
    return pv.Line(cam.t, cam.t + cam.optical_axis * cam.far)


# --------------------------------------------------------------------------- #
# Scene assembly
# --------------------------------------------------------------------------- #
class RigScene:
    """Draws a rig into a PyVista plotter and keeps actors updatable."""

    def __init__(self, plotter: pv.Plotter, rig: Rig,
                 show_solid: bool = True, show_body: bool = True,
                 show_axes_triads: bool = False, show_ground: bool = True,
                 show_labels: bool = True, solid_opacity: float = 0.13,
                 style: str = "frustum", prefix: str = "") -> None:
        self.p = plotter
        self.rig = rig
        self.show_solid = show_solid
        self.show_body = show_body
        self.show_axes_triads = show_axes_triads
        self.show_labels = show_labels
        self.solid_opacity = solid_opacity
        self.style = style
        # Actor names are global to a plotter, so subplots need distinct prefixes.
        self.prefix = prefix
        self._coverage_visible = False
        self._hull_visible = False
        self.coverage_min_views = 1

        if show_ground:
            self._add_ground()
        self._add_target()
        for cam in rig.cameras:
            self.update_camera(cam)
        self._add_labels()

    def _name(self, key: str) -> str:
        return f"{self.prefix}{key}"

    # -- static parts ------------------------------------------------------ #
    def _add_ground(self) -> None:
        b = self.rig.scene_bounds(pad=0.10)
        size = max(b[1] - b[0], b[3] - b[2]) * 1.2
        z0 = min(0.0, b[4])
        grid = pv.Plane(center=(self.rig.target.center[0], self.rig.target.center[1], z0),
                        direction=(0, 0, 1), i_size=size, j_size=size,
                        i_resolution=int(size / 0.05) or 1,
                        j_resolution=int(size / 0.05) or 1)
        self.p.add_mesh(grid, style="wireframe", color="#cbd5e1", line_width=1,
                        opacity=0.55, name=self._name("ground"), pickable=False)

    def _add_target(self) -> None:
        # The target is a region of interest, so keep it nearly transparent and
        # rely on its edges: whatever is drawn inside it stays readable.
        mesh = self.rig.target.to_pyvista()
        self.p.add_mesh(mesh, color="#fbbf24", opacity=0.12,
                        name=self._name("target"), smooth_shading=True,
                        pickable=False)
        self.p.add_mesh(mesh.extract_feature_edges(), color="#92400e",
                        line_width=3, name=self._name("target_edges"),
                        pickable=False)

    def _add_labels(self) -> None:
        if not self.show_labels:
            return
        pts = np.array([c.t for c in self.rig.cameras])
        labels = [f"{c.name} ({c.hfov_deg:.0f}x{c.vfov_deg:.0f} deg)"
                  for c in self.rig.cameras]
        self.p.add_point_labels(pts, labels, name=self._name("cam_labels"),
                                font_size=11, text_color="black",
                                shape_color="white", shape_opacity=0.75,
                                point_size=1, always_visible=True, pickable=False)

    # -- per-camera -------------------------------------------------------- #
    def update_camera(self, cam: Camera) -> None:
        """Rebuild every actor belonging to one camera."""
        self.p.add_mesh(frustum_edges(cam), color=cam.color, line_width=2.5,
                        name=self._name(f"{cam.name}_edges"), pickable=False)

        if self.show_solid:
            solid = (frustum_pyramid(cam) if self.style == "pyramid"
                     else frustum_solid(cam))
            self.p.add_mesh(solid, color=cam.color, opacity=self.solid_opacity,
                            name=self._name(f"{cam.name}_solid"), show_edges=False,
                            pickable=False)
        self.p.add_mesh(image_plane(cam, cam.far), color=cam.color, opacity=0.25,
                        name=self._name(f"{cam.name}_farplane"), pickable=False)

        if self.show_body:
            body_len = max(0.02, self.rig.target.characteristic_size * 0.12)
            self.p.add_mesh(camera_body(cam, length=body_len, width=body_len * 0.6),
                            color="#1f2937", name=self._name(f"{cam.name}_body"),
                            pickable=False)

        if self.show_axes_triads:
            scale = max(0.03, self.rig.target.characteristic_size * 0.25)
            for i, (arrow, color) in enumerate(camera_axes(cam, scale=scale)):
                self.p.add_mesh(arrow, color=color,
                                name=self._name(f"{cam.name}_axis{i}"),
                                pickable=False)

    def refresh_all(self) -> None:
        for cam in self.rig.cameras:
            self.update_camera(cam)
        self._add_labels()
        if self._coverage_visible:
            self.update_coverage(True)
        if self._hull_visible:
            self.update_intersection(True)

    # -- coverage cloud ---------------------------------------------------- #
    def update_coverage(self, visible: bool, pitch: Optional[float] = None,
                        min_views: Optional[int] = None) -> None:
        """Show target voxels coloured by how many cameras see them."""
        self._coverage_visible = bool(visible)
        name = self._name("coverage")
        if not visible:
            self.p.remove_actor(name, reset_camera=False)
            if "views" in self.p.scalar_bars.keys():
                self.p.remove_scalar_bar("views")
            return

        k = self.coverage_min_views if min_views is None else min_views
        cov = volume_coverage(self.rig,
                              pitch=pitch or self.rig.analysis.interactive_voxel_pitch)
        seen = cov.n_views >= max(1, k)
        if not seen.any():
            self.p.remove_actor(name, reset_camera=False)
            return
        cloud = pv.PolyData(cov.points[seen])
        cloud["views"] = cov.n_views[seen].astype(float)
        self.p.add_mesh(cloud, scalars="views", name=name,
                        cmap="turbo", clim=[1, max(2, len(self.rig))],
                        point_size=10, render_points_as_spheres=True,
                        scalar_bar_args={"title": "views",
                                         "n_labels": len(self.rig) + 1,
                                         "vertical": True, "position_x": 0.92,
                                         "position_y": 0.30, "width": 0.05,
                                         "height": 0.40},
                        pickable=False)

    # -- all-camera intersection ------------------------------------------- #
    def update_intersection(self, visible: bool, pitch: Optional[float] = None,
                            min_views: Optional[int] = None) -> None:
        """Draw the volume every camera sees as a solid body.

        Frustums and the supported target shapes are all convex, so the convex
        hull of the qualifying voxels is the intersection volume itself. The hull
        passes through voxel centres, so it under-states the volume by about half
        a pitch all round; use the report's voxel count for the actual number.
        """
        self._hull_visible = bool(visible)
        name = self._name("intersection")
        if not visible:
            self.p.remove_actor(name, reset_camera=False)
            self.p.remove_actor(self._name("intersection_edges"), reset_camera=False)
            return

        k = len(self.rig) if min_views is None else min_views
        cov = volume_coverage(self.rig,
                              pitch=pitch or self.rig.analysis.interactive_voxel_pitch)
        pts = cov.points[cov.n_views >= k]
        if len(pts) < 4:
            self.p.remove_actor(name, reset_camera=False)
            self.p.remove_actor(self._name("intersection_edges"), reset_camera=False)
            return
        try:
            hull = pv.PolyData(pts).delaunay_3d().extract_surface()
        except Exception:
            return
        self.p.add_mesh(hull, color="#a855f7", opacity=0.45, name=name,
                        smooth_shading=True, pickable=False)
        self.p.add_mesh(hull.extract_feature_edges(), color="#6b21a8",
                        line_width=2, name=self._name("intersection_edges"),
                        pickable=False)


def _add_world_axes(plotter: pv.Plotter, rig: Rig, scale: Optional[float] = None,
                    prefix: str = "") -> None:
    scale = scale or max(0.05, rig.target.characteristic_size * 0.9)
    for direction, color, label in (((1, 0, 0), "red", "X"),
                                    ((0, 1, 0), "green", "Y"),
                                    ((0, 0, 1), "blue", "Z")):
        plotter.add_mesh(pv.Arrow(start=(0, 0, 0), direction=direction, scale=scale,
                                  tip_length=0.15, tip_radius=0.035,
                                  shaft_radius=0.010),
                         color=color, name=f"{prefix}world_{label}", pickable=False)


# --------------------------------------------------------------------------- #
# Static figure
# --------------------------------------------------------------------------- #
_VIEWS = {
    "iso": None,
    "top": "xy",
    "front": "xz",
    "side": "yz",
}


def plot_rig(rig: Rig, screenshot: Optional[str] = None, off_screen: bool = False,
             show_coverage: bool = False, show_intersection: bool = False,
             coverage_min_views: int = 1, multiview: bool = False,
             style: str = "frustum", window_size: Sequence[int] = (1500, 950),
             title: Optional[str] = None):
    """Render the rig. With ``multiview`` a 2x2 iso/top/front/side panel is made."""
    pv.set_plot_theme("document")

    if not multiview:
        p = pv.Plotter(off_screen=off_screen, window_size=list(window_size))
        scene = RigScene(p, rig, style=style)
        scene.coverage_min_views = coverage_min_views
        _add_world_axes(p, rig)
        if show_coverage:
            scene.update_coverage(True)
        if show_intersection:
            scene.update_intersection(True)
        p.add_text(title or f"{rig.name} - {len(rig)} cameras", font_size=11)
        p.add_axes()
        p.camera_position = _iso_position(rig)
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
            p.show(screenshot=screenshot, auto_close=True)
        else:
            p.show()
        return p

    p = pv.Plotter(shape=(2, 2), off_screen=off_screen, window_size=list(window_size))
    for idx, (name, view) in enumerate(_VIEWS.items()):
        p.subplot(idx // 2, idx % 2)
        scene = RigScene(p, rig, show_ground=(name != "top"), style=style,
                         show_labels=(name == "iso"), prefix=f"{name}_")
        scene.coverage_min_views = coverage_min_views
        _add_world_axes(p, rig, prefix=f"{name}_")
        if show_coverage:
            scene.update_coverage(True)
        if show_intersection:
            scene.update_intersection(True)
        p.add_text(name, font_size=10)
        if view is None:
            p.camera_position = _iso_position(rig)
        else:
            getattr(p, f"view_{view}")()
            p.reset_camera()
    if screenshot:
        Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
        p.show(screenshot=screenshot, auto_close=True)
    else:
        p.show()
    return p


def _iso_position(rig: Rig):
    b = rig.scene_bounds(pad=0.05)
    center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
    eye = center + np.array([1.1, -1.5, 0.95]) * span
    return [tuple(eye), tuple(center), (0, 0, 1)]


# --------------------------------------------------------------------------- #
# Interactive viewer (Qt control panel + native PyVista plotter)
# --------------------------------------------------------------------------- #
_PANEL_CSS = """
QWidget#RigPanel {
    background: #f8fafc;
    border-left: 1px solid #e2e8f0;
}
QLabel#Section {
    color: #0f172a;
    font-weight: 600;
    font-size: 12px;
    padding-top: 4px;
}
QLabel#Hint {
    color: #64748b;
    font-size: 10px;
}
QLabel#Param {
    color: #334155;
    font-size: 11px;
    min-width: 72px;
}
QLabel#Metrics {
    color: #1e293b;
    font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace;
    font-size: 11px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px;
}
QComboBox, QDoubleSpinBox {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 2px 6px;
    min-height: 22px;
    font-size: 11px;
}
QDoubleSpinBox {
    max-width: 96px;
}
QSlider::groove:horizontal {
    height: 2px;
    background: #cbd5e1;
    border-radius: 1px;
}
QSlider::handle:horizontal {
    width: 12px;
    height: 12px;
    margin: -5px 0;
    background: #2563eb;
    border-radius: 6px;
}
QSlider::sub-page:horizontal {
    background: #93c5fd;
    border-radius: 1px;
}
QCheckBox {
    color: #334155;
    font-size: 11px;
    spacing: 6px;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 11px;
    color: #334155;
}
QPushButton:hover {
    background: #f1f5f9;
    border-color: #94a3b8;
}
"""


class _ParamRow:
    """Label + thin slider + spinbox for one degree of freedom."""

    def __init__(self, parent, key: str, label: str, lo: float, hi: float,
                 value: float, decimals: int, step: float, unit: str,
                 on_change) -> None:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QDoubleSpinBox,
                                    QSlider, QWidget)

        self.key = key
        self.lo = lo
        self.hi = hi
        self._scale = 10 ** decimals
        self._updating = False
        self.on_change = on_change

        self.widget = QWidget(parent)
        row = QHBoxLayout(self.widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        name = QLabel(label, self.widget)
        name.setObjectName("Param")
        row.addWidget(name)

        self.slider = QSlider(Qt.Horizontal, self.widget)
        self.slider.setRange(int(round(lo * self._scale)),
                             int(round(hi * self._scale)))
        self.slider.setValue(int(round(value * self._scale)))
        self.slider.setSingleStep(max(1, int(round(step * self._scale))))
        self.slider.setFixedHeight(18)
        row.addWidget(self.slider, stretch=1)

        self.spin = QDoubleSpinBox(self.widget)
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(f" {unit}" if unit else "")
        self.spin.setValue(value)
        self.spin.setKeyboardTracking(False)
        row.addWidget(self.spin)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, ticks: int) -> None:
        if self._updating:
            return
        value = ticks / self._scale
        self._updating = True
        self.spin.setValue(value)
        self._updating = False
        self.on_change(self.key, value)

    def _from_spin(self, value: float) -> None:
        if self._updating:
            return
        self._updating = True
        self.slider.setValue(int(round(value * self._scale)))
        self._updating = False
        self.on_change(self.key, value)

    def set_value(self, value: float) -> None:
        value = float(np.clip(value, self.lo, self.hi))
        self._updating = True
        self.spin.setValue(value)
        self.slider.setValue(int(round(value * self._scale)))
        self._updating = False

    def set_range(self, lo: float, hi: float) -> None:
        self.lo, self.hi = lo, hi
        self._updating = True
        self.spin.setRange(lo, hi)
        self.slider.setRange(int(round(lo * self._scale)),
                             int(round(hi * self._scale)))
        self._updating = False


class RigViewer:
    """Interactive rig editor: one Qt window with embedded 3D view + controls."""

    def __init__(self, rig: Rig, style: str = "frustum",
                 window_size: Sequence[int] = (1600, 1000),
                 export_dir: str = "design/out") -> None:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox,
                                    QHBoxLayout, QLabel, QMainWindow,
                                    QPushButton, QScrollArea, QSizePolicy,
                                    QVBoxLayout, QWidget)
        from pyvista.plotting.render_window_interactor import RenderWindowInteractor
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

        self.rig = rig
        self.export_dir = Path(export_dir)
        self.active = 0
        self._updating = False
        self._rows: Dict[str, _ParamRow] = {}
        self._view_rows: Dict[str, _ParamRow] = {}

        self._app = QApplication.instance() or QApplication([])
        self._window = QMainWindow()
        self._window.setWindowTitle(f"Rig viewer — {rig.name}")
        self._window.resize(int(window_size[0]), int(window_size[1]))
        self._window.closeEvent = self._on_window_close  # type: ignore[method-assign]

        root = QWidget()
        self._window.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # -- embedded 3D view (Qt owns the event loop → mouse orbit works) -- #
        view_host = QWidget()
        view_layout = QVBoxLayout(view_host)
        view_layout.setContentsMargins(0, 0, 0, 0)
        self._vtk = QVTKRenderWindowInteractor(view_host)
        view_layout.addWidget(self._vtk)
        layout.addWidget(view_host, stretch=1)

        pv.set_plot_theme("document")
        pv.global_theme.font.size = 10
        # Build the scene off-screen, then rebind to the Qt VTK widget so we do
        # not open a second native window that fights Qt for mouse events.
        self.p = pv.Plotter(off_screen=True)
        self.p.set_background("white")
        self.scene = RigScene(self.p, rig, style=style)
        _add_world_axes(self.p, rig)

        self._bind_plotter_to_vtk()
        self.p.enable_trackball_style()
        self.p.add_axes()

        self._view_center = self._scene_center()
        self._view_distance = self._default_view_distance()
        self._view_az, self._view_el = self._iso_orbit_angles()
        self._apply_orbit()

        # -- side panel ---------------------------------------------------- #
        panel = QWidget()
        panel.setObjectName("RigPanel")
        panel.setFixedWidth(340)
        panel.setStyleSheet(_PANEL_CSS)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(14, 14, 14, 14)
        panel_layout.setSpacing(8)

        view_title = QLabel("Scene view")
        view_title.setObjectName("Section")
        panel_layout.addWidget(view_title)

        view_box = QWidget()
        view_box_layout = QVBoxLayout(view_box)
        view_box_layout.setContentsMargins(0, 0, 0, 0)
        view_box_layout.setSpacing(6)
        for key, label, (lo, hi), value, decimals, step, unit in [
            ("orbit_az", "orbit az", (-180.0, 180.0), self._view_az, 1, 1.0, "°"),
            ("orbit_el", "orbit el", (-89.0, 89.0), self._view_el, 1, 1.0, "°"),
            ("orbit_dist", "distance", (0.05, max(2.0, self._view_distance * 4)),
             self._view_distance, 3, 0.01, "m"),
        ]:
            row = _ParamRow(view_box, key, label, lo, hi, float(value),
                            decimals, step, unit, self._on_view_param)
            self._view_rows[key] = row
            view_box_layout.addWidget(row.widget)
        panel_layout.addWidget(view_box)

        presets = QHBoxLayout()
        for name, handler in (
            ("Iso", self._reset_view),
            ("Top", lambda: self._set_preset("top")),
            ("Front", lambda: self._set_preset("front")),
            ("Side", lambda: self._set_preset("side")),
        ):
            btn = QPushButton(name)
            btn.clicked.connect(handler)
            presets.addWidget(btn)
        panel_layout.addLayout(presets)

        title = QLabel("Camera pose")
        title.setObjectName("Section")
        panel_layout.addWidget(title)

        self._camera_box = QComboBox()
        self._camera_box.addItems(self.rig.names)
        self._camera_box.currentIndexChanged.connect(self._on_select)
        panel_layout.addWidget(self._camera_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        pose_host = QWidget()
        pose_layout = QVBoxLayout(pose_host)
        pose_layout.setContentsMargins(0, 4, 0, 0)
        pose_layout.setSpacing(6)

        r = self._position_range()
        cam = self.rig[self.active]
        az, el, roll = cam.azimuth_elevation_roll
        specs = [
            ("x", "x", (-r, r), cam.t[0], 4, 0.001, "m"),
            ("y", "y", (-r, r), cam.t[1], 4, 0.001, "m"),
            ("z", "z", (-r, r), cam.t[2], 4, 0.001, "m"),
            ("azimuth", "azimuth", (-180.0, 180.0), az, 2, 0.5, "°"),
            ("elevation", "elevation", (-90.0, 90.0), el, 2, 0.5, "°"),
            ("roll", "roll", (-180.0, 180.0), roll, 2, 0.5, "°"),
            ("working dist", "far", (0.05, max(3.0, cam.far * 2)), cam.far,
             3, 0.01, "m"),
        ]
        for key, label, (lo, hi), value, decimals, step, unit in specs:
            row = _ParamRow(pose_host, key, label, lo, hi, float(value),
                            decimals, step, unit, self._on_param)
            self._rows[key] = row
            pose_layout.addWidget(row.widget)
        pose_layout.addStretch(1)
        scroll.setWidget(pose_host)
        panel_layout.addWidget(scroll, stretch=1)

        overlays = QLabel("Overlays")
        overlays.setObjectName("Section")
        panel_layout.addWidget(overlays)

        self._cov_box = QCheckBox("Coverage cloud  [c]")
        self._cov_box.toggled.connect(self._on_coverage_toggled)
        panel_layout.addWidget(self._cov_box)
        self._hull_box = QCheckBox("Common volume  [i]")
        self._hull_box.toggled.connect(self._on_intersection_toggled)
        panel_layout.addWidget(self._hull_box)

        actions = QHBoxLayout()
        btn_report = QPushButton("Report [r]")
        btn_report.clicked.connect(self._print_report)
        btn_export = QPushButton("Export [e]")
        btn_export.clicked.connect(self._export)
        btn_reset = QPushButton("Reset view [t]")
        btn_reset.clicked.connect(self._reset_view)
        actions.addWidget(btn_report)
        actions.addWidget(btn_export)
        actions.addWidget(btn_reset)
        panel_layout.addLayout(actions)

        metrics_title = QLabel("Live metrics")
        metrics_title.setObjectName("Section")
        panel_layout.addWidget(metrics_title)

        self._metrics = QLabel()
        self._metrics.setObjectName("Metrics")
        self._metrics.setWordWrap(True)
        self._metrics.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._metrics.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        panel_layout.addWidget(self._metrics)

        hint = QLabel("Left-drag to orbit · scroll / right-drag to zoom · "
                      "shift+left-drag to pan.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        panel_layout.addWidget(hint)

        layout.addWidget(panel)

        self._bind_keys()
        self._update_metrics()
        # Sync orbit readouts after a mouse interaction finishes.
        self.p.iren.add_observer("EndInteractionEvent", self._on_end_interaction)

        self._vtk.Initialize()
        self._window.show()
        self.p.render()

    def _bind_plotter_to_vtk(self) -> None:
        """Move the plotter's renderers onto the Qt VTK widget."""
        from pyvista.plotting.render_window_interactor import RenderWindowInteractor

        old_rw = self.p.ren_win
        new_rw = self._vtk.GetRenderWindow()
        renderers = []
        collection = old_rw.GetRenderers()
        collection.InitTraversal()
        for _ in range(collection.GetNumberOfItems()):
            renderer = collection.GetNextItem()
            if renderer is not None:
                renderers.append(renderer)
        for renderer in renderers:
            old_rw.RemoveRenderer(renderer)
            new_rw.AddRenderer(renderer)
        try:
            old_rw.Finalize()
        except Exception:
            pass
        self.p.ren_win = new_rw
        self.p.off_screen = False
        self.p.iren = RenderWindowInteractor(
            self.p, light_follow_camera=False, interactor=self._vtk
        )
        self.p.iren.set_render_window(new_rw)
        self.p.reset_key_events()

    def _on_window_close(self, event) -> None:
        try:
            self._vtk.Finalize()
        except Exception:
            pass
        try:
            self.p.close()
        except Exception:
            pass
        event.accept()

    def _on_end_interaction(self, *_args) -> None:
        self._sync_orbit_from_camera()

    def _sync_orbit_from_camera(self) -> None:
        if self._updating or not self._view_rows:
            return
        try:
            pos = np.asarray(self.p.camera.position, dtype=float)
            focus = np.asarray(self.p.camera.focal_point, dtype=float)
        except Exception:
            return
        offset = pos - focus
        dist = float(np.linalg.norm(offset))
        if dist < 1e-9:
            return
        direction = offset / dist
        az = float(np.degrees(np.arctan2(direction[1], direction[0])))
        el = float(np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0))))
        self._view_center = focus
        self._view_az = az
        self._view_el = el
        self._view_distance = dist
        self._sync_view_controls()

    def _position_range(self) -> float:
        b = self.rig.scene_bounds(pad=0.2)
        return float(max(abs(b[0]), abs(b[1]), abs(b[2]), abs(b[3]),
                         abs(b[4]), abs(b[5]), 0.5))

    def _scene_center(self) -> np.ndarray:
        b = self.rig.scene_bounds(pad=0.05)
        return np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2,
                         (b[4] + b[5]) / 2], dtype=float)

    def _default_view_distance(self) -> float:
        b = self.rig.scene_bounds(pad=0.05)
        span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4], 0.1)
        return float(np.linalg.norm(np.array([1.1, -1.5, 0.95]) * span))

    @staticmethod
    def _iso_orbit_angles() -> Tuple[float, float]:
        direction = np.array([1.1, -1.5, 0.95], dtype=float)
        direction /= np.linalg.norm(direction)
        az = float(np.degrees(np.arctan2(direction[1], direction[0])))
        el = float(np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0))))
        return az, el

    def _apply_orbit(self) -> None:
        az = np.radians(self._view_az)
        el = np.radians(self._view_el)
        offset = self._view_distance * np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ])
        eye = self._view_center + offset
        self.p.camera_position = [tuple(eye), tuple(self._view_center), (0, 0, 1)]

    def _sync_view_controls(self) -> None:
        if not self._view_rows:
            return
        self._updating = True
        try:
            self._view_rows["orbit_az"].set_value(self._view_az)
            self._view_rows["orbit_el"].set_value(self._view_el)
            self._view_rows["orbit_dist"].set_value(self._view_distance)
        finally:
            self._updating = False

    def _on_view_param(self, key: str, value: float) -> None:
        if self._updating:
            return
        if key == "orbit_az":
            self._view_az = float(value)
        elif key == "orbit_el":
            self._view_el = float(value)
        elif key == "orbit_dist":
            self._view_distance = max(float(value), 0.05)
        self._apply_orbit()
        self.p.render()

    def _set_preset(self, name: str) -> None:
        self._view_center = self._scene_center()
        self._view_distance = self._default_view_distance()
        if name == "top":
            self._view_az, self._view_el = 90.0, 89.0
        elif name == "front":
            self._view_az, self._view_el = -90.0, 0.0
        elif name == "side":
            self._view_az, self._view_el = 0.0, 0.0
        else:
            self._view_az, self._view_el = self._iso_orbit_angles()
        self._sync_view_controls()
        self._apply_orbit()
        self.p.render()

    def _on_param(self, key: str, value: float) -> None:
        if self._updating:
            return
        cam = self.rig[self.active]
        if key in ("x", "y", "z"):
            t = cam.t.copy()
            t["xyz".index(key)] = float(value)
            cam.set_pose(position=t)
        elif key == "azimuth":
            cam.set_pose(azimuth_deg=float(value))
        elif key == "elevation":
            cam.set_pose(elevation_deg=float(value))
        elif key == "roll":
            cam.set_pose(roll_deg=float(value))
        elif key == "working dist":
            cam.far = max(float(value), cam.near + 1e-3)
        self.scene.update_camera(cam)
        self.scene._add_labels()
        if self.scene._coverage_visible:
            self.scene.update_coverage(True)
        if self.scene._hull_visible:
            self.scene.update_intersection(True)
        self._update_metrics()
        self.p.render()

    def _on_select(self, idx: int) -> None:
        if idx < 0 or idx == self.active:
            return
        self.active = idx
        self._sync_controls()
        self._update_metrics()

    def _sync_controls(self) -> None:
        cam = self.rig[self.active]
        az, el, roll = cam.azimuth_elevation_roll
        values = {"x": cam.t[0], "y": cam.t[1], "z": cam.t[2],
                  "azimuth": az, "elevation": el, "roll": roll,
                  "working dist": cam.far}
        far_hi = max(3.0, cam.far * 2)
        self._rows["working dist"].set_range(0.05, far_hi)
        self._updating = True
        try:
            for key, value in values.items():
                self._rows[key].set_value(float(value))
        finally:
            self._updating = False

    def _update_metrics(self) -> None:
        cam = self.rig[self.active]
        az, el, roll = cam.azimuth_elevation_roll
        cov = volume_coverage(self.rig,
                              pitch=self.rig.analysis.interactive_voxel_pitch)
        dist = float(np.linalg.norm(self.rig.target.center - cam.t))
        gx, _ = cam.gsd(dist)

        lines = [
            f"<b>{cam.name}</b>  ·  {cam.modality}  ·  {cam.width}×{cam.height}",
            f"FOV {cam.hfov_deg:.1f}×{cam.vfov_deg:.1f}°"
            f"&nbsp;&nbsp;GSD {gx * 1000:.3f} mm/px",
            f"pos ({cam.t[0]:+.3f}, {cam.t[1]:+.3f}, {cam.t[2]:+.3f}) m",
            f"az {az:+.1f}°&nbsp;&nbsp;el {el:+.1f}°&nbsp;&nbsp;roll {roll:+.1f}°",
            "",
            "<b>Target volume coverage</b>",
        ]
        for k in range(1, len(self.rig) + 1):
            frac = cov.fraction_seen_by_at_least(k) * 100
            lines.append(f"  ≥{k} cam&nbsp;&nbsp;{frac:5.1f}%")

        pairs = list(cov.pairwise_overlap().items())
        if pairs:
            lines.append("")
            lines.append("<b>Pair overlap (IoU)</b>")
            for (a, b), m in pairs:
                lines.append(f"  {a} ∩ {b}&nbsp;&nbsp;{m['iou'] * 100:.1f}%")

        self._metrics.setText("<br>".join(lines))

    def _bind_keys(self) -> None:
        self.p.add_key_event("r", self._print_report)
        self.p.add_key_event("e", self._export)
        self.p.add_key_event("c", self._toggle_coverage)
        self.p.add_key_event("i", self._toggle_intersection)
        self.p.add_key_event("t", self._reset_view)

    def _print_report(self) -> None:
        from .analysis import evaluate
        print("\n" + evaluate(self.rig).text(), flush=True)

    def _on_coverage_toggled(self, checked: bool) -> None:
        self.scene.update_coverage(bool(checked))
        self.p.render()

    def _on_intersection_toggled(self, checked: bool) -> None:
        self.scene.update_intersection(bool(checked))
        self.p.render()

    def _toggle_coverage(self) -> None:
        self._cov_box.toggle()

    def _toggle_intersection(self) -> None:
        self._hull_box.toggle()

    def _reset_view(self) -> None:
        self._set_preset("iso")

    def _export(self) -> None:
        from .export import export_all
        paths = export_all(self.rig, self.export_dir)
        print("\nexported:")
        for p in paths:
            print("  " + str(p))
        print(flush=True)

    def show(self):
        return self._app.exec_()


def run_viewer(rig: Rig, **kwargs) -> RigViewer:
    viewer = RigViewer(rig, **kwargs)
    viewer.show()
    return viewer
