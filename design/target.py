"""Measurement volumes and subject surfaces the rig has to cover."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Target:
    """The region of interest the cameras must observe.

    ``kind`` is one of ``box``, ``plane``, ``cylinder`` or ``sphere``. A target
    provides both a volume sampling (for frustum-intersection studies) and a
    surface sampling with normals (for view-angle and resolution studies).
    """

    kind: str = "box"
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    size: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.2]))
    radius: float = 0.1
    height: float = 0.2
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    exclude_normals: Optional[Sequence[Sequence[float]]] = None
    name: str = "subject"

    def __post_init__(self) -> None:
        self.kind = str(self.kind).lower()
        self.center = np.asarray(self.center, dtype=float).reshape(3)
        self.size = np.asarray(self.size, dtype=float).ravel()
        self.normal = np.asarray(self.normal, dtype=float).reshape(3)
        if self.kind not in {"box", "plane", "cylinder", "sphere"}:
            raise ValueError(f"unknown target kind {self.kind!r}")

    # -- extents ----------------------------------------------------------- #
    @property
    def bounds(self) -> np.ndarray:
        """(xmin, xmax, ymin, ymax, zmin, zmax)."""
        if self.kind == "box":
            half = self.size[:3] / 2.0
        elif self.kind == "plane":
            e1, e2 = self._plane_axes()
            half = (np.abs(e1) * self.size[0] + np.abs(e2) * self.size[1]) / 2.0
        elif self.kind == "cylinder":
            half = np.array([self.radius, self.radius, self.height / 2.0])
        else:
            half = np.full(3, self.radius)
        lo, hi = self.center - half, self.center + half
        return np.array([lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])

    @property
    def characteristic_size(self) -> float:
        b = self.bounds
        return float(max(b[1] - b[0], b[3] - b[2], b[5] - b[4]))

    def _plane_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.normal / np.linalg.norm(self.normal)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(n @ ref) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(ref, n)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        return e1, e2

    # -- sampling ---------------------------------------------------------- #
    def sample_volume(self, pitch: float) -> np.ndarray:
        """Regular grid of points inside the target volume. Shape (N, 3).

        For ``plane`` targets the "volume" degenerates to the surface samples.
        """
        if self.kind == "plane":
            return self.sample_surface(pitch)[0]

        b = self.bounds
        axes = [np.arange(b[2 * i] + pitch / 2.0, b[2 * i + 1], pitch)
                if b[2 * i + 1] - b[2 * i] > pitch
                else np.array([(b[2 * i] + b[2 * i + 1]) / 2.0])
                for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

        if self.kind == "cylinder":
            d = grid[:, :2] - self.center[:2]
            grid = grid[np.hypot(d[:, 0], d[:, 1]) <= self.radius]
        elif self.kind == "sphere":
            grid = grid[np.linalg.norm(grid - self.center, axis=1) <= self.radius]
        return grid

    def sample_surface(self, pitch: float) -> Tuple[np.ndarray, np.ndarray]:
        """Points and outward unit normals on the target surface.

        Faces listed in ``exclude_normals`` are dropped, which is how you stop a
        resting surface (``[0, 0, -1]`` for a box on a table) from counting as an
        uncoverable blind spot.
        """
        if self.kind == "plane":
            pts, nrm = self._plane_surface(pitch)
        elif self.kind == "box":
            pts, nrm = self._box_surface(pitch)
        elif self.kind == "cylinder":
            pts, nrm = self._cylinder_surface(pitch)
        else:
            pts, nrm = self._sphere_surface(pitch)
        return self._apply_exclusions(pts, nrm)

    def _apply_exclusions(self, pts: np.ndarray, nrm: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.exclude_normals:
            return pts, nrm
        keep = np.ones(len(pts), dtype=bool)
        for raw in self.exclude_normals:
            d = np.asarray(raw, dtype=float).reshape(3)
            d /= np.linalg.norm(d)
            keep &= (nrm @ d) < 0.99
        return pts[keep], nrm[keep]

    def _plane_surface(self, pitch: float) -> Tuple[np.ndarray, np.ndarray]:
        e1, e2 = self._plane_axes()
        w, h = float(self.size[0]), float(self.size[1])
        a = np.arange(-w / 2.0 + pitch / 2.0, w / 2.0, pitch)
        b = np.arange(-h / 2.0 + pitch / 2.0, h / 2.0, pitch)
        if a.size == 0:
            a = np.array([0.0])
        if b.size == 0:
            b = np.array([0.0])
        A, B = np.meshgrid(a, b, indexing="ij")
        pts = self.center + A.ravel()[:, None] * e1 + B.ravel()[:, None] * e2
        n = self.normal / np.linalg.norm(self.normal)
        return pts, np.tile(n, (len(pts), 1))

    def _box_surface(self, pitch: float) -> Tuple[np.ndarray, np.ndarray]:
        sx, sy, sz = self.size[:3]
        pts, nrm = [], []
        faces = [
            (np.array([1.0, 0, 0]), sx / 2.0, (sy, sz), (1, 2)),
            (np.array([0, 1.0, 0]), sy / 2.0, (sx, sz), (0, 2)),
            (np.array([0, 0, 1.0]), sz / 2.0, (sx, sy), (0, 1)),
        ]
        for axis, offset, (u_len, v_len), (iu, iv) in faces:
            u = np.arange(-u_len / 2.0 + pitch / 2.0, u_len / 2.0, pitch)
            v = np.arange(-v_len / 2.0 + pitch / 2.0, v_len / 2.0, pitch)
            if u.size == 0:
                u = np.array([0.0])
            if v.size == 0:
                v = np.array([0.0])
            U, V = np.meshgrid(u, v, indexing="ij")
            for sign in (+1.0, -1.0):
                p = np.zeros((U.size, 3))
                p[:, iu] = U.ravel()
                p[:, iv] = V.ravel()
                p += sign * offset * axis
                pts.append(p + self.center)
                nrm.append(np.tile(sign * axis, (len(p), 1)))
        return np.vstack(pts), np.vstack(nrm)

    def _cylinder_surface(self, pitch: float, with_caps: bool = True
                          ) -> Tuple[np.ndarray, np.ndarray]:
        n_theta = max(8, int(np.ceil(2 * np.pi * self.radius / pitch)))
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        z = np.arange(-self.height / 2.0 + pitch / 2.0, self.height / 2.0, pitch)
        if z.size == 0:
            z = np.array([0.0])
        T, Z = np.meshgrid(theta, z, indexing="ij")
        side = np.column_stack([
            self.center[0] + self.radius * np.cos(T.ravel()),
            self.center[1] + self.radius * np.sin(T.ravel()),
            self.center[2] + Z.ravel(),
        ])
        side_n = np.column_stack([np.cos(T.ravel()), np.sin(T.ravel()),
                                  np.zeros(T.size)])
        if not with_caps:
            return side, side_n

        r = np.arange(pitch / 2.0, self.radius, pitch)
        cap_pts, cap_n = [], []
        for sign in (+1.0, -1.0):
            for ri in r:
                m = max(6, int(np.ceil(2 * np.pi * ri / pitch)))
                th = np.linspace(0, 2 * np.pi, m, endpoint=False)
                p = np.column_stack([
                    self.center[0] + ri * np.cos(th),
                    self.center[1] + ri * np.sin(th),
                    np.full(m, self.center[2] + sign * self.height / 2.0),
                ])
                cap_pts.append(p)
                cap_n.append(np.tile([0.0, 0.0, sign], (m, 1)))
        return (np.vstack([side] + cap_pts), np.vstack([side_n] + cap_n))

    def _sphere_surface(self, pitch: float) -> Tuple[np.ndarray, np.ndarray]:
        n = max(64, int(4 * np.pi * self.radius ** 2 / pitch ** 2))
        # Fibonacci sphere: near-uniform coverage without pole clustering.
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        theta = np.pi * (1 + 5 ** 0.5) * i
        dirs = np.column_stack([np.cos(theta) * np.sin(phi),
                                np.sin(theta) * np.sin(phi),
                                np.cos(phi)])
        return self.center + self.radius * dirs, dirs

    # -- io ---------------------------------------------------------------- #
    @classmethod
    def from_dict(cls, cfg: Dict) -> "Target":
        cfg = dict(cfg or {})
        kind = cfg.pop("type", cfg.pop("kind", "box"))
        return cls(kind=kind, **cfg)

    def to_pyvista(self):
        import pyvista as pv

        if self.kind == "box":
            b = self.bounds
            return pv.Box(bounds=tuple(b))
        if self.kind == "plane":
            return pv.Plane(center=self.center, direction=self.normal,
                            i_size=float(self.size[0]), j_size=float(self.size[1]))
        if self.kind == "cylinder":
            return pv.Cylinder(center=self.center, direction=(0, 0, 1),
                               radius=self.radius, height=self.height,
                               resolution=64)
        return pv.Sphere(radius=self.radius, center=self.center,
                         theta_resolution=48, phi_resolution=48)
