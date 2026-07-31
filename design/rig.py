"""Multi-camera rig: a set of posed cameras plus the target they must cover."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from .camera import (Camera, K_from_calibration, K_from_fov, K_from_sensor,
                     R_from_azelroll, R_from_lookat)
from .target import Target

# Display colours for camera frustums and report curves, in assignment order.
# Purely cosmetic: nothing here touches image data or colour calibration.
DEFAULT_COLORS = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed", "#0891b2"]


@dataclass
class AnalysisSettings:
    """Knobs for the coverage and quality metrics."""

    voxel_pitch: float = 0.005
    surface_pitch: float = 0.004
    max_incidence_deg: float = 65.0
    margin_px: float = 0.0
    registration_depth: Optional[float] = None
    disparity_noise_px: float = 0.3
    interactive_voxel_pitch: float = 0.012

    @classmethod
    def from_dict(cls, cfg: Optional[Dict]) -> "AnalysisSettings":
        cfg = dict(cfg or {})
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in cfg.items() if k in known})


def _validate_rotation(R: np.ndarray, name: str) -> np.ndarray:
    """Reject anything that is not a proper rotation before it corrupts a study."""
    if R.shape != (3, 3):
        raise ValueError(f"{name}: rotation must be 3x3, got {R.shape}")
    off = float(np.abs(R @ R.T - np.eye(3)).max())
    if off > 1e-4:
        raise ValueError(f"{name}: rotation is not orthonormal (max |RR^T - I| = "
                         f"{off:.2e}). A transposed or scaled matrix is the usual "
                         f"cause.")
    if float(np.linalg.det(R)) < 0:
        raise ValueError(f"{name}: rotation has determinant < 0, i.e. it mirrors. "
                         f"Check the handedness of the source convention.")
    return R


def _build_pose(pose: Dict, name: str, target: Optional[Target]
                ) -> Tuple[np.ndarray, np.ndarray, str]:
    """Resolve one ``pose`` block to ``(R, t, source)`` of ``T_world_from_camera``.

    Four spellings, in precedence order. ``matrix`` and ``R`` carry *measured*
    extrinsics; ``look_at`` and the angle triple express *intended* mounting. The
    returned source label records which, since it changes how the resulting
    numbers should be read.
    """
    if "matrix" in pose:
        M = np.asarray(pose["matrix"], dtype=float)
        if M.shape != (4, 4):
            raise ValueError(f"{name}: pose.matrix must be 4x4, got {M.shape}")
        return _validate_rotation(M[:3, :3], name), M[:3, 3].copy(), "measured"

    position = np.asarray(pose.get("position", [0.0, 0.0, 0.0]), dtype=float)

    if "R" in pose:
        R = _validate_rotation(np.asarray(pose["R"], dtype=float), name)
        return R, position, "measured"

    if "look_at" in pose:
        aim = pose["look_at"]
        if isinstance(aim, str):
            if aim.strip().lower() != "target":
                raise ValueError(f"{name}: look_at must be coordinates or 'target', "
                                 f"got {aim!r}")
            if target is None:
                raise ValueError(f"{name}: look_at 'target' needs a target block")
            aim = target.center
        R = R_from_lookat(position, aim, float(pose.get("roll_deg", 0.0)))
        return R, position, "intended"

    return (R_from_azelroll(float(pose.get("azimuth_deg", 0.0)),
                            float(pose.get("elevation_deg", 0.0)),
                            float(pose.get("roll_deg", 0.0))),
            position, "intended")


def build_camera(cfg: Dict, base_dir: Path, index: int = 0,
                 target: Optional[Target] = None) -> Camera:
    """Instantiate a :class:`Camera` from one entry of the rig YAML.

    ``pose.look_at`` accepts either explicit coordinates or the string ``target``,
    which resolves to the target centre so that moving the target re-aims the
    camera instead of silently leaving it pointed at the old location.

    See :func:`_build_pose` for the ``matrix`` and ``R`` spellings used to load
    measured extrinsics.
    """
    cfg = dict(cfg)
    name = cfg.get("name", f"cam{index + 1}")
    intr = dict(cfg.get("intrinsics", {}))
    source = str(intr.pop("source", "fov")).lower()
    resolution = intr.pop("resolution", None)
    dist = np.zeros(0)

    if source == "calibration":
        path = Path(intr.pop("file"))
        if not path.is_absolute():
            path = base_dir / path
        K, (w, h), dist = K_from_calibration(path, resolution)
    elif source == "sensor":
        if resolution is None:
            raise ValueError(f"{name}: 'resolution' is required for sensor intrinsics")
        w, h = int(resolution[0]), int(resolution[1])
        K = K_from_sensor(w, h, float(intr["focal_length_mm"]),
                          pixel_pitch_um=intr.get("pixel_pitch_um"),
                          sensor_width_mm=intr.get("sensor_width_mm"))
    elif source == "fov":
        if resolution is None:
            raise ValueError(f"{name}: 'resolution' is required for fov intrinsics")
        w, h = int(resolution[0]), int(resolution[1])
        K = K_from_fov(w, h, float(intr["hfov_deg"]), intr.get("vfov_deg"))
    else:
        raise ValueError(f"{name}: unknown intrinsics source {source!r}")

    pose = dict(cfg.get("pose", {}))
    R, position, pose_source = _build_pose(pose, name, target)

    depth_range = cfg.get("range", cfg.get("depth_range", [0.1, 2.0]))

    return Camera(
        name=name, width=w, height=h, K=K, R=R, t=position,
        near=float(depth_range[0]), far=float(depth_range[1]),
        modality=str(cfg.get("modality", "rgb")),
        color=str(cfg.get("color", DEFAULT_COLORS[index % len(DEFAULT_COLORS)])),
        distortion=dist,
        meta=dict({k: v for k, v in cfg.items()
                   if k not in {"name", "intrinsics", "pose", "range",
                                "depth_range", "modality", "color"}},
                  pose_source=pose_source),
    )


@dataclass
class Rig:
    """A camera rig with its target volume and analysis settings."""

    cameras: List[Camera] = field(default_factory=list)
    target: Target = field(default_factory=Target)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    name: str = "rig"
    source_path: Optional[Path] = None

    # -- io ---------------------------------------------------------------- #
    @classmethod
    def from_yaml(cls, path) -> "Rig":
        path = Path(path)
        cfg = load_config(path)
        base_dir = _resolve_base_dir(cfg, path)
        target = Target.from_dict(cfg.get("target", {}))
        cams = [build_camera(c, base_dir, i, target)
                for i, c in enumerate(cfg.get("cameras", []))]
        if not cams:
            raise ValueError(f"{path}: no cameras defined")
        return cls(
            cameras=cams,
            target=target,
            analysis=AnalysisSettings.from_dict(cfg.get("analysis")),
            name=str(cfg.get("name", path.stem)),
            source_path=path,
        )

    def to_dict(self) -> Dict:
        """Serialise the current poses back into rig-YAML form."""
        cams = []
        for cam in self.cameras:
            az, el, roll = cam.azimuth_elevation_roll
            cams.append({
                "name": cam.name,
                "modality": cam.modality,
                "color": cam.color,
                "intrinsics": {
                    "source": "fov",
                    "resolution": [cam.width, cam.height],
                    "hfov_deg": round(cam.hfov_deg, 6),
                    "vfov_deg": round(cam.vfov_deg, 6),
                },
                "range": [round(cam.near, 6), round(cam.far, 6)],
                "pose": {
                    "position": [round(float(v), 6) for v in cam.t],
                    "azimuth_deg": round(az, 4),
                    "elevation_deg": round(el, 4),
                    "roll_deg": round(roll, 4),
                },
            })
        b = self.target
        target = {"type": b.kind, "center": [round(float(v), 6) for v in b.center]}
        if b.kind in {"box", "plane"}:
            target["size"] = [round(float(v), 6) for v in b.size]
        if b.kind in {"cylinder", "sphere"}:
            target["radius"] = round(b.radius, 6)
        if b.kind == "cylinder":
            target["height"] = round(b.height, 6)
        if b.kind == "plane":
            target["normal"] = [round(float(v), 6) for v in b.normal]
        return {"name": self.name, "target": target,
                "analysis": vars(self.analysis).copy(), "cameras": cams}

    def save_yaml(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path

    # -- access ------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.cameras)

    def __iter__(self):
        return iter(self.cameras)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.by_name(key)
        return self.cameras[key]

    def by_name(self, name: str) -> Camera:
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise KeyError(f"no camera named {name!r}; have {self.names}")

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.cameras]

    def of_modality(self, modality: str) -> List[Camera]:
        return [c for c in self.cameras if c.modality == modality]

    def aim_errors_deg(self) -> Dict[str, float]:
        """Angle between each optical axis and the direction to the target centre.

        A large value usually means the target moved but ``look_at`` did not.
        """
        out: Dict[str, float] = {}
        for cam in self.cameras:
            to_target = self.target.center - cam.t
            n = np.linalg.norm(to_target)
            if n < 1e-9:
                out[cam.name] = 0.0
                continue
            cos = float(np.clip(cam.optical_axis @ (to_target / n), -1.0, 1.0))
            out[cam.name] = float(np.degrees(np.arccos(cos)))
        return out

    def copy(self) -> "Rig":
        return Rig(cameras=[c.copy() for c in self.cameras],
                   target=copy.deepcopy(self.target),
                   analysis=copy.copy(self.analysis),
                   name=self.name, source_path=self.source_path)

    def scene_bounds(self, pad: float = 0.05) -> np.ndarray:
        """World bounds enclosing the target and all frustums."""
        pts = [self.target.bounds.reshape(3, 2).T.reshape(-1, 3)]
        for cam in self.cameras:
            pts.append(cam.frustum_vertices())
            pts.append(cam.t.reshape(1, 3))
        allp = np.vstack(pts)
        lo, hi = allp.min(axis=0) - pad, allp.max(axis=0) + pad
        return np.array([lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])

    def summary(self) -> str:
        lines = [f"rig '{self.name}' - {len(self)} cameras, "
                 f"target: {self.target.kind} at "
                 f"({self.target.center[0]:.3f}, {self.target.center[1]:.3f}, "
                 f"{self.target.center[2]:.3f}) m"]
        lines += ["  " + c.summary() for c in self.cameras]
        measured = [c.name for c in self.cameras
                    if c.meta.get("pose_source") == "measured"]
        if measured:
            lines.append(f"  poses: {', '.join(measured)} measured; "
                         + (", ".join(c.name for c in self.cameras
                                      if c.name not in measured)
                            + " still intended"
                            if len(measured) < len(self.cameras)
                            else "all cameras measured"))
        stray = self.aim_errors_deg()
        for name, err in sorted(stray.items()):
            if err <= 5.0:
                continue
            if self.by_name(name).meta.get("pose_source") == "measured":
                lines.append(f"  note: {name} is measured pointing {err:.1f} deg off "
                             f"the target centre - that is the as-built aim, not a "
                             f"config mistake")
            else:
                lines.append(f"  note: {name} misses the target centre by "
                             f"{err:.1f} deg - use `look_at: target` to aim it")
        vertical = [c.name for c in self.cameras if c.near_vertical]
        if vertical:
            verb = "looks" if len(vertical) == 1 else "look"
            lines.append(f"  note: {', '.join(vertical)} {verb} near-vertically, so "
                         "azimuth and roll are the same degree of freedom there "
                         "(gimbal lock); the azimuth reported above is the "
                         "canonical 0 and the framing is carried by roll")
        return "\n".join(lines)


# Nested maps that an overlay replaces wholesale rather than merging key by key.
# Half-merging these produces contradictory configs, e.g. a `pose` holding both a
# measured `R` and a stale `look_at`, or `intrinsics` naming both a calibration
# file and an explicit FOV.
_REPLACE_WHOLESALE = {"pose", "intrinsics"}


def _merge_cameras(base: List[Dict], over: List[Dict]) -> List[Dict]:
    """Merge camera lists by ``name``, keeping the base order."""
    merged = {c.get("name"): dict(c) for c in base}
    order = [c.get("name") for c in base]
    for cam in over:
        name = cam.get("name")
        if name is None:
            raise ValueError("every camera in an 'extends' overlay needs a 'name' "
                             "so it can be matched against the base rig")
        if name in merged:
            merged[name] = _merge_config(merged[name], cam)
        else:
            merged[name] = dict(cam)
            order.append(name)
    return [merged[n] for n in order]


def _merge_config(base: Dict, over: Dict) -> Dict:
    out = dict(base)
    for key, value in over.items():
        if key == "cameras" and isinstance(value, list):
            out[key] = _merge_cameras(base.get("cameras", []), value)
        elif (key not in _REPLACE_WHOLESALE and isinstance(value, dict)
                and isinstance(base.get(key), dict)):
            out[key] = _merge_config(base[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path, _seen: Optional[set] = None) -> Dict:
    """Read a rig YAML, resolving an ``extends:`` chain if there is one.

    An overlay lets a variant rig restate only what differs - typically measured
    poses - instead of duplicating the intrinsics, target and analysis settings,
    which would then drift apart. Cameras are matched by name; ``pose`` and
    ``intrinsics`` are replaced wholesale, other maps merge key by key.
    """
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"circular 'extends' chain reaching {path} again")
    seen.add(path)

    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg

    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    if not parent_path.exists():
        raise FileNotFoundError(f"{path}: extends '{parent}' -> {parent_path} "
                                f"does not exist")
    return _merge_config(load_config(parent_path, seen), cfg)


def _resolve_base_dir(cfg: Dict, path: Path) -> Path:
    """Directory that relative paths inside the rig YAML are resolved against."""
    base = cfg.get("base_dir")
    if base:
        b = Path(base)
        return b if b.is_absolute() else (path.parent / b).resolve()
    # Default: the project root, i.e. the parent of the `design` package.
    return Path(__file__).resolve().parent.parent
