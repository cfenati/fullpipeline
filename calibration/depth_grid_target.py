"""Depth-grid target: an independent relative-depth ruler for this rig.

``check_line_accuracy.py``'s line ladder validates lateral (X-Y) triangulation
accuracy -- but a flat, fronto-parallel plate has ~zero depth variation across
it by construction, so it can never touch the Z-axis (depth). This target does
the depth-axis equivalent: a flat baseplate carries a grid of blocks, each a
different, independently-measured height, plus a handful of flush (zero-
standoff) corner fiducials used only to fit the plane that stands in for the
baseplate's true orientation -- never assumed to be perpendicular to either
camera.

A block's ground-truth depth is its height **relative to that fitted plane**,
not any assumed standoff from the camera -- this is deliberate: the camera's
internal optical-centre offset is not known, so only relative depth
differences (delta-Z between blocks) are validated here, mirroring
``LineLadderTarget``'s own use of relative gaps rather than absolute plate
position. See ``check_depth_accuracy.py``'s module docstring for the full
measurement pipeline.

Unlike the line ladder, the number of points measured in a session is not
fixed. This target's own rig-specific geometry (a 30 mm height range packed
into a 50 mm footprint) makes self-occlusion from one or both cameras a real
possibility for some cells -- a session that only measures 15 of the grid's
cells is a normal, valid result, not a partial failure. Correspondences are
therefore keyed by ``(row, col)`` grid index, not by click order or count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import yaml

DEFAULT_DEPTH_GRID_TARGET_CONFIG = "calibration/config/depth_grid_target.yaml"


@dataclass(frozen=True)
class DepthGridTarget:
    """A flat baseplate carrying a grid of blocks of known, distinct heights."""

    heights_mm: Tuple[Tuple[float, ...], ...] = ()
    pitch_mm: float = 10.0
    reference_corner_count: int = 4
    height_uncertainty_mm: float = 0.05
    measured_by: str = "nominal (CAD) -- NOT verified"

    def __post_init__(self) -> None:
        if not self.heights_mm or not self.heights_mm[0]:
            raise ValueError("heights_mm must be a non-empty grid")
        row_length = len(self.heights_mm[0])
        if any(len(row) != row_length for row in self.heights_mm):
            raise ValueError("heights_mm rows must all be the same length")
        if any(height < 0.0 for row in self.heights_mm for height in row):
            raise ValueError("heights_mm must not contain negative heights")
        flat_heights = [height for row in self.heights_mm for height in row]
        if len(set(flat_heights)) != len(flat_heights):
            raise ValueError(
                "heights_mm must not contain duplicate heights -- block "
                "identification by engraved height requires every height to "
                "be unique"
            )
        if self.pitch_mm <= 0.0:
            raise ValueError(f"pitch_mm must be positive, got {self.pitch_mm}")
        if self.reference_corner_count < 3:
            raise ValueError(
                "reference_corner_count must be >= 3 for a plane fit, "
                f"got {self.reference_corner_count}"
            )
        if self.height_uncertainty_mm < 0.0:
            raise ValueError(
                f"height_uncertainty_mm must not be negative, got {self.height_uncertainty_mm}"
            )

    @property
    def row_count(self) -> int:
        return len(self.heights_mm)

    @property
    def col_count(self) -> int:
        return len(self.heights_mm[0])

    def height_at(self, row: int, col: int) -> float:
        return self.heights_mm[row][col]

    def cell_at_height(self, height_mm: float, tol: float = 1e-6) -> Tuple[int, int]:
        """Reverse lookup: which (row, col) carries this engraved height.

        Every height in the grid is unique (enforced in __post_init__), so
        this is well-defined. Raises with the sorted list of valid heights on
        a miss, since a typo here would otherwise be silently indistinguishable
        from a real measurement.
        """
        for row in range(self.row_count):
            for col in range(self.col_count):
                if abs(self.heights_mm[row][col] - height_mm) <= tol:
                    return row, col
        valid = sorted({height for row in self.heights_mm for height in row})
        raise ValueError(
            f"no block at height {height_mm} mm (tol {tol}); valid heights: {valid}"
        )

    def pair_depths_mm(
        self, cells: Sequence[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Every pair (i < j) among the CELLS ACTUALLY CLICKED this session,
        and its true depth separation.

        Mirrors LineLadderTarget.pair_distances_mm(): all C(n,2) pairs, not
        just adjacent ones. Unlike the ladder's fixed mark count, ``cells``
        varies session to session -- whichever (row, col) blocks were
        visible and clicked, in any order.
        """
        heights = np.array([self.height_at(row, col) for row, col in cells], dtype=np.float64)
        rows, cols = np.triu_indices(len(heights), k=1)
        return np.column_stack([rows, cols]), heights[cols] - heights[rows]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DepthGridTarget":
        target_type = str(data.get("target_type", "depth_grid")).lower()
        if target_type != "depth_grid":
            raise ValueError(f"Unsupported target_type '{target_type}'")
        if "heights_mm" not in data:
            raise ValueError("heights_mm is required")
        rows = tuple(tuple(float(value) for value in row) for row in data["heights_mm"])
        return cls(
            heights_mm=rows,
            pitch_mm=float(data.get("pitch_mm", 10.0)),
            reference_corner_count=int(data.get("reference_corner_count", 4)),
            height_uncertainty_mm=float(data.get("height_uncertainty_mm", 0.05)),
            measured_by=str(data.get("measured_by", "nominal (CAD) -- NOT verified")),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "DepthGridTarget":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Depth grid target config not found: {path}")
        with path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
        try:
            return cls.from_dict(data)
        except ValueError as exc:
            raise ValueError(f"{exc} in {path}") from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_type": "depth_grid",
            "heights_mm": [list(row) for row in self.heights_mm],
            "pitch_mm": self.pitch_mm,
            "reference_corner_count": self.reference_corner_count,
            "height_uncertainty_mm": self.height_uncertainty_mm,
            "measured_by": self.measured_by,
        }
