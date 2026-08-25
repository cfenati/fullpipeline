"""Parallel-line ladder target: an independent metric ruler for this rig.

Unlike ``target_board.TargetBoard``, this target is never used to *fit* anything.
It exists only to be measured. The calibration's entire metric scale comes from
the ChArUco ``square_size_m``, so checking triangulated lengths against a board
of the same nominal square cannot detect a wrong square size -- the error cancels.
A separately fabricated ladder, whose spacings were measured with a different
instrument, is the only thing here that can.

Two ways to describe a ladder:

``gaps_mm``
    Explicit gap list, so the line count is fixed at ``len(gaps_mm) + 1`` and the
    whole plate must be in frame. Unequal gaps make the pattern self-checking: the
    marks fit it only one way, so a mark lost off the frame edge is caught.

``spacing_mm``
    One uniform gap and *no fixed count*, for a plate longer than the field of
    view. Whatever consecutive run of marks is visible gets measured, and the
    distance from mark i to mark j is simply ``(j - i) * spacing_mm``. Absolute
    identity never enters the measurement, so it does not matter which part of the
    plate is in frame -- only that the visible marks are consecutive, which the
    detector checks by requiring uniform spacing.

Distances are quoted between **line centres**, not edges. Laser kerf, marking
width, exposure and blooming all widen a line symmetrically, so a centre-to-centre
distance is immune to every one of them; an edge-to-edge distance is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

DEFAULT_LINE_TARGET_CONFIG = "calibration/config/line_target.yaml"

VALID_ORIENTATIONS = ("vertical", "horizontal")


@dataclass(frozen=True)
class LineLadderTarget:
    """A flat, rigid plate carrying parallel lines at known centre-to-centre gaps."""

    gaps_mm: Tuple[float, ...] = ()
    spacing_mm: Optional[float] = None
    orientation: str = "vertical"
    line_width_mm: float = 1.0
    gaps_uncertainty_mm: float = 0.05
    measured_by: str = "nominal (CAD) -- NOT verified"

    def __post_init__(self) -> None:
        if bool(self.gaps_mm) == (self.spacing_mm is not None):
            raise ValueError("set exactly one of gaps_mm or spacing_mm")
        if self.spacing_mm is not None and self.spacing_mm <= 0.0:
            raise ValueError(f"spacing_mm must be positive, got {self.spacing_mm}")
        if any(gap <= 0.0 for gap in self.gaps_mm):
            raise ValueError(f"gaps_mm must all be positive, got {list(self.gaps_mm)}")
        if self.orientation not in VALID_ORIENTATIONS:
            raise ValueError(
                f"orientation must be one of {VALID_ORIENTATIONS}, got '{self.orientation}'"
            )
        if self.line_width_mm <= 0.0:
            raise ValueError(f"line_width_mm must be positive, got {self.line_width_mm}")
        if self.gaps_uncertainty_mm < 0.0:
            raise ValueError(
                f"gaps_uncertainty_mm must not be negative, got {self.gaps_uncertainty_mm}"
            )

    @property
    def open_ended(self) -> bool:
        """True when the count is whatever is visible rather than fixed."""
        return self.spacing_mm is not None

    @property
    def line_count(self) -> Optional[int]:
        """Fixed number of marks, or None when the count comes from the image."""
        return None if self.open_ended else len(self.gaps_mm) + 1

    def positions_mm(self, count: Optional[int] = None) -> np.ndarray:
        """Centre of each mark along the axis perpendicular to them, first at 0."""
        if self.open_ended:
            if count is None:
                raise ValueError("an open-ended ladder needs the detected mark count")
            return np.arange(count, dtype=np.float64) * float(self.spacing_mm)
        return np.concatenate([[0.0], np.cumsum(np.asarray(self.gaps_mm, dtype=np.float64))])

    def span_mm(self, count: Optional[int] = None) -> float:
        """First mark centre to last. Must fit the stereo overlap at working distance."""
        return float(self.positions_mm(count)[-1])

    @property
    def strictly_increasing(self) -> bool:
        """Whether the gap pattern pins the indexing on its own.

        Only unequal gaps do. A uniform ladder looks the same shifted by one mark,
        so a missed mark cannot be detected from the pattern -- which is exactly
        why the open-ended mode measures ``(j - i) * spacing_mm`` instead of
        relying on absolute mark identity.
        """
        if self.open_ended:
            return False
        gaps = np.asarray(self.gaps_mm, dtype=np.float64)
        return bool(np.all(np.diff(gaps) > 0.0)) if gaps.size > 1 else True

    def pair_distances_mm(
        self, count: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Every mark pair (i < j) and its true centre-to-centre distance.

        All pairs, not just adjacent ones: sixteen marks give 120 distances
        spanning the full plate instead of fifteen, from one capture.
        """
        positions = self.positions_mm(count)
        rows, cols = np.triu_indices(len(positions), k=1)
        return np.column_stack([rows, cols]), positions[cols] - positions[rows]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineLadderTarget":
        target_type = str(data.get("target_type", "line_ladder")).lower()
        if target_type != "line_ladder":
            raise ValueError(f"Unsupported target_type '{target_type}'")
        if "gaps_mm" not in data and "spacing_mm" not in data:
            raise ValueError("gaps_mm or spacing_mm is required")

        gaps = data.get("gaps_mm") or ()
        spacing = data.get("spacing_mm")
        return cls(
            gaps_mm=tuple(float(gap) for gap in gaps),
            spacing_mm=None if spacing is None else float(spacing),
            orientation=str(data.get("orientation", "vertical")).lower(),
            line_width_mm=float(data.get("line_width_mm", 1.0)),
            gaps_uncertainty_mm=float(data.get("gaps_uncertainty_mm", 0.05)),
            measured_by=str(data.get("measured_by", "nominal (CAD) -- NOT verified")),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "LineLadderTarget":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Line target config not found: {path}")
        with path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
        try:
            return cls.from_dict(data)
        except ValueError as exc:
            raise ValueError(f"{exc} in {path}") from exc

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "target_type": "line_ladder",
            "orientation": self.orientation,
            "line_width_mm": self.line_width_mm,
            "gaps_uncertainty_mm": self.gaps_uncertainty_mm,
            "measured_by": self.measured_by,
        }
        if self.open_ended:
            payload["spacing_mm"] = self.spacing_mm
        else:
            payload["gaps_mm"] = list(self.gaps_mm)
        return payload
