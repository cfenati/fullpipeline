"""ChArUco calibration target description."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml

DEFAULT_BOARD_CONFIG = "calibration/config/charuco_11x8.yaml"


@dataclass(frozen=True)
class TargetBoard:
    squares_x: int
    squares_y: int
    square_size_m: float
    marker_size_m: float
    dictionary: str = "DICT_4X4_50"
    legacy_pattern: bool = True

    @property
    def corners_x(self) -> int:
        return self.squares_x - 1

    @property
    def corners_y(self) -> int:
        return self.squares_y - 1

    @property
    def total_corners(self) -> int:
        return self.corners_x * self.corners_y

    @classmethod
    def from_yaml(cls, path: Path) -> "TargetBoard":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Board config not found: {path}")

        with path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}

        target_type = str(data.get("target_type", "charuco")).lower()
        if target_type != "charuco":
            raise ValueError(f"Unsupported target_type '{target_type}' in {path}")

        return cls(
            squares_x=int(data["squares_x"]),
            squares_y=int(data["squares_y"]),
            square_size_m=float(data["square_size_m"]),
            marker_size_m=float(data["marker_size_m"]),
            dictionary=str(data.get("dictionary", "DICT_4X4_50")),
            legacy_pattern=bool(data.get("legacy_pattern", True)),
        )

    def aruco_dictionary(self) -> Any:
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"Unknown ArUco dictionary: {self.dictionary}")
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))

    def build_board(self) -> Any:
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_size_m,
            self.marker_size_m,
            self.aruco_dictionary(),
        )
        board.setLegacyPattern(self.legacy_pattern)
        return board

    def create_detector(self) -> Any:
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return cv2.aruco.CharucoDetector(
            self.build_board(),
            cv2.aruco.CharucoParameters(),
            detector_params,
        )

    def chessboard_corners(self) -> np.ndarray:
        """Object points of every interior corner, in board coordinates (metres)."""
        return np.asarray(self.build_board().getChessboardCorners(), dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_type": "charuco",
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_size_m": self.square_size_m,
            "marker_size_m": self.marker_size_m,
            "dictionary": self.dictionary,
            "legacy_pattern": self.legacy_pattern,
        }
