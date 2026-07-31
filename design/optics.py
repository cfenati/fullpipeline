"""Sensor and lens catalogues for the hardware in this rig.

Where the numbers come from
---------------------------
Optris Xi 400 LT USB, "OPTICAL" table of the product page / datasheet::

    Field of View               18x14    29x22    53x38    80x54   [deg]
    Focal length                 20.0     12.7      7.7      5.7   [mm]
    F number                      1.1      0.9      0.9      0.9
    Optical resolution (D:S)    390:1    239:1    128:1     78:1
    Minimum distance to target    350      350      250      200   [mm]
    Smallest detectable spot (IFOV, 1 px)
                                  0.3      0.5      0.7      0.8   [mm]
    Smallest measurable spot (MFOV, 3x3 px)
                                  0.9      1.5      2.1      2.4   [mm]

Why the datasheet FOV and not the focal length: a pinhole model gives
``2*atan(sensor/2f)``, which only holds for a rectilinear lens. The two long
lenses agree with the datasheet to a few tenths of a degree, the two wide ones
do not, because they are deliberately barrel-distorted::

    f = 20.0 mm   pinhole 18.4 x 14.0    datasheet 18 x 14    agrees
    f = 12.7 mm   pinhole 28.7 x 21.8    datasheet 29 x 22    agrees
    f =  7.7 mm   pinhole 45.7 x 35.3    datasheet 53 x 38    differs
    f =  5.7 mm   pinhole 59.3 x 46.5    datasheet 80 x 54    differs

The datasheet's own IFOV column settles which one to trust. IFOV is the pixel
footprint at the minimum distance, so it must equal ``z_min / f_px``. Reading
``f_px`` off the datasheet FOV reproduces all four IFOV values; reading it off
the focal length only reproduces the two long ones. See
:func:`verify_optris_lenses`. So use ``source: fov`` with the datasheet angles,
and remember that a wide IR lens then maps non-linearly inside that cone -
pixel-accurate overlays still need a measured distortion model.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np


class Sensor(NamedTuple):
    """A detector: full pixel count and pixel pitch."""

    name: str
    width_px: int
    height_px: int
    pitch_um: float

    @property
    def width_mm(self) -> float:
        return self.width_px * self.pitch_um / 1000.0

    @property
    def height_mm(self) -> float:
        return self.height_px * self.pitch_um / 1000.0

    @property
    def diagonal_mm(self) -> float:
        return float(np.hypot(self.width_mm, self.height_mm))


class Lens(NamedTuple):
    label: str
    focal_mm: float
    hfov_deg: float
    vfov_deg: float
    rectilinear: bool
    min_distance_m: float
    ifov_mm: float          # one-pixel footprint at the minimum distance
    f_number: float

    @property
    def mfov_mm(self) -> float:
        """Smallest *measurable* spot: Optris specifies 3x3 pixels."""
        return 3.0 * self.ifov_mm


# --------------------------------------------------------------------------- #
# Optris Xi 400 LT USB: 382 x 288 px, 17 um pitch, 8-14 um, 80 Hz
# --------------------------------------------------------------------------- #
OPTRIS_XI400 = Sensor("Optris Xi 400 LT USB (uncooled bolometer)", 382, 288, 17.0)

OPTRIS_XI400_LENSES: List[Lens] = [
    Lens("18x14", 20.0, 18.0, 14.0, True, 0.350, 0.3, 1.1),
    Lens("29x22", 12.7, 29.0, 22.0, True, 0.350, 0.5, 0.9),
    Lens("53x38", 7.7, 53.0, 38.0, False, 0.250, 0.7, 0.9),
    Lens("80x54", 5.7, 80.0, 54.0, False, 0.200, 0.8, 0.9),
]

# --------------------------------------------------------------------------- #
# ELP-USB16MP01-MFV: Sony IMX298, 1/2.8 in, 16 MP, 5-50 mm manual zoom CS lens
# 10 fps at 4656x3496, 30 fps at 1080p, MJPEG/YUY2 over UVC.
# --------------------------------------------------------------------------- #
ELP_USB16MP01 = Sensor("Sony IMX298 (1/2.8 in)", 4656, 3496, 1.12)
ELP_USB16MP01_ZOOM_MM: Tuple[float, float] = (5.0, 50.0)


# --------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------- #
def fov_from_focal(sensor: Sensor, focal_mm: float,
                   resolution: Optional[Sequence[int]] = None
                   ) -> Tuple[float, float]:
    """Rectilinear field of view in degrees for a focal length.

    ``resolution`` lets you evaluate a cropped or binned readout mode; it is
    interpreted as a centred window of the same pixel pitch.
    """
    w_px, h_px = (sensor.width_px, sensor.height_px) if resolution is None \
        else (int(resolution[0]), int(resolution[1]))
    half_w = w_px * sensor.pitch_um / 2000.0
    half_h = h_px * sensor.pitch_um / 2000.0
    return (float(np.degrees(2 * np.arctan(half_w / focal_mm))),
            float(np.degrees(2 * np.arctan(half_h / focal_mm))))


def zoom_table(sensor: Sensor, focal_lengths: Sequence[float],
               resolution: Optional[Sequence[int]] = None
               ) -> List[Tuple[float, float, float]]:
    """``(focal_mm, hfov_deg, vfov_deg)`` for each focal length of a zoom lens."""
    return [(f,) + fov_from_focal(sensor, f, resolution) for f in focal_lengths]


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
def verify_optris_lenses() -> List[dict]:
    """Cross-check the datasheet FOV against its own IFOV column.

    IFOV is the footprint of one pixel at the minimum distance, so
    ``ifov = z_min / f_px``. Computing ``f_px`` from the datasheet angles should
    reproduce the published IFOV; computing it from the focal length should not,
    for the two wide lenses.
    """
    out = []
    for lens in OPTRIS_XI400_LENSES:
        f_px_fov = (OPTRIS_XI400.width_px / 2.0) / np.tan(
            np.radians(lens.hfov_deg) / 2.0)
        f_px_focal = lens.focal_mm / (OPTRIS_XI400.pitch_um / 1000.0)
        out.append({
            "lens": lens.label,
            "ifov_datasheet_mm": lens.ifov_mm,
            "ifov_from_fov_mm": lens.min_distance_m / f_px_fov * 1000.0,
            "ifov_from_focal_mm": lens.min_distance_m / f_px_focal * 1000.0,
            "hfov_datasheet_deg": lens.hfov_deg,
            "hfov_from_focal_deg": fov_from_focal(OPTRIS_XI400, lens.focal_mm)[0],
        })
    return out
