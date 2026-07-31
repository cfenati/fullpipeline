"""Flat-field RGB color correction for uniform lighting and white balance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

DEFAULT_BLUR_SIGMA_FRACTION = 0.08
DEFAULT_BORDER_FRACTION = 0.12
DEFAULT_CORRECTION_STRENGTH = 0.65
DEFAULT_MAX_GAIN = 1.35
DEFAULT_MIN_GAIN = 0.85


def effective_gains(
    gain: np.ndarray,
    strength: float = DEFAULT_CORRECTION_STRENGTH,
    max_gain: float = DEFAULT_MAX_GAIN,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> np.ndarray:
    """Clamp and partially apply flat-field gains (1.0 = no correction)."""
    strength = float(np.clip(strength, 0.0, 1.0))
    clipped = np.clip(gain, min_gain, max_gain)
    return 1.0 + strength * (clipped - 1.0)


def center_patch(image: np.ndarray, border_fraction: float = DEFAULT_BORDER_FRACTION) -> np.ndarray:
    height, width = image.shape[:2]
    margin = max(4, int(min(height, width) * border_fraction))
    center_y, center_x = height // 2, width // 2
    return image[
        center_y - margin : center_y + margin,
        center_x - margin : center_x + margin,
    ]


def neutralize_center_white(
    image: np.ndarray,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
) -> np.ndarray:
    """Scale BGR channels so the center patch has equal channel means."""
    patch = center_patch(image, border_fraction=border_fraction).reshape(-1, 3).astype(np.float32)
    b_mean, g_mean, r_mean = patch.mean(axis=0)
    gray_target = float((b_mean + g_mean + r_mean) / 3.0)
    scale = np.array(
        [
            gray_target / (b_mean + 1e-6),
            gray_target / (g_mean + 1e-6),
            gray_target / (r_mean + 1e-6),
        ],
        dtype=np.float32,
    )
    corrected = np.clip(image.astype(np.float32) * scale, 0, 255)
    return corrected.astype(np.uint8)


@dataclass
class FlatFieldMaps:
    gain_b: np.ndarray
    gain_g: np.ndarray
    gain_r: np.ndarray
    blur_sigma_px: float
    source_width: int
    source_height: int

    def apply(
        self,
        image: np.ndarray,
        border_fraction: float = DEFAULT_BORDER_FRACTION,
        neutralize_white: bool = False,
        strength: float = DEFAULT_CORRECTION_STRENGTH,
        max_gain: float = DEFAULT_MAX_GAIN,
        min_gain: float = DEFAULT_MIN_GAIN,
    ) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected BGR image with shape (H, W, 3), got {image.shape}")

        raw_gains = self._gains_for_shape(image.shape[:2])
        gains = tuple(
            effective_gains(gain, strength=strength, max_gain=max_gain, min_gain=min_gain)
            for gain in raw_gains
        )
        image_f = image.astype(np.float32)
        channels = cv2.split(image_f)
        corrected = cv2.merge([channel * gain for channel, gain in zip(channels, gains)])
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        if neutralize_white:
            corrected = neutralize_center_white(corrected, border_fraction=border_fraction)
        return corrected

    def _gains_for_shape(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = shape
        if (height, width) == (self.source_height, self.source_width):
            return self.gain_b, self.gain_g, self.gain_r

        return (
            cv2.resize(self.gain_b, (width, height), interpolation=cv2.INTER_LINEAR),
            cv2.resize(self.gain_g, (width, height), interpolation=cv2.INTER_LINEAR),
            cv2.resize(self.gain_r, (width, height), interpolation=cv2.INTER_LINEAR),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            gain_b=self.gain_b,
            gain_g=self.gain_g,
            gain_r=self.gain_r,
            blur_sigma_px=np.array([self.blur_sigma_px], dtype=np.float32),
            source_width=np.array([self.source_width], dtype=np.int32),
            source_height=np.array([self.source_height], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "FlatFieldMaps":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Flat-field file not found: {path}")

        data = np.load(path)
        return cls(
            gain_b=data["gain_b"].astype(np.float32),
            gain_g=data["gain_g"].astype(np.float32),
            gain_r=data["gain_r"].astype(np.float32),
            blur_sigma_px=float(data["blur_sigma_px"][0]),
            source_width=int(data["source_width"][0]),
            source_height=int(data["source_height"][0]),
        )

    @classmethod
    def from_white_reference(
        cls,
        image: np.ndarray,
        blur_sigma_fraction: float = DEFAULT_BLUR_SIGMA_FRACTION,
        border_fraction: float = DEFAULT_BORDER_FRACTION,
    ) -> "FlatFieldMaps":
        height, width = image.shape[:2]
        sigma = max(8.0, min(height, width) * blur_sigma_fraction)
        image_f = image.astype(np.float32)
        center = center_patch(image_f, border_fraction=border_fraction)
        center_means = center.reshape(-1, 3).mean(axis=0)

        gain_maps = []
        for channel_index, channel in enumerate(cv2.split(image_f)):
            shading = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
            gain_maps.append((center_means[channel_index] / (shading + 1e-6)).astype(np.float32))

        return cls(
            gain_b=gain_maps[0],
            gain_g=gain_maps[1],
            gain_r=gain_maps[2],
            blur_sigma_px=sigma,
            source_width=width,
            source_height=height,
        )


def compute_flat_field_from_white(
    image: np.ndarray,
    blur_sigma_fraction: float = DEFAULT_BLUR_SIGMA_FRACTION,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
) -> FlatFieldMaps:
    """Build per-channel gain maps from a uniform white/gray reference frame."""
    return FlatFieldMaps.from_white_reference(
        image,
        blur_sigma_fraction=blur_sigma_fraction,
        border_fraction=border_fraction,
    )


def apply_flat_field_correction(
    image: np.ndarray,
    flat_field: Optional[FlatFieldMaps] = None,
    blur_sigma_fraction: float = DEFAULT_BLUR_SIGMA_FRACTION,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
    neutralize_white: bool = True,
    strength: float = 1.0,
    max_gain: float = DEFAULT_MAX_GAIN,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> Tuple[np.ndarray, FlatFieldMaps]:
    """Correct vignetting and color tint using flat-field division.

    If ``flat_field`` is omitted, gain maps are estimated from ``image`` itself
    (useful for one-off correction of a white reference capture).
    """
    if flat_field is None:
        flat_field = compute_flat_field_from_white(
            image,
            blur_sigma_fraction=blur_sigma_fraction,
            border_fraction=border_fraction,
        )

    corrected = flat_field.apply(
        image,
        border_fraction=border_fraction,
        neutralize_white=neutralize_white,
        strength=strength,
        max_gain=max_gain,
        min_gain=min_gain,
    )
    return corrected, flat_field


@dataclass
class RGBColorCorrector:
    """Per-camera flat-field correction loaded from config."""

    cam1: Optional[FlatFieldMaps]
    cam2: Optional[FlatFieldMaps]
    blur_sigma_fraction: float = DEFAULT_BLUR_SIGMA_FRACTION
    border_fraction: float = DEFAULT_BORDER_FRACTION
    on_the_fly_fallback: bool = True
    neutralize_white: bool = False
    preview_corrected: bool = True
    strength: float = DEFAULT_CORRECTION_STRENGTH
    max_gain: float = DEFAULT_MAX_GAIN
    min_gain: float = DEFAULT_MIN_GAIN

    def correct_cam1(self, frame: np.ndarray) -> np.ndarray:
        return self._correct_frame(frame, self.cam1)

    def correct_cam2(self, frame: np.ndarray) -> np.ndarray:
        return self._correct_frame(frame, self.cam2)

    def preview_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.preview_corrected:
            return frame1, frame2
        return self.correct_cam1(frame1), self.correct_cam2(frame2)

    def _correct_frame(
        self,
        frame: np.ndarray,
        flat_field: Optional[FlatFieldMaps],
    ) -> np.ndarray:
        if flat_field is not None:
            return flat_field.apply(
                frame,
                border_fraction=self.border_fraction,
                neutralize_white=self.neutralize_white,
                strength=self.strength,
                max_gain=self.max_gain,
                min_gain=self.min_gain,
            )

        if not self.on_the_fly_fallback:
            return frame

        corrected, _ = apply_flat_field_correction(
            frame,
            blur_sigma_fraction=self.blur_sigma_fraction,
            border_fraction=self.border_fraction,
            neutralize_white=self.neutralize_white,
            strength=self.strength,
            max_gain=self.max_gain,
            min_gain=self.min_gain,
        )
        return corrected

    def info(self) -> Dict[str, Any]:
        return {
            "cam1_flat_field": self._flat_field_info(self.cam1),
            "cam2_flat_field": self._flat_field_info(self.cam2),
            "blur_sigma_fraction": self.blur_sigma_fraction,
            "border_fraction": self.border_fraction,
            "on_the_fly_fallback": self.on_the_fly_fallback,
            "neutralize_white": self.neutralize_white,
            "preview_corrected": self.preview_corrected,
            "strength": self.strength,
            "max_gain": self.max_gain,
            "min_gain": self.min_gain,
        }

    @staticmethod
    def _flat_field_info(flat_field: Optional[FlatFieldMaps]) -> Optional[Dict[str, Any]]:
        if flat_field is None:
            return None
        return {
            "source_width": flat_field.source_width,
            "source_height": flat_field.source_height,
            "blur_sigma_px": flat_field.blur_sigma_px,
        }


def _resolve_path(project_root: Path, path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _load_optional_flat_field(path: Optional[Path]) -> Optional[FlatFieldMaps]:
    if path is None:
        return None
    if not path.exists():
        print(f"Color correction: flat-field file not found, skipping: {path}")
        return None
    return FlatFieldMaps.load(path)


def load_rgb_color_corrector(config: dict, project_root: Path) -> Optional[RGBColorCorrector]:
    rgb_config = config.get("rgb", {})
    correction_config = rgb_config.get("color_correction", {})
    if not correction_config.get("enabled", False):
        return None

    cam1_path = _resolve_path(project_root, correction_config.get("flat_field_cam1"))
    cam2_path = _resolve_path(project_root, correction_config.get("flat_field_cam2"))

    corrector = RGBColorCorrector(
        cam1=_load_optional_flat_field(cam1_path),
        cam2=_load_optional_flat_field(cam2_path),
        blur_sigma_fraction=float(
            correction_config.get("blur_sigma_fraction", DEFAULT_BLUR_SIGMA_FRACTION)
        ),
        border_fraction=float(correction_config.get("border_fraction", DEFAULT_BORDER_FRACTION)),
        on_the_fly_fallback=bool(correction_config.get("on_the_fly_fallback", True)),
        neutralize_white=bool(correction_config.get("neutralize_white", False)),
        preview_corrected=bool(correction_config.get("preview_corrected", True)),
        strength=float(correction_config.get("strength", DEFAULT_CORRECTION_STRENGTH)),
        max_gain=float(correction_config.get("max_gain", DEFAULT_MAX_GAIN)),
        min_gain=float(correction_config.get("min_gain", DEFAULT_MIN_GAIN)),
    )

    if corrector.cam1 is None and corrector.cam2 is None and not corrector.on_the_fly_fallback:
        print(
            "Color correction enabled but no flat-field files loaded and "
            "on_the_fly_fallback=false; correction disabled."
        )
        return None

    modes = []
    if corrector.cam1 is not None:
        modes.append("cam1=flat_field")
    elif corrector.on_the_fly_fallback:
        modes.append("cam1=on_the_fly")
    if corrector.cam2 is not None:
        modes.append("cam2=flat_field")
    elif corrector.on_the_fly_fallback:
        modes.append("cam2=on_the_fly")
    print(f"RGB color correction enabled ({', '.join(modes)})")
    print(
        "RGB correction tuning: "
        f"strength={corrector.strength:.2f}, "
        f"gain=[{corrector.min_gain:.2f}, {corrector.max_gain:.2f}]"
    )
    if corrector.preview_corrected:
        print("RGB preview: flat-field correction applied (WYSIWYG)")
    else:
        print("RGB preview: raw (saved captures are still corrected)")
    return corrector
