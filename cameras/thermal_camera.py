import ctypes as ct
from ctypes.util import find_library
import os
from pathlib import Path
from typing import Optional

import numpy as np


class EvoIRFrameMetadata(ct.Structure):
    _fields_ = [
        ("counter", ct.c_uint),
        ("counterHW", ct.c_uint),
        ("timestamp", ct.c_longlong),
        ("timestampMedia", ct.c_longlong),
        ("flagState", ct.c_int),
        ("tempChip", ct.c_float),
        ("tempFlag", ct.c_float),
        ("tempBox", ct.c_float),
    ]


class ThermalFrame:
    def __init__(
        self,
        thermal_raw: np.ndarray,
        temperature_c: np.ndarray,
        palette_bgr: np.ndarray,
        metadata: EvoIRFrameMetadata,
        mean_temp_c: float,
    ):
        self.thermal_raw = thermal_raw
        self.temperature_c = temperature_c
        self.palette_bgr = palette_bgr
        self.metadata = metadata
        self.mean_temp_c = mean_temp_c


class ThermalCamera:
    def __init__(self, config_xml: Path):
        self.config_xml = Path(config_xml)
        self._libir = None
        self._metadata = EvoIRFrameMetadata()
        self._thermal_width = ct.c_int()
        self._thermal_height = ct.c_int()
        self._palette_width = ct.c_int()
        self._palette_height = ct.c_int()
        self._serial = ct.c_ulong()
        self._np_thermal = None
        self._np_palette = None
        self._thermal_pointer = None
        self._palette_pointer = None

    def open(self) -> None:
        if not self.config_xml.is_file():
            raise FileNotFoundError(f"Thermal config not found: {self.config_xml}")

        if os.name == "nt":
            self._libir = ct.CDLL(".\\libirimager.dll")
        else:
            library_path = find_library("irdirectsdk")
            if library_path is None:
                raise RuntimeError(
                    "Could not find irdirectsdk. Install libirimager first."
                )
            self._libir = ct.cdll.LoadLibrary(library_path)

        path_xml = str(self.config_xml.resolve()).encode()
        ret = self._libir.evo_irimager_usb_init(path_xml, b"", b"")
        if ret != 0:
            raise RuntimeError(f"evo_irimager_usb_init failed with code {ret}")

        self._libir.evo_irimager_get_serial(ct.byref(self._serial))
        self._libir.evo_irimager_get_thermal_image_size(
            ct.byref(self._thermal_width),
            ct.byref(self._thermal_height),
        )
        self._libir.evo_irimager_get_palette_image_size(
            ct.byref(self._palette_width),
            ct.byref(self._palette_height),
        )

        thermal_size = self._thermal_width.value * self._thermal_height.value
        palette_size = (
            self._palette_width.value * self._palette_height.value * 3
        )

        self._np_thermal = np.zeros(thermal_size, dtype=np.uint16)
        self._np_palette = np.zeros(palette_size, dtype=np.uint8)
        self._thermal_pointer = self._np_thermal.ctypes.data_as(
            ct.POINTER(ct.c_ushort)
        )
        self._palette_pointer = self._np_palette.ctypes.data_as(
            ct.POINTER(ct.c_ubyte)
        )

        print(
            "Thermal camera: "
            f"serial={self._serial.value}, "
            f"thermal={self._thermal_width.value}x{self._thermal_height.value}, "
            f"palette={self._palette_width.value}x{self._palette_height.value}"
        )

    def grab(self) -> Optional[ThermalFrame]:
        if self._libir is None:
            raise RuntimeError("Thermal camera is not open")

        ret = self._libir.evo_irimager_get_thermal_palette_image_metadata(
            self._thermal_width,
            self._thermal_height,
            self._thermal_pointer,
            self._palette_width,
            self._palette_height,
            self._palette_pointer,
            ct.byref(self._metadata),
        )
        if ret != 0:
            print(f"Thermal grab failed with code {ret}")
            return None

        thermal_raw = self._np_thermal.reshape(
            self._thermal_height.value,
            self._thermal_width.value,
        ).copy()
        palette_rgb = self._np_palette.reshape(
            self._palette_height.value,
            self._palette_width.value,
            3,
        )
        palette_bgr = palette_rgb[:, :, ::-1].copy()
        temperature_c = self.raw_to_celsius(thermal_raw)
        mean_temp_c = float(temperature_c.mean())

        return ThermalFrame(
            thermal_raw=thermal_raw,
            temperature_c=temperature_c,
            palette_bgr=palette_bgr,
            metadata=self._metadata,
            mean_temp_c=mean_temp_c,
        )

    @staticmethod
    def raw_to_celsius(thermal_raw: np.ndarray) -> np.ndarray:
        return thermal_raw.astype(np.float32) / 10.0 - 100.0

    @staticmethod
    def celsius_to_int16_centidegrees(temperature_c: np.ndarray) -> np.ndarray:
        return np.round(temperature_c * 100.0).astype(np.int16)

    def release(self) -> None:
        if self._libir is not None:
            self._libir.evo_irimager_terminate()
            self._libir = None

    def info(self) -> dict:
        return {
            "config_xml": str(self.config_xml),
            "serial": int(self._serial.value) if self._libir is not None else None,
            "thermal_width": self._thermal_width.value,
            "thermal_height": self._thermal_height.value,
            "palette_width": self._palette_width.value,
            "palette_height": self._palette_height.value,
        }

    @staticmethod
    def metadata_to_dict(metadata: EvoIRFrameMetadata) -> dict:
        return {
            "counter": int(metadata.counter),
            "counterHW": int(metadata.counterHW),
            "timestamp": int(metadata.timestamp),
            "timestampMedia": int(metadata.timestampMedia),
            "flagState": int(metadata.flagState),
            "tempChip": float(metadata.tempChip),
            "tempFlag": float(metadata.tempFlag),
            "tempBox": float(metadata.tempBox),
        }
