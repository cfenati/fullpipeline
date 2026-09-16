from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

try:
    import PySpin
except ImportError:
    PySpin = None


def _require_pyspin() -> Any:
    if PySpin is None:
        raise RuntimeError(
            "PySpin is not installed. Install the Spinnaker Python wheel into this environment."
        )
    return PySpin


def _set_color_processing(processor, prefer_fast: bool = True) -> None:
    pyspin = _require_pyspin()
    names = (
        "ColorProcessingAlgorithm_BILINEAR",
        "BILINEAR",
        "ColorProcessingAlgorithm_DEFAULT",
        "DEFAULT",
        "ColorProcessingAlgorithm_HQ_LINEAR",
        "HQ_LINEAR",
    )
    if not prefer_fast:
        names = names[4:] + names[:4]

    for name in names:
        algo = getattr(pyspin, name, None)
        if algo is not None:
            try:
                processor.SetColorProcessing(algo)
                return
            except Exception:
                pass


def _set_enum_by_name(nodemap, node_name: str, entry_name: str) -> bool:
    pyspin = _require_pyspin()
    node = pyspin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not pyspin.IsAvailable(node) or not pyspin.IsWritable(node):
        return False

    entry = node.GetEntryByName(entry_name)
    if not pyspin.IsReadable(entry):
        return False

    node.SetIntValue(entry.GetValue())
    return True


class BlackflyCamera:
    def __init__(
        self,
        name: str = "FLIR Blackfly",
        camera_index: int = 0,
        serial: Optional[str] = None,
        timeout_ms: int = 1000,
        gain_auto: bool = False,
        gain: Optional[float] = None,
        max_fps: Optional[float] = None,
        preview_max_width: Optional[int] = 1280,
        stream_newest_only: bool = True,
    ):
        self.name = name
        self.camera_index = camera_index
        self.serial = serial
        self.timeout_ms = timeout_ms
        self.gain_auto = gain_auto
        self.gain = gain
        self.max_fps = max_fps
        self.preview_max_width = preview_max_width
        self.stream_newest_only = stream_newest_only

        self._system = None
        self._cam_list = None
        self._cam = None
        self._processor = None
        self._model_name = None
        self._serial_number = None
        self._sensor_width = None
        self._sensor_height = None

    def is_open(self) -> bool:
        return self._cam is not None

    def open(self) -> None:
        pyspin = _require_pyspin()

        self._system = pyspin.System.GetInstance()
        self._cam_list = self._system.GetCameras()
        if self._cam_list.GetSize() == 0:
            raise RuntimeError("No FLIR/Spinnaker cameras found")

        cam = self._select_camera(pyspin)
        cam.Init()
        cam.AcquisitionMode.SetValue(pyspin.AcquisitionMode_Continuous)

        try:
            cam.PixelFormat.SetValue(pyspin.PixelFormat_Mono8)
        except Exception:
            pass

        try:
            if cam.GainAuto.GetAccessMode() == pyspin.RW:
                target = pyspin.GainAuto_Continuous if self.gain_auto else pyspin.GainAuto_Off
                cam.GainAuto.SetValue(target)
        except Exception:
            pass

        if not self.gain_auto and self.gain is not None:
            nodemap = cam.GetNodeMap()
            gain_node = pyspin.CFloatPtr(nodemap.GetNode("Gain"))
            if pyspin.IsAvailable(gain_node) and pyspin.IsWritable(gain_node):
                clamped = max(gain_node.GetMin(), min(float(self.gain), gain_node.GetMax()))
                gain_node.SetValue(clamped)

        if self.max_fps is not None:
            nodemap = cam.GetNodeMap()
            try:
                frame_rate_enable = pyspin.CBooleanPtr(
                    nodemap.GetNode("AcquisitionFrameRateEnable")
                )
                frame_rate = pyspin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
                if (
                    pyspin.IsAvailable(frame_rate_enable)
                    and pyspin.IsWritable(frame_rate_enable)
                    and pyspin.IsAvailable(frame_rate)
                    and pyspin.IsWritable(frame_rate)
                ):
                    frame_rate_enable.SetValue(True)
                    clamped = max(
                        frame_rate.GetMin(),
                        min(float(self.max_fps), frame_rate.GetMax()),
                    )
                    frame_rate.SetValue(clamped)
            except Exception:
                pass

        if self.stream_newest_only:
            stream_nodemap = cam.GetTLStreamNodeMap()
            _set_enum_by_name(
                stream_nodemap,
                "StreamBufferHandlingMode",
                "NewestOnly",
            )

        self._read_device_info(pyspin, cam)
        self._sensor_width = int(cam.Width.GetValue())
        self._sensor_height = int(cam.Height.GetValue())
        self._processor = pyspin.ImageProcessor()
        _set_color_processing(self._processor, prefer_fast=True)

        cam.BeginAcquisition()
        self._cam = cam

        print(
            f"{self.name}: {self._model_name} "
            f"(serial {self._serial_number}, index {self.camera_index}, "
            f"{self._sensor_width}x{self._sensor_height}, "
            f"preview<={self.preview_max_width or 'full'})"
        )

    def _select_camera(self, pyspin):
        if self.serial:
            for index in range(self._cam_list.GetSize()):
                cam = self._cam_list[index]
                nodemap = cam.GetTLDeviceNodeMap()
                serial_node = pyspin.CStringPtr(nodemap.GetNode("DeviceSerialNumber"))
                if not pyspin.IsReadable(serial_node):
                    continue
                if serial_node.GetValue() == self.serial:
                    return cam
            raise RuntimeError(f"FLIR camera with serial {self.serial} not found")

        if self.camera_index >= self._cam_list.GetSize():
            raise RuntimeError(
                f"FLIR camera index {self.camera_index} out of range "
                f"(found {self._cam_list.GetSize()} camera(s))"
            )
        return self._cam_list[self.camera_index]

    def _read_device_info(self, pyspin, cam) -> None:
        nodemap = cam.GetNodeMap()
        model_node = pyspin.CStringPtr(nodemap.GetNode("DeviceModelName"))
        serial_node = pyspin.CStringPtr(nodemap.GetNode("DeviceSerialNumber"))
        self._model_name = (
            model_node.GetValue() if pyspin.IsReadable(model_node) else "unknown"
        )
        self._serial_number = (
            serial_node.GetValue() if pyspin.IsReadable(serial_node) else "unknown"
        )

    def _to_bgr(self, img_ptr, full_resolution: bool) -> Optional[np.ndarray]:
        pyspin = _require_pyspin()
        height, width = img_ptr.GetHeight(), img_ptr.GetWidth()
        preview_max_width = None if full_resolution else self.preview_max_width

        if preview_max_width and width > preview_max_width:
            scale = preview_max_width / float(width)
            preview_width = max(1, int(width * scale))
            preview_height = max(1, int(height * scale))
            buffer = np.frombuffer(img_ptr.GetData(), dtype=np.uint8)
            if buffer.size != height * width:
                return None
            gray = buffer.reshape(height, width)
            resized = cv2.resize(
                gray,
                (preview_width, preview_height),
                interpolation=cv2.INTER_AREA,
            )
            return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        try:
            converted = self._processor.Convert(img_ptr, pyspin.PixelFormat_BGR8)
            out_height, out_width = converted.GetHeight(), converted.GetWidth()
            return (
                np.frombuffer(converted.GetData(), dtype=np.uint8)
                .reshape(out_height, out_width, 3)
                .copy()
            )
        except Exception:
            buffer = np.frombuffer(img_ptr.GetData(), dtype=np.uint8)
            if buffer.size != height * width:
                return None
            gray = buffer.reshape(height, width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def grab(self, full_resolution: bool = False) -> Optional[np.ndarray]:
        if not self.is_open():
            raise RuntimeError(f"{self.name} is not open")

        img_ptr = self._cam.GetNextImage(self.timeout_ms)
        if img_ptr.IsIncomplete():
            img_ptr.Release()
            return None

        try:
            return self._to_bgr(img_ptr, full_resolution=full_resolution)
        finally:
            img_ptr.Release()

    def set_gain(self, gain: float) -> float:
        """Set Gain (dB) on an already-open camera, disabling GainAuto first.
        Returns the value actually applied, clamped to the node's min/max."""
        if not self.is_open():
            raise RuntimeError(f"{self.name} is not open")

        pyspin = _require_pyspin()

        try:
            if self._cam.GainAuto.GetAccessMode() == pyspin.RW:
                self._cam.GainAuto.SetValue(pyspin.GainAuto_Off)
        except Exception:
            pass

        nodemap = self._cam.GetNodeMap()
        gain_node = pyspin.CFloatPtr(nodemap.GetNode("Gain"))
        if not (pyspin.IsAvailable(gain_node) and pyspin.IsWritable(gain_node)):
            raise RuntimeError(f"{self.name}: Gain node is not writable")

        clamped = max(gain_node.GetMin(), min(float(gain), gain_node.GetMax()))
        gain_node.SetValue(clamped)
        self.gain_auto = False
        self.gain = clamped
        return clamped

    def release(self) -> None:
        if self._cam is not None:
            try:
                self._cam.EndAcquisition()
            except Exception:
                pass
            try:
                self._cam.DeInit()
            except Exception:
                pass
            del self._cam
            self._cam = None

        if self._cam_list is not None:
            self._cam_list.Clear()
            self._cam_list = None

        if self._system is not None:
            self._system.ReleaseInstance()
            self._system = None

        self._processor = None

    def info(self) -> dict:
        return {
            "name": self.name,
            "model": self._model_name,
            "serial": self._serial_number,
            "camera_index": self.camera_index,
            "timeout_ms": self.timeout_ms,
            "gain_auto": self.gain_auto,
            "gain": self.gain,
            "max_fps": self.max_fps,
            "sensor_width": self._sensor_width,
            "sensor_height": self._sensor_height,
            "preview_max_width": self.preview_max_width,
            "stream_newest_only": self.stream_newest_only,
        }
