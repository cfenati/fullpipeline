import subprocess
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

STARTUP_FRAMES_TO_DISCARD = 8
GRAB_PAIR_MAX_ATTEMPTS = 8
# Auto toggles must be switched off before their manual counterpart becomes
# writable (v4l2 reports e.g. exposure_time_absolute as "inactive" while
# auto_exposure is in Aperture Priority).
CONTROL_ORDER_FIRST = ("white_balance_automatic", "auto_exposure")


def controls_for(rgb_config: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Merge rgb.controls.common with the per-camera overrides for ``key``."""
    controls_config = rgb_config.get("controls") or {}
    merged: Dict[str, Any] = dict(controls_config.get("common") or {})
    merged.update(controls_config.get(key) or {})
    return merged


class RGBCamera:
    def __init__(
        self,
        device: str,
        name: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        controls: Optional[Dict[str, Any]] = None,
    ):
        self.device = device
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.controls = dict(controls or {})
        self._cap = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {self.name} at {self.device}")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(
            chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4)
        )

        print(
            f"{self.name} ({self.device}): "
            f"{self.width}x{self.height}, "
            f"{self.fps:.2f} FPS, "
            f"{fourcc}"
        )

        self._cap = cap
        self._apply_controls()
        self._discard_startup_frames()

    def _apply_controls(self) -> None:
        """Push v4l2 controls from config; they do not survive a replug/reboot."""
        if not self.controls:
            return

        ordered = sorted(
            self.controls.items(),
            key=lambda item: (
                CONTROL_ORDER_FIRST.index(item[0])
                if item[0] in CONTROL_ORDER_FIRST
                else len(CONTROL_ORDER_FIRST)
            ),
        )

        for control, value in ordered:
            command = ["v4l2-ctl", "-d", self.device, "-c", f"{control}={value}"]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except FileNotFoundError:
                print(f"{self.name}: v4l2-ctl not found, skipping controls")
                return
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or error.stdout or "").strip()
                print(f"{self.name}: could not set {control}={value} ({detail})")

        summary = ", ".join(f"{name}={value}" for name, value in ordered)
        print(f"{self.name} controls: {summary}")

    def _discard_startup_frames(self) -> None:
        if self._cap is None:
            return

        for _ in range(STARTUP_FRAMES_TO_DISCARD):
            if not self.grab_only():
                break
            self.retrieve_frame()

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def grab_only(self) -> bool:
        if not self.is_open():
            return False

        try:
            return self._cap.grab()
        except cv2.error:
            return False

    def retrieve_frame(self) -> Optional[np.ndarray]:
        if not self.is_open():
            return None

        try:
            ret, frame = self._cap.retrieve()
        except cv2.error:
            return None

        if not ret or frame is None or frame.size == 0:
            return None

        return frame

    def grab(self) -> Optional[np.ndarray]:
        if not self.grab_only():
            return None

        return self.retrieve_frame()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def recover(self, settle_s: float = 2.0) -> bool:
        print(f"Recovering {self.name}...")
        self.release()
        time.sleep(settle_s)
        try:
            self.open()
        except RuntimeError as error:
            print(f"Recovery failed for {self.name}: {error}")
            return False
        return True

    @staticmethod
    def grab_pair(
        cam1: "RGBCamera",
        cam2: "RGBCamera",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        for _ in range(GRAB_PAIR_MAX_ATTEMPTS):
            if not cam1.grab_only() or not cam2.grab_only():
                continue

            frame1 = cam1.retrieve_frame()
            frame2 = cam2.retrieve_frame()
            if frame1 is not None and frame2 is not None:
                return frame1, frame2

        return None, None

    @staticmethod
    def recover_pair(
        cam1: "RGBCamera",
        cam2: "RGBCamera",
        settle_s: float = 2.0,
    ) -> bool:
        ok1 = cam1.recover(settle_s)
        time.sleep(0.5)
        ok2 = cam2.recover(settle_s)
        return ok1 and ok2

    def info(self) -> dict:
        return {
            "name": self.name,
            "device": self.device,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }
