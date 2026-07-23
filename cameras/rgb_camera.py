import time
from typing import Optional, Tuple

import cv2
import numpy as np

STARTUP_FRAMES_TO_DISCARD = 8
GRAB_PAIR_MAX_ATTEMPTS = 8


class RGBCamera:
    def __init__(
        self,
        device: str,
        name: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        self.device = device
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
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
        self._discard_startup_frames()

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

    @staticmethod
    def recover_failed(
        cam1: "RGBCamera",
        cam2: "RGBCamera",
        frame1: Optional[np.ndarray],
        frame2: Optional[np.ndarray],
        settle_s: float = 1.0,
    ) -> bool:
        """Reopen only the RGB camera(s) that failed to grab."""
        if frame1 is not None and frame2 is not None:
            return True

        ok = True
        if frame1 is None:
            ok = cam1.recover(settle_s) and ok
        if frame2 is None:
            if frame1 is None:
                time.sleep(0.3)
            ok = cam2.recover(settle_s) and ok
        return ok

    def info(self) -> dict:
        return {
            "name": self.name,
            "device": self.device,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }
