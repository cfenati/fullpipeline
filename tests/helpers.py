"""Shared test helpers: a synthetic stereo rig and a headless stand-in for cv2's window calls."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, List
from unittest import mock

import cv2
import numpy as np

from calibration.stereo import StereoExtrinsics


def synthetic_rig() -> StereoExtrinsics:
    """Two identical pinhole cameras 60 mm apart, no distortion, no rotation."""
    k = np.array([[1000.0, 0, 320], [0, 1000.0, 240], [0, 0, 1]])
    r = np.eye(3)
    t = np.array([-0.06, 0.0, 0.0])
    skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return StereoExtrinsics(
        name_a="a", name_b="b", image_size_a=(640, 480), image_size_b=(640, 480),
        camera_matrix_a=k, distortion_a=np.zeros(5),
        camera_matrix_b=k.copy(), distortion_b=np.zeros(5),
        R=r, T=t, essential=skew @ r, fundamental=np.zeros((3, 3)),
        reprojection_error_px=0.1, views_used=1, points_used=1,
    )


class HeadlessCv2:
    """Stands in for OpenCV's window calls so run_interactive runs with no display.

    ``script`` steps run one per ``waitKey`` call; each returns the key code to hand back
    (-1 for "no key"). A step may fire the registered mouse callback first. If the script
    runs dry before the session finishes, the test fails loudly instead of hanging.
    """

    def __init__(self) -> None:
        self.callback = None
        self.frames: List[np.ndarray] = []
        self.script: List[Callable[[], int]] = []
        self.destroyed = False

    def _wait_key(self, _delay: int) -> int:
        if self.destroyed:
            return -1
        if not self.script:
            raise AssertionError("scripted interaction ended before the session finished")
        return self.script.pop(0)()

    @contextmanager
    def patched(self) -> Iterator["HeadlessCv2"]:
        with mock.patch.object(cv2, "namedWindow"), \
             mock.patch.object(cv2, "setMouseCallback",
                               lambda _w, cb: setattr(self, "callback", cb)), \
             mock.patch.object(cv2, "imshow", lambda _w, f: self.frames.append(f.copy())), \
             mock.patch.object(cv2, "waitKey", self._wait_key), \
             mock.patch.object(cv2, "getWindowProperty", lambda *_a: 1), \
             mock.patch.object(cv2, "destroyAllWindows",
                               lambda: setattr(self, "destroyed", True)):
            yield self

    def mouse(self, event: int, x: int, y: int) -> Callable[[], int]:
        def step() -> int:
            self.callback(event, x, y, 0, None)
            return -1
        return step

    def tap(self, x: int, y: int) -> List[Callable[[], int]]:
        return [self.mouse(cv2.EVENT_LBUTTONDOWN, x, y), self.mouse(cv2.EVENT_LBUTTONUP, x, y)]

    def key(self, code: int) -> Callable[[], int]:
        return lambda: code
