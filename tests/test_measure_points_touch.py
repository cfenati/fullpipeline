from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_points
from tests.helpers import HeadlessCv2, synthetic_rig
from touch_controls import ButtonBar, bar_rows

WINDOW = (800, 600)          # panels 400 wide; touch: 330 tall + 150 status + 120 bar
BAR_TOP = 330 + measure_points.STATUS_HEIGHT


def bar_press(headless: HeadlessCv2, key: str, event: int = cv2.EVENT_LBUTTONDOWN):
    for button, (x0, y0, x1, y1) in ButtonBar(800, bar_rows(False)).rects():
        if button.key == key:
            return headless.mouse(event, (x0 + x1) // 2, BAR_TOP + (y0 + y1) // 2)
    raise KeyError(key)


def run(headless: HeadlessCv2, touch: bool):
    image = np.random.RandomState(0).randint(0, 255, (480, 640, 3), np.uint8)
    with headless.patched():
        return measure_points.run_interactive(
            image, image, synthetic_rig(), (0.05, 0.5), 8, WINDOW, 0, 0.15, touch=touch)


class TouchRunInteractiveTest(unittest.TestCase):
    def test_tap_then_place_records_points_and_finish_needs_two_taps(self):
        h = HeadlessCv2()
        h.script = (h.tap(200, 150) + [bar_press(h, "place")]      # A, point 0
                    + h.tap(600, 150) + [bar_press(h, "place")]    # B, point 0
                    + h.tap(260, 200) + [bar_press(h, "place")]    # A, point 1
                    + h.tap(660, 200) + [bar_press(h, "place")]    # B, point 1
                    + [bar_press(h, "finish"), bar_press(h, "finish")])
        result = run(h, touch=True)
        self.assertEqual(len(result["clicks_a"]), 2)
        self.assertEqual(h.frames[0].shape, (600, 800, 3))  # window size unchanged

    def test_a_single_finish_tap_does_not_end_the_session(self):
        h = HeadlessCv2()
        h.script = [bar_press(h, "finish")]
        with self.assertRaises(AssertionError):  # the loop was still running when the script ran dry
            run(h, touch=True)

    def test_placing_in_the_wrong_panel_records_nothing(self):
        h = HeadlessCv2()
        h.script = (h.tap(600, 150) + [bar_press(h, "place")]
                    + [bar_press(h, "finish"), bar_press(h, "finish")])
        self.assertEqual(run(h, touch=True), {})

    def test_undo_button_removes_the_last_point(self):
        h = HeadlessCv2()
        h.script = (h.tap(200, 150) + [bar_press(h, "place")] + h.tap(600, 150) + [bar_press(h, "place")]
                    + h.tap(260, 200) + [bar_press(h, "place")] + h.tap(660, 200) + [bar_press(h, "place")]
                    + h.tap(300, 250) + [bar_press(h, "place")] + h.tap(700, 250) + [bar_press(h, "place")]
                    + [bar_press(h, "undo"), bar_press(h, "finish"), bar_press(h, "finish")])
        self.assertEqual(len(run(h, touch=True)["clicks_a"]), 2)

    def test_keyboard_and_mouse_path_is_unchanged_without_touch(self):
        h = HeadlessCv2()
        down = cv2.EVENT_LBUTTONDOWN
        h.script = [h.mouse(down, 200, 150), h.mouse(down, 600, 150),
                    h.mouse(down, 260, 200), h.mouse(down, 660, 200), h.key(ord("q"))]
        result = run(h, touch=False)
        self.assertEqual(len(result["clicks_a"]), 2)
        self.assertEqual(h.frames[0].shape, (600, 800, 3))


class BuildStatusLinesTest(unittest.TestCase):
    def _state(self, pending=None):
        return {"text_mode": False, "clicks_a": [], "pending_a": pending,
                "consecutive": [], "linked": True}

    def test_non_touch_lines_are_unchanged(self):
        lines = measure_points.build_status_lines(self._state(), None)
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "point 0: click a feature in camera A")
        self.assertIn("wheel = zoom", lines[2])

    def test_non_touch_on_status_still_replaces_everything(self):
        self.assertEqual(measure_points.build_status_lines(self._state(), lambda: ["x"]), ["x"])

    def test_touch_lines_name_the_next_camera_and_drop_keyboard_hints(self):
        lines = measure_points.build_status_lines(self._state(), None, touch=True)
        self.assertIn("tap a feature in camera A", lines[0])
        self.assertNotIn("wheel", " ".join(lines))
        pending = measure_points.build_status_lines(self._state([1.0, 2.0]), None, touch=True)
        self.assertIn("camera B", pending[0])

    def test_touch_prepends_the_prompt_to_a_callers_status_lines(self):
        lines = measure_points.build_status_lines(self._state(), lambda: ["Rim: 1 of 5"], touch=True)
        self.assertEqual(lines[1:], ["Rim: 1 of 5"])
        self.assertIn("camera A", lines[0])


if __name__ == "__main__":
    unittest.main()
