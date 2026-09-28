from __future__ import annotations

import unittest

import cv2
import numpy as np

from measure_points import Panel
from touch_controls import BAR_HEIGHT, ButtonBar, TouchController, bar_rows, draw_bar

PANEL = (400, 330)
STATUS = 150


def make_state():
    image = np.zeros((480, 640, 3), np.uint8)
    return {"panel_a": Panel(image, PANEL), "panel_b": Panel(image, PANEL),
            "pending_a": None, "text_mode": False, "cursor": None}


class Rig:
    """A TouchController plus a recorder standing in for run_interactive's on_mouse."""

    def __init__(self, show_lock: bool = False) -> None:
        self.state = make_state()
        self.controller = TouchController(self.state, STATUS, show_lock=show_lock)
        self.inner_calls = []
        self._callback = self.controller.wrap(lambda *args: self.inner_calls.append(args))

    def mouse(self, event: int, x: int, y: int) -> None:
        self._callback(event, x, y, 0, None)

    def tap(self, x: int, y: int) -> None:
        self.mouse(cv2.EVENT_LBUTTONDOWN, x, y)
        self.mouse(cv2.EVENT_LBUTTONUP, x, y)

    def press(self, key: str, event: int = cv2.EVENT_LBUTTONDOWN) -> None:
        for button, (x0, y0, x1, y1) in self.controller.bar.rects():
            if button.key == key:
                self.mouse(event, (x0 + x1) // 2, self.controller.bar_top + (y0 + y1) // 2)
                return
        raise KeyError(key)


class ButtonBarTest(unittest.TestCase):
    def test_buttons_tile_each_row_without_gaps(self):
        bar = ButtonBar(800, bar_rows(True))
        for row_index in range(2):
            rects = [r for _b, r in bar.rects() if r[1] == row_index * 60]
            self.assertEqual(rects[0][0], 0)
            self.assertEqual(rects[-1][2], 800)
            for left, right in zip(rects, rects[1:]):
                self.assertEqual(left[2], right[0])

    def test_hit_finds_the_button_under_a_point_and_none_outside(self):
        bar = ButtonBar(800, bar_rows(False))
        button, (x0, y0, x1, y1) = bar.rects()[0]
        self.assertIs(bar.hit((x0 + x1) // 2, (y0 + y1) // 2), button)
        self.assertIsNone(bar.hit(800, 0))
        self.assertIsNone(bar.hit(0, bar.height))

    def test_lock_button_only_when_requested(self):
        keys = lambda show: {b.key for b, _r in ButtonBar(800, bar_rows(show)).rects()}
        self.assertIn("lock", keys(True))
        self.assertNotIn("lock", keys(False))

    def test_draw_bar_size_and_armed_state_is_visible(self):
        bar = ButtonBar(800, bar_rows(False))
        plain = draw_bar(bar, {})
        self.assertEqual(plain.shape, (BAR_HEIGHT, 800, 3))
        self.assertFalse(np.array_equal(plain, draw_bar(bar, {"finish": True})))


class TapAndDragTest(unittest.TestCase):
    def test_tap_sets_a_tentative_crosshair_without_recording(self):
        rig = Rig()
        rig.tap(200, 150)
        name, point = rig.controller.tentative
        self.assertEqual(name, "a")
        np.testing.assert_allclose(point, rig.state["panel_a"].to_image((200, 150)))
        self.assertEqual(rig.inner_calls, [])

    def test_tap_on_the_right_half_is_panel_b(self):
        rig = Rig()
        rig.tap(600, 150)
        name, point = rig.controller.tentative
        self.assertEqual(name, "b")
        np.testing.assert_allclose(point, rig.state["panel_b"].to_image((200, 150)))

    def test_tap_in_the_status_strip_is_ignored(self):
        rig = Rig()
        rig.tap(200, 340)  # panel height is 330, so this is the status strip
        self.assertIsNone(rig.controller.tentative)

    def test_drag_pans_the_touched_panel_and_leaves_no_crosshair(self):
        rig = Rig()
        before_a = rig.state["panel_a"].offset.copy()
        before_b = rig.state["panel_b"].offset.copy()
        rig.mouse(cv2.EVENT_LBUTTONDOWN, 200, 150)
        rig.mouse(cv2.EVENT_MOUSEMOVE, 230, 150)
        rig.mouse(cv2.EVENT_LBUTTONUP, 230, 150)
        moved = before_a - rig.state["panel_a"].offset
        self.assertAlmostEqual(moved[0], 30 / rig.state["panel_a"].scale)
        np.testing.assert_allclose(rig.state["panel_b"].offset, before_b)
        self.assertIsNone(rig.controller.tentative)

    def test_text_mode_ignores_touches(self):
        rig = Rig()
        rig.state["text_mode"] = True
        rig.tap(200, 150)
        self.assertIsNone(rig.controller.tentative)


class PlaceTest(unittest.TestCase):
    def test_place_forwards_a_left_click_at_the_crosshair(self):
        rig = Rig()
        rig.tap(200, 150)
        rig.press("place")
        self.assertEqual(len(rig.inner_calls), 1)
        event, x, y, _flags, _param = rig.inner_calls[0]
        self.assertEqual((event, x, y), (cv2.EVENT_LBUTTONDOWN, 200, 150))
        self.assertIsNone(rig.controller.tentative)

    def test_place_in_b_uses_canvas_coordinates_once_a_is_pending(self):
        rig = Rig()
        rig.state["pending_a"] = [1.0, 2.0]
        rig.tap(600, 150)
        rig.press("place")
        _event, x, y, _flags, _param = rig.inner_calls[0]
        self.assertEqual((x, y), (600, 150))

    def test_place_in_the_wrong_panel_is_refused_and_keeps_the_crosshair(self):
        rig = Rig()
        rig.tap(600, 150)  # B, but nothing is pending in A yet
        rig.press("place")
        self.assertEqual(rig.inner_calls, [])
        self.assertIsNotNone(rig.controller.tentative)

    def test_place_without_a_crosshair_does_nothing(self):
        rig = Rig()
        rig.press("place")
        self.assertEqual(rig.inner_calls, [])

    def test_place_is_refused_when_the_crosshair_was_panned_off_screen(self):
        rig = Rig()
        rig.tap(200, 150)
        rig.state["panel_a"].pan((-5000, 0))
        rig.press("place")
        self.assertEqual(rig.inner_calls, [])
        self.assertIsNotNone(rig.controller.tentative)


class NudgeUndoZoomTest(unittest.TestCase):
    def test_arrows_nudge_one_screen_pixel(self):
        rig = Rig()
        rig.tap(200, 150)
        scale = rig.state["panel_a"].scale
        _n, start = rig.controller.tentative
        start = start.copy()
        rig.press("right")
        rig.press("down")
        _n, end = rig.controller.tentative
        self.assertAlmostEqual(end[0] - start[0], 1 / scale)
        self.assertAlmostEqual(end[1] - start[1], 1 / scale)
        rig.press("left")
        rig.press("up")
        _n, back = rig.controller.tentative
        np.testing.assert_allclose(back, start)

    def test_undo_clears_the_crosshair_before_sending_a_key(self):
        rig = Rig()
        rig.tap(200, 150)
        rig.press("undo")
        self.assertIsNone(rig.controller.tentative)
        self.assertEqual(rig.controller.pop_key(), -1)
        rig.press("undo")
        self.assertEqual(rig.controller.pop_key(), ord("u"))

    def test_cancel_clears_the_crosshair(self):
        rig = Rig()
        rig.tap(200, 150)
        rig.press("cancel")
        self.assertIsNone(rig.controller.tentative)

    def test_fit_and_lock_send_their_keys(self):
        rig = Rig(show_lock=True)
        rig.press("fit")
        rig.press("lock")
        self.assertEqual(rig.controller.pop_key(), ord("0"))
        self.assertEqual(rig.controller.pop_key(), ord("n"))
        self.assertEqual(rig.controller.pop_key(), -1)
        with self.assertRaises(KeyError):
            Rig(show_lock=False).press("lock")

    def test_zoom_buttons_scale_the_last_touched_panel_only(self):
        rig = Rig()
        rig.tap(600, 150)  # touch B
        a0, b0 = rig.state["panel_a"].scale, rig.state["panel_b"].scale
        rig.press("zoom_in")
        self.assertAlmostEqual(rig.state["panel_b"].scale, b0 * 1.25)
        self.assertAlmostEqual(rig.state["panel_a"].scale, a0)
        rig.press("zoom_out")
        self.assertAlmostEqual(rig.state["panel_b"].scale, b0)

    def test_zoom_keeps_the_crosshair_fixed_on_screen(self):
        rig = Rig()
        rig.tap(200, 150)
        rig.press("zoom_in")
        _n, point = rig.controller.tentative
        self.assertEqual(rig.state["panel_a"].to_screen(point), (200, 150))


class ConfirmTest(unittest.TestCase):
    def test_reset_and_finish_need_two_taps(self):
        rig = Rig()
        rig.press("reset")
        self.assertEqual(rig.controller.pop_key(), -1)
        rig.press("reset")
        self.assertEqual(rig.controller.pop_key(), ord("r"))
        rig.press("finish")
        self.assertEqual(rig.controller.pop_key(), -1)
        rig.press("finish")
        self.assertEqual(rig.controller.pop_key(), ord("q"))

    def test_pressing_another_button_disarms_a_pending_confirm(self):
        rig = Rig()
        rig.press("finish")
        rig.press("fit")
        self.assertEqual(rig.controller.pop_key(), ord("0"))
        rig.press("finish")
        self.assertEqual(rig.controller.pop_key(), -1)

    def test_a_quick_second_tap_arrives_as_double_click_and_still_confirms(self):
        rig = Rig()
        rig.press("finish")
        rig.press("finish", event=cv2.EVENT_LBUTTONDBLCLK)
        self.assertEqual(rig.controller.pop_key(), ord("q"))


class SyncAndComposeTest(unittest.TestCase):
    def test_sync_points_the_loupe_cursor_at_the_crosshair(self):
        rig = Rig()
        rig.tap(600, 150)
        rig.controller.sync()
        self.assertEqual(rig.state["cursor"], (600, 150))

    def test_compose_appends_the_bar_and_keeps_the_width(self):
        rig = Rig()
        frame = np.zeros((PANEL[1] + STATUS, 800, 3), np.uint8)
        out = rig.controller.compose(frame)
        self.assertEqual(out.shape, (PANEL[1] + STATUS + BAR_HEIGHT, 800, 3))

    def test_compose_draws_the_crosshair_where_the_operator_tapped(self):
        rig = Rig()
        rig.tap(200, 150)
        frame = np.zeros((PANEL[1] + STATUS, 800, 3), np.uint8)
        out = rig.controller.compose(frame)
        self.assertTrue(out[150, 200].any())


if __name__ == "__main__":
    unittest.main()
