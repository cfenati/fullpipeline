from __future__ import annotations

import unittest

from touch_controls import ConfirmGate, Debouncer, GestureTracker


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class ConfirmGateTest(unittest.TestCase):
    def test_second_press_inside_timeout_confirms(self):
        clock = FakeClock()
        gate = ConfirmGate(3.0, clock)
        self.assertFalse(gate.press())
        self.assertTrue(gate.armed())
        clock.now += 2.0
        self.assertTrue(gate.press())
        self.assertFalse(gate.armed())

    def test_press_after_timeout_rearms_instead_of_confirming(self):
        clock = FakeClock()
        gate = ConfirmGate(3.0, clock)
        gate.press()
        clock.now += 3.5
        self.assertFalse(gate.armed())
        self.assertFalse(gate.press())
        self.assertTrue(gate.armed())

    def test_a_confirmation_is_consumed(self):
        gate = ConfirmGate(3.0, FakeClock())
        gate.press()
        gate.press()
        self.assertFalse(gate.press())

    def test_disarm_cancels_a_pending_confirmation(self):
        gate = ConfirmGate(3.0, FakeClock())
        gate.press()
        gate.disarm()
        self.assertFalse(gate.press())


class DebouncerTest(unittest.TestCase):
    def test_ready_until_marked_then_not_until_gap_passes(self):
        clock = FakeClock()
        debounce = Debouncer(1.0, clock)
        self.assertTrue(debounce.ready())
        debounce.mark()
        self.assertFalse(debounce.ready())
        clock.now += 0.5
        self.assertFalse(debounce.ready())
        clock.now += 0.5
        self.assertTrue(debounce.ready())


class GestureTrackerTest(unittest.TestCase):
    def test_small_movement_is_a_tap_at_the_release_point(self):
        tracker = GestureTracker(12)
        tracker.press(100, 100)
        self.assertIsNone(tracker.move(105, 103))
        self.assertEqual(tracker.release(105, 103), (105, 103))

    def test_large_movement_is_a_drag_with_incremental_deltas(self):
        tracker = GestureTracker(12)
        tracker.press(100, 100)
        self.assertEqual(tracker.move(130, 100), (30, 0))  # includes the threshold distance
        self.assertEqual(tracker.move(140, 110), (10, 10))
        self.assertIsNone(tracker.release(140, 110))

    def test_release_without_press_is_nothing(self):
        self.assertIsNone(GestureTracker().release(5, 5))

    def test_state_resets_after_a_drag(self):
        tracker = GestureTracker(12)
        tracker.press(0, 0)
        tracker.move(50, 0)
        tracker.release(50, 0)
        self.assertFalse(tracker.active)
        tracker.press(10, 10)
        self.assertEqual(tracker.release(10, 10), (10, 10))

    def test_move_before_press_is_ignored(self):
        self.assertIsNone(GestureTracker().move(9, 9))


if __name__ == "__main__":
    unittest.main()
