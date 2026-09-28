from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from tkinter import ttk


@unittest.skipUnless(os.environ.get("DISPLAY"), "needs a display")
class LiveGuiTouchTest(unittest.TestCase):
    def setUp(self):
        from live_gui import LiveCaptureGUI
        self.tmp = tempfile.TemporaryDirectory()
        self.gui = LiveCaptureGUI(Path(self.tmp.name), "rgb_cam1", "rgb_cam2", "900x700", touch=True)

    def tearDown(self):
        self.gui.close()
        self.tmp.cleanup()

    def _button_texts(self):
        texts = []

        def walk(widget):
            for child in widget.winfo_children():
                if isinstance(child, ttk.Button):
                    texts.append(str(child.cget("text")))
                walk(child)
        walk(self.gui.root)
        return texts

    def test_touch_toolbar_has_no_keyboard_hints_or_folder_button(self):
        texts = self._button_texts()
        self.assertIn("Discard Worst", texts)
        self.assertIn("Discard & Restart", texts)
        self.assertFalse([t for t in texts if "(r)" in t or "(space)" in t or "(q)" in t])
        self.assertNotIn("Open Captures Folder", texts)

    def test_destructive_button_needs_a_second_tap(self):
        called = []
        button = ttk.Button(self.gui.root, text="Discard")
        handler = self.gui._confirming(button, "Discard", lambda: called.append(1))
        handler()
        self.assertEqual(called, [])
        self.assertEqual(button.cget("text"), "Tap again to confirm")
        handler()
        self.assertEqual(called, [1])
        self.assertEqual(button.cget("text"), "Discard")

    def test_pause_label_has_no_keyboard_hint_in_touch_mode(self):
        self.gui.toggle_pause()
        self.assertEqual(self.gui._pause_button.cget("text"), "Resume Auto-Capture")

    def test_confirming_cancels_a_stale_revert_before_a_later_cycle_can_be_disturbed_by_it(self):
        """Regression test: a leftover root.after revert from a finished cycle
        must not survive to fire during a later cycle. Without cancellation,
        that stale callback can reset the label to plain text while a new
        cycle's first tap has already re-armed the gate - so the operator's
        very next tap, which looks like a fresh first tap, silently fires the
        destructive action instead of showing "Tap again to confirm" again.
        """
        scheduled = {}
        cancelled = []
        next_id = iter(range(1, 1000))

        def fake_after(_ms, fn):
            job_id = next(next_id)
            scheduled[job_id] = fn
            return job_id

        def fake_after_cancel(job_id):
            cancelled.append(job_id)
            scheduled.pop(job_id, None)

        def fire(job_id):
            """Simulate the real Tk event loop reaching a job's due time: a
            no-op if it was already cancelled, exactly like a real
            after_cancel'd job never calling back."""
            fn = scheduled.pop(job_id, None)
            if fn is not None:
                fn()

        self.gui.root.after = fake_after
        self.gui.root.after_cancel = fake_after_cancel

        calls = []
        button = ttk.Button(self.gui.root, text="Discard")
        handler = self.gui._confirming(button, "Discard", lambda: calls.append(1))

        handler()  # cycle 1: first tap arms
        self.assertEqual(len(scheduled), 1)
        first_job_id = next(iter(scheduled))

        handler()  # cycle 1: second tap confirms - must cancel the pending revert
        self.assertEqual(calls, [1])
        self.assertIn(first_job_id, cancelled)
        self.assertEqual(scheduled, {})

        handler()  # cycle 2: first tap arms again
        self.assertEqual(button.cget("text"), "Tap again to confirm")
        second_job_id = next(iter(scheduled))
        self.assertNotEqual(second_job_id, first_job_id)

        fire(first_job_id)  # the stale cycle-1 revert reaching its due time late
        self.assertEqual(
            button.cget("text"), "Tap again to confirm",
            "a stale revert from a finished cycle must not touch a later cycle's label",
        )

        handler()  # cycle 2: the genuine second tap
        self.assertEqual(calls, [1, 1])
        self.assertEqual(button.cget("text"), "Discard")


@unittest.skipUnless(os.environ.get("DISPLAY"), "needs a display")
class LiveGuiDesktopUnchangedTest(unittest.TestCase):
    def test_desktop_toolbar_keeps_its_hints(self):
        from live_gui import LiveCaptureGUI
        with tempfile.TemporaryDirectory() as tmp:
            gui = LiveCaptureGUI(Path(tmp), "rgb_cam1", "rgb_cam2", "900x700")
            try:
                gui.toggle_pause()
                self.assertEqual(gui._pause_button.cget("text"), "Resume Auto-Capture (space)")
            finally:
                gui.close()


if __name__ == "__main__":
    unittest.main()
