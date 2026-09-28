from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from touch_controls import Debouncer


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@unittest.skipUnless(os.environ.get("DISPLAY"), "needs a display")
class CaptureGuiTouchTest(unittest.TestCase):
    def setUp(self):
        from gui import CaptureGUI
        self.tmp = tempfile.TemporaryDirectory()
        self.gui = CaptureGUI(Path(self.tmp.name), "800x600", touch=True)
        self.clock = FakeClock()
        self.gui._save_gate = Debouncer(1.0, self.clock)

    def tearDown(self):
        self.gui.close()
        self.tmp.cleanup()

    def test_a_tap_queued_during_a_slow_save_is_ignored_after_it_finishes(self):
        self.gui.request_save()
        self.assertTrue(self.gui.consume_save_request())
        # ... save_capture runs for seconds; a second tap is delivered only once it ends ...
        self.gui.note_capture_saved(Path("20260925_101500"))
        self.gui.request_save()
        self.assertFalse(self.gui.consume_save_request())

    def test_a_deliberate_second_capture_a_moment_later_works(self):
        self.gui.request_save()
        self.gui.consume_save_request()
        self.gui.note_capture_saved(Path("20260925_101500"))
        self.clock.now += 1.2
        self.gui.request_save()
        self.assertTrue(self.gui.consume_save_request())

    def test_touch_mode_has_no_open_folder_button(self):
        self.gui.pump()
        texts = []

        def walk(widget):
            for child in widget.winfo_children():
                try:
                    texts.append(str(child.cget("text")))
                except Exception:
                    pass
                walk(child)
        walk(self.gui.root)
        self.assertIn("Take photo", texts)
        self.assertNotIn("Open Output Folder", texts)


@unittest.skipUnless(os.environ.get("DISPLAY"), "needs a display")
class CaptureGuiDesktopUnchangedTest(unittest.TestCase):
    def test_desktop_mode_does_not_debounce_and_keeps_its_buttons(self):
        from gui import CaptureGUI
        with tempfile.TemporaryDirectory() as tmp:
            gui = CaptureGUI(Path(tmp), "800x600")
            try:
                gui.note_capture_saved(Path("x"))
                gui.request_save()
                self.assertTrue(gui.consume_save_request())
            finally:
                gui.close()


if __name__ == "__main__":
    unittest.main()
