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
