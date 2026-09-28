from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from menu_logic import STAGES, Report


@unittest.skipUnless(os.environ.get("DISPLAY"), "needs a display")
class MenuSmokeTest(unittest.TestCase):
    def setUp(self):
        from menu import MenuApp
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "config.yaml").write_text("output_dir: captures\n", encoding="utf-8")
        for stamp in ("20260916_100000", "20260917_090000"):
            session = root / "captures" / "wound_a" / stamp
            session.mkdir(parents=True)
            for camera in ("rgb_cam1.jpg", "rgb_cam2.jpg"):
                cv2.imwrite(str(session / camera), np.full((60, 80, 3), 128, np.uint8))
        self.root_dir = root
        self.app = MenuApp(project_root=root, maximize=False)

    def tearDown(self):
        self.app.root.destroy()
        self.tmp.cleanup()

    def _texts(self):
        out = []

        def walk(widget):
            for child in widget.winfo_children():
                try:
                    out.append(str(child.cget("text")))
                except Exception:
                    pass
                walk(child)
        walk(self.app.body)
        return out

    def test_home_has_a_tile_per_stage_plus_output_folder_and_quit(self):
        self.app.root.update()
        texts = self._texts()
        for key in ("capture", "calibrate", "depth", "length", "register"):
            self.assertIn(STAGES[key].title, texts)
        self.assertIn("Output folder", texts)
        self.assertIn("Quit", texts)

    def test_every_screen_builds(self):
        app = self.app
        app.show_session_picker(STAGES["depth"])
        app.root.update()
        self.assertTrue(any(t.startswith("Latest capture") for t in self._texts()))
        app.show_session_picker(STAGES["depth"], folder="wound_a")
        app.root.update()
        app.show_name_keyboard(STAGES["capture"])
        app.root.update()
        self.assertIn("Skip (auto-name)", self._texts())
        app.show_error(STAGES["capture"], "An RGB camera could not be opened.", "log line")
        app.root.update()
        self.assertIn("Retry", self._texts())
        report = self.root_dir / "report.txt"
        report.write_text("DEPTH 3.0 mm\n", encoding="utf-8")
        app.show_result(STAGES["depth"], Report(report, None))
        app.root.update()
        app.show_result(STAGES["depth"], None)
        app.root.update()
        self.assertIn("Done", self._texts())

    def test_empty_captures_shows_a_message_not_a_crash(self):
        from menu import MenuApp
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text("output_dir: captures\n", encoding="utf-8")
            app = MenuApp(project_root=Path(tmp), maximize=False)
            try:
                app.show_session_picker(STAGES["length"])
                app.root.update()
                texts = []

                def walk(widget):
                    for child in widget.winfo_children():
                        try:
                            texts.append(str(child.cget("text")))
                        except Exception:
                            pass
                        walk(child)
                walk(app.body)
                self.assertIn("No captures yet. Use Take images first.", texts)
            finally:
                app.root.destroy()

    def test_a_missing_usb_drive_is_reported_instead_of_being_silently_created(self):
        from menu import MenuApp
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text(
                f"output_dir: {Path(tmp) / 'unplugged' / 'captures'}\n", encoding="utf-8")
            app = MenuApp(project_root=Path(tmp), maximize=False)
            try:
                app.choose(STAGES["capture"])
                app.root.update()
                self.assertFalse((Path(tmp) / "unplugged").exists())
                texts = []

                def walk(widget):
                    for child in widget.winfo_children():
                        try:
                            texts.append(str(child.cget("text")))
                        except Exception:
                            pass
                        walk(child)
                walk(app.body)
                self.assertTrue(any("USB drive" in t for t in texts))
            finally:
                app.root.destroy()


if __name__ == "__main__":
    unittest.main()
