from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from menu_logic import (
    GENERIC_FAILURE, StageRun, desktop_entry_text, find_newest_report, friendly_error,
    install_shortcut, tail_lines,
)


def wait_for_exit(run: StageRun, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code = run.poll()
        if code is not None:
            return code
        time.sleep(0.05)
    raise AssertionError("child did not exit in time")


class StageRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, code: str, **kwargs) -> StageRun:
        run = StageRun([sys.executable, "-c", code], self.root / "logs" / "x.log", self.root, **kwargs)
        run.start()
        return run

    def test_success_records_output_in_the_log(self):
        run = self._run("print('hello from stage')")
        self.assertEqual(wait_for_exit(run), 0)
        self.assertIn("hello from stage", run.log_text())
        self.assertFalse(run.stopped)

    def test_failure_returns_the_exit_code_and_captures_stderr(self):
        run = self._run("import sys; sys.stderr.write('boom\\n'); sys.exit(3)")
        self.assertEqual(wait_for_exit(run), 3)
        self.assertIn("boom", run.log_text())

    def test_stop_interrupts_a_long_running_child(self):
        run = self._run("import time; time.sleep(60)")
        run.stop()
        self.assertTrue(run.stopped)
        wait_for_exit(run)

    def test_stop_escalates_when_the_child_ignores_interrupts(self):
        run = self._run("import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)",
                        term_after_s=0.3, kill_after_s=0.6)
        time.sleep(0.5)  # let the child install its handler
        run.stop()
        self.assertNotEqual(wait_for_exit(run), 0)

    def test_elapsed_and_started_at(self):
        run = self._run("import time; time.sleep(0.3)")
        self.assertLess(abs(run.started_at - time.time()), 5)
        wait_for_exit(run)
        self.assertGreaterEqual(run.elapsed(), 0.0)


class FriendlyErrorTest(unittest.TestCase):
    def test_known_failures_get_plain_language(self):
        self.assertIn("RGB camera", friendly_error("Pipeline error: Cannot open rgb_cam1 at /dev/video0"))
        self.assertIn("thermal", friendly_error("RuntimeError: evo_irimager_usb_init failed with code -1"))
        self.assertIn("FLIR", friendly_error("RuntimeError: No FLIR/Spinnaker cameras found"))
        self.assertIn("calibrated", friendly_error("No stereo extrinsics at /x/extrinsics.json."))
        self.assertIn("missing a camera image", friendly_error("/x/s1 is missing rgb_cam2.jpg."))

    def test_unknown_failure_falls_back_to_the_generic_message(self):
        self.assertEqual(friendly_error("something odd"), GENERIC_FAILURE)

    def test_tail_lines(self):
        text = "\n".join(str(i) for i in range(50)) + "\n"
        self.assertEqual(tail_lines(text, 3), "47\n48\n49")
        self.assertEqual(tail_lines("", 3), "")


class FindNewestReportTest(unittest.TestCase):
    def _write(self, path: Path, mtime: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
        os.utime(path, (mtime, mtime))

    def test_picks_the_newest_report_since_the_run_started_with_its_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root / "old" / "wound_depth" / "report.txt", 500)
            self._write(root / "s1" / "wound_depth" / "report.txt", 2000)
            self._write(root / "s2" / "point_measurements" / "report.txt", 2500)
            self._write(root / "s2" / "point_measurements" / "measured.jpg", 2500)
            found = find_newest_report(root, since=1000)
            self.assertEqual(found.report_path, root / "s2" / "point_measurements" / "report.txt")
            self.assertEqual(found.image_path, root / "s2" / "point_measurements" / "measured.jpg")

    def test_registration_report_and_preview_are_recognised(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root / "s3" / "report_features.txt", 3000)
            self._write(root / "s3" / "preview_features.jpg", 3000)
            found = find_newest_report(root, since=1000)
            self.assertEqual(found.report_path.name, "report_features.txt")
            self.assertEqual(found.image_path.name, "preview_features.jpg")

    def test_nothing_new_or_missing_dir_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root / "s1" / "report.txt", 500)
            self.assertIsNone(find_newest_report(root, since=1000))
            self.assertIsNone(find_newest_report(root / "missing", since=0))


class ShortcutTest(unittest.TestCase):
    def test_desktop_entry_has_absolute_exec_and_path(self):
        text = desktop_entry_text("/venv/bin/python", Path("/proj/menu.py"), Path("/proj"))
        self.assertIn('Exec="/venv/bin/python" "/proj/menu.py"', text)
        self.assertIn("Path=/proj", text)
        self.assertIn("Terminal=false", text)
        self.assertTrue(text.startswith("[Desktop Entry]"))

    def test_install_writes_executable_launchers(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = install_shortcut("/venv/bin/python", Path("/proj/menu.py"), Path("/proj"), home=Path(tmp))
            self.assertEqual({p.parent.name for p in written}, {"Desktop", "applications"})
            for path in written:
                self.assertTrue(path.is_file())
                self.assertTrue(os.access(path, os.X_OK))


if __name__ == "__main__":
    unittest.main()
