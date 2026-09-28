from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from menu_logic import (
    PROJECT_ROOT, STAGE_ORDER, STAGES, UNNAMED_FOLDER, build_command, captures_dir_from,
    find_sessions, group_by_folder, hidden_capture_dirs, latest_session, paginate,
    sanitize_session_name, storage_problem, window_for_screen,
)


def make_session(root: Path, *parts: str, mtime: float = 1000.0) -> Path:
    path = root.joinpath(*parts)
    path.mkdir(parents=True)
    for name in ("rgb_cam1.jpg", "rgb_cam2.jpg"):
        (path / name).write_bytes(b"x")
    os.utime(path, (mtime, mtime))
    return path


class FindSessionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_session(self.root, "20260916_154103", mtime=1000)                 # unnamed capture
        make_session(self.root, "wound_a", "20260916_100000", mtime=2000)
        make_session(self.root, "wound_a", "20260917_090000", mtime=3000)
        make_session(self.root, "stereo", "s1", mtime=9000)                     # hidden folder
        (self.root / "empty").mkdir()
        half = self.root / "half" / "x"
        half.mkdir(parents=True)
        (half / "rgb_cam1.jpg").write_bytes(b"x")                               # only one camera

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_sessions_newest_first_and_skips_hidden_and_incomplete(self):
        sessions = find_sessions(self.root, hidden=[self.root / "stereo"])
        self.assertEqual([s.name for s in sessions],
                         ["20260917_090000", "20260916_100000", "20260916_154103"])

    def test_folder_is_blank_for_unnamed_captures(self):
        by_name = {s.name: s.folder for s in find_sessions(self.root)}
        self.assertEqual(by_name["20260916_154103"], UNNAMED_FOLDER)
        self.assertEqual(by_name["20260917_090000"], "wound_a")

    def test_latest_session_and_empty(self):
        sessions = find_sessions(self.root, hidden=[self.root / "stereo"])
        self.assertEqual(latest_session(sessions).name, "20260917_090000")
        self.assertIsNone(latest_session([]))

    def test_groups_are_ordered_by_their_newest_session(self):
        groups = group_by_folder(find_sessions(self.root, hidden=[self.root / "stereo"]))
        self.assertEqual([name for name, _ in groups], ["wound_a", UNNAMED_FOLDER])
        self.assertEqual(len(groups[0][1]), 2)

    def test_missing_captures_dir_is_empty(self):
        self.assertEqual(find_sessions(self.root / "nope"), [])

    def test_display_formats_a_timestamp_name(self):
        sessions = find_sessions(self.root, hidden=[self.root / "stereo"])
        unnamed = [s for s in sessions if s.folder == UNNAMED_FOLDER][0]
        self.assertEqual(unnamed.display, "16 Sep 2026  15:41:03")
        named = [s for s in sessions if s.folder == "wound_a"][0]
        self.assertTrue(named.display.startswith("wound_a / "))


class ConfigHelpersTest(unittest.TestCase):
    def test_hidden_capture_dirs_come_from_the_calibration_config(self):
        config = {"geometric_calibration": {
            "intrinsics_captures": "captures/stereo",
            "stereo_captures": ["captures/stereo", "captures/other"],
            "cross_validation_captures": "captures/cross-validation",
        }}
        hidden = hidden_capture_dirs(config, Path("/proj"))
        self.assertIn(Path("/proj/captures/stereo"), hidden)
        self.assertIn(Path("/proj/captures/cross-validation"), hidden)
        self.assertIn(Path("/proj/captures/other"), hidden)

    def test_no_calibration_config_hides_nothing(self):
        self.assertEqual(hidden_capture_dirs({}, Path("/proj")), [])

    def test_an_absolute_output_dir_is_kept_and_the_default_is_project_relative(self):
        self.assertEqual(captures_dir_from({"output_dir": "/mnt/usbdrive/captures"}, Path("/proj")),
                         Path("/mnt/usbdrive/captures"))
        self.assertEqual(captures_dir_from({}, Path("/proj")), Path("/proj/captures"))


class StorageProblemTest(unittest.TestCase):
    def test_an_existing_folder_or_one_whose_parent_exists_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(storage_problem(Path(tmp)))
            self.assertIsNone(storage_problem(Path(tmp) / "captures"))  # will be created there

    def test_a_missing_mount_is_reported_with_the_path_and_the_usb_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "usbdrive" / "captures"
            message = storage_problem(missing)
            self.assertIn(str(missing), message)
            self.assertIn("USB drive", message)


class NameAndWindowTest(unittest.TestCase):
    def test_sanitize_keeps_only_safe_characters_and_caps_length(self):
        self.assertEqual(sanitize_session_name("wound a/../x!"), "woundax")
        self.assertEqual(sanitize_session_name("Pig_skin-2"), "Pig_skin-2")
        self.assertEqual(len(sanitize_session_name("a" * 100)), 40)

    def test_window_fits_inside_the_screen_with_a_floor(self):
        self.assertEqual(window_for_screen(1920, 1080), (1880, 940))
        self.assertEqual(window_for_screen(1280, 720), (1240, 580))
        self.assertEqual(window_for_screen(800, 480), (960, 540))


class PaginateTest(unittest.TestCase):
    def test_slices_and_clamps(self):
        items = list(range(10))
        first = paginate(items, 0, 4)
        self.assertEqual((first.items, first.index, first.count), ([0, 1, 2, 3], 0, 3))
        last = paginate(items, 99, 4)
        self.assertEqual((last.items, last.index), ([8, 9], 2))
        self.assertEqual(paginate(items, -5, 4).index, 0)

    def test_empty_list_is_one_empty_page(self):
        page = paginate([], 0, 6)
        self.assertEqual((page.items, page.index, page.count), ([], 0, 1))


class BuildCommandTest(unittest.TestCase):
    ROOT = Path("/proj")

    def test_capture_with_and_without_a_name(self):
        named = build_command(STAGES["capture"], "py", self.ROOT, name="wound_a")
        self.assertEqual(named, ["py", "/proj/capture_pipeline.py", "--touch", "--output", "wound_a"])
        self.assertEqual(build_command(STAGES["capture"], "py", self.ROOT),
                         ["py", "/proj/capture_pipeline.py", "--touch"])

    def test_calibrate_takes_no_session(self):
        self.assertEqual(build_command(STAGES["calibrate"], "py", self.ROOT),
                         ["py", "/proj/calibrate_live.py", "--touch"])

    def test_measure_tools_get_session_and_window(self):
        cmd = build_command(STAGES["depth"], "py", self.ROOT,
                            session=Path("/data/s1"), window=(1240, 580))
        self.assertEqual(cmd, ["py", "/proj/measure_wound_depth.py", "--touch",
                               "--session", "/data/s1", "--window", "1240", "580"])

    def test_register_has_no_touch_flag_and_no_window(self):
        cmd = build_command(STAGES["register"], "py", self.ROOT,
                            session=Path("/data/s1"), window=(1240, 580))
        self.assertEqual(cmd, ["py", "/proj/register_features.py", "--session", "/data/s1"])

    def test_a_session_stage_without_a_session_is_an_error(self):
        with self.assertRaises(ValueError):
            build_command(STAGES["length"], "py", self.ROOT)


class StageTableTest(unittest.TestCase):
    def test_every_stage_script_exists_and_touch_stages_accept_the_flag(self):
        for key in STAGE_ORDER:
            stage = STAGES[key]
            script = PROJECT_ROOT / stage.script
            self.assertTrue(script.is_file(), f"{stage.script} is missing")
            if stage.touch:
                self.assertIn('"--touch"', script.read_text(encoding="utf-8"),
                              f"{stage.script} has no --touch flag")
            if stage.window:
                self.assertIn('"--window"', script.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
