from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from capture_pipeline import unique_session_dir


class UniqueSessionDirTest(unittest.TestCase):
    def test_second_and_third_capture_in_one_second_get_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = unique_session_dir(root, "20260925_101500")
            self.assertEqual(first.name, "20260925_101500")
            first.mkdir()
            second = unique_session_dir(root, "20260925_101500")
            self.assertEqual(second.name, "20260925_101500_2")
            second.mkdir()
            self.assertEqual(unique_session_dir(root, "20260925_101500").name, "20260925_101500_3")

    def test_different_seconds_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "20260925_101500").mkdir()
            self.assertEqual(unique_session_dir(root, "20260925_101501").name, "20260925_101501")


if __name__ == "__main__":
    unittest.main()
