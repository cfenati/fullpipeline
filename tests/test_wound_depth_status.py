from __future__ import annotations

import contextlib
import io
import unittest

from measure_wound_depth import RimAndWoundSession

# mm, camera looking down +z: a flat rim at z=200 and one point 3 mm farther away (a recession)
POINTS = [[0, 0, 200], [10, 0, 200], [0, 10, 200], [10, 10, 200], [5, 5, 200], [5, 5, 203]]


class RimAndWoundStatusLinesTest(unittest.TestCase):
    def test_counts_the_rim_then_locks_then_reports_a_live_depth(self):
        session = RimAndWoundSession(neighbors=5)
        result = {"points_mm": POINTS}
        self.assertEqual(session.status_lines(),
                         ["Rim: 0 of 5 needed", "Tap rim points around the area"])
        with contextlib.redirect_stdout(io.StringIO()):
            for index in range(4):
                session.on_point(index, result)
            self.assertEqual(session.status_lines()[0], "Rim: 4 of 5 needed")
            session.on_point(4, result)
            self.assertEqual(session.status_lines()[0], "Rim: 5 points - tap Lock rim when done")
            session.on_advance()
            self.assertEqual(session.status_lines(),
                             ["Rim locked: 5 points", "Tap points to measure"])
            session.on_point(5, result)
        self.assertEqual(session.status_lines()[1], "point 1: depth 3.00 mm (live estimate)")

    def test_undo_clears_the_live_estimate(self):
        session = RimAndWoundSession(neighbors=5)
        result = {"points_mm": POINTS}
        with contextlib.redirect_stdout(io.StringIO()):
            for index in range(5):
                session.on_point(index, result)
            session.on_advance()
            session.on_point(5, result)
            session.on_undo(5)
        self.assertIsNone(session.last_live)
        self.assertEqual(session.status_lines()[1], "Tap points to measure")


if __name__ == "__main__":
    unittest.main()
