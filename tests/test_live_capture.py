from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from calibrate_live import default_min_corners, promote_to_canonical
from calibration.live_capture import Phase, PhaseFitResult, SequentialLiveCalibrationSession
from calibration.target_board import TargetBoard

BOARD = TargetBoard(squares_x=11, squares_y=8, square_size_m=0.004, marker_size_m=0.002933)


def make_session(tmp: str) -> SequentialLiveCalibrationSession:
    return SequentialLiveCalibrationSession(
        BOARD, Path(tmp) / "board.yaml", Path(tmp) / "captures", Path(tmp) / "results",
    )


class DefaultMinCornersTest(unittest.TestCase):
    def test_half_the_boards_corners(self):
        # 11x8 squares -> 10x7 interior corners = 70; half of that is 35.
        self.assertEqual(BOARD.total_corners, 70)
        self.assertEqual(default_min_corners(BOARD), 35)


class CanAdvanceTest(unittest.TestCase):
    def test_no_fit_yet_is_blocked_on_every_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = make_session(tmp)
            ok, reason = session.can_advance()
            self.assertFalse(ok)
            self.assertIn("Run a fit first", reason)

    def test_cam_a_advances_on_any_fit_regardless_of_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = make_session(tmp)
            session._results[Phase.CAM_A] = PhaseFitResult(
                phase=Phase.CAM_A, ready=False, view_count=5, warnings=["bad fit"],
            )
            ok, _reason = session.can_advance()
            self.assertTrue(ok)

    def test_cam_b_blocks_entering_stereo_on_imprecise_mono_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = make_session(tmp)
            session._phase_index = 1  # Phase.CAM_B
            session._results[Phase.CAM_A] = PhaseFitResult(
                phase=Phase.CAM_A, ready=True, view_count=20,
            )  # no .intrinsics -> "no successful fit"
            session._results[Phase.CAM_B] = PhaseFitResult(
                phase=Phase.CAM_B, ready=True, view_count=20,
            )
            ok, reason = session.can_advance()
            self.assertFalse(ok)
            self.assertIn("no successful fit", reason)

    def test_stereo_finish_is_blocked_when_the_stereo_fit_is_not_ready(self):
        """The gap this whole design closes: stereo_calibrate.py crashing (or any other
        stereo-phase warning) must not let the session reach is_done() and get promoted."""
        with tempfile.TemporaryDirectory() as tmp:
            session = make_session(tmp)
            session._phase_index = 2  # Phase.STEREO
            session._results[Phase.STEREO] = PhaseFitResult(
                phase=Phase.STEREO, ready=False, view_count=19,
                warnings=["stereo_calibrate.py exited with code -6 - extrinsics.json was "
                          "written, but report.txt/rig_pose.yaml/figures may be stale or "
                          "missing; check the console output above."],
            )
            ok, reason = session.can_advance()
            self.assertFalse(ok)
            self.assertIn("exited with code -6", reason)
            self.assertFalse(session.advance())
            self.assertFalse(session.is_done())

    def test_stereo_finish_is_allowed_when_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = make_session(tmp)
            session._phase_index = 2  # Phase.STEREO
            session._results[Phase.STEREO] = PhaseFitResult(
                phase=Phase.STEREO, ready=True, view_count=20,
            )
            ok, reason = session.can_advance()
            self.assertTrue(ok)
            self.assertEqual(reason, "")
            self.assertTrue(session.advance())
            self.assertTrue(session.is_done())


class PromoteToCanonicalTest(unittest.TestCase):
    def _make_run(self, root: Path, names: tuple) -> None:
        for name in names:
            (root / name).mkdir(parents=True)
            (root / name / "marker.txt").write_text(name)

    def test_first_promotion_copies_with_nothing_to_back_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "results" / "live_20260101_000000"
            canonical_root = Path(tmp) / "results"
            self._make_run(output_root, ("rgb_cam1", "rgb_cam2", "stereo_rgb_cam1_rgb_cam2"))

            backups = promote_to_canonical(
                "rgb_cam1", "rgb_cam2", output_root, canonical_root, "20260101_000000",
            )

            self.assertEqual(backups, {})
            for name in ("rgb_cam1", "rgb_cam2", "stereo_rgb_cam1_rgb_cam2"):
                self.assertEqual((canonical_root / name / "marker.txt").read_text(), name)
            # the run's own folder is untouched (copytree, not move)
            self.assertTrue((output_root / "rgb_cam1" / "marker.txt").exists())

    def test_second_promotion_backs_up_the_previous_canonical_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            canonical_root = Path(tmp) / "results"
            self._make_run(canonical_root, ("rgb_cam1", "rgb_cam2", "stereo_rgb_cam1_rgb_cam2"))
            (canonical_root / "rgb_cam1" / "marker.txt").write_text("old")

            new_run = canonical_root / "live_20260202_000000"
            self._make_run(new_run, ("rgb_cam1", "rgb_cam2", "stereo_rgb_cam1_rgb_cam2"))
            (new_run / "rgb_cam1" / "marker.txt").write_text("new")

            backups = promote_to_canonical(
                "rgb_cam1", "rgb_cam2", new_run, canonical_root, "20260202_000000",
            )

            self.assertEqual(
                set(backups), {"rgb_cam1", "rgb_cam2", "stereo_rgb_cam1_rgb_cam2"},
            )
            self.assertEqual((canonical_root / "rgb_cam1" / "marker.txt").read_text(), "new")
            backup_dir = canonical_root / "backup_before_live_20260202_000000"
            self.assertEqual((backup_dir / "rgb_cam1" / "marker.txt").read_text(), "old")

    def test_no_op_when_output_root_is_already_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            canonical_root = Path(tmp) / "results"
            self._make_run(canonical_root, ("rgb_cam1",))

            backups = promote_to_canonical(
                "rgb_cam1", "rgb_cam2", canonical_root, canonical_root, "20260101_000000",
            )

            self.assertEqual(backups, {})
            self.assertEqual((canonical_root / "rgb_cam1" / "marker.txt").read_text(), "rgb_cam1")


if __name__ == "__main__":
    unittest.main()
