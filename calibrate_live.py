#!/usr/bin/env python3
"""Guided live calibration: cam1 intrinsics -> cam2 intrinsics -> stereo pair.

Three sequential phases, each with its own dedicated capture target. Every
``--interval`` seconds, saves a frame - just this phase's camera(s), gated on
whether the ChArUco board is currently usable there - and keeps a live
coverage map so you can see which part of the frame still needs a view.
Reaching a phase's target auto-runs the same offline ``calibrate_cameras.py``/
``stereo_calibrate.py`` this repo already uses, once, not per capture; if the
result isn't ready, capture a few more (top-up) rather than reshooting the
whole phase.

    python calibrate_live.py
    python calibrate_live.py --camera-a rgb_cam1 --camera-b rgb_cam2 --interval 3.0
    python calibrate_live.py --no-preview

Needs the RGB cameras attached - see CLAUDE.md for hardware-dependent scripts.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_cameras import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    load_config as load_raw_config,
    resolve_path,
)
from calibration.opencv_calibrate import DEFAULT_MIN_CORNERS, CameraIntrinsics  # noqa: E402
from calibration.stereo import DEFAULT_MIN_SHARED_CORNERS, StereoExtrinsics  # noqa: E402
from calibration.live_capture import Phase, SequentialLiveCalibrationSession  # noqa: E402
from calibration.target_board import DEFAULT_BOARD_CONFIG, TargetBoard  # noqa: E402
from capture_pipeline import TerminalInput, load_config as load_pipeline_config, preview_frame  # noqa: E402
from cameras.rgb_camera import RGBCamera, controls_for  # noqa: E402

MAX_CONSECUTIVE_GRAB_FAILURES = 50
# Unlike capture_pipeline.py's raw preview loop, every iteration here also runs
# ChArUco detection (check_gate) - real CPU work, not just a redraw. Pacing to
# ~5 Hz keeps the gate feeling responsive to a moving board without spinning
# as fast as the camera can grab.
GATE_CHECK_INTERVAL_S = 0.2


def open_rgb_cameras(config: dict) -> Tuple[RGBCamera, RGBCamera]:
    """Open both RGB cameras once, for the whole run - phases don't reopen."""
    rgb_config = config["rgb"]
    cam1 = RGBCamera(
        device=rgb_config["cam1"], name="RGB Camera 1",
        width=rgb_config["width"], height=rgb_config["height"], fps=rgb_config["fps"],
        controls=controls_for(rgb_config, "cam1"),
    )
    cam2 = RGBCamera(
        device=rgb_config["cam2"], name="RGB Camera 2",
        width=rgb_config["width"], height=rgb_config["height"], fps=rgb_config["fps"],
        controls=controls_for(rgb_config, "cam2"),
    )
    cam1.open()
    time.sleep(0.3)
    cam2.open()
    return cam1, cam2


def review_and_prune(camera_a: str, camera_b: str, captures_dir: Path, output_root: Path) -> None:
    """Offer to delete unambiguously-bad sessions; point at manual pruning for judgment calls."""
    from prune_calibration import MIN_REMAINING_SESSIONS, UNDETECTED_ERROR, build_candidates

    fits: List[Tuple[str, object]] = []
    for camera in (camera_a, camera_b):
        path = output_root / camera / "intrinsics.json"
        if path.exists():
            fits.append((camera, CameraIntrinsics.load_json(path)))

    stereo_path = output_root / f"stereo_{camera_a}_{camera_b}" / "extrinsics.json"
    if stereo_path.exists():
        fits.append(("stereo", StereoExtrinsics.load_json(stereo_path)))

    if not fits:
        return

    no_board: Dict[str, str] = {}
    outliers: List[Tuple[str, float, str]] = []
    for name, fit in fits:
        for candidate in build_candidates(fit, captures_dir):
            if candidate.error_px == UNDETECTED_ERROR:
                no_board.setdefault(candidate.session, candidate.status)
            elif candidate.status == "rejected as outlier":
                outliers.append((candidate.session, candidate.error_px, name))

    if not no_board and not outliers:
        print("\nNo bad sessions detected - every captured session contributed to some fit.")
        return

    print("\n--- Bad-view review ---")
    if no_board:
        print(f"{len(no_board)} session(s) had no usable board in at least one camera:")
        for label, reason in sorted(no_board.items()):
            print(f"  {label}: {reason}")

        existing = [captures_dir / label for label in no_board if (captures_dir / label).is_dir()]
        total_sessions = sum(1 for path in captures_dir.iterdir() if path.is_dir())
        remaining = total_sessions - len(existing)
        if not existing:
            pass
        elif remaining < MIN_REMAINING_SESSIONS:
            print(f"\nNot offering to delete: would leave only {remaining} session(s), fewer than "
                  f"the {MIN_REMAINING_SESSIONS} needed for a usable calibration.")
        else:
            try:
                answer = input(f"\nDelete these {len(existing)} session folder(s) now? [y/N] ").strip().lower()
            except EOFError:
                answer = "n"
                print("\n[no interactive input available - skipping deletion]")
            if answer == "y":
                for path in existing:
                    shutil.rmtree(path)
                    print(f"  deleted {path}")
            else:
                print("  skipped - no folders deleted.")

    if outliers:
        print(f"\n{len(outliers)} high-residual outlier view(s) were excluded from the fit "
              "(kept on disk - a magnitude judgment call, not automatic):")
        for label, error, source in sorted(outliers, key=lambda row: row[1], reverse=True)[:10]:
            print(f"  {label:<20} {error:>8.3f} px   ({source})")
        print("\nReview these yourself and prune if you agree, e.g.:")
        print(f"  python prune_calibration.py --camera {camera_a} --captures {captures_dir} "
              f"--results {output_root} --count 5 --apply")


def parse_args() -> argparse.Namespace:
    calibration_config = load_raw_config().get("geometric_calibration", {}) or {}

    captures_default = calibration_config.get("intrinsics_captures")
    if not captures_default:
        stereo_captures = calibration_config.get("stereo_captures")
        if isinstance(stereo_captures, list) and stereo_captures:
            captures_default = stereo_captures[0]
    captures_default = captures_default or "captures/stereo"

    parser = argparse.ArgumentParser(
        description="Guided live calibration: cam1 intrinsics -> cam2 intrinsics -> stereo pair.",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml",
                         help="Path to pipeline config YAML (default: %(default)s).")
    parser.add_argument("--camera-a", default="rgb_cam1", help="Reference camera (default: %(default)s).")
    parser.add_argument("--camera-b", default="rgb_cam2", help="Second camera (default: %(default)s).")
    parser.add_argument("--in", "--captures", dest="captures", default=captures_default,
                         help=f"Parent capture directory (default: {captures_default}); this run "
                              "writes into its own fresh live_<timestamp> subfolder inside it, "
                              "never reading sessions from a previous run.")
    parser.add_argument("--board", default=calibration_config.get("board", DEFAULT_BOARD_CONFIG),
                         help="ChArUco board YAML (default: %(default)s).")
    parser.add_argument("--out", "--output", dest="output", default=None,
                         help="Results directory (default: geometric_calibration.output_dir/"
                              "live_<timestamp> - a fresh, non-destructive folder; pass this "
                              "explicitly to overwrite a specific location instead).")
    parser.add_argument("--interval", type=float, default=3.0,
                         help="Seconds between auto-capture attempts (default: %(default)s).")
    parser.add_argument("--min-corners", type=int, default=DEFAULT_MIN_CORNERS,
                         help="Minimum ChArUco corners per camera view (default: %(default)s).")
    parser.add_argument("--min-shared-corners", type=int, default=DEFAULT_MIN_SHARED_CORNERS,
                         help="Minimum corners shared by both cameras, stereo phase only "
                              "(default: %(default)s).")
    parser.add_argument("--mono-views", type=int, default=20,
                         help="Target view count for each intrinsics phase (default: %(default)s).")
    parser.add_argument("--stereo-views", type=int, default=15,
                         help="Target view count for the stereo phase (default: %(default)s).")
    parser.add_argument("--top-up", type=int, default=5,
                         help="Extra views to capture when 'capture more' is requested after a "
                              "not-ready fit (default: %(default)s).")
    parser.add_argument("--discard-count", type=int, default=5,
                         help="How many of the fit's own worst-reprojection-error views to remove "
                              "on 'discard worst' - keeps every other view, unlike 'discard & "
                              "restart' which wipes the whole phase. Capture does not resume on "
                              "its own afterward; 'capture more' (top-up) starts it again "
                              "(default: %(default)s).")
    parser.add_argument("--max-mono-rms-for-stereo", type=float, default=1.5,
                         help="Continuing from cam2 intrinsics into the stereo phase is blocked "
                              "unless both mono fits' RMS is at or under this (default: %(default)s "
                              "px) - 'ready' alone is a looser bar than a good stereo fit needs.")
    parser.add_argument("--min-coverage-fraction", type=float, default=0.5,
                         help="...and unless both mono fits' sampled-area fraction (convex hull of "
                              "every corner seen, over frame area) is at or above this (default: "
                              "%(default)s).")
    parser.add_argument("--no-mono-quality-gate", action="store_true",
                         help="Disable both checks above; continuing into the stereo phase only "
                              "ever requires a fit to have run, same as every other phase.")
    parser.add_argument("--show-rigidity-warnings", action="store_true",
                         help="Keep stereo_quality_warnings()'s single-view baseline/rotation-"
                              "scatter ('rigidity') warnings in the stereo phase's result panel. "
                              "Hidden by default - they're a rig-mounting diagnostic, not a "
                              "registration-accuracy one; still present in report.txt regardless.")
    parser.add_argument("--show-depth-range-warning", action="store_true",
                         help="Keep the 'vary the working distance' warning in the stereo phase's "
                              "result panel. Hidden by default - it assumes the rig should "
                              "generalize across a range of distances, which doesn't apply to a "
                              "fixed close-range setup; still present in report.txt regardless.")
    parser.add_argument("--no-preview", action="store_true", help="Headless mode: terminal keys only.")
    parser.add_argument("--touch", action="store_true",
                        help="Touchscreen mode: big two-row toolbar, no keyboard hints, "
                             "tap-again-to-confirm on the discard buttons.")
    return parser.parse_args()


def _discard_worst(session: SequentialLiveCalibrationSession, discard_count: int) -> str:
    """Remove the current phase's worst-error views; returns a status message."""
    result = session.last_fit()
    if result is None or not result.worst_views:
        return "No fit yet to know which views are worst - run a fit first."
    labels = [label for label, _error in result.worst_views[:discard_count]]
    removed = session.discard_views(labels)
    if removed == 0:
        return "Nothing to discard."
    return (f"Discarded {removed} worst view(s) ({session.view_count()}/{session.target()} left) - "
            "capture is paused; click 'Capture More' (u) when you're ready to refill.")


def _print_done(session: SequentialLiveCalibrationSession, args: argparse.Namespace,
                 captures_dir: Path, output_root: Path) -> None:
    print(f"\nDone. All three phases fit. Results written under {output_root}.")
    print("Nothing under calibration/results/ or design/config/rig_as_built.yaml changed.")
    print("Once you're happy with this result, promote it by re-running with the default output:")
    print(f"  python calibrate_cameras.py --camera {args.camera_a} --in {captures_dir}")
    print(f"  python calibrate_cameras.py --camera {args.camera_b} --in {captures_dir}")
    print(f"  python stereo_calibrate.py --camera-a {args.camera_a} --camera-b {args.camera_b} "
          f"--in {captures_dir}")


def main() -> int:
    args = parse_args()
    calibration_config = load_raw_config().get("geometric_calibration", {}) or {}
    output_base = args.output or calibration_config.get("output_dir", DEFAULT_OUTPUT_DIR)
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        output_root = resolve_path(output_base) / f"live_{run_timestamp}"
    else:
        output_root = resolve_path(output_base)

    captures_dir = resolve_path(args.captures) / f"live_{run_timestamp}"
    board_path = resolve_path(args.board)
    board = TargetBoard.from_yaml(board_path)
    config = load_pipeline_config(args.config)

    print(f"Captures: {captures_dir}")
    print(f"Results will be written to: {output_root}")
    print(f"Target: {board.squares_x}x{board.squares_y} ChArUco, {board.dictionary}, "
          f"square={board.square_size_m * 1000:.1f} mm")

    session = SequentialLiveCalibrationSession(
        board, board_path, captures_dir, output_root,
        camera_a=args.camera_a, camera_b=args.camera_b,
        min_corners=args.min_corners, min_shared_corners=args.min_shared_corners,
        mono_target=args.mono_views, stereo_target=args.stereo_views, top_up=args.top_up,
        max_mono_rms_for_stereo=None if args.no_mono_quality_gate else args.max_mono_rms_for_stereo,
        min_coverage_fraction=None if args.no_mono_quality_gate else args.min_coverage_fraction,
        show_rigidity_warnings=args.show_rigidity_warnings,
        show_depth_range_warning=args.show_depth_range_warning,
    )

    show_preview = not args.no_preview
    gui = None
    terminal_input: Optional[TerminalInput] = None
    cam1 = cam2 = None
    quit_requested = False

    try:
        print("Opening cameras...")
        cam1, cam2 = open_rgb_cameras(config)

        if show_preview:
            from live_gui import LiveCaptureGUI
            gui = LiveCaptureGUI(captures_dir, args.camera_a, args.camera_b, touch=args.touch)
            gui.set_status("Ready")
        else:
            terminal_input = TerminalInput()
            terminal_input.__enter__()
            print("Ready. Keys: r = refit now | u = capture more (top-up) | "
                  "w = discard worst views | c = continue to next phase | "
                  "d = discard phase | q = quit")

        preview_max_width = config.get("rgb", {}).get("preview_max_width")
        consecutive_failures = 0
        next_capture_time = time.time() + args.interval

        while not session.is_done():
            loop_start = time.time()
            if gui is not None and not gui.is_open():
                print("GUI closed.")
                quit_requested = True
                break

            phase = session.phase
            need_a = phase in (Phase.CAM_A, Phase.STEREO)
            need_b = phase in (Phase.CAM_B, Phase.STEREO)

            if need_a and need_b:
                frame_a, frame_b = RGBCamera.grab_pair(cam1, cam2)
                grabbed_ok = frame_a is not None and frame_b is not None
            elif need_a:
                frame_a, frame_b = cam1.grab(), None
                grabbed_ok = frame_a is not None
            else:
                frame_a, frame_b = None, cam2.grab()
                grabbed_ok = frame_b is not None

            if not grabbed_ok:
                consecutive_failures += 1
                if consecutive_failures in (1, 10, 25):
                    print("RGB grab failed, attempting camera recovery...")
                    recovered = (
                        RGBCamera.recover_pair(cam1, cam2) if need_a and need_b
                        else (cam1.recover() if need_a else cam2.recover())
                    )
                    if recovered:
                        consecutive_failures = 0
                        continue
                if consecutive_failures >= MAX_CONSECUTIVE_GRAB_FAILURES:
                    print("Failed to grab from the RGB camera(s).")
                    quit_requested = True
                    break
                time.sleep(0.1)
                continue
            consecutive_failures = 0

            preview_a = preview_frame(frame_a, preview_max_width) if frame_a is not None else None
            preview_b = preview_frame(frame_b, preview_max_width) if frame_b is not None else None
            gate = session.check_gate(preview_a, preview_b)
            session.poll()

            now = time.time()
            paused = gui is not None and gui.is_paused()
            can_capture = (
                not paused and now >= next_capture_time and gate.open
                and not session.is_fitting() and session.view_count() < session.target()
            )

            if gui is not None:
                gui.update(session, preview_a, preview_b, gate)

            if can_capture:
                label = session.tick(frame_a if need_a else None, frame_b if need_b else None)
                next_capture_time = now + args.interval
                if gui is not None:
                    gui.note_capture(label)
                else:
                    print(f"captured {label} ({session.view_count()}/{session.target()})")
            elif not gate.open and not paused and now >= next_capture_time:
                message = f"board not visible - waiting ({gate.reason})"
                if gui is not None:
                    gui.set_status(message)

            if session.maybe_start_fit() and gui is None:
                print(f"\nTarget reached ({session.view_count()} views) - fitting {phase.value}...")

            if gui is not None:
                if gui.consume_refit_request():
                    if not session.request_refit():
                        gui.set_status("Nothing to refit yet.")
                if gui.consume_top_up_request():
                    session.request_top_up()
                if gui.consume_continue_request():
                    ok, reason = session.can_advance()
                    if not ok:
                        gui.set_status(reason)
                    else:
                        session.advance()
                if gui.consume_discard_request():
                    session.request_discard()
                if gui.consume_discard_worst_request():
                    gui.set_status(_discard_worst(session, args.discard_count))
                if gui.should_quit():
                    quit_requested = True
                    break
            elif terminal_input is not None:
                key = terminal_input.poll_key()
                if key == ord("r"):
                    if not session.request_refit():
                        print("Nothing to refit yet.")
                elif key == ord("u"):
                    session.request_top_up()
                    print(f"Target raised to {session.target()}.")
                elif key == ord("c"):
                    ok, reason = session.can_advance()
                    if not ok:
                        print(reason)
                    else:
                        session.advance()
                        if not session.is_done():
                            print(f"\nMoving to phase: {session.phase.value} "
                                  f"({session.phase_camera_label()}, target {session.target()})")
                elif key == ord("d"):
                    session.request_discard()
                    print("Phase discarded - starting it over.")
                elif key == ord("w"):
                    print(_discard_worst(session, args.discard_count))
                elif key == ord("q"):
                    quit_requested = True
                    break

            elapsed = time.time() - loop_start
            remaining = GATE_CHECK_INTERVAL_S - elapsed
            if remaining > 0:
                time.sleep(remaining)

        if gui is not None:
            gui.close()

        if session.is_done():
            _print_done(session, args, captures_dir, output_root)
            review_and_prune(args.camera_a, args.camera_b, captures_dir, output_root)
        elif quit_requested:
            print(f"\nQuit before finishing all phases. Captured sessions remain in {captures_dir} - "
                  "run this tool again (it starts a fresh phase 1) or use the offline scripts directly "
                  f"against {captures_dir}.")

        return 0
    finally:
        if terminal_input is not None:
            terminal_input.__exit__(None, None, None)
        if cam1 is not None:
            cam1.release()
        if cam2 is not None:
            cam2.release()


if __name__ == "__main__":
    sys.exit(main())
