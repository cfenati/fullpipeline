#!/usr/bin/env python3
"""Delete the worst capture sessions from a calibration set.

Reads the per-view reprojection errors recorded by calibrate_cameras.py and
removes the capture session folders that hurt the calibration most.

Nothing is deleted unless --apply is given.

Example:
    python prune_calibration.py --camera rgb_cam1 --count 5
    python prune_calibration.py --camera rgb_cam1 --count 5 --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_cameras import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    default_capture_dir,
    load_config,
    resolve_path,
)
from calibration.opencv_calibrate import CameraIntrinsics  # noqa: E402

# Never leave a set too small to calibrate from.
MIN_REMAINING_SESSIONS = 8

UNDETECTED_ERROR = float("inf")


@dataclass
class Candidate:
    session: str
    error_px: float
    status: str
    path: Path

    @property
    def error_text(self) -> str:
        if self.error_px == UNDETECTED_ERROR:
            return "no board"
        return f"{self.error_px:.4f}"


def build_candidates(intrinsics: CameraIntrinsics, capture_dir: Path) -> List[Candidate]:
    """Every session that took part in the calibration, worst first."""
    scored: List[Candidate] = []

    for session, reason in intrinsics.discarded_views.items():
        scored.append(
            Candidate(session, UNDETECTED_ERROR, f"discarded: {reason}", capture_dir / session)
        )
    for session, error in intrinsics.rejected_views.items():
        scored.append(
            Candidate(session, float(error), "rejected as outlier", capture_dir / session)
        )
    for session, error in intrinsics.per_view_errors.items():
        scored.append(Candidate(session, float(error), "used", capture_dir / session))

    scored.sort(key=lambda candidate: candidate.error_px, reverse=True)
    return scored


def select_for_deletion(
    candidates: List[Candidate],
    count: Optional[int],
    max_error: Optional[float],
) -> List[Candidate]:
    if count is None and max_error is None:
        raise SystemExit("Choose what to remove with --count and/or --max-error.")

    selected = candidates
    if max_error is not None:
        selected = [
            candidate for candidate in selected if candidate.error_px > max_error
        ]
    if count is not None:
        selected = selected[:count]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete the worst capture sessions from a calibration set.",
    )
    parser.add_argument(
        "--camera",
        default="rgb_cam1",
        help="Camera whose calibration report should be used (default: %(default)s).",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Number of worst sessions to delete.",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        help="Delete every session above this reprojection error in px.",
    )
    parser.add_argument(
        "--captures",
        help="Capture directory to prune (default: captures/calib_<camera>).",
    )
    parser.add_argument(
        "--results",
        help=f"Calibration results directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--keep-undetected",
        action="store_true",
        help="Do not consider sessions where the board was never detected.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without this the tool only reports what it would remove.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Allow leaving fewer than {MIN_REMAINING_SESSIONS} sessions.",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Prune even when the report lists sessions that were already deleted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calibration_config = load_config().get("geometric_calibration", {}) or {}

    capture_dir = resolve_path(args.captures or default_capture_dir(args.camera))
    results_root = resolve_path(
        args.results or calibration_config.get("output_dir", DEFAULT_OUTPUT_DIR)
    )
    intrinsics_path = results_root / args.camera / "intrinsics.json"

    if not intrinsics_path.exists():
        raise SystemExit(
            f"No calibration report found at {intrinsics_path}\n"
            f"Run: python calibrate_cameras.py --camera {args.camera}"
        )

    intrinsics = CameraIntrinsics.load_json(intrinsics_path)
    candidates = build_candidates(intrinsics, capture_dir)
    if args.keep_undetected:
        candidates = [c for c in candidates if c.error_px != UNDETECTED_ERROR]

    print(f"Report:   {intrinsics_path}")
    print(f"Captures: {capture_dir}")
    print(
        f"Report describes {intrinsics.views_used} views, "
        f"RMS {intrinsics.reprojection_error_px:.4f} px"
    )

    # Sessions already deleted by an earlier run mean the report predates the
    # current capture set, so its per-view errors no longer describe the data.
    already_gone = [c for c in candidates if not c.path.is_dir()]
    candidates = [c for c in candidates if c.path.is_dir()]
    print(f"Sessions in report: {len(candidates) + len(already_gone)}, on disk: {len(candidates)}")

    if already_gone:
        print(
            f"\nThis report is out of date: {len(already_gone)} of its sessions "
            "are already deleted."
        )
        print("Its per-view errors were measured with those views included, so they")
        print("no longer describe the remaining captures.")
        if not args.allow_stale:
            raise SystemExit(
                f"\nRefusing to prune from a stale report. Refresh it first:\n"
                f"  python calibrate_cameras.py --camera {args.camera}\n"
                "Then prune again, or pass --allow-stale to use these old errors anyway."
            )
        print("Continuing with --allow-stale; only sessions still on disk are considered.")

    selected = select_for_deletion(candidates, args.count, args.max_error)
    if not selected:
        print("\nNothing matches the given criteria; no folders to delete.")
        return 0

    remaining = len(candidates) - len(selected)
    print(f"\nWorst {len(selected)} session(s), worst first:")
    print(f"  {'session':<22}{'error px':>10}   role in that calibration")
    for candidate in selected:
        print(f"  {candidate.session:<22}{candidate.error_text:>10}   {candidate.status}")

    kept = [c for c in candidates if c not in selected]
    if kept:
        best, worst = kept[-1], kept[0]
        print(
            f"\nWould keep {remaining} session(s), "
            f"errors {best.error_text} to {worst.error_text} px"
        )

    if remaining < MIN_REMAINING_SESSIONS and not args.force:
        raise SystemExit(
            f"\nRefusing to leave only {remaining} session(s); "
            f"{MIN_REMAINING_SESSIONS} are needed for a usable calibration. "
            "Lower --count, or pass --force."
        )

    if not args.apply:
        print("\nDry run. Nothing was deleted. Re-run with --apply to delete.")
        return 0

    print("")
    deleted = 0
    for candidate in selected:
        # Guard against a malformed session name escaping the capture directory.
        path = candidate.path.resolve()
        if capture_dir not in path.parents:
            print(f"  skipped {candidate.session}: outside {capture_dir}")
            continue
        if not path.is_dir():
            print(f"  skipped {candidate.session}: folder not found")
            continue
        shutil.rmtree(path)
        print(f"  deleted {path}")
        deleted += 1

    print(f"\nDeleted {deleted} session folder(s).")
    print(
        "Each folder held every camera for that timestamp, not just "
        f"{args.camera}.jpg."
    )
    print(f"Now re-run: python calibrate_cameras.py --camera {args.camera}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
