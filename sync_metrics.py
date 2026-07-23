"""Software timing metrics for multi-camera frame acquisition."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cameras.blackfly_camera import BlackflyCamera
from cameras.rgb_camera import GRAB_PAIR_MAX_ATTEMPTS, RGBCamera
from cameras.thermal_camera import ThermalCamera, ThermalFrame


@dataclass(frozen=True)
class GrabTimings:
    origin_s: float
    thermal_end_s: float
    rgb_grab_end_s: float
    rgb1_end_s: float
    rgb2_end_s: float
    blackfly_end_s: Optional[float] = None

    @classmethod
    def from_stamps(
        cls,
        origin_s: float,
        thermal_end_s: float,
        rgb_grab_end_s: float,
        rgb1_end_s: float,
        rgb2_end_s: float,
        blackfly_end_s: Optional[float] = None,
    ) -> "GrabTimings":
        return cls(origin_s, thermal_end_s, rgb_grab_end_s, rgb1_end_s, rgb2_end_s, blackfly_end_s)

    def _ms(self, start: float, end: float) -> float:
        return round((end - start) * 1000.0, 3)

    def to_dict(self, thermal_frame: Optional[ThermalFrame] = None) -> dict:
        thermal_hw = None
        if thermal_frame is not None:
            thermal_hw = int(thermal_frame.metadata.counterHW)

        payload = {
            "grab_order": "thermal -> rgb_grab -> rgb1_retrieve -> rgb2_retrieve",
            "durations_ms": {
                "thermal_grab": self._ms(self.origin_s, self.thermal_end_s),
                "rgb_grab": self._ms(self.thermal_end_s, self.rgb_grab_end_s),
                "rgb1_retrieve": self._ms(self.rgb_grab_end_s, self.rgb1_end_s),
                "rgb2_retrieve": self._ms(self.rgb1_end_s, self.rgb2_end_s),
                "total": self._ms(self.origin_s, self.rgb2_end_s),
            },
            "completion_offsets_ms": {
                "thermal": self._ms(self.origin_s, self.thermal_end_s),
                "rgb1": self._ms(self.origin_s, self.rgb1_end_s),
                "rgb2": self._ms(self.origin_s, self.rgb2_end_s),
            },
            "pair_offsets_ms": {
                "rgb1_vs_rgb2": self._ms(self.rgb1_end_s, self.rgb2_end_s),
                "thermal_vs_rgb1": self._ms(self.thermal_end_s, self.rgb1_end_s),
                "thermal_vs_rgb2": self._ms(self.thermal_end_s, self.rgb2_end_s),
            },
            "spread_ms": self._ms(self.thermal_end_s, self.rgb2_end_s),
            "thermal_counter_hw": thermal_hw,
        }
        if self.blackfly_end_s is not None:
            payload["grab_order"] += " -> blackfly_grab"
            payload["durations_ms"]["blackfly_grab"] = self._ms(self.rgb2_end_s, self.blackfly_end_s)
            payload["durations_ms"]["total"] = self._ms(self.origin_s, self.blackfly_end_s)
            payload["completion_offsets_ms"]["blackfly"] = self._ms(
                self.origin_s, self.blackfly_end_s
            )
            payload["pair_offsets_ms"]["blackfly_vs_rgb2"] = self._ms(
                self.rgb2_end_s, self.blackfly_end_s
            )
            payload["spread_ms"] = self._ms(self.thermal_end_s, self.blackfly_end_s)
        return payload


def grab_all_timed(
    cam1: RGBCamera,
    cam2: RGBCamera,
    thermal: ThermalCamera,
    blackfly: Optional[BlackflyCamera] = None,
) -> tuple[
    Optional[object],
    Optional[object],
    Optional[ThermalFrame],
    Optional[np.ndarray],
    GrabTimings,
]:
    origin_s = time.perf_counter()

    thermal_frame = thermal.grab()
    thermal_end_s = time.perf_counter()

    frame1 = None
    frame2 = None
    rgb_grab_end_s = thermal_end_s
    rgb1_end_s = thermal_end_s
    rgb2_end_s = thermal_end_s

    for _ in range(GRAB_PAIR_MAX_ATTEMPTS):
        if cam1.grab_only() and cam2.grab_only():
            rgb_grab_end_s = time.perf_counter()
            frame1 = cam1.retrieve_frame()
            rgb1_end_s = time.perf_counter()
            frame2 = cam2.retrieve_frame()
            rgb2_end_s = time.perf_counter()
            if frame1 is not None and frame2 is not None:
                break

        rgb_grab_end_s = time.perf_counter()
        rgb1_end_s = rgb_grab_end_s
        rgb2_end_s = rgb_grab_end_s

    blackfly_frame = None
    blackfly_end_s = None
    if blackfly is not None:
        blackfly_frame = blackfly.grab()
        blackfly_end_s = time.perf_counter()

    timings = GrabTimings.from_stamps(
        origin_s,
        thermal_end_s,
        rgb_grab_end_s,
        rgb1_end_s,
        rgb2_end_s,
        blackfly_end_s,
    )
    return frame1, frame2, thermal_frame, blackfly_frame, timings


def format_timings(timings: GrabTimings, thermal_frame: Optional[ThermalFrame] = None) -> str:
    data = timings.to_dict(thermal_frame)
    pair = data["pair_offsets_ms"]
    durations = data["durations_ms"]
    return (
        f"spread={data['spread_ms']:.1f} ms | "
        f"rgb1-rgb2={pair['rgb1_vs_rgb2']:.1f} ms | "
        f"thermal-rgb1={pair['thermal_vs_rgb1']:.1f} ms | "
        f"thermal-rgb2={pair['thermal_vs_rgb2']:.1f} ms | "
        f"total={durations['total']:.1f} ms"
    )


@dataclass
class SyncSummary:
    samples: int
    rgb1_vs_rgb2_ms: list[float]
    thermal_vs_rgb1_ms: list[float]
    thermal_vs_rgb2_ms: list[float]
    spread_ms: list[float]
    total_ms: list[float]

    @classmethod
    def from_timings(cls, timings_list: list[GrabTimings]) -> "SyncSummary":
        rgb1_vs_rgb2 = [t.to_dict()["pair_offsets_ms"]["rgb1_vs_rgb2"] for t in timings_list]
        thermal_vs_rgb1 = [t.to_dict()["pair_offsets_ms"]["thermal_vs_rgb1"] for t in timings_list]
        thermal_vs_rgb2 = [t.to_dict()["pair_offsets_ms"]["thermal_vs_rgb2"] for t in timings_list]
        spread = [t.to_dict()["spread_ms"] for t in timings_list]
        total = [t.to_dict()["durations_ms"]["total"] for t in timings_list]
        return cls(
            samples=len(timings_list),
            rgb1_vs_rgb2_ms=rgb1_vs_rgb2,
            thermal_vs_rgb1_ms=thermal_vs_rgb1,
            thermal_vs_rgb2_ms=thermal_vs_rgb2,
            spread_ms=spread,
            total_ms=total,
        )

    @staticmethod
    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0}
        return {
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "max": round(max(values), 2),
            "min": round(min(values), 2),
        }

    def to_dict(self) -> dict:
        return {
            "samples": self.samples,
            "rgb1_vs_rgb2_ms": self._stats(self.rgb1_vs_rgb2_ms),
            "thermal_vs_rgb1_ms": self._stats(self.thermal_vs_rgb1_ms),
            "thermal_vs_rgb2_ms": self._stats(self.thermal_vs_rgb2_ms),
            "spread_ms": self._stats(self.spread_ms),
            "total_grab_ms": self._stats(self.total_ms),
        }

    def format_report(self) -> str:
        data = self.to_dict()
        lines = [
            f"Sync test summary ({self.samples} samples)",
            (
                "  RGB1 ↔ RGB2:   "
                f"mean {data['rgb1_vs_rgb2_ms']['mean']:.2f} ms, "
                f"max {data['rgb1_vs_rgb2_ms']['max']:.2f} ms"
            ),
            (
                "  Thermal ↔ RGB1:"
                f" mean {data['thermal_vs_rgb1_ms']['mean']:.2f} ms, "
                f"max {data['thermal_vs_rgb1_ms']['max']:.2f} ms"
            ),
            (
                "  Thermal ↔ RGB2:"
                f" mean {data['thermal_vs_rgb2_ms']['mean']:.2f} ms, "
                f"max {data['thermal_vs_rgb2_ms']['max']:.2f} ms"
            ),
            (
                "  3-camera spread:"
                f" mean {data['spread_ms']['mean']:.2f} ms, "
                f"max {data['spread_ms']['max']:.2f} ms"
            ),
            (
                "  Total grab:    "
                f"mean {data['total_grab_ms']['mean']:.2f} ms, "
                f"max {data['total_grab_ms']['max']:.2f} ms"
            ),
        ]
        return "\n".join(lines)
