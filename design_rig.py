#!/usr/bin/env python3
"""Design and evaluate the multi-camera rig (2 x 16 MP RGB + Optris Xi 400 thermal).

    python design_rig.py info                     # intrinsics / FOV / GSD summary
    python design_rig.py view                     # interactive 3D frustum editor
    python design_rig.py plot --multiview         # static figure (iso/top/front/side)
    python design_rig.py report                   # text report + analysis figures
    python design_rig.py sweep elevation -20 -60  # parametric study of one angle
    python design_rig.py --rig design/config/rig_as_built.yaml optimize
                                                  # refine baseline / height / aim
                                                  # around the calibrated poses

Every command accepts ``--rig <path>`` to point at a different rig YAML.
For redesign from measured extrinsics, pass the as-built overlay.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
import yaml

from design import Rig, evaluate
from design.analysis import stereo_pairs, volume_coverage
from design.optics import (ELP_USB16MP01, OPTRIS_XI400, OPTRIS_XI400_LENSES,
                           fov_from_focal, verify_optris_lenses, zoom_table)
from design.report import write_report

DEFAULT_RIG = Path(__file__).resolve().parent / "design" / "config" / "rig.yaml"
DEFAULT_OUT = Path(__file__).resolve().parent / "design" / "out"


def load(args) -> Rig:
    rig = Rig.from_yaml(args.rig)
    if getattr(args, "voxel", None):
        rig.analysis.voxel_pitch = args.voxel
        rig.analysis.surface_pitch = args.voxel
    return rig


# --------------------------------------------------------------------------- #
def cmd_info(args) -> int:
    rig = load(args)
    print(rig.summary())
    print()
    print(f"{'camera':<12s}{'MP':>6s}{'HFOV':>8s}{'VFOV':>8s}"
          f"{'GSD@0.5m':>11s}{'footprint@0.5m':>18s}{'dist to target':>16s}")
    for cam in rig.cameras:
        gx, _ = cam.gsd(0.5)
        fw, fh = cam.footprint(0.5)
        dist = float(np.linalg.norm(rig.target.center - cam.t))
        print(f"{cam.name:<12s}{cam.megapixels:6.1f}{cam.hfov_deg:8.1f}"
              f"{cam.vfov_deg:8.1f}{gx * 1000:9.3f}mm"
              f"{fw * 100:9.1f}x{fh * 100:.1f}cm{dist:14.3f}m")
    print()
    cov = volume_coverage(rig)
    for k in range(1, len(rig) + 1):
        print(f"  target volume seen by >= {k} camera(s): "
              f"{cov.fraction_seen_by_at_least(k) * 100:6.2f} %")
    return 0


def cmd_view(args) -> int:
    from design.visualize import run_viewer
    rig = load(args)
    print(rig.summary())
    print("\ninteractive viewer: left-drag the 3D view to orbit; use the side "
          "panel to edit poses or type exact values.")
    print("keys:  r = full report   e = export   c = coverage cloud   "
          "i = common volume   t = reset view\n", flush=True)
    run_viewer(rig, export_dir=str(args.out))
    return 0


def cmd_plot(args) -> int:
    from design.visualize import plot_rig
    rig = load(args)
    out = Path(args.output) if args.output else (args.out / "rig_views.png")
    plot_rig(rig, screenshot=str(out) if args.save else None,
             off_screen=args.save, multiview=args.multiview)
    if args.save:
        print(f"wrote {out}")
    return 0


def cmd_report(args) -> int:
    rig = load(args)
    rep = evaluate(rig)
    print(rep.text())
    paths = write_report(rep, args.out)
    if args.render:
        from design.visualize import plot_rig
        plot_rig(rig, screenshot=str(args.out / "rig_views.png"), off_screen=True,
                 multiview=True)
        paths.append(args.out / "rig_views.png")
    print("\nwrote:")
    for p in paths:
        print("  " + str(p))
    return 0


def cmd_optics(args) -> int:
    """Show where every intrinsic number comes from, and cross-check it."""
    rig = load(args)
    cfg = yaml.safe_load(Path(args.rig).read_text()) or {}
    sources = {c.get("name"): (c.get("intrinsics") or {}).get("source", "fov")
               for c in cfg.get("cameras", [])}

    print("INTRINSICS PROVENANCE")
    print(f"  {'camera':<10s}{'source':<13s}{'resolution':>12s}{'HFOV':>8s}{'VFOV':>8s}"
          f"{'fx [px]':>10s}{'pitch':>8s}{'focal':>10s}")
    for cam in rig.cameras:
        pitch = cam.pixel_pitch_um
        f_mm = cam.focal_length_mm
        print(f"  {cam.name:<10s}{sources.get(cam.name, '?'):<13s}"
              f"{f'{cam.width}x{cam.height}':>12s}{cam.hfov_deg:8.2f}{cam.vfov_deg:8.2f}"
              f"{cam.fx:10.1f}"
              + (f"{pitch:7.2f}um" if pitch else f"{'-':>8s}")
              + (f"{f_mm:8.2f}mm" if f_mm else f"{'-':>10s}"))

    rgb_cameras = list(rig.of_modality("rgb"))
    if len(rgb_cameras) >= 2:
        print("\nRGB INTRINSICS CROSS-CHECK: are the two cameras actually matched?")
        dist_names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
        for cam_a, cam_b in itertools.combinations(rgb_cameras, 2):
            print(f"  {cam_a.name} vs {cam_b.name}")
            print(f"    {'':10s}{cam_a.name:>14s}{cam_b.name:>14s}{'delta':>10s}")
            for label, va, vb in [
                ("fx", cam_a.fx, cam_b.fx),
                ("fy", cam_a.fy, cam_b.fy),
                ("cx", cam_a.cx, cam_b.cx),
                ("cy", cam_a.cy, cam_b.cy),
                ("HFOV deg", cam_a.hfov_deg, cam_b.hfov_deg),
                ("VFOV deg", cam_a.vfov_deg, cam_b.vfov_deg),
            ]:
                delta = 100.0 * (va - vb) / vb if vb else float("nan")
                print(f"    {label:10s}{va:14.2f}{vb:14.2f}{delta:9.2f}%")
            dist_a = np.asarray(cam_a.distortion).reshape(-1)
            dist_b = np.asarray(cam_b.distortion).reshape(-1)
            n = min(len(dist_a), len(dist_b))
            if n:
                print(f"    {'distortion':10s}{cam_a.name:>14s}{cam_b.name:>14s}")
                for i in range(n):
                    label = dist_names[i] if i < len(dist_names) else f"d{i}"
                    print(f"    {label:10s}{dist_a[i]:14.4f}{dist_b[i]:14.4f}")
            if len(dist_a) != len(dist_b):
                print(f"    (distortion model length differs: "
                      f"{len(dist_a)} vs {len(dist_b)} terms)")

    print("\nRGB ZOOM: the ELP 5-50 mm ring is a design variable")
    print(f"  {ELP_USB16MP01.name}, {ELP_USB16MP01.width_px}x{ELP_USB16MP01.height_px}, "
          f"{ELP_USB16MP01.pitch_um} um pitch")
    print(f"  active area {ELP_USB16MP01.width_mm:.2f} x {ELP_USB16MP01.height_mm:.2f} mm, "
          f"diagonal {ELP_USB16MP01.diagonal_mm:.2f} mm")
    print(f"  {'focal':>8s}{'HFOV':>8s}{'VFOV':>8s}   full-resolution field of view")
    for f, h, v in zoom_table(ELP_USB16MP01, [5, 6, 6.5, 8, 10, 12, 16, 25, 35, 50]):
        print(f"  {f:7.1f}mm{h:8.1f}{v:8.1f}")
    for cam in rig.of_modality("rgb"):
        if cam.focal_length_mm:
            print(f"  -> {cam.name} is calibrated at about "
                  f"{cam.focal_length_mm:.2f} mm on that ring")

    print("\nTHERMAL LENS TABLE (Optris Xi 400 LT USB datasheet)")
    print(f"  {'lens':>7s}{'focal':>8s}{'F/#':>6s}{'min dist':>10s}"
          f"{'IFOV':>8s}{'MFOV':>8s}{'HFOV from focal':>18s}")
    for lens in OPTRIS_XI400_LENSES:
        h, _ = fov_from_focal(OPTRIS_XI400, lens.focal_mm)
        flag = "" if lens.rectilinear else "  <- not rectilinear"
        print(f"  {lens.label:>7s}{lens.focal_mm:7.1f}mm{lens.f_number:6.1f}"
              f"{lens.min_distance_m * 1000:9.0f}mm{lens.ifov_mm:7.1f}mm"
              f"{lens.mfov_mm:7.1f}mm{h:14.1f} deg{flag}")

    print("\n  cross-check: IFOV is one pixel at the minimum distance, so it must")
    print("  equal z_min/f_px. Only the published angles reproduce all four values.")
    print(f"  {'lens':>7s}{'IFOV sheet':>12s}{'from FOV':>11s}{'from focal':>12s}")
    for r in verify_optris_lenses():
        print(f"  {r['lens']:>7s}{r['ifov_datasheet_mm']:11.1f}mm"
              f"{r['ifov_from_fov_mm']:10.2f}mm{r['ifov_from_focal_mm']:11.2f}mm")

    print("\nTHERMAL MEASUREMENT SPOT at the current geometry")
    for cam in rig.of_modality("thermal"):
        dist = float(np.linalg.norm(rig.target.center - cam.t))
        gx, _ = cam.gsd(dist)
        print(f"  {cam.name}: {dist:.3f} m to target centre")
        print(f"    edge-fitted (datasheet FOV): {gx * 1000:.2f} mm/px, "
              f"smallest measurable feature {cam.measurement_spot(dist) * 1000:.2f} mm "
              f"({cam.measurement_pixels}x{cam.measurement_pixels} px)")
        on_axis = cam.paraxial_gsd(dist)
        if on_axis is not None and abs(on_axis - gx) / gx > 0.02:
            print(f"    on-axis (physical focal length): {on_axis * 1000:.2f} mm/px, "
                  f"feature {on_axis * cam.measurement_pixels * 1000:.2f} mm")
            print("    the lens is not rectilinear, so true resolution varies "
                  "between these across the frame")
    return 0


def _pair_geometry(rig: Rig, names):
    """Current baseline vector / midpoint / height for a stereo pair.

    The search keeps the mount's XY midpoint and the *horizontal* baseline
    direction from the loaded poses; height is a shared Z for both cameras.
    """
    a, b = rig.by_name(names[0]), rig.by_name(names[1])
    mid = 0.5 * (a.t + b.t)
    delta = b.t - a.t
    delta_xy = np.array([delta[0], delta[1], 0.0])
    baseline = float(np.linalg.norm(delta_xy))
    if baseline < 1e-9:
        # Cameras stacked vertically — fall back to the full 3D direction.
        baseline = float(np.linalg.norm(delta))
        if baseline < 1e-9:
            raise ValueError(f"{names[0]} and {names[1]} share the same position")
        direction = delta / baseline
    else:
        direction = delta_xy / baseline
    height = float(mid[2])
    elevations = [c.azimuth_elevation_roll[1] for c in (a, b)]
    return mid, direction, baseline, height, elevations


def _score_layout(rig: Rig, names, tri_lo: float, tri_hi: float,
                  pitch: float) -> dict:
    """Coverage + stereo metrics used to rank a candidate layout."""
    cov = volume_coverage(rig, pitch=pitch)
    both = cov.fraction_seen_by(names)
    all_c = cov.fraction_seen_by_at_least(len(rig))
    per = cov.per_camera_fraction()
    pair = stereo_pairs(rig)[(names[0], names[1])]
    tri = pair["triangulation_angle_deg"]
    sigma = pair["depth_sigma_mm"]
    feasible = (both > 0.999
                and tri_lo <= tri <= tri_hi)
    # Rank feasible layouts by depth precision, then by full-rig coverage.
    # Infeasible ones sort after every feasible point.
    rank = (0, sigma, -all_c) if feasible else (1, -both, sigma)
    return {
        "both": both, "all": all_c, "per": per, "tri": tri, "sigma": sigma,
        "feasible": feasible, "rank": rank, "pair": pair,
    }


def _apply_pair_layout(rig: Rig, names, mid_xy, direction, baseline: float,
                       height: float, elevation, aim: str, move_others: bool):
    """Place the stereo pair; optionally lift the remaining cameras with them."""
    a0, b0 = rig.by_name(names[0]), rig.by_name(names[1])
    old_mid_z = 0.5 * (a0.t[2] + b0.t[2])
    dz = height - old_mid_z
    mid = np.array([mid_xy[0], mid_xy[1], height], dtype=float)
    for cam, sign in zip((a0, b0), (-1.0, 1.0)):
        cam.t = mid + sign * (baseline / 2.0) * direction
        cam.t[2] = height
        if aim == "look-at":
            cam.look_at(rig.target.center)
        elif aim == "elevation":
            # Aim the optical axis at the target in azimuth, free elevation.
            to = rig.target.center - cam.t
            az = float(np.degrees(np.arctan2(to[1], to[0])))
            cam.set_pose(azimuth_deg=az, elevation_deg=float(elevation),
                         roll_deg=0.0)
        # aim == "hold": keep the rotation matrices already on the cameras

    if move_others:
        for cam in rig.cameras:
            if cam.name not in names:
                cam.t = cam.t + np.array([0.0, 0.0, dz])
                if aim == "look-at":
                    cam.look_at(rig.target.center)


def cmd_optimize(args) -> int:
    """Refine the *present* rig: baseline, mount height, and camera aim.

    Starts from the poses in the loaded YAML (typically the as-built overlay from
    stereo calibration) and searches nearby layouts. It does **not** invent a
    clean symmetric pair from scratch — the midpoint and baseline direction of
    your current mount are kept, and only the design knobs you can still change
    (separation, height, tilt) are varied so the subject fills the frustums.
    """
    rig = load(args)
    names = args.cameras or [c.name for c in rig.of_modality("rgb")][:2]
    if len(names) != 2:
        print(f"need exactly two cameras, got {names}", file=sys.stderr)
        return 2

    try:
        mid, direction, B0, H0, elevations0 = _pair_geometry(rig, names)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    pitch = args.voxel or max(rig.analysis.voxel_pitch, 0.005)
    tri_lo, tri_hi = args.triangulation

    # Default search windows hug the present design instead of a blank slate.
    if args.baseline is None:
        b_lo, b_hi = 0.6 * B0 * 1000.0, 1.8 * B0 * 1000.0
    else:
        b_lo, b_hi = args.baseline
    if args.height is None:
        h_lo, h_hi = H0 - 0.05, H0 + 0.05
    else:
        h_lo, h_hi = args.height
    if args.elevation is None:
        el0 = float(np.mean(elevations0))
        el_lo, el_hi = el0 - 20.0, el0 + 10.0
    else:
        el_lo, el_hi = args.elevation

    baselines = np.linspace(b_lo, b_hi, args.steps) / 1000.0
    heights = np.linspace(h_lo, h_hi, args.steps)
    # Elevation is only a free parameter when --aim elevation; otherwise one pass.
    elevations = (np.linspace(el_lo, el_hi, args.steps) if args.aim == "elevation"
                  else np.array([float(np.mean(elevations0))]))

    print(f"optimising present layout of {names[0]} / {names[1]}")
    print(f"  source: {rig.source_path or args.rig}")
    print(f"  current baseline {B0 * 1000:.1f} mm along "
          f"[{direction[0]:+.3f}, {direction[1]:+.3f}, {direction[2]:+.3f}]")
    print(f"  current height   {H0 * 1000:.1f} mm  "
          f"(midpoint XY = [{mid[0]:.4f}, {mid[1]:.4f}])")
    print(f"  current elev.    {elevations0[0]:.1f} / {elevations0[1]:.1f} deg")
    print(f"  aim mode         {args.aim}"
          + ("; other cameras move in z with the pair" if args.move_others
             else "; other cameras stay put"))
    print(f"  feasible = both RGB see >= 99.9 % of the target volume and "
          f"triangulation is {tri_lo:.0f}-{tri_hi:.0f} deg\n")

    # Score the as-loaded configuration first so improvement is visible.
    current = _score_layout(rig, names, tri_lo, tri_hi, pitch)
    print(f"  PRESENT: both-RGB {current['both'] * 100:5.1f} %  "
          f"all-cams {current['all'] * 100:5.1f} %  "
          f"tri {current['tri']:5.1f} deg  "
          f"sigma_z {current['sigma'] * 1000:6.0f} um"
          f"{'  [feasible]' if current['feasible'] else '  [not feasible]'}")
    for n, f in current["per"].items():
        print(f"           {n:<12s} {f * 100:5.1f} % of target volume")
    print()

    best = None
    best_rig = None
    # Compact table: for each (height, elevation) row, show baseline columns.
    # When elevation is not swept the row label is just height.
    for elev in elevations:
        if args.aim == "elevation":
            print(f"  elevation {elev:+.1f} deg")
        print(f"  {'height':>8s}" + "".join(f"{f'{b * 1000:.0f}mm':>18s}"
                                            for b in baselines))
        for height in heights:
            cells = []
            for baseline in baselines:
                r = rig.copy()
                _apply_pair_layout(
                    r, names, mid[:2], direction, baseline, height, elev,
                    args.aim, args.move_others)
                m = _score_layout(r, names, tri_lo, tri_hi, pitch)
                cells.append(f"{m['both'] * 100:5.1f}{'*' if m['feasible'] else ' '}"
                             f"|{m['tri']:4.1f}|{m['sigma'] * 1000:5.0f}")
                if best is None or m["rank"] < best["rank"]:
                    best = dict(m, baseline=baseline, height=height,
                                elevation=elev)
                    best_rig = r
            print(f"  {height:7.3f}m" + "".join(f"{c:>18s}" for c in cells))
        print()

    print("  cells: both-RGB coverage % | triangulation deg | sigma_z um")
    print("  * marks a feasible point (full target in both RGB + good tri angle)")

    if best is None:
        print("\n  no candidates evaluated", file=sys.stderr)
        return 1

    print(f"\n  BEST{'  (feasible)' if best['feasible'] else '  (best effort — none fully feasible)'}:")
    print(f"    baseline  {best['baseline'] * 1000:.1f} mm"
          f"  (was {B0 * 1000:.1f} mm, "
          f"delta {best['baseline'] * 1000 - B0 * 1000:+.1f} mm)")
    print(f"    height    {best['height'] * 1000:.1f} mm"
          f"  (was {H0 * 1000:.1f} mm, "
          f"delta {(best['height'] - H0) * 1000:+.1f} mm)")
    if args.aim == "elevation":
        print(f"    elevation {best['elevation']:+.1f} deg"
              f"  (was {float(np.mean(elevations0)):+.1f} deg mean)")
    elif args.aim == "look-at":
        print("    aim       look-at target centre (rotations follow the new positions)")
    else:
        print("    aim       held at the loaded rotations")
    print(f"    both-RGB  {best['both'] * 100:.1f} %   all-cams {best['all'] * 100:.1f} %")
    print(f"    tri       {best['tri']:.1f} deg   sigma_z {best['sigma'] * 1000:.0f} um")

    print("\n  suggested poses (paste into an overlay YAML, or --write):")
    for cam in best_rig.cameras:
        az, el, roll = cam.azimuth_elevation_roll
        print(f"  - name: {cam.name}")
        print(f"    pose:")
        print(f"      position: [{cam.t[0]:.4f}, {cam.t[1]:.4f}, {cam.t[2]:.4f}]")
        if args.aim == "hold":
            print("      # rotation held from the loaded file")
        else:
            print(f"      azimuth_deg: {az:.2f}")
            print(f"      elevation_deg: {el:.2f}")
            print(f"      roll_deg: {roll:.2f}")

    if args.write:
        out = Path(args.write)
        # Keep measured intrinsics / target / analysis via extends when possible.
        src = Path(args.rig)
        payload = {
            "extends": str(src.name) if src.parent == out.parent
            else str(src.resolve()),
            "name": f"{rig.name}_optimised",
            "cameras": [],
        }
        for cam in best_rig.cameras:
            az, el, roll = cam.azimuth_elevation_roll
            entry = {
                "name": cam.name,
                "pose": {
                    "position": [round(float(v), 6) for v in cam.t],
                },
            }
            if args.aim != "hold":
                entry["pose"]["azimuth_deg"] = round(az, 4)
                entry["pose"]["elevation_deg"] = round(el, 4)
                entry["pose"]["roll_deg"] = round(roll, 4)
            else:
                entry["pose"]["R"] = [
                    [round(float(v), 9) for v in row] for row in cam.R
                ]
            payload["cameras"].append(entry)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(payload, sort_keys=False))
        print(f"\n  wrote {out}")
        print(f"  inspect with:  python design_rig.py --rig {out} view")

    if not best["feasible"]:
        print("\n  nothing fully feasible in this window — widen --baseline / "
              "--height / --elevation or relax --triangulation", file=sys.stderr)
        return 1
    return 0


_SWEEP_PARAMS = ("x", "y", "z", "azimuth", "elevation", "roll", "far")


def cmd_sweep(args) -> int:
    """Vary one degree of freedom of one camera and tabulate the metrics."""
    rig = load(args)
    cam = rig.by_name(args.camera) if args.camera else rig.cameras[0]
    values = np.linspace(args.start, args.stop, args.steps)

    print(f"sweeping {args.param} of {cam.name} "
          f"from {args.start} to {args.stop} ({args.steps} steps)\n")
    header = (f"{args.param:>12s}" + "".join(f"{n[:9]:>11s}" for n in rig.names)
              + f"{'>=2 cam':>11s}{'all cam':>11s}{'GSD mm':>10s}")
    print(header)
    print("-" * len(header))

    best = (-1.0, None)
    for v in values:
        if args.param in ("x", "y", "z"):
            t = cam.t.copy()
            t["xyz".index(args.param)] = v
            cam.set_pose(position=t)
        elif args.param == "far":
            cam.far = float(v)
        else:
            cam.set_pose(**{f"{args.param}_deg": float(v)})
        if args.look_at_target and args.param in ("x", "y", "z"):
            cam.look_at(rig.target.center)

        cov = volume_coverage(rig)
        per = cov.per_camera_fraction()
        all_c = cov.fraction_seen_by_at_least(len(rig))
        two_c = cov.fraction_seen_by_at_least(2)
        gx, _ = cam.gsd(float(np.linalg.norm(rig.target.center - cam.t)))
        print(f"{v:12.2f}" + "".join(f"{per[n] * 100:10.1f}%" for n in rig.names)
              + f"{two_c * 100:10.1f}%{all_c * 100:10.1f}%{gx * 1000:10.3f}")
        if all_c > best[0]:
            best = (all_c, v)

    print(f"\nbest {args.param} = {best[1]:.2f} "
          f"({best[0] * 100:.1f} % of the target seen by all {len(rig)} cameras)")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rig", type=Path, default=DEFAULT_RIG, help="rig YAML")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    p.add_argument("--voxel", type=float, default=None,
                   help="override the sampling pitch in metres")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="print intrinsics, FOV and coverage").set_defaults(
        func=cmd_info)
    sub.add_parser("optics", help="show where each intrinsic comes from and "
                                  "cross-check it against the datasheets"
                   ).set_defaults(func=cmd_optics)
    sub.add_parser("view", help="interactive 3D viewer").set_defaults(func=cmd_view)

    sp = sub.add_parser("plot", help="static 3D figure")
    sp.add_argument("-o", "--output", default=None)
    sp.add_argument("--save", action="store_true", help="render off-screen to a file")
    sp.add_argument("--multiview", action="store_true", help="2x2 iso/top/front/side")
    sp.set_defaults(func=cmd_plot)

    sp = sub.add_parser("report", help="text report and analysis figures")
    sp.add_argument("--render", action="store_true",
                    help="also render the 3D views off-screen")
    sp.set_defaults(func=cmd_report)

    opt_help = (
        "refine baseline, mount height and camera aim around the loaded poses "
        "(use with --rig design/config/rig_as_built.yaml)"
    )
    opt_epilog = (
        "Starts from the poses already in the YAML — typically the as-built\n"
        "overlay written by stereo calibration — and searches nearby layouts.\n"
        "The midpoint and baseline direction of your mount are kept; only\n"
        "separation, height and tilt are varied so the subject fills the frames.\n"
        "\n"
        "examples:\n"
        "  design_rig.py --rig design/config/rig_as_built.yaml optimize\n"
        "  design_rig.py --rig design/config/rig_as_built.yaml optimize "
        "--aim elevation\n"
        "  design_rig.py --rig design/config/rig_as_built.yaml optimize "
        "--baseline 40 120 --height 0.25 0.32 --steps 7\n"
        "  design_rig.py --rig design/config/rig_as_built.yaml optimize "
        "--write design/config/optimized_baseline.yaml"
    )
    sp = sub.add_parser(
        "optimize", help=opt_help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=opt_epilog)
    sp.add_argument("--cameras", nargs=2, default=None,
                    help="the pair (default: the first two RGB cameras)")
    sp.add_argument("--baseline", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="baseline range in mm (default: 0.6x-1.8x current)")
    sp.add_argument("--height", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="mount height range in m (default: current +- 50 mm)")
    sp.add_argument("--elevation", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="elevation range in deg (only with --aim elevation)")
    sp.add_argument("--aim", choices=["look-at", "elevation", "hold"],
                    default="look-at",
                    help="how to set rotations: re-aim at the target "
                         "(default), sweep elevation, or keep loaded R")
    sp.add_argument("--move-others", action="store_true",
                    help="also shift non-pair cameras in z with the mount")
    sp.add_argument("--triangulation", type=float, nargs=2,
                    default=[15.0, 30.0], metavar=("MIN", "MAX"),
                    help="acceptable triangulation angle in deg "
                         "(default 15 30)")
    sp.add_argument("--steps", type=int, default=7,
                    help="samples per axis (default 7)")
    sp.add_argument("--write", type=Path, default=None,
                    help="write the best layout as a YAML overlay")
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser(
        "sweep", help="parametric study of one degree of freedom",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="start and stop are positional, in metres for x/y/z/far and in\n"
               "degrees for azimuth/elevation/roll. Examples:\n"
               "  design_rig.py sweep z 0.25 0.40 --steps 10 --camera thermal\n"
               "  design_rig.py sweep x 0.02 0.15 --camera rgb_cam2 --look-at-target\n"
               "  design_rig.py sweep elevation -90 -45 --camera thermal")
    sp.add_argument("param", choices=_SWEEP_PARAMS,
                    help="degree of freedom to vary")
    sp.add_argument("start", type=float, help="first value (m or deg)")
    sp.add_argument("stop", type=float, help="last value (m or deg)")
    sp.add_argument("--steps", type=int, default=13, help="samples (default 13)")
    sp.add_argument("--camera", default=None,
                    help="camera to move (default: the first in the rig)")
    sp.add_argument("--look-at-target", action="store_true",
                    help="re-aim at the target centre after each move")
    sp.set_defaults(func=cmd_sweep)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
