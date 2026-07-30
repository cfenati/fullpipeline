"""Matplotlib figures summarising a rig design."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent
                                          / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import RigReport, overlap_vs_depth, registration_table
from .rig import Rig


def figure_depth_sweep(rep: RigReport, path) -> Optional[Path]:
    """Common field of view and shared area as a function of working distance."""
    ds = rep.depth_sweep or overlap_vs_depth(rep.rig)
    d = ds["depths_m"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for cam in rep.rig.cameras:
        axes[0].plot(d, ds["fraction"][cam.name] * 100, label=cam.name,
                     color=cam.color, lw=1.8)
    axes[0].plot(d, ds["fraction_all"] * 100, "k--", lw=2, label="all cameras")
    axes[0].set_xlabel(f"depth along {ds['axis_camera']} optical axis [m]")
    axes[0].set_ylabel(f"share of {ds['axis_camera']} field of view [%]")
    axes[0].set_title("Field-of-view overlap vs distance")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(d, ds["area_axis_cm2"], color="#64748b", lw=1.6,
                 label=f"{ds['axis_camera']} footprint")
    axes[1].plot(d, ds["area_all_cm2"], "k-", lw=2, label="shared by all")
    axes[1].set_xlabel("depth [m]")
    axes[1].set_ylabel("area [cm$^2$]")
    axes[1].set_title("Imaged area vs distance")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    return _save(fig, path)


def figure_resolution(rig: Rig, path, n: int = 120) -> Optional[Path]:
    """Ground sampling distance and footprint versus working distance."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for cam in rig.cameras:
        z = np.linspace(cam.near, cam.far, n)
        axes[0].plot(z, z / cam.fx * 1000, color=cam.color, lw=1.8,
                     label=f"{cam.name} ({cam.megapixels:.1f} MP)")
        axes[1].plot(z, z * cam.width / cam.fx * 100, color=cam.color, lw=1.8,
                     label=f"{cam.name} width")
        axes[1].plot(z, z * cam.height / cam.fy * 100, color=cam.color, lw=1.2,
                     ls="--", label=f"{cam.name} height")

    size = rig.target.characteristic_size
    axes[1].axhline(size * 100, color="k", ls=":", lw=1.5,
                    label=f"target size {size * 100:.0f} cm")
    axes[0].set_xlabel("distance [m]")
    axes[0].set_ylabel("GSD [mm/px]")
    axes[0].set_title("Sampling density vs distance")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("distance [m]")
    axes[1].set_ylabel("footprint [cm]")
    axes[1].set_title("Imaged extent vs distance")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)

    return _save(fig, path)


def figure_registration(rep: RigReport, path) -> Optional[Path]:
    """Parallax error of a fixed pixel mapping as the subject depth changes."""
    curves = rep.registration or registration_table(rep.rig)
    if not curves:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # Symmetric pairs give near-identical curves, so vary the dash pattern too.
    styles = ["-", "--", "-.", ":"]
    for i, r in enumerate(curves):
        label = f"{r['source']} -> {r['target']}"
        ls = styles[i % len(styles)]
        axes[0].plot(r["depths_m"], r["rms_px"], lw=1.8, ls=ls, label=label)
        axes[1].plot(r["depths_m"], r["rms_mm"], lw=1.8, ls=ls, label=label)
    z0 = curves[0]["reference_depth_m"]
    for ax in axes:
        ax.axvline(z0, color="k", ls=":", lw=1.4,
                   label=f"reference depth {z0:.2f} m")
        ax.set_xlabel("true subject depth [m]")
        ax.grid(alpha=0.3, which="both")
        # The error is a magnitude and vanishes at the reference depth, so use a
        # plain log axis and let the zero be clipped.
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("registration RMS error [px]")
    axes[0].set_title("Depth-dependent registration error (pixels)")
    axes[1].set_ylabel("registration RMS error [mm]")
    axes[1].set_title("Depth-dependent registration error (metric)")

    return _save(fig, path)


def figure_coverage(rep: RigReport, path) -> Optional[Path]:
    """Histogram of simultaneous views plus per-camera coverage bars."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    n_cam = len(rep.rig)

    counts = np.bincount(rep.volume.n_views, minlength=n_cam + 1)
    axes[0].bar(np.arange(n_cam + 1), counts / max(1, rep.volume.n_points) * 100,
                color="#2563eb")
    axes[0].set_xlabel("number of cameras seeing a voxel")
    axes[0].set_ylabel("share of target volume [%]")
    axes[0].set_title("Simultaneous views (volume)")
    axes[0].set_xticks(np.arange(n_cam + 1))
    axes[0].grid(alpha=0.3, axis="y")

    names = rep.rig.names
    vol = [rep.volume.per_camera_fraction()[n] * 100 for n in names]
    surf = [rep.surface.per_camera_fraction()[n] * 100 for n in names]
    x = np.arange(len(names))
    axes[1].bar(x - 0.2, vol, width=0.38, label="volume", color="#2563eb")
    axes[1].bar(x + 0.2, surf, width=0.38, label="surface", color="#16a34a")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15)
    axes[1].set_ylabel("coverage [%]")
    axes[1].set_title("Per-camera coverage of the target")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)

    tri = rep.triangulation
    if tri and tri["multi_view"].any():
        axes[2].hist(tri["angle_deg"][tri["multi_view"]], bins=30,
                     color="#d97706")
        axes[2].set_xlabel("best pairwise triangulation angle [deg]")
        axes[2].set_ylabel("voxels")
        axes[2].set_title("Triangulation geometry")
        axes[2].grid(alpha=0.3, axis="y")
    else:
        axes[2].text(0.5, 0.5, "no multi-view voxels", ha="center", va="center")
        axes[2].set_axis_off()

    return _save(fig, path)


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(rep: RigReport, directory) -> List[Path]:
    """Text report plus all figures."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    txt = directory / "rig_report.txt"
    txt.write_text(rep.text())
    paths = [txt]

    for fn, name in ((figure_coverage, "coverage.png"),
                     (figure_depth_sweep, "overlap_vs_depth.png"),
                     (figure_registration, "registration_error.png")):
        out = fn(rep, directory / name)
        if out:
            paths.append(out)
    paths.append(figure_resolution(rep.rig, directory / "resolution.png"))
    return [p for p in paths if p]
