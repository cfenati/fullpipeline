#!/usr/bin/env python3
"""Analyze RGB camera color uniformity (vignetting, border tint, channel balance)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# OpenCV's bundled Qt plugins break interactive GUI backends; save figures headlessly.
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Import cv2 after matplotlib backend is set.
import cv2

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cameras.rgb_camera import RGBCamera, controls_for
from color_correction import (
    DEFAULT_MAX_GAIN,
    DEFAULT_MIN_GAIN,
    FlatFieldMaps,
    apply_flat_field_correction,
)


@dataclass
class RegionStats:
    name: str
    pixel_count: int
    b_mean: float
    g_mean: float
    r_mean: float
    gray_mean: float
    rg_ratio: float
    bg_ratio: float

    @property
    def brightness(self) -> float:
        return self.gray_mean


@dataclass
class UniformityReport:
    image_path: Path
    shape: tuple[int, int]
    regions: list[RegionStats]
    radial_gray: list[tuple[float, float]]
    center_to_border_drop: float
    max_rg_spread: float
    max_bg_spread: float
    vignette_score: float
    color_uniformity_score: float

    def print_summary(self) -> None:
        h, w = self.shape
        print(f"\nImage: {self.image_path}")
        print(f"Size:  {w}x{h}")
        print("\nRegion statistics (BGR means, gray = luminance proxy):")
        print(f"{'region':<10} {'pixels':>8} {'B':>7} {'G':>7} {'R':>7} {'gray':>7} {'R/G':>7} {'B/G':>7}")
        for region in self.regions:
            print(
                f"{region.name:<10} {region.pixel_count:>8} "
                f"{region.b_mean:>7.1f} {region.g_mean:>7.1f} {region.r_mean:>7.1f} "
                f"{region.gray_mean:>7.1f} {region.rg_ratio:>7.3f} {region.bg_ratio:>7.3f}"
            )

        print("\nRadial brightness profile (0 = center, 1 = corner):")
        for r_lo, gray in self.radial_gray:
            bar = "#" * int(max(0.0, min(40.0, gray / 4.0)))
            print(f"  r={r_lo:0.1f}-{r_lo + 0.1:0.1f}: gray={gray:6.1f}  {bar}")

        print("\nDiagnostics:")
        print(f"  Center-to-border brightness drop: {self.center_to_border_drop:+.1f} (positive = darker edges)")
        print(f"  Vignette score:                   {self.vignette_score:.1f}% edge darkening")
        print(f"  Max R/G spread across regions:    {self.max_rg_spread:.3f}")
        print(f"  Max B/G spread across regions:    {self.max_bg_spread:.3f}")
        print(f"  Color uniformity score:           {self.color_uniformity_score:.1f}% deviation")

        print("\nLikely causes:")
        for line in self._likely_causes():
            print(f"  - {line}")

    def _likely_causes(self) -> list[str]:
        causes: list[str] = []

        if self.vignette_score >= 8.0:
            causes.append(
                "Strong vignetting or uneven illumination: edges are much darker than center. "
                "Check lens FOV, distance to the scene, and whether lighting is diffuse."
            )
        elif self.vignette_score >= 3.0:
            causes.append(
                "Mild edge darkening: common with wide-angle lenses or when the light source "
                "does not cover the full field of view."
            )

        if self.max_rg_spread >= 0.08 or self.max_bg_spread >= 0.08:
            causes.append(
                "Color tint changes across the frame (R/G or B/G not flat). "
                "This can come from auto white balance, lens shading correction, or colored lighting."
            )

        border_names = {"top", "bottom", "left", "right"}
        borders = [region for region in self.regions if region.name in border_names]
        if borders:
            brightest = max(borders, key=lambda region: region.brightness)
            darkest = min(borders, key=lambda region: region.brightness)
            if brightest.brightness - darkest.brightness >= 15.0:
                causes.append(
                    f"Asymmetric borders ({darkest.name} darker than {brightest.name} by "
                    f"{brightest.brightness - darkest.brightness:.1f}). "
                    "Lighting is probably directional, not uniform."
                )

        if not causes:
            causes.append(
                "Frame looks reasonably uniform for a white target. Residual variation may be JPEG "
                "compression or sensor noise."
            )

        causes.append(
            "Camera-side settings to inspect with `v4l2-ctl -d <device> --list-ctrls`: "
            "auto white balance, exposure, gain, gamma, and any lens-shading / WDR options."
        )
        return causes


def _region_stats(name: str, region: np.ndarray) -> RegionStats:
    pixels = region.reshape(-1, 3).astype(np.float32)
    b_mean, g_mean, r_mean = pixels.mean(axis=0)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32).mean()
    return RegionStats(
        name=name,
        pixel_count=int(pixels.shape[0]),
        b_mean=float(b_mean),
        g_mean=float(g_mean),
        r_mean=float(r_mean),
        gray_mean=float(gray),
        rg_ratio=float(r_mean / (g_mean + 1e-6)),
        bg_ratio=float(b_mean / (g_mean + 1e-6)),
    )


def analyze_image(
    image_path: Path,
    border_fraction: float = 0.12,
    image: np.ndarray | None = None,
) -> UniformityReport:
    if image is None:
        image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    margin = max(4, int(min(height, width) * border_fraction))
    center_y, center_x = height // 2, width // 2

    regions = [
        _region_stats("center", image[center_y - margin : center_y + margin, center_x - margin : center_x + margin]),
        _region_stats("top", image[:margin, :]),
        _region_stats("bottom", image[-margin:, :]),
        _region_stats("left", image[:, :margin]),
        _region_stats("right", image[:, -margin:]),
        _region_stats("top_left", image[:margin, :margin]),
        _region_stats("top_right", image[:margin, -margin:]),
        _region_stats("bottom_left", image[-margin:, :margin]),
        _region_stats("bottom_right", image[-margin:, -margin:]),
    ]

    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    radius_norm = radius / radius.max()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    radial_gray: list[tuple[float, float]] = []
    for index in range(10):
        r_lo = index / 10.0
        r_hi = (index + 1) / 10.0
        mask = (radius_norm >= r_lo) & (radius_norm < r_hi)
        if mask.any():
            radial_gray.append((r_lo, float(gray[mask].mean())))

    center_gray = regions[0].gray_mean
    border_regions = regions[1:5]
    border_gray = float(np.mean([region.gray_mean for region in border_regions]))
    center_to_border_drop = center_gray - border_gray
    vignette_score = max(0.0, center_to_border_drop / max(center_gray, 1.0) * 100.0)

    rg_values = [region.rg_ratio for region in regions]
    bg_values = [region.bg_ratio for region in regions]
    max_rg_spread = max(rg_values) - min(rg_values)
    max_bg_spread = max(bg_values) - min(bg_values)
    color_uniformity_score = max(max_rg_spread, max_bg_spread) * 100.0

    return UniformityReport(
        image_path=image_path,
        shape=(height, width),
        regions=regions,
        radial_gray=radial_gray,
        center_to_border_drop=center_to_border_drop,
        max_rg_spread=max_rg_spread,
        max_bg_spread=max_bg_spread,
        vignette_score=vignette_score,
        color_uniformity_score=color_uniformity_score,
    )


def _ratio_map(channel_a: np.ndarray, channel_b: np.ndarray) -> np.ndarray:
    return channel_a / (channel_b + 1e-6)


def save_corrected_image(corrected: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), corrected, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def save_correction_figure(
    original_report: UniformityReport,
    corrected_report: UniformityReport,
    original: np.ndarray,
    corrected: np.ndarray,
    output_path: Path,
) -> None:
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

    b0, g0, r0 = cv2.split(original.astype(np.float32))
    b1, g1, r1 = cv2.split(corrected.astype(np.float32))

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.2)

    for col, (rgb, title) in enumerate([(original_rgb, "Original"), (corrected_rgb, "Corrected")]):
        ax = fig.add_subplot(grid[0, col * 2 : col * 2 + 2])
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis("off")

    ratio_specs = [
        ("R/G before", _ratio_map(r0, g0)),
        ("R/G after", _ratio_map(r1, g1)),
        ("B/G before", _ratio_map(b0, g0)),
        ("B/G after", _ratio_map(b1, g1)),
    ]
    for index, (title, ratio_map) in enumerate(ratio_specs):
        ax = fig.add_subplot(grid[1, index])
        im = ax.imshow(ratio_map, cmap="coolwarm", vmin=0.9, vmax=1.1)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"{original_report.image_path.name} | "
        f"color spread {original_report.color_uniformity_score:.1f}% -> "
        f"{corrected_report.color_uniformity_score:.1f}% | "
        f"vignette {original_report.vignette_score:.1f}% -> "
        f"{corrected_report.vignette_score:.1f}%",
        fontsize=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_correction_tips() -> None:
    print("\nHow to fix color tint (in order of impact):")
    print("  1. Lighting: use diffuse, even illumination across the full FOV.")
    print("  2. Camera AWB: lock white balance after placing a gray/white card in center:")
    print("       v4l2-ctl -d <device> -c white_balance_automatic=0")
    print("       v4l2-ctl -d <device> -c white_balance_temperature=4500   # 2800-6500")
    print("     Temperature is the red/blue axis only; a green cast needs software.")
    print("  2b. Lock exposure too, so reference and real captures match:")
    print("       v4l2-ctl -d <device> -c auto_exposure=1 -c exposure_time_absolute=157")
    print("  3. Software flat-field: this script's --correct step (saved as *_corrected.jpg).")
    print("     Re-capture a white reference under your real lighting and re-run to build gain maps.")
    print("  4. Pipeline: save the .npz flat-field from --save-flat-field and apply it to future captures.")


def save_report_figure(report: UniformityReport, output_path: Path) -> None:
    image = cv2.imread(str(report.image_path))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    b, g, r = cv2.split(image.astype(np.float32))

    ratio_rg = _ratio_map(r, g)
    ratio_bg = _ratio_map(b, g)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    height, width = report.shape
    center_y, center_x = height // 2, width // 2
    yy, xx = np.mgrid[0:height, 0:width]
    radius_norm = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    radius_norm /= radius_norm.max()

    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(rgb)
    ax.set_title("Input image")
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 1])
    im = ax.imshow(gray, cmap="gray")
    ax.set_title("Luminance (gray)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(grid[0, 2])
    radial_x = [point[0] + 0.05 for point in report.radial_gray]
    radial_y = [point[1] for point in report.radial_gray]
    ax.plot(radial_x, radial_y, marker="o")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Normalized radius")
    ax.set_ylabel("Mean gray")
    ax.set_title("Radial vignetting profile")
    ax.grid(True, alpha=0.3)

    for index, (channel, title) in enumerate(zip([r, g, b], ["Red", "Green", "Blue"])):
        ax = fig.add_subplot(grid[1, index])
        im = ax.imshow(channel, cmap="inferno")
        ax.set_title(f"{title} channel")
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(grid[2, 0])
    im = ax.imshow(ratio_rg, cmap="coolwarm", vmin=0.85, vmax=1.15)
    ax.set_title("R/G ratio (flat = uniform color)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(grid[2, 1])
    im = ax.imshow(ratio_bg, cmap="coolwarm", vmin=0.85, vmax=1.15)
    ax.set_title("B/G ratio (flat = uniform color)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(grid[2, 2])
    im = ax.imshow(radius_norm, cmap="magma")
    ax.set_title("Normalized radius mask")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"{report.image_path.name} | vignette={report.vignette_score:.1f}% | "
        f"color spread={report.color_uniformity_score:.1f}%",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_v4l2_controls(device: str) -> None:
    print(f"\nV4L2 controls for {device}:")
    try:
        output = subprocess.check_output(
            ["v4l2-ctl", "-d", device, "--list-ctrls"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        print("  v4l2-ctl not installed (sudo apt install v4l-utils)")
        return
    except subprocess.CalledProcessError as error:
        print(f"  Could not query controls: {error.output.strip()}")
        return

    interesting = (
        "white_balance",
        "wb",
        "exposure",
        "gain",
        "gamma",
        "brightness",
        "contrast",
        "saturation",
        "sharpness",
        "backlight",
        "wdr",
        "shade",
        "hue",
    )
    for line in output.splitlines():
        lower = line.lower()
        if any(keyword in lower for keyword in interesting):
            print(f"  {line.strip()}")


def capture_live(device: str, name: str, output_dir: Path, config: dict, key: str) -> Path:
    rgb_config = config["rgb"]
    camera = RGBCamera(
        device=device,
        name=name,
        width=int(rgb_config["width"]),
        height=int(rgb_config["height"]),
        fps=int(rgb_config["fps"]),
        controls=controls_for(rgb_config, key),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}_live.jpg"

    try:
        camera.open()
        frame = None
        for _ in range(12):
            frame = camera.grab()
            if frame is not None:
                break
        if frame is None:
            raise RuntimeError(f"{name} did not return a frame")
        cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    finally:
        camera.release()

    print(f"Saved live frame to {output_path}")
    return output_path


def resolve_image_paths(args: argparse.Namespace, config: dict) -> list[Path]:
    paths: list[Path] = []

    if args.image:
        paths.extend(Path(path).expanduser().resolve() for path in args.image)

    if args.session:
        session_dir = Path(args.session).expanduser().resolve()
        for name in ("rgb_cam1.jpg", "rgb_cam2.jpg"):
            candidate = session_dir / name
            if candidate.exists():
                paths.append(candidate)

    if args.live:
        live_dir = Path(args.output).expanduser().resolve() / "live_capture"
        paths.append(capture_live(config["rgb"]["cam1"], "rgb_cam1", live_dir, config, "cam1"))
        paths.append(capture_live(config["rgb"]["cam2"], "rgb_cam2", live_dir, config, "cam2"))

    if paths:
        return paths

    default = PROJECT_ROOT / "captures" / "20260723_171707" / "rgb_cam1.jpg"
    if default.exists():
        return [default]

    raise SystemExit(
        "No input image found. Use --image, --session, or --live.\n"
        "Example: python check_color.py --session captures/20260723_171707"
    )


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_flat_field(
    image_path: Path,
    args: argparse.Namespace,
    config: dict,
) -> FlatFieldMaps | None:
    """Load saved maps: --flat-field, else config path matched to rgb_cam1/cam2."""
    if args.flat_field:
        path = Path(args.flat_field).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return FlatFieldMaps.load(path.resolve())

    correction = config.get("rgb", {}).get("color_correction", {})
    stem = image_path.stem.lower()
    key = None
    if "cam1" in stem:
        key = "flat_field_cam1"
    elif "cam2" in stem:
        key = "flat_field_cam2"
    if key is None:
        return None

    rel = correction.get(key)
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"Flat-field not found for {image_path.name}: {path}")
        return None
    return FlatFieldMaps.load(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check RGB color uniformity and likely causes of border tint / vignetting.",
    )
    parser.add_argument(
        "--image",
        action="append",
        help="Path to one RGB image (repeat for multiple).",
    )
    parser.add_argument(
        "--session",
        help="Capture session directory containing rgb_cam1.jpg / rgb_cam2.jpg.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Grab a fresh frame from both RGB cameras using config.yaml.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "color_reports"),
        help="Directory for saved diagnostic plots.",
    )
    parser.add_argument(
        "--border-fraction",
        type=float,
        default=0.12,
        help="Fraction of min(width,height) used as border band thickness.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open matplotlib windows interactively.",
    )
    parser.add_argument(
        "--v4l2",
        action="store_true",
        help="Print relevant V4L2 camera controls from config.yaml.",
    )
    parser.add_argument(
        "--correct",
        dest="correct",
        action="store_true",
        default=True,
        help="Apply flat-field correction and save *_corrected.jpg (default: on).",
    )
    parser.add_argument(
        "--no-correct",
        dest="correct",
        action="store_false",
        help="Skip flat-field correction.",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=0.08,
        help="Gaussian blur sigma as fraction of min(width,height) for flat-field estimation.",
    )
    parser.add_argument(
        "--save-flat-field",
        action="store_true",
        help="Also save per-channel gain maps as *_flat_field.npz for reuse in the pipeline.",
    )
    parser.add_argument(
        "--flat-field",
        help="Path to a *_flat_field.npz (overrides config auto-match for cam1/cam2).",
    )
    parser.add_argument(
        "--no-white-balance",
        dest="white_balance",
        action="store_false",
        default=True,
        help="Build maps without folding in a fixed white balance (shading only).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Correction blend [0,1]. Default: rgb.color_correction.strength from config.yaml.",
    )
    parser.add_argument(
        "--max-gain",
        type=float,
        default=None,
        help="Upper gain clamp. Default: config or 1.35.",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=None,
        help="Lower gain clamp. Default: config or 0.85.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    output_dir = Path(args.output).expanduser().resolve()

    if args.v4l2:
        print_v4l2_controls(config["rgb"]["cam1"])
        print_v4l2_controls(config["rgb"]["cam2"])

    image_paths = resolve_image_paths(args, config)
    reports: list[UniformityReport] = []
    correction_cfg = config.get("rgb", {}).get("color_correction", {})
    strength = float(
        args.strength
        if args.strength is not None
        else correction_cfg.get("strength", 1.0)
    )
    max_gain = float(
        args.max_gain
        if args.max_gain is not None
        else correction_cfg.get("max_gain", DEFAULT_MAX_GAIN)
    )
    min_gain = float(
        args.min_gain
        if args.min_gain is not None
        else correction_cfg.get("min_gain", DEFAULT_MIN_GAIN)
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        report = analyze_image(image_path, border_fraction=args.border_fraction)
        reports.append(report)
        report.print_summary()

        figure_path = output_dir / f"{image_path.stem}_uniformity.png"
        save_report_figure(report, figure_path)
        print(f"\nSaved plot: {figure_path}")

        if args.correct:
            # When saving new maps, always estimate from this image (ignore old .npz).
            if args.save_flat_field:
                flat_field = None
                print(f"Building new flat-field maps from {image_path.name}")
            else:
                flat_field = resolve_flat_field(image_path, args, config)
                if flat_field is not None:
                    balance = "white-balanced" if flat_field.white_balanced else "shading only"
                    print(f"Using saved flat-field for {image_path.name} ({balance})")
                else:
                    print(
                        f"No saved flat-field for {image_path.name}; "
                        "estimating maps from this image"
                    )

            print(
                f"Tuning: strength={strength:.2f}, "
                f"gain=[{min_gain:.2f}, {max_gain:.2f}], "
                f"white_balance={args.white_balance}"
            )
            corrected, flat_field = apply_flat_field_correction(
                image,
                flat_field=flat_field,
                blur_sigma_fraction=args.blur_sigma,
                border_fraction=args.border_fraction,
                strength=strength,
                max_gain=max_gain,
                min_gain=min_gain,
                white_balance=args.white_balance,
            )
            tag = f"s{strength:.2f}"
            corrected_path = output_dir / f"{image_path.stem}_corrected_{tag}.jpg"
            save_corrected_image(corrected, corrected_path)
            print(f"Saved corrected image: {corrected_path}")

            corrected_report = analyze_image(
                corrected_path,
                border_fraction=args.border_fraction,
                image=corrected,
            )
            comparison_path = output_dir / f"{image_path.stem}_correction_{tag}.png"
            save_correction_figure(report, corrected_report, image, corrected, comparison_path)
            print(f"Saved correction comparison: {comparison_path}")

            print(
                "\nAfter flat-field correction:"
                f" color spread {report.color_uniformity_score:.1f}% -> "
                f"{corrected_report.color_uniformity_score:.1f}%,"
                f" vignette {report.vignette_score:.1f}% -> {corrected_report.vignette_score:.1f}%"
            )

            if args.save_flat_field:
                flat_field_path = output_dir / f"{image_path.stem}_flat_field.npz"
                flat_field.save(flat_field_path)
                print(f"Saved flat-field maps: {flat_field_path}")
                print(
                    "  Copy to color_calibration/ and set rgb.color_correction in config.yaml "
                    "to use in capture_pipeline.py"
                )

    if args.correct:
        print_correction_tips()

    if args.show:
        print(
            "\n--show: interactive GUI disabled (OpenCV Qt conflict). "
            f"Open plots under: {output_dir}"
        )

    if len(reports) >= 2:
        print("\nComparison:")
        base = reports[0]
        for report in reports[1:]:
            print(
                f"  {report.image_path.name}: vignette {report.vignette_score:.1f}% "
                f"vs {base.image_path.name} {base.vignette_score:.1f}%, "
                f"color spread {report.color_uniformity_score:.1f}% "
                f"vs {base.color_uniformity_score:.1f}%"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
