"""Tkinter GUI for calibrate_live.py: one phase at a time, capture + coverage + fit result.

Same ttk/StringVar/pump() conventions as gui.py's CaptureGUI. Unlike the
overlapping cam1+cam2+stereo 3-card layout this replaces, only the current
phase's camera(s) and fit result are shown - the phase header names which
camera(s) are active right now.
"""

from __future__ import annotations

import subprocess
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from gui import fit_to_box
from touch_controls import CONFIRM_TIMEOUT_S, ConfirmGate, apply_touch_style
from calibration.live_capture import (
    GateStatus,
    PHASE_ORDER,
    Phase,
    SequentialLiveCalibrationSession,
)

# How long the "just captured" flash border stays on screen after a save.
FLASH_DURATION_S = 0.6

COLOR_OPEN = (0, 200, 0)      # BGR green: gate open, this frame will be saved
COLOR_PARTIAL = (0, 165, 255)  # BGR amber: board seen here, but gate not open yet
COLOR_FLASH = (255, 255, 255)  # BGR white: full-frame border flash at capture instant

PHASE_KIND_TITLE = {
    Phase.CAM_A: "intrinsics",
    Phase.CAM_B: "intrinsics",
    Phase.STEREO: "stereo pair",
}


class LiveCaptureGUI:
    def __init__(
        self,
        captures_dir: Path,
        camera_a: str,
        camera_b: str,
        window_size: str = "1500x950",
        touch: bool = False,
    ) -> None:
        self.captures_dir = Path(captures_dir)
        self.camera_a = camera_a
        self.camera_b = camera_b
        self._touch = touch
        self._closed = False
        self._paused = False
        self._refit_requested = False
        self._top_up_requested = False
        self._continue_requested = False
        self._discard_requested = False
        self._discard_worst_requested = False
        self._quit_requested = False
        self._photos: dict = {}
        self._flash_until = 0.0
        self._capture_count = 0
        # Initial guess (this rig's configured RGB resolution); update()
        # overwrites it from the actual frame shape as soon as one arrives.
        self._camera_aspect = 4656.0 / 3496.0

        self.root = tk.Tk()
        self.root.title("Guided Live Calibration")
        self.root.geometry(window_size)
        self.root.minsize(*((640, 400) if touch else (1100, 760)))

        self._header_var = tk.StringVar(value="Opening cameras...")
        self._status_var = tk.StringVar(value="")
        self._count_var = tk.StringVar(value="New captures: 0")
        self._last_session_var = tk.StringVar(value="Last save: none")
        self._ready_var = tk.StringVar(value="not fit yet")
        self._stats_var = tk.StringVar(value="")
        self._quality_var = tk.StringVar(value="")
        self._gaps_var = tk.StringVar(value="")
        self._warnings_var = tk.StringVar(value="")
        self._worst_views_var = tk.StringVar(value="")

        self._build_layout()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.request_quit)

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_layout(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(fill=tk.X)
        if self._touch:
            apply_touch_style(self.root)
            self._build_touch_toolbar(toolbar)
        else:
            style = ttk.Style()
            if "clam" in style.theme_names():
                style.theme_use("clam")
            self._build_desktop_toolbar(toolbar)

        header = ttk.Frame(self.root, padding=(10, 0, 10, 4))
        header.pack(fill=tk.X)
        ttk.Label(header, textvariable=self._header_var, font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        if not self._touch:
            hint = ttk.Frame(self.root, padding=(10, 0, 10, 6))
            hint.pack(fill=tk.X)
            ttk.Label(
                hint,
                text=(
                    "Each phase is fit once, when its target is reached - not per capture. "
                    "READY just means calibrate_cameras.py/stereo_calibrate.py's own quality checks see "
                    "no warnings; Continue works either way, except moving from cam2 into the stereo "
                    "phase, which also needs both mono RMS/coverage over the --max-mono-rms-for-stereo/"
                    "--min-coverage-fraction bars (imprecise intrinsics make a bad stereo fit look like "
                    "a stereo problem). If not ready, 'Capture More' adds views without losing what you "
                    "have, 'Discard & Restart' starts the phase over."
                ),
                foreground="#888888",
                wraplength=1450,
                justify=tk.LEFT,
            ).pack(anchor="w")

        body = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=3, uniform="col")
        body.columnconfigure(1, weight=2, uniform="col")
        body.rowconfigure(0, weight=1)

        self._left = ttk.Frame(body)
        self._left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._frame_a, self._panel_a = self._create_preview_panel(self._left, f"Camera A ({self.camera_a})")
        self._frame_b, self._panel_b = self._create_preview_panel(self._left, f"Camera B ({self.camera_b})")
        self._layout_phase: Optional[Phase] = None
        self._left.bind("<Configure>", lambda _event: self._reflow_preview_panels())
        self._layout_for_phase(Phase.CAM_A)

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._result_card = self._create_result_card(right)

        status = ttk.Frame(self.root, padding=(10, 8))
        status.pack(fill=tk.X)
        ttk.Label(status, textvariable=self._status_var, wraplength=700, justify=tk.LEFT).pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self._last_session_var).pack(side=tk.RIGHT, padx=(20, 0))
        ttk.Label(status, textvariable=self._count_var).pack(side=tk.RIGHT, padx=(20, 0))

    def _build_desktop_toolbar(self, toolbar: ttk.Frame) -> None:
        ttk.Button(toolbar, text="Refit Now (r)", command=self.request_refit).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Capture More (u)", command=self.request_top_up).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Continue (c)", command=self.request_continue).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Discard Worst (w)", command=self.request_discard_worst).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(toolbar, text="Discard & Restart (d)", command=self.request_discard).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self._pause_button = ttk.Button(
            toolbar, text="Pause Auto-Capture (space)", command=self.toggle_pause
        )
        self._pause_button.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Open Captures Folder", command=self.open_captures_folder).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(toolbar, text="Quit (q)", command=self.request_quit).pack(side=tk.RIGHT)

    def _build_touch_toolbar(self, toolbar: ttk.Frame) -> None:
        """Two rows of big buttons; the destructive ones need a second tap."""
        top, bottom = ttk.Frame(toolbar), ttk.Frame(toolbar)
        top.pack(fill=tk.X, pady=(0, 6))
        bottom.pack(fill=tk.X)

        def add(row: ttk.Frame, text: str, command, confirm: bool = False) -> ttk.Button:
            button = ttk.Button(row, text=text, style="Touch.TButton")
            button.configure(command=self._confirming(button, text, command) if confirm else command)
            button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
            return button

        add(top, "Refit Now", self.request_refit)
        add(top, "Capture More", self.request_top_up)
        add(top, "Continue", self.request_continue)
        self._pause_button = add(top, "Pause Auto-Capture", self.toggle_pause)
        add(bottom, "Discard Worst", self.request_discard_worst, confirm=True)
        add(bottom, "Discard & Restart", self.request_discard, confirm=True)
        add(bottom, "Quit", self.request_quit)

    def _confirming(self, button: ttk.Button, label: str, action):
        """Wrap ``action`` so the first tap only relabels the button; a second tap runs it."""
        gate = ConfirmGate()

        def handler() -> None:
            if gate.press():
                button.configure(text=label)
                action()
                return
            button.configure(text="Tap again to confirm")
            self.root.after(int(CONFIRM_TIMEOUT_S * 1000) + 100,
                            lambda: button.configure(text=label))
        return handler

    def _create_preview_panel(self, parent: ttk.Frame, title: str) -> tuple:
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        label = tk.Label(frame, anchor=tk.CENTER, background="#161616", foreground="#666666")
        label.grid(row=0, column=0, sticky="nsew")
        return frame, label

    def _layout_for_phase(self, phase: Phase) -> None:
        """Mono phases get the whole preview area; stereo stacks A above B.

        A mono phase only has one active camera, so it gets the full left
        column instead of being stuck at half height with the other cell
        idle. The stereo phase always shows both, stacked (camera A on top,
        camera B below) rather than side by side - user preference.
        """
        if phase is self._layout_phase:
            return
        self._layout_phase = phase
        self._reflow_preview_panels()

    def _reflow_preview_panels(self) -> None:
        """Size and centre the active panel(s) to exactly match the camera's
        own aspect ratio, using place() instead of grid()'s stretch-to-fill.

        grid()+sticky="nsew" stretches each panel to its cell's shape, which
        is almost never the camera's own (landscape) aspect ratio - the
        image itself was never distorted (fit_to_box already preserves
        aspect), but the label's background showed through as a gap on
        whichever axis had spare room. Placing each panel at its own
        exactly-fitted size removes that gap instead of just hiding it.
        """
        width = self._left.winfo_width()
        height = self._left.winfo_height()
        if width <= 1 or height <= 1:
            return

        phase = self._layout_phase or Phase.CAM_A
        if phase is Phase.CAM_A:
            panels = [self._frame_a]
            self._frame_b.place_forget()
        elif phase is Phase.CAM_B:
            panels = [self._frame_b]
            self._frame_a.place_forget()
        else:
            panels = [self._frame_a, self._frame_b]
        count = len(panels)

        # Aspect (width:height) of the whole stacked block: N frames of the
        # same width, each camera_aspect tall relative to that width, stacked
        # is N times as tall as one frame at that width.
        block_aspect = self._camera_aspect / count
        candidate_width = height * block_aspect
        if candidate_width <= width:
            block_w, block_h = candidate_width, float(height)
        else:
            block_w, block_h = float(width), width / block_aspect

        x0 = (width - block_w) / 2.0
        y0 = (height - block_h) / 2.0
        panel_h = block_h / count
        for index, panel in enumerate(panels):
            panel.place(x=x0, y=y0 + index * panel_h, width=block_w, height=panel_h)

    def _create_result_card(self, parent: ttk.Frame) -> dict:
        frame = ttk.LabelFrame(parent, text="Phase fit", padding=8)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(7, weight=3)

        ready_label = ttk.Label(frame, textvariable=self._ready_var, font=("TkDefaultFont", 11, "bold"))
        ready_label.grid(row=0, column=0, sticky="w")
        self._ready_label = ready_label
        ttk.Label(frame, textvariable=self._stats_var).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(frame, textvariable=self._quality_var, foreground="#888888").grid(
            row=2, column=0, sticky="w", pady=(2, 0)
        )
        ttk.Label(frame, textvariable=self._gaps_var, foreground="#b35900").grid(
            row=3, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Label(
            frame, textvariable=self._warnings_var, wraplength=520, foreground="#b35900", justify=tk.LEFT
        ).grid(row=4, column=0, sticky="nw", pady=(6, 0))
        ttk.Label(
            frame, textvariable=self._worst_views_var, wraplength=520, foreground="#888888", justify=tk.LEFT
        ).grid(row=5, column=0, sticky="nw", pady=(6, 0))

        ttk.Label(frame, text="coverage", foreground="#888888").grid(row=6, column=0, sticky="w", pady=(10, 0))
        coverage_label = tk.Label(frame, anchor=tk.CENTER, background="#161616")
        coverage_label.grid(row=7, column=0, sticky="nsew", pady=(2, 0))
        return {"coverage_label": coverage_label}

    def _bind_shortcuts(self) -> None:
        self.root.bind("r", lambda _event: self.request_refit())
        self.root.bind("u", lambda _event: self.request_top_up())
        self.root.bind("c", lambda _event: self.request_continue())
        self.root.bind("w", lambda _event: self.request_discard_worst())
        self.root.bind("d", lambda _event: self.request_discard())
        self.root.bind("<Control-q>", lambda _event: self.request_quit())
        self.root.bind("<Escape>", lambda _event: self.request_quit())
        self.root.bind("q", lambda _event: self.request_quit())
        self.root.bind("<space>", lambda _event: self.toggle_pause())

    # ------------------------------------------------------------------ #
    # Requests / state
    # ------------------------------------------------------------------ #
    def request_refit(self) -> None:
        self._refit_requested = True

    def consume_refit_request(self) -> bool:
        if not self._refit_requested:
            return False
        self._refit_requested = False
        return True

    def request_top_up(self) -> None:
        self._top_up_requested = True

    def consume_top_up_request(self) -> bool:
        if not self._top_up_requested:
            return False
        self._top_up_requested = False
        return True

    def request_continue(self) -> None:
        self._continue_requested = True

    def consume_continue_request(self) -> bool:
        if not self._continue_requested:
            return False
        self._continue_requested = False
        return True

    def request_discard(self) -> None:
        self._discard_requested = True

    def consume_discard_request(self) -> bool:
        if not self._discard_requested:
            return False
        self._discard_requested = False
        return True

    def request_discard_worst(self) -> None:
        self._discard_worst_requested = True

    def consume_discard_worst_request(self) -> bool:
        if not self._discard_worst_requested:
            return False
        self._discard_worst_requested = False
        return True

    def request_quit(self) -> None:
        self._quit_requested = True

    def should_quit(self) -> bool:
        return self._quit_requested

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        suffix = "" if self._touch else " (space)"
        label = f"Resume Auto-Capture{suffix}" if self._paused else f"Pause Auto-Capture{suffix}"
        self._pause_button.configure(text=label)

    def is_paused(self) -> bool:
        return self._paused

    def open_captures_folder(self) -> None:
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(
            ["xdg-open", str(self.captures_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def set_status(self, message: str) -> None:
        self._status_var.set(message)

    # ------------------------------------------------------------------ #
    # Live preview + overlay + result panel
    # ------------------------------------------------------------------ #
    def update(
        self,
        session: SequentialLiveCalibrationSession,
        preview_a: Optional[np.ndarray],
        preview_b: Optional[np.ndarray],
        gate: Optional[GateStatus],
    ) -> None:
        if self._closed:
            return

        phase = session.phase
        show_a = phase in (Phase.CAM_A, Phase.STEREO)
        show_b = phase in (Phase.CAM_B, Phase.STEREO)
        gate_open = gate.open if gate else False

        frame_for_aspect = preview_a if preview_a is not None else preview_b
        if frame_for_aspect is not None:
            frame_height, frame_width = frame_for_aspect.shape[:2]
            self._camera_aspect = frame_width / frame_height

        self._layout_for_phase(phase)
        self._reflow_preview_panels()

        self._header_var.set(
            f"Phase {PHASE_ORDER.index(phase) + 1}/3: {session.phase_camera_label()} "
            f"{PHASE_KIND_TITLE[phase]} - {session.view_count()}/{session.target()} captured"
        )
        if session.is_fitting():
            self.set_status("Fitting - shelling out to the offline calibration script...")
        elif not gate_open and not session.is_fitting():
            self.set_status(f"board not visible - waiting ({gate.reason if gate else 'no gate yet'})")

        frame_a = self._annotate(preview_a, gate.left if gate else None, gate_open) if show_a else None
        frame_b = self._annotate(preview_b, gate.right if gate else None, gate_open) if show_b else None

        if time.time() < self._flash_until:
            for frame in (frame_a, frame_b):
                if frame is not None:
                    self._draw_flash(frame)

        if frame_a is not None:
            self._update_panel(self._panel_a, "preview_a", frame_a)
        else:
            self._show_placeholder(self._panel_a, "not used this phase")
        if frame_b is not None:
            self._update_panel(self._panel_b, "preview_b", frame_b)
        else:
            self._show_placeholder(self._panel_b, "not used this phase")

        self._update_result_panel(session)
        self.pump()

    @staticmethod
    def _annotate(
        frame: Optional[np.ndarray], observation, gate_open: bool
    ) -> Optional[np.ndarray]:
        if frame is None:
            return None
        annotated = frame.copy()
        if observation is not None:
            color = COLOR_OPEN if gate_open else COLOR_PARTIAL
            cv2.aruco.drawDetectedCornersCharuco(
                annotated,
                observation.image_points,
                observation.corner_ids.reshape(-1, 1),
                color,
            )
        return annotated

    @staticmethod
    def _draw_flash(frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        thickness = max(8, min(width, height) // 40)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), COLOR_FLASH, thickness)

    def note_capture(self, session_label: str) -> None:
        """Call right after tick() actually saved a frame."""
        self._capture_count += 1
        self._count_var.set(f"New captures: {self._capture_count}")
        self._last_session_var.set(f"Last save: {session_label}")
        self._flash_until = time.time() + FLASH_DURATION_S

    def _show_placeholder(self, label: tk.Label, text: str) -> None:
        self._photos.pop(id(label), None)
        label.configure(image="", text=text)

    # ------------------------------------------------------------------ #
    # Phase-fit result panel
    # ------------------------------------------------------------------ #
    def _update_result_panel(self, session: SequentialLiveCalibrationSession) -> None:
        result = session.last_fit()

        quality = session.capture_quality_summary()
        if quality is None:
            self._quality_var.set("")
        else:
            text = (
                f"corners: min {quality.min_corners} / avg {quality.avg_corners:.0f} / "
                f"max {quality.max_corners} (of {quality.total_corners})"
            )
            if quality.thin_count:
                text += (
                    f"  -  {quality.thin_count} thin view(s) below "
                    f"{quality.thin_threshold} corners (partial board - likely cause of a "
                    "distortion coefficient blowing up later)"
                )
            self._quality_var.set(text)

        gaps = session.edge_gaps()
        fraction_pct = session.current_coverage_fraction() * 100.0
        gaps_text = f"coverage: {fraction_pct:.0f}% of frame"
        gaps_text += ", gaps: " + ", ".join(gaps) if gaps else " - every edge reached"
        self._gaps_var.set(gaps_text)

        if session.is_fitting():
            self._ready_var.set("FITTING...")
            self._ready_label.configure(foreground="#888888")
        elif result is None:
            self._ready_var.set("not fit yet")
            self._ready_label.configure(foreground="#888888")
        elif result.ready:
            self._ready_var.set("READY")
            self._ready_label.configure(foreground="#1a7f1a")
        else:
            self._ready_var.set("not ready")
            self._ready_label.configure(foreground="#b35900")

        if result is None:
            self._stats_var.set("")
            self._warnings_var.set("")
            self._worst_views_var.set("")
        elif result.error:
            self._stats_var.set(result.error)
            self._warnings_var.set("")
            self._worst_views_var.set("")
        else:
            if result.intrinsics is not None:
                self._stats_var.set(
                    f"views: {result.intrinsics.views_used}   "
                    f"RMS: {result.intrinsics.reprojection_error_px:.2f} px"
                )
            elif result.extrinsics is not None:
                baseline_mm = result.extrinsics.baseline_m * 1000.0
                self._stats_var.set(
                    f"views: {result.extrinsics.views_used}   "
                    f"RMS: {result.extrinsics.reprojection_error_px:.2f} px   "
                    f"baseline: {baseline_mm:.2f} mm"
                )
            self._warnings_var.set("\n".join(f"- {w}" for w in result.warnings))
            if result.worst_views:
                shown = ", ".join(f"{label} ({error:.2f}px)" for label, error in result.worst_views[:5])
                self._worst_views_var.set(f"worst views: {shown}  ->  Discard Worst (w) to prune")
            else:
                self._worst_views_var.set("")

        coverage = session.current_coverage()
        if coverage is not None:
            self._update_coverage_thumbnail(self._result_card["coverage_label"], coverage)

    def _update_coverage_thumbnail(self, label: tk.Label, coverage: np.ndarray) -> None:
        width = label.winfo_width()
        height = label.winfo_height()
        if width <= 1 or height <= 1:
            width, height = 480, 320
        fitted = fit_to_box(coverage, width - 4, height - 4)
        rgb_image = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb_image))
        self._photos[id(label)] = photo
        label.configure(image=photo, text="")

    # ------------------------------------------------------------------ #
    # Low-level panel update, pump, lifecycle
    # ------------------------------------------------------------------ #
    def _update_panel(self, label: tk.Label, key: str, bgr_image: np.ndarray) -> None:
        if bgr_image is None or bgr_image.size == 0:
            return
        width = label.winfo_width()
        height = label.winfo_height()
        if width <= 1 or height <= 1:
            width, height = label.master.winfo_width(), label.master.winfo_height()
        if width <= 1 or height <= 1:
            width, height = 640, 480
        fitted = fit_to_box(bgr_image, width - 8, height - 8)
        rgb_image = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb_image))
        self._photos[key] = photo
        label.configure(image=photo, text="")

    def pump(self) -> None:
        if self._closed:
            return
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.root.destroy()

    def is_open(self) -> bool:
        if self._closed:
            return False
        try:
            return bool(self.root.winfo_exists())
        except tk.TclError:
            return False
