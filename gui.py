from __future__ import annotations

import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from cameras.thermal_camera import ThermalFrame
from touch_controls import SAVE_DEBOUNCE_S, Debouncer, apply_touch_style


def fit_to_box(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    if max_width <= 0 or max_height <= 0:
        return image

    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )


class CaptureGUI:
    """Multi-camera capture interface with toolbar, panels, and status bar."""

    def __init__(
        self,
        output_dir: Path,
        window_size: str = "1400x920",
        touch: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self._touch = touch
        self._save_gate = Debouncer(SAVE_DEBOUNCE_S)
        self._closed = False
        self._save_requested = False
        self._quit_requested = False
        self._capture_count = 0
        self._photos: dict[str, ImageTk.PhotoImage] = {}

        self.root = tk.Tk()
        self.root.title("Multi-Camera Capture Pipeline")
        self.root.geometry(window_size)
        self.root.minsize(*((640, 400) if touch else (1000, 720)))

        self._status_var = tk.StringVar(value="Initializing...")
        self._temp_var = tk.StringVar(value="Mean: -- °C")
        self._range_var = tk.StringVar(value="Range: --")
        self._save_var = tk.StringVar(value="Last save: none")
        self._count_var = tk.StringVar(value="Captures: 0")

        self._build_layout()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.request_quit)

    def _build_layout(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(fill=tk.X)

        if self._touch:
            apply_touch_style(self.root)
            ttk.Button(
                toolbar, text="Take photo", style="TouchHuge.TButton",
                command=self.request_save,
            ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 12))
            ttk.Button(
                toolbar, text="Done", style="Touch.TButton", command=self.request_quit,
            ).pack(side=tk.RIGHT)
        else:
            style = ttk.Style()
            if "clam" in style.theme_names():
                style.theme_use("clam")
            ttk.Button(
                toolbar,
                text="Save Capture",
                command=self.request_save,
            ).pack(side=tk.LEFT, padx=(0, 6))
            ttk.Button(
                toolbar,
                text="Open Output Folder",
                command=self.open_output_folder,
            ).pack(side=tk.LEFT, padx=(0, 6))

            ttk.Button(
                toolbar,
                text="Quit",
                command=self.request_quit,
            ).pack(side=tk.RIGHT)

        feeds = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        feeds.pack(fill=tk.BOTH, expand=True)
        feeds.columnconfigure(0, weight=1, uniform="cam")
        feeds.columnconfigure(1, weight=1, uniform="cam")
        feeds.rowconfigure(0, weight=1, uniform="rows")
        feeds.rowconfigure(1, weight=1, uniform="rows")

        self._rgb1_panel = self._create_panel(feeds, "RGB Camera 1", row=0, column=0)
        self._rgb2_panel = self._create_panel(feeds, "RGB Camera 2", row=0, column=1)
        self._blackfly_panel = self._create_panel(feeds, "FLIR Blackfly", row=1, column=0)
        self._thermal_panel = self._create_panel(feeds, "Thermal Camera", row=1, column=1)
        self._show_blackfly = True

        status = ttk.Frame(self.root, padding=(10, 8))
        status.pack(fill=tk.X)
        ttk.Label(status, textvariable=self._status_var).pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self._temp_var).pack(side=tk.LEFT, padx=(20, 0))
        ttk.Label(status, textvariable=self._range_var).pack(side=tk.LEFT, padx=(20, 0))
        ttk.Label(status, textvariable=self._count_var).pack(side=tk.RIGHT, padx=(20, 0))
        ttk.Label(status, textvariable=self._save_var).pack(side=tk.RIGHT)

    def _create_panel(
        self,
        parent: ttk.Frame,
        title: str,
        row: int,
        column: int,
        columnspan: int = 1,
    ) -> tk.Label:
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.grid(
            row=row,
            column=column,
            columnspan=columnspan,
            sticky="nsew",
            padx=4,
            pady=4,
        )
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        label = tk.Label(frame, anchor=tk.CENTER, background="#161616")
        label.grid(row=0, column=0, sticky="nsew")
        return label

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-s>", lambda _event: self.request_save())
        self.root.bind("<Control-q>", lambda _event: self.request_quit())
        self.root.bind("<Escape>", lambda _event: self.request_quit())
        self.root.bind("s", lambda _event: self.request_save())
        self.root.bind("q", lambda _event: self.request_quit())

    def request_save(self) -> None:
        # A save blocks this loop for seconds, so a double-tap's second tap is only delivered
        # after the save ends; refusing taps for a moment after completion drops it.
        if self._touch and not self._save_gate.ready():
            return
        self._save_requested = True

    def request_quit(self) -> None:
        self._quit_requested = True

    def consume_save_request(self) -> bool:
        if not self._save_requested:
            return False
        self._save_requested = False
        return True

    def should_quit(self) -> bool:
        return self._quit_requested

    def set_status(self, message: str) -> None:
        self._status_var.set(message)
        self.pump()

    def set_warmup_progress(self, remaining: float, frame_count: int) -> None:
        self._status_var.set(
            f"Warming up thermal camera... {remaining:.1f}s remaining ({frame_count} frames)"
        )
        self.pump()

    def note_capture_saved(self, session_dir: Path) -> None:
        self._save_gate.mark()
        self._capture_count += 1
        self._count_var.set(f"Captures: {self._capture_count}")
        self._save_var.set(f"Last save: {session_dir.name}")
        self.set_status(
            "Saved. Tap Take photo for another." if self._touch
            else "Ready — press Save Capture or S"
        )

    def open_output_folder(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(
            ["xdg-open", str(self.output_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def set_blackfly_enabled(self, enabled: bool) -> None:
        self._show_blackfly = enabled
        if not enabled:
            self._blackfly_panel.configure(image="")
            self._photos.pop("blackfly", None)

    def update_frames(
        self,
        rgb1: Optional[np.ndarray],
        rgb2: Optional[np.ndarray],
        thermal_frame: Optional[ThermalFrame],
        blackfly: Optional[np.ndarray] = None,
    ) -> None:
        if self._closed:
            return

        if rgb1 is not None:
            self._update_panel(self._rgb1_panel, "rgb1", rgb1)
        if rgb2 is not None:
            self._update_panel(self._rgb2_panel, "rgb2", rgb2)

        if self._show_blackfly and blackfly is not None:
            self._update_panel(self._blackfly_panel, "blackfly", blackfly)

        if thermal_frame is not None:
            self._update_panel(
                self._thermal_panel,
                "thermal",
                thermal_frame.palette_bgr,
            )
            temps = thermal_frame.temperature_c
            self._temp_var.set(f"Mean: {thermal_frame.mean_temp_c:.2f} °C")
            self._range_var.set(
                f"Range: {float(temps.min()):.1f} – {float(temps.max()):.1f} °C"
            )

        self.pump()

    def _update_panel(self, label: tk.Label, key: str, bgr_image: np.ndarray) -> None:
        if bgr_image is None or bgr_image.size == 0:
            return

        width = label.winfo_width()
        height = label.winfo_height()
        if width <= 1 or height <= 1:
            width, height = label.master.winfo_width(), label.master.winfo_height()
        if width <= 1 or height <= 1:
            width, height = 640, 360

        fitted = fit_to_box(bgr_image, width - 8, height - 8)
        rgb_image = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb_image))
        self._photos[key] = photo
        label.configure(image=photo)

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
