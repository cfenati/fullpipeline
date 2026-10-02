#!/usr/bin/env python3
"""Touchscreen home screen: launch capture, calibration, measurement and registration.

Every stage is an existing script run as a subprocess (see menu_logic.STAGES); this process
only draws screens, so it never imports a camera SDK and a crashing stage cannot take it
down. Only one stage runs at a time, so two stages can never fight over the cameras.

    python menu.py                     # open the menu
    python menu.py --install-shortcut  # put a launcher icon on the desktop, then exit
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, List, Optional, Tuple

from PIL import Image, ImageTk

# Works around a glibc heap-corruption bug ("malloc_consolidate(): unaligned fastbin
# chunk detected") in calibrate_cameras.py/stereo_calibrate.py's native (OpenCV/numpy)
# cleanup code, after all of that run's real work already succeeded. Confirmed present
# both on the host (numpy 1.24.4 + unbounded opencv-contrib-python 5.0.0) AND inside the
# irimager-test container (numpy 2.5.3 + the same opencv 5.0.0) - MALLOC_CHECK_=0 only
# looked like a fix because it tells glibc to silently ignore corruption it still finds,
# not because the corruption wasn't happening; leaving it fully unset lets the default
# allocator's own unconditional consistency check catch it and abort, confirmed on real
# hardware. Every stage runs as a subprocess (see module docstring), so setting this
# before any subprocess.Popen call propagates it down the whole chain.
os.environ.setdefault("MALLOC_CHECK_", "3")

from menu_logic import (
    PROJECT_ROOT, STAGE_ORDER, STAGES, Page, Report, Session, Stage, StageRun,
    build_command, captures_dir_from, find_newest_report, find_sessions, friendly_error,
    group_by_folder, hidden_capture_dirs, install_shortcut, latest_session,
    load_menu_config, paginate, results_dir_from, sanitize_session_name, storage_problem,
    tail_lines, window_for_screen,
)
from touch_controls import apply_touch_style, confirming_command

POLL_MS = 300
FOLDERS_PER_PAGE = 8
SESSIONS_PER_PAGE = 6
NAME_CHIPS = 6
THUMB_BOX = (260, 195)
KEY_ROWS = ("1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm_-")


class MenuApp:
    def __init__(self, project_root: Path = PROJECT_ROOT, maximize: bool = True) -> None:
        self.project_root = project_root
        config = load_menu_config(project_root)
        self.captures_dir = captures_dir_from(config, project_root)
        self.results_dir = results_dir_from(config, project_root)
        self.hidden_dirs = hidden_capture_dirs(config, project_root)

        self.root = tk.Tk()
        self.root.title("FullPipeline")
        self.root.geometry("1280x720")
        apply_touch_style(self.root, maximize=maximize)
        self.body = ttk.Frame(self.root, padding=24)
        self.body.pack(fill=tk.BOTH, expand=True)

        self._photos: List[ImageTk.PhotoImage] = []
        self._run: Optional[StageRun] = None
        self._last_launch: Optional[Tuple[Stage, Optional[Path], Optional[str]]] = None
        self._elapsed = tk.StringVar(value="")
        self._name = tk.StringVar(value="")
        self.show_home()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _reset(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()
        self._photos.clear()

    def _title(self, text: str) -> None:
        ttk.Label(self.body, text=text, style="Title.TLabel").pack(anchor="w", pady=(0, 12))

    def _button(self, parent, text: str, command: Callable[[], None],
                style: str = "Touch.TButton") -> ttk.Button:
        return ttk.Button(parent, text=text, style=style, command=command)

    def _load_photo(self, path: Path, box: Tuple[int, int]) -> Optional[ImageTk.PhotoImage]:
        try:
            with Image.open(path) as image:
                image.draft("RGB", (box[0] * 2, box[1] * 2))  # fast reduced-size JPEG decode
                thumb = image.convert("RGB")
            thumb.thumbnail(box)
            photo = ImageTk.PhotoImage(thumb)
        except (OSError, ValueError):
            return None
        self._photos.append(photo)
        return photo

    def _nav_bar(self, page: Page, goto: Callable[[int], None], back: Callable[[], None]) -> None:
        # Packed before the expanding content so it always keeps its space at the bottom.
        bar = ttk.Frame(self.body)
        bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))
        self._button(bar, "Back", back).pack(side=tk.LEFT)
        if page.count > 1:
            nxt = self._button(bar, "Next >", lambda: goto(page.index + 1))
            nxt.pack(side=tk.RIGHT)
            prev = self._button(bar, "< Prev", lambda: goto(page.index - 1))
            prev.pack(side=tk.RIGHT, padx=(0, 12))
            ttk.Label(bar, text=f"{page.index + 1} / {page.count}", style="Body.TLabel").pack(
                side=tk.RIGHT, padx=16)
            if page.index == 0:
                prev.state(["disabled"])
            if page.index >= page.count - 1:
                nxt.state(["disabled"])

    # ------------------------------------------------------------------ #
    # Screens
    # ------------------------------------------------------------------ #
    def show_home(self, notice: str = "") -> None:
        self._reset()
        self._title("FullPipeline")
        if notice:
            ttk.Label(self.body, text=notice, style="Notice.TLabel", wraplength=1100,
                      justify=tk.LEFT).pack(anchor="w")
        self._button(self.body, "Quit", self.root.destroy).pack(side=tk.BOTTOM, anchor="e", pady=(12, 0))
        grid = ttk.Frame(self.body)
        grid.pack(fill=tk.BOTH, expand=True)
        tiles: List[Tuple[str, Callable[[], None]]] = [
            (STAGES[key].title, lambda key=key: self.choose(STAGES[key])) for key in STAGE_ORDER
        ]
        tiles.append(("Output folder", self.open_output_folder))
        for index, (title, command) in enumerate(tiles):
            self._button(grid, title, command, style="Tile.TButton").grid(
                row=index // 3, column=index % 3, sticky="nsew", padx=10, pady=10)
        grid.columnconfigure((0, 1, 2), weight=1, uniform="tile")
        grid.rowconfigure((0, 1), weight=1, uniform="tile")

    def choose(self, stage: Stage) -> None:
        if stage.naming:  # a stage that writes new captures: refuse if the drive is missing
            problem = storage_problem(self.captures_dir, self.project_root)
            if problem:
                self.show_home(problem)
                return
        if stage.needs_session:
            self.show_session_picker(stage)
        elif stage.naming:
            self.show_name_keyboard(stage)
        else:
            self.start_stage(stage)

    def show_session_picker(self, stage: Stage, folder: Optional[str] = None, page: int = 0) -> None:
        self._reset()
        sessions = find_sessions(self.captures_dir, self.hidden_dirs)
        if folder is None:
            self._show_folder_list(stage, sessions, page)
        else:
            members = [s for s in sessions if s.folder == folder]
            self._show_folder_sessions(stage, members, folder, page)

    def _show_folder_list(self, stage: Stage, sessions: List[Session], page: int) -> None:
        self._title(f"{stage.title}: pick a capture")
        if not sessions:
            self._button(self.body, "Back", self.show_home).pack(side=tk.BOTTOM, anchor="w")
            ttk.Label(self.body, text=storage_problem(self.captures_dir, self.project_root)
                      or "No captures yet. Use Take images first.",
                      style="Body.TLabel", wraplength=1000, justify=tk.LEFT).pack(anchor="w", pady=24)
            return
        newest = latest_session(sessions)
        self._button(self.body, f"Latest capture\n{newest.display}",
                     lambda: self.start_stage(stage, session=newest.path),
                     style="TouchHuge.TButton").pack(fill=tk.X, pady=(0, 16))
        chunk = paginate(group_by_folder(sessions), page, FOLDERS_PER_PAGE)
        self._nav_bar(chunk, lambda p: self.show_session_picker(stage, None, p), self.show_home)
        grid = ttk.Frame(self.body)
        grid.pack(fill=tk.BOTH, expand=True)
        for index, (name, members) in enumerate(chunk.items):
            label = f"{name or 'Unnamed captures'}  ({len(members)})"
            self._button(grid, label, lambda name=name: self.show_session_picker(stage, name)).grid(
                row=index // 2, column=index % 2, sticky="nsew", padx=6, pady=6)
        grid.columnconfigure((0, 1), weight=1, uniform="folders")

    def _show_folder_sessions(self, stage: Stage, members: List[Session], folder: str, page: int) -> None:
        self._title(f"{folder or 'Unnamed captures'}: pick a capture")
        chunk = paginate(members, page, SESSIONS_PER_PAGE)
        self._nav_bar(chunk, lambda p: self.show_session_picker(stage, folder, p),
                      lambda: self.show_session_picker(stage))
        grid = ttk.Frame(self.body)
        grid.pack(fill=tk.BOTH, expand=True)
        for index, session in enumerate(chunk.items):
            card = ttk.Frame(grid)
            card.grid(row=index // 3, column=index % 3, sticky="nsew", padx=6, pady=6)
            card.columnconfigure(0, weight=1)
            photo = self._load_photo(session.path / "rgb_cam1.jpg", THUMB_BOX)
            button = ttk.Button(card, text=session.display, style="Touch.TButton",
                                command=lambda s=session: self.start_stage(stage, session=s.path))
            if photo is not None:
                button.configure(image=photo, compound=tk.TOP)
            button.grid(row=0, column=0, sticky="nsew")
            delete_button = ttk.Button(card, text="Delete")
            delete_button.configure(command=confirming_command(
                self.root, delete_button, "Delete",
                lambda s=session: self._delete_session(s, stage, folder, page)))
            delete_button.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        grid.columnconfigure((0, 1, 2), weight=1, uniform="cards")

    def _delete_session(self, session: Session, stage: Stage, folder: str, page: int) -> None:
        shutil.rmtree(session.path, ignore_errors=True)
        members = [s for s in find_sessions(self.captures_dir, self.hidden_dirs) if s.folder == folder]
        if members:
            self.show_session_picker(stage, folder, page)
        else:
            self.show_session_picker(stage)

    def show_name_keyboard(self, stage: Stage) -> None:
        self._reset()
        self._name.set("")
        self._title("Name this capture (optional)")
        ttk.Label(self.body, textvariable=self._name, style="Title.TLabel",
                  relief="sunken", anchor="w", padding=8).pack(fill=tk.X)
        existing = [name for name, _ in group_by_folder(find_sessions(self.captures_dir, self.hidden_dirs))
                    if name][:NAME_CHIPS]
        if existing:
            chips = ttk.Frame(self.body)
            chips.pack(fill=tk.X, pady=8)
            for name in existing:
                self._button(chips, name, lambda name=name: self._name.set(name)).pack(side=tk.LEFT, padx=4)

        keys = ttk.Frame(self.body)
        keys.pack(expand=True)
        for row, letters in enumerate(KEY_ROWS):
            for column, char in enumerate(letters):
                ttk.Button(keys, text=char, style="Key.TButton", width=3,
                           command=lambda char=char: self._name.set(
                               sanitize_session_name(self._name.get() + char))
                           ).grid(row=row, column=column, padx=3, pady=3)
        ttk.Button(keys, text="Backspace", style="Key.TButton",
                   command=lambda: self._name.set(self._name.get()[:-1])
                   ).grid(row=len(KEY_ROWS), column=0, columnspan=5, sticky="nsew", padx=3, pady=3)
        ttk.Button(keys, text="Clear", style="Key.TButton",
                   command=lambda: self._name.set("")
                   ).grid(row=len(KEY_ROWS), column=5, columnspan=5, sticky="nsew", padx=3, pady=3)

        actions = ttk.Frame(self.body)
        actions.pack(fill=tk.X, pady=(8, 0))
        self._button(actions, "Back", self.show_home).pack(side=tk.LEFT)
        self._button(actions, "Skip (auto-name)", lambda: self.start_stage(stage)).pack(side=tk.RIGHT)
        self._button(actions, "Start", lambda: self.start_stage(
            stage, name=sanitize_session_name(self._name.get()) or None)).pack(side=tk.RIGHT, padx=(0, 12))

    def show_running(self, stage: Stage) -> None:
        self._reset()
        self._title(f"Running: {stage.title}")
        ttk.Label(self.body, style="Body.TLabel", wraplength=1000, justify=tk.LEFT,
                  text="The tool's window opens on top of this screen. Use its Done / "
                       "Finish button when you are finished.").pack(anchor="w", pady=8)
        ttk.Label(self.body, textvariable=self._elapsed, style="Body.TLabel").pack(anchor="w")
        progress = ttk.Progressbar(self.body, mode="indeterminate")
        progress.pack(fill=tk.X, pady=16)
        progress.start(15)
        self._button(self.body, "Stop", self._stop_run, style="TouchHuge.TButton").pack(
            side=tk.BOTTOM, fill=tk.X)
        self._poll(stage)

    def show_error(self, stage: Stage, message: str, details: str) -> None:
        self._reset()
        self._title(f"{stage.title} did not finish")
        ttk.Label(self.body, text=message, style="Body.TLabel", wraplength=1000,
                  justify=tk.LEFT).pack(anchor="w", pady=12)
        actions = ttk.Frame(self.body)
        actions.pack(side=tk.BOTTOM, fill=tk.X)
        box = tk.Text(self.body, height=10, font=("TkFixedFont", 14), wrap="word")
        box.insert("1.0", details or "(no output was recorded)")
        box.configure(state="disabled")

        def toggle() -> None:
            if box.winfo_manager():
                box.pack_forget()
            else:
                box.pack(fill=tk.BOTH, expand=True, pady=8)

        self._button(actions, "Home", self.show_home).pack(side=tk.LEFT)
        self._button(actions, "Retry", self._retry).pack(side=tk.RIGHT)
        self._button(actions, "Show details", toggle).pack(side=tk.RIGHT, padx=(0, 12))

    def show_result(self, stage: Stage, report: Optional[Report]) -> None:
        self._reset()
        self._title(f"{stage.title}: result")
        actions = ttk.Frame(self.body)
        actions.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))
        self._button(actions, "Done", self.show_home).pack(side=tk.LEFT)
        if report is None:
            ttk.Label(self.body, style="Body.TLabel", wraplength=1000, justify=tk.LEFT,
                      text="No result was saved. (The measure tools only save once points "
                           "have been placed before Finish.)").pack(anchor="w", pady=12)
            return
        content = ttk.Frame(self.body)
        content.pack(fill=tk.BOTH, expand=True)
        text = tk.Text(content, font=("TkFixedFont", 16), wrap="none", width=44)
        try:
            text.insert("1.0", report.report_path.read_text(encoding="utf-8", errors="replace"))
        except OSError as error:
            text.insert("1.0", f"Could not read {report.report_path}: {error}")
        text.configure(state="disabled")
        self._button(actions, "Page down", lambda: text.yview_scroll(1, "pages")).pack(side=tk.RIGHT)
        self._button(actions, "Page up", lambda: text.yview_scroll(-1, "pages")).pack(
            side=tk.RIGHT, padx=(0, 12))
        if report.image_path is not None:
            box = (int(self.root.winfo_screenwidth() * 0.5), int(self.root.winfo_screenheight() * 0.6))
            photo = self._load_photo(report.image_path, box)
            if photo is not None:
                ttk.Label(content, image=photo).pack(side=tk.LEFT, padx=(0, 16))
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------ #
    # Running stages
    # ------------------------------------------------------------------ #
    def start_stage(self, stage: Stage, session: Optional[Path] = None, name: Optional[str] = None) -> None:
        self._last_launch = (stage, session, name)
        window = window_for_screen(self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        command = build_command(stage, sys.executable, self.project_root,
                                session=session, name=name, window=window)
        log_path = self.project_root / "logs" / "menu" / f"{time.strftime('%Y%m%d_%H%M%S')}_{stage.key}.log"
        run = StageRun(command, log_path, self.project_root)
        try:
            run.start()
        except OSError as error:
            self.show_error(stage, f"Could not start {stage.title}.", str(error))
            return
        self._run = run
        self._elapsed.set("0 s")
        self.show_running(stage)

    def _stop_run(self) -> None:
        if self._run is not None:
            self._run.stop()
            self._elapsed.set("Stopping...")

    def _retry(self) -> None:
        if self._last_launch is not None:
            stage, session, name = self._last_launch
            self.start_stage(stage, session, name)

    def _poll(self, stage: Stage) -> None:
        run = self._run
        if run is None:
            return
        code = run.poll()
        if code is None:
            if not run.stopped:
                self._elapsed.set(f"{int(run.elapsed())} s")
            self.root.after(POLL_MS, lambda: self._poll(stage))
            return
        self._run = None
        self.root.lift()
        if run.stopped:
            self.show_home(f"{stage.title} stopped.")
            return
        log = run.log_text()
        if code != 0:
            self.show_error(stage, friendly_error(log), tail_lines(log))
            return
        if stage.result:
            self.show_result(stage, find_newest_report(self.results_dir, run.started_at))
            return
        self.show_home(f"{stage.title}: finished.")

    def open_output_folder(self) -> None:
        problem = storage_problem(self.captures_dir, self.project_root)
        if problem:  # never mkdir(parents=True) a missing mount point
            self.show_home(problem)
            return
        self.captures_dir.mkdir(exist_ok=True)
        opener = shutil.which("xdg-open")
        if opener is None:
            # A kiosk/touchscreen image often has no file manager installed at all, so
            # xdg-open itself is missing - Popen would otherwise fail silently (stderr is
            # discarded below) and look like the button does nothing. Show the path at
            # least, since that's still useful without a GUI file browser available.
            self.show_home(f"No file manager is available to open a folder.\n"
                           f"Captures are at: {self.captures_dir}")
            return
        try:
            subprocess.Popen([opener, str(self.captures_dir)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as error:
            self.show_home(f"Could not open the output folder: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Touchscreen home screen for the FullPipeline rig.")
    parser.add_argument("--install-shortcut", action="store_true",
                        help="Write a FullPipeline launcher to ~/Desktop and "
                             "~/.local/share/applications, then exit.")
    args = parser.parse_args()
    if args.install_shortcut:
        for path in install_shortcut(sys.executable, Path(__file__).resolve(), PROJECT_ROOT):
            print(f"Wrote {path}")
        return 0
    MenuApp().root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
