from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class VideoPopupManager:
    """Manage two small looping meme video windows with optional audio.

    OpenCV tidak bisa memutar audio. Untuk suara, class ini memakai ffpyplayer
    jika package tersebut sudah terinstall. Jika belum ada, video tetap tampil
    tetapi terminal memberi instruksi install dependency.
    """

    WINDOW_NAMES = ["Scuba Meme 1", "Scuba Meme 2"]

    def __init__(self, video_paths: list[str], window_size: tuple[int, int] = (260, 360)) -> None:
        self.video_paths = [Path(path) for path in video_paths]
        self.window_size = window_size  # width, height
        self.captures: list[cv2.VideoCapture | None] = [None for _ in self.video_paths]
        self.showing_indices: set[int] = set()
        self._reported_missing: set[int] = set()
        self._reported_audio_missing = False
        self.manual_mode = False
        self.audio_players: dict[int, Any] = {}

        for index, path in enumerate(self.video_paths):
            if not path.exists():
                self._print_missing(index)

    def _display_path(self, index: int) -> str:
        path = self.video_paths[index]
        try:
            return path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return path.as_posix()

    def _print_missing(self, index: int) -> None:
        if index in self._reported_missing:
            return
        print(f"Missing video file: {self._display_path(index)}")
        self._reported_missing.add(index)

    def _open_capture(self, index: int) -> bool:
        if index < 0 or index >= len(self.video_paths):
            return False

        path = self.video_paths[index]
        if not path.exists():
            self._print_missing(index)
            return False

        capture = self.captures[index]
        if capture is not None and capture.isOpened():
            return True

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            print(f"Failed to open video file: {self._display_path(index)}")
            capture.release()
            return False

        self.captures[index] = capture
        return True

    def show_all(self, manual: bool = False) -> None:
        """Show both meme videos. Audio may overlap by design."""
        if manual:
            self.manual_mode = True
        for index in range(len(self.video_paths)):
            self.show_one(index, manual=manual)

    def show_one(self, index: int, manual: bool = False) -> None:
        if manual:
            self.manual_mode = True
        if index in self.showing_indices and self._open_capture(index):
            return

        if not self._open_capture(index):
            return

        window_name = self.WINDOW_NAMES[index]
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, *self.window_size)
        self.showing_indices.add(index)
        self._start_audio(index)

    def hide_all(self) -> None:
        self.manual_mode = False
        for index in list(self.showing_indices):
            self._hide_one(index)
        self.showing_indices.clear()

    def _hide_one(self, index: int) -> None:
        try:
            cv2.destroyWindow(self.WINDOW_NAMES[index])
        except cv2.error:
            pass

        self._stop_audio(index)
        capture = self.captures[index]
        if capture is not None:
            capture.release()
            self.captures[index] = None
        self.showing_indices.discard(index)

    def _start_audio(self, index: int) -> None:
        self._stop_audio(index)
        try:
            from ffpyplayer.player import MediaPlayer
        except ModuleNotFoundError:
            if not self._reported_audio_missing:
                print("Audio disabled: install ffpyplayer with `pip install -r requirements.txt`.")
                self._reported_audio_missing = True
            return

        self.audio_players[index] = MediaPlayer(str(self.video_paths[index]))

    def _stop_audio(self, index: int) -> None:
        player = self.audio_players.pop(index, None)
        if player is not None:
            try:
                player.close_player()
            except Exception:
                pass

    def _pump_audio(self, index: int) -> str | None:
        player = self.audio_players.get(index)
        if player is None:
            return None
        try:
            _, status = player.get_frame()
            return status
        except Exception:
            self._stop_audio(index)
            return None

    def _restart_video(self, index: int) -> None:
        capture = self.captures[index]
        if capture is not None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._start_audio(index)

    def update(self) -> None:
        for index in list(self.showing_indices):
            capture = self.captures[index]
            if capture is None or not capture.isOpened():
                self._hide_one(index)
                continue

            audio_status = self._pump_audio(index)
            ok, frame = capture.read()
            if audio_status == "eof" or not ok or frame is None:
                self._restart_video(index)
                capture = self.captures[index]
                if capture is None:
                    continue
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue

            cv2.imshow(self.WINDOW_NAMES[index], self._resize_with_aspect_ratio(frame))

    def _resize_with_aspect_ratio(self, frame):
        target_width, target_height = self.window_size
        height, width = frame.shape[:2]
        if width == 0 or height == 0:
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)

        scale = min(target_width / width, target_height / height)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - resized_width) // 2
        y_offset = (target_height - resized_height) // 2
        canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
        return canvas

    def release(self) -> None:
        self.hide_all()
        for index, capture in enumerate(self.captures):
            if capture is not None:
                capture.release()
                self.captures[index] = None

    def is_showing(self) -> bool:
        return bool(self.showing_indices)