from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal, Optional


Mode = Literal["auto", "manual"]


@dataclass
class ManualCommand:
    speed: int = 0   # -100..100, sign indicates backward/forward
    steer: int = 0   # -30..30 degrees


class ModeManager:
    def __init__(self, initial: Mode = "auto"):
        self._mode: Mode = initial
        self._lock = threading.Lock()
        self._cmd = ManualCommand()

    def get_mode(self) -> Mode:
        with self._lock:
            return self._mode

    def set_mode(self, mode: Mode) -> None:
        if mode not in ("auto", "manual"):
            raise ValueError("mode must be 'auto' or 'manual'")
        with self._lock:
            self._mode = mode

    def set_manual_command(self, *, speed: Optional[int] = None, steer: Optional[int] = None) -> ManualCommand:
        with self._lock:
            if speed is not None:
                self._cmd.speed = max(-100, min(100, int(speed)))
            if steer is not None:
                self._cmd.steer = max(-30, min(30, int(steer)))
            return ManualCommand(self._cmd.speed, self._cmd.steer)

    def get_manual_command(self) -> ManualCommand:
        with self._lock:
            return ManualCommand(self._cmd.speed, self._cmd.steer)
