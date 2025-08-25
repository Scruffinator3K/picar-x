from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal, Optional


Mode = Literal["auto", "manual", "idle"]


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
        if mode not in ("auto", "manual", "idle"):
            raise ValueError("mode must be 'auto', 'manual', or 'idle'")
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

    def adjust_manual(self, *, speed_delta: Optional[int] = None, steer_delta: Optional[int] = None) -> ManualCommand:
        """Adjust manual command by deltas and persist (latched control).

        - speed_delta: increment/decrement current speed (-100..100 clamp)
        - steer_delta: increment/decrement current steering angle (-30..30 clamp)
        """
        with self._lock:
            if speed_delta is not None:
                try:
                    self._cmd.speed = max(-100, min(100, int(self._cmd.speed + int(speed_delta))))
                except Exception:
                    pass
            if steer_delta is not None:
                try:
                    self._cmd.steer = max(-30, min(30, int(self._cmd.steer + int(steer_delta))))
                except Exception:
                    pass
            return ManualCommand(self._cmd.speed, self._cmd.steer)
