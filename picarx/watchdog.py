from __future__ import annotations

import time
import threading
from typing import Callable, Optional

from .logging_setup import init_logger


class SafetyWatchdog:
    """Background safety watchdog.

    - If no heartbeat within timeout, stops motors.
    - If distance is below a hard-stop threshold, stops motors immediately.
    """

    def __init__(
        self,
        stop_fn: Callable[[], None],
        get_distance_fn: Callable[[], Optional[float]],
        heartbeat_timeout_s: float = 0.5,
    hard_stop_distance_cm: float = 8.0,
        logger_name: str = "picarx.watchdog",
    ) -> None:
        self.log = init_logger(logger_name)
        self._stop_fn = stop_fn
        self._get_distance = get_distance_fn
        self._timeout = float(heartbeat_timeout_s)
        self._hard_stop_cm = float(hard_stop_distance_cm)

        self._last_beat = time.time()
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def start(self):
        if self._th is None:
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()
            self.log.info("SafetyWatchdog started")

    def shutdown(self):
        self._stop.set()
        if self._th and self._th.is_alive():
            self._th.join(timeout=0.5)
        self.log.info("SafetyWatchdog stopped")

    def heartbeat(self):
        self._last_beat = time.time()

    def _loop(self):
        while not self._stop.is_set():
            now = time.time()
            # Hard stop check
            try:
                d = self._get_distance()
            except Exception:
                d = None
            if d is not None and d <= self._hard_stop_cm:
                self.log.warning(f"Hard-stop distance reached ({d:.1f} cm). Stopping.")
                try:
                    # Use slow_stop if the stop function belongs to a Picarx instance
                    owner = getattr(self._stop_fn, "__self__", None)
                    slow = getattr(owner, "slow_stop", None)
                    if callable(slow):
                        slow(duration_s=0.4, steps=10)
                    else:
                        self._stop_fn()
                except Exception:
                    pass
            # Heartbeat timeout
            if now - self._last_beat > self._timeout:
                self.log.warning("Heartbeat timeout. Stopping motors.")
                try:
                    owner = getattr(self._stop_fn, "__self__", None)
                    slow = getattr(owner, "slow_stop", None)
                    if callable(slow):
                        slow(duration_s=0.5, steps=12)
                    else:
                        self._stop_fn()
                except Exception:
                    pass
                self._last_beat = now  # avoid spamming
            time.sleep(0.02)
