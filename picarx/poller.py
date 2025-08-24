from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional, Tuple

from .logging_setup import init_logger


class SensorPoller:
    """Background polling for sensors with independent intervals.

    It keeps latest values in-memory and provides a light staleness check.
    """

    def __init__(
        self,
        ultrasonic_read: Optional[Callable[[], Optional[float]]] = None,
        grayscale_read: Optional[Callable[[], Optional[List[float]]]] = None,
        ultrasonic_interval: float = 0.03,  # ~33 Hz
        grayscale_interval: float = 0.02,   # ~50 Hz
        logger_name: str = "picarx.poller",
    ) -> None:
        self.log = init_logger(logger_name)
        self._stop = threading.Event()

        self.ultrasonic_read = ultrasonic_read
        self.grayscale_read = grayscale_read

        self.ultra_interval = max(0.005, float(ultrasonic_interval)) if ultrasonic_read else None
        self.gray_interval = max(0.005, float(grayscale_interval)) if grayscale_read else None

        self._ultra_thread: Optional[threading.Thread] = None
        self._gray_thread: Optional[threading.Thread] = None

        self._ultra_lock = threading.Lock()
        self._gray_lock = threading.Lock()

        self._ultra_last: Optional[float] = None
        self._ultra_last_ts: float = 0.0

        self._gray_last: Optional[List[float]] = None
        self._gray_last_ts: float = 0.0

    # lifecycle
    def start(self) -> None:
        if self.ultra_interval is not None and self._ultra_thread is None:
            self._ultra_thread = threading.Thread(target=self._ultra_loop, daemon=True)
            self._ultra_thread.start()
        if self.gray_interval is not None and self._gray_thread is None:
            self._gray_thread = threading.Thread(target=self._gray_loop, daemon=True)
            self._gray_thread.start()
        self.log.info("SensorPoller started")

    def stop(self, timeout: float = 0.5) -> None:
        self._stop.set()
        for t in (self._ultra_thread, self._gray_thread):
            if t and t.is_alive():
                t.join(timeout=timeout)
        self.log.info("SensorPoller stopped")

    # loops
    def _ultra_loop(self) -> None:
        next_ts = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now >= next_ts:
                try:
                    v = self.ultrasonic_read() if self.ultrasonic_read else None
                    if v is not None:
                        with self._ultra_lock:
                            self._ultra_last = float(v)
                            self._ultra_last_ts = now
                except Exception as e:
                    # errors already handled by SafeUltrasonic; keep loop alive
                    self.log.debug(f"ultrasonic poll error: {e}")
                next_ts = now + (self.ultra_interval or 0.05)
            time.sleep(0.001)

    def _gray_loop(self) -> None:
        next_ts = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now >= next_ts:
                try:
                    vals = self.grayscale_read() if self.grayscale_read else None
                    if vals is not None:
                        with self._gray_lock:
                            self._gray_last = list(vals)
                            self._gray_last_ts = now
                except Exception as e:
                    self.log.debug(f"grayscale poll error: {e}")
                next_ts = now + (self.gray_interval or 0.05)
            time.sleep(0.001)

    # getters
    def get_distance(self, max_age: float = 0.15) -> Optional[float]:
        with self._ultra_lock:
            if self._ultra_last and (time.time() - self._ultra_last_ts) <= max_age:
                return self._ultra_last
        return None

    def get_grayscale(self, max_age: float = 0.1) -> Optional[List[float]]:
        with self._gray_lock:
            if self._gray_last and (time.time() - self._gray_last_ts) <= max_age:
                return list(self._gray_last)
        return None
