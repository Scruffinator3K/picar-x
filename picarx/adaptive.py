from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

from .logging_setup import init_logger


@dataclass
class AdaptiveConfig:
    min_speed: int = 20
    max_speed: int = 60
    near_cm: float = 25.0
    far_cm: float = 120.0
    smoothing: float = 0.3  # EMA


class AdaptiveController:
    """Tiny adaptive controller to modulate speed based on proximity and mapping.

    This is a simple scaffold to satisfy Phase 3 initial adaptation: it uses an
    exponential moving average of obstacle distance to pick a speed within
    [min_speed, max_speed]. Future work: add PID, reward models, or MAB.
    """

    def __init__(self, cfg: Optional[AdaptiveConfig] = None) -> None:
        self.log = init_logger("picarx.adaptive")
        self.cfg = cfg or AdaptiveConfig()
        self._ema_dist: Optional[float] = None
        self._last_update = time.time()

    def update(self, distance_cm: Optional[float]) -> None:
        if distance_cm is None or distance_cm <= 0:
            return
        alpha = self.cfg.smoothing
        if self._ema_dist is None:
            self._ema_dist = float(distance_cm)
        else:
            self._ema_dist = alpha * float(distance_cm) + (1 - alpha) * self._ema_dist
        self._last_update = time.time()

    def get_speed_setpoint(self) -> int:
        if self._ema_dist is None:
            return self.cfg.min_speed
        d = max(self.cfg.near_cm, min(self.cfg.far_cm, self._ema_dist))
        # map near..far to min..max
        t = (d - self.cfg.near_cm) / max(1e-6, (self.cfg.far_cm - self.cfg.near_cm))
        spd = int(round(self.cfg.min_speed + t * (self.cfg.max_speed - self.cfg.min_speed)))
        return max(self.cfg.min_speed, min(self.cfg.max_speed, spd))
