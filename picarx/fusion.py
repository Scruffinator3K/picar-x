from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import time

from .logging_setup import init_logger


@dataclass
class FusionState:
    ts: float
    distance_cm: Optional[float]
    grayscale: Optional[List[float]]
    line_error: Optional[float]
    cliff: bool


class SensorFusion:
    """Combines sensor readings into a compact state for control logic."""

    def __init__(self, px, obstacle_stop_cm: float = 18.0, obstacle_slow_cm: float = 28.0):
        self.px = px
        self.log = init_logger("picarx.fusion")
        self.obstacle_stop_cm = obstacle_stop_cm
        self.obstacle_slow_cm = obstacle_slow_cm
        self._last: Optional[FusionState] = None

    def _compute_line_error(self, gs: List[float]) -> Optional[float]:
        if not gs or len(gs) != 3:
            return None
        left, mid, right = gs
        denom = max(1e-3, left + mid + right)
        # Normalize and compute a simple left-right balance
        err = (right - left) / denom  # right positive => steer right
        # clamp
        return max(-1.0, min(1.0, err))

    def update(self) -> FusionState:
        dist = self.px.get_distance()
        gs = self.px.get_grayscale_data()
        cliff = False
        try:
            cliff = self.px.get_cliff_status(gs)
        except Exception:
            cliff = False
        line_err = self._compute_line_error(gs) if gs else None
        self._last = FusionState(ts=time.time(), distance_cm=dist, grayscale=gs, line_error=line_err, cliff=cliff)
        return self._last

    def last(self) -> Optional[FusionState]:
        return self._last
