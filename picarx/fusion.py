from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import time
from collections import deque

from .logging_setup import init_logger
from .vision import Vision


@dataclass
class FusionState:
    ts: float
    distance_cm: Optional[float]
    grayscale: Optional[List[float]]
    line_error: Optional[float]
    cliff: bool


class SensorFusion:
    """Combines sensor readings into a compact state for control logic."""

    def __init__(self, px, obstacle_stop_cm: float = 40.0, obstacle_slow_cm: float = 60.0, vision: Optional[Vision] = None):
        self.px = px
        self.log = init_logger("picarx.fusion")
        self.obstacle_stop_cm = obstacle_stop_cm
        self.obstacle_slow_cm = obstacle_slow_cm
        self._last: Optional[FusionState] = None
        self.vision = vision or Vision(use_capture=False)
        
        # Enhanced filtering for noisy surfaces like wood grain
        self._line_error_history = deque(maxlen=10)
        self._grayscale_history = deque(maxlen=5)
        self._last_stable_error = 0.0
        self._error_change_threshold = 0.3  # Ignore small changes
        self._min_contrast_threshold = 200  # Minimum contrast to consider line detection valid

    def _compute_line_error(self, gs: List[float]) -> Optional[float]:
        if not gs or len(gs) != 3:
            return None
        
        left, mid, right = gs
        
        # Check if there's sufficient contrast to indicate a real line
        max_val = max(left, mid, right)
        min_val = min(left, mid, right)
        contrast = max_val - min_val
        
        if contrast < self._min_contrast_threshold:
            # Insufficient contrast - likely on uniform surface (wood grain noise)
            # Return None to disable line following on noisy surfaces
            self.log.debug(f"Low contrast ({contrast:.1f}) - disabling line following")
            return None
        
        # Store raw grayscale for trend analysis
        self._grayscale_history.append([left, mid, right])
        
        # Calculate basic error
        denom = max(1e-3, left + mid + right)
        raw_error = (right - left) / denom  # right positive => steer right
        raw_error = max(-1.0, min(1.0, raw_error))
        
        # Apply smoothing and stability filtering
        self._line_error_history.append(raw_error)
        
        if len(self._line_error_history) < 3:
            return None  # Need some history for stability
        
        # Use moving average for smoothing
        recent_errors = list(self._line_error_history)[-5:]
        smoothed_error = sum(recent_errors) / len(recent_errors)
        
        # Only update if the change is significant (reduces jitter from wood grain)
        error_change = abs(smoothed_error - self._last_stable_error)
        
        if error_change > self._error_change_threshold:
            self._last_stable_error = smoothed_error
            self.log.debug(f"Line error updated: {smoothed_error:.3f} (contrast: {contrast:.1f})")
            return smoothed_error
        else:
            # Small change - likely noise, keep previous stable value
            self.log.debug(f"Line error stable: {self._last_stable_error:.3f} (rejected change: {error_change:.3f})")
            return self._last_stable_error

    def update(self) -> FusionState:
        dist = self.px.get_distance()
        gs = self.px.get_grayscale_data()
        cliff = False
        try:
            cliff = self.px.get_cliff_status(gs)
        except Exception:
            cliff = False
        line_err = self._compute_line_error(gs) if gs else None
        # If no grayscale line error (no painted lines), try vision-based road edge error
        if line_err is None and self.vision and self.vision.last():
            v = self.vision.last() or {}
            ve = v.get("line_error")
            if isinstance(ve, (float, int)):
                line_err = float(ve)
        self._last = FusionState(ts=time.time(), distance_cm=dist, grayscale=gs, line_error=line_err, cliff=cliff)
        return self._last

    def last(self) -> Optional[FusionState]:
        return self._last
