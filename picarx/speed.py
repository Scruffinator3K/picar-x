from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeedLimits:
    max_speed: int = 100
    min_speed: int = 0
    max_accel_per_s: float = 80.0      # units per second
    max_decel_per_s: float = 120.0     # units per second
    emergency_decel_per_s: float = 300.0


class SpeedPlanner:
    """Rate-limited speed planner with distance-aware target limiting.

    Maintains a smoothed commanded speed in the 0..100 range.
    """

    def __init__(self, limits: SpeedLimits | None = None):
        self.limits = limits or SpeedLimits()
        self._current: float = 0.0

    @property
    def current(self) -> int:
        return int(max(self.limits.min_speed, min(self.limits.max_speed, round(self._current))))

    def reset(self, value: int = 0) -> None:
        self._current = float(value)

    def limit_by_distance(
        self,
        distance_cm: float | None,
        base_speed: int,
        *,
        far_cm: float = 80.0,
        slow_cm: float = 35.0,
        stop_cm: float = 18.0,
        min_slow_speed: int = 25,
    ) -> int:
        """Compute a distance-limited target speed.

        - >= far_cm: no reduction
        - stop_cm..slow_cm: ramp 0 -> min_slow_speed
        - slow_cm..far_cm: ramp min_slow_speed -> base_speed
        - <= stop_cm: 0
        - None: no reduction
        """
        base_speed = int(max(self.limits.min_speed, min(self.limits.max_speed, base_speed)))
        if distance_cm is None:
            return base_speed
        d = float(distance_cm)
        if d <= stop_cm:
            return 0
        if d >= far_cm:
            return base_speed
        if d <= slow_cm:
            # map [stop_cm, slow_cm] -> [0, min_slow_speed]
            if slow_cm <= stop_cm:
                return min_slow_speed
            ratio = (d - stop_cm) / (slow_cm - stop_cm)
            return int(ratio * min_slow_speed)
        # map [slow_cm, far_cm] -> [min_slow_speed, base_speed]
        if far_cm <= slow_cm:
            return base_speed
        ratio = (d - slow_cm) / (far_cm - slow_cm)
        return int(min_slow_speed + ratio * max(0, base_speed - min_slow_speed))

    def step(self, target_speed: int, dt: float, *, emergency: bool = False) -> int:
        """Advance the smoothed speed toward target with rate limits."""
        target = float(max(self.limits.min_speed, min(self.limits.max_speed, target_speed)))
        delta = target - self._current
        if delta > 0:
            max_step = self.limits.max_accel_per_s * dt
            self._current += min(delta, max_step)
        elif delta < 0:
            rate = self.limits.emergency_decel_per_s if emergency else self.limits.max_decel_per_s
            max_step = rate * dt
            self._current += max(delta, -max_step)
        # clamp
        self._current = max(self.limits.min_speed, min(self.limits.max_speed, self._current))
        return self.current
