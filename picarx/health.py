from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Deque, Optional, Tuple
from collections import deque


@dataclass
class RollingStat:
    size: int
    values: Deque[float] = field(default_factory=deque)

    def add(self, v: float) -> None:
        if len(self.values) >= self.size:
            self.values.popleft()
        self.values.append(v)

    def avg(self) -> Optional[float]:
        return mean(self.values) if self.values else None

    def min(self) -> Optional[float]:
        return min(self.values) if self.values else None

    def max(self) -> Optional[float]:
        return max(self.values) if self.values else None


@dataclass
class SensorHealth:
    name: str
    range_min: float
    range_max: float
    timeout_s: float = 0.2
    rolling: RollingStat = field(default_factory=lambda: RollingStat(20))
    last_value: Optional[float] = None
    last_time: float = field(default_factory=time.time)
    failures: int = 0

    def validate(self, value: float) -> Tuple[bool, Optional[str]]:
        now = time.time()
        if value is None:
            self.failures += 1
            return False, f"{self.name}: None value"
        if not (self.range_min <= value <= self.range_max):
            self.failures += 1
            return False, f"{self.name}: out of range {value} not in [{self.range_min}, {self.range_max}]"

        # Update
        self.rolling.add(value)
        self.last_value = value
        self.last_time = now
        return True, None

    def is_stale(self) -> bool:
        return (time.time() - self.last_time) > self.timeout_s

    def summary(self) -> dict:
        return {
            "name": self.name,
            "last": self.last_value,
            "avg": self.rolling.avg(),
            "min": self.rolling.min(),
            "max": self.rolling.max(),
            "failures": self.failures,
            "stale": self.is_stale(),
        }
