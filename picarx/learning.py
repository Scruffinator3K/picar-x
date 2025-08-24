from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from .logging_setup import init_logger


@dataclass
class BanditConfig:
    eps: float = 0.1
    arms: List[float] = None  # speed scale candidates

    def __post_init__(self):
        if self.arms is None:
            self.arms = [0.7, 0.8, 0.9, 1.0]


class EpsilonGreedyBandit:
    def __init__(self, cfg: BanditConfig) -> None:
        self.cfg = cfg
        self.log = init_logger("picarx.learning")
        self.value: Dict[float, float] = {a: 0.0 for a in cfg.arms}
        self.count: Dict[float, int] = {a: 0 for a in cfg.arms}

    def select(self) -> float:
        if random.random() < self.cfg.eps:
            a = random.choice(self.cfg.arms)
            self.log.debug(f"explore arm={a}")
            return a
        # exploit
        a = max(self.cfg.arms, key=lambda k: self.value[k])
        return a

    def update(self, arm: float, reward: float) -> None:
        n = self.count[arm] + 1
        v = self.value[arm]
        self.count[arm] = n
        self.value[arm] = v + (reward - v) / n
