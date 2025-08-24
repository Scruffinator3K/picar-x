from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .logging_setup import init_logger


log = init_logger("picarx.config")


@dataclass
class VisionConfig:
    hsv_lower: tuple[int, int, int] = (35, 40, 40)
    hsv_upper: tuple[int, int, int] = (85, 255, 255)
    roi_y: float = 0.65


@dataclass
class FusionConfig:
    obstacle_stop_cm: float = 18.0
    obstacle_slow_cm: float = 28.0


@dataclass
class SpeedConfig:
    max_speed: int = 100
    min_speed: int = 0
    max_accel_per_s: float = 80.0
    max_decel_per_s: float = 120.0
    emergency_decel_per_s: float = 300.0


@dataclass
class WatchdogConfig:
    heartbeat_timeout_s: float = 0.5
    hard_stop_distance_cm: float = 10.0


@dataclass
class AppConfig:
    vision: VisionConfig = VisionConfig()
    fusion: FusionConfig = FusionConfig()
    speed: SpeedConfig = SpeedConfig()
    watchdog: WatchdogConfig = WatchdogConfig()


def _default_config_path() -> Path:
    # Prefer user config; can be overridden via PICARX_CONFIG
    env = os.getenv("PICARX_CONFIG")
    if env:
        return Path(env)
    return Path(os.path.expanduser("~")) / ".picarx" / "config.json"


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    path = Path(path) if path else _default_config_path()
    try:
        if not path.exists():
            return AppConfig()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # helper to merge dict into dataclass
        def _merge(dc_cls, dc_default, key):
            section = data.get(key, {}) if isinstance(data, dict) else {}
            if not isinstance(section, dict):
                section = {}
            values = {**asdict(dc_default), **section}
            return dc_cls(**values)

        return AppConfig(
            vision=_merge(VisionConfig, VisionConfig(), "vision"),
            fusion=_merge(FusionConfig, FusionConfig(), "fusion"),
            speed=_merge(SpeedConfig, SpeedConfig(), "speed"),
            watchdog=_merge(WatchdogConfig, WatchdogConfig(), "watchdog"),
        )
    except Exception as e:
        log.warning(f"load_config failed: {e}")
        return AppConfig()
