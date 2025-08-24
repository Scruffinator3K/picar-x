from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .logging_setup import init_logger
from .configuration import AppConfig

try:
    # Prefer robot-hat utils for battery reading
    from robot_hat.utils import get_battery_voltage  # type: ignore
except Exception:
    get_battery_voltage = None  # type: ignore


@dataclass
class BatteryStatus:
    voltage_v: float = 0.0
    low: bool = False
    critical: bool = False


class PowerManager:
    def __init__(self, cfg: AppConfig, on_scale_update: Optional[Callable[[float], None]] = None):
        self.log = init_logger("picarx.power")
        self.cfg = cfg
        self._on_scale_update = on_scale_update
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status = BatteryStatus()
        self._scale = 1.0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=0.5)

    def status(self) -> BatteryStatus:
        return self._status

    def speed_scale(self) -> float:
        return float(self._scale)

    def _read_voltage(self) -> float:
        try:
            if get_battery_voltage:
                return float(get_battery_voltage())
        except Exception as e:
            self.log.debug(f"battery read failed: {e}")
        return 0.0

    def _loop(self):
        low_v = float(self.cfg.power.low_voltage_v)
        crit_v = float(self.cfg.power.critical_voltage_v)
        while not self._stop.is_set():
            v = self._read_voltage()
            low = v > 0 and v <= low_v
            critical = v > 0 and v <= crit_v
            self._status = BatteryStatus(voltage_v=v, low=low, critical=critical)
            # dynamic scaling
            scale = 1.0
            if self.cfg.power.dynamic_speed_scale and v > 0 and low:
                # linearly scale between low_v..crit_v -> 1.0..min_scale
                min_s = max(0.3, float(self.cfg.power.min_speed_scale_at_low))
                if v <= crit_v:
                    scale = min_s
                else:
                    span = max(1e-3, low_v - crit_v)
                    scale = min(1.0, max(min_s, min_s + (v - crit_v) * (1.0 - min_s) / span))
            self._scale = scale
            if self._on_scale_update:
                try:
                    self._on_scale_update(scale)
                except Exception:
                    pass
            # Optional: sleep/idle action on critical
            if critical and self.cfg.power.sleep_on_critical:
                # Currently just logs; system-level suspend would need OS integration
                self.log.warning("Critical battery detected; consider switching to idle mode.")
            time.sleep(1.0)
