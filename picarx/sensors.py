from __future__ import annotations

from typing import List, Optional, Tuple
import time

from robot_hat import Ultrasonic, Grayscale_Module, Pin, ADC

from .health import SensorHealth
from .logging_setup import init_logger


log = init_logger("picarx.sensors")


class SafeUltrasonic:
    """Ultrasonic wrapper with validation and simple debouncing."""

    def __init__(self, trig: str, echo: str, min_cm: float = 2.0, max_cm: float = 400.0):
        self.sensor = Ultrasonic(Pin(trig), Pin(echo, mode=Pin.IN, pull=Pin.PULL_DOWN))
        self.health = SensorHealth("ultrasonic", min_cm, max_cm, timeout_s=0.3)
        self._last_good: Optional[float] = None
        self._last_time: float = 0.0

    def read(self) -> Optional[float]:
        try:
            v = float(self.sensor.read())
        except Exception as e:
            log.warning(f"ultrasonic read error: {e}")
            return self._last_good

        ok, err = self.health.validate(v)
        if ok:
            # basic debounce: ignore spikes with absurd rate-of-change
            now = time.time()
            if self._last_good is not None and now - self._last_time < 0.02:
                # within 20ms window, accept only small changes
                if abs(v - self._last_good) > 50:  # cm jump guard
                    log.debug(f"ultrasonic spike filtered: {self._last_good} -> {v}")
                    return self._last_good
            self._last_good = v
            self._last_time = now
            return v
        else:
            log.debug(err)
            return self._last_good


class SafeGrayscale:
    """Grayscale wrapper that guards read errors and provides status with validation."""

    def __init__(self, a0: str, a1: str, a2: str, line_range: Tuple[float, float] = (0, 4095)):
        adc0, adc1, adc2 = [ADC(pin) for pin in (a0, a1, a2)]
        self.sensor = Grayscale_Module(adc0, adc1, adc2, reference=None)
        self.health = [
            SensorHealth("grayscale_0", *line_range),
            SensorHealth("grayscale_1", *line_range),
            SensorHealth("grayscale_2", *line_range),
        ]
        self._last: Optional[List[float]] = None

    def reference(self, ref: List[float]):
        self.sensor.reference(ref)

    def read(self) -> Optional[List[float]]:
        try:
            vals = list(self.sensor.read())
        except Exception as e:
            log.warning(f"grayscale read error: {e}")
            return self._last
        ok_all = True
        for i, v in enumerate(vals):
            ok, _ = self.health[i].validate(float(v))
            ok_all = ok_all and ok
        if ok_all:
            self._last = vals
        return self._last or vals

    def read_status(self, values: List[float]):
        try:
            return self.sensor.read_status(values)
        except Exception as e:
            log.warning(f"grayscale status error: {e}")
            return [1, 1, 1]  # assume background to fail-safe stop
