from __future__ import annotations

import time

from picarx.picarx import Picarx
from picarx.autonomy import AutonomousController
from picarx.configuration import load_config
from picarx.telemetry import TelemetryServer
from picarx.watchdog import SafetyWatchdog


def main():
    cfg = load_config()
    px = Picarx()
    ctrl = AutonomousController(px)

    # Watchdog and telemetry
    wd = SafetyWatchdog(
        stop_fn=px.stop,
        get_distance_fn=px.get_distance,
        heartbeat_timeout_s=cfg.watchdog.heartbeat_timeout_s,
        hard_stop_distance_cm=cfg.watchdog.hard_stop_distance_cm,
    )
    wd.start()

    tel = TelemetryServer(
        addr="0.0.0.0",
        port=8080,
        get_status=lambda: {
            "distance_cm": px.get_distance(),
            "grayscale": px.get_grayscale_data(),
        },
        on_heartbeat=wd.heartbeat,
    )
    tel.start()
    try:
        while True:
            ctrl.tick()
            wd.heartbeat()
            time.sleep(0.02)  # 50 Hz control loop
    except KeyboardInterrupt:
        pass
    finally:
        try:
            px.stop()
        except Exception:
            pass
        try:
            wd.shutdown()
        except Exception:
            pass
        try:
            tel.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
