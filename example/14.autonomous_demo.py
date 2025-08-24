from __future__ import annotations

import time

from picarx.picarx import Picarx
from picarx.autonomy import AutonomousController
from picarx.configuration import load_config
from picarx.telemetry import TelemetryServer
from picarx.watchdog import SafetyWatchdog
from picarx.mode import ModeManager
from picarx.vision import Vision


def main():
    cfg = load_config()
    px = Picarx()
    ctrl = AutonomousController(px)
    vision = Vision(use_capture=True)
    modes = ModeManager(initial="auto")

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
            "mode": modes.get_mode(),
        },
        on_heartbeat=wd.heartbeat,
        on_set_mode=modes.set_mode,
        on_manual=lambda sp, st: modes.set_manual_command(speed=sp, steer=st),
        get_snapshot=(
            (lambda: (vision.last() and None))  # placeholder if no camera API
            if vision is None else None
        ),
    )
    tel.start()
    try:
        while True:
            if modes.get_mode() == "manual":
                cmd = modes.get_manual_command()
                px.set_dir_servo_angle(cmd.steer)
                if cmd.speed >= 0:
                    px.forward(cmd.speed)
                else:
                    px.backward(-cmd.speed)
            else:
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
