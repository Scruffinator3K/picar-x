from __future__ import annotations

import time

from picarx.picarx import Picarx
from picarx.autonomy import AutonomousController
from picarx.configuration import load_config
from picarx.telemetry import TelemetryServer
from picarx.watchdog import SafetyWatchdog
from picarx.mode import ModeManager
from picarx.perception import Perception
from picarx.slam import OccupancyGridSLAM
from picarx.adaptive import AdaptiveController
from picarx.learning import EpsilonGreedyBandit, BanditConfig
from picarx.power import PowerManager


def main():
    cfg = load_config()
    px = Picarx()
    # Start edge perception (object detection, gestures, tracking)
    perception = Perception(cfg, use_capture=True)
    # Phase 3 modules
    slam = OccupancyGridSLAM()
    adaptive = AdaptiveController()
    bandit = EpsilonGreedyBandit(BanditConfig())
    current_arm = 1.0
    # Power/battery manager
    power = PowerManager(cfg)
    power.start()
    # Autonomy with speed scaling hook
    ctrl = AutonomousController(px, perception=perception, speed_scale_cb=lambda: current_arm)
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
            "battery": {
                "voltage_v": round(power.status().voltage_v, 2),
                "low": power.status().low,
                "critical": power.status().critical,
                "speed_scale": round(power.speed_scale(), 2),
            },
            # Perception summary for dashboard
            **({
                "perception": {
                    "gesture": (perception.last().gesture if perception.last() else None),
                    "inference_ms": (perception.last().inference_ms if perception.last() else 0.0),
                    "fps": (perception.last().fps if perception.last() else 0.0),
                    # limit telemetry object list to avoid large payloads
                    "objects": [
                        {
                            "label": o.label,
                            "score": round(o.score, 3),
                            "bbox": list(map(float, o.bbox_xywh)),
                            "track_id": o.track_id,
                            "depth_cm": o.depth_cm,
                        }
                        for o in ((perception.last().objects if perception.last() else [])[:8])
                    ],
                }
            })
        },
        on_heartbeat=wd.heartbeat,
        on_set_mode=modes.set_mode,
        on_manual=lambda sp, st: modes.set_manual_command(speed=sp, steer=st),
        get_snapshot=perception.get_jpeg,
        get_map=lambda: slam.get_map_jpeg(scale=2),
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
                # scale selection via bandit
                current_arm = bandit.select() * power.speed_scale()
                status = ctrl.tick()
                # reward: higher for speed, penalize near obstacle
                dist = px.get_distance()
                spd = ctrl.last_speed_cmd()
                reward = (spd / 100.0) - (0.5 if (dist and dist < 25) else 0.0)
                # SLAM update with current cmd and sensor distance
                slam.update(speed_cmd=spd, steer_angle_deg=ctrl.last_steer_cmd(), distance_cm=dist)
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
        try:
            power.stop()
        except Exception:
            pass
        try:
            perception.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
