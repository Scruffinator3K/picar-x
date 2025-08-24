from __future__ import annotations

import time

from picarx.picarx import Picarx
from picarx.autonomy import AutonomousController


def main():
    px = Picarx()
    ctrl = AutonomousController(px)
    try:
        while True:
            ctrl.tick()
            time.sleep(0.02)  # 50 Hz control loop
    except KeyboardInterrupt:
        pass
    finally:
        try:
            px.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
