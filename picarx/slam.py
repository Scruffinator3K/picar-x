from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import time

try:
    import numpy as np  # type: ignore
    import cv2  # type: ignore
except Exception:
    np = None
    cv2 = None

from .logging_setup import init_logger


@dataclass
class Pose2D:
    x: float = 0.0  # meters
    y: float = 0.0  # meters
    yaw: float = 0.0  # radians


class OccupancyGridSLAM:
    """Tiny occupancy-grid SLAM using naive dead-reckoning and ultrasonic mapping.

    - Pose integrates commanded speed and steering to estimate motion.
    - Mapping: marks free cells along a forward ray and occupied cell at hit.
    - Designed to run lightweight on Pi; not a full SLAM, but useful for Phase 3 scaffolding.
    """

    def __init__(
        self,
        *,
        map_size: Tuple[int, int] = (200, 200),  # cells
        resolution_m_per_cell: float = 0.02,  # 2cm per cell => 4m square map
        origin_cell: Tuple[int, int] = (100, 100),
        wheel_base_m: float = 0.12,
    ) -> None:
        self.log = init_logger("picarx.slam")
        self.enabled = np is not None
        self.res = float(resolution_m_per_cell)
        self.wheel_base = float(wheel_base_m)
        self.map_w, self.map_h = int(map_size[0]), int(map_size[1])
        self.occ = None if np is None else np.full((self.map_h, self.map_w), 0.5, dtype=np.float32)
        self.pose = Pose2D()
        self._last_ts = time.time()
        self.ox, self.oy = origin_cell

    def reset_map(self) -> None:
        if self.occ is not None:
            self.occ.fill(0.5)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        cx = int(round(self.ox + x / self.res))
        cy = int(round(self.oy - y / self.res))
        return cx, cy

    def _clamp_cell(self, cx: int, cy: int) -> Tuple[int, int]:
        cx = max(0, min(self.map_w - 1, cx))
        cy = max(0, min(self.map_h - 1, cy))
        return cx, cy

    def integrate_motion(self, speed_cmd: int, steer_angle_deg: int, dt: float) -> None:
        # Speed 0..100 maps to ~0..0.6 m/s (tunable)
        vmax = 0.6
        v = vmax * max(0.0, min(100.0, float(speed_cmd))) / 100.0
        steer = math.radians(float(steer_angle_deg))
        if abs(steer) < math.radians(1.0):
            # straight
            dx = v * dt * math.cos(self.pose.yaw)
            dy = v * dt * math.sin(self.pose.yaw)
            self.pose.x += dx
            self.pose.y += dy
        else:
            # bicycle model
            R = self.wheel_base / math.tan(steer)
            omega = v / R
            dtheta = omega * dt
            self.pose.yaw = (self.pose.yaw + dtheta + math.pi) % (2 * math.pi) - math.pi
            # ICC update
            self.pose.x += v * dt * math.cos(self.pose.yaw)
            self.pose.y += v * dt * math.sin(self.pose.yaw)

    def update(self, *, speed_cmd: int, steer_angle_deg: int, distance_cm: Optional[float]) -> None:
        now = time.time()
        dt = max(0.001, now - self._last_ts)
        self._last_ts = now
        # 1) integrate motion
        self.integrate_motion(speed_cmd, steer_angle_deg, dt)
        # 2) map update from ultrasonic
        if self.occ is None or distance_cm is None or distance_cm <= 0:
            return
        # forward ray from pose in heading direction
        max_m = min(3.0, float(distance_cm) / 100.0)
        steps = int(max_m / self.res)
        x0, y0, th = self.pose.x, self.pose.y, self.pose.yaw
        hit_cell = None
        for i in range(1, steps + 1):
            xi = x0 + i * self.res * math.cos(th)
            yi = y0 + i * self.res * math.sin(th)
            cx, cy = self.world_to_cell(xi, yi)
            cx, cy = self._clamp_cell(cx, cy)
            # free space decay
            self.occ[cy, cx] = max(0.05, self.occ[cy, cx] - 0.02)
            if i * self.res >= max_m:
                hit_cell = (cx, cy)
                break
        if hit_cell:
            cx, cy = hit_cell
            self.occ[cy, cx] = min(0.95, self.occ[cy, cx] + 0.1)

    def get_pose(self) -> Pose2D:
        return self.pose

    def get_map_jpeg(self, scale: int = 2) -> Optional[bytes]:
        if self.occ is None or cv2 is None:
            return None
        img = (255.0 * (1.0 - self.occ)).astype("uint8")  # occupied dark
        # draw robot
        cx, cy = self.world_to_cell(self.pose.x, self.pose.y)
        cx, cy = self._clamp_cell(cx, cy)
        cv2.circle(img, (cx, cy), 2, 128, -1)
        # scale
        if scale > 1:
            img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode('.jpg', img)
        return buf.tobytes() if ok else None
