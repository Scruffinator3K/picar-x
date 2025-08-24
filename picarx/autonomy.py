from __future__ import annotations

import time
from typing import Optional

from .logging_setup import init_logger
from .behavior_tree import Selector, Sequence, Condition, Action, Status
from .fusion import SensorFusion


class AutonomousController:
    """Simple autonomous controller powered by a behavior tree.

    Behaviors:
    - Emergency stop if cliff detected
    - Avoid obstacle when distance below threshold
    - Otherwise line-follow, else roam forward
    """

    def __init__(self, px, fusion: Optional[SensorFusion] = None):
        self.px = px
        self.log = init_logger("picarx.autonomy")
        self.fusion = fusion or SensorFusion(px)
        self._build_tree()

    # predicates
    def _is_cliff(self) -> bool:
        state = self.fusion.update()
        return state.cliff

    def _is_obstacle_close(self) -> bool:
        state = self.fusion.update()
        d = state.distance_cm or 999.0
        return d <= self.fusion.obstacle_stop_cm

    def _is_obstacle_near(self) -> bool:
        state = self.fusion.update()
        d = state.distance_cm or 999.0
        return d <= self.fusion.obstacle_slow_cm

    def _has_line(self) -> bool:
        state = self.fusion.update()
        return state.line_error is not None

    # actions
    def _act_stop(self) -> Status:
        self.px.stop()
        return Status.SUCCESS

    def _act_avoid(self) -> Status:
        # simple avoidance: stop, steer away, reverse a bit, then straighten
        self.px.stop()
        time.sleep(0.05)
        self.px.set_dir_servo_angle(-25)
        self.px.backward(50)
        time.sleep(0.3)
        self.px.stop()
        self.px.set_dir_servo_angle(0)
        return Status.SUCCESS

    def _act_slow(self) -> Status:
        self.px.set_power(30)
        return Status.SUCCESS

    def _act_line_follow(self) -> Status:
        st = self.fusion.update()
        if st.line_error is None:
            return Status.FAILURE
        # proportional steering on error
        k = 40.0
        steer = int(max(-30, min(30, k * st.line_error)))
        self.px.set_dir_servo_angle(steer)
        # dynamic speed: reduce speed when steering hard
        base = 55
        speed = max(30, int(base - abs(steer) * 0.6))
        self.px.forward(speed)
        return Status.SUCCESS

    def _act_cruise(self) -> Status:
        self.px.set_dir_servo_angle(0)
        self.px.forward(50)
        return Status.SUCCESS

    def _build_tree(self):
        self.tree = Selector([
            # 1) Emergency stop
            Sequence([Condition(self._is_cliff), Action(self._act_stop)]),
            # 2) Obstacle avoidance
            Sequence([Condition(self._is_obstacle_close), Action(self._act_avoid)]),
            Sequence([Condition(self._is_obstacle_near), Action(self._act_slow)]),
            # 3) Line follow else cruise forward
            Sequence([Condition(self._has_line), Action(self._act_line_follow)]),
            Action(self._act_cruise),
        ])

    def tick(self):
        try:
            return self.tree.tick()
        except Exception as e:
            self.log.error(f"autonomy tick error: {e}")
            self.px.stop()
            return Status.FAILURE
