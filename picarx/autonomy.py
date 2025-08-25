from __future__ import annotations

import time
from typing import Optional, Callable

from .logging_setup import init_logger
from .behavior_tree import Selector, Sequence, Condition, Action, Status
from .fusion import SensorFusion
from .perception import Perception
from .speed import SpeedPlanner


class AutonomousController:
    """Simple autonomous controller powered by a behavior tree.

    Behaviors:
    - Emergency stop if cliff detected
    - Avoid obstacle when distance below threshold
    - Otherwise line-follow, else roam forward
    """

    def __init__(self, px, fusion: Optional[SensorFusion] = None, perception: Optional[Perception] = None,
                 speed_scale_cb: Optional[Callable[[], float]] = None):
        self.px = px
        self.log = init_logger("picarx.autonomy")
        self.fusion = fusion or SensorFusion(px)
        self.perception = perception
        self._planner = SpeedPlanner()
        self._last_tick = time.time()
        self._last_speed_cmd: int = 0
        self._last_steer_cmd: int = 0
        self.get_speed_scale: Callable[[], float] = speed_scale_cb or (lambda: 1.0)
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

    def _gesture_stop(self) -> bool:
        try:
            if self.perception is None:
                return False
            st = self.perception.last()
            return bool(st and st.gesture == "stop")
        except Exception:
            return False

    def _has_line(self) -> bool:
        state = self.fusion.update()
        return state.line_error is not None

    # actions
    def _act_stop(self) -> Status:
        self._planner.reset(0)
        self.px.stop()
        self._last_speed_cmd = 0
        return Status.SUCCESS

    def _act_avoid(self) -> Status:
        # simple avoidance: stop, steer away, reverse a bit, then straighten
        self._planner.reset(0)
        self.px.stop()
        time.sleep(0.05)
        self.px.set_dir_servo_angle(-25)
        self._last_steer_cmd = -25
        self.px.backward(50)
        self._last_speed_cmd = -50
        time.sleep(0.3)
        self.px.stop()
        self._last_speed_cmd = 0
        self.px.set_dir_servo_angle(0)
        self._last_steer_cmd = 0
        return Status.SUCCESS

    def _act_slow(self) -> Status:
        st = self.fusion.update()
        # base slower speed near obstacle; planner will further limit by distance
        base = 50  # Increased from 35 - more confident near obstacles
        target = self._planner.limit_by_distance(st.distance_cm, base_speed=base)
        target = int(max(20, min(100, target * self.get_speed_scale())))  # Minimum speed 20
        now = time.time()
        dt = max(0.005, now - self._last_tick)
        emergency = (st.distance_cm or 999.0) <= (self.fusion.obstacle_stop_cm + 2.0)  # Reduced buffer
        cmd = self._planner.step(target, dt, emergency=emergency)
        self._last_tick = now
        self.px.forward(cmd)
        self._last_speed_cmd = cmd
        return Status.SUCCESS

    def _act_line_follow(self) -> Status:
        st = self.fusion.update()
        if st.line_error is None:
            return Status.FAILURE
        
        # Enhanced steering control for noisy surfaces
        # Reduce proportional gain and add deadband to minimize jitter
        error_deadband = 0.1  # Ignore small errors (wood grain noise)
        
        if abs(st.line_error) < error_deadband:
            # Small error - maintain current steering or go straight
            steer = getattr(self, '_last_stable_steer', 0)
        else:
            # Significant error - apply steering correction with reduced gain
            k = 25.0  # Reduced from 40.0 for smoother response
            steer = int(max(-25, min(25, k * st.line_error)))  # Reduced max steering angle
            self._last_stable_steer = steer
        
        self.px.set_dir_servo_angle(steer)
        self._last_steer_cmd = steer
        
        # Reduce speed more aggressively with steering to improve stability
        base = max(35, int(65 - abs(steer) * 1.2))  # More speed reduction with steering
        base = min(base, 75)  # Lower max base speed for stability
        target = self._planner.limit_by_distance(st.distance_cm, base_speed=base)
        target = int(max(15, min(100, target * self.get_speed_scale())))  # Lower minimum speed
        now = time.time()
        dt = max(0.005, now - self._last_tick)
        emergency = (st.distance_cm or 999.0) <= (self.fusion.obstacle_stop_cm + 2.0)
        cmd = self._planner.step(target, dt, emergency=emergency)
        self._last_tick = now
        self.px.forward(cmd)
        self._last_speed_cmd = cmd
        
        # Log steering decisions for debugging
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 1
            
        if self._debug_counter % 50 == 0:  # Log every ~1 second at 50Hz
            self.log.debug(f"Line follow: error={st.line_error:.3f}, steer={steer}, speed={cmd}")
        
        return Status.SUCCESS

    def _act_cruise(self) -> Status:
        # Enhanced cruise with gentle steering corrections
        # Don't just lock to center - allow gentle corrections based on sensor trends
        st = self.fusion.update()
        
        # If we have some line detection but it's weak, make gentle adjustments
        if st.line_error is not None and abs(st.line_error) > 0.2:
            # Gentle steering correction during cruise
            gentle_steer = int(max(-10, min(10, 15.0 * st.line_error)))
            self.px.set_dir_servo_angle(gentle_steer)
            self._last_steer_cmd = gentle_steer
        else:
            # No strong line signal - go straight
            self.px.set_dir_servo_angle(0)
            self._last_steer_cmd = 0
        
        # Conservative cruise speed for stability on varied surfaces
        base = 60  # Reduced from 70 for better stability
        target = self._planner.limit_by_distance(st.distance_cm, base_speed=base)
        target = int(max(25, min(100, target * self.get_speed_scale())))  # Higher minimum cruise speed
        now = time.time()
        dt = max(0.005, now - self._last_tick)
        cmd = self._planner.step(target, dt)
        self._last_tick = now
        self.px.forward(cmd)
        self._last_speed_cmd = cmd
        return Status.SUCCESS

    def _build_tree(self):
        self.tree = Selector([
            # 0) User gesture stop overrides everything
            Sequence([Condition(self._gesture_stop), Action(self._act_stop)]),
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

    # Phase 3 helpers
    def last_speed_cmd(self) -> int:
        return int(self._last_speed_cmd)

    def last_steer_cmd(self) -> int:
        return int(self._last_steer_cmd)
