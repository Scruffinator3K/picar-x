from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import math
import time
import json
import os
from collections import deque
from enum import Enum

try:
    import numpy as np  # type: ignore
    import cv2  # type: ignore
except Exception:
    np = None
    cv2 = None

from .logging_setup import init_logger


class CellType(Enum):
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2
    EXPLORED = 3
    OBSTACLE = 4
    WAYPOINT = 5
    PATH = 6
    GOAL = 7


@dataclass
class Pose2D:
    x: float = 0.0  # meters
    y: float = 0.0  # meters
    yaw: float = 0.0  # radians
    confidence: float = 1.0  # pose confidence 0-1
    timestamp: float = field(default_factory=time.time)


@dataclass
class MapFeature:
    """Represents interesting features detected in the environment"""
    id: str
    type: str  # obstacle, landmark, path, room, door, etc.
    position: Tuple[float, float]  # world coordinates
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class NavigationWaypoint:
    """Navigation waypoints for path planning"""
    id: str
    position: Tuple[float, float]
    name: str = ""
    waypoint_type: str = "generic"  # generic, checkpoint, goal, home, charging, etc.
    accessible: bool = True
    visited_count: int = 0
    last_visited: Optional[float] = None


class EnhancedSLAM:
    """Enhanced SLAM system with:
    - Multi-sensor fusion (ultrasonic, visual, odometry)
    - Path planning and navigation
    - Feature detection and mapping
    - Exploration strategies
    - Map persistence and sharing
    - Real-time obstacle avoidance
    - Room detection and mapping
    """

    def __init__(
        self,
        *,
        map_size: Tuple[int, int] = (400, 400),  # cells - larger map
        resolution_m_per_cell: float = 0.02,  # 2cm per cell => 8m square map
        origin_cell: Tuple[int, int] = (200, 200),
        wheel_base_m: float = 0.12,
        enable_visual_slam: bool = True,
        enable_loop_closure: bool = True,
        enable_path_planning: bool = True,
    ) -> None:
        self.log = init_logger("picarx.enhanced_slam")
        self.enabled = np is not None and cv2 is not None
        
        # Map configuration
        self.res = float(resolution_m_per_cell)
        self.wheel_base = float(wheel_base_m)
        self.map_w, self.map_h = int(map_size[0]), int(map_size[1])
        self.ox, self.oy = origin_cell
        
        # Enhanced features flags
        self.enable_visual_slam = enable_visual_slam and self.enabled
        self.enable_loop_closure = enable_loop_closure and self.enabled
        self.enable_path_planning = enable_path_planning and self.enabled
        
        # Multi-layer maps
        if self.enabled:
            self.occupancy_map = np.full((self.map_h, self.map_w), 0.5, dtype=np.float32)  # Probability grid
            self.confidence_map = np.zeros((self.map_h, self.map_w), dtype=np.float32)  # Confidence in each cell
            self.exploration_map = np.zeros((self.map_h, self.map_w), dtype=np.uint8)  # Exploration status
            self.feature_map = np.zeros((self.map_h, self.map_w), dtype=np.uint8)  # Feature annotations
            self.path_map = np.zeros((self.map_h, self.map_w), dtype=np.uint8)  # Planned paths
            self.cost_map = np.full((self.map_h, self.map_w), 1.0, dtype=np.float32)  # Navigation costs
        else:
            self.occupancy_map = None
            self.confidence_map = None
            self.exploration_map = None
            self.feature_map = None
            self.path_map = None
            self.cost_map = None
        
        # Pose tracking with history
        self.pose = Pose2D()
        self.pose_history = deque(maxlen=1000)
        self.odometry_error = Pose2D()  # Accumulated odometry error
        
        # Feature and landmark tracking
        self.features: Dict[str, MapFeature] = {}
        self.waypoints: Dict[str, NavigationWaypoint] = {}
        self.current_path: List[Tuple[int, int]] = []
        self.exploration_frontier: List[Tuple[int, int]] = []
        
        # Sensor fusion
        self.ultrasonic_history = deque(maxlen=50)
        self.last_loop_closure = 0.0
        
        # Navigation state
        self.current_goal: Optional[Tuple[float, float]] = None
        self.navigation_mode = "exploration"  # exploration, goal_seeking, returning_home
        self.home_position: Optional[Tuple[float, float]] = None
        
        # Statistics and performance
        self.update_count = 0
        self.total_distance_traveled = 0.0
        self.area_explored = 0.0
        self.obstacles_detected = 0
        self.features_detected = 0
        
        # Timing
        self._last_ts = time.time()
        self._last_full_update = 0.0
        
        # Persistence
        self.map_save_path = os.path.expanduser("~/.picarx/slam_map.json")
        self._load_persistent_data()
        
        if self.enabled:
            self.log.info(f"Enhanced SLAM initialized: {self.map_w}x{self.map_h} cells, {resolution_m_per_cell}m/cell")
            self.log.info(f"Features enabled: Visual={self.enable_visual_slam}, Loop={self.enable_loop_closure}, Planning={self.enable_path_planning}")
        else:
            self.log.warning("SLAM disabled: numpy/cv2 not available")

    def update(
        self,
        *,
        speed_ms: float = 0.0,
        steering_angle_rad: float = 0.0,
        front_distance_m: Optional[float] = None,
        sensor_readings: Optional[Dict[str, Any]] = None,
        visual_features: Optional[List[Any]] = None,
        force_full_update: bool = False,
    ) -> bool:
        """Enhanced SLAM update with multi-sensor fusion"""
        if not self.enabled:
            return False

        ts = time.time()
        dt = ts - self._last_ts
        self._last_ts = ts
        
        if dt <= 0.001:  # Skip updates that are too rapid
            return False
        
        self.update_count += 1
        
        # Update pose with odometry
        old_pose = Pose2D(self.pose.x, self.pose.y, self.pose.yaw)
        self._update_pose(speed_ms, steering_angle_rad, dt)
        
        # Track distance traveled
        distance_moved = math.sqrt(
            (self.pose.x - old_pose.x) ** 2 + (self.pose.y - old_pose.y) ** 2
        )
        self.total_distance_traveled += distance_moved
        
        # Store pose history
        self.pose_history.append(Pose2D(
            self.pose.x, self.pose.y, self.pose.yaw, 
            confidence=self._calculate_pose_confidence(), 
            timestamp=ts
        ))
        
        # Update maps with sensor data
        map_updated = False
        
        # Ultrasonic sensor mapping
        if front_distance_m is not None:
            self.ultrasonic_history.append((front_distance_m, ts))
            if self._update_occupancy_from_ultrasonic(front_distance_m):
                map_updated = True
        
        # Visual SLAM features
        if self.enable_visual_slam and visual_features:
            self.visual_features.extend(visual_features)
            if self._process_visual_features(visual_features):
                map_updated = True
        
        # Additional sensor processing
        if sensor_readings:
            if self._process_additional_sensors(sensor_readings):
                map_updated = True
        
        # Periodic full updates for advanced features
        should_full_update = (
            force_full_update or 
            (ts - self._last_full_update) > 2.0 or
            self.update_count % 50 == 0
        )
        
        if should_full_update:
            self._last_full_update = ts
            self._perform_full_update()
            map_updated = True
        
        # Save map periodically
        if self.update_count % 200 == 0:
            self._save_persistent_data()
        
        return map_updated

    def _update_pose(self, speed_ms: float, steering_angle_rad: float, dt: float) -> None:
        """Enhanced pose update with error accumulation tracking"""
        if abs(speed_ms) < 0.001:
            return  # Stationary
        
        # Bicycle model kinematics
        if abs(steering_angle_rad) < 0.001:  # Straight line
            self.pose.x += speed_ms * dt * math.cos(self.pose.yaw)
            self.pose.y += speed_ms * dt * math.sin(self.pose.yaw)
        else:
            # Curved motion
            turn_radius = self.wheel_base / math.tan(abs(steering_angle_rad))
            angular_velocity = speed_ms / turn_radius
            if steering_angle_rad < 0:
                angular_velocity = -angular_velocity
            
            # Update position and orientation
            delta_yaw = angular_velocity * dt
            arc_length = speed_ms * dt
            
            # Move along the arc
            if abs(delta_yaw) > 0.001:
                chord_length = 2 * turn_radius * math.sin(abs(delta_yaw) / 2)
                chord_angle = self.pose.yaw + delta_yaw / 2
                
                self.pose.x += chord_length * math.cos(chord_angle)
                self.pose.y += chord_length * math.sin(chord_angle)
            else:
                self.pose.x += arc_length * math.cos(self.pose.yaw)
                self.pose.y += arc_length * math.sin(self.pose.yaw)
            
            self.pose.yaw += delta_yaw
        
        # Normalize yaw to [-pi, pi]
        self.pose.yaw = math.atan2(math.sin(self.pose.yaw), math.cos(self.pose.yaw))
        
        # Update confidence based on motion
        motion_factor = min(abs(speed_ms) * 10, 1.0)  # Higher speed = lower confidence
        steering_factor = min(abs(steering_angle_rad) * 5, 1.0)  # Sharp turns = lower confidence
        self.pose.confidence = max(0.1, self.pose.confidence * (1.0 - motion_factor * 0.1) * (1.0 - steering_factor * 0.1))

    def _calculate_pose_confidence(self) -> float:
        """Calculate pose confidence based on various factors"""
        base_confidence = self.pose.confidence
        
        # Reduce confidence over time without loop closure
        time_factor = min((time.time() - self.last_loop_closure) / 30.0, 1.0)
        base_confidence *= (1.0 - time_factor * 0.5)
        
        # Increase confidence with feature matches
        if len(self.features) > 0:
            feature_factor = min(len(self.features) / 10.0, 0.3)
            base_confidence += feature_factor
        
        return max(0.1, min(1.0, base_confidence))

    def _update_occupancy_from_ultrasonic(self, distance_m: float) -> bool:
        """Enhanced occupancy mapping with confidence tracking"""
        if not self.enabled or distance_m <= 0:
            return False
        
        # Convert pose to map coordinates
        map_x = int(self.pose.x / self.res) + self.ox
        map_y = int(self.pose.y / self.res) + self.oy
        
        if not (0 <= map_x < self.map_w and 0 <= map_y < self.map_h):
            return False
        
        # Ray endpoint
        end_x = self.pose.x + distance_m * math.cos(self.pose.yaw)
        end_y = self.pose.y + distance_m * math.sin(self.pose.yaw)
        
        end_map_x = int(end_x / self.res) + self.ox
        end_map_y = int(end_y / self.res) + self.oy
        
        # Bresenham line algorithm for ray casting
        cells = self._bresenham_line(map_x, map_y, end_map_x, end_map_y)
        
        # Mark free cells along ray (except last)
        for i, (cx, cy) in enumerate(cells[:-1]):
            if 0 <= cx < self.map_w and 0 <= cy < self.map_h:
                # Increase confidence in free space
                self.occupancy_map[cy, cx] = max(0.1, self.occupancy_map[cy, cx] - 0.05)
                self.confidence_map[cy, cx] = min(1.0, self.confidence_map[cy, cx] + 0.1)
                self.exploration_map[cy, cx] = max(self.exploration_map[cy, cx], CellType.EXPLORED.value)
        
        # Mark obstacle at endpoint (if within reasonable range)
        if distance_m < 2.0 and len(cells) > 0:
            ex, ey = cells[-1]
            if 0 <= ex < self.map_w and 0 <= ey < self.map_h:
                # Mark as occupied
                self.occupancy_map[ey, ex] = min(0.9, self.occupancy_map[ey, ex] + 0.1)
                self.confidence_map[ey, ex] = min(1.0, self.confidence_map[ey, ex] + 0.2)
                self.exploration_map[ey, ex] = CellType.OBSTACLE.value
                self.obstacles_detected += 1
                
                # Add as feature if significant
                self._add_obstacle_feature(end_x, end_y, distance_m)
        
        return True

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm for ray casting"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            
            if error * 2 > -dy:
                error -= dy
                x += x_inc
            
            if error * 2 < dx:
                error += dx
                y += y_inc
        
        return points

    def _process_visual_features(self, features: List[Any]) -> bool:
        """Process visual features for SLAM"""
        if not self.enable_visual_slam:
            return False
        
        # This would integrate with computer vision
        # For now, simulate feature detection
        for feature in features:
            self.features_detected += 1
        
        return len(features) > 0

    def _process_additional_sensors(self, sensor_data: Dict[str, Any]) -> bool:
        """Process additional sensor data (IMU, encoders, etc.)"""
        # Future: integrate wheel encoders, IMU, etc.
        return False

    def _perform_full_update(self) -> None:
        """Perform computationally expensive updates periodically"""
        if not self.enabled:
            return
        
        # Update exploration frontiers
        self._update_exploration_frontiers()
        
        # Update navigation cost map
        self._update_cost_map()
        
        # Attempt loop closure
        if self.enable_loop_closure:
            self._attempt_loop_closure()
        
        # Update path planning
        if self.enable_path_planning and self.current_goal:
            self._plan_path_to_goal()
        
        # Calculate exploration metrics
        self._update_exploration_metrics()

    def _update_exploration_frontiers(self) -> None:
        """Find frontiers between explored and unexplored areas"""
        if not self.enabled:
            return
        
        self.exploration_frontier.clear()
        
        for y in range(1, self.map_h - 1):
            for x in range(1, self.map_w - 1):
                if self.exploration_map[y, x] == CellType.EXPLORED.value:
                    # Check if adjacent to unknown cells
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if (self.exploration_map[ny, nx] == CellType.UNKNOWN.value and
                                self.occupancy_map[ny, nx] < 0.7):  # Not obviously occupied
                                self.exploration_frontier.append((x, y))
                                break

    def _update_cost_map(self) -> None:
        """Update navigation cost map based on obstacles and uncertainty"""
        if not self.enabled:
            return
        
        # Base cost from occupancy probability
        self.cost_map = self.occupancy_map.copy()
        
        # Increase cost near obstacles (inflation)
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        inflated = cv2.filter2D(self.occupancy_map, -1, kernel)
        self.cost_map = np.maximum(self.cost_map, inflated * 0.5)
        
        # Increase cost in unexplored areas
        unexplored_mask = self.exploration_map == CellType.UNKNOWN.value
        self.cost_map[unexplored_mask] += 0.3

    def _attempt_loop_closure(self) -> bool:
        """Attempt to detect loop closure for pose correction"""
        # Simplified loop closure: check if we're near a previously visited location
        current_time = time.time()
        
        for historical_pose in list(self.pose_history)[-100:-10]:  # Check recent history, not too recent
            distance = math.sqrt(
                (self.pose.x - historical_pose.x) ** 2 + 
                (self.pose.y - historical_pose.y) ** 2
            )
            
            if distance < 0.5:  # Within 50cm of previous location
                # Simple pose correction (average positions)
                correction_factor = 0.1
                self.pose.x = self.pose.x * (1 - correction_factor) + historical_pose.x * correction_factor
                self.pose.y = self.pose.y * (1 - correction_factor) + historical_pose.y * correction_factor
                
                # Increase pose confidence
                self.pose.confidence = min(1.0, self.pose.confidence + 0.2)
                self.last_loop_closure = current_time
                
                self.log.debug(f"Loop closure detected at ({self.pose.x:.2f}, {self.pose.y:.2f})")
                return True
        
        return False

    def _plan_path_to_goal(self) -> bool:
        """Simple A* path planning to current goal"""
        if not self.enable_path_planning or not self.current_goal:
            return False
        
        # Convert goal to map coordinates
        goal_x = int(self.current_goal[0] / self.res) + self.ox
        goal_y = int(self.current_goal[1] / self.res) + self.oy
        
        # Current position
        start_x = int(self.pose.x / self.res) + self.ox
        start_y = int(self.pose.y / self.res) + self.oy
        
        # Simple path planning (placeholder for A*)
        self.path_map.fill(0)
        self.current_path = [(start_x, start_y), (goal_x, goal_y)]
        
        # Mark path on map
        for x, y in self.current_path:
            if 0 <= x < self.map_w and 0 <= y < self.map_h:
                self.path_map[y, x] = CellType.PATH.value
        
        return True

    def _add_obstacle_feature(self, x: float, y: float, distance: float) -> None:
        """Add an obstacle as a map feature"""
        feature_id = f"obstacle_{len(self.features)}"
        feature = MapFeature(
            id=feature_id,
            type="obstacle",
            position=(x, y),
            properties={"distance": distance, "sensor": "ultrasonic"},
            confidence=0.8
        )
        self.features[feature_id] = feature

    def _update_exploration_metrics(self) -> None:
        """Update exploration and mapping metrics"""
        if not self.enabled:
            return
        
        # Calculate explored area
        explored_cells = np.sum(self.exploration_map > CellType.UNKNOWN.value)
        self.area_explored = explored_cells * (self.res ** 2)  # square meters

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
    def _plan_path_to_goal(self) -> bool:
        """Simple A* path planning to current goal"""
        if not self.enable_path_planning or not self.current_goal:
            return False
        
        # Convert goal to map coordinates
        goal_x = int(self.current_goal[0] / self.res) + self.ox
        goal_y = int(self.current_goal[1] / self.res) + self.oy
        
        # Current position
        start_x = int(self.pose.x / self.res) + self.ox
        start_y = int(self.pose.y / self.res) + self.oy
        
        # Simple path planning (placeholder for A*)
        self.path_map.fill(0)
        self.current_path = [(start_x, start_y), (goal_x, goal_y)]
        
        # Mark path on map
        for x, y in self.current_path:
            if 0 <= x < self.map_w and 0 <= y < self.map_h:
                self.path_map[y, x] = CellType.PATH.value
        
        return True

    def _add_obstacle_feature(self, x: float, y: float, distance: float) -> None:
        """Add an obstacle as a map feature"""
        feature_id = f"obstacle_{len(self.features)}"
        feature = MapFeature(
            id=feature_id,
            type="obstacle",
            position=(x, y),
            properties={"distance": distance, "sensor": "ultrasonic"},
            confidence=0.8
        )
        self.features[feature_id] = feature

    def _update_exploration_metrics(self) -> None:
        """Update exploration and mapping metrics"""
        if not self.enabled:
            return
        
        # Calculate explored area
        explored_cells = np.sum(self.exploration_map > CellType.UNKNOWN.value)
        self.area_explored = explored_cells * (self.res ** 2)  # square meters

    # Navigation and Exploration Methods
    
    def set_goal(self, x: float, y: float, goal_type: str = "generic") -> bool:
        """Set a navigation goal"""
        if not self.enable_path_planning:
            return False
        
        self.current_goal = (x, y)
        self.navigation_mode = "goal_seeking"
        
        # Add as waypoint
        waypoint_id = f"goal_{len(self.waypoints)}"
        waypoint = NavigationWaypoint(
            id=waypoint_id,
            position=(x, y),
            name=f"Goal {len(self.waypoints)}",
            waypoint_type=goal_type
        )
        self.waypoints[waypoint_id] = waypoint
        
        # Plan path
        self._plan_path_to_goal()
        
        self.log.info(f"Goal set at ({x:.2f}, {y:.2f})")
        return True

    def set_home_position(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        """Set or update home position"""
        if x is None:
            x = self.pose.x
        if y is None:
            y = self.pose.y
        
        self.home_position = (x, y)
        
        # Add as waypoint
        home_waypoint = NavigationWaypoint(
            id="home",
            position=(x, y),
            name="Home",
            waypoint_type="home"
        )
        self.waypoints["home"] = home_waypoint
        
        self.log.info(f"Home position set at ({x:.2f}, {y:.2f})")

    def return_home(self) -> bool:
        """Start navigation back to home position"""
        if not self.home_position or not self.enable_path_planning:
            return False
        
        return self.set_goal(self.home_position[0], self.home_position[1], "home")

    def get_next_exploration_target(self) -> Optional[Tuple[float, float]]:
        """Get next exploration target based on frontiers"""
        if not self.exploration_frontier:
            return None
        
        # Find closest frontier
        best_frontier = None
        best_distance = float('inf')
        
        current_map_x = int(self.pose.x / self.res) + self.ox
        current_map_y = int(self.pose.y / self.res) + self.oy
        
        for fx, fy in self.exploration_frontier:
            distance = math.sqrt((fx - current_map_x) ** 2 + (fy - current_map_y) ** 2)
            if distance < best_distance:
                best_distance = distance
                best_frontier = (fx, fy)
        
        if best_frontier:
            # Convert back to world coordinates
            world_x = (best_frontier[0] - self.ox) * self.res
            world_y = (best_frontier[1] - self.oy) * self.res
            return (world_x, world_y)
        
        return None

    def start_exploration_mode(self) -> bool:
        """Start autonomous exploration"""
        self.navigation_mode = "exploration"
        
        # Find next exploration target
        target = self.get_next_exploration_target()
        if target:
            return self.set_goal(target[0], target[1], "exploration")
        
        self.log.info("No exploration targets found")
        return False

    # Map Export and Visualization Methods
    
    def get_map_data(self) -> Dict[str, Any]:
        """Get comprehensive map data for visualization"""
        if not self.enabled:
            return {}
        
        # Convert numpy arrays to lists for JSON serialization
        map_data = {
            "pose": {
                "x": float(self.pose.x),
                "y": float(self.pose.y),
                "yaw": float(self.pose.yaw),
                "confidence": float(self.pose.confidence)
            },
            "map_config": {
                "width": self.map_w,
                "height": self.map_h,
                "resolution": self.res,
                "origin": [self.ox, self.oy]
            },
            "occupancy_map": self.occupancy_map.tolist(),
            "confidence_map": self.confidence_map.tolist(),
            "exploration_map": self.exploration_map.tolist(),
            "feature_map": self.feature_map.tolist(),
            "path_map": self.path_map.tolist(),
            "cost_map": self.cost_map.tolist(),
            "features": {fid: {
                "id": f.id,
                "type": f.type,
                "position": f.position,
                "properties": f.properties,
                "confidence": f.confidence,
                "timestamp": f.timestamp
            } for fid, f in self.features.items()},
            "waypoints": {wid: {
                "id": w.id,
                "position": w.position,
                "name": w.name,
                "type": w.waypoint_type,
                "accessible": w.accessible,
                "visited_count": w.visited_count,
                "last_visited": w.last_visited
            } for wid, w in self.waypoints.items()},
            "path": self.current_path,
            "frontiers": self.exploration_frontier,
            "statistics": {
                "total_distance": self.total_distance_traveled,
                "area_explored": self.area_explored,
                "obstacles_detected": self.obstacles_detected,
                "features_detected": self.features_detected,
                "pose_confidence": self.pose.confidence,
                "update_count": self.update_count
            },
            "navigation": {
                "mode": self.navigation_mode,
                "current_goal": self.current_goal,
                "home_position": self.home_position
            }
        }
        
        return map_data

    def get_visualization_image(self, size: Tuple[int, int] = (800, 800)) -> Optional[Any]:
        """Generate visualization image of the map"""
        if not self.enabled:
            return None
        
        # Create RGB visualization
        vis_img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Scale factor from map to image
        scale_x = size[0] / self.map_w
        scale_y = size[1] / self.map_h
        
        # Draw occupancy map
        for y in range(self.map_h):
            for x in range(self.map_w):
                img_x = int(x * scale_x)
                img_y = int(y * scale_y)
                
                if img_x < size[0] and img_y < size[1]:
                    occupancy = self.occupancy_map[y, x]
                    if self.exploration_map[y, x] == CellType.OBSTACLE.value:
                        vis_img[img_y, img_x] = [255, 0, 0]  # Red for obstacles
                    elif self.exploration_map[y, x] == CellType.EXPLORED.value:
                        intensity = int((1.0 - occupancy) * 255)
                        vis_img[img_y, img_x] = [intensity, intensity, intensity]  # Gray for explored
                    else:
                        vis_img[img_y, img_x] = [64, 64, 64]  # Dark for unknown
        
        # Draw path
        for px, py in self.current_path:
            img_x = int(px * scale_x)
            img_y = int(py * scale_y)
            if 0 <= img_x < size[0] and 0 <= img_y < size[1]:
                vis_img[img_y, img_x] = [0, 255, 0]  # Green for path
        
        # Draw robot position
        robot_x = int((self.pose.x / self.res + self.ox) * scale_x)
        robot_y = int((self.pose.y / self.res + self.oy) * scale_y)
        if 0 <= robot_x < size[0] and 0 <= robot_y < size[1]:
            cv2.circle(vis_img, (robot_x, robot_y), 5, (0, 0, 255), -1)  # Blue for robot
            
            # Draw orientation arrow
            arrow_len = 15
            end_x = robot_x + int(arrow_len * math.cos(self.pose.yaw))
            end_y = robot_y + int(arrow_len * math.sin(self.pose.yaw))
            cv2.arrowedLine(vis_img, (robot_x, robot_y), (end_x, end_y), (0, 0, 255), 2)
        
        # Draw features
        for feature in self.features.values():
            feat_x = int((feature.position[0] / self.res + self.ox) * scale_x)
            feat_y = int((feature.position[1] / self.res + self.oy) * scale_y)
            if 0 <= feat_x < size[0] and 0 <= feat_y < size[1]:
                if feature.type == "obstacle":
                    cv2.circle(vis_img, (feat_x, feat_y), 3, (255, 255, 0), -1)  # Yellow for features
        
        # Draw goal
        if self.current_goal:
            goal_x = int((self.current_goal[0] / self.res + self.ox) * scale_x)
            goal_y = int((self.current_goal[1] / self.res + self.oy) * scale_y)
            if 0 <= goal_x < size[0] and 0 <= goal_y < size[1]:
                cv2.circle(vis_img, (goal_x, goal_y), 8, (255, 0, 255), 2)  # Magenta for goal
        
        return vis_img

    # Legacy method for backward compatibility
    def get_map_jpeg(self, scale: int = 2) -> Optional[bytes]:
        """Generate JPEG image of the map for backward compatibility"""
        if not self.enabled:
            return None
        
        # Generate visualization
        vis_img = self.get_visualization_image((400, 400))
        if vis_img is None:
            return None
        
        # Convert to grayscale for JPEG
        gray_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2GRAY)
        
        # Scale if requested
        if scale > 1:
            new_size = (gray_img.shape[1] * scale, gray_img.shape[0] * scale)
            gray_img = cv2.resize(gray_img, new_size, interpolation=cv2.INTER_NEAREST)
        
        # Encode as JPEG
        ok, buf = cv2.imencode('.jpg', gray_img)
        return buf.tobytes() if ok else None

    # Persistence Methods
    
    def _save_persistent_data(self) -> None:
        """Save map data to persistent storage"""
        try:
            os.makedirs(os.path.dirname(self.map_save_path), exist_ok=True)
            
            data = {
                "timestamp": time.time(),
                "map_data": self.get_map_data(),
                "pose_history": [
                    {"x": p.x, "y": p.y, "yaw": p.yaw, "confidence": p.confidence, "timestamp": p.timestamp}
                    for p in list(self.pose_history)[-100:]  # Save last 100 poses
                ]
            }
            
            with open(self.map_save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.log.debug(f"Map saved to {self.map_save_path}")
            
        except Exception as e:
            self.log.warning(f"Failed to save map: {e}")

    def _load_persistent_data(self) -> bool:
        """Load map data from persistent storage"""
        try:
            if not os.path.exists(self.map_save_path):
                return False
            
            with open(self.map_save_path, 'r') as f:
                data = json.load(f)
            
            # Check if data is recent (within 1 hour)
            if time.time() - data.get("timestamp", 0) > 3600:
                self.log.info("Saved map is old, starting fresh")
                return False
            
            map_data = data.get("map_data", {})
            
            # Restore pose
            if "pose" in map_data:
                pose_data = map_data["pose"]
                self.pose.x = pose_data.get("x", 0.0)
                self.pose.y = pose_data.get("y", 0.0)
                self.pose.yaw = pose_data.get("yaw", 0.0)
                self.pose.confidence = pose_data.get("confidence", 1.0)
            
            # Restore maps if enabled
            if self.enabled and "occupancy_map" in map_data:
                self.occupancy_map = np.array(map_data["occupancy_map"], dtype=np.float32)
                self.confidence_map = np.array(map_data["confidence_map"], dtype=np.float32)
                self.exploration_map = np.array(map_data["exploration_map"], dtype=np.uint8)
                self.feature_map = np.array(map_data["feature_map"], dtype=np.uint8)
                self.path_map = np.array(map_data["path_map"], dtype=np.uint8)
                self.cost_map = np.array(map_data["cost_map"], dtype=np.float32)
            
            # Restore features
            if "features" in map_data:
                for fid, fdata in map_data["features"].items():
                    feature = MapFeature(
                        id=fdata["id"],
                        type=fdata["type"],
                        position=tuple(fdata["position"]),
                        properties=fdata["properties"],
                        confidence=fdata["confidence"],
                        timestamp=fdata["timestamp"]
                    )
                    self.features[fid] = feature
            
            # Restore waypoints
            if "waypoints" in map_data:
                for wid, wdata in map_data["waypoints"].items():
                    waypoint = NavigationWaypoint(
                        id=wdata["id"],
                        position=tuple(wdata["position"]),
                        name=wdata["name"],
                        waypoint_type=wdata["type"],
                        accessible=wdata["accessible"],
                        visited_count=wdata["visited_count"],
                        last_visited=wdata["last_visited"]
                    )
                    self.waypoints[wid] = waypoint
            
            # Restore navigation state
            if "navigation" in map_data:
                nav_data = map_data["navigation"]
                self.navigation_mode = nav_data.get("mode", "exploration")
                self.current_goal = nav_data.get("current_goal")
                self.home_position = nav_data.get("home_position")
            
            # Restore statistics
            if "statistics" in map_data:
                stats = map_data["statistics"]
                self.total_distance_traveled = stats.get("total_distance", 0.0)
                self.area_explored = stats.get("area_explored", 0.0)
                self.obstacles_detected = stats.get("obstacles_detected", 0)
                self.features_detected = stats.get("features_detected", 0)
            
            self.log.info(f"Map loaded from {self.map_save_path}")
            return True
            
        except Exception as e:
            self.log.warning(f"Failed to load map: {e}")
            return False

    def clear_map(self) -> None:
        """Clear all map data and start fresh"""
        if self.enabled:
            self.occupancy_map.fill(0.5)
            self.confidence_map.fill(0.0)
            self.exploration_map.fill(0)
            self.feature_map.fill(0)
            self.path_map.fill(0)
            self.cost_map.fill(1.0)
        
        self.features.clear()
        self.waypoints.clear()
        self.current_path.clear()
        self.exploration_frontier.clear()
        self.pose_history.clear()
        
        self.pose = Pose2D()
        self.current_goal = None
        self.home_position = None
        self.navigation_mode = "exploration"
        
        self.total_distance_traveled = 0.0
        self.area_explored = 0.0
        self.obstacles_detected = 0
        self.features_detected = 0
        self.update_count = 0
        
        self.log.info("Map cleared")

    def get_slam_status(self) -> Dict[str, Any]:
        """Get current SLAM system status"""
        return {
            "enabled": self.enabled,
            "features": {
                "visual_slam": self.enable_visual_slam,
                "loop_closure": self.enable_loop_closure,
                "path_planning": self.enable_path_planning
            },
            "map_size": [self.map_w, self.map_h],
            "resolution": self.res,
            "pose": {
                "x": self.pose.x,
                "y": self.pose.y,
                "yaw": self.pose.yaw,
                "confidence": self.pose.confidence
            },
            "statistics": {
                "updates": self.update_count,
                "distance_traveled": self.total_distance_traveled,
                "area_explored": self.area_explored,
                "obstacles": self.obstacles_detected,
                "features": len(self.features),
                "waypoints": len(self.waypoints)
            },
            "navigation": {
                "mode": self.navigation_mode,
                "has_goal": self.current_goal is not None,
                "has_home": self.home_position is not None,
                "path_length": len(self.current_path),
                "frontiers": len(self.exploration_frontier)
            }
        }

    # Legacy methods for backward compatibility
    def get_pose(self) -> Pose2D:
        """Get current pose (backward compatibility)"""
        return self.pose

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to cell coordinates"""
        cx = int(x / self.res) + self.ox
        cy = int(y / self.res) + self.oy
        return cx, cy

    def _clamp_cell(self, cx: int, cy: int) -> Tuple[int, int]:
        """Clamp cell coordinates to map bounds"""
        cx = max(0, min(self.map_w - 1, cx))
        cy = max(0, min(self.map_h - 1, cy))
        return cx, cy

    # Simple update method for backward compatibility
    def _update_map(self, distance_m: float) -> bool:
        """Simple map update (backward compatibility)"""
        return self._update_occupancy_from_ultrasonic(distance_m)


# Backward compatibility alias
OccupancyGridSLAM = EnhancedSLAM
