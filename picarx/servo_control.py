"""
Comprehensive Servo Control System with Advanced Anti-Jitter Protection

This module provides a robust servo control system designed to eliminate
jittering through multiple layers of protection including rate limiting,
deadband filtering, command queuing, exponential backoff, and hardware
power management.
"""

from __future__ import annotations

import time
import threading
from typing import Optional, Dict, Any, Callable
from collections import deque
import math

from .logging_setup import init_logger


class ServoCommand:
    """Represents a servo movement command with metadata"""
    def __init__(self, servo_id: str, target_angle: float, priority: int = 0, source: str = "unknown"):
        self.servo_id = servo_id
        self.target_angle = target_angle
        self.priority = priority
        self.source = source
        self.timestamp = time.time()
        self.retry_count = 0


class ServoState:
    """Tracks the current state of a servo"""
    def __init__(self, servo_id: str):
        self.servo_id = servo_id
        self.current_angle = 0.0
        self.target_angle = 0.0
        self.last_command_time = 0.0
        self.last_actual_move_time = 0.0
        self.command_count = 0
        self.is_moving = False
        self.is_enabled = True
        self.temperature = 0.0
        self.error_count = 0
        self.consecutive_rejections = 0
        
        # Anti-jitter tracking
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        self.acceleration_history = deque(maxlen=3)


class ComprehensiveServoController:
    """
    Advanced servo controller with comprehensive anti-jitter protection
    
    Features:
    - Multi-layer rate limiting with exponential backoff
    - Large deadband filtering (configurable)
    - Command queuing and debouncing
    - Servo power management and idle detection
    - Position prediction and smoothing
    - Global servo mutex for thread safety
    - Hardware health monitoring
    - Adaptive control based on system load
    """
    
    def __init__(self, picarx_instance, logger_name: str = "picarx.servo_control"):
        self.log = init_logger(logger_name)
        self.px = picarx_instance
        
        # Configuration - much more aggressive anti-jitter settings
        self.config = {
            "min_rate_limit": 1.0,  # Minimum 1 second between commands
            "max_rate_limit": 5.0,  # Maximum 5 seconds for problematic servos
            "deadband_degrees": 8.0,  # Large 8-degree deadband
            "startup_suppression_time": 15.0,  # 15 seconds of startup suppression
            "max_queue_size": 3,  # Small queue to prevent spam
            "exponential_backoff_factor": 1.5,
            "max_consecutive_rejections": 5,
            "servo_idle_timeout": 30.0,  # Disable servo after 30s of inactivity
            "position_smoothing_factor": 0.7,  # Smooth position changes
            "velocity_limit_deg_per_sec": 45.0,  # Maximum velocity
            "acceleration_limit_deg_per_sec2": 90.0,  # Maximum acceleration
            
            # Per-servo specific configurations
            "servo_specific": {
                "cam_pan": {
                    "min_rate_limit": 1.5,  # Very aggressive for pan
                    "deadband_degrees": 10.0,  # Large deadband for pan
                },
                "cam_tilt": {
                    "min_rate_limit": 1.5,  # Very aggressive for tilt
                    "deadband_degrees": 10.0,  # Large deadband for tilt
                },
                "dir_servo": {
                    "min_rate_limit": 0.2,  # More responsive for steering
                    "deadband_degrees": 2.0,  # Smaller deadband for steering
                    "startup_suppression_time": 2.0,  # Less startup suppression
                }
            }
        }
        
        # Global state
        self.servo_states: Dict[str, ServoState] = {}
        self.command_queue = deque(maxlen=self.config["max_queue_size"])
        self.global_lock = threading.RLock()
        self.boot_time = time.time()
        self.system_load_factor = 1.0
        self.emergency_stop = False
        
        # Initialize servo states
        self._initialize_servo_states()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_servo_states(self):
        """Initialize tracking for all servos"""
        servo_ids = ["cam_pan", "cam_tilt", "dir_servo"]
        for servo_id in servo_ids:
            self.servo_states[servo_id] = ServoState(servo_id)
            
        # Initialize current positions from hardware if possible
        try:
            if hasattr(self.px, 'cam_pan_angle'):
                self.servo_states["cam_pan"].current_angle = self.px.cam_pan_angle
            if hasattr(self.px, 'cam_tilt_angle'):
                self.servo_states["cam_tilt"].current_angle = self.px.cam_tilt_angle
        except Exception as e:
            self.log.warning(f"Could not initialize servo positions: {e}")
    
    def _start_background_tasks(self):
        """Start background monitoring and control tasks"""
        self.monitor_thread = threading.Thread(target=self._monitor_servo_health, daemon=True)
        self.monitor_thread.start()
        
        self.queue_processor_thread = threading.Thread(target=self._process_command_queue, daemon=True)
        self.queue_processor_thread.start()
    
    def _monitor_servo_health(self):
        """Background thread to monitor servo health and manage power"""
        while not self.emergency_stop:
            try:
                current_time = time.time()
                
                with self.global_lock:
                    for servo_state in self.servo_states.values():
                        # Check for idle servos and disable them
                        if (servo_state.is_enabled and 
                            current_time - servo_state.last_actual_move_time > self.config["servo_idle_timeout"]):
                            self._disable_servo(servo_state.servo_id)
                        
                        # Reset consecutive rejections periodically
                        if (current_time - servo_state.last_command_time > 10.0 and 
                            servo_state.consecutive_rejections > 0):
                            servo_state.consecutive_rejections = max(0, servo_state.consecutive_rejections - 1)
                            self.log.debug(f"Reducing rejection count for {servo_state.servo_id}")
                
                # Update system load factor based on recent activity
                self._update_system_load_factor()
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.log.error(f"Error in servo health monitor: {e}")
                time.sleep(5.0)
    
    def _process_command_queue(self):
        """Background thread to process queued servo commands"""
        while not self.emergency_stop:
            try:
                if self.command_queue:
                    with self.global_lock:
                        if self.command_queue:
                            command = self.command_queue.popleft()
                            self._execute_servo_command(command)
                
                time.sleep(0.1)  # Process queue at 10Hz
                
            except Exception as e:
                self.log.error(f"Error in command queue processor: {e}")
                time.sleep(1.0)
    
    def _update_system_load_factor(self):
        """Update system load factor based on recent servo activity"""
        current_time = time.time()
        recent_commands = sum(
            1 for state in self.servo_states.values()
            if current_time - state.last_command_time < 5.0
        )
        
        # Higher load = more restrictive rate limiting
        self.system_load_factor = min(3.0, 1.0 + (recent_commands * 0.5))
    
    def set_servo_angle(self, servo_id: str, angle: float, priority: int = 0, source: str = "user") -> bool:
        """
        Set servo angle with comprehensive anti-jitter protection
        
        Args:
            servo_id: Identifier for the servo ("cam_pan", "cam_tilt", "dir_servo")
            angle: Target angle in degrees
            priority: Command priority (higher = more important)
            source: Source of the command for debugging
            
        Returns:
            bool: True if command was accepted, False if rejected
        """
        current_time = time.time()
        
        # Startup suppression - use servo-specific timing
        servo_config = self.config["servo_specific"].get(servo_id, {})
        startup_time = servo_config.get("startup_suppression_time", self.config["startup_suppression_time"])
        
        if current_time - self.boot_time < startup_time:
            self.log.debug(f"Rejecting {servo_id} command during startup suppression")
            return False
        
        # Emergency stop check
        if self.emergency_stop:
            self.log.warning(f"Rejecting {servo_id} command due to emergency stop")
            return False
        
        with self.global_lock:
            # Get or create servo state
            if servo_id not in self.servo_states:
                self.servo_states[servo_id] = ServoState(servo_id)
            
            servo_state = self.servo_states[servo_id]
            
            # Rate limiting with exponential backoff
            min_interval = self._calculate_rate_limit(servo_state)
            if current_time - servo_state.last_command_time < min_interval:
                servo_state.consecutive_rejections += 1
                self.log.debug(f"Rate limiting {servo_id}: {current_time - servo_state.last_command_time:.3f}s < {min_interval:.3f}s")
                return False
            
            # Deadband filtering - use servo-specific deadband
            servo_config = self.config["servo_specific"].get(servo_id, {})
            deadband = servo_config.get("deadband_degrees", self.config["deadband_degrees"])
            
            if abs(angle - servo_state.current_angle) < deadband:
                self.log.debug(f"Deadband filtering {servo_id}: {abs(angle - servo_state.current_angle):.1f}° < {deadband:.1f}°")
                return False
            
            # Velocity and acceleration limiting
            if not self._check_motion_limits(servo_state, angle, current_time):
                servo_state.consecutive_rejections += 1
                self.log.debug(f"Motion limits exceeded for {servo_id}")
                return False
            
            # Position smoothing
            smoothed_angle = self._apply_position_smoothing(servo_state, angle)
            
            # Create command
            command = ServoCommand(servo_id, smoothed_angle, priority, source)
            
            # Queue management
            if len(self.command_queue) >= self.config["max_queue_size"]:
                # Remove lowest priority command
                self.command_queue = deque(
                    sorted(self.command_queue, key=lambda c: c.priority, reverse=True)[:self.config["max_queue_size"]-1],
                    maxlen=self.config["max_queue_size"]
                )
            
            self.command_queue.append(command)
            servo_state.last_command_time = current_time
            servo_state.command_count += 1
            servo_state.consecutive_rejections = 0
            
            self.log.debug(f"Queued {servo_id} command: {angle:.1f}° -> {smoothed_angle:.1f}° (queue size: {len(self.command_queue)})")
            return True
    
    def _calculate_rate_limit(self, servo_state: ServoState) -> float:
        """Calculate dynamic rate limit based on servo history and system load"""
        servo_config = self.config["servo_specific"].get(servo_state.servo_id, {})
        base_limit = servo_config.get("min_rate_limit", self.config["min_rate_limit"])
        
        # Exponential backoff for problematic servos
        if servo_state.consecutive_rejections > 0:
            backoff_factor = self.config["exponential_backoff_factor"] ** servo_state.consecutive_rejections
            base_limit *= min(backoff_factor, self.config["max_rate_limit"] / base_limit)
        
        # Apply system load factor
        base_limit *= self.system_load_factor
        
        # Cap at maximum
        return min(base_limit, self.config["max_rate_limit"])
    
    def _check_motion_limits(self, servo_state: ServoState, target_angle: float, current_time: float) -> bool:
        """Check if the requested motion exceeds velocity/acceleration limits"""
        if not servo_state.position_history:
            return True  # No history, allow movement
        
        # Calculate time delta
        if servo_state.last_actual_move_time == 0:
            return True  # First movement
        
        dt = current_time - servo_state.last_actual_move_time
        if dt < 0.01:  # Avoid division by very small numbers
            return False
        
        # Calculate velocity
        position_delta = target_angle - servo_state.current_angle
        velocity = abs(position_delta) / dt
        
        if velocity > self.config["velocity_limit_deg_per_sec"]:
            self.log.debug(f"Velocity limit exceeded: {velocity:.1f} > {self.config['velocity_limit_deg_per_sec']}")
            return False
        
        # Calculate acceleration if we have velocity history
        if len(servo_state.velocity_history) > 0:
            last_velocity = servo_state.velocity_history[-1]
            acceleration = abs(velocity - last_velocity) / dt
            
            if acceleration > self.config["acceleration_limit_deg_per_sec2"]:
                self.log.debug(f"Acceleration limit exceeded: {acceleration:.1f} > {self.config['acceleration_limit_deg_per_sec2']}")
                return False
        
        return True
    
    def _apply_position_smoothing(self, servo_state: ServoState, target_angle: float) -> float:
        """Apply position smoothing to reduce abrupt movements"""
        if not servo_state.position_history:
            return target_angle
        
        # Weighted average with current position
        smoothing_factor = self.config["position_smoothing_factor"]
        smoothed = servo_state.current_angle * smoothing_factor + target_angle * (1 - smoothing_factor)
        
        return smoothed
    
    def _execute_servo_command(self, command: ServoCommand):
        """Execute a servo command with hardware interaction"""
        try:
            servo_state = self.servo_states[command.servo_id]
            current_time = time.time()
            
            # Re-enable servo if it was disabled
            if not servo_state.is_enabled:
                self._enable_servo(command.servo_id)
            
            # Execute the actual hardware command
            success = self._send_hardware_command(command.servo_id, command.target_angle)
            
            if success:
                # Update servo state
                old_angle = servo_state.current_angle
                servo_state.current_angle = command.target_angle
                servo_state.target_angle = command.target_angle
                servo_state.last_actual_move_time = current_time
                servo_state.is_moving = True
                
                # Update history for motion analysis
                servo_state.position_history.append(command.target_angle)
                if servo_state.last_actual_move_time > 0:
                    dt = current_time - servo_state.last_actual_move_time
                    if dt > 0:
                        velocity = abs(command.target_angle - old_angle) / dt
                        servo_state.velocity_history.append(velocity)
                
                self.log.info(f"Executed {command.servo_id}: {old_angle:.1f}° -> {command.target_angle:.1f}°")
                
                # Schedule servo to stop moving after a delay
                threading.Timer(0.5, lambda: setattr(servo_state, 'is_moving', False)).start()
            else:
                servo_state.error_count += 1
                self.log.error(f"Failed to execute {command.servo_id} command")
                
        except Exception as e:
            self.log.error(f"Error executing servo command: {e}")
            if command.servo_id in self.servo_states:
                self.servo_states[command.servo_id].error_count += 1
    
    def _send_hardware_command(self, servo_id: str, angle: float) -> bool:
        """Send command to actual hardware"""
        try:
            if servo_id == "cam_pan":
                self.px.set_cam_pan_angle(angle)
            elif servo_id == "cam_tilt":
                self.px.set_cam_tilt_angle(angle)
            elif servo_id == "dir_servo":
                self.px.set_dir_servo_angle(angle)
            else:
                self.log.error(f"Unknown servo ID: {servo_id}")
                return False
            
            return True
            
        except Exception as e:
            self.log.error(f"Hardware command failed for {servo_id}: {e}")
            return False
    
    def _enable_servo(self, servo_id: str):
        """Enable a servo that was previously disabled"""
        if servo_id in self.servo_states:
            self.servo_states[servo_id].is_enabled = True
            self.log.info(f"Re-enabled servo {servo_id}")
    
    def _disable_servo(self, servo_id: str):
        """Disable a servo to save power and reduce jitter"""
        if servo_id in self.servo_states:
            servo_state = self.servo_states[servo_id]
            servo_state.is_enabled = False
            self.log.info(f"Disabled idle servo {servo_id}")
            
            # Send a "neutral" command to stop any residual jitter
            try:
                self._send_hardware_command(servo_id, servo_state.current_angle)
            except Exception as e:
                self.log.warning(f"Could not send neutral command to {servo_id}: {e}")
    
    def emergency_stop_all_servos(self):
        """Emergency stop all servo operations"""
        self.emergency_stop = True
        with self.global_lock:
            self.command_queue.clear()
            for servo_state in self.servo_states.values():
                servo_state.is_moving = False
        self.log.critical("Emergency stop activated for all servos")
    
    def resume_servo_operations(self):
        """Resume servo operations after emergency stop"""
        self.emergency_stop = False
        self.log.info("Servo operations resumed")
    
    def get_servo_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all servos"""
        with self.global_lock:
            status = {
                "system_load_factor": self.system_load_factor,
                "emergency_stop": self.emergency_stop,
                "queue_size": len(self.command_queue),
                "servos": {}
            }
            
            for servo_id, state in self.servo_states.items():
                status["servos"][servo_id] = {
                    "current_angle": state.current_angle,
                    "target_angle": state.target_angle,
                    "is_enabled": state.is_enabled,
                    "is_moving": state.is_moving,
                    "command_count": state.command_count,
                    "error_count": state.error_count,
                    "consecutive_rejections": state.consecutive_rejections,
                    "last_command_age": time.time() - state.last_command_time,
                    "last_move_age": time.time() - state.last_actual_move_time,
                }
            
            return status
    
    def force_servo_position(self, servo_id: str, angle: float) -> bool:
        """Force servo to specific position, bypassing all anti-jitter protection (emergency use only)"""
        self.log.warning(f"FORCING servo {servo_id} to {angle}° - bypassing all protection!")
        try:
            success = self._send_hardware_command(servo_id, angle)
            if success and servo_id in self.servo_states:
                self.servo_states[servo_id].current_angle = angle
                self.servo_states[servo_id].target_angle = angle
            return success
        except Exception as e:
            self.log.error(f"Force command failed: {e}")
            return False
