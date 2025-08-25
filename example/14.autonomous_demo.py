from __future__ import annotations

import time

from picarx.picarx import Picarx
from picarx.autonomy import AutonomousController
from picarx.configuration import load_config
from picarx.telemetry import TelemetryServer
from picarx.watchdog import SafetyWatchdog
from picarx.mode import ModeManager
from picarx.perception import Perception
from picarx.slam import EnhancedSLAM
from picarx.adaptive import AdaptiveController
from picarx.learning import AdaptiveBandit, BanditConfig, ContextState, PerformanceMetrics
from picarx.power import PowerManager


def main():
    cfg = load_config()
    
    # Start Vilib for camera streaming (SunFounder recommended approach)
    try:
        import socket
        print("Starting Vilib camera streaming...")
        # Import vilib correctly - use capital V in Vilib class
        from vilib import Vilib
        print("Vilib imported successfully")
        
        print("Starting vilib camera...")
        Vilib.camera_start(vflip=False, hflip=False)
        print("Starting vilib web display...")
        Vilib.display(local=False, web=True)  # Note: web=True enables port 9000
        
        # Wait briefly for web server to bind
        def _port_listening(port: int, host: str = "127.0.0.1") -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.25)
                try:
                    return s.connect_ex((host, port)) == 0
                except Exception:
                    return False
        for _ in range(20):  # ~5s
            if _port_listening(9000) or _port_listening(9000, "0.0.0.0"):
                break
            time.sleep(0.25)
        else:
            print("Vilib web server not listening on :9000 after 5s; retrying once...")
            try:
                Vilib.camera_close()
            except Exception:
                pass
            time.sleep(0.5)
            Vilib.camera_start(vflip=False, hflip=False)
            Vilib.display(local=False, web=True)
        print("📹 Camera streaming available at http://0.0.0.0:9000/mjpg")
        vilib_started = True
    except Exception as e:
        print(f"Warning: Failed to start Vilib camera streaming: {e}")
        import traceback
        traceback.print_exc()
        vilib_started = False
    
    # Create Picarx object after camera is initialized
    px = Picarx()
    
    # Start edge perception (object detection, gestures, tracking)
    # Note: For Pi Camera, disable use_capture to avoid conflicts with Vilib
    perception = Perception(cfg, use_capture=not vilib_started)
    # Phase 3 modules
    slam = EnhancedSLAM()
    adaptive = AdaptiveController()
    
    # Enhanced adaptive bandit with context awareness - more conservative speeds
    bandit_config = BanditConfig(
        eps=0.10,  # Lower initial exploration for more stability
        arms=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Slower, more conservative speeds
        decay_rate=0.999,  # Faster exploration decay for stability
        min_eps=0.02,  # Lower minimum exploration
        context_window=50,  # Smaller context window for faster adaptation
        adaptation_rate=0.10  # Slower adaptation for stability
    )
    bandit = AdaptiveBandit(bandit_config)
    current_arm = 0.6  # Start with slower speed
    
    # Power/battery manager
    power = PowerManager(cfg)
    power.start()
    # Autonomy with speed scaling hook
    ctrl = AutonomousController(px, perception=perception, speed_scale_cb=lambda: current_arm)
    # Default to manual mode on startup
    modes = ModeManager(initial="manual")

    # Watchdog and telemetry
    wd = SafetyWatchdog(
        stop_fn=px.stop,
        get_distance_fn=px.get_distance,
        heartbeat_timeout_s=cfg.watchdog.heartbeat_timeout_s,
        hard_stop_distance_cm=cfg.watchdog.hard_stop_distance_cm,
    )
    wd.start()

    # COMPREHENSIVE ANTI-JITTER Pan/Tilt handler with aggressive rate limiting
    last_pan_tilt_command_time = 0.0
    pan_tilt_rate_limit = 2.0  # Minimum 2 seconds between commands
    
    def handle_pan_tilt(pan, tilt, pan_d, tilt_d, center=False):
        nonlocal last_pan_tilt_command_time
        
        current_time = time.time()
        
        # AGGRESSIVE rate limiting - reject rapid commands
        if current_time - last_pan_tilt_command_time < pan_tilt_rate_limit:
            print(f"Pan/tilt command REJECTED - too soon (interval: {current_time - last_pan_tilt_command_time:.2f}s)")
            return
        
        last_pan_tilt_command_time = current_time
        print(f"Pan/tilt command ACCEPTED - processing...")
        
        if center:
            px.set_cam_pan_angle(0)
            px.set_cam_tilt_angle(0)
            return
        
        # Recover current logical angles from last outputs; fall back to 0
        try:
            cur_pan = (px.cam_pan_cali_val - float(px._last_pan_angle_sent)) if px._last_pan_angle_sent is not None else 0.0
        except Exception:
            cur_pan = 0.0
        try:
            cur_tilt = (px.cam_tilt_cali_val - float(px._last_tilt_angle_sent)) if px._last_tilt_angle_sent is not None else 0.0
        except Exception:
            cur_tilt = 0.0
        
        new_pan = None
        new_tilt = None
        
        if pan is not None:
            new_pan = pan
        elif pan_d is not None:
            new_pan = cur_pan + pan_d
        
        if tilt is not None:
            new_tilt = tilt
        elif tilt_d is not None:
            new_tilt = cur_tilt + tilt_d
        
        # Execute commands with built-in servo anti-jitter protection
        if new_pan is not None:
            print(f"Setting pan to {new_pan:.1f}°")
            px.set_cam_pan_angle(new_pan)
        if new_tilt is not None:
            print(f"Setting tilt to {new_tilt:.1f}°")
            px.set_cam_tilt_angle(new_tilt)

    def on_slam(action, data):
        """Handle SLAM commands from UI"""
        try:
            if not slam:
                return None
                
            if action == "set_home":
                slam.set_home_position()
                return True
                
            elif action == "clear":
                slam.clear_map()
                return True
                
            elif action == "set_goal":
                x = data.get("x", 0)
                y = data.get("y", 0)
                slam.set_goal(x, y)
                return True
                
            elif action == "explore":
                return slam.start_exploration_mode()
                
            elif action == "return_home":
                return slam.return_home()
                
            elif action == "export":
                return slam.get_map_data()
                
            elif action == "status":
                return slam.get_slam_status()
                
            return None
        except Exception as e:
            print(f"Error handling SLAM command: {e}")
            return None

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
            # Enhanced learning statistics
            "learning": bandit.get_learning_stats(),
            # Perception summary for dashboard
            "perception": perception.get_perception_summary(),
            # Enhanced SLAM data for navigation
            "slam": slam.get_slam_status() if slam else None,
            # Comprehensive servo status for monitoring jitter
            "servo_status": px.get_servo_status(),
            **({
                "perception": {
                    "gesture": (perception.last().gesture if perception.last() else None),
                    "inference_ms": (perception.last().inference_ms if perception.last() else 0.0),
                    "fps": (perception.last().fps if perception.last() else 0.0),
                    "scene_description": (perception.last().scene_description if perception.last() else "unknown"),
                    "dominant_objects": (perception.last().dominant_objects if perception.last() else []),
                    "safety_assessment": (perception.last().safety_assessment if perception.last() else "unknown"),
                    "recommended_action": (perception.last().recommended_action if perception.last() else None),
                    # limit telemetry object list to avoid large payloads
                    "objects": [
                        {
                            "label": o.label,
                            "score": round(o.score, 3),
                            "bbox": list(map(float, o.bbox_xywh)),
                            "track_id": o.track_id,
                            "depth_cm": o.depth_cm,
                            "classification": getattr(o, 'classification', 'unknown'),
                            "confidence_level": getattr(o, 'confidence_level', 'unknown'),
                            "relative_size": getattr(o, 'relative_size', 'unknown'),
                            "position": getattr(o, 'position', 'unknown'),
                            "motion": getattr(o, 'motion', 'unknown'),
                            "threat_level": getattr(o, 'threat_level', 'unknown'),
                        }
                        for o in ((perception.last().objects if perception.last() else [])[:8])
                    ],
                }
            })
        },
        on_heartbeat=wd.heartbeat,
        on_set_mode=modes.set_mode,
        # Support both direct set and incremental (latched) adjustments from UI/CLI
        on_manual=lambda sp, st, sdp=None, std=None: (
            modes.adjust_manual(speed_delta=sdp, steer_delta=std)
            if (sdp is not None or std is not None) else
            modes.set_manual_command(speed=sp, steer=st)
        ),
    on_pan_tilt=handle_pan_tilt,
        on_slam=on_slam,
        # Remove get_snapshot since we use Vilib for camera streaming
        get_snapshot=None,
        get_map=lambda: slam.get_map_jpeg(scale=2),
    )
    tel.start()
    
    # Initialize performance tracking for enhanced learning
    last_performance_update = time.time()
    collision_count = 0
    maneuver_count = 0
    
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
                # Enhanced autonomous control with rich perception data and context-aware learning
                perception_state = perception.last()
                
                # Gather context information for adaptive learning
                battery_status = power.status()
                current_context = ContextState(
                    battery_level=battery_status.voltage_v / 12.6,  # Normalize to 0-1 range
                    obstacle_density=0.0,  # Will be calculated from perception
                    lighting_quality=1.0,  # Assume good lighting for now
                    surface_type="smooth",  # Default assumption
                    time_of_day="day",  # Could be enhanced with actual time
                    recent_collisions=collision_count,
                    successful_maneuvers=maneuver_count
                )
                
                # Calculate obstacle density from perception data
                if perception_state and perception_state.objects:
                    obstacle_density = min(1.0, len(perception_state.objects) / 10.0)
                    current_context.obstacle_density = obstacle_density
                
                # Use perception-based navigation when available
                speed_reduction = 1.0
                emergency_action = None
                
                if perception_state and perception_state.recommended_action:
                    action = perception_state.recommended_action
                    if action == "stop":
                        px.stop()
                        collision_count += 1  # Count near-misses as learning events
                        continue
                    elif action == "avoid_left":
                        px.set_dir_servo_angle(-20)  # Steer left
                        px.forward(20)  # Slow speed
                        maneuver_count += 1
                        emergency_action = "avoid_left"
                        continue
                    elif action == "avoid_right":
                        px.set_dir_servo_angle(20)   # Steer right
                        px.forward(20)  # Slow speed
                        maneuver_count += 1
                        emergency_action = "avoid_right"
                        continue
                    elif action == "slow":
                        speed_reduction = 0.5  # Reduce speed but continue with normal navigation
                
                # Context-aware bandit selection
                selected_arm = bandit.select(current_context)
                current_arm = selected_arm * power.speed_scale() * speed_reduction
                status = ctrl.tick()
                
                # Enhanced reward function incorporating perception and performance metrics
                dist = px.get_distance()
                spd = ctrl.last_speed_cmd()
                
                # Base reward calculation with more nuanced scoring
                base_reward = 0.0
                if dist and dist > 40:
                    base_reward = (spd / 70.0) * 0.8  # High reward for confident movement in clear areas
                elif dist and dist > 25:
                    base_reward = (spd / 90.0) * 0.5  # Moderate reward for cautious movement
                elif dist and dist > 15:
                    base_reward = (spd / 120.0) * 0.2  # Small reward for very cautious movement
                else:
                    base_reward = -0.1 if (dist and dist < 10) else 0.0  # Penalty for very close obstacles
                
                # Perception-based reward adjustments
                perception_bonus = 0.0
                if perception_state:
                    if perception_state.safety_assessment == "danger":
                        perception_bonus = -0.4  # Strong penalty for dangerous situations
                    elif perception_state.safety_assessment == "caution":
                        perception_bonus = -0.1  # Small penalty for caution
                    elif perception_state.safety_assessment == "safe":
                        perception_bonus = 0.15  # Bonus for safe conditions
                    
                    # Bonus for maintaining good scene understanding
                    if perception_state.scene_description in ["clear"]:
                        perception_bonus += 0.1
                    elif perception_state.scene_description in ["person_nearby"]:
                        perception_bonus += 0.05  # Smaller bonus for person detection
                
                # Emergency action penalties/bonuses
                emergency_bonus = 0.0
                if emergency_action:
                    emergency_bonus = 0.2  # Reward successful avoidance maneuvers
                
                # Combine all reward components
                total_reward = base_reward + perception_bonus + emergency_bonus
                
                # Create performance metrics for multi-objective learning
                current_performance = PerformanceMetrics(
                    speed_efficiency=min(1.0, spd / 100.0),
                    safety_score=min(1.0, (dist or 0) / 50.0) if dist else 0.0,
                    exploration_score=bandit.current_eps,  # Reward exploration
                    energy_efficiency=battery_status.voltage_v / 12.6,  # Battery efficiency
                    mission_progress=0.5 + total_reward * 0.5,  # Combined mission score
                )
                
                # Update bandit with context and performance metrics
                bandit.update(
                    arm=selected_arm,
                    reward=total_reward,
                    context=current_context,
                    metrics=current_performance
                )
                
                # Periodic performance logging
                now = time.time()
                if now - last_performance_update > 30.0:  # Every 30 seconds
                    learning_stats = bandit.get_learning_stats()
                    print(f"Learning Progress: Success Rate: {learning_stats['success_rate']:.1%}, "
                          f"Exploration: {learning_stats['current_exploration']:.3f}, "
                          f"Best Arm: {learning_stats['best_arm']}, "
                          f"Adaptations: {learning_stats['adaptation_triggers']}")
                    last_performance_update = now
                    
                    # Reset counters periodically
                    if collision_count > 10:
                        collision_count = collision_count // 2
                    if maneuver_count > 20:
                        maneuver_count = maneuver_count // 2
                
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
