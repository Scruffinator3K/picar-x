#!/usr/bin/env python3
"""
Updated autonomous demo with proper camera sharing for streaming
"""

from __future__ import annotations

import time
import cv2
import threading
from typing import Optional

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


class SharedCamera:
    """Shared camera instance that can provide frames to both perception and streaming"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.last_frame = None
        self.last_jpeg = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        
    def start(self):
        """Start the camera capture thread"""
        if self.running:
            return
            
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"❌ Failed to open camera {self.camera_index}")
                return False
                
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print("✅ Shared camera started")
            return True
            
        except Exception as e:
            print(f"❌ Camera start failed: {e}")
            return False
    
    def stop(self):
        """Stop the camera capture"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print("📹 Shared camera stopped")
    
    def _capture_loop(self):
        """Main capture loop running in background thread"""
        while self.running and self.cap:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.last_frame = frame.copy()
                        # Encode to JPEG for streaming
                        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if success:
                            self.last_jpeg = buffer.tobytes()
                else:
                    time.sleep(0.01)  # Brief pause if frame read failed
                    
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Camera capture error: {e}")
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[any]:
        """Get the latest frame for perception processing"""
        with self.lock:
            return self.last_frame.copy() if self.last_frame is not None else None
    
    def get_jpeg(self) -> Optional[bytes]:
        """Get the latest JPEG data for streaming"""
        with self.lock:
            return self.last_jpeg


def main():
    cfg = load_config()
    px = Picarx()
    
    # Create shared camera instance
    shared_camera = SharedCamera(camera_index=0)
    if not shared_camera.start():
        print("❌ Failed to start camera, exiting")
        return
    
    # Wait for camera to stabilize
    time.sleep(2)
    
    # Start perception WITHOUT its own capture (use_capture=False)
    perception = Perception(cfg, use_capture=False)
    
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
        get_snapshot=shared_camera.get_jpeg,  # Use shared camera for streaming
        get_map=lambda: slam.get_map_jpeg(scale=2),
    )
    tel.start()
    
    print("🚗 Autonomous demo starting with shared camera...")
    print("📹 Camera stream available at: http://192.168.86.22:8080/stream")
    print("🎛️  Dashboard available at: http://192.168.86.22:8080/")
    
    try:
        while True:
            # Get frame from shared camera for perception
            frame = shared_camera.get_frame()
            if frame is not None and perception.enabled:
                perception.process(frame)
            
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
                bandit.update(reward)
                
                # Log adaptive learning
                if (int(time.time()) % 30) == 0:  # every 30s
                    adaptive.log_state({"speed": spd, "distance": dist, "reward": reward})
                
                # SLAM update
                if dist and dist > 0:
                    slam.update_scan([dist], px.dir_servo_angle, *px.get_position_simulation())
                
            time.sleep(0.05)  # 20 Hz main loop
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        shared_camera.stop()
        power.stop()
        wd.stop()
        tel.stop()
        px.stop()


if __name__ == "__main__":
    main()
