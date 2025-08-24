from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .logging_setup import init_logger


log = init_logger("picarx.config")


@dataclass
class VisionConfig:
    hsv_lower: tuple[int, int, int] = (35, 40, 40)
    hsv_upper: tuple[int, int, int] = (85, 255, 255)
    roi_y: float = 0.65
    adaptive_v_min: int = 0  # 0 disables; else overrides V low bound dynamically up to this min based on ROI stats
    enable_multi_edges: bool = True


@dataclass
class TFLiteConfig:
    enabled: bool = False
    model_path: str = ""  # e.g., /opt/vilib/detect.tflite or efficientdet path
    labels_path: str = ""  # optional COCO labels
    score_threshold: float = 0.4
    max_results: int = 10
    num_threads: int = 1  # CPU threads for tflite
    delegate: str = ""    # e.g., 'edgetpu' to try Coral, or path to delegate .so


@dataclass
class GestureConfig:
    enabled: bool = False
    # simple mapping outputs: stop/go/left/right; additional gestures can be added later
    left_right_deadband_px: int = 40  # horizontal deadband around center for left/right
    fist_threshold: float = 0.15  # heuristic ratio for fist vs open palm


@dataclass
class DepthConfig:
    enabled: bool = False
    focal_length_px: float = 600.0  # approximate for 640x480 webcam; calibrate per camera
    ref_object_height_cm: float = 16.0  # default height of object of interest (e.g., traffic sign diameter)
    target_label: str = "person"  # label to estimate depth for


@dataclass
class TrackingConfig:
    enabled: bool = True
    max_age_frames: int = 10
    process_every_n: int = 1


@dataclass
class PerformanceConfig:
    # OpenCV optimizations
    cv2_use_optimized: bool = True
    cv2_threads: int = 0  # 0 lets OpenCV decide; >0 to pin
    # Perception throttling
    detection_every_n: int = 1  # run detector every n frames
    camera_target_fps: int = 30


@dataclass
class FusionConfig:
    obstacle_stop_cm: float = 18.0
    obstacle_slow_cm: float = 28.0


@dataclass
class SpeedConfig:
    max_speed: int = 100
    min_speed: int = 0
    max_accel_per_s: float = 80.0
    max_decel_per_s: float = 120.0
    emergency_decel_per_s: float = 300.0


@dataclass
class WatchdogConfig:
    heartbeat_timeout_s: float = 0.5
    hard_stop_distance_cm: float = 10.0


@dataclass
class PowerConfig:
    low_voltage_v: float = 6.8    # 2S Li-ion pack ~3.4V/cell
    critical_voltage_v: float = 6.4
    dynamic_speed_scale: bool = True
    min_speed_scale_at_low: float = 0.6
    sleep_on_critical: bool = False


@dataclass
class AppConfig:
    vision: VisionConfig = VisionConfig()
    tflite: TFLiteConfig = TFLiteConfig()
    gestures: GestureConfig = GestureConfig()
    depth: DepthConfig = DepthConfig()
    tracking: TrackingConfig = TrackingConfig()
    performance: PerformanceConfig = PerformanceConfig()
    fusion: FusionConfig = FusionConfig()
    speed: SpeedConfig = SpeedConfig()
    watchdog: WatchdogConfig = WatchdogConfig()
    power: PowerConfig = PowerConfig()


def _default_config_path() -> Path:
    # Prefer user config; can be overridden via PICARX_CONFIG
    env = os.getenv("PICARX_CONFIG")
    if env:
        return Path(env)
    return Path(os.path.expanduser("~")) / ".picarx" / "config.json"


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    path = Path(path) if path else _default_config_path()
    try:
        if not path.exists():
            return AppConfig()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # helper to merge dict into dataclass
        def _merge(dc_cls, dc_default, key):
            section = data.get(key, {}) if isinstance(data, dict) else {}
            if not isinstance(section, dict):
                section = {}
            values = {**asdict(dc_default), **section}
            return dc_cls(**values)

        return AppConfig(
            vision=_merge(VisionConfig, VisionConfig(), "vision"),
            tflite=_merge(TFLiteConfig, TFLiteConfig(), "tflite"),
            gestures=_merge(GestureConfig, GestureConfig(), "gestures"),
            depth=_merge(DepthConfig, DepthConfig(), "depth"),
            tracking=_merge(TrackingConfig, TrackingConfig(), "tracking"),
            performance=_merge(PerformanceConfig, PerformanceConfig(), "performance"),
            fusion=_merge(FusionConfig, FusionConfig(), "fusion"),
            speed=_merge(SpeedConfig, SpeedConfig(), "speed"),
            watchdog=_merge(WatchdogConfig, WatchdogConfig(), "watchdog"),
            power=_merge(PowerConfig, PowerConfig(), "power"),
        )
    except Exception as e:
        log.warning(f"load_config failed: {e}")
        return AppConfig()
