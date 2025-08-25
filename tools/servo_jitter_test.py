#!/usr/bin/env python3
"""
Comprehensive Servo Jitter Test and Monitoring Tool

This script provides extensive testing and monitoring of the servo anti-jitter system.
Use this to verify that the comprehensive servo controller is working correctly
and to diagnose any remaining jitter issues.
"""

import time
import json
import argparse
from picarx.picarx import Picarx
from picarx.logging_setup import init_logger


def test_servo_anti_jitter(px, servo_id, angle_range, test_duration=30, command_frequency=10):
    """Test anti-jitter protection by sending rapid commands"""
    logger = init_logger("servo_test")
    
    logger.info(f"Starting anti-jitter test for {servo_id}")
    logger.info(f"Angle range: {angle_range}, Duration: {test_duration}s, Frequency: {command_frequency}Hz")
    
    start_time = time.time()
    command_count = 0
    accepted_count = 0
    rejected_count = 0
    
    min_angle, max_angle = angle_range
    angle = min_angle
    direction = 1
    
    while time.time() - start_time < test_duration:
        # Send command
        if servo_id == "cam_pan":
            success = px.servo_controller.set_servo_angle("cam_pan", angle, source="jitter_test")
        elif servo_id == "cam_tilt":
            success = px.servo_controller.set_servo_angle("cam_tilt", angle, source="jitter_test")
        elif servo_id == "dir_servo":
            success = px.servo_controller.set_servo_angle("dir_servo", angle, source="jitter_test")
        else:
            logger.error(f"Unknown servo: {servo_id}")
            return
        
        command_count += 1
        if success:
            accepted_count += 1
        else:
            rejected_count += 1
        
        # Move angle for next command
        angle += direction * 5
        if angle >= max_angle:
            direction = -1
            angle = max_angle
        elif angle <= min_angle:
            direction = 1
            angle = min_angle
        
        # Wait for next command
        time.sleep(1.0 / command_frequency)
    
    # Results
    logger.info(f"Test completed for {servo_id}")
    logger.info(f"Commands sent: {command_count}")
    logger.info(f"Commands accepted: {accepted_count} ({accepted_count/command_count*100:.1f}%)")
    logger.info(f"Commands rejected: {rejected_count} ({rejected_count/command_count*100:.1f}%)")
    
    return {
        "servo_id": servo_id,
        "commands_sent": command_count,
        "commands_accepted": accepted_count,
        "commands_rejected": rejected_count,
        "acceptance_rate": accepted_count / command_count if command_count > 0 else 0,
        "test_duration": test_duration
    }


def monitor_servo_status(px, duration=60, interval=5):
    """Monitor servo status over time"""
    logger = init_logger("servo_monitor")
    
    logger.info(f"Monitoring servo status for {duration}s (every {interval}s)")
    
    start_time = time.time()
    status_history = []
    
    while time.time() - start_time < duration:
        status = px.get_servo_status()
        status["timestamp"] = time.time()
        status_history.append(status)
        
        logger.info(f"Servo Status at {time.time() - start_time:.1f}s:")
        logger.info(f"  System Load: {status['system_load_factor']:.2f}")
        logger.info(f"  Queue Size: {status['queue_size']}")
        logger.info(f"  Emergency Stop: {status['emergency_stop']}")
        
        for servo_id, servo_status in status["servos"].items():
            logger.info(f"  {servo_id}:")
            logger.info(f"    Enabled: {servo_status['is_enabled']}")
            logger.info(f"    Current Angle: {servo_status['current_angle']:.1f}°")
            logger.info(f"    Commands: {servo_status['command_count']}")
            logger.info(f"    Errors: {servo_status['error_count']}")
            logger.info(f"    Rejections: {servo_status['consecutive_rejections']}")
            logger.info(f"    Last Command: {servo_status['last_command_age']:.1f}s ago")
        
        time.sleep(interval)
    
    return status_history


def stress_test_all_servos(px, duration=60):
    """Stress test all servos simultaneously"""
    logger = init_logger("servo_stress_test")
    
    logger.info(f"Starting stress test for all servos ({duration}s)")
    
    import threading
    import random
    
    # Test parameters for each servo
    servo_configs = {
        "cam_pan": {"range": [-45, 45], "frequency": 5},
        "cam_tilt": {"range": [-20, 20], "frequency": 5},
        "dir_servo": {"range": [-15, 15], "frequency": 3}
    }
    
    results = {}
    threads = []
    
    def servo_stress_worker(servo_id, config):
        """Worker function for individual servo stress testing"""
        start_time = time.time()
        command_count = 0
        accepted_count = 0
        
        min_angle, max_angle = config["range"]
        frequency = config["frequency"]
        
        while time.time() - start_time < duration:
            # Random angle within range
            angle = random.uniform(min_angle, max_angle)
            
            success = px.servo_controller.set_servo_angle(servo_id, angle, source="stress_test")
            command_count += 1
            if success:
                accepted_count += 1
            
            time.sleep(1.0 / frequency + random.uniform(0, 0.1))  # Add some jitter
        
        results[servo_id] = {
            "commands_sent": command_count,
            "commands_accepted": accepted_count,
            "acceptance_rate": accepted_count / command_count if command_count > 0 else 0
        }
    
    # Start threads for each servo
    for servo_id, config in servo_configs.items():
        thread = threading.Thread(target=servo_stress_worker, args=(servo_id, config))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Print results
    logger.info("Stress test completed!")
    for servo_id, result in results.items():
        logger.info(f"{servo_id}: {result['commands_accepted']}/{result['commands_sent']} "
                   f"({result['acceptance_rate']*100:.1f}% acceptance)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive servo jitter testing and monitoring")
    parser.add_argument("--test", choices=["anti-jitter", "monitor", "stress"], 
                       help="Type of test to run")
    parser.add_argument("--servo", choices=["cam_pan", "cam_tilt", "dir_servo"], 
                       help="Servo to test (for anti-jitter test)")
    parser.add_argument("--duration", type=int, default=30, 
                       help="Test duration in seconds")
    parser.add_argument("--output", type=str, 
                       help="Output file for results (JSON format)")
    
    args = parser.parse_args()
    
    logger = init_logger("servo_test_main")
    logger.info("Initializing Picarx with comprehensive servo controller...")
    
    # Initialize Picarx
    px = Picarx()
    
    # Wait for servo controller to initialize
    time.sleep(2.0)
    
    results = {}
    
    try:
        if args.test == "anti-jitter":
            if not args.servo:
                logger.error("--servo required for anti-jitter test")
                return
            
            servo_ranges = {
                "cam_pan": [-45, 45],
                "cam_tilt": [-20, 20], 
                "dir_servo": [-15, 15]
            }
            
            results = test_servo_anti_jitter(
                px, args.servo, servo_ranges[args.servo], 
                test_duration=args.duration
            )
            
        elif args.test == "monitor":
            results = monitor_servo_status(px, duration=args.duration)
            
        elif args.test == "stress":
            results = stress_test_all_servos(px, duration=args.duration)
            
        else:
            logger.error("No test specified. Use --test [anti-jitter|monitor|stress]")
            return
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")
        
        # Emergency stop all servos at the end
        logger.info("Emergency stopping all servos...")
        px.emergency_stop_servos()
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        px.emergency_stop_servos()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        px.emergency_stop_servos()
        raise


if __name__ == "__main__":
    main()
