#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from picarx.perception import Perception
from picarx.configuration import load_config
import time

def test_perception():
    print("Loading config...")
    cfg = load_config()
    
    print("Creating Perception instance...")
    p = Perception(cfg, use_capture=True)
    
    print("Waiting 3 seconds for camera initialization...")
    time.sleep(3)
    
    print("Testing get_jpeg()...")
    jpeg_data = p.get_jpeg()
    
    if jpeg_data:
        print(f"✅ JPEG data obtained: {len(jpeg_data)} bytes")
        # Test if it's valid JPEG
        if jpeg_data.startswith(b'\xff\xd8') and jpeg_data.endswith(b'\xff\xd9'):
            print("✅ Valid JPEG header/footer detected")
        else:
            print("❌ Invalid JPEG format")
    else:
        print("❌ No JPEG data returned")
    
    # Check if perception is enabled
    print(f"Perception enabled: {p.enabled}")
    print(f"Last frame available: {p._last_frame is not None}")
    
    p.stop()

if __name__ == "__main__":
    test_perception()
