#!/usr/bin/env python3

try:
    from vilib import Vilib
    print("✅ Vilib is available")
    
    # Test camera start
    Vilib.camera_start(vflip=False, hflip=False)
    print("✅ Camera started with Vilib")
    
    # Test display
    Vilib.display(local=False, web=True)
    print("✅ Vilib display started on web")
    print("📹 Video should be available at http://192.168.86.22:9000/mjpg")
    
    import time
    time.sleep(3)
    
    # Stop
    Vilib.camera_close()
    print("✅ Camera closed")
    
except ImportError as e:
    print(f"❌ Vilib not available: {e}")
except Exception as e:
    print(f"❌ Error with Vilib: {e}")
