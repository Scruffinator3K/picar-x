from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import time
import threading

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    cv2 = None  # optional in headless or no-op environments
    np = None

from .logging_setup import init_logger


class Vision:
    """Vision tuned for pea-gravel road with grass edges.

    - Uses HSV thresholding to detect grass (greens) as a mask.
    - Estimates road edges at the bottom ROI and computes a center-line error.
    - Optionally runs a background capture loop when OpenCV is available.
    """

    def __init__(self, *, camera_index: int = 0, use_capture: bool = True):
        self.log = init_logger("picarx.vision")
        self.enabled = cv2 is not None and np is not None
        self._last: Optional[Dict[str, Any]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap = None

        # HSV ranges tuned for typical grass; may need field tuning
        # OpenCV Hue in [0,179]. Green approx: 35-85
        self.hsv_lower = (35, 40, 40)
        self.hsv_upper = (85, 255, 255)
        # ROI: use bottom 35% of the frame
        self.roi_y = 0.65

        if not self.enabled:
            self.log.info("OpenCV/NumPy not available; vision disabled")
            return

        if use_capture:
            try:
                self._cap = cv2.VideoCapture(camera_index)
                if not self._cap or not self._cap.isOpened():
                    self.log.warning("Camera capture not available; running in manual frame mode")
                    self._cap = None
                else:
                    self._thread = threading.Thread(target=self._loop, daemon=True)
                    self._thread.start()
                    self.log.info("Vision capture loop started")
            except Exception as e:
                self.log.warning(f"Vision capture init failed: {e}")
                self._cap = None

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=0.5)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

    def last(self) -> Optional[Dict[str, Any]]:
        return self._last

    def set_frame(self, frame) -> Optional[Dict[str, Any]]:
        """Process an externally-provided frame and update results."""
        return self.process(frame)

    # core
    def _loop(self):
        while not self._stop.is_set() and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self.process(frame)
            time.sleep(0.02)  # ~50 Hz

    def process(self, frame) -> Optional[Dict[str, Any]]:
        if not self.enabled or frame is None:
            return self._last
        try:
            h, w = frame.shape[:2]
            # downscale for speed
            scale = 320.0 / float(w) if w > 320 else 1.0
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                h, w = frame.shape[:2]

            # ROI crop: bottom portion
            y0 = int(h * self.roi_y)
            roi = frame[y0:, :]

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(self.hsv_lower), np.array(self.hsv_upper))
            # morphology to clean up
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # compute column transitions using gradient along x of mask
            # focus on near-bottom scanline average to stabilize
            scan = mask[-5:, :].mean(axis=0)  # average last 5 rows
            # normalize to 0..1
            scan = scan / 255.0
            # gradient
            grad = np.diff(scan)

            # find left edge (road -> grass) near left half: look for positive peak
            left_region = grad[: w // 2]
            right_region = grad[w // 2 :]
            left_idx = int(np.argmax(left_region)) if left_region.size else None
            right_idx = (w // 2) + int(np.argmin(right_region)) if right_region.size else None

            left_edge = left_idx if left_idx is not None and left_region.size else None
            right_edge = right_idx if right_region.size else None

            center_x = w / 2.0
            road_center_x: Optional[float] = None

            if left_edge is not None and right_edge is not None:
                road_center_x = (left_edge + right_edge) / 2.0
            elif left_edge is not None:
                road_center_x = left_edge + 0.4 * w  # assume ~40% half-width
            elif right_edge is not None:
                road_center_x = right_edge - 0.4 * w

            line_error = None
            if road_center_x is not None:
                line_error = float((road_center_x - center_x) / max(1.0, w))
                # clamp ~road_error to [-1,1]
                line_error = max(-1.0, min(1.0, line_error))

            result = {
                "ts": time.time(),
                "w": w,
                "h": h,
                "roi_y": self.roi_y,
                "left_edge": left_edge,
                "right_edge": right_edge,
                "line_error": line_error,
            }
            self._last = result
            return result
        except Exception as e:
            self.log.debug(f"vision process error: {e}")
            return self._last
