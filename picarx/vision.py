from __future__ import annotations

from typing import Any, Dict, Optional
import time

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # optional in headless or no-op environments

from .logging_setup import init_logger


class Vision:
    """Lightweight vision pipeline with optional OpenCV. Placeholder for TFLite integration.

    This class is intentionally simple to keep Phase 2 incremental.
    """

    def __init__(self):
        self.log = init_logger("picarx.vision")
        self.enabled = cv2 is not None
        self._last: Optional[Dict[str, Any]] = None
        if not self.enabled:
            self.log.info("OpenCV not available; vision disabled")

    def process(self, frame) -> Optional[Dict[str, Any]]:
        if not self.enabled or frame is None:
            return self._last
        # Placeholder: compute simple brightness metric
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            result = {"brightness": brightness, "ts": time.time()}
            self._last = result
            return result
        except Exception as e:
            self.log.debug(f"vision process error: {e}")
            return self._last
