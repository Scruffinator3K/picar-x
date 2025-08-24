from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import time
import threading

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    cv2 = None
    np = None

try:
    # Use tflite runtime if available (lighter than full TF)
    from tflite_runtime.interpreter import Interpreter, load_delegate  # type: ignore
except Exception:
    try:
        # Fall back to full TF if present
        from tensorflow.lite import Interpreter  # type: ignore
        load_delegate = None  # type: ignore
    except Exception:
        Interpreter = None  # type: ignore
        load_delegate = None  # type: ignore

try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None

from .logging_setup import init_logger
from .configuration import AppConfig


@dataclass
class DetectedObject:
    label: str
    score: float
    bbox_xywh: Tuple[float, float, float, float]  # x, y, w, h in pixels
    track_id: Optional[int] = None
    depth_cm: Optional[float] = None


@dataclass
class PerceptionState:
    ts: float
    objects: List[DetectedObject] = field(default_factory=list)
    gesture: Optional[str] = None  # 'stop'|'go'|'left'|'right'|None
    fps: float = 0.0
    inference_ms: float = 0.0


class _SimpleTracker:
    """Lightweight centroid-based tracker with exponential smoothing and ID assignment."""

    def __init__(self, max_age_frames: int = 10):
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1
        self._max_age = max_age_frames

    @staticmethod
    def _centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x, y, w, h = bbox
        return x + w / 2.0, y + h / 2.0

    def update(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        # Age tracks
        for t in self._tracks.values():
            t["age"] += 1

        # Assign by nearest neighbor on centroids
        unmatched = list(range(len(detections)))
        for tid, t in list(self._tracks.items()):
            if not unmatched:
                continue
            tx, ty = t["cx"], t["cy"]
            # find closest detection
            best_i = None
            best_d = 1e9
            for i in unmatched:
                cx, cy = self._centroid(detections[i].bbox_xywh)
                d = (cx - tx) ** 2 + (cy - ty) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i is not None and best_d < (t.get("gate", 80.0) ** 2):
                # associate
                cx, cy = self._centroid(detections[best_i].bbox_xywh)
                # EMA smoothing
                t["cx"] = 0.6 * t["cx"] + 0.4 * cx
                t["cy"] = 0.6 * t["cy"] + 0.4 * cy
                t["age"] = 0
                detections[best_i].track_id = tid
                unmatched.remove(best_i)

        # Create new tracks for unmatched
        for i in unmatched:
            cx, cy = self._centroid(detections[i].bbox_xywh)
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {"cx": cx, "cy": cy, "age": 0, "gate": 120.0}
            detections[i].track_id = tid

        # Drop old tracks
        for tid in [tid for tid, t in self._tracks.items() if t["age"] > self._max_age]:
            self._tracks.pop(tid, None)

        return detections


class Perception:
    """Edge perception: TFLite object detection, optional gesture recognition,
    light tracking, and approximate depth estimation. Can own its own capture loop.
    """

    def __init__(self, cfg: AppConfig, camera_index: int = 0, use_capture: bool = True):
        self.log = init_logger("picarx.perception")
        self.cfg = cfg
        self.enabled = cv2 is not None and np is not None
        self._cap = None
        self._thread = None
        self._stop = threading.Event()
        self._last_state = None
        # Avoid direct np.ndarray type in annotation when numpy may be missing
        # Use plain assignment to maximize compatibility with older type checkers
        self._last_frame = None

        # Detector
        self._interpreter = None
        self._input_shape = None
        self._input_buf = None
        if Interpreter and cfg.tflite.enabled and cfg.tflite.model_path:
            try:
                kwargs = {}
                # threads
                try:
                    nt = int(getattr(cfg.tflite, "num_threads", 1) or 1)
                    if nt > 0:
                        kwargs["num_threads"] = nt
                except Exception:
                    pass
                # delegate
                try:
                    delegate_name = getattr(cfg.tflite, "delegate", "") or ""
                    if delegate_name and load_delegate:
                        try:
                            kwargs["experimental_delegates"] = [load_delegate(delegate_name)]
                            self.log.info(f"Using TFLite delegate: {delegate_name}")
                        except Exception as e:
                            self.log.warning(f"Delegate load failed: {e}; continuing without delegate")
                except Exception:
                    pass
                self._interpreter = Interpreter(model_path=cfg.tflite.model_path, **kwargs)
                self._interpreter.allocate_tensors()
                self._input_shape = self._interpreter.get_input_details()[0]["shape"]
                self.log.info(f"Loaded TFLite model: {cfg.tflite.model_path}")
            except Exception as e:
                self.log.warning(f"TFLite init failed: {e}")
                self._interpreter = None

        # Optional labels mapping
        self._label_map: Optional[dict[int, str]] = None
        self._label_list: Optional[list[str]] = None
        if getattr(cfg.tflite, "labels_path", None):
            try:
                path = cfg.tflite.labels_path
                lines: list[str] = []
                with open(path, "r", encoding="utf-8") as f:
                    for ln in f:
                        s = ln.strip()
                        if s:
                            lines.append(s)
                lm: dict[int, str] = {}
                ordered: list[str] = []
                # Parse formats like:
                #  - "person" (one per line; index = line number)
                #  - "0 person" or "1:person" or "1 person"
                for idx, s in enumerate(lines):
                    name = s
                    id_parsed: Optional[int] = None
                    if ":" in s:
                        parts = s.split(":", 1)
                        if parts[0].strip().isdigit():
                            id_parsed = int(parts[0].strip())
                            name = parts[1].strip()
                    else:
                        parts = s.split()
                        if len(parts) > 1 and parts[0].isdigit():
                            id_parsed = int(parts[0])
                            name = " ".join(parts[1:]).strip()
                    if id_parsed is not None:
                        lm[id_parsed] = name
                    ordered.append(name)
                self._label_map = lm or None
                self._label_list = ordered or None
                self.log.info(f"Loaded labels: {path} ({len(ordered)} entries)")
            except Exception as e:
                self.log.warning(f"Label map load failed: {e}")
                self._label_map = None
                self._label_list = None

        # Gestures
        self._hands = None
        if mp and cfg.gestures.enabled:
            try:
                self._hands = mp.solutions.hands.Hands(max_num_hands=1,
                                                       min_detection_confidence=0.5,
                                                       min_tracking_confidence=0.5)
            except Exception as e:
                self.log.warning(f"MediaPipe Hands init failed: {e}")
                self._hands = None

        # Tracking
        self._tracker = _SimpleTracker(max_age_frames=cfg.tracking.max_age_frames)

        # OpenCV performance knobs
        try:
            if cv2 is not None:
                if getattr(self.cfg.performance, "cv2_use_optimized", True):
                    cv2.setUseOptimized(True)
                thr = int(getattr(self.cfg.performance, "cv2_threads", 0) or 0)
                if thr > 0:
                    cv2.setNumThreads(thr)
        except Exception:
            pass

        # Start capture
        if self.enabled and use_capture and (self._interpreter or self._hands):
            try:
                self._cap = cv2.VideoCapture(camera_index)
                try:
                    target_fps = int(getattr(self.cfg.performance, "camera_target_fps", 0) or 0)
                    if target_fps > 0:
                        self._cap.set(cv2.CAP_PROP_FPS, target_fps)
                except Exception:
                    pass
                if not self._cap or not self._cap.isOpened():
                    self.log.warning("Perception camera not available; external frames only")
                    self._cap = None
                else:
                    self._thread = threading.Thread(target=self._loop, daemon=True)
                    self._thread.start()
                    self.log.info("Perception capture loop started")
            except Exception as e:
                self.log.warning(f"Perception capture init failed: {e}")
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
        if self._hands:
            try:
                self._hands.close()
            except Exception:
                pass

    def last(self) -> Optional[PerceptionState]:
        return self._last_state

    def get_jpeg(self) -> Optional[bytes]:
        if self._last_frame is None or cv2 is None:
            return None
        try:
            ok, buf = cv2.imencode('.jpg', self._last_frame)
            return buf.tobytes() if ok else None
        except Exception:
            return None

    # loop
    def _loop(self):
        last_t = time.time()
        framecount = 0
        fps = 0.0
        det_every = 1
        try:
            det_every = max(1, int(getattr(self.cfg.performance, "detection_every_n", 1) or 1))
        except Exception:
            det_every = 1
        while not self._stop.is_set() and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self._last_frame = frame
            if (framecount % det_every) == 0:
                st = self.process(frame)
            else:
                st = self._last_state
            framecount += 1
            now = time.time()
            if now - last_t >= 1.0:
                fps = framecount / max(1e-3, (now - last_t))
                framecount = 0
                last_t = now
            if st:
                st.fps = fps
            time.sleep(0.005)

    # core
    def process(self, frame) -> Optional[PerceptionState]:
        if not self.enabled:
            return self._last_state

        t0 = time.time()
        dets: List[DetectedObject] = []

        # 1) Object detection via TFLite SSD-like models
        if self._interpreter is not None and self._input_shape is not None:
            try:
                _, in_h, in_w, _ = self._input_shape
                # Preallocate and reuse input buffer to reduce allocations
                if self._input_buf is None and np is not None:
                    self._input_buf = np.zeros((1, in_h, in_w, 3), dtype=np.uint8)
                img = cv2.resize(frame, (in_w, in_h))
                if self._input_buf is not None:
                    self._input_buf[0] = img
                inp = self._interpreter.get_input_details()[0]
                self._interpreter.set_tensor(inp['index'], self._input_buf if self._input_buf is not None else np.expand_dims(img, axis=0).astype('uint8'))
                self._interpreter.invoke()

                out0 = self._interpreter.get_output_details()
                # Common SSD outputs: boxes, classes, scores, count
                try:
                    boxes = self._interpreter.get_tensor(out0[0]['index'])[0]
                    classes = self._interpreter.get_tensor(out0[1]['index'])[0]
                    scores = self._interpreter.get_tensor(out0[2]['index'])[0]
                    count = int(self._interpreter.get_tensor(out0[3]['index'])[0])
                except Exception:
                    # fallback: some models omit count
                    boxes = self._interpreter.get_tensor(out0[0]['index'])[0]
                    classes = self._interpreter.get_tensor(out0[1]['index'])[0]
                    scores = self._interpreter.get_tensor(out0[2]['index'])[0]
                    count = len(scores)

                H, W = frame.shape[:2]
                thr = self.cfg.tflite.score_threshold
                for i in range(count):
                    s = float(scores[i])
                    if s < thr:
                        continue
                    y1, x1, y2, x2 = boxes[i]
                    x = float(x1 * W)
                    y = float(y1 * H)
                    w = float(max(1.0, (x2 - x1) * W))
                    h = float(max(1.0, (y2 - y1) * H))
                    class_id = int(classes[i])
                    label = self._class_to_label(class_id)
                    dets.append(DetectedObject(label=label, score=s, bbox_xywh=(x, y, w, h)))
            except Exception as e:
                self.log.debug(f"tflite detect error: {e}")

        # 2) Gesture recognition via MediaPipe Hands (heuristic commands)
        gesture: Optional[str] = None
        if self._hands is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = self._hands.process(rgb)
                rgb.flags.writeable = True
                if res.multi_hand_landmarks:
                    # Use simple heuristic: open palm => go, fist => stop, lateral position => left/right
                    h, w = frame.shape[:2]
                    lm = res.multi_hand_landmarks[0]
                    xs = [p.x * w for p in lm.landmark]
                    ys = [p.y * h for p in lm.landmark]
                    # Bounding box of landmarks
                    minx, maxx = min(xs), max(xs)
                    miny, maxy = min(ys), max(ys)
                    spanx = maxx - minx
                    spany = maxy - miny
                    # Fist vs open: ratio of bounding box area to convex hull? Use simple aspect: smaller area ~ fist
                    area = max(1.0, spanx * spany)
                    norm_area = area / float(w * h)
                    if norm_area < self.cfg.gestures.fist_threshold:
                        gesture = "stop"
                    else:
                        # lateral cue
                        cx = sum(xs) / len(xs)
                        center = w / 2.0
                        if cx < center - self.cfg.gestures.left_right_deadband_px:
                            gesture = "left"
                        elif cx > center + self.cfg.gestures.left_right_deadband_px:
                            gesture = "right"
                        else:
                            gesture = "go"
            except Exception as e:
                self.log.debug(f"gesture error: {e}")

        # Optionally keep only top-N detections by score to reduce downstream load
        try:
            max_n = int(getattr(self.cfg.tflite, "max_results", 0) or 0)
        except Exception:
            max_n = 0
        if max_n and len(dets) > max_n:
            dets.sort(key=lambda d: d.score, reverse=True)
            dets = dets[:max_n]

        # 3) Tracking assignment
        if dets and self.cfg.tracking.enabled:
            dets = self._tracker.update(dets)

        # 4) Depth estimation for target label (single monocular estimate)
        if dets and self.cfg.depth.enabled:
            H = frame.shape[0]
            f = self.cfg.depth.focal_length_px
            ref_h = self.cfg.depth.ref_object_height_cm
            target_lbl = self.cfg.depth.target_label
            for d in dets:
                if d.label == target_lbl and d.depth_cm is None:
                    _, _, _, h = d.bbox_xywh
                    if h > 1.0:
                        d.depth_cm = float((ref_h * f) / h)

        t1 = time.time()
        state = PerceptionState(ts=t1, objects=dets, gesture=gesture, inference_ms=(t1 - t0) * 1000.0)
        self._last_state = state
        return state

    # helpers
    def _class_to_label(self, cid: int) -> str:
        """Map class id to a human-friendly label if labels are available.
        Handles common 0/1-based offsets by trying cid, cid-1, cid+1.
        """
        try:
            if self._label_map:
                for k in (cid, cid - 1, cid + 1):
                    if k in self._label_map:
                        return self._label_map[k]
            if self._label_list and 0 <= cid < len(self._label_list):
                return self._label_list[cid]
            if self._label_list and 0 <= cid - 1 < len(self._label_list):
                return self._label_list[cid - 1]
        except Exception:
            pass
        return str(cid)
