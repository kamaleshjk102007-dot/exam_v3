"""
SentinelEye — AI Detector
Headless server wrapper for exam_detector_v3.py logic.
Includes PyTorch 2.6 compatibility fix.

Detection classes (from your best.pt model):
  ACTIVE  : phone, Hand Under Table, Look Around, Wave, Bend Over The Desk
  IGNORED : Normal, Stand Up  (no alert, no box)
"""

import cv2
import time
import base64
import logging
import threading
import datetime
import functools
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from typing import Callable, Optional

logger = logging.getLogger("sentinel.detector")

# ── YOLO ─────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed — run: pip install ultralytics")

# ── Constants (mirrors exam_detector_v3.py) ───────────────────────────────────
CLASS_NAMES = [
    "Bend Over The Desk",
    "Hand Under Table",
    "Look Around",
    "Normal",
    "Stand Up",
    "Wave",
    "phone",
]

IGNORED_CLASSES = {"Stand Up", "Normal"}

ALERT_CLASSES = {
    "phone",
    "Hand Under Table",
    "Look Around",
    "Wave",
    "Bend Over The Desk",
}

# BGR colours for bounding boxes
CLASS_COLORS = {
    "Bend Over The Desk": (0,   165, 255),   # orange
    "Hand Under Table":   (255, 255,   0),   # cyan
    "Look Around":        (0,   255, 255),   # yellow
    "Wave":               (255,   0, 255),   # magenta
    "phone":              (0,     0, 255),   # red
}

# Per-class confidence thresholds
CLASS_CONF = {
    "phone":              0.45,
    "Look Around":        0.40,
    "Hand Under Table":   0.45,
    "Wave":               0.45,
    "Bend Over The Desk": 0.35,
}
DEFAULT_CONF = 0.40

# Min/max bounding box area ratios (filters tiny/huge false positives)
MIN_BOX_RATIO = {
    "phone":              0.003,
    "Wave":               0.010,
    "Look Around":        0.005,
    "Hand Under Table":   0.008,
    "Bend Over The Desk": 0.010,
}
MAX_BOX_RATIO = {"phone": 0.20}

# Alert severity and messages
SEVERITY_MAP = {
    "phone":              "critical",
    "Hand Under Table":   "critical",
    "Look Around":        "warning",
    "Wave":               "warning",
    "Bend Over The Desk": "info",
}
ALERT_MESSAGES = {
    "phone":              "Mobile phone detected — exam violation",
    "Hand Under Table":   "Hand under desk — possible concealed device",
    "Look Around":        "Student looking around — possible copying",
    "Wave":               "Waving gesture — possible signalling",
    "Bend Over The Desk": "Student bending — unusual posture",
}

# Snapshot directory — exported so detection.py can import it
SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
JPEG_QUALITY = 75


# ── Frame Result ──────────────────────────────────────────────────────────────
class FrameResult:
    """Holds everything produced by analyzing one frame."""
    __slots__ = [
        "frame_number", "timestamp", "current_stats",
        "alerts", "fps", "annotated_jpg_b64",
        "total_violations", "cumulative_stats",
    ]

    def __init__(self):
        self.frame_number:      int   = 0
        self.timestamp:         float = 0.0
        self.current_stats:     dict  = {}
        self.alerts:            list  = []
        self.fps:               float = 0.0
        self.annotated_jpg_b64: str   = ""
        self.total_violations:  int   = 0
        self.cumulative_stats:  dict  = {}


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _class_name(results, cls_i: int) -> str:
    name = results[0].names.get(cls_i)
    if name:
        return name
    if 0 <= cls_i < len(CLASS_NAMES):
        return CLASS_NAMES[cls_i]
    return f"class_{cls_i}"


def draw_detections(frame: np.ndarray, results, global_conf=None):
    """
    Validate detections, draw boxes, return (annotated_frame, stats, alert_reasons).
    Mirrors exam_detector_v3.draw_detections — headless version.
    """
    h, w       = frame.shape[:2]
    frame_area = max(h * w, 1)
    stats      = defaultdict(int)
    alerts     = []
    validated  = []

    for box in results[0].boxes:
        conf     = float(box.conf[0])
        cls_i    = int(box.cls[0])
        cls_name = _class_name(results, cls_i)

        if cls_name in IGNORED_CLASSES:
            continue

        threshold = global_conf if global_conf is not None else CLASS_CONF.get(cls_name, DEFAULT_CONF)
        if conf < threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        ratio = (bw * bh) / frame_area
        if ratio < MIN_BOX_RATIO.get(cls_name, 0.008):
            continue
        if ratio > MAX_BOX_RATIO.get(cls_name, 1.0):
            continue
        # Phone aspect ratio filter (must be taller than wide)
        if cls_name == "phone" and (bh / max(bw, 1)) < 0.7:
            continue

        validated.append((cls_name, conf, x1, y1, x2, y2))

    for cls_name, conf, x1, y1, x2, y2 in validated:
        stats[cls_name] += 1
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name}  {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, lh + 6)
        cv2.rectangle(frame, (x1, label_y - lh - 6), (x1 + lw + 4, label_y), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if cls_name in ALERT_CLASSES:
            alerts.append(cls_name)

    return frame, dict(stats), alerts


def draw_hud(frame: np.ndarray, cumulative_stats: dict, fps: float,
             total_violations: int, camera_name: str = "") -> np.ndarray:
    """Draw status bar and cumulative detection panel."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 62), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    ts  = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cam = f"  [{camera_name}]" if camera_name else ""
    cv2.putText(frame, f"SENTINEL EYE{cam}  |  {ts}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:5.1f}   Violations: {total_violations}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 1, cv2.LINE_AA)

    display = [c for c in CLASS_NAMES if c not in IGNORED_CLASSES]
    pw, ph  = 240, len(display) * 26 + 32
    px, py  = w - pw - 6, 70
    ov2 = frame.copy()
    cv2.rectangle(ov2, (px - 4, py - 4), (px + pw, py + ph), (10, 10, 10), -1)
    cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "CUMULATIVE DETECTIONS", (px + 2, py + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    for i, cls in enumerate(display):
        cnt   = cumulative_stats.get(cls, 0)
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        if cls in ALERT_CLASSES and cnt > 0:
            color = (80, 80, 255)
        y = py + 30 + i * 26
        cv2.putText(frame, f"{cls[:22]:<22}  {cnt:>4}",
                    (px + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    return frame


def save_snapshot(frame: np.ndarray, reason: str) -> str:
    """Save violation frame to snapshots/ folder (with annotations)."""
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tag   = reason.replace(" ", "_")[:20]
    fname = SNAPSHOT_DIR / f"violation_{ts}_{tag}.jpg"
    cv2.imwrite(str(fname), frame)
    logger.info(f"Snapshot saved: {fname}")
    return str(fname)


def encode_frame(frame: np.ndarray, quality: int = JPEG_QUALITY) -> str:
    """JPEG-encode a frame to base64 string for WebSocket transport."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


# ── Main Detector Class ───────────────────────────────────────────────────────
class ExamDetector:
    """
    Thread-safe wrapper around YOLOv8 for use in FastAPI async context.
    Lazy-loads the model on first use so server starts instantly.
    """

    def __init__(self, model_path: str = "best.pt"):
        self.model_path  = model_path
        self._model      = None
        self._lock       = threading.Lock()
        self._fps_buf:    deque = deque(maxlen=30)
        self._prev_time:  float = time.time()

    # ── Lazy model property ──────────────────────────────────────────────────
    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if not YOLO_AVAILABLE:
                        raise RuntimeError(
                            "ultralytics not installed — run: pip install ultralytics"
                        )
                    if not Path(self.model_path).is_file():
                        raise FileNotFoundError(
                            f"Model not found: {self.model_path}\n"
                            "Place your best.pt inside the backend/ folder."
                        )

                    logger.info(f"Loading YOLO model: {self.model_path}")

                    # ── PyTorch 2.6 fix ──────────────────────────────────────
                    import torch

                    _orig_load = torch.load

                    @functools.wraps(_orig_load)
                    def _patched_load(*args, **kwargs):
                        kwargs.setdefault("weights_only", False)
                        return _orig_load(*args, **kwargs)

                    torch.load = _patched_load
                    try:
                        self._model = YOLO(self.model_path)
                    finally:
                        torch.load = _orig_load  # always restore original

                    logger.info(
                        f"YOLO model loaded — {len(self._model.names)} classes: "
                        f"{list(self._model.names.values())}"
                    )
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Single frame analysis ────────────────────────────────────────────────
    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        cumulative_stats: Optional[dict] = None,
        total_violations: int = 0,
        camera_name: str = "",
        global_conf: Optional[float] = None,
    ) -> FrameResult:

        result                  = FrameResult()
        result.frame_number     = frame_number
        result.timestamp        = time.time()
        result.cumulative_stats = cumulative_stats or {}
        result.total_violations = total_violations

        # FPS tracking
        now = time.time()
        self._fps_buf.append(1.0 / max(now - self._prev_time, 1e-6))
        self._prev_time = now
        result.fps = sum(self._fps_buf) / max(len(self._fps_buf), 1)

        try:
            predictions = self.model(frame, verbose=False)

            annotated, cur_stats, alert_reasons = draw_detections(
                frame.copy(), predictions, global_conf
            )

            # Red border flash on violation frame
            if alert_reasons:
                h, w = annotated.shape[:2]
                cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 0, 200), 5)

            annotated = draw_hud(
                annotated,
                cumulative_stats or {},
                result.fps,
                total_violations,
                camera_name,
            )

            result.current_stats     = cur_stats
            result.annotated_jpg_b64 = encode_frame(annotated)

            # De-duplicate alerts per frame
            seen = set()
            for cls_name in alert_reasons:
                if cls_name not in seen:
                    seen.add(cls_name)
                    result.alerts.append({
                        "type":     cls_name,
                        "severity": SEVERITY_MAP.get(cls_name, "warning"),
                        "message":  ALERT_MESSAGES.get(cls_name, cls_name),
                    })

        except Exception as e:
            logger.error(f"Frame {frame_number} analysis error: {e}", exc_info=True)

        return result

    # ── Full video file analysis ─────────────────────────────────────────────
    def analyze_video(
        self,
        video_path: str,
        on_frame: Optional[Callable] = None,
        global_conf: Optional[float] = None,
    ) -> dict:
        """
        Analyze entire video file.
        Snapshots saved as clean frames (no bounding boxes).
        Calls on_frame(FrameResult) after each analyzed frame.
        Returns summary dict.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps_v        = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num    = 0
        analyzed     = 0
        total_viol   = 0
        cum_stats:  dict  = defaultdict(int)
        all_alerts: list  = []
        last_snap:  float = 0.0

        from config import settings as cfg
        skip = cfg.FRAME_SKIP

        logger.info(f"Analyzing: {video_path} | {total_frames} frames @ {fps_v:.1f}fps")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                if frame_num % skip != 0:
                    continue

                # Keep clean copy BEFORE analyze_frame draws on it
                clean_frame = frame.copy()

                result = self.analyze_frame(
                    frame, frame_num,
                    dict(cum_stats),
                    total_viol,
                    "Video",
                    global_conf,
                )
                analyzed += 1

                for cls, cnt in result.current_stats.items():
                    cum_stats[cls] += cnt

                if result.alerts:
                    total_viol += len(result.alerts)
                    all_alerts.extend(result.alerts)

                    # Save clean snapshot (no boxes) every 5 seconds
                    if time.time() - last_snap >= 5.0:
                        save_snapshot(clean_frame, result.alerts[0]["type"])
                        last_snap = time.time()

                result.total_violations = total_viol
                result.cumulative_stats  = dict(cum_stats)

                if on_frame:
                    on_frame(result)

        finally:
            cap.release()

        logger.info(f"Analysis complete: {analyzed} frames, {total_viol} violations")
        return {
            "total_frames_analyzed": analyzed,
            "total_violations":      total_viol,
            "cumulative_stats":      dict(cum_stats),
            "all_alerts":            all_alerts,
            "fps_video":             fps_v,
            "duration_seconds":      total_frames / fps_v if fps_v > 0 else 0,
        }

    # ── Open RTSP/IP camera stream ────────────────────────────────────────────
    def open_stream(self, stream_url: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # ── Simulation frame (when no camera / no model) ──────────────────────────
    @staticmethod
    def sim_frame(frame_num: int) -> np.ndarray:
        h, w  = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (12, 17, 24)
        noise = np.random.randint(0, 6, (h, w, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        for x in [60, 200, 340, 480]:
            cv2.rectangle(frame, (x, 280), (x + 110, 330), (28, 40, 55), -1)
            cv2.rectangle(frame, (x, 330), (x + 110, 420), (22, 33, 45), -1)
        cv2.putText(
            frame,
            f"SIMULATION — add best.pt to activate AI | Frame {frame_num}",
            (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 200, 100), 1,
        )
        return frame


# ── Singleton ─────────────────────────────────────────────────────────────────
_detector_instance: Optional[ExamDetector] = None


def get_detector(model_path: str = "best.pt") -> ExamDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ExamDetector(model_path)
    return _detector_instance
