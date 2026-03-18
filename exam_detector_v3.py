"""
============================================================
  EXAM BEHAVIOR REAL-TIME DETECTION SYSTEM
  YOLOv8 | 7 Classes | Alarm + Screenshot + Violation Count
============================================================

IGNORED CLASSES (never drawn, counted or alarmed):
  - Stand Up  (false positives on leaning students)
  - Normal    (not a violation — no need to show)

ACTIVE ALERT CLASSES:
  - phone
  - Hand Under Table
  - Look Around
  - Wave

USAGE:
  python exam_detector_v3.py --source cctv.mp4
  python exam_detector_v3.py --source 0
  python exam_detector_v3.py --model best.pt
  python exam_detector_v3.py --conf 0.50

REQUIREMENTS:
  pip install ultralytics opencv-python pygame numpy
"""

import cv2
import sys
import argparse
import threading
import time
import datetime
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARN] pygame not found — audio alarms disabled. pip install pygame")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not found — pip install ultralytics")


# =============================================================================
#  CONFIGURATION
# =============================================================================

CLASS_NAMES: list = [
    "Bend Over The Desk",
    "Hand Under Table",
    "Look Around",
    "Normal",
    "Stand Up",
    "Wave",
    "phone",
]

# Add any class here to fully disable it — no box, no alarm, no count
IGNORED_CLASSES: set = {
    "Stand Up",   # false positives on leaning students
    "Normal",     # not a violation — no need to detect or display
}

# Classes that trigger alarm + red border + screenshot
ALERT_CLASSES: set = {
    "phone",
    "Hand Under Table",
    "Look Around",
    "Wave",
}

CLASS_COLORS: dict = {
    "Bend Over The Desk": (255, 165,   0),
    "Hand Under Table":   (  0, 255, 255),
    "Look Around":        (255, 255,   0),
    "Wave":               (128,   0, 128),
    "phone":              (  0,   0, 255),
}

CLASS_CONF: dict = {
    "phone":              0.70,
    "Look Around":        0.45,
    "Hand Under Table":   0.55,
    "Wave":               0.60,
    "Bend Over The Desk": 0.40,
}
DEFAULT_CONF: float = 0.45

MIN_BOX_RATIO: dict = {
    "phone":              0.005,
    "Wave":               0.020,
    "Look Around":        0.008,
    "Hand Under Table":   0.015,
    "Bend Over The Desk": 0.020,
}

MAX_BOX_RATIO: dict = {
    "phone": 0.12,
}

ALARM_COOLDOWN:      float = 5.0
SCREENSHOT_COOLDOWN: float = 5.0
SCREENSHOT_DIR:      str   = "exam_violations"


# =============================================================================
#  ALARM MANAGER
# =============================================================================

class AlarmManager:

    def __init__(self) -> None:
        self.last_alarm_time: float = 0.0
        self._sound = None
        self._init_sound()

    def _init_sound(self) -> None:
        if not PYGAME_AVAILABLE:
            return
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._sound = self._make_beep()
        except Exception as exc:
            print(f"[WARN] AlarmManager._init_sound failed: {exc}")

    def _make_beep(self):
        t    = np.linspace(0, 0.4, int(44100 * 0.4), endpoint=False)
        mono = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(np.column_stack([mono, mono]))

    def trigger(self, reason: str) -> None:
        now = time.time()
        if now - self.last_alarm_time < ALARM_COOLDOWN:
            return
        self.last_alarm_time = now
        print(f"  [ALARM] {reason}")
        if self._sound is not None:
            threading.Thread(target=self._sound.play, daemon=True).start()


# =============================================================================
#  SCREENSHOT MANAGER
# =============================================================================

class ScreenshotManager:
    def __init__(self, save_dir: str) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.count: int = 0

    def save(self, frame: np.ndarray, reason: str) -> str:
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag   = reason.replace(" ", "_")[:20]
        fname = self.save_dir / f"violation_{ts}_{tag}_{self.count:04d}.jpg"
        cv2.imwrite(str(fname), frame)
        self.count += 1
        print(f"  [SCREENSHOT] {fname}")
        return str(fname)


# =============================================================================
#  HUD / OVERLAY DRAWING
# =============================================================================

def draw_hud(
    frame: np.ndarray,
    cumulative_stats: dict,
    fps: float,
    total_violations: int,
) -> np.ndarray:
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, f"EXAM MONITOR  |  {ts}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:5.1f}   Violations: {total_violations}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)

    # Only show non-ignored classes in panel
    display_classes = [c for c in CLASS_NAMES if c not in IGNORED_CLASSES]
    panel_w = 240
    panel_h = len(display_classes) * 28 + 34
    px, py  = w - panel_w - 8, 70
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (px - 4, py - 4), (px + panel_w, py + panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "CUMULATIVE DETECTIONS", (px + 2, py + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

    for i, cls in enumerate(display_classes):
        cnt   = cumulative_stats.get(cls, 0)
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        if cls in ALERT_CLASSES and cnt > 0:
            color = (0, 0, 255)
        y = py + 34 + i * 28
        cv2.putText(frame, f"{cls[:24]:<24}  {cnt:>4}",
                    (px + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    return frame


# =============================================================================
#  DETECTION DRAWING
# =============================================================================

def _safe_class_name(results, cls_i: int) -> str:
    name = results[0].names.get(cls_i)
    if name is not None:
        return name
    if 0 <= cls_i < len(CLASS_NAMES):
        return CLASS_NAMES[cls_i]
    return f"class_{cls_i}"


def draw_detections(
    frame: np.ndarray,
    results,
    global_conf_override=None,
) -> tuple:

    h_frame, w_frame = frame.shape[:2]
    frame_area = max(h_frame * w_frame, 1)

    current_stats: dict = defaultdict(int)
    alert_reasons: list = []
    validated:     list = []

    for box in results[0].boxes:
        conf     = float(box.conf[0])
        cls_i    = int(box.cls[0])
        cls_name = _safe_class_name(results, cls_i)

        # Skip ignored classes entirely — no box, no count, no alarm
        if cls_name in IGNORED_CLASSES:
            continue

        threshold = (global_conf_override
                     if global_conf_override is not None
                     else CLASS_CONF.get(cls_name, DEFAULT_CONF))
        if conf < threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            continue

        box_area_ratio = (box_w * box_h) / frame_area

        if box_area_ratio < MIN_BOX_RATIO.get(cls_name, 0.010):
            continue
        if box_area_ratio > MAX_BOX_RATIO.get(cls_name, 1.0):
            continue

        if cls_name == "phone":
            if (box_h / box_w) < 0.8:
                continue

        validated.append((cls_name, conf, x1, y1, x2, y2))

    for cls_name, conf, x1, y1, x2, y2 in validated:
        current_stats[cls_name] += 1
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name}  {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, lh + 6)
        cv2.rectangle(frame, (x1, label_y - lh - 6), (x1 + lw + 4, label_y), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if cls_name in ALERT_CLASSES:
            alert_reasons.append(cls_name)

    return frame, dict(current_stats), alert_reasons


# =============================================================================
#  MAIN DETECTION LOOP
# =============================================================================

def run(source="0", model_path: str = "best.pt", global_conf=None) -> None:

    if not YOLO_AVAILABLE:
        print("ERROR: ultralytics not installed. pip install ultralytics")
        sys.exit(1)

    if not Path(model_path).is_file():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    print(f"\n{'='*57}")
    print("  EXAM BEHAVIOR DETECTION SYSTEM")
    print(f"{'='*57}")
    print(f"  Model    : {model_path}")
    print(f"  Source   : {source}")
    print(f"  Saving   : {SCREENSHOT_DIR}/")
    print(f"  Ignored  : {', '.join(sorted(IGNORED_CLASSES))}")
    print(f"  Alerts   : {', '.join(sorted(ALERT_CLASSES))}")
    print(f"  Keys     : [S] Screenshot   [Q / ESC] Quit")
    print(f"{'='*57}\n")

    model = YOLO(model_path)
    print(f"  Model loaded — {len(model.names)} classes\n")

    alarm         = AlarmManager()
    screenshotter = ScreenshotManager(SCREENSHOT_DIR)

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        sys.exit(1)

    fps_buf:   deque = deque(maxlen=30)
    prev_time: float = time.time()

    total_violations:     int   = 0
    cumulative_stats:     dict  = defaultdict(int)
    last_screenshot_time: float = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or frame read error.")
                break

            results = model(frame, verbose=False)
            frame, cur_stats, alerts = draw_detections(frame, results, global_conf)

            for cls, cnt in cur_stats.items():
                cumulative_stats[cls] += cnt

            now = time.time()
            fps_buf.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            fps = sum(fps_buf) / len(fps_buf)

            if alerts:
                unique_alerts = set(alerts)
                alarm.trigger(", ".join(unique_alerts))
                total_violations += len(unique_alerts)

                if now - last_screenshot_time >= SCREENSHOT_COOLDOWN:
                    screenshotter.save(frame.copy(), next(iter(unique_alerts)))
                    last_screenshot_time = now

                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

            frame = draw_hud(frame, cumulative_stats, fps, total_violations)
            cv2.imshow("Exam Behavior Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                screenshotter.save(frame.copy(), "manual")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n{'='*57}")
    print("  SESSION SUMMARY")
    print(f"{'='*57}")
    print(f"  Screenshots saved : {screenshotter.count}  ->  ./{SCREENSHOT_DIR}/")
    print(f"{'='*57}\n")


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exam Behavior Real-Time Detector")
    parser.add_argument("--source", default="0",
                        help="0=webcam, or path to video file")
    parser.add_argument("--model",  default="best.pt",
                        help="Path to YOLOv8 weights (default: best.pt)")
    parser.add_argument("--conf",   type=float, default=None,
                        help="Global confidence override e.g. --conf 0.50")
    args = parser.parse_args()
    run(source=args.source, model_path=args.model, global_conf=args.conf)