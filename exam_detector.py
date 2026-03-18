"""
============================================================
  🎓 EXAM BEHAVIOR REAL-TIME DETECTION SYSTEM
  YOLOv8 | 7 Classes | Alarm + Screenshot + Violation Count
============================================================

USAGE:
  python exam_detector.py                        # webcam (default)
  python exam_detector.py --source video.mp4     # video file
  python exam_detector.py --source 0             # webcam index 0
  python exam_detector.py --model best.pt        # custom model path

REQUIREMENTS:
  pip install ultralytics opencv-python pygame numpy
"""

import cv2
import numpy as np
import argparse
import os
import time
import datetime
from collections import defaultdict, deque
from pathlib import Path

# ── Optional pygame for alarm sound ──────────────────────
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARN] pygame not found. Install with: pip install pygame")

# ── Try to import ultralytics ─────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not found. Install with: pip install ultralytics")

# ─────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Bend Over The Desk",
    "Hand Under Table",
    "Look Around",
    "Normal",
    "Stand Up",
    "Wave",
    "phone",
]

# Colors per class (BGR)
CLASS_COLORS = {
    "Bend Over The Desk": (255, 165,   0),   # Orange
    "Hand Under Table":   (  0, 255, 255),   # Cyan
    "Look Around":        (255, 255,   0),   # Yellow
    "Normal":             (  0, 255,   0),   # Green
    "Stand Up":           (255,   0, 255),   # Magenta
    "Wave":               (128,   0, 128),   # Purple
    "phone":              (  0,   0, 255),   # Red  ← HIGH RISK
}

# Classes that trigger alarm
ALERT_CLASSES = {"phone", "Hand Under Table", "Look Around", "Wave"}

# Confidence threshold
CONF_THRESHOLD = 0.45

# Alarm cooldown (seconds) — don't spam alarm
ALARM_COOLDOWN = 5

# Screenshot save folder
SCREENSHOT_DIR = "exam_violations"

# ─────────────────────────────────────────────────────────
#  ALARM MANAGER
# ─────────────────────────────────────────────────────────
class AlarmManager:
    def __init__(self):
        self.last_alarm_time = 0
        self._init_sound()

    def _init_sound(self):
        if PYGAME_AVAILABLE:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        self.beep_buf = self._make_beep()

    def _make_beep(self):
        """Generate a 440 Hz beep in memory."""
        sample_rate = 44100
        duration    = 0.4
        freq        = 880
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        return wave

    def trigger(self, reason: str):
        now = time.time()
        if now - self.last_alarm_time < ALARM_COOLDOWN:
            return
        self.last_alarm_time = now
        print(f"  🚨 ALARM  → {reason}")
        if PYGAME_AVAILABLE:
            try:
                sound = pygame.sndarray.make_sound(self.beep_buf)
                sound.play()
            except Exception:
                pass  # silent fallback

# ─────────────────────────────────────────────────────────
#  SCREENSHOT MANAGER
# ─────────────────────────────────────────────────────────
class ScreenshotManager:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.count = 0

    def save(self, frame: np.ndarray, reason: str) -> str:
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = reason.replace(" ", "_")[:20]
        fname = self.save_dir / f"violation_{ts}_{tag}_{self.count:04d}.jpg"
        cv2.imwrite(str(fname), frame)
        self.count += 1
        print(f"  📸 Saved  → {fname}")
        return str(fname)

# ─────────────────────────────────────────────────────────
#  HUD / OVERLAY DRAWING
# ─────────────────────────────────────────────────────────
def draw_hud(frame, stats: dict, fps: float, total_violations: int):
    h, w = frame.shape[:2]

    # ── Semi-transparent top bar ──────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, f"EXAM MONITOR  |  {ts}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}   Violations: {total_violations}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)

    # ── Right-side class counter panel ───────────────────
    panel_w  = 230
    panel_h  = len(CLASS_NAMES) * 28 + 30
    px, py   = w - panel_w - 8, 70
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (px-4, py-4), (px+panel_w, py+panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "DETECTED", (px+2, py+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    for i, cls in enumerate(CLASS_NAMES):
        cnt   = stats.get(cls, 0)
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        if cls in ALERT_CLASSES and cnt > 0:
            color = (0, 0, 255)
        y = py + 34 + i * 28
        cv2.putText(frame, f"{cls[:22]:<22}  {cnt}",
                    (px+2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)

    return frame


def draw_detections(frame, results):
    """Draw bounding boxes + labels on frame."""
    current_stats = defaultdict(int)
    alert_reasons = []

    for box in results[0].boxes:
        conf  = float(box.conf[0])
        cls_i = int(box.cls[0])
        if conf < CONF_THRESHOLD:
            continue

        cls_name = results[0].names.get(cls_i, CLASS_NAMES[cls_i] if cls_i < len(CLASS_NAMES) else "unknown")
        color    = CLASS_COLORS.get(cls_name, (200, 200, 200))
        current_stats[cls_name] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name}  {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if cls_name in ALERT_CLASSES:
            alert_reasons.append(cls_name)

    return frame, dict(current_stats), alert_reasons

# ─────────────────────────────────────────────────────────
#  MAIN DETECTION LOOP
# ─────────────────────────────────────────────────────────
def run(source=0, model_path="best.pt"):
    if not YOLO_AVAILABLE:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return

    print(f"\n{'='*55}")
    print("  🎓 EXAM BEHAVIOR DETECTION SYSTEM")
    print(f"{'='*55}")
    print(f"  Model  : {model_path}")
    print(f"  Source : {source}")
    print(f"  Saving : {SCREENSHOT_DIR}/")
    print(f"  Keys   : [S] Screenshot  [Q/ESC] Quit")
    print(f"{'='*55}\n")

    # Load model
    model = YOLO(model_path)
    print(f"  ✅ Model loaded — {len(model.names)} classes\n")

    alarm      = AlarmManager()
    screenshotter = ScreenshotManager(SCREENSHOT_DIR)

    # Open video/webcam
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        return

    # FPS tracking
    fps_buf    = deque(maxlen=30)
    prev_time  = time.time()

    # Cumulative stats
    total_violations  = 0
    cumulative_stats  = defaultdict(int)
    screenshot_cooldown = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended.")
            break

        # ── Inference ────────────────────────────────────
        results = model(frame, verbose=False)

        # ── Draw detections ───────────────────────────────
        frame, cur_stats, alerts = draw_detections(frame, results)

        # ── Update cumulative stats ───────────────────────
        for k, v in cur_stats.items():
            cumulative_stats[k] += v

        # ── FPS ───────────────────────────────────────────
        now = time.time()
        fps_buf.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = sum(fps_buf) / len(fps_buf)

        # ── Alarm + auto-screenshot on violations ─────────
        if alerts:
            reason = ", ".join(set(alerts))
            alarm.trigger(reason)
            total_violations += len(alerts)

            # Auto-screenshot (throttled)
            if time.time() > screenshot_cooldown:
                screenshotter.save(frame.copy(), alerts[0])
                screenshot_cooldown = time.time() + ALARM_COOLDOWN

            # Red border flash
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 6)

        # ── HUD overlay ───────────────────────────────────
        frame = draw_hud(frame, cur_stats, fps, total_violations)

        cv2.imshow("Exam Behavior Monitor", frame)

        # ── Key handling ──────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # Q or ESC
            break
        elif key == ord('s'):       # Manual screenshot
            screenshotter.save(frame.copy(), "manual")

    # ── Cleanup ───────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    # ── Final report ─────────────────────────────────────
    print(f"\n{'='*55}")
    print("  📊 SESSION SUMMARY")
    print(f"{'='*55}")
    print(f"  Total violation alerts  : {total_violations}")
    print(f"  Screenshots saved       : {screenshotter.count}  →  ./{SCREENSHOT_DIR}/")
    print()
    for cls, cnt in sorted(cumulative_stats.items(), key=lambda x: -x[1]):
        flag = " ⚠️" if cls in ALERT_CLASSES else ""
        print(f"    {cls:<25} {cnt:>5}{flag}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exam Behavior Real-Time Detector")
    parser.add_argument("--source", default="0",
                        help="Video source: 0=webcam, or path to video file")
    parser.add_argument("--model",  default="best.pt",
                        help="Path to YOLOv8 model weights (best.pt)")
    parser.add_argument("--conf",   default=0.45, type=float,
                        help="Confidence threshold (default 0.45)")
    args = parser.parse_args()

    CONF_THRESHOLD = args.conf
    run(source=args.source, model_path=args.model)
