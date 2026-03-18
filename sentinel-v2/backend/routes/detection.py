"""
SentinelEye — Detection Routes

Endpoints:
  POST /upload                  — upload video file
  GET  /status/{session_id}     — poll analysis progress
  GET  /summary/{session_id}    — completed session summary
  GET  /snapshots               — list violation snapshots
  GET  /snapshot/{filename}     — serve snapshot (NO auth — opens in browser)
  WS   /ws/video/{video_id}     — stream video file frame-by-frame
  WS   /ws/{session_id}         — real-time RTSP camera stream

Behaviour:
  - Per-type alert cooldown (10s) — no duplicate alerts in browser
  - Snapshots saved WITH bounding boxes (annotated frame from YOLO output)
  - Per-type snapshot cooldown (10s)
  - Snapshot GET endpoint has no auth — opens directly in browser/new tab
  - DB records every violation for accurate session totals
"""

import asyncio
import json
import os
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np
from bson import ObjectId
from fastapi import (
    APIRouter, Depends, File, Form, HTTPException,
    Query, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import FileResponse

from config import settings
from database.connection import get_db
from detector import get_detector, SNAPSHOT_DIR
from routes.auth import _decode_token, get_current_user

router = APIRouter()
logger = logging.getLogger("sentinel.detection")
_pool  = ThreadPoolExecutor(max_workers=3)
_jobs: dict = {}

# ── Cooldown durations ────────────────────────────────────────────────────────
ALERT_COOLDOWN_SEC = 10   # min seconds between same-type alerts in browser
SNAP_COOLDOWN_SEC  = 10   # min seconds between same-type snapshots on disk


# ── Save annotated snapshot WITH bounding boxes ───────────────────────────────
def save_annotated_snapshot(jpg_b64: str, reason: str) -> str:
    """
    Decode the annotated JPEG already produced by YOLO (has boxes + HUD)
    and save it to the snapshots folder.
    """
    import base64 as _b64
    import datetime as _dt
    ts    = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tag   = reason.replace(" ", "_")[:20]
    fname = SNAPSHOT_DIR / f"violation_{ts}_{tag}.jpg"
    try:
        jpg_bytes = _b64.b64decode(jpg_b64)
        arr       = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            cv2.imwrite(str(fname), frame)
            logger.info(f"Snapshot saved (annotated): {fname}")
    except Exception as e:
        logger.error(f"Snapshot save error: {e}")
    return str(fname)


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
@router.post("/upload")
async def upload_video(
    file:         UploadFile = File(...),
    classroom_id: str        = Form(...),
    current_user: dict       = Depends(get_current_user),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {ext}. Use MP4, AVI or MOV.")

    video_id = str(uuid.uuid4())
    filename = f"{video_id}{ext}"
    dest     = os.path.join(settings.UPLOAD_DIR, filename)

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB)")

    with open(dest, "wb") as f:
        f.write(content)

    cap      = cv2.VideoCapture(dest)
    fps_v    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = n_frames / fps_v if fps_v > 0 else 0

    db = get_db()
    await db.videos.insert_one({
        "_id":           video_id,
        "filename":      filename,
        "original_name": file.filename,
        "path":          dest,
        "classroom_id":  classroom_id,
        "size_mb":       round(size_mb, 2),
        "fps":           fps_v,
        "total_frames":  n_frames,
        "duration":      round(duration, 2),
        "uploaded_by":   current_user["_id"],
        "uploaded_at":   datetime.utcnow(),
    })

    logger.info(f"Uploaded: {file.filename} ({size_mb:.1f} MB, {n_frames} frames)")
    return {
        "video_id":         video_id,
        "filename":         file.filename,
        "size_mb":          round(size_mb, 2),
        "fps":              round(fps_v, 2),
        "total_frames":     n_frames,
        "duration_seconds": round(duration, 2),
        "message":          "Upload successful — ready to analyze",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STATUS & SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
@router.get("/status/{session_id}")
async def analysis_status(
    session_id:   str,
    current_user: dict = Depends(get_current_user),
):
    job = _jobs.get(session_id)
    if job:
        return job
    db = get_db()
    s  = await db.sessions.find_one({"_id": ObjectId(session_id)})
    if not s:
        raise HTTPException(404, "Session not found")
    return {
        "status":     s.get("status"),
        "progress":   100,
        "violations": s.get("total_violations", 0),
    }


@router.get("/summary/{session_id}")
async def get_summary(
    session_id:   str,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    s  = await db.sessions.find_one({"_id": ObjectId(session_id)})
    if not s:
        raise HTTPException(404, "Session not found")

    total  = await db.alerts.count_documents({"session_id": session_id})
    crit   = await db.alerts.count_documents({"session_id": session_id, "severity": "critical"})
    warn   = await db.alerts.count_documents({"session_id": session_id, "severity": "warning"})
    phones = await db.alerts.count_documents({"session_id": session_id, "alert_type": "phone"})
    look   = await db.alerts.count_documents({"session_id": session_id, "alert_type": "Look Around"})
    hand   = await db.alerts.count_documents({"session_id": session_id, "alert_type": "Hand Under Table"})
    wave   = await db.alerts.count_documents({"session_id": session_id, "alert_type": "Wave"})

    ended = s.get("ended_at") or datetime.utcnow()
    dur   = (ended - s["started_at"]).total_seconds()

    return {
        "session_id":            session_id,
        "classroom_id":          s["classroom_id"],
        "status":                s.get("status"),
        "total_frames_analyzed": s.get("total_frames", 0),
        "total_violations":      total,
        "critical":              crit,
        "warning":               warn,
        "by_type": {
            "phone":            phones,
            "Look Around":      look,
            "Hand Under Table": hand,
            "Wave":             wave,
        },
        "duration_seconds":  round(dur, 1),
        "started_at":        s["started_at"].isoformat(),
        "completed_at":      ended.isoformat() if s.get("ended_at") else None,
        "cumulative_stats":  s.get("cumulative_stats", {}),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SNAPSHOTS
# ══════════════════════════════════════════════════════════════════════════════
@router.get("/snapshots")
async def list_snapshots(current_user: dict = Depends(get_current_user)):
    snap_dir = Path("snapshots")
    files    = sorted(snap_dir.glob("*.jpg"), key=os.path.getmtime, reverse=True)
    return [
        {
            "filename": f.name,
            "url":      f"/api/detection/snapshot/{f.name}",
            "size_kb":  round(f.stat().st_size / 1024, 1),
        }
        for f in files[:50]
    ]


@router.get("/snapshot/{filename}")
async def get_snapshot(filename: str):
    # NO auth required — browser can open this URL directly in a new tab
    path = Path("snapshots") / filename
    if not path.exists():
        raise HTTPException(404, "Snapshot not found")
    return FileResponse(str(path), media_type="image/jpeg")


# ══════════════════════════════════════════════════════════════════════════════
#  WebSocket — VIDEO FILE  (streams annotated frames live)
# ══════════════════════════════════════════════════════════════════════════════
@router.websocket("/ws/video/{video_id}")
async def ws_video(
    websocket:    WebSocket,
    video_id:     str,
    token:        Optional[str] = Query(None),
    classroom_id: Optional[str] = Query(None),
):
    try:
        if token:
            _decode_token(token)
    except Exception:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    logger.info(f"WS video: video_id={video_id}")

    db    = get_db()
    video = await db.videos.find_one({"_id": video_id})
    if not video:
        await _send(websocket, "error", {"message": "Video not found"})
        await websocket.close()
        return

    session_doc = {
        "classroom_id":     classroom_id or video.get("classroom_id", ""),
        "video_id":         video_id,
        "mode":             "video",
        "status":           "running",
        "started_at":       datetime.utcnow(),
        "ended_at":         None,
        "total_alerts":     0,
        "total_frames":     0,
        "total_violations": 0,
    }
    res        = await db.sessions.insert_one(session_doc)
    session_id = str(res.inserted_id)

    await _send(websocket, "status", {
        "status":       "connected",
        "session_id":   session_id,
        "total_frames": video.get("total_frames", 0),
        "fps":          video.get("fps", 25.0),
        "message":      f"Analyzing: {video['original_name']}",
    })

    detector          = get_detector(settings.YOLO_MODEL)
    cumulative_stats: dict = {}
    total_violations: int  = 0
    frame_num:        int  = 0
    analyzed:         int  = 0
    alert_docs:       list = []
    total_frames            = video.get("total_frames", 0)
    loop                    = asyncio.get_event_loop()

    alert_last_sent: Dict[str, float] = {}
    snap_last_saved: Dict[str, float] = {}

    cap = cv2.VideoCapture(video["path"])
    if not cap.isOpened():
        await _send(websocket, "error", {"message": "Cannot open video file"})
        await websocket.close()
        return

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                cmd = json.loads(raw)
                if cmd.get("action") == "stop":
                    break
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

            ret, frame = cap.read()
            if not ret:
                await _send(websocket, "status", {
                    "status":           "completed",
                    "session_id":       session_id,
                    "total_violations": total_violations,
                    "total_frames":     analyzed,
                    "cumulative_stats": cumulative_stats,
                    "message":          "Analysis complete",
                })
                break

            frame_num += 1
            if frame_num % settings.FRAME_SKIP != 0:
                continue

            analyzed += 1
            progress = int((frame_num / max(total_frames, 1)) * 100)

            result = await loop.run_in_executor(
                _pool,
                lambda f=frame: detector.analyze_frame(
                    f, frame_num, dict(cumulative_stats),
                    total_violations, "Video Analysis", settings.YOLO_CONF,
                ),
            )

            for cls, cnt in result.current_stats.items():
                cumulative_stats[cls] = cumulative_stats.get(cls, 0) + cnt

            now = time.time()
            alerts_for_browser: list = []

            if result.alerts:
                total_violations += len(result.alerts)

                for a in result.alerts:
                    atype = a["type"]

                    # Always record in DB
                    alert_docs.append({
                        "session_id":   session_id,
                        "classroom_id": classroom_id or video.get("classroom_id", ""),
                        "camera_id":    None,
                        "alert_type":   atype,
                        "severity":     a["severity"],
                        "message":      a["message"],
                        "frame_number": frame_num,
                        "timestamp":    result.timestamp,
                        "created_at":   datetime.utcnow(),
                    })

                    # Browser — cooldown per type
                    if now - alert_last_sent.get(atype, 0) >= ALERT_COOLDOWN_SEC:
                        alert_last_sent[atype] = now
                        alerts_for_browser.append(a)

                    # Snapshot WITH bounding boxes — cooldown per type
                    if now - snap_last_saved.get(atype, 0) >= SNAP_COOLDOWN_SEC:
                        snap_last_saved[atype] = now
                        save_annotated_snapshot(result.annotated_jpg_b64, atype)

                if len(alert_docs) >= 10:
                    await db.alerts.insert_many(alert_docs)
                    alert_docs = []

            await _send(websocket, "frame_result", {
                "frame_number":     frame_num,
                "fps":              round(result.fps, 1),
                "progress":         progress,
                "current_stats":    result.current_stats,
                "cumulative_stats": cumulative_stats,
                "alerts":           alerts_for_browser,
                "total_violations": total_violations,
                "frame_base64":     result.annotated_jpg_b64,
            })

            await asyncio.sleep(0.04)

    except WebSocketDisconnect:
        logger.info(f"WS video disconnected: {video_id}")
    except Exception as e:
        logger.error(f"WS video error: {e}", exc_info=True)
        await _send(websocket, "error", {"message": str(e)})
    finally:
        cap.release()
        if alert_docs:
            await db.alerts.insert_many(alert_docs)
        await db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {
                "status":           "completed",
                "ended_at":         datetime.utcnow(),
                "total_alerts":     total_violations,
                "total_violations": total_violations,
                "total_frames":     analyzed,
                "cumulative_stats": cumulative_stats,
            }},
        )
        logger.info(f"WS video done: {video_id} — {total_violations} violations in {analyzed} frames")


# ══════════════════════════════════════════════════════════════════════════════
#  WebSocket — REAL-TIME camera stream
# ══════════════════════════════════════════════════════════════════════════════
@router.websocket("/ws/{session_id}")
async def ws_realtime(
    websocket:    WebSocket,
    session_id:   str,
    token:        Optional[str] = Query(None),
    camera_id:    Optional[str] = Query(None),
    stream_url:   Optional[str] = Query(None),
    classroom_id: Optional[str] = Query(None),
    camera_name:  Optional[str] = Query(None),
):
    try:
        if token:
            _decode_token(token)
    except Exception:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    logger.info(f"WS realtime: session={session_id}")

    db = get_db()
    await db.sessions.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {
            "status":       "running",
            "started_at":   datetime.utcnow(),
            "classroom_id": classroom_id or "",
            "mode":         "realtime",
        }},
        upsert=True,
    )

    detector          = get_detector(settings.YOLO_MODEL)
    cumulative_stats: dict = {}
    total_violations: int  = 0
    frame_num:        int  = 0
    cap                    = None

    alert_last_sent: Dict[str, float] = {}
    snap_last_saved: Dict[str, float] = {}

    try:
        if stream_url:
            cap = detector.open_stream(stream_url)
            if not cap.isOpened():
                cap = None
                await _send(websocket, "status", {
                    "status":  "sim_mode",
                    "message": "Cannot open camera stream — simulation mode",
                })

        await _send(websocket, "status", {
            "status":     "connected",
            "session_id": session_id,
        })

        loop = asyncio.get_event_loop()

        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.04)
                cmd = json.loads(raw)
                if cmd.get("action") == "stop":
                    break
                if cmd.get("action") == "switch_camera":
                    if cap:
                        cap.release()
                    new_url = cmd.get("stream_url")
                    cap = detector.open_stream(new_url) if new_url else None
                    camera_name = cmd.get("camera_name", camera_name)
                    alert_last_sent.clear()
                    snap_last_saved.clear()
                    await _send(websocket, "status", {"status": "switched"})
                    continue
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

            frame_num += 1

            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    await _send(websocket, "status", {"status": "stream_lost"})
                    await asyncio.sleep(1)
                    continue
            else:
                frame = detector.sim_frame(frame_num)

            if frame_num % settings.FRAME_SKIP != 0:
                await asyncio.sleep(0.01)
                continue

            result = await loop.run_in_executor(
                _pool,
                lambda f=frame: detector.analyze_frame(
                    f, frame_num, dict(cumulative_stats),
                    total_violations, camera_name or "Camera", settings.YOLO_CONF,
                ),
            )

            for cls, cnt in result.current_stats.items():
                cumulative_stats[cls] = cumulative_stats.get(cls, 0) + cnt

            now = time.time()
            alerts_for_browser: list = []

            if result.alerts:
                total_violations += len(result.alerts)
                alert_docs = []

                for a in result.alerts:
                    atype = a["type"]

                    # Always record in DB
                    alert_docs.append({
                        "session_id":   session_id,
                        "classroom_id": classroom_id or "",
                        "camera_id":    camera_id,
                        "alert_type":   atype,
                        "severity":     a["severity"],
                        "message":      a["message"],
                        "frame_number": frame_num,
                        "timestamp":    result.timestamp,
                        "created_at":   datetime.utcnow(),
                    })

                    # Browser — cooldown per type
                    if now - alert_last_sent.get(atype, 0) >= ALERT_COOLDOWN_SEC:
                        alert_last_sent[atype] = now
                        alerts_for_browser.append(a)

                    # Snapshot WITH bounding boxes — cooldown per type
                    if now - snap_last_saved.get(atype, 0) >= SNAP_COOLDOWN_SEC:
                        snap_last_saved[atype] = now
                        save_annotated_snapshot(result.annotated_jpg_b64, atype)

                if alert_docs:
                    await db.alerts.insert_many(alert_docs)

            await _send(websocket, "frame_result", {
                "frame_number":     frame_num,
                "fps":              round(result.fps, 1),
                "current_stats":    result.current_stats,
                "cumulative_stats": cumulative_stats,
                "alerts":           alerts_for_browser,
                "total_violations": total_violations,
                "frame_base64":     result.annotated_jpg_b64,
            })

            await asyncio.sleep(0.02)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WS realtime error: {e}", exc_info=True)
    finally:
        if cap:
            cap.release()
        await db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {
                "status":           "completed",
                "ended_at":         datetime.utcnow(),
                "total_violations": total_violations,
                "cumulative_stats": cumulative_stats,
            }},
        )
        logger.info(f"WS realtime ended: {session_id} — {total_violations} violations")


# ── Shared send helper ────────────────────────────────────────────────────────
async def _send(ws: WebSocket, msg_type: str, data: dict):
    try:
        await ws.send_text(json.dumps({"type": msg_type, "data": data}))
    except Exception:
        pass
