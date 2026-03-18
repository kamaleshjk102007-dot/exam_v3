"""SentinelEye — Alerts routes"""
from typing import Optional
from fastapi import APIRouter, Depends, Query
from database.connection import get_db
from routes.auth import get_current_user

router = APIRouter()

def _serialize(doc: dict) -> dict:
    return {
        "id":           str(doc["_id"]),
        "session_id":   doc["session_id"],
        "classroom_id": doc.get("classroom_id", ""),
        "camera_id":    doc.get("camera_id"),
        "alert_type":   doc["alert_type"],
        "severity":     doc["severity"],
        "message":      doc["message"],
        "frame_number": doc.get("frame_number"),
        "snapshot":     doc.get("snapshot"),
        "created_at":   doc["created_at"].isoformat(),
    }

@router.get("/session/{sid}")
async def session_alerts(
    sid: str,
    severity: Optional[str] = None,
    limit: int = Query(200, le=500),
    u=Depends(get_current_user),
):
    db = get_db()
    q = {"session_id": sid}
    if severity:
        q["severity"] = severity
    docs = await db.alerts.find(q).sort("created_at", -1).limit(limit).to_list(limit)
    return [_serialize(d) for d in docs]

@router.get("/classroom/{cid}")
async def classroom_alerts(
    cid: str,
    limit: int = Query(50, le=200),
    u=Depends(get_current_user),
):
    db = get_db()
    docs = await db.alerts.find(
        {"classroom_id": cid}
    ).sort("created_at", -1).limit(limit).to_list(limit)
    return [_serialize(d) for d in docs]

@router.get("/stats/{sid}")
async def alert_stats(sid: str, u=Depends(get_current_user)):
    db = get_db()
    total = await db.alerts.count_documents({"session_id": sid})
    pipeline = [
        {"$match": {"session_id": sid}},
        {"$group": {
            "_id":      "$alert_type",
            "count":    {"$sum": 1},
            "critical": {"$sum": {"$cond": [{"$eq": ["$severity", "critical"]}, 1, 0]}},
        }},
    ]
    by_type = await db.alerts.aggregate(pipeline).to_list(20)
    return {
        "total":   total,
        "by_type": {
            r["_id"]: {"count": r["count"], "critical": r["critical"]}
            for r in by_type
        },
    }

@router.delete("/session/{sid}", status_code=204)
async def clear_session_alerts(sid: str, u=Depends(get_current_user)):
    db = get_db()
    await db.alerts.delete_many({"session_id": sid})

@router.delete("/classroom/{cid}", status_code=204)
async def clear_classroom_alerts(cid: str, u=Depends(get_current_user)):
    db = get_db()
    await db.alerts.delete_many({"classroom_id": cid})