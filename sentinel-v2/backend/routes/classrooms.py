"""SentinelEye — Classroom CRUD"""
from datetime import datetime
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from database.connection import get_db
from routes.auth import get_current_user

router = APIRouter()


class ClassroomIn(BaseModel):
    name:        str
    description: Optional[str] = None
    capacity:    Optional[int] = None


def _serialize(doc: dict, cam_count: int = 0) -> dict:
    return {
        "id":             str(doc["_id"]),
        "name":           doc["name"],
        "description":    doc.get("description"),
        "capacity":       doc.get("capacity"),
        "camera_count":   cam_count,
        "active_session": doc.get("active_session"),
        "status":         doc.get("status", "idle"),
        "created_at":     doc["created_at"].isoformat(),
    }


@router.get("/")
async def list_classrooms(u=Depends(get_current_user)):
    db   = get_db()
    docs = await db.classrooms.find(
        {"created_by": u["_id"]}
    ).sort("name", 1).to_list(200)
    result = []
    for doc in docs:
        cnt = await db.cameras.count_documents({"classroom_id": str(doc["_id"])})
        result.append(_serialize(doc, cnt))
    return result


@router.post("/", status_code=201)
async def create_classroom(data: ClassroomIn, u=Depends(get_current_user)):
    db  = get_db()
    doc = {
        "name":           data.name,
        "description":    data.description,
        "capacity":       data.capacity,
        "created_by":     u["_id"],
        "created_at":     datetime.utcnow(),
        "active_session": None,
        "status":         "idle",
    }
    res = await db.classrooms.insert_one(doc)
    cid = str(res.inserted_id)

    # Auto-create 2 default cameras for every classroom
    for name, ip, loc in [
        ("Front Camera", "192.168.1.101", "Front wall"),
        ("Back Camera",  "192.168.1.102", "Rear corner"),
    ]:
        await db.cameras.insert_one({
            "name":         name,
            "ip_address":   ip,
            "location":     loc,
            "classroom_id": cid,
            "stream_url":   f"rtsp://{ip}:554/stream",
            "status":       "online",
            "created_at":   datetime.utcnow(),
        })

    doc["_id"] = res.inserted_id
    return _serialize(doc, 2)


@router.get("/{cid}")
async def get_classroom(cid: str, u=Depends(get_current_user)):
    db  = get_db()
    doc = await db.classrooms.find_one({"_id": ObjectId(cid)})
    if not doc:
        raise HTTPException(404, "Classroom not found")
    cnt = await db.cameras.count_documents({"classroom_id": cid})
    return _serialize(doc, cnt)


@router.patch("/{cid}")
async def update_classroom(cid: str, data: ClassroomIn, u=Depends(get_current_user)):
    db     = get_db()
    update = {k: v for k, v in data.dict().items() if v is not None}
    if update:
        await db.classrooms.update_one({"_id": ObjectId(cid)}, {"$set": update})
    doc = await db.classrooms.find_one({"_id": ObjectId(cid)})
    cnt = await db.cameras.count_documents({"classroom_id": cid})
    return _serialize(doc, cnt)


@router.delete("/{cid}", status_code=204)
async def delete_classroom(cid: str, u=Depends(get_current_user)):
    db = get_db()
    await db.classrooms.delete_one({"_id": ObjectId(cid)})
    await db.cameras.delete_many({"classroom_id": cid})
