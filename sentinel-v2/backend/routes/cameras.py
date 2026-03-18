"""SentinelEye — Camera CRUD"""
from datetime import datetime
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from database.connection import get_db
from routes.auth import get_current_user

router = APIRouter()


class CameraIn(BaseModel):
    name:         str
    ip_address:   str
    location:     Optional[str] = None
    classroom_id: str
    stream_url:   Optional[str] = None


class CameraUpdate(BaseModel):
    name:       Optional[str] = None
    ip_address: Optional[str] = None
    location:   Optional[str] = None
    stream_url: Optional[str] = None
    status:     Optional[str] = None


def _serialize(doc: dict) -> dict:
    return {
        "id":           str(doc["_id"]),
        "name":         doc["name"],
        "ip_address":   doc["ip_address"],
        "location":     doc.get("location"),
        "classroom_id": doc["classroom_id"],
        "stream_url":   doc.get("stream_url"),
        "status":       doc.get("status", "online"),
        "created_at":   doc["created_at"].isoformat(),
    }


@router.get("/classroom/{cid}")
async def list_cameras(cid: str, u=Depends(get_current_user)):
    db   = get_db()
    docs = await db.cameras.find({"classroom_id": cid}).sort("name", 1).to_list(50)
    return [_serialize(d) for d in docs]


@router.post("/", status_code=201)
async def create_camera(data: CameraIn, u=Depends(get_current_user)):
    db  = get_db()
    doc = {
        "name":         data.name,
        "ip_address":   data.ip_address,
        "location":     data.location,
        "classroom_id": data.classroom_id,
        "stream_url":   data.stream_url or f"rtsp://{data.ip_address}:554/stream",
        "status":       "online",
        "created_at":   datetime.utcnow(),
    }
    res       = await db.cameras.insert_one(doc)
    doc["_id"] = res.inserted_id
    return _serialize(doc)


@router.get("/{cam_id}")
async def get_camera(cam_id: str, u=Depends(get_current_user)):
    db  = get_db()
    doc = await db.cameras.find_one({"_id": ObjectId(cam_id)})
    if not doc:
        raise HTTPException(404, "Camera not found")
    return _serialize(doc)


@router.patch("/{cam_id}")
async def update_camera(cam_id: str, data: CameraUpdate, u=Depends(get_current_user)):
    db     = get_db()
    update = {k: v for k, v in data.dict().items() if v is not None}
    # Auto-update stream URL if IP changed
    if "ip_address" in update and "stream_url" not in update:
        update["stream_url"] = f"rtsp://{update['ip_address']}:554/stream"
    if update:
        await db.cameras.update_one({"_id": ObjectId(cam_id)}, {"$set": update})
    doc = await db.cameras.find_one({"_id": ObjectId(cam_id)})
    return _serialize(doc)


@router.delete("/{cam_id}", status_code=204)
async def delete_camera(cam_id: str, u=Depends(get_current_user)):
    db = get_db()
    await db.cameras.delete_one({"_id": ObjectId(cam_id)})
