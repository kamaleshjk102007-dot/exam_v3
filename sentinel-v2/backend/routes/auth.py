"""SentinelEye — Authentication (JWT + bcrypt)"""
import jwt
import bcrypt
import logging
from datetime import datetime, timedelta
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr

from config import settings
from database.connection import get_db

router = APIRouter()
logger = logging.getLogger("sentinel.auth")
oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    username:  str
    email:     EmailStr
    password:  str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email:    str
    password: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def _make_token(user_id: str) -> str:
    exp = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": user_id, "exp": exp},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired — please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")


def _serialize_user(u: dict) -> dict:
    return {
        "id":         str(u["_id"]),
        "username":   u["username"],
        "email":      u["email"],
        "full_name":  u.get("full_name"),
        "created_at": u["created_at"].isoformat(),
    }


# ── Dependency — get current logged-in user ────────────────────────────────────
async def get_current_user(token: str = Depends(oauth2)) -> dict:
    payload = _decode_token(token)
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(401, "Bad token payload")
    db   = get_db()
    user = await db.users.find_one({"_id": ObjectId(uid)})
    if not user:
        raise HTTPException(401, "User not found")
    return user


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/register", status_code=201)
async def register(data: UserCreate):
    db = get_db()
    if await db.users.find_one({"email": data.email}):
        raise HTTPException(400, "Email already registered")
    if await db.users.find_one({"username": data.username}):
        raise HTTPException(400, "Username already taken")
    doc = {
        "username":   data.username,
        "email":      data.email,
        "full_name":  data.full_name,
        "password":   _hash_password(data.password),
        "created_at": datetime.utcnow(),
        "is_active":  True,
    }
    res       = await db.users.insert_one(doc)
    doc["_id"] = res.inserted_id
    token      = _make_token(str(res.inserted_id))
    logger.info(f"Registered: {data.email}")
    return {"access_token": token, "token_type": "bearer", "user": _serialize_user(doc)}


@router.post("/login")
async def login(data: UserLogin):
    db   = get_db()
    user = await db.users.find_one({"email": data.email})
    if not user or not _verify_password(data.password, user["password"]):
        raise HTTPException(401, "Invalid email or password")
    token = _make_token(str(user["_id"]))
    logger.info(f"Login: {data.email}")
    return {"access_token": token, "token_type": "bearer", "user": _serialize_user(user)}


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return _serialize_user(current_user)


@router.post("/logout")
async def logout():
    # JWT is stateless — client discards the token
    return {"message": "Logged out successfully"}
