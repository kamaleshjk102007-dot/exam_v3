"""SentinelEye — Configuration (reads from .env)"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    APP_NAME: str = "SentinelEye"
    DEBUG: bool = False
    SECRET_KEY: str = "sentinel-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "sentinel_eye"

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 500
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv"]

    YOLO_MODEL: str = "best.pt"
    YOLO_CONF: float = 0.40
    FRAME_SKIP: int = 2

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure all needed directories exist on startup
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs("static",    exist_ok=True)
os.makedirs("snapshots", exist_ok=True)
