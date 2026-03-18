"""
SentinelEye — FastAPI Application Entry Point
Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from database.connection import connect_db, disconnect_db
from routes.alerts import router as alerts_router
from routes.auth import router as auth_router
from routes.cameras import router as cameras_router
from routes.classrooms import router as classrooms_router
from routes.detection import router as detection_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sentinel")


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("  SentinelEye starting up")
    logger.info("=" * 50)
    try:
        await connect_db()
    except Exception as e:
        logger.warning(f"MongoDB not available: {e}")
        logger.warning("Running without database — some features disabled")
    yield
    logger.info("SentinelEye shutting down")
    await disconnect_db()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "SentinelEye API",
    description = "AI-powered exam cheating detection system",
    version     = "2.4.0",
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc",
    lifespan    = lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── API routers ───────────────────────────────────────────────────────────────
app.include_router(auth_router,        prefix="/api/auth",       tags=["Auth"])
app.include_router(classrooms_router,  prefix="/api/classrooms", tags=["Classrooms"])
app.include_router(cameras_router,     prefix="/api/cameras",    tags=["Cameras"])
app.include_router(detection_router,   prefix="/api/detection",  tags=["Detection"])
app.include_router(alerts_router,      prefix="/api/alerts",     tags=["Alerts"])

# ── Static file serving ───────────────────────────────────────────────────────
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")
app.mount("/uploads",   StaticFiles(directory="uploads"),   name="uploads")
app.mount("/static",    StaticFiles(directory="static"),    name="static")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    from pathlib import Path
    model_ready = Path(settings.YOLO_MODEL).is_file()
    return {
        "status":      "ok",
        "version":     "2.4.0",
        "model":       settings.YOLO_MODEL,
        "model_ready": model_ready,
        "frame_skip":  settings.FRAME_SKIP,
        "yolo_conf":   settings.YOLO_CONF,
    }


# ── Serve frontend SPA (catch-all) ────────────────────────────────────────────
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str):
    return FileResponse("static/index.html")
