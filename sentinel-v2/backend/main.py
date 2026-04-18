"""
SentinelEye — FastAPI Application Entry Point
Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Routes:
  /          → Landing page (exam_detector_web.html)
  /app       → SentinelEye app (login / dashboard)
  /api/...   → API endpoints
  /api/docs  → Swagger UI
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from database.connection import connect_db, disconnect_db, is_db_connected
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

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
FRONTEND_DIR = BASE_DIR.parent / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"
SNAPSHOTS_DIR = BASE_DIR / "snapshots"
APP_PAGE = STATIC_DIR / "index.html"
LANDING_PAGE = STATIC_DIR / "landing.html"
FRONTEND_LANDING_PAGE = FRONTEND_DIR / "index.html"
ROOT_ASSET_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif", ".ico",
    ".css", ".js", ".map", ".json", ".woff", ".woff2", ".ttf", ".otf",
}


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
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")
app.mount("/uploads",   StaticFiles(directory=str(UPLOADS_DIR)),   name="uploads")
app.mount("/static",    StaticFiles(directory=str(STATIC_DIR)),    name="static")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    model_ready = Path(settings.YOLO_MODEL).is_file()
    landing_exists = LANDING_PAGE.is_file() or FRONTEND_LANDING_PAGE.is_file()
    return {
        "status":          "ok",
        "version":         "2.4.0",
        "model":           settings.YOLO_MODEL,
        "model_ready":     model_ready,
        "database_ready":  is_db_connected(),
        "landing_page":    landing_exists,
        "frame_skip":      settings.FRAME_SKIP,
        "yolo_conf":       settings.YOLO_CONF,
    }


# ── Landing page — public homepage ────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_landing():
    if LANDING_PAGE.is_file():
        return FileResponse(LANDING_PAGE)
    if FRONTEND_LANDING_PAGE.is_file():
        return FileResponse(FRONTEND_LANDING_PAGE)
    # Fallback to app if no landing page present
    return FileResponse(APP_PAGE)


# ── App page — SentinelEye dashboard (login required) ────────────────────────
@app.get("/app", include_in_schema=False)
async def serve_app():
    return FileResponse(APP_PAGE)


# ── Catch-all — everything else goes to the SPA ──────────────────────────────
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str):
    # Don't catch API routes accidentally
    if full_path.startswith("api/"):
        from fastapi import HTTPException
        raise HTTPException(404, "Not found")
    if full_path:
        static_candidate = STATIC_DIR / full_path
        frontend_candidate = FRONTEND_DIR / full_path
        for candidate in (static_candidate, frontend_candidate):
            if candidate.is_file() and candidate.suffix.lower() in ROOT_ASSET_EXTS:
                return FileResponse(candidate)
    return FileResponse(APP_PAGE)
