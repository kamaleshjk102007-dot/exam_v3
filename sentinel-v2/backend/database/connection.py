"""SentinelEye — MongoDB async connection via Motor"""
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from config import settings

logger = logging.getLogger("sentinel.db")

client = None
db     = None


async def connect_db():
    global client, db
    logger.info(f"Connecting to MongoDB: {settings.MONGODB_URL}")
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db     = client[settings.MONGODB_DB]
    # Verify connection
    await client.admin.command("ping")
    logger.info(f"MongoDB connected — database: {settings.MONGODB_DB}")
    await _create_indexes()


async def disconnect_db():
    global client
    if client:
        client.close()
        logger.info("MongoDB disconnected")


async def _create_indexes():
    """Create performance indexes on first run."""
    await db.users.create_index("email",    unique=True)
    await db.users.create_index("username", unique=True)
    await db.classrooms.create_index("created_by")
    await db.cameras.create_index("classroom_id")
    await db.alerts.create_index([("session_id", 1), ("created_at", -1)])
    await db.sessions.create_index([("classroom_id", 1), ("started_at", -1)])
    logger.info("Database indexes created")


def get_db():
    return db


def is_db_connected() -> bool:
    return db is not None
