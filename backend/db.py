from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import BaseModel
from config import settings
import logging

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None

DB_NAME_FALLBACK = "talentmind"

async def init_mongo() -> None:
    global _client, _db
    if _client is not None:
        return
    uri = settings.MONGO_URI if hasattr(settings, 'MONGO_URI') else f"mongodb://localhost:27017/{DB_NAME_FALLBACK}"
    _client = AsyncIOMotorClient(uri)
    try:
        # get_default_database works when URI contains db name; otherwise use fallback
        try:
            _db = _client.get_default_database()
        except Exception:
            _db = None
        if _db is None:
            _db = _client[DB_NAME_FALLBACK]
        await _db.command({'ping': 1})
        logger.info("✅ Connected to MongoDB")
        await ensure_indexes()
    except Exception as e:
        logger.error(f"❌ Mongo connection failed: {e}")
        raise

async def close_mongo() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None

def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("MongoDB not initialized")
    return _db

async def ensure_indexes() -> None:
    db = get_db()
    # Users: unique email
    await db.users.create_index("email", unique=True)
    # Sessions: by user_id and updated_at
    await db.sessions.create_index("user_id")
    await db.sessions.create_index("updated_at")
    await db.sessions.create_index("session_id", unique=True)
    # Messages: by session_id and timestamp
    await db.messages.create_index("session_id")
    await db.messages.create_index("timestamp")
