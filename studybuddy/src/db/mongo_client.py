"""
studybuddy/src/db/mongo_client.py

MongoDB Atlas connection singleton for DataPilot.

Public API:
  get_users_collection()    -> pymongo.collection.Collection
  get_attempts_collection() -> pymongo.collection.Collection
  ping()                    -> bool   (health-check, safe to call from UI)

The connection is created once per server process and reused across
all Streamlit reruns. Indexes are created on first connect.
"""

import os
import sys
from pathlib import Path

import certifi
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ConfigurationError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DB_NAME                    = "studybuddy"
USERS_COLLECTION           = "users"
ATTEMPTS_COLLECTION        = "quiz_attempts"
CHAT_SESSIONS_COLLECTION   = "chat_sessions"
WEAK_INTERACTIONS_COLLECTION = "weak_interactions"
CONNECT_TIMEOUT_MS         = 5_000   # 5 s — fail fast, don't hang the UI


# ── Module-level singleton ────────────────────────────────────────────────────
# Streamlit preserves module globals between reruns within the same server
# process, so this acts as an in-process connection pool.

_client: MongoClient | None = None
_db:     Database    | None = None


def _load_uri() -> str:
    """Read MONGO_URI from environment (populated by python-dotenv in config.py)."""
    uri = os.environ.get("MONGO_URI", "").strip()
    if not uri or "<username>" in uri:
        raise EnvironmentError(
            "MONGO_URI is not configured. "
            "Open studybuddy/.env and replace the placeholder with your "
            "real MongoDB Atlas connection string."
        )
    return uri


def _ensure_indexes(db: Database) -> None:
    """Create indexes if they don't already exist. Safe to call repeatedly."""
    users = db[USERS_COLLECTION]
    users.create_index(
        [("username", ASCENDING)],
        unique=True,
        name="username_unique",
    )

    attempts = db[ATTEMPTS_COLLECTION]
    attempts.create_index(
        [("username", ASCENDING), ("timestamp", DESCENDING)],
        name="username_timestamp",
    )

    chat = db[CHAT_SESSIONS_COLLECTION]
    # Unique index for upsert lookups by (username, session_id)
    chat.create_index(
        [("username", ASCENDING), ("session_id", ASCENDING)],
        unique=True,
        sparse=True,
        name="username_session_id",
    )
    # For get_active_session() and get_chat_sessions() sort queries
    chat.create_index(
        [("username", ASCENDING), ("is_active", ASCENDING), ("updated_at", DESCENDING)],
        name="username_active_updated",
    )

    weak = db[WEAK_INTERACTIONS_COLLECTION]
    # Primary lookup: all weak interactions for a user (dashboard aggregation)
    weak.create_index(
        [("username", ASCENDING), ("timestamp", DESCENDING)],
        name="weak_username_timestamp",
    )
    # Chapter mastery aggregation: filter by user + source
    weak.create_index(
        [("username", ASCENDING), ("source_file", ASCENDING)],
        name="weak_username_source",
    )


def _connect() -> Database:
    """
    Establish the connection on first call; return the cached database
    on subsequent calls.
    """
    global _client, _db

    if _db is not None:
        return _db

    uri    = _load_uri()
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=CONNECT_TIMEOUT_MS,
        tlsCAFile=certifi.where(),
    )
    # Connection is validated lazily by the first real operation.
    # Keeping the explicit ping out of the hot path saves one Atlas RTT (~2-3 s).
    db = client[DB_NAME]
    _ensure_indexes(db)

    _client = client
    _db     = db
    return _db


# ── Public helpers ────────────────────────────────────────────────────────────

def get_users_collection() -> Collection:
    """Return the 'users' collection, connecting if necessary."""
    return _connect()[USERS_COLLECTION]


def get_attempts_collection() -> Collection:
    """Return the 'quiz_attempts' collection, connecting if necessary."""
    return _connect()[ATTEMPTS_COLLECTION]


def get_chat_sessions_collection() -> Collection:
    """Return the 'chat_sessions' collection, connecting if necessary."""
    return _connect()[CHAT_SESSIONS_COLLECTION]


def get_weak_interactions_collection() -> Collection:
    """Return the 'weak_interactions' collection, connecting if necessary."""
    return _connect()[WEAK_INTERACTIONS_COLLECTION]


def ping() -> bool:
    """
    Return True if Atlas is reachable, False otherwise.
    Never raises — safe to call from the Streamlit sidebar status panel.
    """
    try:
        _connect()
        return True
    except Exception:
        return False
