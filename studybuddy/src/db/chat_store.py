"""
studybuddy/src/db/chat_store.py

Real-time chat session persistence for DataPilot.

Design:
  • Each chat session has a stable session_id (UUID4) created at login.
  • upsert_active_session() is called after every assistant response —
    it creates the document on first call and updates it on every
    subsequent call. MongoDB $setOnInsert ensures started_at is only
    written once.
  • finalize_session() marks is_active=False on New Chat / Logout so
    the session appears in the History page.
  • get_active_session() is called on login to restore an in-progress
    conversation instead of starting with an empty chat.

Public API:
  upsert_active_session(username, session_id, messages) -> None
  finalize_session(username, session_id)                -> None
  get_active_session(username)                          -> dict | None
  get_chat_sessions(username, limit)                    -> list[dict]
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.db.mongo_client import get_chat_sessions_collection

_SKIP_CONTENT = frozenset({
    "⏳ Thinking...",
    "⚠️ Error occurred. Please try again.",
})


def _clean_messages(messages: list) -> list[dict]:
    """Strip metadata, placeholders, and system noise — store only role+content."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant")
        and not m.get("is_placeholder", False)
        and m.get("content", "").strip()
        and m.get("content") not in _SKIP_CONTENT
    ]


def upsert_active_session(username: str, session_id: str, messages: list) -> None:
    """
    Create or update the active session document for this user.

    Called after every assistant response for real-time persistence.
    Uses upsert=True so the first call creates the document and all
    subsequent calls update it without touching started_at.
    """
    clean = _clean_messages(messages)
    if len(clean) < 2:
        return

    now = datetime.now(timezone.utc)
    get_chat_sessions_collection().update_one(
        {"username": username.strip().lower(), "session_id": session_id},
        {
            "$set": {
                "updated_at":    now,
                "message_count": len(clean),
                "messages":      clean,
                "is_active":     True,
            },
            "$setOnInsert": {
                "started_at": now,
            },
        },
        upsert=True,
    )


def finalize_session(username: str, session_id: str) -> None:
    """
    Mark a session as complete (is_active=False).

    Called on New Chat and Logout. Finalized sessions appear in the
    History page; active sessions are excluded from that view.
    """
    get_chat_sessions_collection().update_one(
        {"username": username.strip().lower(), "session_id": session_id},
        {"$set": {"is_active": False, "ended_at": datetime.now(timezone.utc)}},
    )


def get_active_session(username: str) -> dict | None:
    """
    Return the most recently updated active session for this user, or None.

    Called on login so an interrupted conversation is restored automatically.
    """
    return get_chat_sessions_collection().find_one(
        {"username": username.strip().lower(), "is_active": True},
        sort=[("updated_at", -1)],
    )


def get_chat_sessions(username: str, limit: int = 10) -> list[dict]:
    """
    Return the most recent finalized (completed) sessions, newest first.

    Each dict: {started_at, ended_at, message_count, messages}
    Active/in-progress sessions are excluded from the History view.
    """
    username = username.strip().lower()
    cursor   = (
        get_chat_sessions_collection()
        .find(
            {"username": username, "is_active": False},
            {"username": 0, "session_id": 0},
        )
        .sort("ended_at", -1)
        .limit(limit)
    )

    sessions = []
    for doc in cursor:
        ended   = doc.get("ended_at")
        started = doc.get("started_at")
        sessions.append({
            "started_at":    started.strftime("%Y-%m-%d %H:%M")
                             if isinstance(started, datetime) else str(started or ""),
            "ended_at":      ended.strftime("%Y-%m-%d %H:%M")
                             if isinstance(ended, datetime)   else str(ended or ""),
            "message_count": doc.get("message_count", 0),
            "messages":      doc.get("messages", []),
        })
    return sessions
