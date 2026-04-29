"""
studybuddy/src/db/auth.py

User registration and login for DataPilot.
Passwords are hashed with bcrypt — plain-text passwords are never stored.

Public API:
  register_user(username, password) -> dict        (raises ValueError on bad input / duplicate)
  login_user(username, password)    -> dict | None (None = wrong credentials)
  get_user(username)                -> dict | None
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import bcrypt
from pymongo.errors import DuplicateKeyError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.db.mongo_client import get_users_collection

# ── Validation constants ──────────────────────────────────────────────────────

MIN_USERNAME_LEN = 3
MAX_USERNAME_LEN = 30
MIN_PASSWORD_LEN = 6


# ── Private helpers ───────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def _validate(username: str, password: str) -> None:
    """Raise ValueError with a user-friendly message if inputs are invalid."""
    if not username or not password:
        raise ValueError("Username and password are required.")
    if len(username) < MIN_USERNAME_LEN:
        raise ValueError(f"Username must be at least {MIN_USERNAME_LEN} characters.")
    if len(username) > MAX_USERNAME_LEN:
        raise ValueError(f"Username must be {MAX_USERNAME_LEN} characters or fewer.")
    if len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")


# ── Public API ────────────────────────────────────────────────────────────────

def register_user(username: str, password: str) -> dict:
    """
    Create a new user document in MongoDB.

    Raises:
      ValueError — invalid input or username already taken.

    Returns the inserted document (including the new _id).
    """
    username = username.strip().lower()
    _validate(username, password)

    doc = {
        "username":      username,
        "password_hash": _hash_password(password),
        "created_at":    datetime.now(timezone.utc),
    }

    try:
        result  = get_users_collection().insert_one(doc)
        doc["_id"] = result.inserted_id
        return doc
    except DuplicateKeyError:
        raise ValueError(f"Username '{username}' is already taken. Please choose another.")


def login_user(username: str, password: str) -> dict | None:
    """
    Verify credentials against the stored bcrypt hash.

    Returns the user document on success.
    Returns None on wrong username or wrong password.
    Never raises for authentication failures — the caller decides the UX.
    """
    username = username.strip().lower()
    if not username or not password:
        return None

    user = get_users_collection().find_one({"username": username})
    if user is None:
        return None

    if not _verify_password(password, user["password_hash"]):
        return None

    return user


def get_user(username: str) -> dict | None:
    """Fetch a user document by username. Returns None if not found."""
    return get_users_collection().find_one({"username": username.strip().lower()})
