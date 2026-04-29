"""
studybuddy/src/db/quiz_store.py

Persist and retrieve quiz attempts from MongoDB Atlas.

Public API:
  save_attempt(username, quiz, results)  -> str              (inserted _id)
  get_history_and_stats(username)        -> (list[dict], dict)  ONE Atlas RTT
  get_history(username, limit)           -> list[dict]       (kept for compat)
  get_stats(username)                    -> dict             (kept for compat)
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.db.mongo_client import get_attempts_collection


# ── Private helpers ───────────────────────────────────────────────────────────

def _serialize_citations(citations: list) -> list[dict]:
    """Convert Citation dataclass objects to plain dicts for MongoDB storage."""
    out = []
    for cit in citations:
        out.append({
            "source_file":    getattr(cit, "source_file",    ""),
            "page_number":    getattr(cit, "page_number",    None),
            "confidence_pct": getattr(cit, "confidence_pct", 0),
        })
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def save_attempt(username: str, quiz: Any, results: dict) -> str:
    """
    Persist a completed, graded quiz attempt to Atlas.

    Accepts the Quiz dataclass from app.py (duck-typed to avoid a circular
    import from the UI layer) and the results dict from _grade_deterministically.

    Returns the inserted document's _id as a string.
    """
    # Sanitise the per_question list — drop the internal '_counted' key if present
    per_question = [
        {k: v for k, v in q.items() if k != "_counted"}
        for q in results.get("per_question", [])
    ]

    doc = {
        "username":     username.strip().lower(),
        "topic":        getattr(quiz, "topic", ""),
        "difficulty":   getattr(quiz, "difficulty", "Medium"),
        "generated_at": getattr(quiz, "generated_at", ""),
        "score":        results["correct"],
        "total":        results["total"],
        "percentage":   round(results["percentage"], 1),
        "timestamp":    datetime.now(timezone.utc),
        "per_question": per_question,
        "sources":      _serialize_citations(getattr(quiz, "citations", [])),
    }

    result = get_attempts_collection().insert_one(doc)
    return str(result.inserted_id)


def get_history(username: str, limit: int = 50) -> list[dict]:
    """
    Return the most recent `limit` attempts for a user, newest first.

    Each dict in the returned list is shaped for direct use in
    st.session_state.quiz_history (label, correct, incorrect) plus
    extra fields for a richer history table (topic, percentage, timestamp).
    """
    username = username.strip().lower()
    cursor   = (
        get_attempts_collection()
        .find({"username": username})
        .sort("timestamp", -1)
        .limit(limit)
    )

    history = []
    for i, doc in enumerate(cursor, start=1):
        history.append({
            # Fields consumed by the Plotly chart in app.py
            "label":     f"Session {i}",
            "correct":   doc["score"],
            "incorrect": doc["total"] - doc["score"],
            # Extra fields for the history table (used in Task 3d)
            "topic":       doc.get("topic", ""),
            "percentage":  doc.get("percentage", 0.0),
            "total":       doc["total"],
            "timestamp":   doc["timestamp"].strftime("%Y-%m-%d %H:%M")
                           if isinstance(doc.get("timestamp"), datetime)
                           else str(doc.get("timestamp", "")),
        })

    # Reverse so label "Session 1" = oldest, "Session N" = newest
    history.reverse()
    for i, item in enumerate(history, start=1):
        item["label"] = f"Session {i}"

    return history


def get_stats(username: str) -> dict:
    """
    Return cumulative quiz statistics for a user, computed server-side
    via an aggregation pipeline (no full document scan in Python).

    Returns: {"total": int, "correct": int, "percentage": float}
    """
    username = username.strip().lower()
    pipeline = [
        {"$match": {"username": username}},
        {"$group": {
            "_id":           None,
            "total_questions": {"$sum": "$total"},
            "total_correct":   {"$sum": "$score"},
        }},
    ]

    results = list(get_attempts_collection().aggregate(pipeline))

    if not results:
        return {"total": 0, "correct": 0, "percentage": 0.0}

    total   = results[0]["total_questions"]
    correct = results[0]["total_correct"]
    return {
        "total":      total,
        "correct":    correct,
        "percentage": round(correct / total * 100, 1) if total else 0.0,
    }


def get_history_and_stats(username: str) -> tuple[list[dict], dict]:
    """
    Return (history, stats) in a SINGLE Atlas round-trip.

    Replaces calling get_history() + get_stats() separately (which cost 2 RTTs).
    Fetches attempts in chronological order and computes cumulative stats in Python.
    """
    username = username.strip().lower()
    cursor   = (
        get_attempts_collection()
        .find(
            {"username": username},
            {"score": 1, "total": 1, "topic": 1, "difficulty": 1,
             "percentage": 1, "timestamp": 1, "sources": 1},
        )
        .sort("timestamp", 1)   # 1 = ASCENDING — oldest first so labels are stable
    )

    history         = []
    total_questions = 0
    total_correct   = 0

    for i, doc in enumerate(cursor, start=1):
        _sources = [
            s.get("source_file", "")
            for s in doc.get("sources", [])
            if s.get("source_file")
        ]
        history.append({
            "label":      f"Session {i}",
            "correct":    doc["score"],
            "incorrect":  doc["total"] - doc["score"],
            "topic":      doc.get("topic", ""),
            "difficulty": doc.get("difficulty", "Medium"),
            "percentage": doc.get("percentage", 0.0),
            "total":      doc["total"],
            "sources":    _sources,
            "timestamp":  doc["timestamp"].strftime("%Y-%m-%d %H:%M")
                          if isinstance(doc.get("timestamp"), datetime)
                          else str(doc.get("timestamp", "")),
        })
        total_questions += doc["total"]
        total_correct   += doc["score"]

    stats = {
        "total":      total_questions,
        "correct":    total_correct,
        "percentage": round(total_correct / total_questions * 100, 1)
                      if total_questions else 0.0,
    }
    return history, stats


def get_difficulty_stats(username: str) -> dict[str, dict]:
    """
    Return per-difficulty mastery stats for a user via a single aggregation.

    Returns a dict keyed by difficulty level, e.g.:
      {
        "Easy":   {"total": 20, "correct": 18, "attempts": 4, "percentage": 90.0},
        "Medium": {"total": 35, "correct": 22, "attempts": 7, "percentage": 62.9},
        "Hard":   {"total": 15, "correct": 6,  "attempts": 3, "percentage": 40.0},
      }
    Only levels with at least one recorded attempt are included.
    """
    username = username.strip().lower()
    pipeline = [
        {"$match": {"username": username}},
        {"$group": {
            "_id":             {"$ifNull": ["$difficulty", "Medium"]},
            "total_questions": {"$sum": "$total"},
            "total_correct":   {"$sum": "$score"},
            "attempts":        {"$sum": 1},
        }},
    ]
    rows = list(get_attempts_collection().aggregate(pipeline))

    result: dict[str, dict] = {}
    for row in rows:
        lvl   = row["_id"] or "Medium"
        total = row["total_questions"]
        right = row["total_correct"]
        result[lvl] = {
            "total":      total,
            "correct":    right,
            "attempts":   row["attempts"],
            "percentage": round(right / total * 100, 1) if total else 0.0,
        }
    return result


def get_topic_mastery(username: str) -> dict:
    """
    Aggregate mastery per quiz topic from quiz_attempts.

    Returns a dict keyed by topic title, e.g.:
      {"HDFS Block Replication": {"total": 15, "correct": 12,
                                   "attempts": 3, "mastery_pct": 80.0}}
    Only topics with a non-empty title are included.
    """
    username = username.strip().lower()
    pipeline = [
        {"$match": {"username": username,
                    "topic":    {"$exists": True, "$ne": ""}}},
        {"$group": {
            "_id":      "$topic",
            "total":    {"$sum": "$total"},
            "correct":  {"$sum": "$score"},
            "attempts": {"$sum": 1},
        }},
    ]
    rows   = list(get_attempts_collection().aggregate(pipeline))
    result = {}
    for row in rows:
        topic = row["_id"] or "General"
        total = row["total"]
        right = row["correct"]
        result[topic] = {
            "total":       total,
            "correct":     right,
            "attempts":    row["attempts"],
            "mastery_pct": round(right / total * 100, 1) if total else 0.0,
        }
    return result


def get_attempt_question_details(
    username: str, attempt_ids: list[str]
) -> dict[str, str]:
    """
    Return {question_text: correct_text} for the given attempt IDs.

    Used as a fallback to enrich flashcard records that were saved before
    correct_answer_text was stored directly on the weak_interaction doc.
    Looks up per_question arrays from the quiz_attempts collection.
    """
    from bson import ObjectId

    username = username.strip().lower()
    oid_list = []
    for aid in attempt_ids:
        try:
            oid_list.append(ObjectId(aid))
        except Exception:
            pass
    if not oid_list:
        return {}

    docs = get_attempts_collection().find(
        {"_id": {"$in": oid_list}, "username": username},
        {"per_question": 1, "_id": 0},
    )
    result: dict[str, str] = {}
    for doc in docs:
        for q in doc.get("per_question", []):
            qt = q.get("question", "")
            ct = q.get("correct_text", "")
            if qt and ct and qt not in result:
                result[qt] = ct
    return result


def delete_user_attempts(username: str) -> int:
    """Delete all quiz attempts for a user. Returns the count of deleted documents."""
    username = username.strip().lower()
    result = get_attempts_collection().delete_many({"username": username})
    return result.deleted_count
