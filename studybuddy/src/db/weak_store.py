"""
studybuddy/src/db/weak_store.py

Adaptive learning — stores every incorrect quiz answer linked to its
source chapter, and provides the aggregation queries that power the
chapter mastery dashboard and adaptive quiz generation.

Public API:
  save_weak_interactions(username, attempt_id, per_question, quiz) -> None
  get_chapter_mastery(username)                                     -> dict
  get_weak_sources(username)                                        -> list[str]
  get_weak_interaction_texts(username, limit)                       -> list[str]
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.db.mongo_client import get_weak_interactions_collection, get_attempts_collection


def save_weak_interactions(
    username: str,
    attempt_id: str,
    per_question: list[dict],
    quiz,
) -> None:
    """
    Persist every incorrect answer as a weak interaction document.

    Each doc is linked to the primary source chapter so the chapter mastery
    dashboard can aggregate by source_file.  Called immediately after
    save_attempt() — only writes when there are wrong answers.
    """
    username = username.strip().lower()
    now      = datetime.now(timezone.utc)

    citations     = getattr(quiz, "citations", [])
    primary_source = (
        citations[0].source_file if citations
        else getattr(quiz, "topic", "Unknown")
    )

    docs = []
    for q in per_question:
        if q.get("is_correct"):
            continue
        docs.append({
            "username":            username,
            "source_file":         primary_source,
            "question_text":       q["question"],
            "correct_answer":      q["correct_letter"],
            "correct_answer_text": q.get("correct_text", ""),
            "student_answer":      q.get("student_letter", ""),
            "topic":               getattr(quiz, "topic", ""),
            "difficulty":          getattr(quiz, "difficulty", "Medium"),
            "attempt_id":          attempt_id,
            "timestamp":           now,
        })

    if docs:
        get_weak_interactions_collection().insert_many(docs)


def get_chapter_mastery(username: str) -> dict:
    """
    Aggregate per-chapter mastery from quiz_attempts.

    Unwinds the `sources` array on each attempt so every cited chapter
    gets credit for that attempt's score/total.  Returns a dict keyed by
    source_file, e.g.:
      {
        "HDFS_Lecture.pdf": {"total": 25, "correct": 18, "attempts": 5,
                             "mastery_pct": 72.0},
        ...
      }
    """
    username = username.strip().lower()
    pipeline = [
        {"$match": {"username": username,
                    "sources":  {"$exists": True, "$not": {"$size": 0}}}},
        {"$unwind": "$sources"},
        {"$group": {
            "_id":      "$sources.source_file",
            "total":    {"$sum": "$total"},
            "correct":  {"$sum": "$score"},
            "attempts": {"$sum": 1},
        }},
    ]
    rows   = list(get_attempts_collection().aggregate(pipeline))
    result = {}
    for row in rows:
        src   = row["_id"] or "Unknown"
        total = row["total"]
        right = row["correct"]
        result[src] = {
            "total":       total,
            "correct":     right,
            "attempts":    row["attempts"],
            "mastery_pct": round(right / total * 100, 1) if total else 0.0,
        }
    return result


def get_weak_sources(username: str) -> list[str]:
    """Return distinct source_files where the user has stored incorrect answers."""
    return get_weak_interactions_collection().distinct(
        "source_file", {"username": username.strip().lower()}
    )


def delete_weak_interactions(username: str) -> int:
    """Delete all weak interaction records for a user. Returns the count deleted."""
    username = username.strip().lower()
    result = get_weak_interactions_collection().delete_many({"username": username})
    return result.deleted_count


def get_weak_interactions(username: str, limit: int = 200) -> list[dict]:
    """
    Return the most recent incorrect-answer records for a user.

    Used by the Flashcard page to display wrong questions grouped by
    source chapter.  Each returned dict includes: question_text,
    correct_answer, topic, source_file, difficulty, timestamp (str).
    """
    username = username.strip().lower()
    docs = (
        get_weak_interactions_collection()
        .find(
            {"username": username},
            {
                "question_text":       1,
                "correct_answer":      1,
                "correct_answer_text": 1,
                "topic":               1,
                "source_file":         1,
                "difficulty":          1,
                "attempt_id":          1,
                "timestamp":           1,
                "_id":                 0,
            },
        )
        .sort("timestamp", -1)
        .limit(limit)
    )
    result = []
    for d in docs:
        ts = d.get("timestamp")
        result.append({
            "question_text":       d.get("question_text", ""),
            "correct_answer":      d.get("correct_answer", ""),
            "correct_answer_text": d.get("correct_answer_text", ""),
            "topic":               d.get("topic", ""),
            "source_file":         d.get("source_file", ""),
            "difficulty":          d.get("difficulty", "Medium"),
            "attempt_id":          d.get("attempt_id", ""),
            "timestamp":           ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else str(ts or ""),
        })
    return result


def get_weak_interaction_texts(username: str, limit: int = 50) -> list[str]:
    """
    Return question texts from the user's most recent incorrect answers.

    Used by the adaptive quiz generator to build a semantically focused
    retrieval query aimed at the student's weakest areas.
    """
    username = username.strip().lower()
    docs     = (
        get_weak_interactions_collection()
        .find({"username": username}, {"question_text": 1, "_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return [d["question_text"] for d in docs if d.get("question_text")]
