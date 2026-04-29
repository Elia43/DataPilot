"""
studybuddy/src/rag/retriever.py

Orchestrates vector retrieval for a classified student query.
Embeds the query via Gemini, queries ChromaDB, and returns
filtered, tier-weighted chunks ready for context assembly.

Public API:
  retrieve(query_text, collection, api_key,
           top_k, threshold, source_filter)  -> list[dict]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.vectorstore.embedder import embed_query
from src.vectorstore.chroma_store import query_collection

DEFAULT_TOP_K = 10
DEFAULT_THRESHOLD = 0.3


def retrieve(
    query_text: str,
    collection,
    api_key: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    source_filter: list[str] | None = None,
) -> list[dict]:
    if not query_text or not query_text.strip():
        return []

    try:
        query_vector = embed_query(query_text, api_key)
    except Exception as e:
        raise RuntimeError(f"Query embedding failed: {e}")

    where_filter: dict | None = None
    if source_filter:
        if len(source_filter) == 1:
            where_filter = {"source_file": {"$eq": source_filter[0]}}
        else:
            where_filter = {"source_file": {"$in": list(source_filter)}}

    return query_collection(
        collection=collection,
        query_embedding=query_vector,
        top_k=top_k,
        threshold=threshold,
        where_filter=where_filter,
    )
