"""
studybuddy/src/vectorstore/embedder.py

Gemini vector embeddings via google.genai SDK.
Model: gemini-embedding-001 (3072-dim, stable GA)

Public API:
  embed_chunks(chunks, api_key="") -> list[dict]
  embed_query(query_text, api_key="") -> list[float]
"""

import time
from google import genai
from google.genai import types

EMBEDDING_MODEL     = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072
BATCH_SIZE          = 20
BATCH_DELAY_SECONDS = 1.0


def _get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def _get_embedding(client: genai.Client, text: str) -> list:
    """Embeds a single document chunk."""
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            ),
        )
        return response.embeddings[0].values
    except Exception as e:
        raise RuntimeError(
            f"Embedding failed: {text[:60]}... Error: {e}"
        ) from e


def _get_query_embedding(client: genai.Client, text: str) -> list:
    """Embeds a single query string."""
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            ),
        )
        return response.embeddings[0].values
    except Exception as e:
        raise RuntimeError(
            f"Query embedding failed: {text[:60]}... Error: {e}"
        ) from e


def embed_chunks(chunks: list, api_key: str = "") -> list:
    """Embeds chunk dicts in batches. Adds 'embedding' key."""
    if not chunks:
        return []

    # Get api_key from environment if not passed
    if not api_key:
        import os
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY required for embedding. "
            "Set it in your .env file."
        )

    client  = _get_client(api_key)
    batches = [
        chunks[i:i + BATCH_SIZE]
        for i in range(0, len(chunks), BATCH_SIZE)
    ]

    for batch_idx, batch in enumerate(batches):
        for chunk in batch:
            chunk["embedding"] = _get_embedding(
                client, chunk["chunk_text"]
            )
        if batch_idx < len(batches) - 1:
            time.sleep(BATCH_DELAY_SECONDS)

    for chunk in chunks:
        dim = len(chunk["embedding"])
        if dim != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Unexpected embedding dimension {dim} for chunk "
                f"{chunk['chunk_id']}. Expected {EMBEDDING_DIMENSION}."
            )
    return chunks


def embed_query(query_text: str, api_key: str = "") -> list:
    """Embeds a single query string."""
    if not query_text.strip():
        raise ValueError("Query text must not be empty.")
    if not api_key:
        import os
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is required for query embedding. "
            "Set it in your .env file or environment."
        )
    client = _get_client(api_key)
    return _get_query_embedding(client, query_text)