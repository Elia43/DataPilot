"""Tests for the vector store retrieval pipeline."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings

import pytest
from src.rag.retriever import retrieve
from src.vectorstore.chroma_store import upsert_chunks

# Small embedding dimension — keeps fixture fast; ChromaDB imposes no minimum.
_DIM = 8
_UNIT_0 = [1.0] + [0.0] * (_DIM - 1)   # unit vector in dim 0
_UNIT_1 = [0.0, 1.0] + [0.0] * (_DIM - 2)  # orthogonal to _UNIT_0


def _make_chunk(i: int, embed=None) -> dict:
    return {
        "chunk_id":    f"test_p1_c{i}",
        "chunk_text":  f"HDFS chunk {i}: distributed storage.",
        "source_file": "test.pdf",
        "page_number": i + 1,
        "timestamp":   None,
        "source_tier": "primary",
        "char_count":  40,
        "created_at":  "2026-01-01T00:00:00Z",
        "embedding":   embed if embed is not None else _UNIT_0,
    }


@pytest.fixture
def collection():
    """Real ChromaDB collection in a managed temp dir.

    The client is explicitly stopped before temp dir cleanup to release
    Windows file handles held by ChromaDB's SQLite backend.
    ignore_cleanup_errors=True suppresses any residual PermissionError so
    pytest teardown never fails because of leftover lock files.
    """
    chunks  = [_make_chunk(i) for i in range(5)]
    tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    client  = chromadb.PersistentClient(
        path=tmp_dir.name,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(
        name="test_retriever",
        metadata={"hnsw:space": "cosine"},
    )
    upsert_chunks(col, chunks)

    yield col

    try:
        client._system.stop()
    except Exception:
        pass
    tmp_dir.cleanup()


def test_retrieve_returns_list(collection):
    with patch("src.rag.retriever.embed_query", return_value=_UNIT_0):
        result = retrieve("what is HDFS", collection, api_key="mock", threshold=0.0)
    assert isinstance(result, list)


def test_retrieve_empty_at_threshold_one(collection):
    # _UNIT_1 is orthogonal to _UNIT_0 (cosine similarity ≈ 0.0).
    # With threshold=1.0 only similarity >= 1.0 passes — nothing qualifies.
    with patch("src.rag.retriever.embed_query", return_value=_UNIT_1):
        result = retrieve("what is HDFS", collection, api_key="mock", threshold=1.0)
    assert result == []


def test_retrieve_respects_top_k(collection):
    # Collection has 5 chunks; top_k=2 must cap the result at 2.
    with patch("src.rag.retriever.embed_query", return_value=_UNIT_0):
        result = retrieve(
            "what is HDFS", collection, api_key="mock", top_k=2, threshold=0.0
        )
    assert len(result) <= 2
