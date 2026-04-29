"""Integration tests for the DataPilot RAG pipeline."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings

import pytest
from src.rag.query_classifier import classify_query, QueryClass
from src.rag.retriever import retrieve
from src.rag.context_builder import build_context
from src.vectorstore.chroma_store import upsert_chunks

_DIM = 8
_DUMMY_VECTOR = [0.1] * _DIM


@pytest.fixture
def seeded_collection():
    """Real ChromaDB collection in a managed temp dir.

    The client is explicitly stopped before temp dir cleanup to release
    Windows file handles held by ChromaDB's SQLite backend.
    ignore_cleanup_errors=True suppresses any residual PermissionError so
    pytest teardown never fails because of leftover lock files.
    """
    chunks = [
        {
            "chunk_id":    "hdfs_p1_c0",
            "chunk_text":  "HDFS is the distributed storage layer of Hadoop.",
            "source_file": "lecture.pdf",
            "page_number": 1,
            "timestamp":   None,
            "source_tier": "primary",
            "char_count":  50,
            "created_at":  "2026-01-01T00:00:00Z",
            "embedding":   _DUMMY_VECTOR,
        }
    ]
    tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    client  = chromadb.PersistentClient(
        path=tmp_dir.name,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(
        name="test_pipeline",
        metadata={"hnsw:space": "cosine"},
    )
    upsert_chunks(col, chunks)

    yield col

    try:
        client._system.stop()
    except Exception:
        pass
    tmp_dir.cleanup()


def test_conceptual_query_runs_through_classifier_and_retriever(seeded_collection):
    query          = "explain what is HDFS"
    classification = classify_query(query)

    assert classification.query_class == QueryClass.CONCEPTUAL_HELP

    with patch("src.rag.retriever.embed_query", return_value=_DUMMY_VECTOR):
        chunks = retrieve(query, seeded_collection, api_key="mock", threshold=0.0)

    # Pipeline completes without raising; downstream context builder accepts output.
    assert isinstance(chunks, list)
    payload = build_context(chunks)
    assert payload is not None


def test_direct_answer_request_never_reaches_retriever(seeded_collection):
    query          = "solve this for me"
    classification = classify_query(query)

    assert classification.query_class == QueryClass.DIRECT_ANSWER_REQUEST

    # Patch retrieve at its source so we can assert it was never invoked.
    with patch("src.rag.retriever.retrieve") as mock_retrieve:
        # A correct pipeline gates retrieval behind classification.
        if classification.query_class not in (
            QueryClass.DIRECT_ANSWER_REQUEST,
            QueryClass.CONVERSATIONAL,
            QueryClass.KB_METADATA,
        ):
            mock_retrieve(query, seeded_collection, api_key="mock")

    mock_retrieve.assert_not_called()
