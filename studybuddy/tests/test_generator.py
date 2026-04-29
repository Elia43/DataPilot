"""Tests for the RAG response generator."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.generator import generate_response, GeneratorResponse
from src.rag.context_builder import ContextPayload
from src.rag.query_classifier import classify_query


def _mock_model(response_text: str) -> dict:
    """Builds the model dict that generate_response() expects."""
    mock_client = MagicMock()
    mock_resp   = MagicMock()
    mock_resp.text = response_text
    mock_client.models.generate_content.return_value = mock_resp
    return {
        "client":         mock_client,
        "model_name":     "gemini-test",
        "fallback_model": "gemini-fallback",
    }


def test_generate_response_returns_string():
    model = _mock_model(
        "HDFS stores data in blocks distributed across DataNodes.\n\n"
        "Reasoning Trace: Answer drawn from lecture context."
    )
    payload = ContextPayload(
        context_str      = "HDFS is a distributed file system that splits files into blocks.",
        citations        = [],
        chunk_count      = 1,
        is_grounded      = True,
        is_refusal       = False,
        uncertainty_flag = False,
    )
    classification = classify_query("what is HDFS")

    response = generate_response("what is HDFS", payload, classification, model)

    assert isinstance(response, GeneratorResponse)
    assert isinstance(response.answer, str)
    assert len(response.answer) > 0


def test_generate_response_handles_empty_context_gracefully():
    # uncertainty_flag=True triggers an early return in generate_response
    # without making any API call — model dict is never accessed.
    empty_payload = ContextPayload(
        context_str      = "",
        citations        = [],
        chunk_count      = 0,
        is_grounded      = False,
        is_refusal       = False,
        uncertainty_flag = True,
    )
    classification = classify_query("what is HDFS")

    response = generate_response("what is HDFS", empty_payload, classification, {})

    assert isinstance(response, GeneratorResponse)
    assert response.was_uncertain is True
    assert isinstance(response.answer, str)
    assert len(response.answer) > 0
