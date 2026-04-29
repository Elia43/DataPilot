"""Tests for the query classifier routing logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.query_classifier import classify_query, QueryClass


def test_hello_is_conversational():
    result = classify_query("hello")
    assert result.query_class == QueryClass.CONVERSATIONAL


def test_how_many_pdfs_is_kb_metadata():
    result = classify_query("how many pdfs")
    assert result.query_class == QueryClass.KB_METADATA


def test_solve_with_options_is_direct_answer():
    # "solve this" triggers the direct-answer pattern before MCQ detection
    result = classify_query(
        "solve this for me: A) option one B) option two C) option three D) option four"
    )
    assert result.query_class == QueryClass.DIRECT_ANSWER_REQUEST


def test_when_is_exam_is_administrative():
    result = classify_query("when is the exam")
    assert result.query_class == QueryClass.ADMINISTRATIVE


def test_explain_hdfs_is_conceptual():
    result = classify_query("explain what is HDFS")
    assert result.query_class == QueryClass.CONCEPTUAL_HELP


def test_what_is_mapreduce_is_conceptual():
    result = classify_query("what is MapReduce")
    assert result.query_class == QueryClass.CONCEPTUAL_HELP
