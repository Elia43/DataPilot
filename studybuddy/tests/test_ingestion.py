"""Tests for the ingestion pipeline (pdf_parser)."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import fitz  # PyMuPDF — already in project requirements
from src.ingestion.pdf_parser import chunk_text, parse_pdf, load_parsed_chunks


def test_chunk_text_returns_list():
    text   = "HDFS is a distributed file system. It stores data blocks across DataNodes."
    result = chunk_text(text)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(c, str) for c in result)


def test_parse_pdf_raises_on_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            parse_pdf(os.path.join(tmp, "does_not_exist.pdf"), tmp)


def test_parsed_chunks_have_expected_keys():
    # Build a minimal single-page PDF in memory using PyMuPDF.
    doc  = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "HDFS distributes data blocks across DataNodes in a cluster.")
    doc.insert_page(-1)  # second page so total_pages > 1
    doc[-1].insert_text((72, 72), "The NameNode manages filesystem metadata.")

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, "test_lecture.pdf")
        doc.save(pdf_path)
        doc.close()

        # api_key="" skips Vision calls — text-only parsing, no network needed.
        json_path = parse_pdf(pdf_path, tmp, api_key="")
        chunks    = load_parsed_chunks(json_path)

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    required_keys = {"chunk_id", "chunk_text", "source_file", "page_number", "source_tier"}
    assert required_keys.issubset(chunks[0].keys())
