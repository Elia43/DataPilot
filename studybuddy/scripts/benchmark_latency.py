"""
studybuddy/scripts/benchmark_latency.py

Latency benchmarking for the DataPilot RAG pipeline.
Times embedding, ChromaDB retrieval, and Gemini generation separately
for 10 hardcoded test questions, then prints a summary table and saves
results to benchmark_results.json.

Run from the project root (Studybuddy/):
    python studybuddy/scripts/benchmark_latency.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent   # studybuddy/scripts/
PKG_DIR    = SCRIPT_DIR.parent                 # studybuddy/

sys.path.insert(0, str(PKG_DIR))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from config import (
    load_api_key,
    initialize_gemini_model,
    CHROMA_DB_PATH,
    SIMILARITY_THRESHOLD,
    TOP_K_RESULTS,
)
from src.vectorstore.chroma_store import get_collection
from src.vectorstore.embedder import embed_query
from src.vectorstore.chroma_store import query_collection
from src.rag.query_classifier import classify_query
from src.rag.context_builder import build_context
from src.rag.generator import generate_response

# ── Test questions ────────────────────────────────────────────────────────────
QUESTIONS = [
    "What is HDFS and how does it store data?",
    "What is the difference between MapReduce and Spark?",
    "What are the main components of the Hadoop ecosystem?",
    "Explain the role of the NameNode in HDFS",
    "What is a RDD in Apache Spark?",
    "What are the advantages of batch processing?",
    "How does MapReduce handle fault tolerance?",
    "What is the difference between structured and unstructured data?",
    "Explain data replication in HDFS",
    "What is big data and what are the three Vs?",
]

RESULTS_PATH = SCRIPT_DIR / "benchmark_results.json"


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 2)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"min": None, "max": None, "avg": None}
    return {
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "avg": round(sum(values) / len(values), 2),
    }


def run_benchmark() -> None:
    print("DataPilot Latency Benchmark")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────────
    print("Loading API key and initialising clients…")
    api_key    = load_api_key()
    model      = initialize_gemini_model()
    collection = get_collection(CHROMA_DB_PATH)
    print(f"ChromaDB collection: {collection.name}  ({collection.count()} chunks)")
    print()

    results: list[dict] = []

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"[{i:2d}/10] {question[:70]}")
        record: dict = {
            "index":            i,
            "question":         question,
            "embed_ms":         None,
            "retrieve_ms":      None,
            "generate_ms":      None,
            "total_ms":         None,
            "chunks_retrieved": 0,
            "success":          False,
            "error":            None,
        }

        try:
            t_start = time.perf_counter()

            # Stage 1 — Embedding
            t0 = time.perf_counter()
            query_vector = embed_query(question, api_key)
            t1 = time.perf_counter()
            record["embed_ms"] = _ms(t0, t1)

            # Stage 2 — ChromaDB retrieval
            t2 = time.perf_counter()
            chunks = query_collection(
                collection      = collection,
                query_embedding = query_vector,
                top_k           = TOP_K_RESULTS,
                threshold       = SIMILARITY_THRESHOLD,
            )
            t3 = time.perf_counter()
            record["retrieve_ms"]      = _ms(t2, t3)
            record["chunks_retrieved"] = len(chunks)

            # Stage 3 — Gemini generation
            classification  = classify_query(question)
            context_payload = build_context(chunks)

            t4 = time.perf_counter()
            generate_response(
                query_text            = question,
                context_payload       = context_payload,
                classification_result = classification,
                model                 = model,
            )
            t5 = time.perf_counter()
            record["generate_ms"] = _ms(t4, t5)

            t_end = time.perf_counter()
            record["total_ms"] = _ms(t_start, t_end)
            record["success"]  = True

            print(
                f"        embed={record['embed_ms']:7.1f} ms  "
                f"retrieve={record['retrieve_ms']:6.1f} ms  "
                f"generate={record['generate_ms']:7.1f} ms  "
                f"total={record['total_ms']:7.1f} ms  "
                f"chunks={record['chunks_retrieved']}"
            )

        except Exception as exc:
            record["error"] = type(exc).__name__
            print(f"        ERROR: {type(exc).__name__}: {exc}")

        results.append(record)

    # ── Aggregate statistics ───────────────────────────────────────
    successful = [r for r in results if r["success"]]

    embed_times    = [r["embed_ms"]    for r in successful]
    retrieve_times = [r["retrieve_ms"] for r in successful]
    generate_times = [r["generate_ms"] for r in successful]
    total_times    = [r["total_ms"]    for r in successful]

    summary = {
        "embed_ms":    _stats(embed_times),
        "retrieve_ms": _stats(retrieve_times),
        "generate_ms": _stats(generate_times),
        "total_ms":    _stats(total_times),
    }

    # ── Print results table ────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Results  ({len(successful)}/{len(QUESTIONS)} successful)")
    print("=" * 60)
    print(f"{'Stage':<16} {'Min (ms)':>10} {'Max (ms)':>10} {'Avg (ms)':>10}")
    print("-" * 50)
    for stage, label in [
        ("embed_ms",    "Embedding"),
        ("retrieve_ms", "Retrieval"),
        ("generate_ms", "Generation"),
        ("total_ms",    "Total E2E"),
    ]:
        s = summary[stage]
        if s["min"] is None:
            print(f"{label:<16} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        else:
            print(
                f"{label:<16} {s['min']:>10.1f} {s['max']:>10.1f} {s['avg']:>10.1f}"
            )
    print("=" * 60)

    # ── Save JSON ──────────────────────────────────────────────────
    output = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "total_runs":   len(QUESTIONS),
        "successful":   len(successful),
        "questions":    results,
        "summary":      summary,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_benchmark()
