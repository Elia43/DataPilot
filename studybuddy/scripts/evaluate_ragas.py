"""
studybuddy/scripts/evaluate_ragas.py

Evaluates DataPilot's RAG pipeline using direct Gemini API scoring.
For each of the 20 test questions the script:
  1. Runs the full RAG pipeline (embed → retrieve → generate)
  2. Makes three separate gemini-3.1-flash-lite-preview calls to score:
       - faithfulness        (answer grounded in retrieved context)
       - answer_relevancy    (answer addresses the question)
       - context_precision   (context is relevant to the question)
  Each scoring call asks for a single float in [0.0, 1.0].

No external evaluation framework required — only the packages already
used by the project (google-genai).

Run from the project root (Studybuddy/):
    python studybuddy/scripts/evaluate_ragas.py
"""

import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Path setup (identical to benchmark_latency.py) ────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent   # studybuddy/scripts/
PKG_DIR    = SCRIPT_DIR.parent                 # studybuddy/

sys.path.insert(0, str(PKG_DIR))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ── Project imports ────────────────────────────────────────────────────────────
from config import (
    load_api_key,
    initialize_gemini_model,
    CHROMA_DB_PATH,
    SIMILARITY_THRESHOLD,
    TOP_K_RESULTS,
)
from src.vectorstore.chroma_store import get_collection, query_collection
from src.vectorstore.embedder import embed_query
from src.rag.query_classifier import classify_query
from src.rag.context_builder import build_context
from src.rag.generator import generate_response

# ── Evaluation questions and ground truths ─────────────────────────────────────
QUESTIONS = [
    "What is Big Data?",
    "Why is Big Data important today?",
    "What are the main stages of the Big Data process?",
    "What are the three main characteristics of Big Data?",
    "What is the role of Big Data Architecture?",
    "What is Batch Processing?",
    "When is Batch Processing mainly used?",
    "What is Hadoop?",
    "What are the main components of Hadoop?",
    "What is the role of HDFS in Hadoop?",
    "What is Hadoop MapReduce?",
    "What are the main advantages of MapReduce?",
    "What are the main phases of the MapReduce computation model?",
    "How does MapReduce handle node failures?",
    "What is the role of a Combiner in MapReduce?",
    "What is Apache Spark?",
    "Why is Spark faster than Hadoop MapReduce?",
    "What is an RDD in Spark?",
    "What is caching in Spark?",
    "What is a shuffle operation in Spark?",
]

GROUND_TRUTHS = [
    "Big Data refers to extremely large and complex datasets that cannot be processed efficiently using traditional data processing systems.",
    "Big Data is important because enormous amounts of data are generated every minute from social media, sensors, mobile devices, and online systems.",
    "The Big Data process includes: Capture, Organize, Integrate, Analyze, and Act.",
    "The three main characteristics are Volume, Velocity, and Variety.",
    "Big Data Architecture provides the structure needed to store, manage, secure, and analyze large-scale data efficiently using distributed systems.",
    "Batch processing is a method where data is collected over a period of time and processed together in large blocks instead of being processed immediately.",
    "Batch processing is used when real-time results are not required and when processing large volumes of stored data is more important than fast analytics.",
    "Hadoop is a distributed software framework that allows processing and storing large datasets across clusters of computers using simple programming models.",
    "The main Hadoop components are HDFS, MapReduce, Hadoop Common, and YARN.",
    "HDFS is the storage system of Hadoop. It splits files into blocks and distributes them across multiple DataNodes while a NameNode manages metadata.",
    "Hadoop MapReduce is a distributed programming framework used to process very large datasets in parallel across clusters in a reliable and fault-tolerant way.",
    "MapReduce allows redundant data storage, moving computation close to data, reduced data movement, and a simple programming model.",
    "The main phases are Map, Shuffle and Sort, and Reduce.",
    "If a worker node fails, map tasks are reassigned. If the master fails, the job stops and the client is notified.",
    "A Combiner performs local aggregation on mapper outputs before sending data to reducers, reducing network traffic.",
    "Apache Spark is a fast and general engine for large-scale data processing supporting batch, streaming, machine learning, and SQL.",
    "Spark is faster because it processes data in memory while Hadoop MapReduce processes data on disk.",
    "RDD is a distributed collection of data split across multiple machines and processed in parallel.",
    "Caching stores data in memory so it can be reused quickly without reloading from disk.",
    "A shuffle operation redistributes data across partitions to group related data together.",
]

RESULTS_PATH = SCRIPT_DIR / "ragas_results.json"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision"]
EVAL_MODEL   = "gemini-3.1-flash-lite-preview"

# Cap chunks fed to scoring prompts so we don't exceed token limits.
_MAX_SCORE_CHUNKS    = 6
_MAX_CHUNK_CHARS     = 600


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _avg(values: list) -> float | None:
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def _fmt_contexts(contexts: list[str]) -> str:
    chunks = contexts[:_MAX_SCORE_CHUNKS]
    return "\n\n".join(
        f"[Chunk {i + 1}]\n{c[:_MAX_CHUNK_CHARS]}"
        for i, c in enumerate(chunks)
    )


# ── Gemini scoring ─────────────────────────────────────────────────────────────

def _gemini_score(client, prompt: str) -> float | None:
    """
    Calls EVAL_MODEL with `prompt` and parses the first float from the
    response. Returns a value clamped to [0.0, 1.0], or None on failure.
    """
    try:
        response = client.models.generate_content(
            model    = EVAL_MODEL,
            contents = prompt,
        )
        text  = response.text.strip()
        match = re.search(r'[0-9]+\.?[0-9]*', text)
        if match:
            return round(min(max(float(match.group()), 0.0), 1.0), 4)
        return None
    except Exception as exc:
        print(f"         SCORE ERROR ({type(exc).__name__}): {exc}")
        return None


def score_faithfulness(client, answer: str, contexts: list[str]) -> float | None:
    """
    Measures whether every claim in the answer is supported by the
    retrieved context (1.0 = fully grounded, 0.0 = hallucinated).
    """
    prompt = (
        "You are a strict factual evaluator.\n\n"
        "RETRIEVED CONTEXT:\n"
        f"{_fmt_contexts(contexts)}\n\n"
        "GENERATED ANSWER:\n"
        f"{answer[:1200]}\n\n"
        "TASK: Score how faithfully the generated answer is supported by "
        "the retrieved context above.\n"
        "  1.0 = every claim in the answer is directly found in the context\n"
        "  0.5 = some claims are supported, others are not\n"
        "  0.0 = the answer introduces facts not present in the context\n\n"
        "Reply with ONLY a single decimal number between 0.0 and 1.0. "
        "No explanation, no other text."
    )
    return _gemini_score(client, prompt)


def score_answer_relevancy(client, question: str, answer: str) -> float | None:
    """
    Measures whether the generated answer directly addresses the question
    (1.0 = fully relevant, 0.0 = off-topic).
    """
    prompt = (
        "You are evaluating answer quality.\n\n"
        f"QUESTION:\n{question}\n\n"
        "GENERATED ANSWER:\n"
        f"{answer[:1200]}\n\n"
        "TASK: Score how well the answer addresses the question.\n"
        "  1.0 = the answer directly and completely answers the question\n"
        "  0.5 = the answer is partially relevant but misses key aspects\n"
        "  0.0 = the answer does not address the question at all\n\n"
        "Reply with ONLY a single decimal number between 0.0 and 1.0. "
        "No explanation, no other text."
    )
    return _gemini_score(client, prompt)


def score_context_precision(
    client,
    question: str,
    contexts: list[str],
    ground_truth: str,
) -> float | None:
    """
    Measures what proportion of the retrieved context is actually relevant
    to answering the question (1.0 = all chunks relevant, 0.0 = none).
    """
    prompt = (
        "You are evaluating retrieval quality.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EXPECTED ANSWER:\n{ground_truth}\n\n"
        "RETRIEVED CONTEXT:\n"
        f"{_fmt_contexts(contexts)}\n\n"
        "TASK: Score what proportion of the retrieved context chunks are "
        "relevant to answering the question.\n"
        "  1.0 = all chunks are directly relevant\n"
        "  0.5 = about half the chunks are relevant\n"
        "  0.0 = none of the chunks are relevant\n\n"
        "Reply with ONLY a single decimal number between 0.0 and 1.0. "
        "No explanation, no other text."
    )
    return _gemini_score(client, prompt)


# ── Phase 1: RAG pipeline ──────────────────────────────────────────────────────

def collect_rag_outputs(
    questions: list[str],
    api_key: str,
    model: dict,
    collection,
) -> tuple[list[str], list[list[str]]]:
    answers:  list[str]       = []
    contexts: list[list[str]] = []

    for i, question in enumerate(questions, start=1):
        print(f"  [{i:2d}/{len(questions)}] {question[:65]}")
        try:
            query_vector    = embed_query(question, api_key)
            chunks          = query_collection(
                collection      = collection,
                query_embedding = query_vector,
                top_k           = TOP_K_RESULTS,
                threshold       = SIMILARITY_THRESHOLD,
            )
            context_payload = build_context(chunks)
            classification  = classify_query(question)
            response        = generate_response(
                query_text            = question,
                context_payload       = context_payload,
                classification_result = classification,
                model                 = model,
            )
            answers.append(response.answer)
            contexts.append(
                [c["chunk_text"] for c in chunks] if chunks else [""]
            )
        except Exception as exc:
            print(f"         ERROR: {type(exc).__name__}: {exc}")
            answers.append("")
            contexts.append([""])

    return answers, contexts


# ── Phase 2: Gemini scoring ────────────────────────────────────────────────────

def score_all(
    client,
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> list[dict]:
    records: list[dict] = []

    for i, (q, a, ctx, gt) in enumerate(
        zip(questions, answers, contexts, ground_truths), start=1
    ):
        print(f"  [{i:2d}/{len(questions)}] {q[:60]}")

        faith = score_faithfulness(client, a, ctx)
        relev = score_answer_relevancy(client, q, a)
        prec  = score_context_precision(client, q, ctx, gt)

        f_str = f"{faith:.2f}" if faith is not None else " N/A"
        r_str = f"{relev:.2f}" if relev is not None else " N/A"
        p_str = f"{prec:.2f}"  if prec  is not None else " N/A"
        print(f"         faith={f_str}  relev={r_str}  prec={p_str}")

        records.append({
            "index":             i,
            "question":          q,
            "generated_answer":  a,
            "ground_truth":      gt,
            "chunks_retrieved":  len(ctx),
            "faithfulness":      faith,
            "answer_relevancy":  relev,
            "context_precision": prec,
        })

    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def run_evaluation() -> None:
    print("DataPilot RAG Evaluation  (scoring model: gemini-3.1-flash-lite-preview)")
    print("=" * 66)

    print("Loading API key and initialising clients…")
    api_key    = load_api_key()
    model      = initialize_gemini_model()
    collection = get_collection(CHROMA_DB_PATH)
    print(f"ChromaDB collection: {collection.name}  ({collection.count()} chunks)\n")

    # Phase 1 — generate RAG answers
    print("Phase 1 — Running RAG pipeline for all 20 questions…")
    answers, contexts = collect_rag_outputs(QUESTIONS, api_key, model, collection)
    print()

    # Phase 2 — score with Gemini (3 calls × 20 questions = 60 API calls)
    print("Phase 2 — Scoring each answer (3 Gemini calls per question)…")
    gemini_client = model["client"]
    per_question  = score_all(
        gemini_client, QUESTIONS, answers, contexts, GROUND_TRUTHS
    )
    print()

    averages = {m: _avg([r[m] for r in per_question]) for m in METRIC_NAMES}

    # ── Results table ──────────────────────────────────────────────
    W = 66
    print("=" * W)
    print(f"{'#':>2}  {'Question':<44} {'Faith':>6} {'Relev':>6} {'Prec':>6}")
    print("-" * W)
    for r in per_question:
        faith = f"{r['faithfulness']:.2f}"    if r["faithfulness"]    is not None else "  N/A"
        relev = f"{r['answer_relevancy']:.2f}" if r["answer_relevancy"] is not None else "  N/A"
        prec  = f"{r['context_precision']:.2f}" if r["context_precision"] is not None else "  N/A"
        print(f"{r['index']:>2}  {r['question'][:43]:<44} {faith:>6} {relev:>6} {prec:>6}")
    print("-" * W)
    avg_faith = f"{averages['faithfulness']:.2f}"    if averages["faithfulness"]    is not None else "  N/A"
    avg_relev = f"{averages['answer_relevancy']:.2f}" if averages["answer_relevancy"] is not None else "  N/A"
    avg_prec  = f"{averages['context_precision']:.2f}" if averages["context_precision"] is not None else "  N/A"
    print(f"{'':>2}  {'AVERAGE':<44} {avg_faith:>6} {avg_relev:>6} {avg_prec:>6}")
    print("=" * W)

    # ── Save JSON ──────────────────────────────────────────────────
    output = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "eval_model":        EVAL_MODEL,
        "total_questions":   len(QUESTIONS),
        "metrics_evaluated": METRIC_NAMES,
        "results":           per_question,
        "averages":          averages,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()
