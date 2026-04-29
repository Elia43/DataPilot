"""
studybuddy/src/rag/context_builder.py

Assembles retrieved chunks into a structured context string and
citation list for injection into the Gemini prompt. Also builds
the Academic Integrity refusal payload for DIRECT_ANSWER_REQUEST
queries.

Public API:
  build_context(retrieved_chunks)         -> ContextPayload
  build_refusal_context(retrieved_chunks,
                        query_text)       -> ContextPayload
  build_empty_context()                   -> ContextPayload
  build_conversational_context()          -> ContextPayload
"""

from dataclasses import dataclass
from typing import List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from config import MAX_CONTEXT_CHUNKS
except ImportError:
    MAX_CONTEXT_CHUNKS = 3  # safe default if config unavailable

@dataclass
class Citation:
    source_file:    str
    page_number:    int | None
    timestamp:      str | None
    confidence_pct: int
    source_tier:    str

@dataclass
class ContextPayload:
    context_str:      str
    citations:        List[Citation]
    chunk_count:      int
    is_grounded:      bool
    is_refusal:       bool
    uncertainty_flag: bool

def _format_citation_tag(citation: Citation) -> str:
    # 3.5: If page_number is -1, treat as None
    pg = citation.page_number
    if pg == -1:
        pg = None
        
    ts = citation.timestamp
    
    if pg is None and ts is not None:
        return f"[Source: {citation.source_file}, Timestamp: {ts}, Confidence: {citation.confidence_pct}%]"
    elif pg is None and ts is None:
        return f"[Source: {citation.source_file}, Confidence: {citation.confidence_pct}%]"
    else:
        return f"[Source: {citation.source_file}, Page: {pg}, Confidence: {citation.confidence_pct}%]"

def _diversify_chunks(chunks: list[dict], max_count: int) -> list[dict]:
    """
    Select up to max_count chunks with guaranteed source-file diversity.

    Algorithm:
      1. Pick the single highest-scoring chunk from each unique source file.
      2. Fill remaining slots (up to max_count) from the leftover pool,
         still ranked by weighted_score.
      3. Re-sort the final selection by weighted_score so the generator
         always sees the most relevant material first.

    This prevents a single PDF from monopolising the context window when
    multiple source files are indexed — the root cause of the "I see only
    2 PDFs" hallucination when 4 are present.
    """
    if len(chunks) <= max_count:
        return chunks

    best_per_source: dict[str, dict] = {}
    remainder: list[dict] = []

    for chunk in chunks:   # chunks arrive sorted by weighted_score desc
        src = chunk.get("source_file", "__unknown__")
        if src not in best_per_source:
            best_per_source[src] = chunk
        else:
            remainder.append(chunk)

    selected = list(best_per_source.values())

    for chunk in remainder:
        if len(selected) >= max_count:
            break
        selected.append(chunk)

    selected.sort(key=lambda c: c.get("weighted_score", 0.0), reverse=True)
    return selected[:max_count]


def build_context(retrieved_chunks: list[dict]) -> ContextPayload:
    if not retrieved_chunks:
        return build_empty_context()

    # Apply diversity-aware selection before capping to MAX_CONTEXT_CHUNKS.
    # _diversify_chunks guarantees at least one chunk per source file, then
    # fills the remaining budget with the highest-scoring chunks overall.
    retrieved_chunks = _diversify_chunks(retrieved_chunks, MAX_CONTEXT_CHUNKS)
        
    n = len(retrieved_chunks)
    context_blocks = []
    
    context_blocks.append(f"RETRIEVED CONTEXT ({n} sources):\n")
    
    citations = []
    
    for i, chunk in enumerate(retrieved_chunks, start=1):
        pg = chunk.get("page_number")
        ts = chunk.get("timestamp")
        
        if pg == -1:
            pg = None
            
        location_line = ""
        if pg is not None:
            location_line = f"Page: {pg}"
        elif ts is not None:
            location_line = f"Timestamp: {ts}"
            
        block = (
            f"--- SOURCE {i} ---\n"
            f"File: {chunk['source_file']}\n"
        )
        if location_line:
            block += f"{location_line}\n"
            
        block += (
            f"Confidence: {chunk['confidence_pct']}%\n"
            f"Tier: {chunk['source_tier']}\n\n"
            f"{chunk['chunk_text']}"
        )
        
        context_blocks.append(block)
        
        citations.append(Citation(
            source_file=chunk["source_file"],
            page_number=pg,
            timestamp=ts,
            confidence_pct=chunk["confidence_pct"],
            source_tier=chunk["source_tier"]
        ))
        
    context_str = "\n\n".join(context_blocks)
    
    return ContextPayload(
        context_str=context_str,
        citations=citations,
        chunk_count=n,
        is_grounded=True,
        is_refusal=False,
        uncertainty_flag=False
    )

def build_refusal_context(retrieved_chunks: list[dict], query_text: str) -> ContextPayload:
    retrieved_chunks = retrieved_chunks[:MAX_CONTEXT_CHUNKS]
    if retrieved_chunks:
        first_chunk = retrieved_chunks[0]
        pg = first_chunk.get("page_number")
        if pg == -1: pg = None
            
        cit = Citation(
            source_file=first_chunk["source_file"],
            page_number=pg,
            timestamp=first_chunk.get("timestamp"),
            confidence_pct=first_chunk["confidence_pct"],
            source_tier=first_chunk["source_tier"]
        )
        hint_source = _format_citation_tag(cit)
        citations = [cit]
        chunk_count = len(retrieved_chunks)
    else:
        hint_source = "[No specific source found — refer to course materials]"
        citations = []
        chunk_count = 0
        
    context_str = (
        "ACADEMIC INTEGRITY BLOCK\n\n"
        "The student has requested a direct answer to what appears to "
        "be a graded assignment or exam question.\n\n"
        "INSTRUCTION FOR GEMINI: Do NOT provide the final answer or "
        "solution. Instead:\n"
        "  1. Acknowledge the student's question warmly.\n"
        "  2. Provide ONE conceptual hint that points toward the "
        "approach or principle needed.\n"
        f"  3. Direct the student to this source: {hint_source}\n"
        "  4. Encourage independent problem-solving.\n\n"
        f"Original query: {query_text}\n"
    )
    
    return ContextPayload(
        context_str=context_str,
        citations=citations,
        chunk_count=chunk_count,
        is_grounded=chunk_count > 0,
        is_refusal=True,
        uncertainty_flag=False
    )

def build_empty_context() -> ContextPayload:
    """Used when RAG retrieval returns no results for a course-content query."""
    context_str = (
        "NO COURSE MATERIAL FOUND FOR THIS QUERY\n\n"
        "The knowledge base does not contain content directly relevant to this query. "
        "Acknowledge that you cannot find this specific topic in the loaded course materials. "
        "If you have general knowledge about the subject, share a brief overview and "
        "encourage the student to consult their uploaded materials or rephrase the question. "
        "Do NOT fabricate course-specific facts or citations."
    )
    return ContextPayload(
        context_str=context_str,
        citations=[],
        chunk_count=0,
        is_grounded=False,
        is_refusal=False,
        uncertainty_flag=False,
    )


def build_conversational_context() -> ContextPayload:
    """Used for greetings and casual messages that need no PDF retrieval."""
    context_str = (
        "CONVERSATIONAL MESSAGE — NO PDF CONTEXT NEEDED\n\n"
        "The student is sending a greeting or casual message. "
        "Respond warmly and naturally. Introduce yourself briefly as DataPilot, "
        "an AI-powered Virtual Teaching Assistant. You can invite them to ask "
        "course questions, generate a quiz, or explore their study materials. "
        "Keep the response short, friendly, and encouraging. "
        "Do NOT use the citation format for this response."
    )
    return ContextPayload(
        context_str=context_str,
        citations=[],
        chunk_count=0,
        is_grounded=False,
        is_refusal=False,
        uncertainty_flag=False,
    )
