"""
studybuddy/src/rag/generator.py

Gemini response generation with exponential backoff retry.
Handles 503, 429, and 500 errors gracefully.

Public API:
  generate_response(query_text, context_payload,
                    classification_result, model,
                    conversation_history=None) -> GeneratorResponse
"""

import sys
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.rag.context_builder import ContextPayload, Citation
from src.rag.query_classifier import ClassificationResult, QueryClass
from google.genai import types

try:
    from config import MASTER_SYSTEM_INSTRUCTION
except ImportError:
    MASTER_SYSTEM_INSTRUCTION = (
        "You are DataPilot, a helpful Virtual Teaching Assistant. "
        "Answer only from provided context. Always include a "
        "Reasoning Trace. Never solve homework directly. "
        "Use bullet points and bold text for all responses."
    )

MAX_RETRIES    = 3
BASE_DELAY_SEC = 2.0


@dataclass
class GeneratorResponse:
    answer:          str
    reasoning_trace: str
    citations:       List[Citation]
    was_refused:     bool
    was_uncertain:   bool
    query_class:     str
    confidence:      float


def _build_prompt(
    query_text: str,
    context_payload: ContextPayload,
    classification_result: ClassificationResult,
    conversation_history: list = None,
) -> str:
    prompt = ""

    if conversation_history:
        recent      = conversation_history[-3:]
        history_str = ""
        for turn in recent:
            role    = "Student" if turn["role"] == "user" \
                      else "DataPilot"
            content = turn["content"][:300]
            history_str += f"{role}: {content}\n"
        prompt += (
            "=== RECENT CONVERSATION ===\n"
            "Use this to resolve follow-up references "
            "like 'them', 'it', 'explain more':\n\n"
            f"{history_str}\n\n"
        )

    prompt += (
        "=== QUERY CLASSIFICATION ===\n"
        f"Class: {classification_result.query_class.value}\n"
        f"Confidence: {classification_result.confidence}\n"
        f"Matched Pattern: {classification_result.matched_pattern}\n\n"
    )
    prompt += context_payload.context_str + "\n\n"
    prompt += (
        "=== STUDENT QUERY ===\n"
        f"{query_text}\n\n"
        "=== RESPONSE INSTRUCTIONS ===\n"
        "Respond using ONLY the context above. Follow ALL rules "
        "in your system instructions.\n\n"
        "MANDATORY FORMAT:\n"
        "1. One-sentence direct answer first\n"
        "2. Bullet points for any list of 3+ items\n"
        "3. **Bold** all key terms and technology names\n"
        "4. After EVERY factual claim, append an inline citation in the\n"
        "   form `[filename, p.X]` — e.g. `[HDFS_Lecture.pdf, p.4]`.\n"
        "   Use the source_file and page_number from the context chunks.\n"
        "   Do NOT omit inline citations — every sentence that states a\n"
        "   fact from the context MUST end with one.\n"
        "5. Reasoning Trace: section at the end\n"
        "6. Trailing summary: Citation: [Source: filename, Page: X, Confidence: Y%]\n"
    )
    return prompt


def _parse_reasoning_trace(response_text: str):
    lower = response_text.lower()
    idx   = lower.find("reasoning trace:")
    if idx != -1:
        answer          = response_text[:idx].strip()
        reasoning_trace = response_text[
            idx + len("reasoning trace:"):
        ].strip()
    else:
        answer          = response_text.strip()
        reasoning_trace = "No explicit reasoning trace provided."
    return answer, reasoning_trace


def generate_response(
    query_text: str,
    context_payload: ContextPayload,
    classification_result: ClassificationResult,
    model: dict,
    conversation_history: list = None,
) -> GeneratorResponse:
    """
    model dict: {"client": genai.Client, "model_name": str,
                 "fallback_model": str}
    """
    if context_payload.uncertainty_flag:
        return GeneratorResponse(
            answer=(
                "I am not sure — the course knowledge base does not "
                "contain information relevant to this query."
            ),
            reasoning_trace="No relevant sources found.",
            citations       = [],
            was_refused     = False,
            was_uncertain   = True,
            query_class     = classification_result.query_class.value,
            confidence      = classification_result.confidence,
        )

    prompt_str = _build_prompt(
        query_text, context_payload, classification_result,
        conversation_history=conversation_history
    )

    client     = model["client"]
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        # Use fallback model on final attempt
        if attempt == MAX_RETRIES and "fallback_model" in model:
            model_name = model["fallback_model"]
        else:
            model_name = model["model_name"]

        try:
            response = client.models.generate_content(
                model    = model_name,
                contents = prompt_str,
                config   = types.GenerateContentConfig(
                    system_instruction = MASTER_SYSTEM_INSTRUCTION
                ),
            )
            raw_text = response.text
            break

        except Exception as e:
            last_error   = e
            error_str    = str(e).lower()
            is_retryable = any(code in error_str for code in [
                "503", "unavailable", "resource_exhausted",
                "429", "internal", "500", "overloaded"
            ])

            if is_retryable and attempt < MAX_RETRIES:
                delay  = BASE_DELAY_SEC * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, 1.0)
                time.sleep(delay + jitter)
                continue
            else:
                raise RuntimeError(
                    f"Gemini generation failed after {attempt} "
                    f"attempt(s): {e}"
                ) from e
    else:
        raise RuntimeError(
            f"Gemini generation failed after {MAX_RETRIES} "
            f"retries: {last_error}"
        )

    answer, reasoning_trace = _parse_reasoning_trace(raw_text)

    return GeneratorResponse(
        answer          = answer,
        reasoning_trace = reasoning_trace,
        citations       = context_payload.citations,
        was_refused     = context_payload.is_refusal,
        was_uncertain   = False,
        query_class     = classification_result.query_class.value,
        confidence      = classification_result.confidence,
    )