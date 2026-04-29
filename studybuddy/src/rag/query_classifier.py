"""
studybuddy/src/rag/query_classifier.py

Classifies incoming student queries before any retrieval or generation
occurs. Enforces the Academic Integrity Layer from the Master System
Instruction by detecting direct homework/exam answer requests.

Public API:
  classify_query(query_text)  -> ClassificationResult
"""

import re
from dataclasses import dataclass
from enum import Enum


class QueryClass(Enum):
    ADMINISTRATIVE        = "administrative"
    CONCEPTUAL_HELP       = "conceptual_help"
    DIRECT_ANSWER_REQUEST = "direct_answer_request"
    KB_METADATA           = "kb_metadata"       # questions about the KB itself
    CONVERSATIONAL        = "conversational"    # greetings / small talk


@dataclass
class ClassificationResult:
    query_class:      QueryClass
    confidence:       float
    matched_pattern:  str
    original_query:   str
    sanitized_query:  str


# ── KB metadata patterns ──────────────────────────────────────────────────────
# Catches any query asking what documents/files the system has indexed.
# These are answered directly from ChromaDB metadata — never via RAG —
# so the LLM can never hallucinate a wrong PDF count or filename.
KB_METADATA_PATTERNS = [
    re.compile(r'\bhow many (pdfs?|documents?|files?|sources?|topics?|books?|slides?)\b', re.IGNORECASE),
    re.compile(r'\bwhat (pdfs?|documents?|files?|sources?) (do you|have you|are (you|there))', re.IGNORECASE),
    re.compile(r'\blist (the |all |your )?(pdfs?|documents?|files?|sources?|course materials?)', re.IGNORECASE),
    re.compile(r'\bwhat (course materials?|content|topics?) (do you|have you|are|did you)', re.IGNORECASE),
    re.compile(r'\bwhat (do you|can you) (know|see|access|find|read)\b', re.IGNORECASE),
    re.compile(r'\bwhat (is|are) (in|inside|indexed in) (your|the) (knowledge base|database|kb|vector store)\b', re.IGNORECASE),
    re.compile(r'\bwhat (pdfs?|documents?|books?|slides?|lectures?) (have you|do you) (indexed|loaded|uploaded|available)\b', re.IGNORECASE),
    re.compile(r'\bwhich (pdfs?|documents?|files?|sources?) (are|have been) (indexed|loaded|available)\b', re.IGNORECASE),
    re.compile(r'\bcan you see (all|my|the) (pdfs?|documents?|files?|materials?)\b', re.IGNORECASE),
    re.compile(r'\bhow many (pdf|doc|file|source)\b', re.IGNORECASE),
]

CONVERSATIONAL_PATTERNS = [
    re.compile(r'^(hi|hello|hey|howdy|hiya|yo|sup)\b',                          re.IGNORECASE),
    re.compile(r'^good\s+(morning|afternoon|evening|day)\b',                    re.IGNORECASE),
    re.compile(r'^(thanks?|thank you|cheers|ty)\b',                             re.IGNORECASE),
    re.compile(r'^how are you\b',                                                re.IGNORECASE),
    re.compile(r'^(who are you|what (are|can) you do|tell me about yourself)\b', re.IGNORECASE),
    re.compile(r'^(help|help me)\s*[.!?]?\s*$',                                  re.IGNORECASE),
    re.compile(r'^(great|awesome|cool|nice|ok|okay|got it|sounds good)\b',      re.IGNORECASE),
    re.compile(r'^(bye|goodbye|see you|later|cya)\b',                           re.IGNORECASE),
]

ADMINISTRATIVE_PATTERNS = [
    re.compile(r'\bwhen\b.*(exam|quiz|test|due|deadline|submission|office hour)', re.IGNORECASE),
    re.compile(r'\bwhere\b.*(exam|class|lecture|office)', re.IGNORECASE),
    re.compile(r'\bwhat time\b', re.IGNORECASE),
    re.compile(r'\bhow (many|much).*(point|mark|grade|credit|weight)', re.IGNORECASE),
    re.compile(r'\bis (there|the).*(exam|quiz|class|lecture|session)\b', re.IGNORECASE),
    re.compile(r'\bsyllabus\b', re.IGNORECASE),
    re.compile(r'\boffice hours?\b', re.IGNORECASE),
    re.compile(r'\bgrading (policy|scheme|rubric)\b', re.IGNORECASE),
]

DIRECT_ANSWER_PATTERNS = [
    re.compile(r'\bsolve (this|the|for)\b', re.IGNORECASE),
    re.compile(r'\bgive me (the |a )?(answer|solution|code|result)\b', re.IGNORECASE),
    re.compile(r'\bwhat is the answer (to|for)\b', re.IGNORECASE),
    re.compile(r'\bcomplete (this|the|my) (assignment|homework|problem|question|task|exercise)\b', re.IGNORECASE),
    re.compile(r'\bdo (this|my|the) (homework|assignment|problem|question|exercise)\b', re.IGNORECASE),
    re.compile(r'\bwrite (the|a|my) (code|solution|answer|essay|report) for\b', re.IGNORECASE),
    re.compile(r'\b(homework|assignment|graded|exam|quiz|problem set)\b.{0,40}\b(answer|solution|solve|complete|finish)\b', re.IGNORECASE),
    re.compile(r'\bjust (give|tell|show) me (the answer|the solution|the code)\b', re.IGNORECASE),
    # Pasted multiple-choice question: lines with A) B) C) D) options
    re.compile(r'\ba\)\s.+\n\s*b\)\s.+\n\s*c\)\s', re.IGNORECASE),
    re.compile(r'\ba\.\s.+\n\s*b\.\s.+\n\s*c\.\s', re.IGNORECASE),
]

CONCEPTUAL_HELP_PATTERNS = [
    re.compile(r'\bexplain\b', re.IGNORECASE),
    re.compile(r'\bwhat (is|are|does|do)\b', re.IGNORECASE),
    re.compile(r'\bhow (does|do|can|should)\b', re.IGNORECASE),
    re.compile(r'\bwhy (is|are|does|do|did)\b', re.IGNORECASE),
    re.compile(r'\bhelp me understand\b', re.IGNORECASE),
    re.compile(r'\bcan you (describe|summarize|clarify|walk me through)\b', re.IGNORECASE),
    re.compile(r'\bdifference between\b', re.IGNORECASE),
    re.compile(r'\bexample of\b', re.IGNORECASE),
    re.compile(r'\bdefine\b', re.IGNORECASE),
]


def _match_patterns(text: str, patterns: list) -> str:
    for pattern in patterns:
        if pattern.search(text):
            return pattern.pattern
    return ""


def classify_query(query_text: str) -> ClassificationResult:
    sanitized = query_text.lower().strip()

    if not sanitized:
        return ClassificationResult(
            query_class=QueryClass.CONCEPTUAL_HELP,
            confidence=0.5,
            matched_pattern="empty_input",
            original_query=query_text,
            sanitized_query=sanitized,
        )

    # Conversational / greeting — bypass RAG entirely
    conv_match = _match_patterns(sanitized, CONVERSATIONAL_PATTERNS)
    if conv_match:
        return ClassificationResult(
            query_class=QueryClass.CONVERSATIONAL,
            confidence=0.93,
            matched_pattern=conv_match,
            original_query=query_text,
            sanitized_query=sanitized,
        )

    # KB metadata checked next — must bypass RAG entirely
    kb_match = _match_patterns(sanitized, KB_METADATA_PATTERNS)
    if kb_match:
        return ClassificationResult(
            query_class=QueryClass.KB_METADATA,
            confidence=0.97,
            matched_pattern=kb_match,
            original_query=query_text,
            sanitized_query=sanitized,
        )

    direct_match = _match_patterns(sanitized, DIRECT_ANSWER_PATTERNS)
    if direct_match:
        return ClassificationResult(
            query_class=QueryClass.DIRECT_ANSWER_REQUEST,
            confidence=0.95,
            matched_pattern=direct_match,
            original_query=query_text,
            sanitized_query=sanitized,
        )

    admin_match = _match_patterns(sanitized, ADMINISTRATIVE_PATTERNS)
    if admin_match:
        return ClassificationResult(
            query_class=QueryClass.ADMINISTRATIVE,
            confidence=0.90,
            matched_pattern=admin_match,
            original_query=query_text,
            sanitized_query=sanitized,
        )

    conceptual_match = _match_patterns(sanitized, CONCEPTUAL_HELP_PATTERNS)
    if conceptual_match:
        return ClassificationResult(
            query_class=QueryClass.CONCEPTUAL_HELP,
            confidence=0.85,
            matched_pattern=conceptual_match,
            original_query=query_text,
            sanitized_query=sanitized,
        )

    return ClassificationResult(
        query_class=QueryClass.CONCEPTUAL_HELP,
        confidence=0.60,
        matched_pattern="no_match_fallback",
        original_query=query_text,
        sanitized_query=sanitized,
    )
