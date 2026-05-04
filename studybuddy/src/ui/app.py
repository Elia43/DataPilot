"""
studybuddy/src/ui/app.py

Streamlit multi-page web UI for DataPilot Virtual TA.
Four pages: Chat, Quiz, Dashboard, History.

Run with:
  studybuddy\.venv\Scripts\streamlit.exe run studybuddy/src/ui/app.py
"""

import sys
import os
import re
import time
import uuid
import random
import concurrent.futures
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

import streamlit as st

import pandas as pd
import plotly.graph_objects as go

_UI_FILE  = Path(__file__).resolve()
_SRC_DIR  = _UI_FILE.parent.parent
_PKG_ROOT = _SRC_DIR.parent
sys.path.insert(0, str(_PKG_ROOT))

# PDFs live one level above the package root (Studybuddy/data/raw/)
_RAW_DIR       = _PKG_ROOT.parent / "data" / "raw"
_PROCESSED_DIR = _PKG_ROOT / "data" / "processed"

st.set_page_config(
    page_title="DataPilot — Virtual TA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import (
    initialize_model as initialize_gemini_model,
    SIMILARITY_THRESHOLD,
    TOP_K_RESULTS,
    CHROMA_DB_PATH,
)
from src.rag.query_classifier import classify_query, QueryClass
from src.rag.retriever import retrieve
from src.rag.context_builder import (
    build_context,
    build_refusal_context,
    build_empty_context,
    build_conversational_context,
)
from src.rag.generator import generate_response
from src.vectorstore.chroma_store import get_collection, get_collection_stats
from src.vectorstore.embedder import embed_chunks
from src.ingestion.pdf_parser import parse_pdf, load_parsed_chunks

try:
    from src.db.auth import register_user, login_user
    from src.db.quiz_store import (
        save_attempt, delete_user_attempts,
        get_topic_mastery, get_attempt_question_details,
    )
    _DB_AVAILABLE = True
except Exception as _e:
    print(f"[DataPilot] WARNING: auth/quiz_store import failed — DB features disabled. Reason: {_e}")
    _DB_AVAILABLE = False

try:
    from src.db.chat_store import (
        upsert_active_session,
        finalize_session,
        get_active_session,
        get_chat_sessions,
    )
    _CHAT_DB_AVAILABLE = True
except Exception as _e:
    print(f"[DataPilot] WARNING: chat_store import failed — chat persistence disabled. Reason: {_e}")
    _CHAT_DB_AVAILABLE = False

try:
    from src.db.weak_store import (
        save_weak_interactions,
        get_chapter_mastery,
        get_weak_sources,
        get_weak_interaction_texts,
        get_weak_interactions,
        delete_weak_interactions,
    )
    _WEAK_DB_AVAILABLE = True
except Exception as _e:
    print(f"[DataPilot] WARNING: weak_store import failed — flashcard features disabled. Reason: {_e}")
    _WEAK_DB_AVAILABLE = False

try:
    from studybuddy.src.ui.admin_panel import render as _render_admin_panel
    from studybuddy.src.db.mongo_client import is_admin as _is_admin
    _ADMIN_AVAILABLE = True
except Exception as _e:
    import traceback
    traceback.print_exc()
    _ADMIN_AVAILABLE = False


@st.cache_resource(show_spinner=False)
def _get_chroma_collection(db_path: str):
    """Load the ChromaDB PersistentClient once per server process."""
    return get_collection(db_path, "studybuddy_knowledge_base")


@st.cache_data(ttl=300, show_spinner=False)
def _load_user_data(username: str) -> tuple:
    """Cache quiz history + stats for 5 minutes. Call .clear() after save_attempt."""
    if not _DB_AVAILABLE:
        return [], {"total": 0, "correct": 0, "percentage": 0.0}
    from src.db.quiz_store import get_history_and_stats
    return get_history_and_stats(username)


@st.cache_data(ttl=300, show_spinner=False)
def _load_difficulty_stats(username: str) -> dict:
    """Cache per-difficulty mastery for 5 minutes. Call .clear() after save_attempt."""
    if not _DB_AVAILABLE:
        return {}
    from src.db.quiz_store import get_difficulty_stats
    return get_difficulty_stats(username)


@st.cache_data(ttl=300, show_spinner=False)
def _load_chapter_mastery(username: str) -> dict:
    """Cache chapter mastery for 5 minutes. Call .clear() after save_attempt."""
    if not _WEAK_DB_AVAILABLE:
        return {}
    return get_chapter_mastery(username)


@st.cache_data(ttl=300, show_spinner=False)
def _load_topic_mastery(username: str) -> dict:
    """Cache per-topic mastery for 5 minutes. Call .clear() after save_attempt."""
    if not _DB_AVAILABLE:
        return {}
    return get_topic_mastery(username)


@st.cache_data(ttl=300, show_spinner=False)
def _load_flashcards(username: str) -> list:
    """Cache wrong-answer flashcard records for 5 minutes."""
    if not _WEAK_DB_AVAILABLE:
        return []
    return get_weak_interactions(username)


# ══════════════════════════════════════════════════════════════════
# EXAM TIMER
# ══════════════════════════════════════════════════════════════════

def _render_timer_widget(deadline_epoch: float) -> None:
    """
    Render a self-ticking JS countdown timer inside a real iframe.

    Accepts deadline_epoch (Unix timestamp = start_time + total_secs) so the
    JS computes remaining time from Date.now() on every tick.  This means
    Streamlit reruns — triggered by radio selections, button clicks, etc. —
    never reset the counter: the new iframe always re-derives the correct
    remaining time from the same absolute deadline.

    Uses st.iframe() so the <script> runs inside a sandboxed iframe where
    window.parent access is permitted by Streamlit's same-origin policy,
    allowing the auto-submit DOM click to reach the form.
    """
    html = f"""
<div id="sb-timer" style="
    font-family:monospace;font-size:1.9rem;font-weight:700;
    color:#2ecc71;padding:4px 14px;border:2px solid #2ecc71;
    border-radius:8px;display:inline-block;letter-spacing:2px;
    background:rgba(0,0,0,0.04);">
⏱ --:--
</div>
<script>
(function(){{
    var deadline = {deadline_epoch};
    var el = document.getElementById('sb-timer');
    var submitted = false;

    function fmt(n){{ return n < 10 ? '0'+n : ''+n; }}

    function setColor(r){{
        var c = r < 60 ? '#e74c3c' : r < 120 ? '#f39c12' : '#2ecc71';
        el.style.color = c;
        el.style.borderColor = c;
    }}

    function autoSubmit(){{
        if (submitted) return;
        submitted = true;
        var p = window.parent.document;
        var btn = p.querySelector('[data-testid="stFormSubmitButton"] button')
               || p.querySelector('button[kind="primaryFormSubmit"]')
               || p.querySelector('button[type="submit"]');
        if (btn) btn.click();
    }}

    function tick(){{
        var rem = Math.max(0, Math.round(deadline - Date.now() / 1000));
        if (rem <= 0){{
            el.innerHTML = '⏱ TIME UP!';
            el.style.color = '#e74c3c';
            el.style.borderColor = '#e74c3c';
            autoSubmit();
            return;
        }}
        el.innerHTML = '⏱ ' + fmt(Math.floor(rem / 60)) + ':' + fmt(rem % 60);
        setColor(rem);
        setTimeout(tick, 500);
    }}

    tick();
}})();
</script>"""
    st.iframe(html, height=70)


# ══════════════════════════════════════════════════════════════════
# QUIZ CONFIG
# ══════════════════════════════════════════════════════════════════

DEFAULT_NUM_QUESTIONS  = 5
QUIZ_TOP_K             = 6
MIN_QUIZ_CONTEXT_CHARS = 200

_DIFFICULTY_CONFIGS = {
    "Easy": {
        "focus":        "definition and recall questions — 'What is X?', 'Which term describes Y?'",
        "option_rule":  "Each option must be a short phrase of 3–6 words.",
        "distractor":   "simple wrong answers that are clearly distinct from the correct one",
    },
    "Medium": {
        "focus":        "conceptual understanding, scenario-based, and comparison questions",
        "option_rule":  "Each option must be a concise phrase of 5–10 words.",
        "distractor":   "plausible distractors that require real understanding to distinguish",
    },
    "Hard": {
        "focus":        "application, analysis, and edge-case questions requiring multi-step reasoning",
        "option_rule":  "Each option must be a concise phrase of 5–12 words.",
        "distractor":   "highly plausible distractors with subtle but critical differences from the correct answer",
    },
}


# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════

def _inject_css():
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        /* Scrollbar styling for the sources column (applied via JS to the column element) */
        [data-testid="column"]::-webkit-scrollbar        { width: 4px; }
        [data-testid="column"]::-webkit-scrollbar-thumb  {
            background: rgba(0,0,0,0.15); border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# STRUCTURED QUIZ DATA MODEL
# ══════════════════════════════════════════════════════════════════

@dataclass
class QuizQuestion:
    number:  int
    text:    str
    options: dict          # {"A": "...", "B": "...", ...}
    correct: str           # "A" / "B" / "C" / "D"


@dataclass
class Quiz:
    questions:    List[QuizQuestion]
    topic:        str
    citations:    list
    difficulty:   str = "Medium"
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def render_for_student(self) -> str:
        blocks = []
        for q in self.questions:
            block = f"**Q{q.number}.** {q.text}\n\n"
            for letter in ("A", "B", "C", "D"):
                if letter in q.options:
                    block += f"**{letter})** {q.options[letter]}\n\n"
            blocks.append(block.rstrip())
        return "\n\n---\n\n".join(blocks)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════

def _init_session_state():
    defaults = {
        "messages":              [],
        "model":                 None,
        "collection":            None,
        "ready":                 False,
        "api_key_set":           False,
        "last_response_meta":    None,
        "_pending_query":        None,
        "_ingesting":            False,
        "student_name":          "",
        "session_started":       False,
        "logged_in":             False,
        # Navigation
        "page":                  "Chat",
        # Chat persistence
        "chat_session_id":       str(uuid.uuid4()),
        # Quiz system
        "quiz_mode":             False,
        "current_quiz":          None,
        "quiz_results":          None,
        "quiz_report_generated": False,
        "quiz_input_counter":    0,
        "is_admin":              False,
        "quiz_difficulty":       "Medium",
        "quiz_num_questions":    DEFAULT_NUM_QUESTIONS,
        "quiz_source_filter":    [],   # empty = all sources
        "_quiz_auto_generate":   False,
        # Analytics
        "quiz_total":            0,
        "quiz_correct":          0,
        "quiz_history":          [],
        # Exam / timed mode
        "quiz_timer_minutes":    0,
        "quiz_start_time":       None,
        "_timer_force_submit":   False,
        # Exam Crunch Mode
        "exam_crunch_mode":      False,
        # Rate limiting: timestamps of recent quiz generation calls
        "_quiz_gen_timestamps":  [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.api_key_set:
        try:
            st.session_state.model       = initialize_gemini_model()
            st.session_state.api_key_set = True
            from config import load_api_key
            os.environ["GEMINI_API_KEY"] = load_api_key()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
# GEMINI CLIENT HELPERS
# ══════════════════════════════════════════════════════════════════

def _get_gemini_model():
    model_obj = st.session_state.get("model")
    if model_obj is None:
        model_obj = initialize_gemini_model()
        st.session_state.model = model_obj
    if isinstance(model_obj, dict):
        return model_obj.get("client") or next(iter(model_obj.values()))
    return model_obj


_GEMINI_TIMEOUT_SECS = 45
_GEMINI_MAX_RETRIES  = 3   # 1 initial attempt + up to 3 retries on 503/429


def _classify_gemini_error(exc: Exception) -> str:
    """Return 'server_busy', 'rate_limit', 'timeout', or 'other'."""
    msg = str(exc).lower()
    if any(k in msg for k in ("503", "unavailable", "service unavailable", "overloaded")):
        return "server_busy"
    if any(k in msg for k in ("429", "resource_exhausted", "quota", "rate_limit", "rate limit")):
        return "rate_limit"
    if isinstance(exc, TimeoutError) or "timeout" in msg:
        return "timeout"
    return "other"


def _call_gemini(prompt: str, on_retry=None) -> str:
    """
    Call Gemini with exponential backoff for 503/429 errors.

    on_retry(attempt, max_retries, wait_secs, err_type) is called before each
    sleep so the UI can update a status widget with retry progress.
    """
    model = _get_gemini_model()

    def _invoke() -> str:
        if hasattr(model, "models"):
            from config import GEMINI_CHAT_MODEL
            return model.models.generate_content(
                model=GEMINI_CHAT_MODEL, contents=prompt
            ).text
        if hasattr(model, "generate_content"):
            return model.generate_content(prompt).text
        if callable(model):
            response = model(prompt)
            return getattr(response, "text", str(response))
        raise TypeError(f"Unsupported Gemini model type: {type(model)}")

    for attempt in range(_GEMINI_MAX_RETRIES + 1):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_invoke)
            try:
                return future.result(timeout=_GEMINI_TIMEOUT_SECS)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Gemini did not respond within {_GEMINI_TIMEOUT_SECS}s. "
                    "Check your network and try again."
                )
            except Exception as exc:
                err_type = _classify_gemini_error(exc)
                if err_type not in ("server_busy", "rate_limit") or attempt >= _GEMINI_MAX_RETRIES:
                    raise
                wait = 2 ** (attempt + 1)   # 2 s, 4 s, 8 s
                if on_retry is not None:
                    on_retry(attempt + 1, _GEMINI_MAX_RETRIES, wait, err_type)
                time.sleep(wait)
    raise RuntimeError("Retry loop exhausted without raising — unreachable")


# ══════════════════════════════════════════════════════════════════
# QUIZ: TOPIC SELECTION
# ══════════════════════════════════════════════════════════════════

def _select_quiz_topic() -> str:
    for msg in reversed(st.session_state.get("messages", [])):
        if msg.get("role") == "user" and len(msg.get("content", "")) > 15:
            return msg["content"]

    try:
        collection = st.session_state.collection
        if collection is not None:
            sample = collection.peek(limit=20)
            docs   = sample.get("documents", [])
            if docs:
                chunk          = random.choice(docs)
                first_sentence = chunk.split(".")[0][:120]
                first_sentence = re.sub(r"[\d\s]+$", "", first_sentence).strip()
                if first_sentence:
                    return first_sentence
    except Exception:
        pass

    return "core concepts from the course"


# ══════════════════════════════════════════════════════════════════
# QUIZ: EDUCATIONAL TOPIC EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _extract_educational_topic(context_str: str, fallback: str) -> str:
    """
    Ask Gemini to derive a clean academic title from the retrieved context.

    Why: _select_quiz_topic() returns the user's last chat message verbatim
    (e.g. "how many pdfs can you see?"). That string is then stored as the
    quiz topic in MongoDB and shown in History/Dashboard — completely
    meaningless educationally. This function replaces it with a proper
    subject title like "HDFS Data Replication and Fault Tolerance".

    Falls back to the raw user query (truncated) if Gemini fails or
    returns something that looks like a question / is too long.
    """
    if not context_str or len(context_str) < 80:
        return fallback[:80]

    prompt = (
        "You are generating a quiz title for an educational tracking system.\n"
        "Based ONLY on the course content below, write a concise academic "
        "subject title of 4–8 words.\n"
        "Rules:\n"
        "- Describe the SUBJECT MATTER being tested, not the student's question\n"
        "- Do NOT start with a verb (no 'Explain', 'Describe', 'What is')\n"
        "- Do NOT end with a question mark\n"
        "- Output ONLY the title — no quotes, no punctuation at the end\n\n"
        "Good examples: HDFS Block Replication Strategy, "
        "MapReduce Shuffle and Sort Phase, Big Data Volume Characteristics\n\n"
        f"Course content:\n{context_str[:1200]}"
    )
    try:
        title = _call_gemini(prompt).strip().strip('"').rstrip(".")
        if len(title) > 90 or "?" in title or len(title) < 5:
            return fallback[:80]
        return title
    except Exception:
        return fallback[:80]


# ══════════════════════════════════════════════════════════════════
# QUIZ: GENERATION EXCEPTION HIERARCHY
# ══════════════════════════════════════════════════════════════════

class QuizGenerationError(Exception):
    """Base class for all quiz generation failures."""

class QuizAPIError(QuizGenerationError):
    """Gemini returned 503/429 — server busy or rate limited."""
    def __init__(self, original_exc: Exception, err_type: str):
        super().__init__(str(original_exc))
        self.err_type = err_type   # "server_busy" | "rate_limit" | "other"
        self.original = original_exc

class QuizNoContextError(QuizGenerationError):
    """Retrieval returned too little text to ground a quiz."""

class QuizParseError(QuizGenerationError):
    """Gemini returned output that could not be parsed into questions."""


# ══════════════════════════════════════════════════════════════════
# QUIZ: GENERATION (RAG-GROUNDED)
# ══════════════════════════════════════════════════════════════════

def _parse_quiz_response(raw_text: str) -> List[QuizQuestion]:
    questions: List[QuizQuestion] = []
    blocks = re.split(r"\n(?=\s*\d+\.\s)", "\n" + raw_text.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        m_q = re.match(r"^\s*(\d+)\.\s*(.+?)(?=\n\s*[A-D]\))", block, re.S)
        if not m_q:
            continue
        q_num  = int(m_q.group(1))
        q_text = m_q.group(2).strip().replace("\n", " ")

        options = {}
        for letter in ("A", "B", "C", "D"):
            m_opt = re.search(
                rf"^\s*{letter}\)\s*(.+?)(?=\n\s*[A-D]\)|\n?\s*Answer:|\Z)",
                block, re.M | re.S
            )
            if m_opt:
                opt_text = m_opt.group(1).strip().replace("\n", " ")
                opt_text = re.sub(r"\s*Answer:\s*[A-D]\b.*$", "", opt_text, flags=re.I).strip()
                options[letter] = opt_text

        m_ans = re.search(r"Answer:\s*([A-D])", block, re.I)
        correct = m_ans.group(1).upper() if m_ans else ""

        if q_text and len(options) == 4 and correct in options:
            questions.append(QuizQuestion(
                number=q_num, text=q_text,
                options=options, correct=correct,
            ))

    return questions


def _generate_quiz_via_rag(
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    difficulty: str = "Medium",
    source_filter: list[str] | None = None,
    on_retry=None,
    exam_crunch: bool = False,
    topic_override: str | None = None,
) -> Quiz:
    """Raises QuizGenerationError subclasses instead of returning None."""
    collection = st.session_state.collection
    if collection is None:
        raise QuizNoContextError("No ChromaDB collection is loaded.")

    # Exam Crunch scans every indexed PDF for maximum coverage
    if exam_crunch:
        source_filter = None

    api_key = os.environ.get("GEMINI_API_KEY", "")

    # Exam Crunch uses a broad, high-yield query so retrieval doesn't depend on
    # whatever the student's last chat message happened to be.
    if topic_override:
        user_query = topic_override
    elif exam_crunch:
        user_query = (
            "key definitions core concepts architecture components "
            "comparison advantages disadvantages"
        )
    else:
        user_query = _select_quiz_topic()

    dcfg = _DIFFICULTY_CONFIGS.get(difficulty, _DIFFICULTY_CONFIGS["Medium"])

    # Exam Crunch pulls twice as many chunks to maximise content coverage.
    _quiz_top_k = QUIZ_TOP_K * 2 if exam_crunch else QUIZ_TOP_K

    try:
        retrieved = retrieve(
            user_query, collection, api_key,
            _quiz_top_k, SIMILARITY_THRESHOLD,
            source_filter=source_filter or None,
        )
    except Exception:
        retrieved = []

    if retrieved:
        context_str = "\n\n".join(c["chunk_text"] for c in retrieved)
        citations   = build_context(retrieved).citations
    else:
        if source_filter:
            raise QuizNoContextError(
                "No content matched the selected source(s). "
                "Try selecting more sources or deselecting filters."
            )
        # Peek more documents in Exam Crunch mode for a wider content base.
        _peek_limit = 30 if exam_crunch else 10
        _peek_slice = 15 if exam_crunch else 5
        try:
            sample      = collection.peek(limit=_peek_limit)
            docs        = sample.get("documents", [])
            context_str = "\n\n".join(docs[:_peek_slice]) if docs else ""
        except Exception:
            context_str = ""
        citations = []

    if len(context_str) < MIN_QUIZ_CONTEXT_CHARS:
        if exam_crunch and context_str:
            # Partial content: tell Gemini it may supplement from general knowledge
            # rather than throwing a hard failure — the exam crunch prompt already
            # focuses on high-yield fundamentals so this is safe.
            context_str += (
                f"\n\n[Only {len(context_str)} chars of specific course content found. "
                "Generate questions from the concepts above and supplement with "
                "general course knowledge to reach the full question count.]"
            )
        else:
            raise QuizNoContextError(
                "The knowledge base doesn't contain enough content to generate a quiz. "
                "Try ingesting more PDFs."
            )

    # Derive a clean academic topic title from the context — not the raw
    # user query — so History/Dashboard show "HDFS Block Replication" not
    # "how many pdfs can you see?".
    topic = _extract_educational_topic(context_str, user_query)

    # Build optional exam-crunch and remediation sections
    _exam_crunch_section = ""
    if exam_crunch:
        _exam_crunch_section = """
══ EXAM CRUNCH MODE — PRIORITIZE HIGH-YIELD CONTENT ══
The student has an upcoming exam. Focus exclusively on:
• Key DEFINITIONS of core terms ("What is X?" / "Which term describes Y?")
• Core ARCHITECTURES and their components and roles
• COMPARISON questions (advantages vs disadvantages, X vs Y trade-offs)
• Cause-and-effect relationships between concepts
Skip obscure details — maximize exam-readiness with high-impact fundamentals.
"""

    _remediation_section = ""
    if topic_override:
        _remediation_section = f"""
══ REMEDIATION FOCUS ══
The student has repeatedly struggled with: {topic_override}
Every question must directly re-test and reinforce this specific concept.
"""

    prompt = f"""You are a senior university professor writing a high-stakes exam for an Advanced Big Data course.

TASK: Generate exactly {num_questions} multiple-choice questions at {difficulty.upper()} difficulty.
Focus on {dcfg['focus']}.

══ ABSOLUTE PROHIBITIONS — any violation makes the question unacceptable ══
• NEVER mention a page number, slide number, figure number, document name, or filename.
• NEVER write "According to the text…", "As stated in the document…", "The passage mentions…"
• NEVER ask what a document "mentions", "describes", or "states" — ask what a CONCEPT IS or DOES.
• {dcfg['option_rule']} NEVER write full sentences as answer options.

══ QUESTION QUALITY STANDARD ({difficulty} level) ══
Focus: {dcfg['focus']}.
Distractors must be {dcfg['distractor']}.
Every correct answer must be directly supported by the course material.

══ COURSE MATERIAL ══
{context_str}

══ TOPIC FOCUS ══
{topic}
{_exam_crunch_section}{_remediation_section}
══ OUTPUT FORMAT — follow exactly, zero deviations ══

1. <Question text>
A) <short phrase>
B) <short phrase>
C) <short phrase>
D) <short phrase>
Answer: <letter>

[Continue for all {num_questions} questions]

CRITICAL FORMAT RULES:
- "Answer:" MUST be on its own new line, immediately after option D.
- Do NOT embed the answer letter inside any option text.
- Do NOT add explanations, asterisks, or parentheses after the answer letter.
- Output ONLY the questions — zero preamble, zero introduction, zero closing remarks."""

    try:
        raw = _call_gemini(prompt, on_retry=on_retry)
    except Exception as exc:
        err_type = _classify_gemini_error(exc)
        if err_type in ("server_busy", "rate_limit"):
            raise QuizAPIError(exc, err_type) from exc
        raise QuizAPIError(exc, "other") from exc

    questions = _parse_quiz_response(raw)
    if len(questions) < 2:
        raise QuizParseError(
            "The AI returned a malformed response — not enough parseable questions."
        )

    return Quiz(questions=questions, topic=topic, citations=citations, difficulty=difficulty)


# ══════════════════════════════════════════════════════════════════
# QUIZ: ADAPTIVE GENERATION (WEAK-POINT TARGETED)
# ══════════════════════════════════════════════════════════════════

def _generate_adaptive_quiz(
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    difficulty: str = "Medium",
    on_retry=None,
) -> Quiz:
    """
    Generate a quiz that targets the student's previously failed topics.

    Strategy:
      1. Pull up to 20 recent incorrect question texts from MongoDB.
      2. Build a focused retrieval query from those texts.
      3. Restrict ChromaDB search to the source files where errors occurred.
      4. Generate a quiz that explicitly revisits the weak concepts.
    """
    if not _WEAK_DB_AVAILABLE or not st.session_state.get("logged_in"):
        raise QuizNoContextError(
            "Adaptive quizzes require you to be logged in with database access."
        )

    collection = st.session_state.collection
    if collection is None:
        raise QuizNoContextError("No ChromaDB collection is loaded.")

    username      = st.session_state.student_name
    weak_texts    = get_weak_interaction_texts(username, limit=20)
    weak_sources  = get_weak_sources(username)

    if not weak_texts:
        raise QuizNoContextError(
            "No weak-point data found yet. Complete some standard quizzes first, "
            "then come back for a targeted review."
        )

    api_key  = os.environ.get("GEMINI_API_KEY", "")
    dcfg     = _DIFFICULTY_CONFIGS.get(difficulty, _DIFFICULTY_CONFIGS["Medium"])

    # Build a dense retrieval query from the student's failed question texts
    seed_query = "Concepts related to: " + " | ".join(weak_texts[:8])

    try:
        retrieved = retrieve(
            seed_query, collection, api_key,
            QUIZ_TOP_K, SIMILARITY_THRESHOLD,
            source_filter=weak_sources or None,
        )
    except Exception:
        retrieved = []

    if not retrieved or len("\n\n".join(c["chunk_text"] for c in retrieved)) < MIN_QUIZ_CONTEXT_CHARS:
        raise QuizNoContextError(
            "Not enough content retrieved for your weak areas. "
            "Complete more standard quizzes to build up weak-point data."
        )

    context_str = "\n\n".join(c["chunk_text"] for c in retrieved)
    citations   = build_context(retrieved).citations
    topic       = _extract_educational_topic(context_str, "Weak Area Review")

    weak_summary = "; ".join(dict.fromkeys(t[:70] for t in weak_texts[:6]))

    dcfg = _DIFFICULTY_CONFIGS.get(difficulty, _DIFFICULTY_CONFIGS["Medium"])
    prompt = f"""You are a tutor creating a targeted remediation quiz.

The student previously struggled with these concepts:
{weak_summary}

TASK: Generate exactly {num_questions} multiple-choice questions at {difficulty.upper()} difficulty
that directly re-test and reinforce the weak areas listed above.
Focus on {dcfg['focus']}.

══ ABSOLUTE PROHIBITIONS ══
• NEVER mention a page number, slide number, figure number, document name, or filename.
• NEVER write "According to the text…" or "As stated in the document…"
• NEVER ask what a document "mentions" — ask what a CONCEPT IS or DOES.
• {dcfg['option_rule']} NEVER write full sentences as answer options.

══ COURSE MATERIAL ══
{context_str}

══ OUTPUT FORMAT — follow exactly ══

1. <Question text>
A) <short phrase>
B) <short phrase>
C) <short phrase>
D) <short phrase>
Answer: <letter>

[Continue for all {num_questions} questions]

CRITICAL: "Answer:" MUST be on its own new line immediately after option D.
Output ONLY the questions — zero preamble, zero closing remarks."""

    try:
        raw = _call_gemini(prompt, on_retry=on_retry)
    except Exception as exc:
        err_type = _classify_gemini_error(exc)
        if err_type in ("server_busy", "rate_limit"):
            raise QuizAPIError(exc, err_type) from exc
        raise QuizAPIError(exc, "other") from exc

    questions = _parse_quiz_response(raw)
    if len(questions) < 2:
        raise QuizParseError(
            "The AI returned a malformed response for the adaptive quiz."
        )

    return Quiz(
        questions=questions,
        topic=f"[Adaptive] {topic}",
        citations=citations,
        difficulty=difficulty,
    )


# ══════════════════════════════════════════════════════════════════
# QUIZ: DETERMINISTIC GRADING + EXPLANATIONS
# ══════════════════════════════════════════════════════════════════

def _parse_student_answers(raw: str, num_questions: int) -> dict:
    raw     = raw.strip().upper()
    answers = {}

    pairs = re.findall(r"(\d+)\s*[-.):]?\s*([A-D])\b", raw)
    if pairs:
        for num_str, letter in pairs:
            try:
                answers[int(num_str)] = letter
            except ValueError:
                continue
        if answers:
            return answers

    letters = re.findall(r"\b([A-D])\b", raw)
    for i, letter in enumerate(letters[:num_questions], start=1):
        answers[i] = letter

    return answers


def _grade_deterministically(quiz: Quiz, student_raw: str) -> dict:
    student_answers = _parse_student_answers(student_raw, len(quiz.questions))

    per_question  = []
    correct_count = 0

    for q in quiz.questions:
        student_letter = student_answers.get(q.number, "")
        is_correct     = (student_letter == q.correct)
        if is_correct:
            correct_count += 1
        per_question.append({
            "number":         q.number,
            "question":       q.text,
            "student_letter": student_letter,
            "student_text":   q.options.get(student_letter, "(not answered)"),
            "correct_letter": q.correct,
            "correct_text":   q.options.get(q.correct, ""),
            "is_correct":     is_correct,
        })

    total = len(quiz.questions)
    return {
        "per_question": per_question,
        "correct":      correct_count,
        "total":        total,
        "percentage":   (correct_count / total * 100) if total else 0.0,
        "raw_answers":  student_raw,
        "timestamp":    datetime.now().isoformat(),
        "_counted":     False,
    }


def _fetch_explanations(quiz: Quiz, per_question: list) -> dict:
    wrong_items = [p for p in per_question if not p["is_correct"]]
    if not wrong_items:
        return {}

    # Collect source filenames from quiz citations for attribution
    source_names: list[str] = []
    for cit in getattr(quiz, "citations", [])[:3]:
        sf = cit.get("source_file", "") if isinstance(cit, dict) else getattr(cit, "source_file", "")
        if sf and sf not in source_names:
            source_names.append(sf)

    source_block = ""
    if source_names:
        joined = ", ".join(source_names)
        source_block = (
            f"\nThis quiz was generated from these course materials: {joined}. "
            f"End each explanation with the most relevant source in brackets, "
            f"e.g. [Source: {source_names[0]}].\n"
        )

    items_str = "\n\n".join(
        f"Q{p['number']}. {p['question']}\n"
        f"Correct answer: {p['correct_letter']}) {p['correct_text']}"
        for p in wrong_items
    )

    source_fmt = " [Source: <filename>]" if source_names else ""
    prompt = (
        "You are a tutor explaining why specific answers are correct.\n"
        f"For each question below, write ONE concise sentence (max 30 words) "
        f"explaining why the given answer is correct.{source_block}\n"
        f"{items_str}\n\n"
        f"Format your response EXACTLY as:\n"
        f"Q<number>: <one-sentence explanation>{source_fmt}\n\n"
        "Provide exactly one line per question. No preamble."
    )

    try:
        raw = _call_gemini(prompt)
    except Exception:
        return {}

    explanations = {}
    for line in raw.split("\n"):
        m = re.match(r"\s*Q(\d+)\s*:\s*(.+)", line.strip())
        if m:
            explanations[int(m.group(1))] = m.group(2).strip()
    return explanations


def _render_grading_markdown(results: dict, explanations: dict) -> str:
    out = ["## 🎯 Quiz Results\n"]

    for p in results["per_question"]:
        symbol = "✅" if p["is_correct"] else "❌"
        out.append(f"**Q{p['number']}.** {p['question']}\n")
        if p["student_letter"]:
            out.append(
                f"{symbol} You answered **{p['student_letter']}**: "
                f"_{p['student_text']}_"
            )
        else:
            out.append(f"{symbol} _(not answered)_")

        if not p["is_correct"]:
            out.append(
                f"Correct answer: **{p['correct_letter']}**) "
                f"_{p['correct_text']}_"
            )
            expl = explanations.get(p["number"])
            if expl:
                out.append(f"💡 {expl}")
        out.append("")

    pct = results["percentage"]
    out.append("---")
    out.append(
        f"### Final Score: **{results['correct']} / {results['total']}** "
        f"({pct:.1f}%)"
    )
    if pct >= 80:
        out.append("🌟 Excellent work!")
    elif pct >= 60:
        out.append("📈 Good — review the explanations above.")
    else:
        out.append("📚 Let's keep studying — try asking about the topics you missed.")

    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════
# QUIZ UI COMPONENTS
# ══════════════════════════════════════════════════════════════════

def _render_quiz_interface():
    quiz: Optional[Quiz] = st.session_state.current_quiz
    if not quiz:
        return

    difficulty = st.session_state.get("quiz_difficulty", "Medium")
    st.markdown(f"### 🎯 Quiz — **{difficulty}** Difficulty")
    if quiz.topic and len(quiz.topic) < 150:
        st.caption(f"📖 Topic: _{quiz.topic}_")

    if not st.session_state.quiz_results:
        # ── Exam timer ─────────────────────────────────────────────
        _timer_mins  = st.session_state.get("quiz_timer_minutes", 0)
        _start_time  = st.session_state.get("quiz_start_time")
        _force_grade = st.session_state.get("_timer_force_submit", False)

        if _timer_mins > 0 and _start_time:
            _elapsed_secs   = int((datetime.now() - _start_time).total_seconds())
            _deadline_epoch = _start_time.timestamp() + _timer_mins * 60
            _render_timer_widget(_deadline_epoch)
            if _elapsed_secs >= _timer_mins * 60:
                st.session_state["_timer_force_submit"] = True
                _force_grade = True

        quiz_counter = st.session_state.quiz_input_counter

        with st.form(key=f"quiz_form_{quiz_counter}"):
            for q in quiz.questions:
                st.markdown(f"**Q{q.number}.** {q.text}")
                options = [
                    f"{letter}) {q.options[letter]}"
                    for letter in ("A", "B", "C", "D")
                    if letter in q.options
                ]
                st.radio(
                    label=f"Question {q.number}",
                    options=options,
                    index=None,
                    key=f"radio_q{q.number}_{quiz_counter}",
                    label_visibility="collapsed",
                )
                if q.number < len(quiz.questions):
                    st.markdown("---")

            st.markdown("")
            submitted = st.form_submit_button(
                "✅ Submit Quiz", type="primary", use_container_width=True
            )

        if submitted or _force_grade:
            st.session_state["_timer_force_submit"] = False
            if _force_grade and not submitted:
                st.info("⏱ Time's up! Your answers have been auto-submitted.")

            answer_parts = []
            for q in quiz.questions:
                val = st.session_state.get(f"radio_q{q.number}_{quiz_counter}")
                if val:
                    answer_parts.append(f"{q.number}{val[0]}")
            user_answer = " ".join(answer_parts)

            with st.spinner("Grading..."):
                results      = _grade_deterministically(quiz, user_answer)
                explanations = _fetch_explanations(quiz, results["per_question"])
                results["explanations"] = explanations
                if _DB_AVAILABLE and st.session_state.get("logged_in"):
                    try:
                        attempt_id = save_attempt(
                            st.session_state.student_name, quiz, results
                        )
                        _load_user_data.clear()
                        _load_difficulty_stats.clear()
                        _load_chapter_mastery.clear()
                        _load_topic_mastery.clear()
                    except Exception:
                        attempt_id = None
                    if _WEAK_DB_AVAILABLE and attempt_id:
                        try:
                            save_weak_interactions(
                                st.session_state.student_name,
                                attempt_id,
                                results["per_question"],
                                quiz,
                            )
                        except Exception:
                            pass

            st.session_state.quiz_results          = results
            st.session_state.quiz_report_generated = True
            st.session_state.quiz_input_counter   += 1
            st.rerun()

    if st.session_state.quiz_results:
        st.markdown("---")
        st.markdown(
            _render_grading_markdown(
                st.session_state.quiz_results,
                st.session_state.quiz_results.get("explanations", {}),
            )
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Quiz", use_container_width=True):
                st.session_state.quiz_mode             = False
                st.session_state.current_quiz          = None
                st.session_state.quiz_results          = None
                st.session_state.quiz_report_generated = False
                st.rerun()
        with col2:
            if st.button("📊 View Dashboard", use_container_width=True):
                st.session_state.page = "Dashboard"
                st.rerun()


def _render_quiz_report():
    if not (st.session_state.quiz_report_generated and st.session_state.quiz_results):
        return

    results  = st.session_state.quiz_results
    correct  = results["correct"]
    total    = results["total"]
    accuracy = results["percentage"]

    st.markdown("### 📈 **Session Report**")

    if not results.get("_counted", False):
        st.session_state.quiz_total   += total
        st.session_state.quiz_correct += correct
        st.session_state.quiz_history.append({
            "label":     f"Session {len(st.session_state.quiz_history) + 1}",
            "correct":   correct,
            "incorrect": total - correct,
        })
        results["_counted"] = True

    c1, c2, c3 = st.columns(3)
    c1.metric("Correct",  correct)
    c2.metric("Total",    total)
    c3.metric("Accuracy", f"{accuracy:.1f}%")

    st.progress(min(accuracy / 100, 1.0))
    if accuracy >= 80:
        st.success("🌟 Excellent work!")
    elif accuracy >= 60:
        st.info("📈 Good — review incorrect answers.")
    else:
        st.warning("📚 Keep studying!")

    fig = go.Figure(data=[
        go.Bar(name="Correct",   x=["This Quiz"], y=[correct],         marker_color="#2ecc71"),
        go.Bar(name="Incorrect", x=["This Quiz"], y=[total - correct], marker_color="#e74c3c"),
    ])
    fig.update_layout(
        title="Quiz Breakdown", yaxis_title="Questions",
        barmode="stack", height=240, margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig)

    if st.session_state.quiz_total > 0:
        overall = st.session_state.quiz_correct / st.session_state.quiz_total * 100
        st.caption(
            f"Overall: {st.session_state.quiz_correct}/{st.session_state.quiz_total} "
            f"({overall:.1f}%)"
        )


# ══════════════════════════════════════════════════════════════════
# RESET STATS DIALOG
# ══════════════════════════════════════════════════════════════════

@st.dialog("⚠️ Reset All Quiz Stats")
def _show_reset_dialog():
    st.warning(
        "This will **permanently delete** all your quiz attempts and weak-point "
        "records from the database. This action cannot be undone."
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Yes, Delete Everything", type="primary", use_container_width=True):
            username = st.session_state.get("student_name", "")
            if username:
                if _DB_AVAILABLE:
                    try:
                        delete_user_attempts(username)
                    except Exception:
                        pass
                if _WEAK_DB_AVAILABLE:
                    try:
                        delete_weak_interactions(username)
                    except Exception:
                        pass
            _load_user_data.clear()
            _load_difficulty_stats.clear()
            _load_chapter_mastery.clear()
            _load_topic_mastery.clear()
            st.session_state.quiz_mode             = False
            st.session_state.current_quiz          = None
            st.session_state.quiz_results          = None
            st.session_state.quiz_report_generated = False
            st.session_state.quiz_total            = 0
            st.session_state.quiz_correct          = 0
            st.session_state.quiz_history          = []
            st.session_state.quiz_timer_minutes    = 0
            st.session_state.quiz_start_time       = None
            st.rerun()
    with c2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# QUIZ RATE LIMITER
# ══════════════════════════════════════════════════════════════════

_QUIZ_RATE_LIMIT     = 3   # max generations
_QUIZ_RATE_WINDOW    = 60  # per N seconds


def _quiz_rate_limit_ok() -> bool:
    """
    Return True if the user is allowed to generate another quiz.
    Tracks timestamps in session state; enforces max 3 per 60 seconds.
    """
    now = time.time()
    timestamps: list = st.session_state.get("_quiz_gen_timestamps", [])
    # Drop timestamps outside the rolling window
    timestamps = [t for t in timestamps if now - t < _QUIZ_RATE_WINDOW]
    if len(timestamps) >= _QUIZ_RATE_LIMIT:
        st.session_state["_quiz_gen_timestamps"] = timestamps
        return False
    timestamps.append(now)
    st.session_state["_quiz_gen_timestamps"] = timestamps
    return True


# ══════════════════════════════════════════════════════════════════
# QUIZ SETUP WIZARD
# ══════════════════════════════════════════════════════════════════

def _render_quiz_setup_wizard():
    """Full-page quiz configuration shown when no quiz is currently loaded."""
    st.header("🎯 Build Your Quiz")
    st.markdown("Configure the settings below, then hit **Generate**.")
    st.divider()

    _, col_form, _ = st.columns([1, 2, 1])

    with col_form:
        # ── Difficulty ────────────────────────────────────────────
        st.subheader("Difficulty")
        _diff_options = ["Easy", "Medium", "Hard", "🤖 Auto-Adjust"]
        _stored_diff  = st.session_state.get("quiz_difficulty", "Medium")
        _diff_idx     = _diff_options.index(_stored_diff) if _stored_diff in _diff_options else 1

        difficulty_selection = st.radio(
            "Difficulty",
            options=_diff_options,
            index=_diff_idx,
            horizontal=True,
            label_visibility="collapsed",
            key="wizard_difficulty",
        )
        if difficulty_selection == "🤖 Auto-Adjust":
            effective_difficulty = _recommend_difficulty(st.session_state.get("quiz_history", []))
            st.info(
                f"🤖 **Auto-Adjust**: Based on your last 5 sessions, recommending "
                f"**{effective_difficulty}** difficulty."
            )
        else:
            effective_difficulty = difficulty_selection
            st.session_state.quiz_difficulty = effective_difficulty

        st.markdown("")

        # ── Study Sources ─────────────────────────────────────────
        if st.session_state.ready and st.session_state.collection:
            try:
                _meta   = st.session_state.collection.get(include=["metadatas"])
                _all_sf = sorted({
                    m.get("source_file", "")
                    for m in (_meta.get("metadatas") or [])
                    if m and m.get("source_file")
                })
            except Exception:
                _all_sf = []

            if _all_sf:
                st.subheader("Study Sources")
                st.caption("Uncheck sources you want to exclude from this quiz.")

                # Select All / Deselect All
                _sa_col, _da_col = st.columns(2)
                with _sa_col:
                    if st.button("☑ Select All", key="wiz_src_btn_select_all",
                                 use_container_width=True):
                        for _j in range(len(_all_sf)):
                            st.session_state[f"wiz_src_chk_{_j}"] = True
                        st.rerun()
                with _da_col:
                    if st.button("☐ Deselect All", key="wiz_src_btn_deselect_all",
                                 use_container_width=True):
                        for _j in range(len(_all_sf)):
                            st.session_state[f"wiz_src_chk_{_j}"] = False
                        st.rerun()

                _mastery_data: dict = {}
                if _WEAK_DB_AVAILABLE and st.session_state.get("logged_in"):
                    _mastery_data = _load_chapter_mastery(st.session_state.student_name)

                _prev_filter     = st.session_state.get("quiz_source_filter") or []
                _checked_sources = []
                for _i, _sf in enumerate(_all_sf):
                    _chk_key     = f"wiz_src_chk_{_i}"
                    _default_val = True
                    # Use setdefault so we never pass both value= and a pre-set
                    # session-state key to the same widget (Streamlit raises a
                    # warning and causes unnecessary reruns when both are present).
                    st.session_state.setdefault(_chk_key, _default_val)
                    _checked = st.checkbox(
                        _sf,
                        key=_chk_key,
                    )
                    if _checked:
                        _checked_sources.append(_sf)
                    if _sf in _mastery_data:
                        _pct = _mastery_data[_sf].get("mastery_pct", 0.0)
                        st.progress(int(_pct), text=f"Mastery: {_pct:.0f}%")

                if set(_checked_sources) == set(_all_sf):
                    st.session_state.quiz_source_filter = []
                else:
                    st.session_state.quiz_source_filter = _checked_sources

                _n_sel = len(_checked_sources)
                _n_tot = len(_all_sf)
                if _n_sel < _n_tot:
                    st.caption(f"Filtering to {_n_sel} of {_n_tot} source(s)")
                st.markdown("")

        _src_filter = st.session_state.get("quiz_source_filter") or None

        # ── Question Count ────────────────────────────────────────
        st.subheader("Questions")
        _num_q = st.select_slider(
            "Number of questions",
            options=[3, 5, 10, 15],
            value=st.session_state.get("quiz_num_questions", DEFAULT_NUM_QUESTIONS),
            format_func=lambda x: f"{x} questions",
            key="wizard_num_questions_slider",
            label_visibility="collapsed",
        )
        st.session_state.quiz_num_questions = _num_q
        st.markdown("")

        # ── Exam / Timed Mode ─────────────────────────────────────
        st.subheader("Exam Mode")
        _timed = st.toggle(
            "⏱ Timed Mode",
            value=st.session_state.get("quiz_timer_minutes", 0) > 0,
            key="wizard_timed_mode",
        )
        if _timed:
            _timer_choice = st.select_slider(
                "Time limit",
                options=[3, 5, 10, 15, 20, 30],
                value=max(st.session_state.get("quiz_timer_minutes", 0) or 10,
                          3),
                format_func=lambda x: f"{x} min",
                key="wizard_timer_mins_slider",
            )
            st.caption(
                f"The quiz will auto-submit after **{_timer_choice} minutes**. "
                "A countdown timer will appear above the questions."
            )
        else:
            _timer_choice = 0

        st.markdown("")
        st.session_state.setdefault("exam_crunch_mode", False)
        _exam_crunch = st.toggle(
            "🚨 Exam Crunch Mode — I have an exam soon!",
            key="exam_crunch_mode",
        )
        if _exam_crunch:
            st.info(
                "📋 **Exam Crunch Active**: All PDFs will be scanned and questions "
                "will prioritize key definitions, core architectures, and comparison "
                "questions to maximize your exam score."
            )
        st.markdown("")

        # ── Generate buttons ──────────────────────────────────────
        if not st.session_state.ready:
            st.warning("⬅️ Knowledge base not loaded — use the sidebar to ingest PDFs first.")
        else:
            if st.button("🎯 Generate Quiz", type="primary", use_container_width=True):
                _quiz_err_slot = st.empty()
                if not _quiz_rate_limit_ok():
                    _quiz_err_slot.warning(
                        "⏳ You've generated 3 quizzes in the last minute. "
                        "Please wait a moment before generating another."
                    )
                    st.stop()
                with st.status("Generating quiz…", expanded=True) as _gen_status:
                    _gen_status.write("📚 Retrieving relevant content from your PDFs…")

                    def _on_quiz_retry(attempt, max_r, wait, err_type):
                        label = "AI server busy" if err_type == "server_busy" else "Rate limit reached"
                        _gen_status.write(
                            f"⏳ {label} — retrying (attempt {attempt + 1}/{max_r + 1}) "
                            f"in {wait}s…"
                        )
                        _gen_status.update(
                            label=f"Generating quiz… (retry {attempt}/{max_r})",
                            state="running",
                        )

                    try:
                        quiz = _generate_quiz_via_rag(
                            _num_q, effective_difficulty,
                            source_filter=_src_filter,
                            on_retry=_on_quiz_retry,
                            exam_crunch=_exam_crunch,
                        )
                        _gen_status.update(label="Quiz ready!", state="complete")
                    except QuizAPIError as exc:
                        _gen_status.update(label="Generation failed", state="error")
                        if exc.err_type == "server_busy":
                            _quiz_err_slot.error(
                                "⚠️ The AI is under high demand and all retries were exhausted. "
                                "Please wait a moment and try again."
                            )
                        elif exc.err_type == "rate_limit":
                            _quiz_err_slot.error(
                                "⚠️ API rate limit reached. Please wait 30 seconds and try again."
                            )
                        else:
                            _quiz_err_slot.error(f"⚠️ Quiz generation failed: {exc}")
                        quiz = None
                    except QuizNoContextError as exc:
                        _gen_status.update(label="Insufficient content", state="error")
                        _quiz_err_slot.error(
                            f"📭 {exc} "
                            "Try selecting more sources or ingesting additional PDFs."
                        )
                        quiz = None
                    except QuizParseError:
                        _gen_status.update(label="Malformed response", state="error")
                        _quiz_err_slot.warning(
                            "The AI returned an unexpected format. "
                            "Click **Generate Quiz** again to retry."
                        )
                        quiz = None

                if quiz is not None:
                    st.session_state.current_quiz          = quiz
                    st.session_state.quiz_mode             = True
                    st.session_state.quiz_results          = None
                    st.session_state.quiz_report_generated = False
                    st.session_state.quiz_timer_minutes    = _timer_choice
                    st.session_state.quiz_start_time       = datetime.now() if _timer_choice > 0 else None
                    st.session_state.quiz_difficulty       = effective_difficulty
                    st.rerun()

            # Adaptive quiz — only shown when weak records exist
            if _WEAK_DB_AVAILABLE and st.session_state.get("logged_in"):
                if bool(get_weak_sources(st.session_state.student_name)):
                    st.markdown("")
                    if st.button(
                        "🔁 Weak Point Quiz", use_container_width=True,
                        help="Quiz generated from topics you previously answered incorrectly",
                    ):
                        _adp_err_slot = st.empty()
                        if not _quiz_rate_limit_ok():
                            _adp_err_slot.warning(
                                "⏳ You've generated 3 quizzes in the last minute. "
                                "Please wait a moment before generating another."
                            )
                            st.stop()
                        with st.status("Building adaptive quiz…", expanded=True) as _adp_status:
                            _adp_status.write("🔍 Analysing your weak areas…")

                            def _on_adp_retry(attempt, max_r, wait, err_type):
                                label = "AI server busy" if err_type == "server_busy" else "Rate limit"
                                _adp_status.write(
                                    f"⏳ {label} — retrying (attempt {attempt + 1}/{max_r + 1}) "
                                    f"in {wait}s…"
                                )

                            try:
                                quiz = _generate_adaptive_quiz(
                                    _num_q, effective_difficulty,
                                    on_retry=_on_adp_retry,
                                )
                                _adp_status.update(label="Adaptive quiz ready!", state="complete")
                            except QuizAPIError as exc:
                                _adp_status.update(label="Generation failed", state="error")
                                _adp_err_slot.error(
                                    "⚠️ AI server busy — all retries exhausted. "
                                    "Please try again in a moment."
                                    if exc.err_type == "server_busy" else
                                    f"⚠️ Quiz generation failed: {exc}"
                                )
                                quiz = None
                            except QuizNoContextError as exc:
                                _adp_status.update(label="Not enough weak-point data", state="error")
                                _adp_err_slot.info(str(exc))
                                quiz = None
                            except QuizParseError:
                                _adp_status.update(label="Malformed response", state="error")
                                _adp_err_slot.warning(
                                    "Adaptive quiz returned unexpected format — please try again."
                                )
                                quiz = None

                        if quiz is not None:
                            st.session_state.current_quiz          = quiz
                            st.session_state.quiz_mode             = True
                            st.session_state.quiz_results          = None
                            st.session_state.quiz_report_generated = False
                            st.session_state.quiz_timer_minutes    = _timer_choice
                            st.session_state.quiz_start_time       = datetime.now() if _timer_choice > 0 else None
                            st.rerun()


# ══════════════════════════════════════════════════════════════════
# LOGIN / REGISTER SCREEN
# ══════════════════════════════════════════════════════════════════

def _restore_history_from_db(username: str) -> None:
    try:
        history, stats = _load_user_data(username)
        st.session_state.quiz_history  = history
        st.session_state.quiz_total    = stats["total"]
        st.session_state.quiz_correct  = stats["correct"]
    except Exception:
        pass

    if _CHAT_DB_AVAILABLE:
        try:
            active = get_active_session(username)
            if active and active.get("messages"):
                st.session_state.messages        = active["messages"]
                st.session_state.chat_session_id = active["session_id"]
        except Exception:
            pass


def _render_login_screen() -> None:
    st.markdown("## 👋 Welcome to DataPilot!")
    st.markdown("Your AI-powered Virtual Teaching Assistant.")
    st.markdown("")

    if not _DB_AVAILABLE:
        st.error("Database unavailable — check that MONGO_URI is set in your .env file.")
        return

    tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="e.g. sarah_j")
            password = st.text_input("Password", type="password")
            submit   = st.form_submit_button("Login", type="primary", use_container_width=True)

        if submit:
            if not username.strip() or not password:
                st.error("Please enter both username and password.")
            else:
                with st.status("Signing in…", expanded=True) as _status:
                    try:
                        st.write("Verifying credentials…")
                        user = login_user(username.strip(), password)
                        if user is None:
                            _status.update(label="Incorrect username or password", state="error")
                        else:
                            st.write("Loading your quiz history…")
                            _restore_history_from_db(user["username"])
                            st.session_state.student_name    = user["username"]
                            st.session_state.session_started = True
                            st.session_state.logged_in       = True
                            st.session_state.is_admin        = _ADMIN_AVAILABLE and _is_admin(user["username"])
                            st.write(f"DEBUG is_admin: {st.session_state.get('is_admin')}")
                            _status.update(label="Signed in!", state="complete")
                            st.rerun()
                    except Exception as e:
                        _status.update(label="Connection error", state="error")
                        st.error(f"Login failed: {e}")

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username", placeholder="e.g. sarah_j")
            new_password = st.text_input("Choose a password", type="password")
            confirm_pw   = st.text_input("Confirm password",  type="password")
            submit_reg   = st.form_submit_button("Create Account", type="primary", use_container_width=True)

        if submit_reg:
            if not new_username.strip() or not new_password:
                st.error("Username and password are required.")
            elif new_password != confirm_pw:
                st.error("Passwords do not match.")
            else:
                try:
                    user = register_user(new_username.strip(), new_password)
                    st.session_state.student_name    = user["username"]
                    st.session_state.session_started = True
                    st.session_state.logged_in       = True
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Registration failed: {e}")


# ══════════════════════════════════════════════════════════════════
# INGESTION HELPER
# ══════════════════════════════════════════════════════════════════

def _run_inline_ingestion() -> None:
    """Parse, embed, and upsert all PDFs found in _RAW_DIR."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("GEMINI_API_KEY not set — cannot embed documents.")
        return

    pdf_files = sorted(_RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        st.warning(f"No PDFs found in: {_RAW_DIR}")
        return

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure collection is initialised before we start
    chroma_path = Path(str(CHROMA_DB_PATH))
    chroma_path.mkdir(parents=True, exist_ok=True)
    if st.session_state.collection is None:
        st.session_state.collection = _get_chroma_collection(str(chroma_path))
    collection = st.session_state.collection

    from src.vectorstore.chroma_store import upsert_chunks

    st.subheader(f"⚡ Indexing {len(pdf_files)} PDF(s)…")
    overall_bar = st.progress(0.0)

    failed = []

    for i, pdf_path in enumerate(pdf_files):
        with st.status(f"[{i+1}/{len(pdf_files)}] {pdf_path.name}", expanded=True) as s:
            try:
                json_path = _PROCESSED_DIR / f"{pdf_path.stem}_parsed.json"

                # ── Step 1: Parse (with stale-cache guard) ────────────────
                # A previous run may have left a JSON with 0 chunks (corrupted
                # source PDF). Always validate the cache before trusting it.
                if json_path.exists():
                    try:
                        cached = load_parsed_chunks(str(json_path))
                    except Exception:
                        cached = []

                    if cached:
                        st.write(f"JSON cache valid — {len(cached)} chunks.")
                        chunks = cached
                    else:
                        st.write("⚠️ Cached JSON is empty — deleting and re-parsing…")
                        json_path.unlink(missing_ok=True)
                        st.write("Parsing PDF…")
                        parse_pdf(str(pdf_path), str(_PROCESSED_DIR), api_key=api_key)
                        chunks = load_parsed_chunks(str(json_path))
                else:
                    st.write("Parsing PDF…")
                    parse_pdf(str(pdf_path), str(_PROCESSED_DIR), api_key=api_key)
                    chunks = load_parsed_chunks(str(json_path))

                # ── Step 2: Validate chunk count before embedding ─────────
                if not chunks:
                    msg = f"❌ {pdf_path.name} — no text extracted (check PDF quality)"
                    s.update(label=msg, state="error")
                    failed.append(pdf_path.name)
                    overall_bar.progress((i + 1) / len(pdf_files))
                    continue

                # ── Step 3: Embed ─────────────────────────────────────────
                st.write(f"Extracted {len(chunks)} chunks. Embedding…")
                embedded  = []
                n_batches = max(1, (len(chunks) + 19) // 20)
                batch_bar = st.progress(0.0)
                for b_idx, b_start in enumerate(range(0, len(chunks), 20)):
                    batch = embed_chunks(chunks[b_start : b_start + 20], api_key=api_key)
                    embedded.extend(batch)
                    batch_bar.progress((b_idx + 1) / n_batches)
                    if b_start + 20 < len(chunks):
                        time.sleep(1.0)

                # ── Step 4: Upsert ────────────────────────────────────────
                st.write(f"Upserting {len(embedded)} vectors…")
                n_up = upsert_chunks(collection, embedded)
                s.update(label=f"✅ {pdf_path.name} — {n_up} chunks indexed", state="complete")

            except Exception as exc:
                s.update(label=f"❌ {pdf_path.name} — error: {exc}", state="error")
                failed.append(pdf_path.name)

        overall_bar.progress((i + 1) / len(pdf_files))

    stats = get_collection_stats(collection)
    st.session_state.ready = True
    ok_count = len(pdf_files) - len(failed)
    st.success(
        f"🎉 Indexing complete! "
        f"**{ok_count}/{len(pdf_files)}** PDFs indexed. "
        f"Knowledge base: **{stats['total_chunks']}** chunks total."
    )
    if failed:
        st.error(
            "The following PDFs produced 0 chunks and were skipped:\n"
            + "\n".join(f"- {f}" for f in failed)
            + "\n\nCheck that the files are valid, non-password-protected PDFs "
            "and re-run ingestion."
        )


# ══════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════

def _render_chat_page():
    col_chat, col_sources = st.columns([3, 2])

    with col_chat:
        st.header("💬 DataPilot Chat")

        # ── Render committed messages ──────────────────────────────
        _BADGE = {
            "direct_answer_request": "#e74c3c",
            "administrative":        "#3498db",
            "conceptual_help":       "#2ecc71",
            "kb_metadata":           "#8e44ad",
            "conversational":        "#1abc9c",
        }
        for msg in st.session_state.messages:
            if msg.get("is_placeholder", False):
                continue
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("metadata"):
                    meta        = msg["metadata"]
                    badge_color = _BADGE.get(meta.get("query_class", ""), "#95a5a6")
                    st.markdown(
                        f'<span style="background:{badge_color};color:white;'
                        f'padding:2px 8px;border-radius:4px;font-size:0.75em;">'
                        f'{meta.get("query_class","").upper()}</span>',
                        unsafe_allow_html=True,
                    )
                    if meta.get("query_class") == "conceptual_help":
                        _cits = meta.get("citations", [])
                        if _cits:
                            with st.expander("📚 Recommended Review"):
                                for _cit in _cits[:3]:
                                    _src = (
                                        _cit.source_file
                                        if hasattr(_cit, "source_file")
                                        else _cit.get("source_file", "")
                                    )
                                    _pg = (
                                        _cit.page_number
                                        if hasattr(_cit, "page_number")
                                        else _cit.get("page_number")
                                    )
                                    if _src and _pg and _pg != -1:
                                        st.markdown(
                                            f"📄 Re-read **{_src}** — page **{_pg}**"
                                        )
                                        st.caption(
                                            f"Based on your question, review {_src} page {_pg} for more depth."
                                        )
                                    elif _src:
                                        st.markdown(f"📄 Review **{_src}**")
                                        st.caption(
                                            f"Based on your question, review {_src} for more depth."
                                        )

        # ── Thinking state: process pending query inline ────────────
        # This runs AFTER committed messages are rendered so the student
        # can always see the conversation while the API call is in flight.
        pending = st.session_state.get("_pending_query")
        if pending:
            # Clear immediately so a page-refresh can't re-trigger it
            st.session_state["_pending_query"] = None
            # Flush any leftover placeholder messages from old code paths
            st.session_state.messages = [
                m for m in st.session_state.messages
                if not m.get("is_placeholder", False)
            ]
            with st.chat_message("assistant"):
                st.markdown("⏳ Thinking…")
                try:
                    _execute_chat_query(pending)
                except Exception as _outer_exc:
                    # Absolute last resort — _execute_chat_query has its own
                    # try/except, but this catches anything that escapes it
                    # (e.g. Streamlit internals, import errors) so st.rerun()
                    # always fires and the session is never permanently locked.
                    st.session_state.messages.append({
                        "role":           "assistant",
                        "content":        "I encountered a hiccup! Please try rephrasing your question.",
                        "metadata":       {
                            "citations": [], "reasoning_trace": type(_outer_exc).__name__,
                            "was_refused": False, "was_uncertain": False,
                            "query_class": "error", "confidence": 0.0,
                        },
                        "is_placeholder": False,
                    })
            st.rerun()
            return

        # ── Empty-state hints ───────────────────────────────────────
        if not st.session_state.ready:
            st.info("⬅️ Knowledge base not loaded — check sidebar.")
        elif not st.session_state.messages:
            st.info(
                "👋 Ask about your course materials, or use the sidebar to generate a quiz!"
            )

        # ── Chat input ──────────────────────────────────────────────
        user_input = st.chat_input(
            "Ask about your course...",
            disabled=not st.session_state.ready,
        )
        if user_input:
            # Only treat as a quiz-generation request when the input is a short,
            # explicit command (≤120 chars, single-line).  Long or multi-line
            # inputs are pasted course content and must go through the chat path.
            _is_quiz_request = (
                len(user_input) <= 120
                and "\n" not in user_input
                and any(w in user_input.lower() for w in ["quiz", "test me", "question me"])
            )
            if _is_quiz_request:
                if not _quiz_rate_limit_ok():
                    st.warning(
                        "⏳ You've generated 3 quizzes in the last minute. "
                        "Please wait a moment before generating another."
                    )
                    st.stop()
                try:
                    with st.spinner("Generating quiz…"):
                        quiz = _generate_quiz_via_rag(
                            st.session_state.get("quiz_num_questions", DEFAULT_NUM_QUESTIONS),
                            st.session_state.quiz_difficulty,
                            source_filter=st.session_state.get("quiz_source_filter") or None,
                        )
                except QuizAPIError as exc:
                    if exc.err_type == "server_busy":
                        st.error(
                            "⚠️ The AI is currently under high demand. "
                            "Try again in a moment, or visit the Quiz page to configure a quiz."
                        )
                    else:
                        st.error(f"⚠️ Quiz generation failed: {exc}")
                    return
                except QuizNoContextError as exc:
                    st.error(
                        f"📭 {exc} "
                        "Visit the **Quiz** page to select specific sources."
                    )
                    return
                except QuizParseError:
                    st.warning("AI returned an unexpected format — please try again.")
                    return
                st.session_state.current_quiz          = quiz
                st.session_state.quiz_mode             = True
                st.session_state.quiz_results          = None
                st.session_state.quiz_report_generated = False
                st.session_state.page                  = "Quiz"
                st.rerun()
                return

            # Append user message, set pending query, rerun.
            # No placeholder needed — the thinking bubble is rendered
            # on the very next run before the API call starts.
            st.session_state.messages.append(
                {"role": "user", "content": user_input, "metadata": None}
            )
            st.session_state["_pending_query"] = user_input
            st.rerun()

    with col_sources:
        # JavaScript sticky: walk up from this iframe to the Streamlit column
        # element and apply sticky positioning directly.  CSS alone cannot do
        # this because Streamlit's app container has overflow:hidden which
        # blocks sticky on any descendant element.
        st.html("""
<script>
(function(){
  function pin(){
    try {
      var pd = window.parent.document;
      var me = Array.from(pd.querySelectorAll('iframe'))
                    .find(function(f){ return f.contentWindow === window; });
      if (!me) return;
      var el = me.parentElement;
      while (el) {
        if (el.dataset && el.dataset.testid === 'column') {
          el.style.position  = 'sticky';
          el.style.top       = '3.5rem';
          el.style.maxHeight = 'calc(100vh - 4.5rem)';
          el.style.overflowY = 'auto';
          el.style.alignSelf = 'flex-start';
          return;
        }
        el = el.parentElement;
      }
    } catch(e) {}
  }
  pin(); setTimeout(pin, 300); setTimeout(pin, 1000);
})();
</script>
""")
        st.header("📚 Sources")

        meta = st.session_state.get("last_response_meta")
        if meta:
            with st.expander("🧠 Last Response Analysis", expanded=False):
                st.markdown(f"**Class:** `{meta.get('query_class','')}`")
                st.markdown(f"**Confidence:** `{int(meta.get('confidence', 0) * 100)}%`")
                st.markdown(f"**Refused:** `{'Yes' if meta.get('was_refused') else 'No'}`")
                if meta.get("reasoning_trace"):
                    st.markdown(f"**Reasoning:** {meta['reasoning_trace']}")

            citations = meta.get("citations", [])
            if citations:
                st.markdown("**📚 Sources Used**")
                for i, cit in enumerate(citations[:3], 1):
                    with st.expander(
                        f"Source {i}: {cit.source_file} ({cit.confidence_pct}% match)"
                    ):
                        st.markdown(f"**File:** `{cit.source_file}`")
                        if cit.page_number is not None:
                            st.markdown(f"**Page:** {cit.page_number}")
                        st.markdown(f"**Confidence:** `{cit.confidence_pct}%`")
        else:
            st.caption("Source citations will appear here after your first question.")


def _render_quiz_page():
    # Auto-generate triggered by "New Quiz" button on the results screen.
    # Runs generation here (with spinner) instead of leaving a blank page.
    if st.session_state.get("_quiz_auto_generate"):
        st.session_state._quiz_auto_generate = False
        _coach_concept = st.session_state.pop("_coach_review_concept", None)
        _coach_sources = st.session_state.pop("_coach_review_sources", None)

        st.header("🎯 Quiz")
        if _coach_concept:
            st.info(
                f"📋 **Coach's Insight Mode**: Generating mini-quiz focused on "
                f"_{_coach_concept[:60]}_"
            )
        _auto_err = st.empty()
        with st.status("Generating quiz…", expanded=True) as _auto_status:
            _auto_status.write("📚 Retrieving relevant content…")

            def _on_auto_retry(attempt, max_r, wait, err_type):
                label = "AI server busy" if err_type == "server_busy" else "Rate limit"
                _auto_status.write(
                    f"⏳ {label} — retrying (attempt {attempt + 1}/{max_r + 1}) in {wait}s…"
                )

            _effective_sources = _coach_sources or st.session_state.get("quiz_source_filter") or None

            try:
                quiz = _generate_quiz_via_rag(
                    st.session_state.get("quiz_num_questions", DEFAULT_NUM_QUESTIONS),
                    st.session_state.quiz_difficulty,
                    source_filter=_effective_sources,
                    on_retry=_on_auto_retry,
                    topic_override=_coach_concept,
                )
                _auto_status.update(label="Quiz ready!", state="complete")
            except QuizAPIError as exc:
                _auto_status.update(label="Generation failed", state="error")
                _auto_err.error(
                    "⚠️ The AI is under high demand. "
                    "Please wait a moment and generate a new quiz."
                    if exc.err_type == "server_busy" else
                    f"⚠️ Quiz generation failed: {exc}"
                )
                return
            except QuizNoContextError as exc:
                _auto_status.update(label="Insufficient content", state="error")
                _auto_err.error(str(exc))
                return
            except QuizParseError:
                _auto_status.update(label="Malformed response", state="error")
                _auto_err.warning("AI returned unexpected format — try generating again.")
                return

        st.session_state.current_quiz          = quiz
        st.session_state.quiz_mode             = True
        st.session_state.quiz_results          = None
        st.session_state.quiz_report_generated = False
        st.rerun()
        return

    if not st.session_state.get("current_quiz"):
        _render_quiz_setup_wizard()
        return

    col_quiz, col_report = st.columns([3, 2])

    with col_quiz:
        _render_quiz_interface()

    with col_report:
        _render_quiz_report()
        quiz = st.session_state.current_quiz
        if quiz and quiz.citations:
            st.markdown("**📚 Quiz Sources**")
            for i, cit in enumerate(quiz.citations[:3], 1):
                with st.expander(f"Source {i}: {cit.source_file}"):
                    st.markdown(f"**File:** `{cit.source_file}`")
                    if getattr(cit, "page_number", None) is not None:
                        st.markdown(f"**Page:** {cit.page_number}")
                    st.markdown(f"**Confidence:** `{cit.confidence_pct}%`")


# ──────────────────────────────────────────────────────────────────
# Dashboard stat helpers — all computation from the history list,
# never from DB aggregations, so every number on the page is
# mathematically consistent with the filter the user selected.
# ──────────────────────────────────────────────────────────────────

def _mastery_pct(correct: int, total: int) -> float:
    """Single source of truth: (correct / total) * 100, rounded to 1 dp."""
    return round(correct / total * 100, 1) if total else 0.0


def _get_rank(pct: float) -> tuple:
    """Return (emoji, title, hex_color) for a given mastery percentage."""
    if pct >= 91:
        return "💎", "The Master",  "#00bcd4"   # cyan / platinum
    if pct >= 71:
        return "🥇", "The Sage",    "#FFD700"   # gold
    if pct >= 41:
        return "🥈", "The Scholar", "#C0C0C0"   # silver
    return   "🥉", "The Novice",   "#CD7F32"   # bronze


def _compute_streak(history: list[dict]) -> int:
    """Return the longest consecutive-day streak found anywhere in quiz history."""
    from datetime import date, timedelta
    dates: set = set()
    for h in history:
        ts = h.get("timestamp", "")
        if ts and len(ts) >= 10:
            try:
                dates.add(date.fromisoformat(ts[:10]))
            except ValueError:
                pass
    if not dates:
        return 0
    sorted_dates = sorted(dates)
    best = cur = 1
    for i in range(1, len(sorted_dates)):
        if sorted_dates[i] - sorted_dates[i - 1] == timedelta(days=1):
            cur  += 1
            best  = max(best, cur)
        else:
            cur = 1
    return best


def _compute_difficulty_stats(history: list[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for h in history:
        d = h.get("difficulty", "Medium")
        t = h["correct"] + h["incorrect"]
        stats.setdefault(d, {"total": 0, "correct": 0, "attempts": 0})
        stats[d]["total"]    += t
        stats[d]["correct"]  += h["correct"]
        stats[d]["attempts"] += 1
    for d in stats:
        stats[d]["percentage"] = _mastery_pct(stats[d]["correct"], stats[d]["total"])
    return stats


def _compute_chapter_stats(history: list[dict]) -> dict[str, dict]:
    """Credit each session's questions to every unique source_file it cited."""
    stats: dict[str, dict] = {}
    for h in history:
        t = h["correct"] + h["incorrect"]
        for src in dict.fromkeys(h.get("sources", [])):  # deduplicate: same PDF cited on multiple pages counts once
            if not src:
                continue
            stats.setdefault(src, {"total": 0, "correct": 0, "attempts": 0})
            stats[src]["total"]    += t
            stats[src]["correct"]  += h["correct"]
            stats[src]["attempts"] += 1
    for src in stats:
        stats[src]["mastery_pct"] = _mastery_pct(stats[src]["correct"], stats[src]["total"])
    return stats


def _compute_topic_stats(history: list[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for h in history:
        topic = h.get("topic", "")
        if not topic:
            continue
        t = h["correct"] + h["incorrect"]
        stats.setdefault(topic, {"total": 0, "correct": 0, "attempts": 0})
        stats[topic]["total"]    += t
        stats[topic]["correct"]  += h["correct"]
        stats[topic]["attempts"] += 1
    for topic in stats:
        stats[topic]["mastery_pct"] = _mastery_pct(stats[topic]["correct"], stats[topic]["total"])
    return stats


def _recommend_difficulty(history: list[dict]) -> str:
    """Suggest next difficulty based on last 5 quiz sessions."""
    recent = [h for h in history[-5:] if "correct" in h and "incorrect" in h]
    if not recent:
        return "Medium"
    avg = sum(
        _mastery_pct(h["correct"], h["correct"] + h["incorrect"])
        for h in recent
    ) / len(recent)
    if avg > 85:
        return "Hard"
    if avg < 50:
        return "Easy"
    return "Medium"


def _render_dashboard_page():
    st.header("📊 Performance Dashboard")

    history = st.session_state.get("quiz_history", [])

    if not history:
        st.info(
            "No quiz data yet. Generate a quiz from the sidebar to start "
            "building your performance profile!"
        )
        return

    # Always derive totals from history so header and charts share
    # the exact same dataset — never rely on session-state aggregates.
    total_q = sum(h["correct"] + h["incorrect"] for h in history)
    correct = sum(h["correct"] for h in history)

    # ── PDF / Chapter drill-down filter ──────────────────────────
    # Collect every source_file referenced across all attempts
    _all_pdfs = sorted({
        src
        for h in history
        for src in h.get("sources", [])
        if src
    })
    # Fall back to ChromaDB metadata when no sources are stored yet
    if not _all_pdfs and st.session_state.get("collection"):
        try:
            _kb_meta  = st.session_state.collection.get(include=["metadatas"])
            _all_pdfs = sorted({
                m.get("source_file", "")
                for m in (_kb_meta.get("metadatas") or [])
                if m and m.get("source_file")
            })
        except Exception:
            pass

    _sel_pdfs: list[str] = []      # tracks active filter for downstream sections

    if _all_pdfs:
        _sel_pdfs = st.multiselect(
            "Filter by PDF / Chapter",
            options=_all_pdfs,
            default=[],
            placeholder="All PDFs — select one or more to drill down",
            key="dash_pdf_filter",
        )
        if _sel_pdfs:
            history = [
                h for h in history
                if any(src in _sel_pdfs for src in h.get("sources", []))
            ]
            if not history:
                st.warning("No quiz sessions found for the selected PDF(s).")
                return
            total_q = sum(h["correct"] + h["incorrect"] for h in history)
            correct = sum(h["correct"] for h in history)

    overall_pct = _mastery_pct(correct, total_q)

    # ── Rank badge ────────────────────────────────────────────────
    _r_emoji, _r_title, _r_color = _get_rank(overall_pct)
    st.markdown(
        f'<div style="background:{_r_color}22;border:2px solid {_r_color};'
        f'border-radius:12px;padding:10px 20px;text-align:center;margin-bottom:10px">'
        f'<span style="font-size:2em;vertical-align:middle">{_r_emoji}</span>'
        f'&nbsp;&nbsp;'
        f'<span style="font-size:1.35em;font-weight:700;color:{_r_color};vertical-align:middle">'
        f'{_r_title}</span>'
        f'&nbsp;&nbsp;'
        f'<span style="font-size:0.95em;color:#888;vertical-align:middle">'
        f'({overall_pct:.1f}% overall mastery)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Headline metrics ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Mastery",  f"{overall_pct:.1f}%")
    c2.metric("Total Questions",  total_q)
    c3.metric("Correct Answers",  correct)
    c4.metric("Quiz Sessions",    len(history))

    st.markdown("**Overall Mastery**")
    st.progress(min(overall_pct / 100, 1.0))

    # ── CSV export ────────────────────────────────────────────────
    _export_rows = []
    for h in history:
        _t = h["correct"] + h["incorrect"]
        _export_rows.append({
            "Session":          h["label"],
            "Topic":            h.get("topic", ""),
            "Difficulty":       h.get("difficulty", ""),
            "Correct_Answers":  h["correct"],
            "Total_Questions":  _t,
            "Accuracy %":       _mastery_pct(h["correct"], _t),
            "Sources":          "; ".join(h.get("sources", [])),
            "Date":             h.get("timestamp", ""),
        })
    _csv_bytes = pd.DataFrame(_export_rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Analytics Report (CSV)",
        data=_csv_bytes,
        file_name="datapilot_analytics_report.csv",
        mime="text/csv",
    )

    st.divider()

    # ── Score trend (line chart) ──────────────────────────────────
    st.subheader("📈 Score Trend")
    if len(history) >= 2:
        _trend_pcts = [
            _mastery_pct(h["correct"], h["correct"] + h["incorrect"])
            for h in history
        ]
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=[h["label"] for h in history],
            y=_trend_pcts,
            mode="lines+markers",
            name="Score %",
            line=dict(color="#3498db", width=2.5),
            marker=dict(size=8, color="#3498db"),
            fill="tozeroy",
            fillcolor="rgba(52,152,219,0.08)",
        ))
        fig_line.add_hline(
            y=80, line_dash="dash", line_color="#2ecc71",
            annotation_text="Target 80%", annotation_position="bottom right",
        )
        fig_line.add_hline(
            y=60, line_dash="dot", line_color="#f39c12",
            annotation_text="Review zone", annotation_position="bottom right",
        )
        fig_line.update_layout(
            yaxis=dict(title="Score %", range=[0, 105]),
            height=280, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_line)
    else:
        st.caption("Complete at least 2 quizzes to see your score trend.")

    st.divider()

    # ── Performance breakdown (stacked bar) ───────────────────────
    st.subheader("📊 Performance Breakdown")
    df  = pd.DataFrame(history)
    fig = go.Figure(data=[
        go.Bar(name="Correct",   x=df["label"], y=df["correct"],   marker_color="#2ecc71"),
        go.Bar(name="Incorrect", x=df["label"], y=df["incorrect"], marker_color="#e74c3c"),
    ])
    fig.update_layout(
        barmode="stack", yaxis_title="Questions",
        height=280, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig)

    st.divider()

    # ── Per-session breakdown ─────────────────────────────────────
    st.subheader("📋 Session Breakdown")
    rows = []
    for h in history:
        t = h["correct"] + h["incorrect"]
        rows.append({
            "Session":  h["label"],
            "Score":    f"{h['correct']}/{t}",
            "Accuracy": f"{_mastery_pct(h['correct'], t):.1f}%",
            "Topic":    (h.get("topic") or "")[:60],
            "Date":     h.get("timestamp", ""),
        })
    st.dataframe(pd.DataFrame(rows))

    st.divider()

    # ── Per-difficulty mastery ─────────────────────────────────────
    diff_stats = _compute_difficulty_stats(history)
    if diff_stats:
        st.subheader("🎯 Mastery by Difficulty")
        _DIFF_COLORS = {"Easy": "#2ecc71", "Medium": "#f39c12", "Hard": "#e74c3c"}
        diff_cols = st.columns(len(diff_stats))
        for col, level in zip(diff_cols, ["Easy", "Medium", "Hard"]):
            if level not in diff_stats:
                continue
            d     = diff_stats[level]
            color = _DIFF_COLORS.get(level, "#95a5a6")
            col.markdown(
                f'<div style="text-align:center">'
                f'<span style="font-size:1.1em;font-weight:600;color:{color}">'
                f'{level}</span><br>'
                f'<span style="font-size:1.8em;font-weight:700">{d["percentage"]:.1f}%</span><br>'
                f'<span style="font-size:0.85em;color:#888">'
                f'{d["correct"]}/{d["total"]} correct · {d["attempts"]} attempt(s)'
                f'</span></div>',
                unsafe_allow_html=True,
            )
            col.progress(min(d["percentage"] / 100, 1.0))
        st.divider()

    # ── Chapter mastery ───────────────────────────────────────────
    chapter_data = _compute_chapter_stats(history)
    if _sel_pdfs:
        chapter_data = {k: v for k, v in chapter_data.items() if k in _sel_pdfs}
    if chapter_data:
        if not _sel_pdfs:
            _ch_title = "📖 Global Chapter Mastery"
        elif len(_sel_pdfs) == 1:
            _ch_title = f"📖 Mastery Breakdown: {_sel_pdfs[0]}"
        else:
            _ch_title = f"📖 Mastery Breakdown: {', '.join(_sel_pdfs)}"
        st.subheader(_ch_title)
        sorted_chapters = sorted(
            chapter_data.items(), key=lambda kv: kv[1]["mastery_pct"]
        )

        ch_labels = [
            ch[:40] + (" 🏆" if d["mastery_pct"] >= 100.0 else "")
            for ch, d in sorted_chapters
        ]
        ch_pcts   = [d["mastery_pct"] for _, d in sorted_chapters]

        fig_ch = go.Figure(go.Bar(
            x=ch_pcts,
            y=ch_labels,
            orientation="h",
            marker_color=[
                "#FFD700" if p >= 100.0
                else "#2ecc71" if p >= 80
                else "#f39c12" if p >= 60
                else "#e74c3c"
                for p in ch_pcts
            ],
            text=[f"{p:.1f}%" for p in ch_pcts],
            textposition="auto",
        ))
        fig_ch.update_layout(
            xaxis=dict(title="Mastery %", range=[0, 105]),
            yaxis=dict(automargin=True),
            height=max(200, len(sorted_chapters) * 45),
            margin=dict(t=10, b=30, l=10, r=10),
        )
        st.plotly_chart(fig_ch)

        ch_rows = []
        for chapter, d in sorted_chapters:
            _mastered = d["mastery_pct"] >= 100.0
            ch_rows.append({
                "Chapter":  chapter,
                "Mastery":  f"{d['mastery_pct']:.1f}%",
                "Correct":  d["correct"],
                "Total":    d["total"],
                "Sessions": d["attempts"],
                "Status":   "🏆 Mastered!"  if _mastered
                            else "🟢 Strong"     if d["mastery_pct"] >= 80
                            else "🟡 Improving"  if d["mastery_pct"] >= 60
                            else "🔴 Review",
            })
        st.dataframe(pd.DataFrame(ch_rows))
        st.divider()

    # ── Topic mastery ─────────────────────────────────────────────
    topic_data = _compute_topic_stats(history)
    if topic_data:
        st.subheader("🏷️ Topic Mastery")
        sorted_topics = sorted(
            topic_data.items(), key=lambda kv: kv[1]["mastery_pct"]
        )

        tp_labels = [t[:50] for t, _ in sorted_topics]
        tp_pcts   = [d["mastery_pct"] for _, d in sorted_topics]

        fig_tp = go.Figure(go.Bar(
            x=tp_pcts,
            y=tp_labels,
            orientation="h",
            marker_color=[
                "#e74c3c" if p < 60 else "#f39c12" if p < 80 else "#2ecc71"
                for p in tp_pcts
            ],
            text=[f"{p:.1f}%" for p in tp_pcts],
            textposition="auto",
        ))
        fig_tp.update_layout(
            xaxis=dict(title="Mastery %", range=[0, 100]),
            yaxis=dict(automargin=True),
            height=max(200, len(sorted_topics) * 45),
            margin=dict(t=10, b=30, l=10, r=10),
        )
        st.plotly_chart(fig_tp)

        tp_rows = []
        for topic, d in sorted_topics:
            tp_rows.append({
                "Topic":    topic,
                "Mastery":  f"{d['mastery_pct']:.1f}%",
                "Correct":  d["correct"],
                "Total":    d["total"],
                "Sessions": d["attempts"],
                "Status":   "🔴 Review" if d["mastery_pct"] < 60
                            else "🟡 Improving" if d["mastery_pct"] < 80
                            else "🟢 Strong",
            })
        st.dataframe(pd.DataFrame(tp_rows))
        st.divider()

    # ── Areas to review ───────────────────────────────────────────
    st.subheader("⚠️ Areas to Review")
    weak = [
        h for h in history
        if _mastery_pct(h["correct"], h["correct"] + h["incorrect"]) < 60.0
    ]
    if weak:
        for h in weak[-3:]:
            t   = h["correct"] + h["incorrect"]
            pct = _mastery_pct(h["correct"], t)
            topic_short = (h.get("topic") or "General")[:70]
            st.markdown(
                f"- **{h['label']}** — {h['correct']}/{t} ({pct:.1f}%) "
                f"— _{topic_short}_"
            )
    else:
        st.success("Great work! No weak sessions detected. Keep it up!")

    # ── Coach's Insight ───────────────────────────────────────────
    if _WEAK_DB_AVAILABLE and st.session_state.get("logged_in"):
        st.divider()
        st.subheader("🧑‍🏫 Coach's Insight")

        _coach_cards = _load_flashcards(st.session_state.student_name)

        if not _coach_cards:
            st.info("Complete more quizzes to unlock personalized coaching insights.")
        else:
            # Group all wrong-answer interactions by topic
            _stopwords = {
                "what", "which", "when", "does", "have", "that", "this", "with",
                "from", "your", "their", "where", "how", "why", "the", "and",
                "for", "are", "was", "not", "can", "will", "its", "into",
            }
            _topic_failures: dict[str, list] = {}
            for _card in _coach_cards:
                _t = (_card.get("topic") or "").strip()
                if not _t:
                    # Keyword fallback: first 3 meaningful words from question text
                    _words = [
                        w.strip("?,.!")
                        for w in _card.get("question_text", "").split()[:10]
                        if len(w) > 3 and w.lower().strip("?,.!") not in _stopwords
                    ]
                    _t = " ".join(_words[:3]) if _words else "General"
                _topic_failures.setdefault(_t, []).append(_card)

            # Only surface concepts with 3+ failures
            _struggling = {
                t: cards
                for t, cards in _topic_failures.items()
                if len(cards) >= 3
            }

            if not _struggling:
                st.success(
                    "🎉 No single concept has been missed 3+ times. "
                    "You're spreading your knowledge well — keep it up!"
                )
            else:
                st.markdown(
                    f"**{len(_struggling)} concept(s)** flagged for targeted review "
                    f"(each missed 3+ times):"
                )
                st.markdown("")
                for _ci, (_concept, _concept_cards) in enumerate(sorted(
                    _struggling.items(), key=lambda x: -len(x[1])
                )):
                    _miss_count = len(_concept_cards)
                    _src_files  = list({
                        c.get("source_file", "")
                        for c in _concept_cards
                        if c.get("source_file")
                    })
                    _col_info, _col_btn = st.columns([3, 1])
                    with _col_info:
                        st.markdown(
                            f"🔴 **{_concept[:60]}** — missed **{_miss_count}×**"
                        )
                        if _src_files:
                            st.caption(
                                "Source(s): "
                                + ", ".join(s[:40] for s in _src_files[:2])
                            )
                    with _col_btn:
                        _btn_key = f"coach_review_{_ci}"
                        if st.button(
                            "🎯 Force Review",
                            key=_btn_key,
                            use_container_width=True,
                            help=f"3-question mini-quiz targeting '{_concept[:40]}'",
                        ):
                            st.session_state["_coach_review_concept"] = _concept
                            st.session_state["_coach_review_sources"] = _src_files or None
                            st.session_state.quiz_num_questions       = 3
                            st.session_state.quiz_difficulty          = "Medium"
                            st.session_state.quiz_source_filter       = _src_files
                            st.session_state["_quiz_auto_generate"]   = True
                            st.session_state.page                     = "Quiz"
                            st.rerun()


# ══════════════════════════════════════════════════════════════════
# FLASHCARD PAGE
# ══════════════════════════════════════════════════════════════════

def _render_flashcard_page():
    st.header("🃏 Weakness Flashcards")
    st.caption("Questions you previously answered incorrectly — review them to close knowledge gaps.")

    if not st.session_state.get("logged_in"):
        st.info("Please log in to view your flashcards.")
        return

    if not _WEAK_DB_AVAILABLE:
        st.warning("Database not available — flashcards require a MongoDB connection.")
        return

    username = st.session_state.student_name
    cards    = _load_flashcards(username)

    if not cards:
        st.success("🎉 No weak points recorded yet. Keep taking quizzes to build your flashcard deck!")
        return

    # ── Deduplicate by question text, count repeat misses ─────────
    _seen: dict[str, dict] = {}
    for c in cards:
        q = c["question_text"]
        if not q:
            continue
        if q not in _seen:
            _seen[q] = {**c, "miss_count": 1}
        else:
            _seen[q]["miss_count"] += 1
    unique_cards = list(_seen.values())

    # ── Enrich old records that lack correct_answer_text ──────────
    # New records store the text directly; old ones only have the letter.
    # For those, batch-look up per_question from the parent attempt.
    if _DB_AVAILABLE:
        _missing_ids = list({
            c["attempt_id"]
            for c in unique_cards
            if not c.get("correct_answer_text") and c.get("attempt_id")
        })
        _fallback: dict[str, str] = (
            get_attempt_question_details(username, _missing_ids)
            if _missing_ids else {}
        )
        for c in unique_cards:
            if not c.get("correct_answer_text"):
                c["correct_answer_text"] = _fallback.get(c["question_text"], "")

    # ── Filter controls ───────────────────────────────────────────
    _all_sources = sorted({c["source_file"] for c in unique_cards if c["source_file"]})
    _all_diffs   = [d for d in ["Easy", "Medium", "Hard"]
                    if any(c["difficulty"] == d for c in unique_cards)]

    fc1, fc2 = st.columns(2)
    with fc1:
        _src_filter = st.multiselect(
            "Filter by Chapter / PDF",
            options=_all_sources,
            default=[],
            placeholder="All chapters",
            key="fc_source_filter",
        )
    with fc2:
        _diff_filter = st.multiselect(
            "Filter by Difficulty",
            options=_all_diffs,
            default=[],
            placeholder="All difficulties",
            key="fc_diff_filter",
        )

    if _src_filter:
        unique_cards = [c for c in unique_cards if c["source_file"] in _src_filter]
    if _diff_filter:
        unique_cards = [c for c in unique_cards if c["difficulty"] in _diff_filter]

    if not unique_cards:
        st.warning("No flashcards match the selected filters.")
        return

    # Sort: most-missed first
    unique_cards.sort(key=lambda c: c["miss_count"], reverse=True)

    st.markdown(f"**{len(unique_cards)} unique question(s)** to review")
    st.divider()

    # ── Difficulty badge helper ────────────────────────────────────
    _DIFF_COLOR = {"Easy": "#2ecc71", "Medium": "#f39c12", "Hard": "#e74c3c"}

    # ── Render cards grouped by source ────────────────────────────
    _by_source: dict[str, list] = {}
    for c in unique_cards:
        src = c["source_file"] or "Unknown Source"
        _by_source.setdefault(src, []).append(c)

    for source, group in sorted(_by_source.items()):
        st.subheader(f"📄 {source}")
        for idx, card in enumerate(group, start=1):
            diff   = card["difficulty"]
            color  = _DIFF_COLOR.get(diff, "#95a5a6")
            misses = card["miss_count"]
            badge  = (
                f'<span style="background:{color};color:#fff;padding:2px 8px;'
                f'border-radius:4px;font-size:0.8em;font-weight:600">{diff}</span>'
            )
            miss_badge = (
                f'<span style="background:#e74c3c;color:#fff;padding:2px 8px;'
                f'border-radius:4px;font-size:0.8em">✗ {misses}×</span>'
                if misses > 1 else ""
            )
            st.markdown(
                f"**Q{idx}.** {card['question_text']} &nbsp;{badge} {miss_badge}",
                unsafe_allow_html=True,
            )
            with st.expander("Reveal correct answer"):
                _letter = card["correct_answer"]
                _text   = card.get("correct_answer_text", "")
                if _text:
                    st.success(f"**{_letter})** {_text}")
                else:
                    st.success(f"**Correct answer: {_letter}**")
                if card.get("topic"):
                    st.caption(f"Topic: _{card['topic']}_")
        st.divider()


def _render_history_page():
    st.header("📜 History")

    tab_quiz, tab_chat = st.tabs(["🎯 Quiz History", "💬 Chat History"])

    with tab_quiz:
        history = st.session_state.get("quiz_history", [])
        if history:
            rows = []
            for h in reversed(history):
                t   = h["correct"] + h["incorrect"]
                pct = h.get("percentage") or (round(h["correct"] / t * 100, 1) if t else 0.0)
                rows.append({
                    "Session":    h["label"],
                    "Topic":      (h.get("topic") or "")[:60],
                    "Difficulty": h.get("difficulty", ""),
                    "Score":      f"{h['correct']}/{t}",
                    "Accuracy":   f"{pct:.1f}%",
                    "Date":       h.get("timestamp", ""),
                })
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("No quiz attempts yet. Generate a quiz from the sidebar!")

    with tab_chat:
        if not _CHAT_DB_AVAILABLE:
            st.info("Chat history requires database connectivity.")
            return
        if not st.session_state.get("logged_in"):
            st.info("Please log in to view chat history.")
            return

        try:
            sessions = get_chat_sessions(st.session_state.student_name)
        except Exception as e:
            st.error(f"Could not load chat history: {e}")
            return

        if not sessions:
            st.info("No saved chat sessions yet. Start a conversation on the Chat page!")
            return

        for s in sessions:
            label = (
                f"Chat — {s.get('started_at', 'Unknown')} "
                f"({s.get('message_count', 0)} messages)"
            )
            with st.expander(label):
                for msg in s.get("messages", []):
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

def _persist_chat() -> None:
    """Upsert the current messages into MongoDB after every assistant turn."""
    if not _CHAT_DB_AVAILABLE or not st.session_state.get("logged_in"):
        return
    session_id = st.session_state.get("chat_session_id")
    if not session_id:
        return
    try:
        upsert_active_session(
            st.session_state.student_name,
            session_id,
            st.session_state.messages,
        )
    except Exception:
        pass  # never block the UI for a persistence failure


def _save_current_chat() -> None:
    """Finalize the active session (called on New Chat / Logout)."""
    if not _CHAT_DB_AVAILABLE or not st.session_state.get("logged_in"):
        return
    session_id = st.session_state.get("chat_session_id")
    if not session_id:
        return
    try:
        finalize_session(st.session_state.student_name, session_id)
    except Exception:
        pass


def _render_sidebar():
    st.sidebar.write(f"ADMIN_AVAILABLE: {_ADMIN_AVAILABLE}")
    st.sidebar.title("📚 DataPilot")
    st.sidebar.caption("AI-Powered Virtual TA + Quiz Master")

    if not st.session_state.session_started:
        return

    st.sidebar.markdown(f"**👤 {st.session_state.student_name}**")

    _streak = _compute_streak(st.session_state.get("quiz_history", []))
    if _streak >= 5:
        st.sidebar.success(f"🔥 **{_streak}-Day Study Streak!**")
    elif _streak > 1:
        st.sidebar.caption(f"📅 {_streak}-day streak — keep it up!")

    st.sidebar.divider()

    # ── Navigation ─────────────────────────────────────────────────
    st.sidebar.subheader("Navigation")
    current_page = st.session_state.get("page", "Chat")
    for icon, page_name in [("💬", "Chat"), ("🎯", "Quiz"), ("📊", "Dashboard"), ("🃏", "Flashcards"), ("📜", "History")]:
        btn_type = "primary" if current_page == page_name else "secondary"
        if st.sidebar.button(
            f"{icon}  {page_name}",
            use_container_width=True,
            type=btn_type,
            key=f"nav_{page_name}",
        ):
            st.session_state.page = page_name
            st.rerun()

    if st.session_state.get("is_admin") and _ADMIN_AVAILABLE:
        btn_type = "primary" if current_page == "Admin Panel" else "secondary"
        if st.sidebar.button(
            "🔐  Admin Panel",
            use_container_width=True,
            type=btn_type,
            key="nav_Admin Panel",
        ):
            st.session_state.page = "Admin Panel"
            st.rerun()

    st.sidebar.divider()

    # ── Knowledge Base ─────────────────────────────────────────────
    st.sidebar.subheader("📚 Knowledge Base")

    if st.session_state.collection is None and st.session_state.api_key_set:
        chroma_path = Path(str(CHROMA_DB_PATH))
        if chroma_path.exists() and any(chroma_path.iterdir()):
            try:
                st.session_state.collection = _get_chroma_collection(str(chroma_path))
                stats = get_collection_stats(st.session_state.collection)
                st.session_state.ready = True
                st.sidebar.success(f"✅ {stats['total_chunks']} chunks loaded")
            except Exception as e:
                st.sidebar.error(f"KB load failed: {e}")
        else:
            st.sidebar.warning("⚠️ No knowledge base found.")

    if st.session_state.ready and st.session_state.collection:
        try:
            stats = get_collection_stats(st.session_state.collection)
            st.sidebar.caption(f"📄 {stats['total_chunks']} chunks indexed")
        except Exception:
            pass

    raw_pdfs = sorted(_RAW_DIR.glob("*.pdf")) if _RAW_DIR.exists() else []
    if raw_pdfs:
        st.sidebar.caption(f"Found {len(raw_pdfs)} PDF(s) in data/raw/")
        if st.sidebar.button("⚡ Ingest / Re-index PDFs", use_container_width=True):
            st.session_state._ingesting = True
            st.rerun()
    else:
        st.sidebar.caption("No PDFs in data/raw/ — add course files there.")

    st.sidebar.divider()

    # ── Utility ────────────────────────────────────────────────────
    if st.sidebar.button("🗑️ New Chat", use_container_width=True):
        _save_current_chat()
        st.session_state.messages           = []
        st.session_state.last_response_meta = None
        st.session_state["_pending_query"]  = None
        st.session_state.chat_session_id    = str(uuid.uuid4())
        st.session_state.page               = "Chat"
        st.rerun()

    if st.sidebar.button("🔄 Reset Quiz Stats", use_container_width=True):
        _show_reset_dialog()

    st.sidebar.divider()

    # ── Status ─────────────────────────────────────────────────────
    st.sidebar.subheader("⚙️ Status")
    st.sidebar.markdown(
        f"- **Model:** {'✅ Active' if st.session_state.api_key_set else '❌ Not configured'}"
    )
    st.sidebar.markdown(
        f"- **KB:** {'✅ Loaded' if st.session_state.collection else '❌ Empty'}"
    )
    st.sidebar.markdown(
        f"- **DB:** {'✅ Connected' if _DB_AVAILABLE else '❌ Offline'}"
    )
    st.sidebar.caption(
        f"Threshold: {SIMILARITY_THRESHOLD} | Top-K: {TOP_K_RESULTS} | Quiz K: {QUIZ_TOP_K}"
    )

    st.sidebar.divider()
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        _save_current_chat()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# PAGE ROUTER
# ══════════════════════════════════════════════════════════════════

def _render_page():
    if not st.session_state.session_started:
        _render_login_screen()
        return

    # Ingestion triggered from sidebar button
    if st.session_state.get("_ingesting"):
        st.session_state._ingesting = False
        _run_inline_ingestion()
        return

    page = st.session_state.get("page", "Chat")
    if page == "Chat":
        _render_chat_page()
    elif page == "Quiz":
        _render_quiz_page()
    elif page == "Dashboard":
        _render_dashboard_page()
    elif page == "Flashcards":
        _render_flashcard_page()
    elif page == "History":
        _render_history_page()
    elif page == "Admin Panel" and st.session_state.get("is_admin") and _ADMIN_AVAILABLE:
        _render_admin_panel()


# ══════════════════════════════════════════════════════════════════
# KB METADATA DIRECT ANSWER
# ══════════════════════════════════════════════════════════════════

def _answer_kb_metadata_query() -> str:
    """
    Answer "how many PDFs / what documents do you have?" directly from
    ChromaDB metadata — bypasses RAG entirely to prevent hallucination.
    """
    collection = st.session_state.get("collection")
    if collection is None:
        return "The knowledge base is not loaded yet."
    try:
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        source_files = sorted({
            m.get("source_file", "")
            for m in metadatas
            if m and m.get("source_file")
        })
        if not source_files:
            return "The knowledge base appears to be empty."
        lines = [f"I have access to **{len(source_files)} document(s)** in my knowledge base:\n"]
        for i, sf in enumerate(source_files, 1):
            lines.append(f"{i}. `{sf}`")
        lines.append(
            f"\nAll {len(source_files)} document(s) are indexed and available "
            "for answering your questions."
        )
        return "\n".join(lines)
    except Exception as e:
        return f"Could not query knowledge base metadata: {e}"


# ══════════════════════════════════════════════════════════════════
# CHAT QUERY EXECUTOR
# ══════════════════════════════════════════════════════════════════

def _execute_chat_query(pending: str) -> None:
    """
    Classify, retrieve, generate, and append to st.session_state.messages.

    Called from _render_chat_page() INSIDE a visible "Thinking…" assistant
    bubble so the student always has feedback during the API call.
    This function must NOT call any st.* rendering functions — it only
    mutates session state.
    """
    source_filter = st.session_state.get("quiz_source_filter") or None

    try:
        classification = classify_query(pending)

        # KB_METADATA: answered directly from ChromaDB — no RAG, no hallucination
        if classification.query_class == QueryClass.KB_METADATA:
            answer      = _answer_kb_metadata_query()
            result_meta = {
                "citations":       [],
                "reasoning_trace": "KB_METADATA direct lookup — no RAG",
                "was_refused":     False,
                "was_uncertain":   False,
                "query_class":     "kb_metadata",
                "confidence":      classification.confidence,
            }
            st.session_state.messages.append({
                "role":           "assistant",
                "content":        answer,
                "metadata":       result_meta,
                "is_placeholder": False,
            })
            st.session_state.last_response_meta = result_meta
            _persist_chat()
            return

        # CONVERSATIONAL: greetings / small-talk — skip RAG entirely
        if classification.query_class == QueryClass.CONVERSATIONAL:
            payload          = build_conversational_context()
            retrieved_chunks = []
        else:
            # Truncate to first 500 chars for retrieval — for pasted MCQs the
            # question stem is all we need; full option text adds noise and can
            # exceed safe embedding API limits.
            _retrieval_query = pending[:500] if len(pending) > 500 else pending
            retrieved_chunks = retrieve(
                _retrieval_query,
                st.session_state.collection,
                os.environ.get("GEMINI_API_KEY", ""),
                TOP_K_RESULTS,
                SIMILARITY_THRESHOLD,
                source_filter=source_filter,
            )

            if classification.query_class == QueryClass.DIRECT_ANSWER_REQUEST:
                payload = build_refusal_context(retrieved_chunks, pending)
            elif not retrieved_chunks:
                payload = build_empty_context()
            else:
                payload = build_context(retrieved_chunks)

        # Build conversation history for multi-turn context
        recent_messages: list[dict] = []
        for m in st.session_state.messages[-10:]:
            if m.get("is_placeholder", False):
                continue
            content = m.get("content", "")
            role    = m.get("role", "")
            if not content.strip() or content in (
                "⏳ Thinking…",
                "⚠️ Error occurred. Please try again.",
            ):
                continue
            if role == "user":
                recent_messages.append({"role": "user", "content": content})
            elif role == "assistant" and m.get("metadata") and len(content) > 20:
                recent_messages.append({"role": "assistant", "content": content[:1500]})
        recent_messages = recent_messages[-8:]

        response = generate_response(
            pending, payload, classification,
            st.session_state.model,
            conversation_history=recent_messages,
        )

        # Guard: never let a blank answer reach the UI
        _answer = response.answer.strip() if response.answer else ""
        if not _answer:
            _answer = (
                "I'm here to help! Try asking a question about your course materials, "
                "or say **'quiz me'** to start a practice session."
            )

        result_meta = {
            "citations":       response.citations,
            "reasoning_trace": response.reasoning_trace,
            "was_refused":     response.was_refused,
            "was_uncertain":   response.was_uncertain,
            "query_class":     response.query_class,
            "confidence":      response.confidence,
        }
        st.session_state.messages.append({
            "role":           "assistant",
            "content":        _answer,
            "metadata":       result_meta,
            "is_placeholder": False,
        })
        st.session_state.last_response_meta = result_meta
        _persist_chat()

    except Exception as exc:
        err_type = _classify_gemini_error(exc)
        if err_type == "server_busy":
            user_msg = (
                "⚠️ The AI is currently under high demand. "
                "Your question has been noted — please try again in a moment."
            )
        elif err_type == "rate_limit":
            user_msg = (
                "⚠️ API rate limit reached. Please wait 30 seconds and ask again."
            )
        elif err_type == "timeout":
            user_msg = (
                "⚠️ The AI took too long to respond. "
                "Check your network connection and try again."
            )
        else:
            user_msg = "⚠️ An error occurred. Please try again."

        err_meta = {
            "citations":       [],
            "reasoning_trace": type(exc).__name__,
            "was_refused":     False,
            "was_uncertain":   False,
            "query_class":     "error",
            "confidence":      0.0,
        }
        st.session_state.messages.append({
            "role":           "assistant",
            "content":        user_msg,
            "metadata":       err_meta,
            "is_placeholder": False,
        })
        st.session_state.last_response_meta = err_meta


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    _inject_css()
    _init_session_state()
    _render_sidebar()
    _render_page()


if __name__ == "__main__":
    main()
