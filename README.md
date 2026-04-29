# 🧠 DataPilot — Adaptive AI Virtual TA for USJ Big Data Engineering students

> Built specifically for **USJ (Université Saint-Joseph) Data Science students** studying the *Introduction to Big Data* course — an intelligent, context-aware learning platform that forces conceptual mastery through Retrieval-Augmented Generation (RAG), adaptive quizzes, and targeted weakness coaching.

---

## 🚀 The Vision (Problem & Solution)

### The Problem

Static studying methods fail to prepare students for high-stakes technical exams. Reading slides passively, re-watching lectures, and Googling answers all share the same fatal flaw: they allow students to feel productive without building genuine, testable understanding.

### The Solution

DataPilot acts as a relentless, intelligent tutor that **refuses to give direct answers**, forcing conceptual mastery through dynamic quizzes, spaced repetition, and targeted weakness coaching. RAG responses are grounded in course materials with source citations — every answer, every hint, and every explanation references a specific document, page, and confidence score.

---

## ✨ Core Capabilities (v2.0 Adaptive Engine)

### 🚨 Exam Crunch Mode

When a student toggles *"I have an exam soon!"*, the system overrides all source filters and scans the entire ingested knowledge base with a broad, high-yield retrieval query. The Gemini prompt is injected with an `EXAM CRUNCH MODE` block that instructs the model to prioritise:

- **Key definitions** of core terms
- **Core architectures** and their components
- **Comparison questions** (X vs Y trade-offs, advantages vs disadvantages)
- **Cause-and-effect relationships** between concepts

The retrieval budget is doubled (`QUIZ_TOP_K × 2`) to maximise content coverage. If the knowledge base yields partial content, the system degrades gracefully — appending a synthesis note to the context instead of hard-failing — so the student always receives a quiz.

### 🧠 Smart Adaptive Difficulty

The difficulty radio in the Quiz Setup Wizard includes a **🤖 Auto-Adjust** option. Selecting it triggers `_recommend_difficulty()`, which analyses the student's last 5 quiz sessions:

| Average Accuracy | Recommended Difficulty |
|---|---|
| > 85% | Hard |
| 50–85% | Medium |
| < 50% | Easy |

The resolved difficulty is shown to the student before generation and stored against the quiz attempt in MongoDB, keeping all historical analytics mathematically consistent.

### 🎯 Coach's Insight (Aggressive Weakness Targeting)

The Dashboard's **Coach's Insight** section aggregates the student's entire `weak_interactions` history from MongoDB and groups wrong answers by topic. Any concept missed **3 or more times** surfaces with a **🎯 Force Review** button. Clicking it:

1. Stores the failing concept in session state as `_coach_review_concept`
2. Routes to the Quiz page and fires `_quiz_auto_generate`
3. Calls `_generate_quiz_via_rag` with `topic_override` set to the struggling concept, injecting a `REMEDIATION FOCUS` block into the Gemini prompt that forces every question to directly re-test that specific concept

This creates a closed feedback loop: fail a concept → Coach surfaces it → targeted mini-quiz → mastery tracked.

### 🛡️ Academic Integrity Guardrails

Every incoming query is classified by `query_classifier.py` before any retrieval occurs:

| Class | Action |
|---|---|
| `CONCEPTUAL_HELP` | Full RAG-grounded explanation with citations |
| `ADMINISTRATIVE` | Direct answer from course metadata |
| `DIRECT_ANSWER_REQUEST` | Hard refusal + one conceptual hint + source citation |
| `KB_METADATA` | Direct ChromaDB metadata lookup (bypasses LLM entirely) |
| `CONVERSATIONAL` | Warm greeting, no retrieval |

The `DIRECT_ANSWER_REQUEST` classifier catches pasted MCQs, "solve this for me" requests, and "complete my homework" patterns via regex pattern matching — no LLM call is made, so the classification itself cannot be manipulated by prompt injection.

### 🏆 The Scholar's Ascent

Built-in gamification drives consistent study habits:

- **Rank System** — Dynamic rank badge computed from overall mastery percentage: 🥉 The Novice (0–40%), 🥈 The Scholar (41–70%), 🥇 The Sage (71–90%), 💎 The Master (91–100%)
- **Study Streaks** — Consecutive-day quiz activity tracked from ISO timestamps; a 🔥 streak banner fires in the sidebar at 5+ days
- **Mastery Seals** — Chapter bars capped at 100% are rendered in gold with a 🏆 seal; the status column shows `🏆 Mastered!` in the breakdown table

---

## 🏗️ Technical Architecture

### Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit (Python) — stateful, multi-page UI with JS-injected sticky sidebar |
| **Database** | MongoDB Atlas — user profiles, quiz attempts, chat sessions, weak interactions |
| **LLM (Generation)** | `gemini-3.1-flash-lite-preview` (primary), `gemini-2.5-flash` (fallback) |
| **LLM (Embeddings)** | `gemini-embedding-001` — 3072-dimension dense vectors |
| **Vector Store** | ChromaDB (HNSW index, persistent local client) |
| **PDF Parsing** | PyMuPDF (`pymupdf`) |
| **Auth** | bcrypt password hashing via `pymongo` |
| **Visualisation** | Plotly + Pandas |

### RAG Pipeline

```
Student Query
    │
    ▼
query_classifier.py          ← Regex pattern-matched; no LLM call
    │
    ├─ KB_METADATA ──────────► ChromaDB metadata lookup (no RAG)
    ├─ CONVERSATIONAL ───────► Warm response (no RAG)
    ├─ DIRECT_ANSWER ────────► build_refusal_context() → Gemini (hint only)
    └─ CONCEPTUAL_HELP ──────┐
                             ▼
                     retriever.py
                     gemini-embedding-001 → ChromaDB HNSW
                     Similarity threshold: 0.3 | Top-K: 10
                             │
                             ▼
                     context_builder.py
                     Diversity-aware chunk selection (max 6 chunks,
                     guaranteed ≥1 chunk per source file)
                             │
                             ▼
                     generator.py
                     gemini-3.1-flash-lite-preview
                     MASTER_SYSTEM_INSTRUCTION + context + conversation history
                             │
                             ▼
                     Grounded response + source citations
```

### MongoDB Collections

| Collection | Purpose |
|---|---|
| `users` | Username + bcrypt-hashed password |
| `quiz_attempts` | Per-attempt scores, difficulty, topic, source citations, per-question results |
| `chat_sessions` | Full message history per session, active/finalized state |
| `weak_interactions` | Every incorrect answer: question text, correct answer, topic, source file, difficulty |

### Key Configuration (`config.py`)

| Parameter | Value |
|---|---|
| Similarity threshold | `0.3` |
| Chunk size | `800` characters |
| Top-K retrieval | `10` |
| Max context chunks | `6` |
| Embedding dimensions | `3072` |

---

## 📊 Performance Benchmarks

Benchmarked across 10 Introduction to Big Data questions, 10/10 successful:

| Stage | Min | Max | Avg |
|---|---|---|---|
| Embedding | 1009.7ms | 3449.3ms | 2280.0ms |
| Retrieval (ChromaDB) | 2.4ms | 8.2ms | 4.6ms |
| Generation (Gemini) | 8324.9ms | 22364.9ms | 13242.9ms |
| Total End-to-End | 9615.6ms | 25580.3ms | 15527.6ms |

ChromaDB HNSW retrieval averages under 5ms. Embedding and generation times reflect network latency to the Google Gemini API.

**RAG Quality Evaluation (20 questions, Introduction to Big Data course)**

| Metric | Score | Interpretation |
|---|---|---|
| Answer Relevancy | 0.97 / 1.0 | Excellent |
| Faithfulness | 0.64 / 1.0 | Good |
| Context Precision | 0.48 / 1.0 | Acceptable |

---

## ⚙️ Local Setup & Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Studybuddy
```

### 2. Create a virtual environment

```bash
python -m venv studybuddy/.venv

# Windows
studybuddy\.venv\Scripts\activate

# macOS / Linux
source studybuddy/.venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r studybuddy/requirements.txt
```

### 4. Configure environment variables

Create `studybuddy/.env` with the following keys:

```env
# Google Gemini API key (https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_api_key_here

# MongoDB Atlas connection string
# Format: mongodb+srv://<username>:<password>@<cluster>.mongodb.net/
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/datapilot
```

> **For Streamlit Cloud deployment**, add these same keys to `.streamlit/secrets.toml` instead. The app reads `st.secrets` with priority over `.env`.

### 5. Add course PDFs

Place your lecture slides and course documents in:

```
Studybuddy/data/raw/
```

### 6. Run the application

```bash
studybuddy\.venv\Scripts\streamlit.exe run studybuddy/src/ui/app.py
```

On first launch, click **⚡ Ingest / Re-index PDFs** in the sidebar to parse, embed, and index your documents into ChromaDB. This is a one-time operation per document set.

---

## 🔒 Security & Stability Patches Applied

The following issues were identified during a codebase audit and patched before submission:

| ID | Severity | What was fixed |
|---|---|---|
| B1 | Medium | `retriever.py` default similarity threshold corrected from `0.75` to `0.3` to match `config.py` |
| B3 | High | Three silent `except Exception` import blocks now print the actual error to the server terminal, making startup failures diagnosable |
| B4 | High | Exception tracebacks removed from session state and UI rendering; only the error class name is stored, preventing leakage of internal file paths |
| B5 | High | `embed_query()` now raises a clear `ValueError` if `GEMINI_API_KEY` is missing before attempting an API call |
| S1 | High | Per-user quiz generation rate limit added: maximum 3 generations per 60-second rolling window, enforced across all three generation entry points (Generate Quiz, Weak Point Quiz, chat trigger) |
| Dead code | Low | `src/vectorstore/retriever.py` (a TODO-only stub) deleted to remove architectural confusion |
| Timer | Bug | Quiz countdown timer migrated from `st.html()` to `st.components.v1.html()` so the JavaScript actually executes; timer now uses an absolute deadline epoch instead of decrementing remaining seconds, so Streamlit reruns do not reset the clock |

---

## ⚠️ Known Limitations

- **Gemini API dependency** — All generation and embedding calls require an active internet connection and a valid `GEMINI_API_KEY`. There is no offline fallback.
- **ChromaDB is local** — The vector store lives on disk in `data/vectorstore/`. It is not shared between machines. Each deployment requires its own ingestion run.
- **PDF quality matters** — Scanned image-only PDFs with no embedded text layer will produce 0 chunks after parsing. Run ingestion and check the status panel to confirm chunk counts.
- **RAG responses are bounded by what was ingested** — If a concept does not appear in the uploaded PDFs, the system will acknowledge it cannot find it rather than answering from general knowledge (except in Exam Crunch Mode, where partial synthesis is permitted).
- **No real-time multi-user isolation** — The app is designed for single-user local deployment. Running it as a shared Streamlit Cloud app with multiple concurrent users will cause session state to bleed between users if they share the same server process.
- **Timer auto-submit requires browser JS** — The timed quiz countdown and auto-submit rely on `window.parent` DOM access from inside a `st.components.v1.html()` iframe. If the student's browser blocks cross-frame scripting, the auto-submit will not fire (the timer will still display correctly).
- **Adaptive difficulty needs quiz history** — The 🤖 Auto-Adjust option falls back to Medium if fewer than 1 completed session exists in history.
- **Coach's Insight threshold is fixed** — A concept must be missed exactly 3 or more times to surface. This cannot be changed from the UI.

---

## 🛠️ Troubleshooting

### "The knowledge base doesn't contain enough content to generate a quiz"

The ChromaDB collection is empty or the ingestion produced 0 chunks.

1. Check that your PDFs are in `data/raw/` and are not password-protected or image-only scans.
2. Click **⚡ Ingest / Re-index PDFs** in the sidebar.
3. Watch the ingestion status panel — each PDF should show a chunk count. Any PDF showing `0 chunks` failed to parse.
4. If a PDF consistently produces 0 chunks, open it in a PDF reader and confirm it contains selectable text (not a scanned image).

### "GEMINI_API_KEY not found" on startup

The `.env` file is missing, empty, or not in the right location.

1. Confirm `studybuddy/.env` exists (not `.env` in the project root).
2. Confirm it contains `GEMINI_API_KEY=...` with no extra spaces or quotes around the value.
3. For Streamlit Cloud, add the key under **Settings → Secrets** in the dashboard, not in a `.env` file.

### Database features show as offline / "DB unavailable"

The `MONGO_URI` is not set or the Atlas cluster is unreachable.

1. Confirm `MONGO_URI` is set in `studybuddy/.env`.
2. Check that your IP address is in the MongoDB Atlas **Network Access** allowlist (or set it to `0.0.0.0/0` for development).
3. On startup, the server terminal will now print the exact import error — check the terminal output for the specific failure reason.

### The quiz countdown timer shows but does not auto-submit when it reaches zero

The browser is blocking cross-frame JavaScript.

1. Try a different browser (Chrome and Edge are most compatible).
2. Disable any browser extensions that block scripts or iframes (uBlock Origin, NoScript, etc.).
3. As a fallback, the server-side `_timer_force_submit` flag will still mark the quiz for grading on the next user interaction even if the JS click does not fire.

### "You've generated 3 quizzes in the last minute"

The per-user rate limit has been reached to prevent runaway Gemini API usage.

1. Wait 60 seconds and try again.
2. The limit resets on a rolling basis — you do not need to wait a full minute from your first generation.

### Dashboard shows 0% mastery despite completed quizzes

Quiz history is stored in `st.session_state` during a session and persisted to MongoDB on submit. If you are not logged in, history is session-only and will be lost on page refresh.

1. Log in before taking quizzes to ensure all attempts are saved.
2. If you were logged in and history is still missing, click **🔄 Reset Quiz Stats** — if that shows data, your data is present and the issue is a display filter. Check whether a PDF filter is active on the Dashboard.

---

## 🎓 Academic Context

Developed as a comprehensive **final-year Big Data Engineering project at USJ**, DataPilot demonstrates end-to-end full-stack engineering across four disciplines:

- **Prompt Engineering** — Multi-phase system instructions, query classification, academic integrity enforcement, and adaptive quiz generation directives
- **Database Architecture** — Four-collection MongoDB schema with compound indexes optimised for mastery aggregation queries
- **Adaptive Algorithm Design** — Spaced-repetition targeting via `weak_interactions`, real-time difficulty adjustment from rolling session accuracy, and exam-mode content prioritisation
- **Production UI Engineering** — Streamlit session state management, JS DOM injection for sticky layout, deterministic quiz grading, multi-turn conversation persistence, and a self-correcting countdown timer using absolute deadline epochs

---

*Built with the Google Gemini API, ChromaDB, Streamlit, and MongoDB Atlas.*
