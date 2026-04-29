"""
studybuddy/src/ingestion/pdf_parser.py

Parses PDF course materials into sentence-aware 800-character chunks
and saves structured JSON output to data/processed/.

Public API:
  parse_pdf(pdf_path, output_dir) -> str (path to saved JSON)
  load_parsed_chunks(json_path)   -> list[dict]
"""

import os
import json
import re
from datetime import datetime
import base64
import fitz
from pathlib import Path
from google import genai
from google.genai import types as genai_types

# ---------------------------------------------------------------------------
# Module-level constant — intentionally NOT imported from config.py so this
# module can run without a Gemini API key being present.
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800
VISION_CHUNK_SIZE        = 400
MIN_TEXT_FOR_TEXT_ONLY   = 50
VISION_MODEL             = "gemini-3.1-flash-lite-preview"




# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _describe_page_image(
    page, api_key: str, page_num: int, source_file: str
) -> str | None:
    """
    Renders a PDF page as an image and sends it to Gemini Vision
    for detailed content extraction. Returns None on any failure.
    """
    if not api_key:
        return None
    try:
        # Render page at 2x resolution for clarity
        mat      = fitz.Matrix(2.0, 2.0)
        pix      = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img_b64  = base64.b64encode(img_bytes).decode("utf-8")

        client   = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model    = VISION_MODEL,
            contents = [
                genai_types.Part.from_bytes(
                    data      = base64.b64decode(img_b64),
                    mime_type = "image/png",
                ),
                genai_types.Part.from_text(
                    "You are extracting educational content from a "
                    "university course PDF page for a search index.\n\n"
                    "CRITICAL INSTRUCTIONS:\n"
                    "1. List EVERY technology, tool, framework, "
                    "database, or software shown — write each name "
                    "explicitly (e.g. Hadoop, Cassandra, Spark, "
                    "HBase, MongoDB, MapReduce, HDFS, etc.)\n"
                    "2. For every diagram or chart: describe its "
                    "structure AND list all labeled items by their "
                    "exact names\n"
                    "3. For the '3 Vs' or any characteristics "
                    "diagram: name each characteristic explicitly\n"
                    "4. Transcribe all visible text: titles, bullets,"
                    " captions, labels\n"
                    "5. For tables: write each row as 'Column: Value'"
                    "\n\nFormat as a structured list. Each distinct "
                    "concept or technology on its own line.\n\n"
                    f"Page {page_num} of: {source_file}"
                ),
            ],
        )
        description = response.text.strip() if response.text else None
        return description if description else None

    except Exception:
        return None

def _clean_text(text: str) -> str:
    """Returns text with whitespace normalised and control characters removed."""
    # Remove Unicode control characters (ord < 32, except space = 32)
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch == " ")
    # Collapse all sequences of 2+ whitespace characters into a single space
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """Returns a list of sentence-aware chunks no larger than chunk_size chars."""
    # STEP 1 – split on sentence boundaries (keep punctuation in the sentence)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        prospective = (current_chunk + " " + sentence).strip() if current_chunk else sentence

        if len(prospective) <= chunk_size:
            # STEP 3a – fits: accumulate
            current_chunk = prospective
        elif current_chunk:
            # STEP 3b – doesn't fit and we have a chunk buffered: flush it
            chunks.append(current_chunk.strip())
            if len(sentence) > chunk_size:
                # STEP 3c – single sentence exceeds limit: append as-is
                _warning = (
                    f"WARNING: oversized chunk ({len(sentence)} chars) "
                    "from file appended as-is"
                )
                chunks.append(sentence)
                current_chunk = ""
            else:
                current_chunk = sentence
        else:
            # current_chunk is empty but sentence alone already exceeds limit
            _warning = (
                f"WARNING: oversized chunk ({len(sentence)} chars) "
                "from file appended as-is"
            )
            chunks.append(sentence)
            current_chunk = ""

    # STEP 4 – flush remaining buffer
    if current_chunk:
        chunks.append(current_chunk.strip())

    # STEP 5 – filter empty strings
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str, output_dir: str, api_key: str = "") -> str:
    """Returns the path to the saved JSON file produced from parsing pdf_path"""
    # STEP 1 – validate inputs
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists() or pdf_path_obj.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"Invalid PDF path: {pdf_path}")

    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    stem = pdf_path_obj.stem
    source_filename = pdf_path_obj.name
    parsed_at = datetime.utcnow().isoformat() + "Z"

    all_chunks = []
    total_pages = 0

    # STEP 2 – open PDF
    with fitz.open(str(pdf_path_obj)) as doc:
        total_pages = len(doc)

        # STEP 3 – extract text page by page
        for page_num, page in enumerate(doc, start=1):
            raw_text     = page.get_text("text")
            cleaned_text = _clean_text(raw_text)

            text_content   = cleaned_text if len(cleaned_text) >= 10 \
                             else ""
            vision_content = ""
            is_vision      = False

            if api_key:
                description = _describe_page_image(
                    page, api_key, page_num, Path(pdf_path).name
                )
                if description:
                    vision_content = description
                    is_vision      = True

            # Merge strategy
            if text_content and vision_content:
                page_text = (
                    f"[DIAGRAM AND VISUAL CONTENT — Page {page_num}]\n"
                    f"{vision_content}\n\n"
                    f"[SLIDE TEXT — Page {page_num}]\n"
                    f"{text_content}"
                )
            elif vision_content:
                page_text = vision_content
            elif text_content:
                page_text          = text_content
                is_vision          = False
            else:
                continue

            effective_chunk_size = VISION_CHUNK_SIZE if is_vision \
                                   else CHUNK_SIZE
            chunks_for_page      = chunk_text(page_text,
                                              effective_chunk_size)

            for chunk_idx, chunk_str in enumerate(chunks_for_page):
                if not chunk_str.strip():
                    continue
                record = {
                    "chunk_id":    f"{stem}_p{page_num}_c{chunk_idx}",
                    "chunk_text":  chunk_str,
                    "source_file": Path(pdf_path).name,
                    "page_number": page_num,
                    "timestamp":   None,
                    "source_tier": "primary",
                    "char_count":  len(chunk_str),
                    "created_at":  datetime.utcnow().isoformat() + "Z",
                    "is_vision_extracted": is_vision
                }
                all_chunks.append(record)

    # STEP 6 – build top-level JSON structure
    output_data = {
        "source_file": source_filename,
        "total_pages": total_pages,
        "total_chunks": len(all_chunks),
        "parsed_at": parsed_at,
        "chunks": all_chunks,
    }

    # STEP 7 – write to output_dir
    output_filename = f"{stem}_parsed.json"
    output_path = output_dir_obj / output_filename
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2, ensure_ascii=False)

    # STEP 8 – return path string
    return str(output_path)


def load_parsed_chunks(json_path: str) -> list:
    """Returns the list of chunk dicts from a previously saved *_parsed.json file."""
    json_path_obj = Path(json_path)
    if not json_path_obj.exists():
        raise FileNotFoundError(f"Parsed JSON not found: {json_path}")

    with open(json_path_obj, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    return data["chunks"]
