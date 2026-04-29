"""
studybuddy/scripts/ingest_course.py

Ingestion pipeline: PDF → parsed chunks → Gemini embeddings → ChromaDB.

Run from the project root (Studybuddy/):
    python studybuddy/scripts/ingest_course.py

Options:
    --force         Re-parse PDFs even if a processed JSON already exists.
    --skip-vision   Skip Gemini Vision for diagram pages (faster, text-only).
    --file NAME     Ingest a single PDF by filename instead of the whole directory.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent   # studybuddy/scripts/
PKG_DIR      = SCRIPT_DIR.parent                 # studybuddy/
PROJECT_ROOT = PKG_DIR.parent                    # Studybuddy/

sys.path.insert(0, str(PKG_DIR))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from config import CHROMA_DB_PATH, EMBEDDING_MODEL_NAME
from src.ingestion.pdf_parser import parse_pdf, load_parsed_chunks
from src.vectorstore.embedder import embed_chunks
from src.vectorstore.chroma_store import get_collection, upsert_chunks, get_collection_stats

# ── Directory constants ───────────────────────────────────────────────────────
RAW_PDF_DIR    = PROJECT_ROOT / "data" / "raw"      # where the PDFs live
PROCESSED_DIR  = PKG_DIR / "data" / "processed"    # parsed JSON output
COLLECTION_NAME = "studybuddy_knowledge_base"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _progress_bar(current: int, total: int, width: int = 28) -> str:
    filled = int(width * current / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {current}/{total}"


def _ingest_one(
    pdf_path: Path,
    api_key: str,
    collection,
    force: bool,
    skip_vision: bool,
) -> dict:
    """Parse, embed, and upsert a single PDF. Returns a stats dict."""
    stem      = pdf_path.stem
    json_path = PROCESSED_DIR / f"{stem}_parsed.json"
    t_start   = time.time()

    # ── Step 1: Parse ─────────────────────────────────────────────
    # A cached JSON with 0 chunks means a previous parse failed on a
    # corrupt file. Always re-parse if the cache is empty, regardless
    # of --force, so a replaced PDF is automatically re-ingested.
    cached_ok = False
    if json_path.exists() and not force:
        try:
            chunks = load_parsed_chunks(str(json_path))
        except Exception:
            chunks = []
        if chunks:
            print(f"    ↩  JSON cache valid ({len(chunks)} chunks) — skipping parse.")
            cached_ok = True
        else:
            print(f"    ⚠  Cached JSON is empty — deleting and re-parsing…")
            json_path.unlink(missing_ok=True)

    if not cached_ok:
        vision_key = "" if skip_vision else api_key
        label = "text-only" if skip_vision else "text + Gemini Vision"
        print(f"    ⚙  Parsing ({label})…", flush=True)
        parse_pdf(str(pdf_path), str(PROCESSED_DIR), api_key=vision_key)
        chunks = load_parsed_chunks(str(json_path))

    if not chunks:
        print(f"    ❌ 0 chunks extracted — skipping embed/upsert for this file.")
        return {"file": pdf_path.name, "chunks": 0, "upserted": 0,
                "elapsed": time.time() - t_start, "failed": True}

    print(f"    📄 {len(chunks)} chunks extracted")

    # ── Step 2: Embed ─────────────────────────────────────────────
    print(f"    🔢 Embedding ({EMBEDDING_MODEL_NAME})…", flush=True)
    batch_size    = 20
    total_batches = max(1, (len(chunks) + batch_size - 1) // batch_size)
    embedded      = []

    for b_idx in range(total_batches):
        batch    = chunks[b_idx * batch_size : (b_idx + 1) * batch_size]
        batch    = embed_chunks(batch, api_key=api_key)
        embedded.extend(batch)
        print(f"       {_progress_bar(b_idx + 1, total_batches)}  "
              f"({len(embedded)}/{len(chunks)} chunks)", flush=True)
        if b_idx < total_batches - 1:
            time.sleep(1.0)  # respect Gemini embedding rate limit

    # ── Step 3: Upsert ────────────────────────────────────────────
    print(f"    💾 Upserting to ChromaDB…", flush=True)
    n_upserted = upsert_chunks(collection, embedded)
    elapsed    = time.time() - t_start

    return {"file": pdf_path.name, "chunks": len(chunks),
            "upserted": n_upserted, "elapsed": elapsed}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF course materials into the DataPilot ChromaDB knowledge base."
    )
    parser.add_argument("--force",       action="store_true",
                        help="Re-parse PDFs even if a processed JSON already exists")
    parser.add_argument("--skip-vision", action="store_true",
                        help="Skip Gemini Vision (text extraction only, much faster)")
    parser.add_argument("--file",        type=str, default=None,
                        help="Ingest a single PDF by filename (e.g. '2-Batch Processing - HDFS.pdf')")
    args = parser.parse_args()

    # ── Preflight checks ──────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set. Add it to studybuddy/.env and retry.")
        sys.exit(1)

    if not RAW_PDF_DIR.exists():
        print(f"ERROR: PDF directory not found:\n  {RAW_PDF_DIR}")
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect PDFs ──────────────────────────────────────────────
    if args.file:
        target = RAW_PDF_DIR / args.file
        if not target.exists():
            print(f"ERROR: File not found:\n  {target}")
            sys.exit(1)
        pdf_files = [target]
    else:
        pdf_files = sorted(RAW_PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in:\n  {RAW_PDF_DIR}")
        sys.exit(0)

    # ── Header ────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  📚  DataPilot — Course Ingestion Pipeline")
    print("═" * 55)
    print(f"  Source    : {RAW_PDF_DIR}")
    print(f"  ChromaDB  : {CHROMA_DB_PATH}")
    print(f"  PDFs      : {len(pdf_files)}")
    print(f"  Vision    : {'disabled (--skip-vision)' if args.skip_vision else 'enabled'}")
    print(f"  Force     : {args.force}")
    print("═" * 55 + "\n")

    collection   = get_collection(str(CHROMA_DB_PATH), COLLECTION_NAME)
    before_count = get_collection_stats(collection)["total_chunks"]
    print(f"  Chunks in KB before ingestion: {before_count}\n")

    # ── Process each PDF ─────────────────────────────────────────
    results    = []
    total_time = 0.0

    for i, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
        result = _ingest_one(
            pdf_path, api_key, collection,
            force=args.force, skip_vision=args.skip_vision,
        )
        results.append(result)
        total_time += result["elapsed"]
        status = "❌ FAILED" if result.get("failed") else "✅ Done"
        print(f"    {status} in {result['elapsed']:.1f}s\n")

    # ── Summary ───────────────────────────────────────────────────
    after_count    = get_collection_stats(collection)["total_chunks"]
    total_chunks   = sum(r["chunks"]   for r in results)
    total_upserted = sum(r["upserted"] for r in results)
    failed_files   = [r["file"] for r in results if r.get("failed")]

    all_ok = not failed_files
    print("═" * 55)
    print(f"  {'✅' if all_ok else '⚠️ '}  Ingestion {'complete' if all_ok else 'finished with errors'}")
    print("─" * 55)
    for r in results:
        print(f"  {r['file']}")
        print(f"    chunks: {r['chunks']}  |  upserted: {r['upserted']}  |  {r['elapsed']:.1f}s")
    print("─" * 55)
    print(f"  Total chunks   : {total_chunks}")
    print(f"  Total upserted : {total_upserted}")
    print(f"  KB size after  : {after_count} chunks")
    print(f"  Total time     : {total_time:.1f}s")
    if failed_files:
        print("─" * 55)
        print("  ❌ FILES WITH 0 CHUNKS (check PDF quality and re-run):")
        for f in failed_files:
            print(f"     • {f}")
    print("═" * 55)
    print()
    print("  Start the app:")
    print(r"  studybuddy\.venv\Scripts\streamlit.exe run studybuddy/src/ui/app.py")
    print()


if __name__ == "__main__":
    main()
