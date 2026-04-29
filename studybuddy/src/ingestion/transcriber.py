"""
studybuddy/src/ingestion/transcriber.py

Transcribes lecture audio/video files (MP4, MP3, WAV, etc.) into
sentence-aware 800-character chunks using faster-whisper ASR.
Produces structured JSON output to data/processed/ using the same
schema as pdf_parser.py for seamless downstream RAG compatibility.

Public API:
  transcribe_media(media_path, output_dir, model_size) -> str
  load_parsed_chunks(json_path)                        -> list[dict]
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel

CHUNK_SIZE = 800
SUPPORTED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg"}
DEFAULT_WHISPER_MODEL = "base"

def _format_timestamp(seconds: float) -> str:
    # Use integer division and modulo per specification
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def _clean_text(text: str) -> str:
    text = text.strip()
    # Remove Unicode control characters (ord < 32, except space = 32)
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch == " ")
    # Collapse 2+ whitespace characters into a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _chunk_segments(segments: list, chunk_size: int) -> list:
    current_text = ""
    current_start = None
    current_end = None
    chunks = []

    for segment in segments:
        cleaned_text = _clean_text(segment.text)
        if not cleaned_text:
            continue
            
        if current_start is None:
            current_start = segment.start
            current_text = cleaned_text
            current_end = segment.end
        else:
            prospective = current_text + " " + cleaned_text
            if len(prospective) <= chunk_size:
                current_text = prospective
                current_end = segment.end
            else:
                # Finalize the current chunk
                chunks.append({
                    "text": current_text.strip(),
                    "start_sec": current_start,
                    "end_sec": current_end,
                    "timestamp": _format_timestamp(current_start)
                })
                # Start a new chunk
                current_text = cleaned_text
                current_start = segment.start
                current_end = segment.end
                
    if current_text:
        chunks.append({
            "text": current_text.strip(),
            "start_sec": current_start,
            "end_sec": current_end,
            "timestamp": _format_timestamp(current_start)
        })
        
    return [c for c in chunks if c["text"]]

def transcribe_media(media_path: str, output_dir: str, model_size: str = DEFAULT_WHISPER_MODEL) -> str:
    media_path_obj = Path(media_path)
    if not media_path_obj.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")
        
    ext = media_path_obj.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")
        
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    
    # Load Whisper Model unconditionally with CPU and int8 
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    # Transcribe
    segments_gen, info = model.transcribe(media_path, beam_size=5, word_timestamps=False)
    segments = list(segments_gen)
    detected_language = info.language
    
    # Chunking
    chunk_dicts = _chunk_segments(segments, CHUNK_SIZE)
    
    stem = media_path_obj.stem
    source_filename = media_path_obj.name
    
    all_chunks = []
    created_at = datetime.utcnow().isoformat() + "Z"
    
    for i, chunk in enumerate(chunk_dicts):
        all_chunks.append({
            "chunk_id": f"{stem}_t{i}",
            "chunk_text": chunk["text"],
            "source_file": source_filename,
            "page_number": None,
            "timestamp": chunk["timestamp"],
            "start_sec": chunk["start_sec"],
            "end_sec": chunk["end_sec"],
            "source_tier": "primary",
            "char_count": len(chunk["text"]),
            "created_at": created_at
        })
        
    output_data = {
        "source_file": source_filename,
        "detected_language": detected_language,
        "duration_seconds": info.duration,
        "total_chunks": len(all_chunks),
        "model_used": model_size,
        "parsed_at": created_at,
        "chunks": all_chunks
    }
    
    output_filename = f"{stem}_transcribed.json"
    output_path = output_dir_obj / output_filename
    
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2, ensure_ascii=False)
        
    return str(output_path)

def load_parsed_chunks(json_path: str) -> list:
    json_path_obj = Path(json_path)
    if not json_path_obj.exists():
        raise FileNotFoundError(f"Transcription JSON not found: {json_path}")
        
    with open(json_path_obj, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        
    return data["chunks"]
