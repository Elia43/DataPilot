"""
studybuddy/src/vectorstore/chroma_store.py

Manages the ChromaDB persistent vector store for DataPilot.
Handles collection initialization, chunk upsert, and cosine
similarity querying with tier-weighted result ranking.

Public API:
  get_collection(db_path, collection_name)       -> chromadb.Collection
  upsert_chunks(collection, embedded_chunks)     -> int
  query_collection(collection, query_embedding,
                   top_k, threshold)             -> list[dict]
  get_collection_stats(collection)               -> dict
"""

import chromadb
from chromadb.config import Settings

TIER_WEIGHTS = {
    "primary":   1.0,
    "secondary": 0.85,
    "tertiary":  0.70
}

DEFAULT_COLLECTION_NAME = "studybuddy_knowledge_base"


def get_collection(db_path: str, collection_name: str = DEFAULT_COLLECTION_NAME) -> chromadb.Collection:
    try:
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        return collection
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChromaDB collection: {e}")


def upsert_chunks(collection, embedded_chunks: list[dict]) -> int:
    if not embedded_chunks:
        return 0
        
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for chunk in embedded_chunks:
        if "embedding" not in chunk or not chunk["embedding"]:
            raise ValueError(f"Chunk {chunk.get('chunk_id', 'unknown')} is missing its embedding vector.")
            
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        documents.append(chunk["chunk_text"])
        
        metadatas.append({
            "source_file":  chunk["source_file"],
            "page_number":  chunk["page_number"] if chunk["page_number"] is not None else -1,
            "timestamp":    chunk["timestamp"] if chunk["timestamp"] is not None else "",
            "start_sec":    chunk.get("start_sec", -1.0),
            "end_sec":      chunk.get("end_sec", -1.0),
            "source_tier":  chunk["source_tier"],
            "char_count":   chunk["char_count"],
            "created_at":   chunk["created_at"]
        })
        
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    return len(embedded_chunks)


def query_collection(
    collection,
    query_embedding: list[float],
    top_k: int = 10,
    threshold: float = 0.75,
    where_filter: dict | None = None,
) -> list[dict]:
    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results":        top_k,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    results = collection.query(**kwargs)
    
    if not results["documents"] or not results["documents"][0]:
        return []
        
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    passing_results = []
    
    for doc, metadata, dist in zip(documents, metadatas, distances):
        raw_similarity = 1.0 - dist
        
        if raw_similarity < threshold:
            continue
            
        tier = metadata.get("source_tier", "primary")
        tier_weight = TIER_WEIGHTS.get(tier, 1.0)
        weighted_score = raw_similarity * tier_weight
        
        # Reverse None encoding for metadatas
        pg_num = metadata.get("page_number")
        if pg_num == -1:
            pg_num = None
            
        ts = metadata.get("timestamp")
        if ts == "":
            ts = None
            
        passing_results.append({
            "chunk_text":      doc,
            "source_file":     metadata.get("source_file"),
            "page_number":     pg_num,
            "timestamp":       ts,
            "source_tier":     tier,
            "raw_similarity":  round(raw_similarity, 4),
            "weighted_score":  round(weighted_score, 4),
            "confidence_pct":  int(raw_similarity * 100)
        })
        
    passing_results.sort(key=lambda x: x["weighted_score"], reverse=True)
    
    return passing_results


def get_collection_stats(collection) -> dict:
    try:
        return {
            "total_chunks":     collection.count(),
            "collection_name":  collection.name
        }
    except Exception as e:
        return {
            "total_chunks": -1,
            "collection_name": "unknown",
            "error": str(e)
        }
