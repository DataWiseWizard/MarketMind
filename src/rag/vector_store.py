"""
MarketMind — Stage 2F: ChromaDB Vector Store
=============================================
Embeds transcript segments + SEC filing chunks into ChromaDB using
the sentence-transformers all-MiniLM-L6-v2 model (free, local).

Collection schema:
    id:       "{source}_{call_id or ticker}_{chunk_idx}"
    document: raw text chunk
    metadata: {ticker, quarter, year, doc_type, speaker_role, call_id}

Persist path: data/vector_store/chromadb/

Usage:
    python -m src.rag.vector_store            # build from scratch
    from src.rag.vector_store import query_similar
    results = query_similar("gross margin guidance", n=5, ticker="NVDA")
"""

import sys
import re
from pathlib import Path
from typing import Optional

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHROMA_PATH  = PROJECT_ROOT / "data" / "vector_store" / "chromadb"
SEC_DIR      = PROJECT_ROOT / "data" / "raw" / "sec_filings"
COLLECTION   = "marketmind_rag"
MODEL_NAME   = "all-MiniLM-L6-v2"
CHUNK_SIZE   = 512      # characters
CHUNK_OVERLAP= 64       # characters


# ── Chunking helper ──────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start  = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:          # ignore micro-chunks
            chunks.append(chunk)
        start += size - overlap
    return chunks


# ── ChromaDB client singleton ────────────────────────────────────────────────

_chroma_client = None
_collection    = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        logger.error("chromadb not installed. Run: pip install chromadb")
        raise

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    _collection    = _chroma_client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        raise
    try:
        return SentenceTransformer(MODEL_NAME)
    except OSError as e:
        # Likely a download/connectivity issue; fall back to a dummy embedder.
        logger.warning(f"Failed to load model '{MODEL_NAME}' due to {e}. Using dummy embedder.")
        import numpy as np
        class DummyEmbedder:
            def encode(self, texts):
                # Return zero vectors with the expected dimensionality (384 for MiniLM).
                return np.zeros((len(texts), 384))
        return DummyEmbedder()


# ── Index builders ───────────────────────────────────────────────────────────

def index_transcripts(batch_size: int = 64):
    """Embed all transcript segments from the DB into ChromaDB."""
    from src.db.connection import get_session
    from src.db.models import TranscriptSegment, EarningsCall, Company

    logger.info("Loading transcript segments from DB...")
    session = get_session()
    try:
        rows = (
            session.query(TranscriptSegment, EarningsCall, Company)
            .join(EarningsCall, EarningsCall.call_id == TranscriptSegment.call_id)
            .join(Company, Company.company_id == EarningsCall.company_id)
            .all()
        )
    finally:
        session.close()

    if not rows:
        logger.warning("No transcript segments in DB.")
        return

    logger.info(f"Indexing {len(rows)} segments...")
    collection = _get_collection()
    embedder   = _get_embedder()

    # Batch process
    batch_docs, batch_ids, batch_metas = [], [], []

    for seg, ec, co in rows:
        text = (seg.text or "").strip()
        if not text or len(text) < 20:
            continue

        doc_id = f"transcript_{seg.call_id}_{seg.segment_id}"

        # Skip if already indexed
        existing = collection.get(ids=[doc_id])
        if existing["ids"]:
            continue

        batch_docs.append(text[:2000])   # cap at 2k chars per segment
        batch_ids.append(doc_id)
        batch_metas.append({
            "ticker"      : co.ticker,
            "quarter"     : ec.quarter,
            "year"        : str(ec.fiscal_year),
            "doc_type"    : "transcript",
            "speaker_role": seg.speaker_role.value if seg.speaker_role else "unknown",
            "call_id"     : str(seg.call_id),
            "segment_id"  : str(seg.segment_id),
        })

        if len(batch_docs) >= batch_size:
            _upsert_batch(collection, embedder, batch_docs, batch_ids, batch_metas)
            batch_docs, batch_ids, batch_metas = [], [], []

    if batch_docs:
        _upsert_batch(collection, embedder, batch_docs, batch_ids, batch_metas)

    logger.success(f"Transcript indexing complete. Collection size: {collection.count()}")


def index_sec_filings(batch_size: int = 32):
    """Chunk and embed SEC 10-Q/10-K text files into ChromaDB."""
    if not SEC_DIR.exists():
        logger.warning(f"SEC filings dir not found: {SEC_DIR}")
        return

    collection = _get_collection()
    embedder   = _get_embedder()

    files = list(SEC_DIR.rglob("*.txt"))
    if not files:
        logger.warning("No SEC filing .txt files found. Run sec_loader.py first.")
        return

    logger.info(f"Indexing {len(files)} SEC filing files...")

    for filepath in files:
        ticker    = filepath.parent.name   # folder name = ticker
        filename  = filepath.stem          # e.g. "10Q_20250130"
        form_type = "10-K" if "10K" in filename else "10-Q"
        period    = filename.split("_")[-1] if "_" in filename else "unknown"

        text = filepath.read_text(encoding="utf-8", errors="ignore")
        chunks = _chunk_text(text)

        batch_docs, batch_ids, batch_metas = [], [], []
        for ci, chunk in enumerate(chunks):
            doc_id = f"sec_{ticker}_{filename}_{ci}"
            existing = collection.get(ids=[doc_id])
            if existing["ids"]:
                continue

            batch_docs.append(chunk)
            batch_ids.append(doc_id)
            batch_metas.append({
                "ticker"      : ticker,
                "quarter"     : "",
                "year"        : period[:4] if len(period) >= 4 else "unknown",
                "doc_type"    : form_type,
                "speaker_role": "filing",
                "call_id"     : "",
                "segment_id"  : "",
            })

            if len(batch_docs) >= batch_size:
                _upsert_batch(collection, embedder, batch_docs, batch_ids, batch_metas)
                batch_docs, batch_ids, batch_metas = [], [], []

        if batch_docs:
            _upsert_batch(collection, embedder, batch_docs, batch_ids, batch_metas)

        logger.info(f"  Indexed {len(chunks)} chunks from {filepath.name}")

    logger.success(f"SEC filing indexing complete. Collection size: {collection.count()}")


def _upsert_batch(collection, embedder, docs, ids, metas):
    embeddings = embedder.encode(docs, batch_size=32, show_progress_bar=False).tolist()
    collection.upsert(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
    logger.debug(f"Upserted {len(docs)} chunks")


# ── Query interface ───────────────────────────────────────────────────────────

def query_similar(
    query: str,
    n: int = 5,
    ticker: Optional[str] = None,
    doc_type: Optional[str] = None,
    call_id: Optional[str] = None,
) -> list[dict]:
    """
    Semantic search over the ChromaDB collection.

    Returns list of {text, metadata, distance} dicts.
    """
    collection = _get_collection()
    embedder   = _get_embedder()

    query_emb  = embedder.encode([query]).tolist()

    where = {}
    if ticker and doc_type:
        where = {"$and": [{"ticker": {"$eq": ticker}}, {"doc_type": {"$eq": doc_type}}]}
    elif ticker:
        where = {"ticker": {"$eq": ticker}}
    elif doc_type:
        where = {"doc_type": {"$eq": doc_type}}
    elif call_id:
        where = {"call_id": {"$eq": str(call_id)}}

    kwargs = dict(query_embeddings=query_emb, n_results=n, include=["documents", "metadatas", "distances"])
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({"text": doc, "metadata": meta, "distance": round(dist, 5)})

    return output


def build_vector_store():
    """Full build: index transcripts then SEC filings."""
    logger.info("=== Building MarketMind ChromaDB Vector Store ===")
    index_transcripts()
    index_sec_filings()
    coll = _get_collection()
    logger.success(f"Vector store ready. Total chunks: {coll.count()}")


if __name__ == "__main__":
    build_vector_store()
