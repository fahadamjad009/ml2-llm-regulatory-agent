"""
src/ingest.py
ML2 — LLM Regulatory Financial Intelligence Agent

Document ingestion pipeline:
  - Loads PDFs and .txt files from data/sample_docs/
  - Chunks with overlap
  - Builds FAISS dense index (MiniLM-L6-v2 embeddings)
  - Builds BM25 lexical index
  - Saves both indexes to data/processed/

Usage:
    python -m src.ingest
    python -m src.ingest --docs-dir data/sample_docs --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pdfplumber
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
DOCS_DIR  = ROOT / "data" / "sample_docs"
PROC_DIR  = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH    = PROC_DIR / "index.faiss"
BM25_PATH     = PROC_DIR / "bm25.pkl"
CHUNKS_PATH   = PROC_DIR / "chunks.json"
MANIFEST_PATH = PROC_DIR / "manifest.json"

# ── constants ─────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 512     # characters
CHUNK_OVERLAP = 64
EMBED_DIM     = 384


# ── text extraction ───────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> str:
    """Extract text from a PDF file using pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text.strip())
    except Exception as e:
        logger.warning("Failed to extract %s: %s", path.name, e)
    return "\n\n".join(text_parts)


def extract_txt(path: Path) -> str:
    """Read a plain text file."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Failed to read %s: %s", path.name, e)
        return ""


def extract_document(path: Path) -> str:
    """Dispatch extraction by file type."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    elif suffix in (".txt", ".md"):
        return extract_txt(path)
    else:
        logger.debug("Unsupported file type: %s", path.suffix)
        return ""


# ── chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split text into overlapping character chunks.
    Returns list of dicts: {id, text, source, chunk_index}.
    """
    chunks = []
    start = 0
    idx = 0
    text = text.strip()

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if len(chunk) > 50:  # skip tiny fragments
            chunk_id = hashlib.md5(f"{source}_{idx}".encode()).hexdigest()[:12]
            chunks.append({
                "id":          chunk_id,
                "text":        chunk,
                "source":      source,
                "chunk_index": idx,
            })
            idx += 1

        start += chunk_size - overlap

    return chunks


# ── embedding + indexing ──────────────────────────────────────────────────────

def build_indexes(
    chunks: list[dict[str, Any]],
    embed_model_name: str = EMBED_MODEL,
) -> tuple[faiss.IndexFlatIP, BM25Okapi]:
    """
    Build FAISS (dense) and BM25 (lexical) indexes from chunks.

    Returns
    -------
    faiss_index : IndexFlatIP (inner product on L2-normalised vectors = cosine sim)
    bm25_index  : BM25Okapi
    """
    texts = [c["text"] for c in chunks]

    # ── FAISS ─────────────────────────────────────────────────────────────────
    logger.info("Encoding %d chunks with %s ...", len(texts), embed_model_name)
    t0 = time.time()
    model = SentenceTransformer(embed_model_name)
    # encode with convert_to_numpy=False to avoid torch/numpy bridge issue
    raw_embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=False,
        normalize_embeddings=True,
    )
    embeddings = np.array([e.tolist() for e in raw_embeddings], dtype=np.float32)
    logger.info("Encoded in %.1fs", time.time() - t0)

    faiss_index = faiss.IndexFlatIP(EMBED_DIM)
    faiss_index.add(embeddings)
    logger.info("FAISS index: %d vectors", faiss_index.ntotal)

    # ── BM25 ──────────────────────────────────────────────────────────────────
    tokenized = [t.lower().split() for t in texts]
    bm25_index = BM25Okapi(tokenized)
    logger.info("BM25 index: %d documents", len(tokenized))

    return faiss_index, bm25_index


# ── file manifest ─────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_manifest() -> dict[str, str]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict[str, str]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


# ── main entry point ──────────────────────────────────────────────────────────

def ingest(docs_dir: Path = DOCS_DIR, force: bool = False) -> int:
    """
    Ingest all supported documents in docs_dir.
    Skips files unchanged since last run (based on MD5 manifest) unless force=True.

    Returns number of chunks indexed.
    """
    supported = (".pdf", ".txt", ".md")
    files = [f for f in docs_dir.iterdir() if f.suffix.lower() in supported]

    if not files:
        logger.warning("No supported documents found in %s", docs_dir)
        logger.info("Add .pdf or .txt files to %s and re-run.", docs_dir)
        return 0

    logger.info("Found %d documents in %s", len(files), docs_dir)

    manifest = {} if force else load_manifest()
    all_chunks: list[dict[str, Any]] = []

    # Load existing chunks if incremental
    if not force and CHUNKS_PATH.exists():
        existing = json.loads(CHUNKS_PATH.read_text())
        # keep chunks from unchanged files
        changed_sources = set()
        for f in files:
            fhash = _file_hash(f)
            if manifest.get(f.name) != fhash:
                changed_sources.add(f.name)
                manifest[f.name] = fhash
        all_chunks = [c for c in existing if c["source"] not in changed_sources]
        process_files = [f for f in files if f.name in changed_sources]
        logger.info(
            "Incremental: %d unchanged files, %d to reprocess",
            len(files) - len(process_files), len(process_files),
        )
    else:
        process_files = files
        for f in files:
            manifest[f.name] = _file_hash(f)

    # Process changed / new files
    for path in process_files:
        logger.info("Processing: %s", path.name)
        text = extract_document(path)
        if not text.strip():
            logger.warning("Empty text from %s — skipping.", path.name)
            continue
        chunks = chunk_text(text, source=path.name)
        logger.info("  → %d chunks from %s", len(chunks), path.name)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.error("No chunks produced. Check your documents.")
        return 0

    logger.info("Total chunks: %d", len(all_chunks))

    # Build indexes
    faiss_index, bm25_index = build_indexes(all_chunks)

    # Save everything
    faiss.write_index(faiss_index, str(FAISS_PATH))
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_index, f)
    CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False))
    save_manifest(manifest)

    logger.info("Saved FAISS index → %s", FAISS_PATH)
    logger.info("Saved BM25 index  → %s", BM25_PATH)
    logger.info("Saved chunks      → %s", CHUNKS_PATH)
    logger.info("Ingestion complete: %d chunks across %d documents",
                len(all_chunks), len(files))

    return len(all_chunks)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Ingest documents into FAISS + BM25 indexes")
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild indexes from scratch ignoring manifest")
    args = parser.parse_args()

    n = ingest(docs_dir=args.docs_dir, force=args.force)
    print(f"\nIngested {n} chunks successfully." if n else "\nNo chunks produced.")
