"""
src/retriever.py
ML2 — LLM Regulatory Financial Intelligence Agent

Hybrid retriever: FAISS (dense) + BM25 (lexical) with Reciprocal Rank Fusion.

Usage:
    from src.retriever import HybridRetriever
    r = HybridRetriever()
    results = r.retrieve("What are APRA capital requirements?", top_k=4)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

FAISS_PATH  = PROC_DIR / "index.faiss"
BM25_PATH   = PROC_DIR / "bm25.pkl"
CHUNKS_PATH = PROC_DIR / "chunks.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
RRF_K       = 60   # RRF constant — higher = smoother rank fusion


class HybridRetriever:
    """
    Hybrid retriever combining FAISS dense search and BM25 lexical search
    via Reciprocal Rank Fusion (RRF).

    RRF score = Σ 1 / (k + rank_i)  for each retrieval system i.
    Higher RRF score = better combined rank.
    """

    def __init__(
        self,
        faiss_path: Path  = FAISS_PATH,
        bm25_path: Path   = BM25_PATH,
        chunks_path: Path = CHUNKS_PATH,
        embed_model: str  = EMBED_MODEL,
        rrf_k: int        = RRF_K,
    ):
        self.rrf_k = rrf_k
        self._load_indexes(faiss_path, bm25_path, chunks_path)
        self._load_embedder(embed_model)

    # ── loading ───────────────────────────────────────────────────────────────

    def _load_indexes(
        self,
        faiss_path: Path,
        bm25_path: Path,
        chunks_path: Path,
    ) -> None:
        if not faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {faiss_path}. Run `python -m src.ingest` first."
            )
        self.faiss_index = faiss.read_index(str(faiss_path))
        logger.info("FAISS index loaded: %d vectors", self.faiss_index.ntotal)

        with open(bm25_path, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        logger.info("BM25 index loaded")

        self.chunks: list[dict[str, Any]] = json.loads(chunks_path.read_text())
        logger.info("Chunks loaded: %d", len(self.chunks))

    def _load_embedder(self, model_name: str) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self.embedder = SentenceTransformer(model_name)

    # ── embedding ─────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string, returning a normalised float32 vector."""
        raw = self.embedder.encode(
            [query],
            convert_to_numpy=False,
            normalize_embeddings=True,
        )
        return np.array([raw[0].tolist()], dtype=np.float32)

    # ── retrieval ─────────────────────────────────────────────────────────────

    def _faiss_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (chunk_index, score) pairs from FAISS dense search."""
        q_vec = self._embed_query(query)
        scores, indices = self.faiss_index.search(q_vec, top_k)
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (chunk_index, score) pairs from BM25 lexical search."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _reciprocal_rank_fusion(
        self,
        faiss_results: list[tuple[int, float]],
        bm25_results:  list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """
        Combine ranked lists via RRF.
        Returns (chunk_index, rrf_score) sorted descending.
        """
        rrf_scores: dict[int, float] = {}

        for rank, (idx, _) in enumerate(faiss_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        faiss_candidates: int = 10,
        bm25_candidates:  int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top_k most relevant chunks for a query using hybrid search.

        Parameters
        ----------
        query            : natural language query
        top_k            : number of final results to return after RRF
        faiss_candidates : how many candidates to pull from FAISS before fusion
        bm25_candidates  : how many candidates to pull from BM25 before fusion

        Returns
        -------
        List of chunk dicts with added 'rrf_score' and 'rank' fields.
        """
        faiss_results = self._faiss_search(query, faiss_candidates)
        bm25_results  = self._bm25_search(query,  bm25_candidates)

        fused = self._reciprocal_rank_fusion(faiss_results, bm25_results)

        results = []
        for rank, (chunk_idx, rrf_score) in enumerate(fused[:top_k]):
            chunk = dict(self.chunks[chunk_idx])
            chunk["rrf_score"] = round(rrf_score, 6)
            chunk["rank"]      = rank + 1
            results.append(chunk)

        logger.debug(
            "Query: %r → %d results (top rrf=%.4f)",
            query[:60], len(results),
            results[0]["rrf_score"] if results else 0,
        )
        return results

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 4,
    ) -> tuple[list[dict], list[float], list[float]]:
        """
        Extended retrieve returning individual FAISS and BM25 scores for
        analysis and evaluation.

        Returns
        -------
        chunks       : list of chunk dicts
        faiss_scores : dense similarity scores (same order as chunks)
        bm25_scores  : lexical scores (same order as chunks)
        """
        faiss_results = self._faiss_search(query, top_k * 3)
        bm25_results  = self._bm25_search(query,  top_k * 3)

        faiss_map = {idx: score for idx, score in faiss_results}
        bm25_map  = {idx: score for idx, score in bm25_results}

        fused = self._reciprocal_rank_fusion(faiss_results, bm25_results)
        top   = fused[:top_k]

        chunks, faiss_scores, bm25_scores = [], [], []
        for chunk_idx, rrf_score in top:
            chunk = dict(self.chunks[chunk_idx])
            chunk["rrf_score"] = round(rrf_score, 6)
            chunks.append(chunk)
            faiss_scores.append(faiss_map.get(chunk_idx, 0.0))
            bm25_scores.append(bm25_map.get(chunk_idx, 0.0))

        return chunks, faiss_scores, bm25_scores


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    retriever = HybridRetriever()

    test_queries = [
        "What are APRA capital adequacy requirements?",
        "How should suspicious transactions be reported to AUSTRAC?",
        "What is credit risk management framework?",
        "SAR narrative requirements for AML reporting",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        results = retriever.retrieve(query, top_k=3)
        for r in results:
            print(f"\n  [{r['rank']}] RRF={r['rrf_score']:.4f} | {r['source']} | chunk {r['chunk_index']}")
            print(f"  {r['text'][:200]}...")
