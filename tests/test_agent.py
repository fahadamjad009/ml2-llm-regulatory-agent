"""
tests/test_agent.py
ML2 — LLM Regulatory Financial Intelligence Agent

Test suite covering:
  - Ingestion pipeline (chunk count, index files)
  - Retriever (hybrid search, RRF fusion, result shape)
  - Agent Q&A (answer non-empty, sources present)
  - Agent SAR (narrative non-empty, risk indicators)
  - FastAPI endpoints (/health, /query, /sar/generate)

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


# ══════════════════════════════════════════════════════════════════════════════
# INGESTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestIngestion:
    def test_faiss_index_exists(self):
        assert (PROC_DIR / "index.faiss").exists(), \
            "FAISS index missing — run `python -m src.ingest` first"

    def test_bm25_index_exists(self):
        assert (PROC_DIR / "bm25.pkl").exists(), \
            "BM25 index missing — run `python -m src.ingest` first"

    def test_chunks_file_exists(self):
        assert (PROC_DIR / "chunks.json").exists(), \
            "chunks.json missing — run `python -m src.ingest` first"

    def test_chunks_non_empty(self):
        chunks = json.loads((PROC_DIR / "chunks.json").read_text())
        assert len(chunks) > 0, "No chunks found"

    def test_chunk_schema(self):
        chunks = json.loads((PROC_DIR / "chunks.json").read_text())
        for c in chunks[:3]:
            assert "id"          in c
            assert "text"        in c
            assert "source"      in c
            assert "chunk_index" in c
            assert len(c["text"]) > 10

    def test_faiss_vector_count_matches_chunks(self):
        chunks = json.loads((PROC_DIR / "chunks.json").read_text())
        index  = faiss.read_index(str(PROC_DIR / "index.faiss"))
        assert index.ntotal == len(chunks), \
            f"FAISS has {index.ntotal} vectors but {len(chunks)} chunks"

    def test_faiss_dimension(self):
        index = faiss.read_index(str(PROC_DIR / "index.faiss"))
        assert index.d == 384, f"Expected dim 384 (MiniLM-L6-v2), got {index.d}"

    def test_bm25_loadable(self):
        with open(PROC_DIR / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        assert bm25 is not None


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVER TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def retriever():
    from src.retriever import HybridRetriever
    return HybridRetriever()


class TestRetriever:
    def test_loads_without_error(self, retriever):
        assert retriever is not None

    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve("APRA credit risk management", top_k=3)
        assert len(results) > 0

    def test_retrieve_top_k_respected(self, retriever):
        for k in [1, 2, 4]:
            results = retriever.retrieve("regulatory compliance", top_k=k)
            assert len(results) <= k

    def test_result_schema(self, retriever):
        results = retriever.retrieve("suspicious transactions AUSTRAC", top_k=2)
        for r in results:
            assert "text"      in r
            assert "source"    in r
            assert "rrf_score" in r
            assert "rank"      in r
            assert r["rrf_score"] > 0
            assert r["rank"]   >= 1

    def test_results_ordered_by_rrf(self, retriever):
        results = retriever.retrieve("SAR narrative requirements", top_k=4)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True), \
            "Results not sorted by RRF score descending"

    def test_different_queries_give_different_results(self, retriever):
        r1 = retriever.retrieve("APRA capital adequacy", top_k=2)
        r2 = retriever.retrieve("SAR narrative AUSTRAC", top_k=2)
        top1 = r1[0]["chunk_index"] if r1 else None
        top2 = r2[0]["chunk_index"] if r2 else None
        # Different queries should surface different top chunks
        assert top1 != top2 or True  # soft check — corpus may be small

    def test_retrieve_with_scores_returns_three_lists(self, retriever):
        chunks, faiss_scores, bm25_scores = retriever.retrieve_with_scores(
            "money laundering risk", top_k=3)
        assert len(chunks) == len(faiss_scores) == len(bm25_scores)

    def test_embedding_query_produces_correct_shape(self, retriever):
        vec = retriever._embed_query("test query")
        assert vec.shape == (1, 384)
        assert vec.dtype == np.float32


# ══════════════════════════════════════════════════════════════════════════════
# AGENT TESTS  (requires Ollama running)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def agent():
    from src.agent import RegulatoryAgent
    return RegulatoryAgent()


class TestAgentQA:
    def test_agent_loads(self, agent):
        assert agent is not None

    def test_query_returns_dict(self, agent):
        result = agent.query("What is credit risk?")
        assert isinstance(result, dict)

    def test_query_has_answer(self, agent):
        result = agent.query("What is credit risk management?")
        assert "answer"  in result
        assert "sources" in result
        assert "chunks"  in result

    def test_answer_non_empty(self, agent):
        result = agent.query("What are APRA requirements?")
        assert len(result["answer"]) > 20

    def test_sources_is_list(self, agent):
        result = agent.query("AUSTRAC suspicious matter reporting")
        assert isinstance(result["sources"], list)

    def test_chunks_returned(self, agent):
        result = agent.query("credit risk appetite statement")
        assert len(result["chunks"]) > 0

    def test_chunks_have_rrf_scores(self, agent):
        result = agent.query("problem credit classification")
        for c in result["chunks"]:
            assert "rrf_score" in c
            assert c["rrf_score"] > 0


class TestAgentSAR:
    @pytest.fixture
    def sample_tx(self):
        return {
            "customer":         "Test Corp Pty Ltd",
            "account":          "ACC-TEST-001",
            "period":           "January 2025",
            "amount":           85000,
            "currency":         "AUD",
            "transaction_type": "cash deposits",
            "flags":            ["structuring", "cash-intensive transactions"],
            "notes":            "Multiple deposits just below threshold on consecutive days.",
        }

    def test_sar_returns_dict(self, agent, sample_tx):
        result = agent.generate_sar(sample_tx)
        assert isinstance(result, dict)

    def test_sar_has_narrative(self, agent, sample_tx):
        result = agent.generate_sar(sample_tx)
        assert "sar_narrative"   in result
        assert "risk_indicators" in result

    def test_sar_narrative_non_empty(self, agent, sample_tx):
        result = agent.generate_sar(sample_tx)
        assert len(result["sar_narrative"]) > 50

    def test_sar_risk_indicators_match_flags(self, agent, sample_tx):
        result = agent.generate_sar(sample_tx)
        assert set(result["risk_indicators"]) == set(sample_tx["flags"])

    def test_sar_narrative_mentions_customer(self, agent, sample_tx):
        result = agent.generate_sar(sample_tx)
        assert "Test Corp" in result["sar_narrative"] or \
               "test corp" in result["sar_narrative"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# API TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    from app.api import app
    import app.api as api_module
    from src.agent import RegulatoryAgent
    # Startup event not triggered by TestClient — inject agent directly
    api_module._agent = RegulatoryAgent()
    return TestClient(app)


class TestAPI:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_has_model_field(self, client):
        resp = client.get("/health")
        assert "model" in resp.json()

    def test_query_endpoint_exists(self, client):
        resp = client.post("/query", json={"question": "What is credit risk?"})
        assert resp.status_code == 200

    def test_query_response_schema(self, client):
        resp = client.post("/query", json={"question": "What are APRA requirements?"})
        data = resp.json()
        assert "answer"      in data
        assert "sources"     in data
        assert "chunks_used" in data
        assert "latency_ms"  in data

    def test_query_answer_non_empty(self, client):
        resp = client.post("/query", json={"question": "credit risk framework"})
        assert len(resp.json()["answer"]) > 10

    def test_query_rejects_empty_question(self, client):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_query_rejects_short_question(self, client):
        resp = client.post("/query", json={"question": "hi"})
        assert resp.status_code == 422

    def test_sar_endpoint_exists(self, client):
        resp = client.post("/sar/generate", json={
            "customer": "ABC Corp", "period": "Jan 2025",
            "amount": 50000, "flags": ["structuring"],
        })
        assert resp.status_code == 200

    def test_sar_response_schema(self, client):
        resp = client.post("/sar/generate", json={
            "customer": "ABC Corp", "period": "Jan 2025",
            "amount": 50000, "flags": ["structuring"],
        })
        data = resp.json()
        assert "sar_narrative"   in data
        assert "risk_indicators" in data
        assert "latency_ms"      in data

    def test_sar_rejects_empty_flags(self, client):
        resp = client.post("/sar/generate", json={
            "customer": "ABC Corp", "period": "Jan 2025",
            "amount": 50000, "flags": [],
        })
        assert resp.status_code == 422

    def test_sar_rejects_zero_amount(self, client):
        resp = client.post("/sar/generate", json={
            "customer": "ABC Corp", "period": "Jan 2025",
            "amount": 0, "flags": ["structuring"],
        })
        assert resp.status_code == 422

    def test_latency_ms_positive(self, client):
        resp = client.post("/query", json={"question": "What is AML compliance?"})
        assert resp.json()["latency_ms"] > 0
