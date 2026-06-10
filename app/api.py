"""
app/api.py
ML2 — LLM Regulatory Financial Intelligence Agent

FastAPI service exposing:
    GET  /health
    POST /query          — regulatory Q&A
    POST /sar/generate   — SAR narrative generation

Usage:
    uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent import RegulatoryAgent

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML2 — Regulatory Financial Intelligence Agent",
    description="LangGraph agent for regulatory Q&A and SAR narrative generation using local Ollama LLM",
    version="1.0.0",
)

_agent: RegulatoryAgent | None = None


@app.on_event("startup")
def startup():
    logging.basicConfig(level=logging.INFO)
    global _agent
    _agent = RegulatoryAgent()
    logger.info("Agent loaded and ready.")


# ── schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Regulatory or financial question")

    class Config:
        json_schema_extra = {"example": {"question": "What are APRA's capital adequacy requirements?"}}


class QueryResponse(BaseModel):
    answer:       str
    sources:      list[str]
    chunks_used:  int
    latency_ms:   float


class SARRequest(BaseModel):
    customer:         str   = Field(..., description="Customer or entity name")
    account:          str   = Field("N/A", description="Account identifier")
    period:           str   = Field(..., description="Reporting period e.g. 'November 2024'")
    amount:           float = Field(..., gt=0, description="Transaction amount in AUD")
    currency:         str   = Field("AUD")
    transaction_type: str   = Field("cash deposits", description="Type of transaction")
    flags:            list[str] = Field(..., min_length=1, description="AML indicator flags triggered")
    notes:            str   = Field("", description="Additional context")

    class Config:
        json_schema_extra = {"example": {
            "customer": "XYZ Trading Pty Ltd",
            "account": "ACC-2024-8871",
            "period": "November 2024",
            "amount": 95000,
            "currency": "AUD",
            "transaction_type": "cash deposits",
            "flags": ["structuring", "cash-intensive transactions"],
            "notes": "Three deposits on consecutive days just below $32K each."
        }}


class SARResponse(BaseModel):
    sar_narrative:   str
    risk_indicators: list[str]
    latency_ms:      float


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "llama3.2:3b", "agent": "ready" if _agent else "loading"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not _agent:
        raise HTTPException(503, "Agent not ready")
    t0 = time.perf_counter()
    try:
        result = _agent.query(req.question)
    except Exception as e:
        logger.exception("Query error")
        raise HTTPException(500, str(e))
    return QueryResponse(
        answer      = result["answer"],
        sources     = result["sources"],
        chunks_used = len(result["chunks"]),
        latency_ms  = round((time.perf_counter() - t0) * 1000, 1),
    )


@app.post("/sar/generate", response_model=SARResponse)
def generate_sar(req: SARRequest):
    if not _agent:
        raise HTTPException(503, "Agent not ready")
    t0 = time.perf_counter()
    try:
        result = _agent.generate_sar(req.model_dump())
    except Exception as e:
        logger.exception("SAR generation error")
        raise HTTPException(500, str(e))
    return SARResponse(
        sar_narrative   = result["sar_narrative"],
        risk_indicators = result["risk_indicators"],
        latency_ms      = round((time.perf_counter() - t0) * 1000, 1),
    )
