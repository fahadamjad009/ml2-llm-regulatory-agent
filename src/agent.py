"""
src/agent.py
ML2 — LLM Regulatory Financial Intelligence Agent

LangGraph agent with two modes:
  1. Financial/Regulatory Q&A  — retrieve → generate → cite
  2. SAR Narrative Generator   — extract_flags → generate_sar → format

Usage:
    from src.agent import RegulatoryAgent
    agent = RegulatoryAgent()

    # Q&A mode
    result = agent.query("What are APRA capital requirements?")

    # SAR mode
    result = agent.generate_sar({
        "customer": "ABC Corp",
        "flags": ["structuring", "rapid fund movement"],
        "amount": 95000,
        "period": "2024-Q4"
    })
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)

OLLAMA_MODEL    = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
TOP_K           = 4


# ── state definitions ─────────────────────────────────────────────────────────

class QAState(TypedDict):
    """State for the Q&A agent graph."""
    query:    str
    chunks:   list[dict]
    answer:   str
    sources:  list[str]
    mode:     str          # "qa" or "sar"


class SARState(TypedDict):
    """State for the SAR generator graph."""
    transaction_data: dict[str, Any]
    regulatory_context: list[dict]
    sar_narrative: str
    risk_indicators: list[str]


# ── LLM ───────────────────────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.1) -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Q&A AGENT
# ══════════════════════════════════════════════════════════════════════════════

QA_SYSTEM_PROMPT = """You are a regulatory and financial compliance expert assistant.
You answer questions about financial regulations, prudential standards, AML/CTF requirements,
and corporate financial disclosures.

RULES:
- Answer only based on the provided context chunks.
- If the context does not contain enough information to answer, say so clearly.
- Be precise and cite specific regulatory standards when mentioned in the context.
- Keep answers concise but complete — typically 3-6 sentences.
- Do not hallucinate regulatory requirements not present in the context.
"""

def qa_retrieve_node(state: QAState) -> QAState:
    """Retrieve relevant chunks for the query."""
    retriever = HybridRetriever()
    chunks = retriever.retrieve(state["query"], top_k=TOP_K)
    logger.info("Retrieved %d chunks for query: %r", len(chunks), state["query"][:60])
    return {**state, "chunks": chunks}


def qa_generate_node(state: QAState) -> QAState:
    """Generate an answer grounded in retrieved chunks."""
    llm = _get_llm()

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}"
        for c in state["chunks"]
    )

    messages = [
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(content=f"""Context:
{context}

Question: {state['query']}

Answer based only on the context above:"""),
    ]

    logger.info("Generating answer with %s ...", OLLAMA_MODEL)
    response = llm.invoke(messages)
    answer   = response.content.strip()

    sources = list({c["source"] for c in state["chunks"]})
    logger.info("Answer generated (%d chars)", len(answer))

    return {**state, "answer": answer, "sources": sources}


def build_qa_graph() -> Any:
    """Build and compile the Q&A LangGraph."""
    graph = StateGraph(QAState)
    graph.add_node("retrieve", qa_retrieve_node)
    graph.add_node("generate", qa_generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# SAR GENERATOR AGENT
# ══════════════════════════════════════════════════════════════════════════════

SAR_SYSTEM_PROMPT = """You are a compliance officer specialising in Anti-Money Laundering (AML)
and Counter-Terrorism Financing (CTF) reporting under Australian AUSTRAC regulations.

Your task is to draft Suspicious Activity Report (SAR) narratives that are:
- Factual and objective
- Structured according to AUSTRAC requirements
- Clear about the basis for suspicion
- Free of speculation beyond the provided transaction data

NARRATIVE STRUCTURE:
1. Subject identification
2. Nature and description of suspicious activity
3. Transaction details (amounts, dates, patterns)
4. Basis for suspicion (which AML indicators were triggered)
5. Action taken / recommended

Keep the narrative professional and suitable for regulatory submission.
"""

def sar_context_node(state: SARState) -> SARState:
    """Retrieve relevant regulatory context for SAR generation."""
    retriever = HybridRetriever()

    # Build a query from the transaction flags
    flags_str = ", ".join(state["transaction_data"].get("flags", []))
    query = f"SAR narrative requirements suspicious activity {flags_str} AUSTRAC reporting"

    chunks = retriever.retrieve(query, top_k=3)
    logger.info("Retrieved %d regulatory context chunks for SAR", len(chunks))
    return {**state, "regulatory_context": chunks}


def sar_generate_node(state: SARState) -> SARState:
    """Generate the SAR narrative using transaction data + regulatory context."""
    llm = _get_llm(temperature=0.2)

    td = state["transaction_data"]
    reg_context = "\n\n".join(c["text"] for c in state["regulatory_context"])

    # Extract risk indicators from flags
    risk_indicators = td.get("flags", [])

    transaction_summary = f"""
Customer/Entity:  {td.get('customer', 'Unknown')}
Account:          {td.get('account', 'N/A')}
Period:           {td.get('period', 'N/A')}
Amount(s):        AUD {td.get('amount', 'N/A'):,} {td.get('currency', '')}
Transaction type: {td.get('transaction_type', 'N/A')}
Flags triggered:  {', '.join(risk_indicators)}
Additional notes: {td.get('notes', 'None')}
""".strip()

    messages = [
        SystemMessage(content=SAR_SYSTEM_PROMPT),
        HumanMessage(content=f"""Regulatory Reference:
{reg_context}

Transaction Data:
{transaction_summary}

Draft a complete SAR narrative for AUSTRAC submission:"""),
    ]

    logger.info("Generating SAR narrative with %s ...", OLLAMA_MODEL)
    response = llm.invoke(messages)
    narrative = response.content.strip()
    logger.info("SAR narrative generated (%d chars)", len(narrative))

    return {**state, "sar_narrative": narrative, "risk_indicators": risk_indicators}


def build_sar_graph() -> Any:
    """Build and compile the SAR generator LangGraph."""
    graph = StateGraph(SARState)
    graph.add_node("get_context", sar_context_node)
    graph.add_node("generate_sar", sar_generate_node)
    graph.set_entry_point("get_context")
    graph.add_edge("get_context", "generate_sar")
    graph.add_edge("generate_sar", END)
    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

class RegulatoryAgent:
    """
    Unified interface for both agent modes.

    Methods
    -------
    query(question)          → {"answer", "sources", "chunks"}
    generate_sar(tx_data)    → {"sar_narrative", "risk_indicators"}
    """

    def __init__(self):
        logger.info("Building agent graphs...")
        self._qa_graph  = build_qa_graph()
        self._sar_graph = build_sar_graph()
        logger.info("Agents ready.")

    def query(self, question: str) -> dict[str, Any]:
        """
        Answer a regulatory or financial question using hybrid RAG.

        Returns
        -------
        dict with keys: answer (str), sources (list), chunks (list)
        """
        state: QAState = {
            "query":   question,
            "chunks":  [],
            "answer":  "",
            "sources": [],
            "mode":    "qa",
        }
        result = self._qa_graph.invoke(state)
        return {
            "answer":  result["answer"],
            "sources": result["sources"],
            "chunks":  result["chunks"],
        }

    def generate_sar(self, transaction_data: dict[str, Any]) -> dict[str, Any]:
        """
        Generate a SAR narrative from transaction data.

        Parameters
        ----------
        transaction_data : dict with keys:
            customer        : entity name
            account         : account identifier (optional)
            period          : reporting period
            amount          : transaction amount
            currency        : currency code (default AUD)
            transaction_type: e.g. "cash deposit", "wire transfer"
            flags           : list of AML indicator strings
            notes           : additional context (optional)

        Returns
        -------
        dict with keys: sar_narrative (str), risk_indicators (list)
        """
        state: SARState = {
            "transaction_data":   transaction_data,
            "regulatory_context": [],
            "sar_narrative":      "",
            "risk_indicators":    [],
        }
        result = self._sar_graph.invoke(state)
        return {
            "sar_narrative":   result["sar_narrative"],
            "risk_indicators": result["risk_indicators"],
        }


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    agent = RegulatoryAgent()

    # Test 1: Q&A
    print("\n" + "="*60)
    print("TEST 1: Regulatory Q&A")
    print("="*60)
    result = agent.query("What are the key requirements of APRA's credit risk management framework?")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")

    # Test 2: SAR generation
    print("\n" + "="*60)
    print("TEST 2: SAR Narrative Generation")
    print("="*60)
    tx_data = {
        "customer":         "XYZ Trading Pty Ltd",
        "account":          "ACC-2024-8871",
        "period":           "November 2024",
        "amount":           95000,
        "currency":         "AUD",
        "transaction_type": "cash deposits",
        "flags":            ["structuring", "cash-intensive transactions", "inconsistency with business profile"],
        "notes":            "Three separate cash deposits of $31,000, $32,000, and $32,000 made on consecutive days. Customer is registered as a software consultancy."
    }
    sar_result = agent.generate_sar(tx_data)
    print(f"\nSAR Narrative:\n{sar_result['sar_narrative']}")
    print(f"\nRisk Indicators: {sar_result['risk_indicators']}")
