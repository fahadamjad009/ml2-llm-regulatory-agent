"""
app/ui.py  —  ML2 Regulatory Intelligence Agent
Matches portfolio v2: no sidebar, Inter/JetBrains Mono, teal accent,
interactive hover badges, card layout.
"""
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.agent import RegulatoryAgent

st.set_page_config(
    page_title="ML2 — Regulatory Intelligence Agent",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

*{box-sizing:border-box;margin:0;padding:0}
.stApp,[data-testid="stAppViewContainer"]{background:#0a0a0a !important;color:#ededed}
[data-testid="stSidebar"]{display:none !important}
section[data-testid="stSidebarContent"]{display:none !important}
button[kind="header"]{display:none !important}
#MainMenu,footer,header,.stDeployButton{visibility:hidden}
.block-container{padding:2rem 3rem;max-width:1100px;margin:0 auto}

/* ── fonts ── */
*{font-family:'Inter',sans-serif}
code,pre,.mono{font-family:'JetBrains Mono',monospace}

/* ── nav bar ── */
.navbar{display:flex;align-items:center;justify-content:space-between;
        padding:0 3rem;height:60px;background:rgba(10,10,10,.9);
        backdrop-filter:blur(12px);border-bottom:1px solid #1a1a1a;
        position:fixed;top:0;left:0;right:0;z-index:100}
.nav-logo{font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:600;
          color:#64ffda;letter-spacing:2px}
.nav-links{display:flex;gap:28px}
.nav-links a{color:#999;text-decoration:none;font-size:.82rem;font-weight:500;
             transition:color .2s;cursor:pointer}
.nav-links a:hover{color:#64ffda}

/* ── hero ── */
.hero{padding:100px 0 48px}
.hero-title{font-size:3rem;font-weight:900;color:#ffffff;letter-spacing:-2px;
            line-height:1.05;margin-bottom:16px}
.hero-sub{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#444;
          letter-spacing:3px;text-transform:uppercase;margin-bottom:24px}

/* ── tag badges (interactive) ── */
.tags{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:32px}
.tag{font-family:'JetBrains Mono',monospace;font-size:.65rem;font-weight:600;
     letter-spacing:1px;padding:5px 14px;border-radius:20px;
     background:rgba(100,255,218,.06);color:#64ffda;
     border:1px solid rgba(100,255,218,.15);cursor:default;
     transition:all .2s;display:inline-block}
.tag:hover{background:rgba(100,255,218,.15);border-color:#64ffda;
           transform:translateY(-1px);box-shadow:0 4px 16px rgba(100,255,218,.12)}

/* ── stat cards ── */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:40px}
.stat{background:#111;border:1px solid #1e1e1e;border-radius:8px;
      padding:20px;text-align:center;transition:border-color .2s, transform .2s}
.stat:hover{border-color:#64ffda;transform:translateY(-2px)}
.stat-label{font-family:'JetBrains Mono',monospace;font-size:.55rem;font-weight:600;
            color:#64ffda;letter-spacing:3px;text-transform:uppercase;margin-bottom:6px}
.stat-value{font-size:1.6rem;font-weight:800;color:#ffffff;line-height:1}
.stat-sub{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#555;margin-top:4px}

/* ── section heading ── */
.sh{font-family:'JetBrains Mono',monospace;font-size:.62rem;font-weight:600;
    color:#64ffda;letter-spacing:4px;text-transform:uppercase;
    border-left:2px solid #64ffda;padding-left:12px;margin:28px 0 14px}

/* ── answer / sar boxes ── */
.abox{background:#111;border:1px solid #1e1e1e;border-radius:8px;
      padding:22px;font-size:.88rem;line-height:1.85;color:#ededed;
      white-space:pre-wrap;border-left:2px solid #64ffda}
.sarbox{background:#0c180c;border:1px solid #1e3a1e;border-radius:8px;
        padding:22px;font-family:'JetBrains Mono',monospace;font-size:.78rem;
        line-height:1.95;color:#b9e4b9;white-space:pre-wrap}

/* ── chunk cards ── */
.chunk{background:#111;border:1px solid #1e1e1e;border-left:2px solid #64ffda;
       border-radius:4px;padding:14px 16px;margin-bottom:8px;
       transition:border-color .2s}
.chunk:hover{border-color:#64ffda;background:#141414}
.chunk-hdr{font-family:'JetBrains Mono',monospace;font-size:.62rem;font-weight:600;
           color:#64ffda;margin-bottom:6px;letter-spacing:1px}
.chunk-txt{font-size:.8rem;color:#888;line-height:1.65}

/* ── pipeline card ── */
.pipe{background:#111;border:1px solid #1e1e1e;border-radius:8px;padding:22px;
      transition:border-color .2s}
.pipe:hover{border-color:#64ffda}
.pipe-title{font-family:'JetBrains Mono',monospace;font-size:.62rem;font-weight:600;
            color:#64ffda;letter-spacing:2px;text-transform:uppercase;margin-bottom:14px}
.pipe-step{font-family:'JetBrains Mono',monospace;font-size:.75rem;color:#888;
           padding:5px 0;border-bottom:1px solid #1a1a1a}
.pipe-step:last-child{border-bottom:none}
.pipe-arrow{color:#64ffda;margin-right:8px}

/* ── risk badge ── */
.rbadge{background:rgba(100,255,218,.06);color:#64ffda;
        border:1px solid rgba(100,255,218,.15);border-radius:20px;
        font-family:'JetBrains Mono',monospace;font-size:.6rem;font-weight:600;
        letter-spacing:1px;padding:4px 12px;display:inline-block;margin:3px;
        transition:all .2s;cursor:default}
.rbadge:hover{background:rgba(100,255,218,.15);border-color:#64ffda}

/* ── streamlit widget overrides ── */
.stTextArea textarea,.stTextInput input{
  background:#111 !important;color:#ededed !important;
  border:1px solid #2a2a2a !important;border-radius:6px !important;
  font-family:'JetBrains Mono',monospace !important;font-size:.82rem !important}
.stTextArea textarea:focus,.stTextInput input:focus{
  border-color:#64ffda !important;outline:none !important}
.stSelectbox [data-baseweb="select"]>div,.stMultiSelect [data-baseweb="select"]>div{
  background:#111 !important;border-color:#2a2a2a !important;color:#ededed !important;
  font-family:'JetBrains Mono',monospace !important}
.stNumberInput input{background:#111 !important;color:#ededed !important;
                     border-color:#2a2a2a !important;font-family:'JetBrains Mono',monospace !important}
.stButton button{background:transparent !important;
                 border:1px solid #64ffda !important;color:#64ffda !important;
                 font-family:'JetBrains Mono',monospace !important;
                 font-weight:600 !important;letter-spacing:2px !important;
                 font-size:.72rem !important;border-radius:4px !important;
                 padding:10px 28px !important;transition:all .2s !important}
.stButton button:hover{background:rgba(100,255,218,.1) !important;
                       box-shadow:0 0 20px rgba(100,255,218,.15) !important}
.stDownloadButton button{border-color:#2a2a2a !important;color:#666 !important;
                         font-size:.65rem !important}
.stDownloadButton button:hover{border-color:#64ffda !important;color:#64ffda !important}
.stTabs [data-baseweb="tab-list"]{background:transparent !important;
                                   border-bottom:1px solid #1e1e1e !important;gap:4px}
.stTabs [data-baseweb="tab"]{color:#555 !important;
  font-family:'JetBrains Mono',monospace !important;font-size:.68rem !important;
  font-weight:600 !important;letter-spacing:2px !important;
  padding:10px 16px !important;border-radius:0 !important;
  transition:color .2s !important}
.stTabs [aria-selected="true"]{color:#64ffda !important;
  border-bottom:2px solid #64ffda !important}
.stTabs [data-baseweb="tab"]:hover{color:#ededed !important}
label,.stSelectbox label,.stMultiSelect label,.stTextInput label,
.stTextArea label,.stNumberInput label{
  color:#ededed !important;font-weight:600 !important;font-size:.82rem !important}
</style>

<div class="navbar">
  <div class="nav-logo">ML2</div>
  <div class="nav-links">
    <a>Q&A</a>
    <a>SAR</a>
    <a>Architecture</a>
    <a href="https://github.com/fahadamjad009" target="_blank">github ↗</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ── agent ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_agent():
    return RegulatoryAgent()
agent = load_agent()

# ── hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">Regulatory Intelligence Agent</div>
  <div class="hero-sub">LangGraph · llama3.2:3b · FAISS + BM25 · APRA · AUSTRAC · Local LLM · No API Key</div>
  <div class="tags">
    <span class="tag">LangGraph</span>
    <span class="tag">llama3.2:3b</span>
    <span class="tag">Ollama</span>
    <span class="tag">FAISS</span>
    <span class="tag">BM25</span>
    <span class="tag">RRF Fusion</span>
    <span class="tag">FastAPI</span>
    <span class="tag">APRA APS 220</span>
    <span class="tag">AUSTRAC AML/CTF</span>
    <span class="tag">Local LLM</span>
    <span class="tag">No API Key</span>
  </div>
  <div class="stats">
    <div class="stat"><div class="stat-label">LLM Model</div><div class="stat-value">3b</div><div class="stat-sub">llama3.2 · Ollama · local</div></div>
    <div class="stat"><div class="stat-label">Retrieval</div><div class="stat-value">Hybrid</div><div class="stat-sub">FAISS + BM25 + RRF</div></div>
    <div class="stat"><div class="stat-label">Corpus</div><div class="stat-value">3</div><div class="stat-sub">APRA · AUSTRAC · 10-K</div></div>
    <div class="stat"><div class="stat-label">Agent Nodes</div><div class="stat-value">2</div><div class="stat-sub">retrieve → generate</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3 = st.tabs(["REGULATORY Q&A", "SAR GENERATOR", "ARCHITECTURE"])

# ══════════════════════════════════════════════════════
# TAB 1 — Q&A
# ══════════════════════════════════════════════════════
with t1:
    st.markdown('<div class="sh">Ask a Regulatory Question</div>', unsafe_allow_html=True)

    sample_qs = [
        "",
        "What are APRA's key requirements for a credit risk management framework?",
        "How should suspicious transactions be reported to AUSTRAC?",
        "What transaction patterns indicate potential money laundering?",
        "What capital adequacy ratios must ADIs maintain under APRA?",
        "What must a SAR narrative include according to AUSTRAC guidelines?",
        "How does an ADI classify problem credits under APRA APS 220?",
    ]

    selected = st.selectbox("Sample questions", sample_qs,
                            format_func=lambda x: "— Select a sample —" if x == "" else x,
                            label_visibility="collapsed")
    question = st.text_area("Question", value=selected,
                            placeholder="Ask about APRA, AUSTRAC, AML/CTF, capital requirements, credit risk...",
                            height=80, label_visibility="collapsed")

    if st.button("▶ ASK AGENT", key="qa_btn"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            with st.spinner("Retrieving · generating · citing..."):
                result = agent.query(question)

            st.markdown('<div class="sh">Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="abox">{result["answer"]}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            for col, lbl, val, sub in [
                (m1, "Chunks Used", str(len(result["chunks"])), "retrieved + RRF fused"),
                (m2, "Sources",     str(len(result["sources"])), "document files"),
                (m3, "Model",       "llama3.2",                  "3b · local · no API"),
            ]:
                with col:
                    st.markdown(f'<div class="stat"><div class="stat-label">{lbl}</div><div class="stat-value">{val}</div><div class="stat-sub">{sub}</div></div>', unsafe_allow_html=True)

            # RRF bar chart
            st.markdown('<div class="sh">Retrieval Scores (RRF Fusion)</div>', unsafe_allow_html=True)
            chunks = result["chunks"]
            fig = go.Figure(go.Bar(
                x=[c["rrf_score"] for c in chunks],
                y=[f"chunk {c['chunk_index']}" for c in chunks],
                orientation="h",
                marker_color="#64ffda", marker_opacity=0.75,
                text=[f"{c['rrf_score']:.4f}" for c in chunks],
                textposition="outside",
                textfont=dict(color="#555", size=10, family="JetBrains Mono"),
            ))
            fig.update_layout(
                height=160, margin=dict(l=70, r=70, t=8, b=8),
                paper_bgcolor="#0a0a0a", plot_bgcolor="#111111",
                font=dict(family="JetBrains Mono", color="#555", size=10),
                xaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#1a1a1a",
                           tickfont=dict(color="#555"), title_font=dict(color="#555")),
                yaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="#888")),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="sh">Retrieved Context</div>', unsafe_allow_html=True)
            for c in chunks:
                st.markdown(f"""<div class="chunk">
                  <div class="chunk-hdr">Rank {c['rank']} · RRF={c['rrf_score']:.4f} · {c['source']} · chunk {c['chunk_index']}</div>
                  <div class="chunk-txt">{c['text'][:300]}{'...' if len(c['text'])>300 else ''}</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 2 — SAR Generator
# ══════════════════════════════════════════════════════
with t2:
    st.markdown('<div class="sh">SAR Narrative Generator</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:.82rem;color:#666;margin-bottom:20px">Generate AUSTRAC-compliant Suspicious Activity Report narratives from transaction data.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        customer = st.text_input("Customer / Entity", value="XYZ Trading Pty Ltd")
        account  = st.text_input("Account ID", value="ACC-2024-8871")
        period   = st.text_input("Reporting period", value="November 2024")
        amount   = st.number_input("Amount (AUD)", value=95000, step=1000)
    with c2:
        tx_type = st.selectbox("Transaction type", [
            "cash deposits","wire transfers","cryptocurrency",
            "trade finance","foreign exchange","structured deposits","other"])
        all_flags = [
            "structuring","cash-intensive transactions","inconsistency with business profile",
            "rapid movement of funds","dormant account activity","layering",
            "geographic risk","threshold avoidance","unusual cash withdrawals",
        ]
        flags = st.multiselect("AML indicators triggered", all_flags,
                               default=["structuring","cash-intensive transactions"])
        notes = st.text_area("Additional context",
                             value="Three separate cash deposits made on consecutive days just below $32K each. Customer registered as a software consultancy.",
                             height=104)

    if st.button("▶ GENERATE SAR", key="sar_btn"):
        if not flags:
            st.warning("Select at least one AML indicator.")
        else:
            tx = dict(customer=customer, account=account, period=period,
                      amount=float(amount), currency="AUD",
                      transaction_type=tx_type, flags=flags, notes=notes)
            with st.spinner("Retrieving regulatory context · generating narrative..."):
                result = agent.generate_sar(tx)

            st.markdown('<div class="sh">SAR Narrative</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sarbox">{result["sar_narrative"]}</div>', unsafe_allow_html=True)

            st.markdown('<div class="sh">Risk Indicators</div>', unsafe_allow_html=True)
            badges = " ".join(f'<span class="rbadge">{f.upper()}</span>' for f in result["risk_indicators"])
            st.markdown(f'<div style="margin:10px 0 20px">{badges}</div>', unsafe_allow_html=True)

            st.download_button("⬇ DOWNLOAD SAR .txt",
                               data=result["sar_narrative"],
                               file_name=f"SAR_{customer.replace(' ','_')}_{period.replace(' ','_')}.txt",
                               mime="text/plain")

# ══════════════════════════════════════════════════════
# TAB 3 — Architecture
# ══════════════════════════════════════════════════════
with t3:
    st.markdown('<div class="sh">Agent Pipelines</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""<div class="pipe">
          <div class="pipe-title">Q&A Pipeline</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>User query</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>FAISS dense search (MiniLM-L6-v2)</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>BM25 lexical search</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>Reciprocal Rank Fusion (k=60)</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>Top-4 chunks selected</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>llama3.2:3b via Ollama</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>Grounded answer + citations</div>
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""<div class="pipe">
          <div class="pipe-title">SAR Pipeline</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>Transaction flags + data</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>Hybrid retrieval (AUSTRAC context)</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>Regulatory context injection</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>llama3.2:3b via Ollama</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>Structured SAR narrative</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>Risk indicators extracted</div>
          <div class="pipe-step"><span class="pipe-arrow">▹</span>AUSTRAC-compliant output</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh">Stack</div>', unsafe_allow_html=True)
    df = pd.DataFrame({
        "Layer":      ["Agent","LLM","Retrieval","Embeddings","Lexical","API","UI"],
        "Technology": ["LangGraph","llama3.2:3b","FAISS IndexFlatIP","MiniLM-L6-v2","BM25Okapi","FastAPI","Streamlit"],
        "Detail":     ["StateGraph nodes","Ollama local","cosine similarity","sentence-transformers","rank-bm25","3 endpoints","dark theme"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""<div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;
color:#2a2a2a;text-align:center;border-top:1px solid #1a1a1a;
padding:16px 0 4px;margin-top:40px">
ML2 · LLM Regulatory Financial Intelligence Agent · LangGraph + llama3.2:3b + FAISS + BM25 ·
Fahad Amjad · github.com/fahadamjad009
</div>""", unsafe_allow_html=True)
