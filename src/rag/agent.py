"""
MarketMind — Stage 2G: AI Agent (Merged Option A + B)
======================================================
Merged dual-mode agent:
  Mode 1 → Adversarial Forensic Audit (auto on call load)
           Cross-examines transcript vs 10-Q discrepancies,
           peer dissonance, and accountability gaps.
  Mode 2 → Interactive RAG Chat
           Evidence-grounded Q&A with source citations.

LLM: Gemini 2.0 Flash (free tier via Google AI Studio)
     Requires: GEMINI_API_KEY in .env

Vector DB: ChromaDB (local, no cost)
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local)

Usage:
    from src.rag.agent import MarketMindAgent
    agent = MarketMindAgent()
    # Chat
    reply = agent.chat(call_id=1, question="What did the CFO say about margins?")
    # Audit
    report = agent.run_forensic_audit(call_id=1)
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CANDIDATE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
]

# Rate limit guard (free tier: 15 RPM)
import time
_last_call_ts = 0.0
_MIN_INTERVAL  = 4.1  # seconds between calls → ~14.6 RPM


def _rate_limit():
    global _last_call_ts
    elapsed = time.time() - _last_call_ts
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call_ts = time.time()


# ── Gemini client ─────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, temperature: float = 0.3) -> str:
    """Call Gemini API using modern google-genai with automatic model fallback."""
    if not GEMINI_API_KEY:
        return (
            "⚠️ GEMINI_API_KEY not set in .env file.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey\n"
            "Add it to your .env as: GEMINI_API_KEY=your_key_here"
        )

    try:
        from google import genai
    except ImportError:
        logger.error("google-genai not installed. Run: pip install google-genai")
        return "Error: google-genai package not installed."

    client = genai.Client(api_key=GEMINI_API_KEY)

    last_error = None
    for model_name in CANDIDATE_MODELS:
        _rate_limit()
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini {model_name} failed: {e}. Trying fallback...")
            last_error = e

    logger.error(f"All Gemini candidate models failed: {last_error}")
    return f"[Gemini API error: {last_error}]"


# ── Context builders ──────────────────────────────────────────────────────────

def _get_call_context(call_id: int) -> dict:
    """Pull call metadata + segments + signals from DB."""
    from src.db.connection import get_session
    from src.db.models import (
        EarningsCall, Company, TranscriptSegment,
        SentimentScore, EvasionScore, Signal, ExtractedMetric
    )

    session = get_session()
    try:
        ec = session.query(EarningsCall).filter_by(call_id=call_id).first()
        if not ec:
            return {}

        co = session.query(Company).filter_by(company_id=ec.company_id).first()

        # Segments (top 15 by order)
        segs = (session.query(TranscriptSegment)
                .filter_by(call_id=call_id)
                .order_by(TranscriptSegment.segment_order)
                .limit(15).all())

        # Sentiment
        sents = session.query(SentimentScore).filter_by(call_id=call_id).all()
        role_sent = {}
        for s in sents:
            role = s.speaker_role.value if s.speaker_role else "unknown"
            role_sent.setdefault(role, []).append(s.compound_score)
        avg_sent = {r: round(sum(v)/len(v), 3) for r, v in role_sent.items()}

        # Evasion
        evas = session.query(EvasionScore).filter_by(call_id=call_id).all()
        avg_evasion = round(sum(e.evasion_score for e in evas) / len(evas), 3) if evas else 0.0
        evasive_cnt = sum(1 for e in evas if e.is_evasive)

        # Metrics
        metrics = session.query(ExtractedMetric).filter_by(call_id=call_id).limit(10).all()
        metrics_summary = [
            {"name": m.metric_name, "value": m.value, "unit": m.unit, "is_guidance": m.is_guidance}
            for m in metrics
        ]

        # Signal
        sig = session.query(Signal).filter_by(call_id=call_id).first()

        # Transcript text (first 3000 chars)
        transcript_preview = "\n\n".join(
            f"[{s.speaker_label or s.speaker_role.value}]\n{s.text[:500]}"
            for s in segs if s.text
        )[:3000]

        return {
            "ticker"            : co.ticker if co else "UNKNOWN",
            "company_name"      : co.company_name if co else "Unknown",
            "quarter"           : ec.quarter,
            "fiscal_year"       : ec.fiscal_year,
            "call_date"         : str(ec.call_date),
            "avg_sentiment"     : avg_sent,
            "avg_evasion"       : avg_evasion,
            "evasive_count"     : evasive_cnt,
            "metrics"           : metrics_summary,
            "predicted_signal"  : sig.predicted_direction.value if sig else "UNKNOWN",
            "signal_confidence" : sig.confidence if sig else 0.0,
            "transcript_preview": transcript_preview,
            "call_id"           : call_id,
        }
    finally:
        session.close()


def _get_peer_context(ticker: str, quarter: str, year: int) -> str:
    """Get same-quarter sentiment summary for peer companies."""
    from src.db.connection import get_session
    from src.db.models import EarningsCall, Company, SentimentScore

    peers = ["NVDA", "AMD", "INTC", "QCOM", "TSM", "ASML", "MRVL"]
    peers = [p for p in peers if p != ticker]

    session = get_session()
    try:
        lines = []
        for peer in peers:
            co = session.query(Company).filter_by(ticker=peer).first()
            if not co:
                continue
            ec = (session.query(EarningsCall)
                  .filter_by(company_id=co.company_id, quarter=quarter, fiscal_year=year)
                  .first())
            if not ec:
                continue
            sents = session.query(SentimentScore).filter_by(call_id=ec.call_id).all()
            if sents:
                avg = round(sum(s.compound_score for s in sents) / len(sents), 3)
                lines.append(f"  {peer}: compound_sentiment={avg}")

        return "\n".join(lines) if lines else "  No peer data for this quarter."
    finally:
        session.close()


# ── RAG context retrieval ─────────────────────────────────────────────────────

def _retrieve_context(query: str, ticker: str, call_id: int, n: int = 5) -> str:
    """Pull top-n relevant chunks from ChromaDB."""
    try:
        from src.rag.vector_store import query_similar
        results = query_similar(query, n=n, ticker=ticker)
        if not results:
            return "No relevant context found in vector store."

        lines = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            src  = f"[{meta.get('ticker','')} {meta.get('doc_type','')} {meta.get('quarter','')} {meta.get('year','')}]"
            lines.append(f"Source {i} {src}:\n{r['text'][:600]}")

        return "\n\n---\n\n".join(lines)
    except Exception as e:
        logger.warning(f"Vector store retrieval failed: {e}")
        return "Vector store unavailable — using transcript preview only."


# ── Prompt builders ───────────────────────────────────────────────────────────

SYSTEM_PERSONA = """You are MarketMind's AI Financial Intelligence Agent.
You are an adversarial forensic analyst specializing in earnings call analysis.
Your role combines:
  1. Adversarial Auditor: Cross-examine management statements against SEC 10-Q/10-K filings.
  2. RAG Research Co-Pilot: Answer questions grounded in evidence.

Rules:
- Always cite your sources (transcript section, 10-Q page/section, prior call date).
- Flag discrepancies between verbal statements and filed numbers.
- Never hallucinate numbers — if data is unavailable, say so.
- End every response with a "Confidence Level: XX%" based on evidence quality.
- Format lists and tables in markdown.
- Always include the disclaimer: "Educational use only. Not financial advice."
"""

def _build_chat_prompt(ctx: dict, question: str, rag_context: str) -> str:
    return f"""{SYSTEM_PERSONA}

## Earnings Call Context
Company: {ctx['company_name']} ({ctx['ticker']})
Quarter: {ctx['quarter']} FY{ctx['fiscal_year']} | Call Date: {ctx['call_date']}
Predicted Signal: {ctx['predicted_signal']} (confidence {ctx.get('signal_confidence',0)*100:.0f}%)
Avg Sentiment: CEO={ctx['avg_sentiment'].get('CEO', 'N/A')}, CFO={ctx['avg_sentiment'].get('CFO', 'N/A')}, Analyst={ctx['avg_sentiment'].get('ANALYST', 'N/A')}
Evasion Score: {ctx['avg_evasion']} | Evasive responses: {ctx['evasive_count']}

## Extracted Financial Metrics
{json.dumps(ctx['metrics'], indent=2)}

## Relevant Retrieved Context (RAG)
{rag_context}

## Transcript Preview
{ctx['transcript_preview']}

## User Question
{question}

## Instructions
Answer the question using the context above. Be specific, cite evidence, flag discrepancies.
"""


def _build_audit_prompt(ctx: dict, peer_context: str, rag_context: str) -> str:
    return f"""{SYSTEM_PERSONA}

## FORENSIC AUDIT TASK
Perform an adversarial forensic audit of this earnings call.

Company: {ctx['company_name']} ({ctx['ticker']})
Quarter: {ctx['quarter']} FY{ctx['fiscal_year']} | Call Date: {ctx['call_date']}

## Call Statistics
Sentiment: {json.dumps(ctx['avg_sentiment'])}
Evasion Score: {ctx['avg_evasion']} (evasive responses: {ctx['evasive_count']})
ML Signal: {ctx['predicted_signal']} ({ctx.get('signal_confidence',0)*100:.0f}% confidence)

## Extracted Metrics
{json.dumps(ctx['metrics'], indent=2)}

## Same-Quarter Peer Sentiment
{peer_context}

## Retrieved Context (Transcript + 10-Q/10-K)
{rag_context}

## Transcript Preview
{ctx['transcript_preview']}

## Audit Instructions
Produce a structured JSON audit report with the following fields:
{{
  "overall_risk": "LOW|MEDIUM|HIGH|CRITICAL",
  "findings": [
    {{
      "severity": "CRITICAL|WARNING|INFO",
      "type": "CALL_VS_10Q|PEER_DISSONANCE|ACCOUNTABILITY_GAP|EVASION_PATTERN|METRIC_AMBIGUITY",
      "title": "...",
      "description": "...",
      "evidence": "...",
      "citation": "..."
    }}
  ],
  "accountability_assessment": "...",
  "recommendation": "..."
}}

Output ONLY the JSON — no prose before or after.
"""


# ── Main Agent class ──────────────────────────────────────────────────────────

class MarketMindAgent:
    """
    Dual-mode AI agent for earnings call forensic intelligence.

    Args:
        use_rag: If True, enables ChromaDB retrieval (requires vector store built).
    """

    def __init__(self, use_rag: bool = True):
        self.use_rag = use_rag

    def chat(self, call_id: int, question: str) -> dict:
        """
        Answer a user question about an earnings call.

        Returns:
            {
              "answer": str,
              "sources": list[dict],
              "call_id": int,
            }
        """
        ctx = _get_call_context(call_id)
        if not ctx:
            return {"answer": f"No data found for call_id={call_id}.", "sources": [], "call_id": call_id}

        rag_context = ""
        sources     = []
        if self.use_rag:
            try:
                from src.rag.vector_store import query_similar
                results = query_similar(question, n=5, ticker=ctx["ticker"])
                rag_context = _retrieve_context(question, ctx["ticker"], call_id)
                sources = [{"text": r["text"][:200], "metadata": r["metadata"]} for r in results]
            except Exception as e:
                logger.warning(f"RAG unavailable: {e}")
                rag_context = ctx["transcript_preview"]

        prompt = _build_chat_prompt(ctx, question, rag_context)
        answer = _call_gemini(prompt, temperature=0.3)

        return {
            "answer"  : answer,
            "sources" : sources,
            "call_id" : call_id,
            "ticker"  : ctx["ticker"],
            "quarter" : f"{ctx['quarter']} FY{ctx['fiscal_year']}",
        }

    def run_forensic_audit(self, call_id: int) -> dict:
        """
        Run the autonomous adversarial audit for a call.

        Returns structured audit findings dict.
        """
        ctx = _get_call_context(call_id)
        if not ctx:
            return {"error": f"No data for call_id={call_id}"}

        peer_context = _get_peer_context(ctx["ticker"], ctx["quarter"], ctx["fiscal_year"])

        audit_query  = f"{ctx['ticker']} {ctx['quarter']} FY{ctx['fiscal_year']} revenue margin guidance discrepancy"
        rag_context  = _retrieve_context(audit_query, ctx["ticker"], call_id, n=6) if self.use_rag else ""

        prompt  = _build_audit_prompt(ctx, peer_context, rag_context)
        raw_out = _call_gemini(prompt, temperature=0.1)

        # Parse JSON from response
        audit_result = {}
        try:
            # Extract JSON block if wrapped in markdown code fences
            match = __import__("re").search(r'\{[\s\S]+\}', raw_out)
            if match:
                audit_result = json.loads(match.group())
            else:
                audit_result = json.loads(raw_out)
        except Exception:
            logger.warning("Could not parse audit JSON — returning raw text.")
            audit_result = {
                "overall_risk": "UNKNOWN",
                "findings": [],
                "raw_response": raw_out,
            }

        audit_result["call_id"]   = call_id
        audit_result["ticker"]    = ctx["ticker"]
        audit_result["quarter"]   = f"{ctx['quarter']} FY{ctx['fiscal_year']}"
        audit_result["generated_at"] = datetime.utcnow().isoformat()

        # Persist to DB
        self._save_audit_to_db(call_id, audit_result)

        return audit_result

    def _save_audit_to_db(self, call_id: int, audit: dict):
        """Persist audit findings to accountability_findings table."""
        try:
            from src.db.connection import get_session
            from src.db.models import AccountabilityFinding, FindingType, Severity

            session = get_session()
            try:
                for f in audit.get("findings", []):
                    sev_str  = f.get("severity", "INFO")
                    type_str = f.get("type", "CALL_VS_10Q")

                    sev_map  = {"CRITICAL": Severity.HIGH, "WARNING": Severity.MEDIUM, "INFO": Severity.LOW}
                    type_map = {
                        "CALL_VS_10Q"       : FindingType.DISCREPANCY,
                        "PEER_DISSONANCE"   : FindingType.INCONSISTENCY,
                        "ACCOUNTABILITY_GAP": FindingType.EVASION,
                        "EVASION_PATTERN"   : FindingType.EVASION,
                        "METRIC_AMBIGUITY"  : FindingType.DISCREPANCY,
                    }

                    finding = AccountabilityFinding(
                        call_id      = call_id,
                        finding_type = type_map.get(type_str, FindingType.DISCREPANCY),
                        severity     = sev_map.get(sev_str, Severity.LOW),
                        description  = f.get("description", "")[:1000],
                        evidence     = f.get("evidence", "")[:1000],
                        created_at   = datetime.utcnow(),
                    )
                    session.add(finding)
                session.commit()
                logger.info(f"Saved {len(audit.get('findings', []))} audit findings to DB.")
            finally:
                session.close()
        except Exception as e:
            logger.warning(f"Could not save audit to DB: {e}")


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MarketMind AI Agent")
    parser.add_argument("--call-id", type=int, required=True, help="Call ID from DB")
    parser.add_argument("--mode",    choices=["chat", "audit"], default="audit")
    parser.add_argument("--question", type=str, default="What are the key risks for this company?")
    parser.add_argument("--no-rag",  action="store_true", help="Disable RAG (faster)")
    args = parser.parse_args()

    agent = MarketMindAgent(use_rag=not args.no_rag)

    if args.mode == "chat":
        result = agent.chat(args.call_id, args.question)
        print("\n" + "="*60)
        print(result["answer"])
        print("="*60)
    else:
        result = agent.run_forensic_audit(args.call_id)
        print(json.dumps(result, indent=2))
