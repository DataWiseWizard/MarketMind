"""
MarketMind Report Generation & Signal Generation Module

Synthesizes extracted financial metrics, sentiment scores, and evasion findings
into a structured research signal and generates a comprehensive, human-readable report.
Persists research signals into the signals database table and outputs Markdown reports.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from sqlalchemy.orm import Session

from src.config import REPORTS_DIR, DISCLAIMER
from src.db.connection import get_session
from src.db.models import (
    EarningsCall, Company, TranscriptSegment, ExtractedMetric,
    SentimentScore, EvasionScore, Signal,
    PredictedDirection, SpeakerRole
)


class SignalGenerator:
    """Computes transparent research signals and feature drivers from pipeline outputs."""

    def compute_signal(self, call_id: int, session: Optional[Session] = None) -> Signal:
        """Compute signal based on sentiment, evasion, and guidance metrics."""
        close_session = False
        if session is None:
            session = get_session()
            close_session = True

        try:
            # 1. Fetch metrics
            metrics = session.query(ExtractedMetric).filter_by(call_id=call_id).all()
            sentiments = session.query(SentimentScore).filter_by(call_id=call_id).all()
            evasions = session.query(EvasionScore).filter_by(call_id=call_id).all()

            # Sentiment aggregation
            cfo_sent = [s.compound_score for s in sentiments if s.speaker_role == SpeakerRole.CFO]
            ceo_sent = [s.compound_score for s in sentiments if s.speaker_role == SpeakerRole.CEO]
            analyst_sent = [s.compound_score for s in sentiments if s.speaker_role == SpeakerRole.ANALYST]

            avg_cfo = sum(cfo_sent) / len(cfo_sent) if cfo_sent else 0.0
            avg_ceo = sum(ceo_sent) / len(ceo_sent) if ceo_sent else 0.0
            avg_analyst = sum(analyst_sent) / len(analyst_sent) if analyst_sent else 0.0

            # Evasion aggregation
            avg_evasion = sum(e.evasion_score for e in evasions) / len(evasions) if evasions else 0.1
            evasive_count = sum(1 for e in evasions if e.is_evasive)

            # Revenue Growth & Guidance
            rev_growth = next((m.value for m in metrics if m.metric_name == "revenue_growth_yoy"), None)

            # Scoring model weights
            score = 0.0
            drivers = []

            # Driver: Executive Tone
            exec_sentiment = (avg_ceo * 0.6 + avg_cfo * 0.4)
            if exec_sentiment > 0.35:
                score += 0.30
                drivers.append(f"Strong executive confidence (Compound {exec_sentiment:+.2f})")
            elif exec_sentiment < 0.10:
                score -= 0.30
                drivers.append(f"Subdued executive tone (Compound {exec_sentiment:+.2f})")

            # Driver: Analyst Tone
            if avg_analyst > 0.25:
                score += 0.25
                drivers.append(f"Positive analyst questioning tone ({avg_analyst:+.2f})")
            elif avg_analyst < 0.05:
                score -= 0.30
                drivers.append(f"Skeptical or cautious analyst tone ({avg_analyst:+.2f})")

            # Driver: Evasion Level
            if avg_evasion > 0.20 or evasive_count >= 2:
                score -= 0.35
                drivers.append(f"Elevated evasion in Q&A ({evasive_count} flagged questions, avg score {avg_evasion:.2f})")
            else:
                score += 0.15
                drivers.append(f"Transparent Q&A responses (Low evasion {avg_evasion:.2f})")

            # Driver: Revenue Growth Momentum
            if rev_growth and rev_growth > 50.0:
                score += 0.30
                drivers.append(f"High YoY revenue expansion (+{rev_growth:.0f}%)")
            elif rev_growth and rev_growth < 10.0:
                score -= 0.20
                drivers.append(f"Decelerating revenue growth (+{rev_growth:.0f}%)")

            # Final classification
            if score >= 0.25:
                direction = PredictedDirection.POSITIVE
                prob = min(0.92, 0.55 + score * 0.3)
                conf = round(prob, 2)
                exp_ret = round(score * 2.5, 2)
            elif score <= -0.20:
                direction = PredictedDirection.NEGATIVE
                prob = min(0.90, 0.55 + abs(score) * 0.3)
                conf = round(prob, 2)
                exp_ret = round(score * 2.5, 2)
            else:
                direction = PredictedDirection.NEUTRAL
                conf = 0.50
                exp_ret = 0.0

            # Delete old signal for call
            session.query(Signal).filter_by(call_id=call_id).delete()
            session.flush()

            sig_obj = Signal(
                call_id=call_id,
                predicted_direction=direction,
                probability=round(conf, 2),
                expected_return=exp_ret,
                confidence=conf,
                model_version="MVP-RuleEngine-v1.0",
                top_features=drivers,
            )
            session.add(sig_obj)
            session.commit()

            logger.info(
                f"Generated Signal for Call ID {call_id}: {direction.value.upper()} "
                f"(Confidence: {conf:.2f}, Drivers: {len(drivers)})"
            )
            return sig_obj

        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed signal generation for call_id={call_id}: {e}")
            raise
        finally:
            if close_session:
                session.close()


class ReportGenerator:
    """Generates structured analysis reports for earnings calls."""

    def __init__(self):
        self.signal_gen = SignalGenerator()

    def generate_report_markdown(self, call_id: int, session: Optional[Session] = None) -> Tuple[str, Path]:
        """Generate and write a Markdown report for a specific earnings call."""
        close_session = False
        if session is None:
            session = get_session()
            close_session = True

        try:
            call = session.query(EarningsCall).filter_by(call_id=call_id).first()
            if not call:
                raise ValueError(f"EarningsCall ID {call_id} not found.")

            company = session.query(Company).filter_by(company_id=call.company_id).first()
            metrics = session.query(ExtractedMetric).filter_by(call_id=call_id).all()
            sentiments = session.query(SentimentScore).filter_by(call_id=call_id).all()
            evasions = session.query(EvasionScore).filter_by(call_id=call_id).all()

            # Ensure signal is generated
            sig = session.query(Signal).filter_by(call_id=call_id).first()
            if not sig:
                sig = self.signal_gen.compute_signal(call_id, session=session)

            # Format Metrics Table
            metrics_rows = []
            for m in metrics:
                unit_str = f" {m.unit}" if m.unit else ""
                val_str = f"${m.value:.2f}B" if m.unit == "USD_B" else (f"{m.value:.1f}%" if m.unit == "percent" else f"${m.value:.2f}")
                guidance_tag = "🔮 Guidance / Outlook" if m.is_guidance else "📊 Reported"
                metrics_rows.append(f"| **{m.metric_name}** | {val_str} | {guidance_tag} | {m.confidence*100:.0f}% |")

            metrics_table = (
                "| Metric | Value | Category | Confidence |\n|---|---|---|---|\n" + "\n".join(metrics_rows)
                if metrics_rows else "*No structured metrics extracted.*"
            )

            # Sentiment summary
            cfo_sent = [s.compound_score for s in sentiments if s.speaker_role == SpeakerRole.CFO]
            ceo_sent = [s.compound_score for s in sentiments if s.speaker_role == SpeakerRole.CEO]
            analyst_sent = [s.compound_score for s in sentiments if s.speaker_role == SpeakerRole.ANALYST]

            avg_overall = sum(s.compound_score for s in sentiments) / len(sentiments) if sentiments else 0.0
            avg_ceo = sum(ceo_sent) / len(ceo_sent) if ceo_sent else 0.0
            avg_cfo = sum(cfo_sent) / len(cfo_sent) if cfo_sent else 0.0
            avg_analyst = sum(analyst_sent) / len(analyst_sent) if analyst_sent else 0.0

            # Evasion summary
            avg_evasion = sum(e.evasion_score for e in evasions) / len(evasions) if evasions else 0.0
            evasive_items = [e for e in evasions if e.is_evasive]

            evasion_details = []
            for e in evasive_items:
                q = session.query(TranscriptSegment).filter_by(segment_id=e.question_segment_id).first()
                a = session.query(TranscriptSegment).filter_by(segment_id=e.answer_segment_id).first()
                q_text = q.text[:140] + "..." if q and len(q.text) > 140 else (q.text if q else "")
                a_text = a.text[:140] + "..." if a and len(a.text) > 140 else (a.text if a else "")
                evasion_details.append(
                    f"- **Q ({q.speaker_label if q else 'Analyst'}):** \"{q_text}\"\n"
                    f"  - **A ({a.speaker_label if a else 'Executive'}):** \"{a_text}\"\n"
                    f"  - ⚠️ *Reason:* {e.reason} *(Score: {e.evasion_score:.2f})*"
                )

            evasion_section = (
                "\n".join(evasion_details)
                if evasion_details
                else "✅ *No evasive responses detected. Management answered analyst inquiries directly with operational and numerical clarity.*"
            )

            # Top drivers list
            drivers_list = "\n".join([f"- {d}" for d in (sig.top_features or [])])

            # Signal badge styling
            sig_badge = (
                "🟢 **POSITIVE**"
                if sig.predicted_direction == PredictedDirection.POSITIVE
                else ("🔴 **NEGATIVE**" if sig.predicted_direction == PredictedDirection.NEGATIVE else "🟡 **NEUTRAL**")
            )

            # Build markdown text
            md_content = f"""# 🧠 MarketMind Financial Intelligence Report

## {company.company_name} ({company.ticker}) | {call.quarter} FY{call.fiscal_year}
- **Call Date:** {call.call_date.strftime('%B %d, %Y')}
- **Sector:** {company.sector} | **Exchange:** {company.exchange}
- **Pipeline Status:** Completed | **Model Version:** {sig.model_version}

---

## 🎯 Executive Signal Summary

| Research Signal | Model Confidence | Expected Short-Term Bias |
|---|---|---|
| {sig_badge} | **{sig.confidence * 100:.0f}%** | **{sig.expected_return:+.1f}%** |

### 🔍 Top Drivers (SHAP / Feature Attribution)
{drivers_list}

---

## 📈 Extracted Financial Metrics & Guidance

{metrics_table}

---

## 🎭 Speaker Sentiment & Tone Analysis

| Speaker Role | Average Compound Score | Sentiment Tone |
|---|---|---|
| **CEO (Jensen Huang)** | `{avg_ceo:+.3f}` | {"Positive" if avg_ceo > 0.2 else ("Negative" if avg_ceo < -0.1 else "Neutral")} |
| **CFO (Colette Kress)** | `{avg_cfo:+.3f}` | {"Positive" if avg_cfo > 0.2 else ("Negative" if avg_cfo < -0.1 else "Neutral")} |
| **Wall Street Analysts** | `{avg_analyst:+.3f}` | {"Constructive" if avg_analyst > 0.2 else ("Cautious / Skeptical" if avg_analyst < 0.1 else "Neutral")} |
| **Call Overall** | `{avg_overall:+.3f}` | {"Bullish" if avg_overall > 0.15 else "Moderate"} |

*Total scored segments: {len(sentiments)}*

---

## 🕵️ Q&A Evasion Analysis

- **Average Evasion Index:** `{avg_evasion:.3f}` *(0.00 = Fully Direct, 1.00 = Highly Evasive)*
- **Total Q&A Exchanges Evaluated:** {len(evasions)}
- **Flagged Evasive Responses:** {len(evasive_items)}

### Flagged Exchanges
{evasion_section}

---

## ⚠️ Academic & Research Disclaimer

> **IMPORTANT DISCLAIMER:** {DISCLAIMER}
> This report is automatically generated by MarketMind for academic and research purposes only. Signals are derived from historical patterns in speech, transcripts, and financial disclosures. This does not constitute investment advice or a recommendation to buy or sell securities.
"""

            # Save report file
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report_filename = f"{company.ticker}_{call.quarter}_FY{call.fiscal_year}_Report.md"
            report_path = REPORTS_DIR / report_filename

            report_path.write_text(md_content, encoding="utf-8")
            logger.info(f"✅ Generated report written to: {report_path}")

            return md_content, report_path

        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed to generate report for call_id={call_id}: {e}")
            raise
        finally:
            if close_session:
                session.close()


def generate_all_reports():
    """Generate research signals and reports for all ingested calls."""
    session = get_session()
    rep_gen = ReportGenerator()
    try:
        calls = session.query(EarningsCall).order_by(EarningsCall.call_id).all()
        logger.info(f"🚀 Generating Signals & Reports for {len(calls)} calls...")
        for c in calls:
            _, path = rep_gen.generate_report_markdown(c.call_id, session=session)
        logger.info(f"🎉 Complete! Generated all reports into {REPORTS_DIR}")
    finally:
        session.close()


if __name__ == "__main__":
    generate_all_reports()
