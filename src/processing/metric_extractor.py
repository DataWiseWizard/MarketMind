"""
MarketMind Financial Metric Extraction Module

Extracts key financial metrics (revenue, gross margin, EPS, guidance outlook, capex)
from parsed transcript segments with exact source quotes and confidence scoring.
Persists structured records into the extracted_metrics database table.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from sqlalchemy.orm import Session

from src.db.connection import get_session
from src.db.models import (
    EarningsCall, TranscriptSegment, ExtractedMetric,
    SpeakerRole, SegmentType
)


class FinancialMetricExtractor:
    """Extracts reported financial metrics and guidance from earnings call segments."""

    def __init__(self):
        # Patterns for values
        self.dollar_num_re = r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(billion|million|trillion|B|M|T)?"
        self.pct_re = r"([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)"

    def _parse_dollar_value(self, num_str: str, magnitude: Optional[str]) -> Tuple[float, str]:
        """Convert number + magnitude into standard numeric value and unit."""
        val = float(num_str)
        mag = (magnitude or "").lower()
        if mag in ["billion", "b"]:
            return val, "USD_B"
        elif mag in ["million", "m"]:
            return val / 1000.0, "USD_B"
        elif mag in ["trillion", "t"]:
            return val * 1000.0, "USD_B"
        return val, "USD"

    def extract_from_sentence(
        self,
        sentence: str,
        speaker_role: SpeakerRole,
        is_outlook_context: bool,
        call_period: str,
    ) -> List[Dict[str, Any]]:
        """Extract financial metrics from a single sentence."""
        metrics = []
        s_clean = sentence.strip()
        s_lower = s_clean.lower()

        # 1. Total Revenue (Reported)
        if ("revenue of $" in s_lower or "revenue was $" in s_lower or "revenue reached $" in s_lower or "total revenue was $" in s_lower) and not is_outlook_context:
            m = re.search(r"(?:total\s+)?revenue\s+(?:of|was|reached)\s+" + self.dollar_num_re, s_clean, re.I)
            if m:
                val, unit = self._parse_dollar_value(m.group(1), m.group(2))
                metrics.append({
                    "metric_name": "revenue",
                    "value": val,
                    "unit": unit,
                    "currency": "USD",
                    "period": call_period,
                    "is_guidance": False,
                    "speaker_role": speaker_role,
                    "quote": s_clean,
                    "confidence": 0.95,
                })

        # 2. Data Center Revenue
        if "data center" in s_lower and ("revenue of $" in s_lower or "revenue was $" in s_lower or "record $" in s_lower):
            m = re.search(r"data\s+center(?:\s+revenue)?\s+(?:of|was|reached|record)\s+" + self.dollar_num_re, s_clean, re.I)
            if m:
                val, unit = self._parse_dollar_value(m.group(1), m.group(2))
                metrics.append({
                    "metric_name": "datacenter_revenue",
                    "value": val,
                    "unit": unit,
                    "currency": "USD",
                    "period": call_period,
                    "is_guidance": False,
                    "speaker_role": speaker_role,
                    "quote": s_clean,
                    "confidence": 0.95,
                })

        # 3. Revenue Growth %
        if ("revenue" in s_lower or "up" in s_lower or "grew" in s_lower) and ("year-over-year" in s_lower or "year-on-year" in s_lower or "yoy" in s_lower):
            m = re.search(r"(?:up|grew|growth of)\s+" + self.pct_re + r"\s*(?:year-over-year|year-on-year|yoy)", s_clean, re.I)
            if m:
                val = float(m.group(1))
                metrics.append({
                    "metric_name": "revenue_growth_yoy",
                    "value": val,
                    "unit": "percent",
                    "currency": None,
                    "period": call_period,
                    "is_guidance": False,
                    "speaker_role": speaker_role,
                    "quote": s_clean,
                    "confidence": 0.90,
                })

        # 4. Gross Margin % (Reported)
        if "gross margin" in s_lower and not is_outlook_context:
            # Check GAAP vs Non-GAAP
            is_non_gaap = "non-gaap" in s_lower
            m = re.search(r"gross\s+margin\s+(?:was|of|reached)\s+" + self.pct_re, s_clean, re.I)
            if m:
                val = float(m.group(1))
                metric_name = "non_gaap_gross_margin" if is_non_gaap else "gaap_gross_margin"
                metrics.append({
                    "metric_name": metric_name,
                    "value": val,
                    "unit": "percent",
                    "currency": None,
                    "period": call_period,
                    "is_guidance": False,
                    "speaker_role": speaker_role,
                    "quote": s_clean,
                    "confidence": 0.95,
                })

        # 5. Diluted EPS (Reported)
        if ("diluted earnings per share" in s_lower or "diluted eps" in s_lower or "eps was $" in s_lower) and not is_outlook_context:
            is_non_gaap = "non-gaap" in s_lower
            m = re.search(r"(?:eps|earnings per share)\s+(?:was|of)\s+\$([0-9]+(?:\.[0-9]+)?)", s_clean, re.I)
            if m:
                val = float(m.group(1))
                metric_name = "non_gaap_eps" if is_non_gaap else "gaap_eps"
                metrics.append({
                    "metric_name": metric_name,
                    "value": val,
                    "unit": "USD",
                    "currency": "USD",
                    "period": call_period,
                    "is_guidance": False,
                    "speaker_role": speaker_role,
                    "quote": s_clean,
                    "confidence": 0.95,
                })

        # 6. Forward Guidance - Revenue
        if is_outlook_context or "outlook" in s_lower or "guidance" in s_lower or "expected to be" in s_lower:
            m = re.search(
                r"(?:total\s+)?revenue\s+is\s+expected\s+to\s+be\s+" + self.dollar_num_re,
                s_clean,
                re.I,
            ) or re.search(
                r"(?:expect|forecast)\s+(?:total\s+)?revenue\s+(?:of|to be)\s+" + self.dollar_num_re,
                s_clean,
                re.I,
            )
            if m:
                val, unit = self._parse_dollar_value(m.group(1), m.group(2))
                metrics.append({
                    "metric_name": "guidance_revenue",
                    "value": val,
                    "unit": unit,
                    "currency": "USD",
                    "period": "Next Quarter Outlook",
                    "is_guidance": True,
                    "speaker_role": speaker_role,
                    "quote": s_clean,
                    "confidence": 0.95,
                })

            # Guidance Gross Margin %
            m_gm = re.search(
                r"gross\s+margins?\s+(?:are|is)?\s*expected\s+to\s+be\s+" + self.pct_re,
                s_clean,
                re.I,
            ) or re.search(
                r"gross\s+margins?\s+outlook\s+is\s+" + self.pct_re,
                s_clean,
                re.I,
            )
            if m_gm:
                val = float(m_gm.group(1))
                is_non_gaap = "non-gaap" in s_lower
                metric_name = "guidance_non_gaap_gross_margin" if is_non_gaap else "guidance_gaap_gross_margin"
                metrics.append({
                    "metric_name": metric_name,
                    "value": val,
                    "unit": "percent",
                    "currency": None,
                    "period": "Next Quarter Outlook",
                    "is_guidance": True,
                    "speaker_role": speaker_role,
                    "quote": s_clean,
                    "confidence": 0.95,
                })

        return metrics

    def extract_from_segment(self, segment: TranscriptSegment, call_period: str) -> List[Dict[str, Any]]:
        """Split segment into sentences and extract financial metrics."""
        text = segment.text
        # Check if segment or paragraph is an outlook section
        sentences = re.split(r"(?<=[.!?])\s+", text)
        is_outlook_active = False

        extracted = []
        for s in sentences:
            s_low = s.lower()
            if any(k in s_low for k in ["turn to the outlook", "outlook for the", "guidance for the"]):
                is_outlook_active = True

            results = self.extract_from_sentence(
                sentence=s,
                speaker_role=segment.speaker_role,
                is_outlook_context=is_outlook_active,
                call_period=call_period,
            )
            for item in results:
                item["call_id"] = segment.call_id
                extracted.append(item)

        return extracted

    def extract_for_call(self, call_id: int, session: Optional[Session] = None) -> List[ExtractedMetric]:
        """Extract and persist all metrics for a given call ID."""
        close_session = False
        if session is None:
            session = get_session()
            close_session = True

        try:
            call = session.query(EarningsCall).filter_by(call_id=call_id).first()
            if not call:
                raise ValueError(f"EarningsCall ID {call_id} not found.")

            call_period = f"{call.quarter} FY{call.fiscal_year}"

            # Query segments (prioritizing prepared remarks and CFO/CEO responses)
            segments = (
                session.query(TranscriptSegment)
                .filter_by(call_id=call_id)
                .order_by(TranscriptSegment.segment_order)
                .all()
            )

            # Delete any prior metrics for this call to allow clean re-run
            session.query(ExtractedMetric).filter_by(call_id=call_id).delete()
            session.flush()

            saved_metrics = []
            seen_signatures = set()

            for seg in segments:
                # We extract from prepared remarks and executive answers
                if seg.segment_type in [SegmentType.PREPARED_REMARKS, SegmentType.QA_ANSWER]:
                    items = self.extract_from_segment(seg, call_period)
                    for item in items:
                        # Avoid duplicates
                        sig = (item["metric_name"], item["value"], item["is_guidance"])
                        if sig in seen_signatures:
                            continue
                        seen_signatures.add(sig)

                        metric_obj = ExtractedMetric(
                            call_id=call_id,
                            metric_name=item["metric_name"],
                            value=item["value"],
                            unit=item["unit"],
                            currency=item["currency"],
                            period=item["period"],
                            is_guidance=item["is_guidance"],
                            speaker_role=item["speaker_role"],
                            quote=item["quote"],
                            confidence=item["confidence"],
                        )
                        session.add(metric_obj)
                        saved_metrics.append(metric_obj)

            session.commit()
            logger.info(f"✅ Extracted {len(saved_metrics)} metrics for Call ID {call_id} ({call_period}).")
            return saved_metrics

        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed metric extraction for call_id={call_id}: {e}")
            raise
        finally:
            if close_session:
                session.close()


def extract_all_calls():
    """Run financial metric extraction across all ingested calls in the database."""
    session = get_session()
    extractor = FinancialMetricExtractor()
    try:
        calls = session.query(EarningsCall).order_by(EarningsCall.call_id).all()
        logger.info(f"🚀 Running Financial Metric Extraction on {len(calls)} calls...")
        total_extracted = 0
        for c in calls:
            metrics = extractor.extract_for_call(c.call_id, session=session)
            total_extracted += len(metrics)
        logger.info(f"🎉 Complete! Extracted a total of {total_extracted} metrics.")
    finally:
        session.close()


if __name__ == "__main__":
    extract_all_calls()
