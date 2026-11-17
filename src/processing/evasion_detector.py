"""
MarketMind Evasion Detection Module

Analyzes Question & Answer (Q&A) pairs during earnings calls to detect
evasive, vague, hedging, or deflective executive responses.
Persists structured scores and explainable rationales into the evasion_scores table.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from sqlalchemy.orm import Session

from src.db.connection import get_session
from src.db.models import (
    EarningsCall, TranscriptSegment, EvasionScore,
    SpeakerRole, SegmentType
)

# Linguistic patterns indicative of evasion or refusal to answer directly
EVASIVE_PHRASES = [
    r"don'?t break (that|it) out",
    r"not breaking (that|it) down",
    r"too early to (tell|say|predict|call)",
    r"hard to (predict|say|tell|quantify)",
    r"monitoring the situation",
    r"remain focused on the long[- ]term",
    r"focused on long[- ]term growth",
    r"not providing specific (guidance|color|numbers|breakdowns)",
    r"not going to give a specific number",
    r"we will see as things evolve",
    r"time will tell",
    r"can'?t comment on (specific|customer|future)",
    r"that remains to be seen",
    r"at this point in time.*?premature",
    r"fluid situation",
    r"moving target",
]

# Patterns where analyst specifically asks for quantification
QUANT_QUESTIONS = [
    r"how much",
    r"how many",
    r"what percent(?:age)?",
    r"can you quantify",
    r"could you quantify",
    r"give us a sense (?:for|of) (?:how much|the size|the number)",
    r"what was the margin impact",
    r"dollar amount",
    r"what is the breakdown",
    r"what run rate",
    r"what volume",
    r"what portion",
]


class EvasionDetector:
    """Detects evasiveness in earnings call executive responses."""

    def __init__(self, evasion_threshold: float = 0.50):
        self.threshold = evasion_threshold

    def is_quant_requested(self, question_text: str) -> bool:
        """Check if the analyst asked for numerical or quantifiable data."""
        q_low = question_text.lower()
        return any(re.search(p, q_low) for p in QUANT_QUESTIONS)

    def contains_numbers_or_metrics(self, answer_text: str) -> bool:
        """Check if the answer provided numbers, dollar figures, or percentages."""
        has_num = bool(re.search(r"\$\s*[0-9]+(?:\.[0-9]+)?", answer_text))
        has_pct = bool(re.search(r"[0-9]+(?:\.[0-9]+)?\s*%", answer_text))
        has_digit = bool(re.search(r"\b[0-9]+(?:\.[0-9]+)?\s*(?:billion|million|thousand|basis points|bps)\b", answer_text, re.I))
        return has_num or has_pct or has_digit

    def find_evasive_markers(self, answer_text: str) -> List[str]:
        """Find explicit deflection or hedging phrases in the answer."""
        a_low = answer_text.lower()
        matches = []
        for p in EVASIVE_PHRASES:
            m = re.search(p, a_low)
            if m:
                matches.append(m.group(0))
        return matches

    def evaluate_pair(
        self,
        question_text: str,
        answer_text: str,
        questioner_name: str = "Analyst",
        executive_name: str = "Executive",
    ) -> Dict[str, Any]:
        """
        Evaluate a single Q&A pair.
        Returns evasion_score (0.0 to 1.0), is_evasive (bool), and explanation reason.
        """
        score = 0.10  # Baseline low score
        reasons = []

        quant_requested = self.is_quant_requested(question_text)
        numbers_provided = self.contains_numbers_or_metrics(answer_text)
        evasion_markers = self.find_evasive_markers(answer_text)

        # 1. Quantification Mismatch
        if quant_requested and not numbers_provided:
            score += 0.40
            reasons.append("Analyst requested specific quantification or numbers, but the executive gave none.")
        elif quant_requested and numbers_provided:
            score -= 0.10  # Rewarded for direct figures

        # 2. Deflection / Hedging Markers
        if evasion_markers:
            score += min(0.40, 0.20 * len(evasion_markers))
            markers_str = ", ".join(f'"{m}"' for m in evasion_markers[:2])
            reasons.append(f"Executive used non-committal or hedging phrasing ({markers_str}).")

        # 3. Answer Length Anomaly (Extremely short evasion)
        words = answer_text.strip().split()
        if len(words) < 25 and not numbers_provided:
            score += 0.15
            reasons.append("Very brief response without substantive operational details.")

        # Clamp score between 0.0 and 1.0
        final_score = round(max(0.05, min(0.95, score)), 3)
        is_evasive = final_score >= self.threshold

        if not reasons:
            if numbers_provided:
                reason = "Direct response with specific figures and operational detail."
            else:
                reason = "Qualitative response addressing question scope directly."
        else:
            reason = " ".join(reasons)

        return {
            "evasion_score": final_score,
            "is_evasive": is_evasive,
            "reason": reason,
        }

    def pair_qa_segments(self, segments: List[TranscriptSegment]) -> List[Tuple[TranscriptSegment, TranscriptSegment]]:
        """Pair sequential QA_QUESTION segments with their matching QA_ANSWER segments."""
        pairs = []
        current_q = None

        for seg in segments:
            if seg.segment_type == SegmentType.QA_QUESTION:
                current_q = seg
            elif seg.segment_type == SegmentType.QA_ANSWER and current_q is not None:
                pairs.append((current_q, seg))
                # Note: We don't reset current_q to None immediately if multiple executives answer,
                # but to avoid duplicate pairings we pair with the primary respondent.
                current_q = None

        return pairs

    def analyze_for_call(self, call_id: int, session: Optional[Session] = None) -> List[EvasionScore]:
        """Run evasion detection for all Q&A pairs in an earnings call."""
        close_session = False
        if session is None:
            session = get_session()
            close_session = True

        try:
            segments = (
                session.query(TranscriptSegment)
                .filter_by(call_id=call_id)
                .order_by(TranscriptSegment.segment_order)
                .all()
            )

            pairs = self.pair_qa_segments(segments)
            if not pairs:
                logger.info(f"No Q&A pairs found for Call ID {call_id}.")
                return []

            # Clear existing evasion scores for clean run
            session.query(EvasionScore).filter_by(call_id=call_id).delete()
            session.flush()

            saved_scores = []
            for q_seg, a_seg in pairs:
                result = self.evaluate_pair(
                    question_text=q_seg.text,
                    answer_text=a_seg.text,
                    questioner_name=q_seg.speaker_label or "Analyst",
                    executive_name=a_seg.speaker_label or "Executive",
                )

                ev_obj = EvasionScore(
                    call_id=call_id,
                    question_segment_id=q_seg.segment_id,
                    answer_segment_id=a_seg.segment_id,
                    evasion_score=result["evasion_score"],
                    is_evasive=result["is_evasive"],
                    reason=result["reason"],
                )
                session.add(ev_obj)
                saved_scores.append(ev_obj)

            session.commit()
            evasive_count = sum(1 for s in saved_scores if s.is_evasive)
            avg_evasion = sum(s.evasion_score for s in saved_scores) / len(saved_scores) if saved_scores else 0.0

            logger.info(
                f"✅ Scored {len(saved_scores)} Q&A pairs for Call ID {call_id}: "
                f"Avg Evasion={avg_evasion:.3f}, Evasive Flagged={evasive_count}."
            )
            return saved_scores

        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed evasion analysis for call_id={call_id}: {e}")
            raise
        finally:
            if close_session:
                session.close()


def analyze_all_calls():
    """Run evasion detection on all earnings calls in database."""
    session = get_session()
    detector = EvasionDetector()
    try:
        calls = session.query(EarningsCall).order_by(EarningsCall.call_id).all()
        logger.info(f"🚀 Running Evasion Detection on {len(calls)} calls...")
        total_pairs = 0
        total_evasive = 0
        for c in calls:
            scores = detector.analyze_for_call(c.call_id, session=session)
            total_pairs += len(scores)
            total_evasive += sum(1 for s in scores if s.is_evasive)
        logger.info(f"🎉 Complete! Analyzed {total_pairs} Q&A pairs ({total_evasive} flagged as evasive).")
    finally:
        session.close()


if __name__ == "__main__":
    analyze_all_calls()
