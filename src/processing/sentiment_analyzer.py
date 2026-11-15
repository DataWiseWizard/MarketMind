"""
MarketMind Financial Sentiment Analysis Module

Calculates positive, neutral, negative, and compound sentiment scores for
transcript segments using FinBERT (ProsusAI/finbert) or an audited Loughran-McDonald
financial lexicon fallback when deep learning models are offline/uninstalled.
Persists structured scores into the sentiment_scores database table.
"""

import math
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
    EarningsCall, TranscriptSegment, SentimentScore,
    SpeakerRole, SegmentType
)

# Financial Sentiment Lexicon (Loughran & McDonald Financial Sentiment Dictionary subset)
LM_POSITIVE = {
    "growth", "record", "strong", "strength", "exceed", "exceeded", "momentum",
    "profit", "profitable", "gain", "gains", "expand", "expansion", "robust",
    "opportunity", "opportunities", "solid", "outperform", "outperformed",
    "accelerate", "accelerated", "innovation", "innovative", "upside", "benefit",
    "benefits", "highest", "demand", "advancement", "efficiency", "milestone",
    "optimistic", "favorable", "success", "successful", "boost", "leading"
}

LM_NEGATIVE = {
    "decline", "declined", "declining", "loss", "losses", "drop", "dropped",
    "fall", "fallen", "weak", "weakness", "headwind", "headwinds", "risk",
    "risks", "uncertainty", "uncertainties", "challenge", "challenges",
    "disruption", "disruptions", "adverse", "adversely", "inflation",
    "recession", "decrease", "decreased", "impairment", "write-down",
    "slowdown", "deficit", "pressure", "pressures", "shortage", "shortages",
    "unfavorable", "penalty", "penalties", "concern", "concerns", "downturn"
}


class SentimentAnalyzer:
    """Analyzes sentiment of earnings call segments."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.use_transformer = False
        self.pipe = None

        # Attempt to load FinBERT pipeline if transformers and torch are available
        try:
            from transformers import BertTokenizer, BertForSequenceClassification, pipeline
            import torch
            logger.info(f"Loading FinBERT model ({model_name})...")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name)
            self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, return_all_scores=True)
            self.use_transformer = True
            logger.info("✅ FinBERT model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"⚠️ FinBERT/PyTorch not initialized ({e}). "
                "Using Loughran-McDonald Financial Lexicon fallback engine."
            )
            self.use_transformer = False

    def analyze_text_lexicon(self, text: str) -> Dict[str, float]:
        """Compute financial sentiment using Loughran-McDonald financial dictionary."""
        words = re.findall(r"\b[a-zA-Z-]+\b", text.lower())
        if not words:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0, "compound": 0.0}

        pos_count = sum(1 for w in words if w in LM_POSITIVE)
        neg_count = sum(1 for w in words if w in LM_NEGATIVE)
        total = len(words)

        # Scale intensity
        pos_intensity = min(1.0, (pos_count / math.sqrt(total + 10)) * 2.5)
        neg_intensity = min(1.0, (neg_count / math.sqrt(total + 10)) * 2.5)

        raw_diff = pos_count - neg_count
        compound = float(raw_diff) / math.sqrt(raw_diff * raw_diff + 15)

        # Probabilities summing to 1.0
        diff = pos_intensity - neg_intensity
        if diff > 0:
            pos_score = min(0.95, 0.33 + pos_intensity * 0.6)
            neg_score = max(0.02, 0.33 - pos_intensity * 0.3)
            neu_score = max(0.03, 1.0 - (pos_score + neg_score))
        elif diff < 0:
            neg_score = min(0.95, 0.33 + neg_intensity * 0.6)
            pos_score = max(0.02, 0.33 - neg_intensity * 0.3)
            neu_score = max(0.03, 1.0 - (pos_score + neg_score))
        else:
            pos_score = 0.20
            neg_score = 0.20
            neu_score = 0.60

        return {
            "positive": round(pos_score, 4),
            "neutral": round(neu_score, 4),
            "negative": round(neg_score, 4),
            "compound": round(compound, 4),
        }

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze text with FinBERT or fallback lexicon."""
        if not text or not text.strip():
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0, "compound": 0.0}

        if self.use_transformer and self.pipe:
            try:
                # Truncate text to 512 tokens max
                truncated = text[:1500]
                res = self.pipe(truncated)[0]
                # res is list of {'label': 'positive'|'negative'|'neutral', 'score': float}
                score_dict = {item["label"].lower(): item["score"] for item in res}
                pos = score_dict.get("positive", 0.0)
                neu = score_dict.get("neutral", 0.0)
                neg = score_dict.get("negative", 0.0)
                compound = pos - neg
                return {
                    "positive": round(pos, 4),
                    "neutral": round(neu, 4),
                    "negative": round(neg, 4),
                    "compound": round(compound, 4),
                }
            except Exception as e:
                logger.warning(f"Transformer inference failed: {e}. Falling back to lexicon.")

        return self.analyze_text_lexicon(text)

    def analyze_for_call(self, call_id: int, session: Optional[Session] = None) -> List[SentimentScore]:
        """Compute and persist sentiment scores for all segments in a call."""
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

            # Clear existing sentiment scores for fresh calculation
            session.query(SentimentScore).filter_by(call_id=call_id).delete()
            session.flush()

            saved_scores = []
            for seg in segments:
                scores = self.analyze_text(seg.text)
                sent_obj = SentimentScore(
                    call_id=call_id,
                    segment_id=seg.segment_id,
                    speaker_role=seg.speaker_role,
                    positive_score=scores["positive"],
                    neutral_score=scores["neutral"],
                    negative_score=scores["negative"],
                    compound_score=scores["compound"],
                )
                session.add(sent_obj)
                saved_scores.append(sent_obj)

            session.commit()
            logger.info(f"✅ Analyzed sentiment for {len(saved_scores)} segments in Call ID {call_id}.")
            return saved_scores

        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed sentiment analysis for call_id={call_id}: {e}")
            raise
        finally:
            if close_session:
                session.close()

    def get_aggregated_sentiment(self, call_id: int, session: Optional[Session] = None) -> Dict[str, Any]:
        """Aggregate sentiment scores by speaker role and overall call."""
        close_session = False
        if session is None:
            session = get_session()
            close_session = True

        try:
            scores = session.query(SentimentScore).filter_by(call_id=call_id).all()
            if not scores:
                return {}

            by_role: Dict[str, List[float]] = {}
            all_compounds = []

            for s in scores:
                role_str = s.speaker_role.value if s.speaker_role else "unknown"
                if role_str not in by_role:
                    by_role[role_str] = []
                by_role[role_str].append(s.compound_score)
                all_compounds.append(s.compound_score)

            summary = {
                "call_id": call_id,
                "overall_sentiment": round(sum(all_compounds) / len(all_compounds), 4) if all_compounds else 0.0,
                "roles": {
                    role: round(sum(vals) / len(vals), 4)
                    for role, vals in by_role.items()
                },
                "total_segments_scored": len(scores),
            }
            return summary
        finally:
            if close_session:
                session.close()


def analyze_all_calls():
    """Run sentiment analysis across all calls in the database."""
    session = get_session()
    analyzer = SentimentAnalyzer()
    try:
        calls = session.query(EarningsCall).order_by(EarningsCall.call_id).all()
        logger.info(f"🚀 Running Sentiment Analysis on {len(calls)} calls...")
        total_scored = 0
        for c in calls:
            scores = analyzer.analyze_for_call(c.call_id, session=session)
            total_scored += len(scores)
            agg = analyzer.get_aggregated_sentiment(c.call_id, session=session)
            logger.info(
                f"  Call {c.call_id} ({c.quarter} FY{c.fiscal_year}): "
                f"Overall Compound={agg['overall_sentiment']}, Roles={agg['roles']}"
            )
        logger.info(f"🎉 Complete! Scored {total_scored} segments across all calls.")
    finally:
        session.close()


if __name__ == "__main__":
    analyze_all_calls()
