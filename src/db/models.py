"""
MarketMind Database Models

SQLAlchemy 2.0 ORM models for all database tables.
"""

from datetime import UTC, datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Base class for all MarketMind models."""
    pass


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

class Company(Base):
    """Company metadata."""
    __tablename__ = "companies"

    company_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(
        String(10), unique=True, nullable=False, index=True
    )
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    exchange: Mapped[Optional[str]] = mapped_column(String(50))

    # Relationships
    earnings_calls: Mapped[List["EarningsCall"]] = relationship(
        back_populates="company", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Company(ticker='{self.ticker}', name='{self.company_name}')>"


# ---------------------------------------------------------------------------
# Earnings Calls
# ---------------------------------------------------------------------------

class EarningsCall(Base):
    """Individual earnings call record."""
    __tablename__ = "earnings_calls"

    call_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.company_id"), nullable=False
    )
    quarter: Mapped[str] = mapped_column(String(10), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    call_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    audio_path: Mapped[Optional[str]] = mapped_column(String(500))
    transcript_path: Mapped[Optional[str]] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        UniqueConstraint(
            "company_id", "quarter", "fiscal_year",
            name="uq_company_quarter_year",
        ),
        Index("ix_earnings_calls_status", "status"),
    )

    # Relationships
    company: Mapped["Company"] = relationship(back_populates="earnings_calls")
    segments: Mapped[List["TranscriptSegment"]] = relationship(
        back_populates="earnings_call", cascade="all, delete-orphan"
    )
    metrics: Mapped[List["ExtractedMetric"]] = relationship(
        back_populates="earnings_call", cascade="all, delete-orphan"
    )
    sentiments: Mapped[List["SentimentScore"]] = relationship(
        back_populates="earnings_call", cascade="all, delete-orphan"
    )
    evasions: Mapped[List["EvasionScore"]] = relationship(
        back_populates="earnings_call", cascade="all, delete-orphan"
    )
    findings: Mapped[List["AccountabilityFinding"]] = relationship(
        back_populates="earnings_call", cascade="all, delete-orphan"
    )
    signals: Mapped[List["Signal"]] = relationship(
        back_populates="earnings_call", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<EarningsCall(id={self.call_id}, {self.quarter} FY{self.fiscal_year})>"


# ---------------------------------------------------------------------------
# Transcript Segments
# ---------------------------------------------------------------------------

class TranscriptSegment(Base):
    """Individual segment / speaker turn of a transcript."""
    __tablename__ = "transcript_segments"

    segment_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    call_id: Mapped[int] = mapped_column(
        ForeignKey("earnings_calls.call_id"), nullable=False
    )
    start_time: Mapped[Optional[float]] = mapped_column(Float)
    end_time: Mapped[Optional[float]] = mapped_column(Float)
    speaker_label: Mapped[Optional[str]] = mapped_column(String(255))
    speaker_role: Mapped[Optional[str]] = mapped_column(String(50))
    segment_type: Mapped[Optional[str]] = mapped_column(String(50))
    text: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        Index("ix_segments_call_id", "call_id"),
        Index("ix_segments_speaker_role", "speaker_role"),
    )

    # Relationships
    earnings_call: Mapped["EarningsCall"] = relationship(back_populates="segments")

    def __repr__(self) -> str:
        preview = self.text[:40] + "..." if len(self.text) > 40 else self.text
        return f"<TranscriptSegment(id={self.segment_id}, speaker='{self.speaker_label}', text='{preview}')>"


# ---------------------------------------------------------------------------
# Extracted Metrics
# ---------------------------------------------------------------------------

class ExtractedMetric(Base):
    """Financial metric extracted from a transcript."""
    __tablename__ = "extracted_metrics"

    metric_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    call_id: Mapped[int] = mapped_column(
        ForeignKey("earnings_calls.call_id"), nullable=False
    )
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[Optional[float]] = mapped_column(Float)
    unit: Mapped[Optional[str]] = mapped_column(String(50))
    currency: Mapped[Optional[str]] = mapped_column(String(10))
    period: Mapped[Optional[str]] = mapped_column(String(50))
    is_guidance: Mapped[bool] = mapped_column(Boolean, default=False)
    speaker_role: Mapped[Optional[str]] = mapped_column(String(50))
    quote: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[Optional[float]] = mapped_column(Float)

    __table_args__ = (
        Index("ix_metrics_call_id", "call_id"),
        Index("ix_metrics_name", "metric_name"),
    )

    # Relationships
    earnings_call: Mapped["EarningsCall"] = relationship(back_populates="metrics")

    def __repr__(self) -> str:
        return f"<ExtractedMetric(name='{self.metric_name}', value={self.value})>"


# ---------------------------------------------------------------------------
# Sentiment Scores
# ---------------------------------------------------------------------------

class SentimentScore(Base):
    """Sentiment analysis result for a transcript segment."""
    __tablename__ = "sentiment_scores"

    sentiment_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    call_id: Mapped[int] = mapped_column(
        ForeignKey("earnings_calls.call_id"), nullable=False
    )
    segment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("transcript_segments.segment_id")
    )
    speaker_role: Mapped[Optional[str]] = mapped_column(String(50))
    positive_score: Mapped[float] = mapped_column(Float, nullable=False)
    neutral_score: Mapped[float] = mapped_column(Float, nullable=False)
    negative_score: Mapped[float] = mapped_column(Float, nullable=False)
    compound_score: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (Index("ix_sentiment_call_id", "call_id"),)

    # Relationships
    earnings_call: Mapped["EarningsCall"] = relationship(back_populates="sentiments")

    def __repr__(self) -> str:
        return f"<SentimentScore(compound={self.compound_score:.3f})>"


# ---------------------------------------------------------------------------
# Evasion Scores
# ---------------------------------------------------------------------------

class EvasionScore(Base):
    """Evasion detection result for a Q&A pair."""
    __tablename__ = "evasion_scores"

    evasion_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    call_id: Mapped[int] = mapped_column(
        ForeignKey("earnings_calls.call_id"), nullable=False
    )
    question_segment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("transcript_segments.segment_id")
    )
    answer_segment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("transcript_segments.segment_id")
    )
    evasion_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_evasive: Mapped[bool] = mapped_column(Boolean, default=False)
    reason: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (Index("ix_evasion_call_id", "call_id"),)

    # Relationships
    earnings_call: Mapped["EarningsCall"] = relationship(back_populates="evasions")

    def __repr__(self) -> str:
        return f"<EvasionScore(score={self.evasion_score:.3f}, evasive={self.is_evasive})>"


# ---------------------------------------------------------------------------
# Accountability Findings
# ---------------------------------------------------------------------------

class AccountabilityFinding(Base):
    """Historical accountability comparison result."""
    __tablename__ = "accountability_findings"

    finding_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    call_id: Mapped[int] = mapped_column(
        ForeignKey("earnings_calls.call_id"), nullable=False
    )
    finding_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[Optional[str]] = mapped_column(String(20))
    current_statement: Mapped[Optional[str]] = mapped_column(Text)
    historical_statement: Mapped[Optional[str]] = mapped_column(Text)
    historical_source: Mapped[Optional[str]] = mapped_column(String(255))
    explanation: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("ix_findings_call_id", "call_id"),
        Index("ix_findings_type", "finding_type"),
    )

    # Relationships
    earnings_call: Mapped["EarningsCall"] = relationship(back_populates="findings")

    def __repr__(self) -> str:
        return f"<AccountabilityFinding(type='{self.finding_type}', severity='{self.severity}')>"


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

class Signal(Base):
    """Market reaction prediction signal."""
    __tablename__ = "signals"

    signal_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    call_id: Mapped[int] = mapped_column(
        ForeignKey("earnings_calls.call_id"), nullable=False
    )
    predicted_direction: Mapped[str] = mapped_column(String(10), nullable=False)
    probability: Mapped[Optional[float]] = mapped_column(Float)
    expected_return: Mapped[Optional[float]] = mapped_column(Float)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))

    __table_args__ = (Index("ix_signals_call_id", "call_id"),)

    # Relationships
    earnings_call: Mapped["EarningsCall"] = relationship(back_populates="signals")

    def __repr__(self) -> str:
        return f"<Signal(direction='{self.predicted_direction}', prob={self.probability})>"


# ---------------------------------------------------------------------------
# Model Runs
# ---------------------------------------------------------------------------

class ModelRun(Base):
    """ML model training run metadata."""
    __tablename__ = "model_runs"

    run_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[Optional[str]] = mapped_column(String(50))
    metrics: Mapped[Optional[str]] = mapped_column(Text)  # JSON string
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

    def __repr__(self) -> str:
        return f"<ModelRun(model='{self.model_name}', version='{self.version}')>"
