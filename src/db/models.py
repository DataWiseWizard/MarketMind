"""
MarketMind Database Models

<<<<<<< HEAD
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
=======
Database schema using SQLAlchemy ORM.
Tables: companies, earnings_calls, transcript_segments, extracted_metrics,
        sentiment_scores, evasion_scores, accountability_findings, signals, model_runs
"""

from datetime import datetime, date
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, Boolean,
    DateTime, Date, ForeignKey, JSON, Numeric, Enum as SQLEnum
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import QueuePool
import enum

Base = declarative_base()


# ---- Enums ----

class CallStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SpeakerRole(enum.Enum):
    CEO = "ceo"
    CFO = "cfo"
    COO = "coo"
    CTO = "cto"
    ANALYST = "analyst"
    OPERATOR = "operator"
    OTHER = "other"
    UNKNOWN = "unknown"


class SegmentType(enum.Enum):
    PREPARED_REMARKS = "prepared_remarks"
    QA_QUESTION = "qa_question"
    QA_ANSWER = "qa_answer"
    OPERATOR = "operator"
    OTHER = "other"


class FindingType(enum.Enum):
    GUIDANCE_RAISED = "guidance_raised"
    GUIDANCE_LOWERED = "guidance_lowered"
    BROKEN_PROMISE = "broken_promise"
    CONTRADICTION = "contradiction"
    AVOIDED_TOPIC = "avoided_topic"


class Severity(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PredictedDirection(enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# ---- Models ----

class Company(Base):
    """Company information"""
    __tablename__ = "companies"

    company_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), unique=True, nullable=False, index=True)
    company_name = Column(String(255), nullable=False)
    sector = Column(String(100))
    exchange = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    earnings_calls = relationship("EarningsCall", back_populates="company")

    def __repr__(self):
        return f"<Company(ticker='{self.ticker}', name='{self.company_name}')>"


class EarningsCall(Base):
    """Individual earnings call record"""
    __tablename__ = "earnings_calls"

    call_id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("companies.company_id"), nullable=False)
    quarter = Column(String(5), nullable=False)  # Q1, Q2, Q3, Q4
    fiscal_year = Column(Integer, nullable=False)
    call_date = Column(Date, nullable=False)
    audio_path = Column(String(500))
    audio_uri = Column(String(500))  # Rev. 2: Explicit URI/path for audio playback
    transcript_path = Column(String(500))
    status = Column(SQLEnum(CallStatus), default=CallStatus.PENDING)
    source = Column(String(100))  # e.g., "uploaded", "sec_edgar", "seeking_alpha"
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    company = relationship("Company", back_populates="earnings_calls")
    segments = relationship("TranscriptSegment", back_populates="earnings_call")
    metrics = relationship("ExtractedMetric", back_populates="earnings_call")
    sentiments = relationship("SentimentScore", back_populates="earnings_call")
    evasions = relationship("EvasionScore", back_populates="earnings_call")
    findings = relationship("AccountabilityFinding", back_populates="earnings_call")
    signals = relationship("Signal", back_populates="earnings_call")
    due_diligence_questions = relationship(
        "DueDiligenceQuestion",
        back_populates="earnings_call",
        foreign_keys="[DueDiligenceQuestion.call_id]"
    )

    def __repr__(self):
        return f"<EarningsCall(call_id={self.call_id}, {self.quarter} FY{self.fiscal_year})>"


class TranscriptSegment(Base):
    """Individual segment of a transcript"""
    __tablename__ = "transcript_segments"

    segment_id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey("earnings_calls.call_id"), nullable=False)
    segment_order = Column(Integer, nullable=False)
    start_time = Column(Float)  # seconds (if from audio)
    end_time = Column(Float)
    speaker_label = Column(String(100))  # raw speaker name
    speaker_role = Column(SQLEnum(SpeakerRole), default=SpeakerRole.UNKNOWN)
    segment_type = Column(SQLEnum(SegmentType), default=SegmentType.OTHER)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    earnings_call = relationship("EarningsCall", back_populates="segments")
    sentiments = relationship("SentimentScore", back_populates="segment")

    def __repr__(self):
        return f"<Segment(id={self.segment_id}, speaker='{self.speaker_label}')>"


class ExtractedMetric(Base):
    """Financial metrics extracted from transcript"""
    __tablename__ = "extracted_metrics"

    metric_id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey("earnings_calls.call_id"), nullable=False)
    metric_name = Column(String(100), nullable=False)  # e.g., "revenue", "eps", "gross_margin"
    value = Column(Float)
    unit = Column(String(50))  # e.g., "USD", "percent", "ratio"
    currency = Column(String(10))
    period = Column(String(50))  # e.g., "Q2 FY2026", "FY2026"
    is_guidance = Column(Boolean, default=False)
    speaker_role = Column(SQLEnum(SpeakerRole))
    quote = Column(Text)  # original text for verification
    confidence = Column(Float)  # extraction confidence 0-1
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    earnings_call = relationship("EarningsCall", back_populates="metrics")

    def __repr__(self):
        return f"<Metric(name='{self.metric_name}', value={self.value})>"


class SentimentScore(Base):
    """Sentiment analysis scores"""
    __tablename__ = "sentiment_scores"

    sentiment_id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey("earnings_calls.call_id"), nullable=False)
    segment_id = Column(Integer, ForeignKey("transcript_segments.segment_id"))
    speaker_role = Column(SQLEnum(SpeakerRole))
    positive_score = Column(Float, nullable=False)
    neutral_score = Column(Float, nullable=False)
    negative_score = Column(Float, nullable=False)
    compound_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    earnings_call = relationship("EarningsCall", back_populates="sentiments")
    segment = relationship("TranscriptSegment", back_populates="sentiments")

    def __repr__(self):
        return f"<Sentiment(compound={self.compound_score:.3f})>"


class EvasionScore(Base):
    """Evasion detection scores for Q&A pairs"""
    __tablename__ = "evasion_scores"

    evasion_id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey("earnings_calls.call_id"), nullable=False)
    question_segment_id = Column(Integer, ForeignKey("transcript_segments.segment_id"))
    answer_segment_id = Column(Integer, ForeignKey("transcript_segments.segment_id"))
    evasion_score = Column(Float, nullable=False)  # 0-1
    is_evasive = Column(Boolean, default=False)
    reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    earnings_call = relationship("EarningsCall", back_populates="evasions")
    question = relationship("TranscriptSegment", foreign_keys=[question_segment_id])
    answer = relationship("TranscriptSegment", foreign_keys=[answer_segment_id])

    def __repr__(self):
        return f"<Evasion(score={self.evasion_score:.3f}, evasive={self.is_evasive})>"


class AccountabilityFinding(Base):
    """Findings from historical comparison (RAG)"""
    __tablename__ = "accountability_findings"

    finding_id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey("earnings_calls.call_id"), nullable=False)
    finding_type = Column(SQLEnum(FindingType), nullable=False)
    severity = Column(SQLEnum(Severity), default=Severity.MEDIUM)
    current_statement = Column(Text, nullable=False)
    historical_statement = Column(Text)
    historical_source = Column(String(255))  # e.g., "Q1 FY2026 call"
    explanation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    earnings_call = relationship("EarningsCall", back_populates="findings")

    def __repr__(self):
        return f"<Finding(type='{self.finding_type}', severity='{self.severity}')>"


class Signal(Base):
    """Market reaction prediction signals"""
    __tablename__ = "signals"

    signal_id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey("earnings_calls.call_id"), nullable=False)
    predicted_direction = Column(SQLEnum(PredictedDirection), nullable=False)
    probability = Column(Float)  # 0-1
    expected_return = Column(Float)  # percentage
    confidence = Column(Float)  # 0-1
    model_version = Column(String(50))
    top_features = Column(JSON)  # SHAP top drivers
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    earnings_call = relationship("EarningsCall", back_populates="signals")

    def __repr__(self):
        return f"<Signal(direction='{self.predicted_direction}', confidence={self.confidence:.3f})>"


class ModelRun(Base):
    """ML model experiment tracking"""
    __tablename__ = "model_runs"

    run_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50))
    metrics = Column(JSON)  # accuracy, precision, recall, f1, etc.
    parameters = Column(JSON)  # hyperparameters
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ModelRun(model='{self.model_name}', version='{self.version}')>"


class DueDiligenceQuestion(Base):
    """Top unasked counter-questions grounded in peer/historical analysis (Rev. 2)"""
    __tablename__ = "due_diligence_questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey("earnings_calls.call_id"), nullable=False)
    rank = Column(Integer, nullable=False)  # 1 to 5
    question_text = Column(Text, nullable=False)
    peer_evidence_quote = Column(Text)
    peer_call_id = Column(Integer, ForeignKey("earnings_calls.call_id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    earnings_call = relationship("EarningsCall", back_populates="due_diligence_questions", foreign_keys=[call_id])
    peer_call = relationship("EarningsCall", foreign_keys=[peer_call_id])

    def __repr__(self):
        return f"<DueDiligenceQuestion(rank={self.rank}, call_id={self.call_id})>"

>>>>>>> 11a2f64dd7a562063e9d3a66d9182b0d5cb0070a
