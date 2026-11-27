"""
MarketMind Database Models

Database schema using SQLAlchemy ORM.
Tables: companies, earnings_calls, transcript_segments, extracted_metrics,
        sentiment_scores, evasion_scores, accountability_findings, signals, model_runs
"""

from datetime import datetime, date, timezone
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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

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
    speaker_label = Column(String(500))  # raw speaker name
    speaker_role = Column(SQLEnum(SpeakerRole), default=SpeakerRole.UNKNOWN)
    segment_type = Column(SQLEnum(SegmentType), default=SegmentType.OTHER)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    earnings_call = relationship("EarningsCall", back_populates="due_diligence_questions", foreign_keys=[call_id])
    peer_call = relationship("EarningsCall", foreign_keys=[peer_call_id])

    def __repr__(self):
        return f"<DueDiligenceQuestion(rank={self.rank}, call_id={self.call_id})>"

