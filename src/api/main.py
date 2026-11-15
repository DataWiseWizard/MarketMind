"""
MarketMind FastAPI REST API

Provides endpoints to query companies, earnings calls, financial metrics,
sentiment scores, evasion analysis, research signals, and trigger analysis runs.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from loguru import logger

from src.config import API_HOST, API_PORT, DEBUG, DISCLAIMER, RAW_DATA_DIR
from src.db.connection import get_session, init_database
from src.db.models import (
    Company, EarningsCall, TranscriptSegment, ExtractedMetric,
    SentimentScore, EvasionScore, Signal
)
from src.pipeline import MarketMindPipeline
from src.reporting.report_generator import ReportGenerator

app = FastAPI(
    title="MarketMind Financial Intelligence API",
    description="Automated Financial Earnings Call Analysis and Market Signal Generation Pipeline",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_runner = MarketMindPipeline()
report_gen = ReportGenerator()


# ---- Pydantic Schemas ----

class CompanyResponse(BaseModel):
    company_id: int
    ticker: str
    company_name: str
    sector: Optional[str] = None
    exchange: Optional[str] = None

    class Config:
        from_attributes = True


class EarningsCallSummary(BaseModel):
    call_id: int
    company_id: int
    ticker: str
    quarter: str
    fiscal_year: int
    call_date: str
    status: str
    signal: Optional[str] = None
    confidence: Optional[float] = None


class MetricResponse(BaseModel):
    metric_id: int
    metric_name: str
    value: float
    unit: Optional[str] = None
    period: Optional[str] = None
    is_guidance: bool
    speaker_role: Optional[str] = None
    quote: Optional[str] = None
    confidence: Optional[float] = None


class EvasionResponse(BaseModel):
    evasion_id: int
    evasion_score: float
    is_evasive: bool
    reason: Optional[str] = None
    question_speaker: Optional[str] = None
    question_text: Optional[str] = None
    answer_speaker: Optional[str] = None
    answer_text: Optional[str] = None


# ---- API Endpoints ----

@app.on_event("startup")
def startup_event():
    init_database()
    logger.info("MarketMind API started. Database ready.")


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "MarketMind API",
        "version": "1.0.0",
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/companies", response_model=List[CompanyResponse], tags=["Companies"])
def list_companies():
    """List all tracked companies."""
    session = get_session()
    try:
        companies = session.query(Company).all()
        return companies
    finally:
        session.close()


@app.get("/api/calls", response_model=List[EarningsCallSummary], tags=["Earnings Calls"])
def list_calls():
    """List all processed earnings calls with their prediction signals."""
    session = get_session()
    try:
        calls = session.query(EarningsCall).order_by(EarningsCall.call_date.desc()).all()
        summaries = []
        for c in calls:
            company = session.query(Company).filter_by(company_id=c.company_id).first()
            sig = session.query(Signal).filter_by(call_id=c.call_id).first()
            summaries.append(
                EarningsCallSummary(
                    call_id=c.call_id,
                    company_id=c.company_id,
                    ticker=company.ticker if company else "UNKNOWN",
                    quarter=c.quarter,
                    fiscal_year=c.fiscal_year,
                    call_date=c.call_date.isoformat(),
                    status=c.status.value if c.status else "completed",
                    signal=sig.predicted_direction.value if sig else None,
                    confidence=sig.confidence if sig else None,
                )
            )
        return summaries
    finally:
        session.close()


@app.get("/api/calls/{call_id}", tags=["Earnings Calls"])
def get_call_details(call_id: int):
    """Get full details for a single earnings call."""
    session = get_session()
    try:
        call = session.query(EarningsCall).filter_by(call_id=call_id).first()
        if not call:
            raise HTTPException(status_code=404, detail="Earnings call not found")

        company = session.query(Company).filter_by(company_id=call.company_id).first()
        signal = session.query(Signal).filter_by(call_id=call_id).first()
        segments_count = session.query(TranscriptSegment).filter_by(call_id=call_id).count()

        return {
            "call_id": call.call_id,
            "ticker": company.ticker if company else None,
            "company_name": company.company_name if company else None,
            "quarter": call.quarter,
            "fiscal_year": call.fiscal_year,
            "call_date": call.call_date.isoformat(),
            "status": call.status.value if call.status else "completed",
            "segments_count": segments_count,
            "signal": {
                "direction": signal.predicted_direction.value if signal else None,
                "confidence": signal.confidence if signal else None,
                "expected_return": signal.expected_return if signal else None,
                "top_drivers": signal.top_features if signal else [],
            } if signal else None,
        }
    finally:
        session.close()


@app.get("/api/calls/{call_id}/metrics", response_model=List[MetricResponse], tags=["Analysis"])
def get_call_metrics(call_id: int):
    """Get all extracted financial metrics and forward guidance for a call."""
    session = get_session()
    try:
        metrics = session.query(ExtractedMetric).filter_by(call_id=call_id).all()
        return [
            MetricResponse(
                metric_id=m.metric_id,
                metric_name=m.metric_name,
                value=m.value,
                unit=m.unit,
                period=m.period,
                is_guidance=m.is_guidance,
                speaker_role=m.speaker_role.value if m.speaker_role else None,
                quote=m.quote,
                confidence=m.confidence,
            )
            for m in metrics
        ]
    finally:
        session.close()


@app.get("/api/calls/{call_id}/sentiment", tags=["Analysis"])
def get_call_sentiment(call_id: int):
    """Get aggregated sentiment scores and role breakdowns."""
    session = get_session()
    try:
        sentiments = session.query(SentimentScore).filter_by(call_id=call_id).all()
        if not sentiments:
            return {"call_id": call_id, "scored_segments": 0, "roles": {}}

        roles_dict = {}
        for s in sentiments:
            role = s.speaker_role.value if s.speaker_role else "unknown"
            roles_dict.setdefault(role, []).append(s.compound_score)

        return {
            "call_id": call_id,
            "total_segments_scored": len(sentiments),
            "overall_compound": round(sum(s.compound_score for s in sentiments) / len(sentiments), 4),
            "roles": {
                r: round(sum(v) / len(v), 4) for r, v in roles_dict.items()
            },
        }
    finally:
        session.close()


@app.get("/api/calls/{call_id}/evasion", response_model=List[EvasionResponse], tags=["Analysis"])
def get_call_evasion(call_id: int):
    """Get question-and-answer pairs with evasion scores and rationales."""
    session = get_session()
    try:
        evasions = session.query(EvasionScore).filter_by(call_id=call_id).all()
        results = []
        for e in evasions:
            q = session.query(TranscriptSegment).filter_by(segment_id=e.question_segment_id).first()
            a = session.query(TranscriptSegment).filter_by(segment_id=e.answer_segment_id).first()
            results.append(
                EvasionResponse(
                    evasion_id=e.evasion_id,
                    evasion_score=e.evasion_score,
                    is_evasive=e.is_evasive,
                    reason=e.reason,
                    question_speaker=q.speaker_label if q else None,
                    question_text=q.text if q else None,
                    answer_speaker=a.speaker_label if a else None,
                    answer_text=a.text if a else None,
                )
            )
        return results
    finally:
        session.close()


@app.get("/api/calls/{call_id}/report", response_class=PlainTextResponse, tags=["Reports"])
def get_call_report(call_id: int):
    """Retrieve the generated markdown intelligence report for a call."""
    session = get_session()
    try:
        md_text, _ = report_gen.generate_report_markdown(call_id, session=session)
        return md_text
    finally:
        session.close()


@app.post("/api/pipeline/upload", tags=["Pipeline"])
async def upload_transcript(file: UploadFile = File(...)):
    """Upload a new earnings call PDF transcript and run the pipeline."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF transcripts are supported.")

    save_dir = RAW_DATA_DIR / "transcripts" / "uploaded"
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / file.filename

    content = await file.read()
    file_path.write_bytes(content)

    logger.info(f"Received upload: {file.filename}, processing...")
    result = pipeline_runner.process_pdf(file_path)
    return {
        "status": "success",
        "message": f"Processed {file.filename}",
        "result": result,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=DEBUG)
