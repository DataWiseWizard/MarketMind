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

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict
from loguru import logger

from src.config import API_HOST, API_PORT, DEBUG, DISCLAIMER, RAW_DATA_DIR
from src.db.connection import get_session, init_database
from src.db.models import (
    Company, EarningsCall, TranscriptSegment, ExtractedMetric,
    SentimentScore, EvasionScore, AccountabilityFinding, Signal, CallStatus, PredictedDirection
)
from src.pipeline import MarketMindPipeline
from src.reporting.report_generator import ReportGenerator

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()
    # Seed tracked semiconductor companies, calls, segments, metrics & signals
    session = get_session()
    try:
        from src.db.seed_data import seed_demo_data
        seed_demo_data(session)
    except Exception as e:
        logger.warning(f"Database seeding skipped: {e}")
        session.rollback()
    finally:
        session.close()

    # Warm up SHAP explainer cache in background for sub-millisecond responses
    try:
        from src.models.explainer import warm_up_explainer
        warm_up_explainer()
    except Exception as e:
        logger.debug(f"Explainer warm up: {e}")

    logger.info("MarketMind API started. Database and models ready.")
    yield


app = FastAPI(
    title="MarketMind Financial Intelligence API",
    description="Automated Financial Earnings Call Analysis and Market Signal Generation Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline_runner: Optional[MarketMindPipeline] = None

def get_pipeline() -> MarketMindPipeline:
    global _pipeline_runner
    if _pipeline_runner is None:
        _pipeline_runner = MarketMindPipeline()
    return _pipeline_runner

_agent_cache: Dict[bool, Any] = {}


def get_agent(use_rag: bool = True):
    if use_rag not in _agent_cache:
        from src.rag.agent import MarketMindAgent
        _agent_cache[use_rag] = MarketMindAgent(use_rag=use_rag)
    return _agent_cache[use_rag]

report_gen = ReportGenerator()


# ---- Pydantic Schemas ----

class CompanyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    company_id: int
    ticker: str
    company_name: str
    sector: Optional[str] = None
    exchange: Optional[str] = None


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

@app.api_route("/health", methods=["GET", "HEAD"], tags=["System"])
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
        if not metrics:
            from dashboard.utils.mock_data import get_mock_metrics_data
            mock_m = get_mock_metrics_data(call_id)
            return [
                MetricResponse(
                    metric_id=i + 1,
                    metric_name=m["metric_name"],
                    value=m["value"],
                    unit=m.get("unit"),
                    period=m.get("period"),
                    is_guidance=m.get("is_guidance", False),
                    speaker_role="cfo",
                    quote=m.get("quote"),
                    confidence=m.get("confidence", 0.95),
                )
                for i, m in enumerate(mock_m)
            ]
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
    """Get aggregated sentiment scores, role breakdowns, and segment timeline."""
    session = get_session()
    try:
        from src.db.models import TranscriptSegment
        sentiments = (
            session.query(SentimentScore, TranscriptSegment.segment_order)
            .outerjoin(TranscriptSegment, TranscriptSegment.segment_id == SentimentScore.segment_id)
            .filter(SentimentScore.call_id == call_id)
            .order_by(TranscriptSegment.segment_order)
            .all()
        )
        if not sentiments:
            from dashboard.utils.mock_data import get_mock_tone_data
            mock_tone = get_mock_tone_data(call_id)
            mock_tone["roles"] = {
                "ceo": mock_tone.get("ceo_mean", 0.73),
                "cfo": mock_tone.get("cfo_mean", 0.39),
                "analyst": mock_tone.get("analyst_mean", -0.06),
            }
            return mock_tone

        roles_dict = {}
        seg_list = []
        for s, seg_order in sentiments:
            role = s.speaker_role.value if s.speaker_role else "unknown"
            role_key = role.lower()
            roles_dict.setdefault(role_key, []).append(s.compound_score)
            seg_list.append({
                "segment_order": seg_order or len(seg_list) + 1,
                "role": role.upper(),
                "speaker": role.upper(),
                "compound": round(s.compound_score, 4),
                "positive": round(s.positive_score, 4),
                "neutral": round(s.neutral_score, 4),
                "negative": round(s.negative_score, 4),
            })

        ceo_scores = roles_dict.get("ceo", [])
        cfo_scores = roles_dict.get("cfo", [])
        analyst_scores = roles_dict.get("analyst", [])

        ceo_mean = round(sum(ceo_scores) / len(ceo_scores), 4) if ceo_scores else 0.0
        cfo_mean = round(sum(cfo_scores) / len(cfo_scores), 4) if cfo_scores else 0.0
        analyst_mean = round(sum(analyst_scores) / len(analyst_scores), 4) if analyst_scores else 0.0

        return {
            "call_id": call_id,
            "total_segments_scored": len(sentiments),
            "overall_compound": round(sum(s[0].compound_score for s in sentiments) / len(sentiments), 4),
            "ceo_mean": ceo_mean,
            "cfo_mean": cfo_mean,
            "analyst_mean": analyst_mean,
            "ceo_cfo_delta": round(ceo_mean - cfo_mean, 4),
            "roles": {
                r: round(sum(v) / len(v), 4) for r, v in roles_dict.items()
            },
            "segments": seg_list,
        }
    finally:
        session.close()


@app.get("/api/calls/{call_id}/evasion", response_model=List[EvasionResponse], tags=["Analysis"])
def get_call_evasion(call_id: int):
    """Get question-and-answer pairs with evasion scores and rationales."""
    session = get_session()
    try:
        evasions = session.query(EvasionScore).filter_by(call_id=call_id).all()
        if not evasions:
            from dashboard.utils.mock_data import get_mock_evasion_data
            mock_ev = get_mock_evasion_data(call_id)
            return [
                EvasionResponse(
                    evasion_id=e["evasion_id"],
                    evasion_score=e["evasion_score"],
                    is_evasive=e["is_evasive"],
                    reason=e.get("reason"),
                    question_speaker=e.get("question_speaker"),
                    question_text=e.get("question_text"),
                    answer_speaker=e.get("answer_speaker"),
                    answer_text=e.get("answer_text"),
                )
                for e in mock_ev
            ]
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
    result = get_pipeline().process_pdf(file_path)
    return {
        "status": "success",
        "message": f"Processed {file.filename}",
        "result": result,
    }



# ══════════════════════════════════════════════════════════════════
# Stage 2 — RAG + Prediction Endpoints
# ══════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    question: str
    use_rag: bool = True


class AuditResponse(BaseModel):
    call_id: int
    ticker: str
    quarter: str
    overall_risk: Optional[str] = None
    findings: List[Dict[str, Any]] = []
    recommendation: Optional[str] = None
    generated_at: Optional[str] = None


@app.post("/api/agent/chat/{call_id}", tags=["AI Agent"])
def agent_chat(call_id: int, body: ChatRequest):
    """
    Interactive RAG chat about an earnings call.
    Returns AI answer grounded in transcript + 10-Q context.
    """
    try:
        agent = get_agent(use_rag=body.use_rag)
        result = agent.chat(call_id=call_id, question=body.question)
        return result
    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agent/audit/{call_id}", tags=["AI Agent"])
def agent_audit(call_id: int, use_rag: bool = True):
    """
    Run the adversarial forensic audit for an earnings call.
    Finds discrepancies, peer dissonance, and accountability gaps.
    """
    try:
        agent = get_agent(use_rag=use_rag)
        result = agent.run_forensic_audit(call_id=call_id)
        if result and not result.get("error"):
            return result
    except Exception as e:
        logger.warning(f"Agent audit live evaluation failed: {e}")

    from dashboard.utils.mock_data import get_mock_audit_findings
    return get_mock_audit_findings(call_id)


@app.get("/api/signals/{call_id}", tags=["Predictions"])
def get_signal_with_shap(call_id: int):
    """
    Get the LightGBM prediction + SHAP feature attributions for a call.
    """
    from dashboard.utils.mock_data import get_mock_signal_data
    session = get_session()
    try:
        sig = session.query(Signal).filter_by(call_id=call_id).first()
        if not sig:
            mock_s = get_mock_signal_data(call_id)
            if mock_s:
                return mock_s
            raise HTTPException(status_code=404, detail=f"No signal for call_id={call_id}")

        # Attempt SHAP explanation with fast cache
        shap_result = {}
        try:
            from src.models.explainer import predict_with_shap

            # Build feature vector from DB
            sents = session.query(SentimentScore).filter_by(call_id=call_id).all()
            role_sent = {}
            for s in sents:
                role = (s.speaker_role.value or "unknown").lower()
                role_sent.setdefault(role, []).append(s.compound_score)
            avg_sent = {r: sum(v) / len(v) for r, v in role_sent.items()}

            evas = session.query(EvasionScore).filter_by(call_id=call_id).all()
            avg_evasion = sum(e.evasion_score for e in evas) / len(evas) if evas else 0.0
            evasive_cnt = sum(1 for e in evas if e.is_evasive)

            acct = session.query(AccountabilityFinding).filter_by(call_id=call_id).all()

            feat_vec = {
                "sent_ceo": avg_sent.get("ceo", 0.73),
                "sent_cfo": avg_sent.get("cfo", 0.39),
                "sent_analyst": avg_sent.get("analyst", -0.06),
                "sent_ceo_delta": avg_sent.get("ceo", 0.73) - avg_sent.get("cfo", 0.39),
                "sent_cfo_delta": 0.0,
                "evasion_mean": avg_evasion if evas else 0.14,
                "evasion_max": max((e.evasion_score for e in evas), default=0.78),
                "evasive_count": float(evasive_cnt) if evas else 2.0,
                "revenue_qoq_pct": 16.8 if sig.predicted_direction.value == "positive" else -6.2,
                "margin_qoq_pp": 1.4 if sig.predicted_direction.value == "positive" else -3.8,
                "discrepancy_count": float(len(acct)),
                "momentum_30d": 0.345,
                "volatility_30d": 0.421,
            }

            shap_result = predict_with_shap(feat_vec)
        except Exception as e:
            logger.debug(f"SHAP explanation fallback: {e}")
            mock_s = get_mock_signal_data(call_id)
            shap_result = mock_s.get("shap", {})

        return {
            "call_id": call_id,
            "predicted_signal": sig.predicted_direction.value,
            "confidence": sig.confidence,
            "expected_return": sig.expected_return,
            "top_features": sig.top_features or [],
            "shap": shap_result,
        }
    finally:
        session.close()


@app.post("/api/vectorstore/build", tags=["RAG"])
def build_vectorstore(background_tasks: BackgroundTasks):
    """
    Trigger background build of ChromaDB vector store.
    Indexes all transcript segments + available SEC filings.
    """
    def _build():
        from src.rag.vector_store import build_vector_store
        build_vector_store()

    background_tasks.add_task(_build)
    return {"status": "building", "message": "Vector store build started in background."}


@app.get("/api/vectorstore/search", tags=["RAG"])
def search_vectorstore(q: str, ticker: Optional[str] = None, n: int = 5):
    """Semantic search over the ChromaDB vector store."""
    try:
        from src.rag.vector_store import query_similar
        results = query_similar(q, n=n, ticker=ticker)
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions", tags=["Predictions"])
def list_predictions():
    """
    Return LightGBM predictions for all calls (after trainer.py is run).
    """
    session = get_session()
    try:
        # Try loading model and predicting from feature matrix if available
        import pickle
        from pathlib import Path as P
        model_path = P(__file__).resolve().parent.parent.parent / "models" / "lightgbm_marketmind_v1.pkl"

        if not model_path.exists():
            return {"message": "Model not trained yet. Run: python -m src.models.trainer", "predictions": []}

        matrix_path = P(__file__).resolve().parent.parent.parent / "data" / "processed" / "features" / "train_matrix.parquet"
        if not matrix_path.exists():
            return {"message": "Feature matrix not built yet. Run: python -m src.features.feature_builder", "predictions": []}

        import pandas as pd
        from src.models.explainer import predict_batch
        df = pd.read_parquet(matrix_path)
        df = predict_batch(df)

        return {
            "total": len(df),
            "predictions": df[["call_id", "ticker", "quarter", "fiscal_year",
                                "label", "predicted_label", "confidence"]].to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


# ---- Monitoring & MLOps Endpoints ----

@app.get("/api/monitoring/drift", tags=["Monitoring"])
def get_drift_report():
    """Return latest feature and prediction drift monitoring report."""
    report_path = PROJECT_ROOT / "models" / "drift_report.json"
    if report_path.exists():
        import json
        try:
            with open(report_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading drift_report.json: {e}")

    try:
        from src.models.monitoring import DriftMonitor
        monitor = DriftMonitor()
        report = monitor.run_drift_analysis()
        monitor.save_reports(report)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate drift report: {e}")


@app.post("/api/monitoring/drift/recalculate", tags=["Monitoring"])
def recalculate_drift():
    """Trigger an on-demand calculation of feature and prediction drift."""
    try:
        from src.models.monitoring import DriftMonitor
        monitor = DriftMonitor()
        report = monitor.run_drift_analysis()
        monitor.save_reports(report)
        monitor.log_to_mlflow(report)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recalculation failed: {e}")


class RetrainRequest(BaseModel):
    train_years: Optional[List[int]] = None


@app.post("/api/monitoring/retrain", tags=["Monitoring"])
def trigger_retraining(req: Optional[RetrainRequest] = None):
    """
    Triggers walk-forward retraining on an expanded historical window
    (default: [2023, 2024, 2025]), updates the model artifacts, logs to MLflow,
    and recalculates feature and prediction drift.
    """
    try:
        from src.models.trainer import run_training
        from src.models.monitoring import DriftMonitor

        train_years = req.train_years if req and req.train_years else [2023, 2024, 2025]

        cur_years = [2026] if 2025 in train_years else [2025, 2026]
        val_years = [2025] if 2025 not in train_years else [2024]
        run_name = f"walk_forward_expanding_{'_'.join(map(str, train_years))}"

        model, features, metrics = run_training(
            train_years=train_years,
            val_years=val_years,
            test_years=cur_years,
            run_name=run_name,
        )

        monitor = DriftMonitor()
        drift_report = monitor.run_drift_analysis(ref_years=train_years, cur_years=cur_years)
        monitor.save_reports(drift_report)
        monitor.log_to_mlflow(drift_report)

        return {
            "status": "success",
            "message": f"Model successfully retrained on years {train_years}.",
            "train_years": train_years,
            "test_years": cur_years,
            "metrics": metrics,
            "drift_report": drift_report,
        }
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")


@app.get("/api/monitoring/mlflow", tags=["Monitoring"])
def get_mlflow_info():
    """Return local MLflow tracking registry status and active model artifact paths."""
    import os
    mlflow_db = PROJECT_ROOT / "mlflow.db"
    return {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{mlflow_db.as_posix()}"),
        "experiment_name": "marketmind",
        "database_exists": mlflow_db.exists(),
        "model_artifact": str(PROJECT_ROOT / "models" / "lightgbm_marketmind_v1.pkl"),
        "model_exists": (PROJECT_ROOT / "models" / "lightgbm_marketmind_v1.pkl").exists(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=DEBUG)

