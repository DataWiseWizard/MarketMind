"""
MarketMind End-to-End Pipeline Runner

Coordinates the full transcript analysis workflow:
1. Ingestion & Segment Parsing
2. Financial Metric Extraction
3. Sentiment Analysis (FinBERT / Lexicon)
4. Evasion Detection in Q&A
5. Market Reaction Research Signal
6. Report Generation
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.db.connection import get_session, init_database
from src.db.models import EarningsCall, Company
from src.ingestion.transcript_parser import TranscriptParser, save_parsed_to_db, ingest_all_transcripts
from src.processing.metric_extractor import FinancialMetricExtractor
from src.processing.sentiment_analyzer import SentimentAnalyzer
from src.processing.evasion_detector import EvasionDetector
from src.reporting.report_generator import SignalGenerator, ReportGenerator


class MarketMindPipeline:
    """Orchestrates end-to-end execution of MarketMind modules."""

    def __init__(self):
        self.parser = TranscriptParser()
        self.metric_extractor = FinancialMetricExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.evasion_detector = EvasionDetector()
        self.signal_generator = SignalGenerator()
        self.report_generator = ReportGenerator()

    def process_pdf(self, pdf_path: Path | str) -> Dict[str, Any]:
        """Run complete pipeline on a single raw PDF transcript."""
        pdf_path = Path(pdf_path)
        logger.info(f"🚀 [Stage 1/6] Ingesting & Parsing Transcript: {pdf_path.name}")
        init_database()

        session = get_session()
        try:
            # 1. Parse & Ingest
            parsed = self.parser.parse(pdf_path)
            company, call, segments = save_parsed_to_db(parsed, session=session)
            call_id = call.call_id

            # 2. Extract Metrics
            logger.info(f"📊 [Stage 2/6] Extracting Financial Metrics (Call ID: {call_id})...")
            metrics = self.metric_extractor.extract_for_call(call_id, session=session)

            # 3. Sentiment Analysis
            logger.info(f"🎭 [Stage 3/6] Analyzing Sentiment across {len(segments)} segments...")
            sentiments = self.sentiment_analyzer.analyze_for_call(call_id, session=session)

            # 4. Evasion Detection
            logger.info("🕵️ [Stage 4/6] Detecting Evasion in Q&A pairs...")
            evasions = self.evasion_detector.analyze_for_call(call_id, session=session)

            # 5. Signal Generation
            logger.info("🎯 [Stage 5/6] Generating Research Signal & Feature Drivers...")
            signal = self.signal_generator.compute_signal(call_id, session=session)

            # 6. Report Generation
            logger.info("📑 [Stage 6/6] Generating Comprehensive Report...")
            report_md, report_path = self.report_generator.generate_report_markdown(call_id, session=session)

            logger.info(f"🎉 Pipeline Completed for {company.ticker} {call.quarter} FY{call.fiscal_year}!")
            return {
                "call_id": call_id,
                "ticker": company.ticker,
                "quarter": call.quarter,
                "fiscal_year": call.fiscal_year,
                "signal": signal.predicted_direction.value,
                "confidence": signal.confidence,
                "metrics_count": len(metrics),
                "segments_count": len(segments),
                "evasion_pairs": len(evasions),
                "report_path": str(report_path),
            }

        finally:
            session.close()

    def run_all(self) -> List[Dict[str, Any]]:
        """Run the pipeline across all raw transcripts."""
        init_database()
        session = get_session()
        try:
            calls = session.query(EarningsCall).order_by(EarningsCall.call_id).all()
            if not calls:
                logger.info("No calls found in database. Ingesting raw transcripts first...")
                ingest_all_transcripts()
                calls = session.query(EarningsCall).order_by(EarningsCall.call_id).all()

            results = []
            for call in calls:
                call_id = call.call_id
                company = session.query(Company).filter_by(company_id=call.company_id).first()
                logger.info(f"\n{'='*50}\nProcessing Call ID {call_id}: {company.ticker} {call.quarter} FY{call.fiscal_year}\n{'='*50}")

                self.metric_extractor.extract_for_call(call_id, session=session)
                self.sentiment_analyzer.analyze_for_call(call_id, session=session)
                self.evasion_detector.analyze_for_call(call_id, session=session)
                sig = self.signal_generator.compute_signal(call_id, session=session)
                _, report_path = self.report_generator.generate_report_markdown(call_id, session=session)

                results.append({
                    "call_id": call_id,
                    "ticker": company.ticker,
                    "quarter": call.quarter,
                    "fiscal_year": call.fiscal_year,
                    "signal": sig.predicted_direction.value,
                    "confidence": sig.confidence,
                    "report_path": str(report_path),
                })
            return results
        finally:
            session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketMind End-to-End Pipeline")
    parser.add_argument("--pdf", type=str, help="Path to raw transcript PDF to process")
    parser.add_argument("--all", action="store_true", help="Process all available transcripts")
    args = parser.parse_args()

    pipeline = MarketMindPipeline()

    if args.pdf:
        res = pipeline.process_pdf(args.pdf)
        print("\nPipeline Result:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    else:
        results = pipeline.run_all()
        print("\nAll Calls Processed Successfully:")
        for r in results:
            print(f"  • {r['ticker']} {r['quarter']} FY{r['fiscal_year']} -> Signal: {r['signal'].upper()} (Conf: {r['confidence']*100:.0f}%) -> {r['report_path']}")
