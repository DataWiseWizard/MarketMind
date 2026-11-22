"""
Tests for Database Schema & Model Integrity
"""

import unittest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import (
    Base, Company, EarningsCall, TranscriptSegment,
    ExtractedMetric, SentimentScore, EvasionScore,
    AccountabilityFinding, Signal, ModelRun
)


class TestDatabaseSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create an isolated in-memory SQLite database for schema validation
        cls.engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(cls.engine)
        cls.Session = sessionmaker(bind=cls.engine)

    def setUp(self):
        self.session = self.Session()

    def tearDown(self):
        self.session.close()

    def test_all_tables_created(self):
        """Assert all expected tables are registered in metadata."""
        tables = set(Base.metadata.tables.keys())
        expected_tables = {
            "companies", "earnings_calls", "transcript_segments",
            "extracted_metrics", "sentiment_scores", "evasion_scores",
            "accountability_findings", "signals", "model_runs"
        }
        self.assertTrue(expected_tables.issubset(tables))

    def test_insert_company_and_call(self):
        """Test inserting and querying relational entities."""
        co = Company(
            ticker="TEST",
            company_name="Test Semiconductor Corp",
            sector="Technology",
            exchange="NASDAQ"
        )
        self.session.add(co)
        self.session.commit()

        call = EarningsCall(
            company_id=co.company_id,
            quarter="Q1",
            fiscal_year=2026,
            call_date=date(2026, 1, 15),
            transcript_path="data/raw/transcripts/test.pdf",
            source="test"
        )
        self.session.add(call)
        self.session.commit()

        queried_co = self.session.query(Company).filter_by(ticker="TEST").first()
        self.assertIsNotNone(queried_co)
        self.assertEqual(queried_co.company_name, "Test Semiconductor Corp")

        queried_call = self.session.query(EarningsCall).filter_by(company_id=co.company_id).first()
        self.assertIsNotNone(queried_call)
        self.assertEqual(queried_call.quarter, "Q1")


if __name__ == "__main__":
    unittest.main()
