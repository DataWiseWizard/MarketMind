"""
Tests for Financial Metric Extraction Module
"""

import unittest
from src.processing.metric_extractor import FinancialMetricExtractor
from src.db.models import SpeakerRole


class TestFinancialMetricExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FinancialMetricExtractor()

    def test_extract_reported_revenue(self):
        sentence = "We delivered another strong quarter with revenue of $44 billion, up 69% year-over-year."
        metrics = self.extractor.extract_from_sentence(
            sentence=sentence,
            speaker_role=SpeakerRole.CFO,
            is_outlook_context=False,
            call_period="Q1 FY2026",
        )
        metric_names = [m["metric_name"] for m in metrics]
        self.assertIn("revenue", metric_names)
        self.assertIn("revenue_growth_yoy", metric_names)

        rev = next(m for m in metrics if m["metric_name"] == "revenue")
        self.assertEqual(rev["value"], 44.0)
        self.assertEqual(rev["unit"], "USD_B")
        self.assertFalse(rev["is_guidance"])

    def test_extract_guidance_revenue(self):
        sentence = "Total revenue is expected to be $45 billion, plus or minus 2%."
        metrics = self.extractor.extract_from_sentence(
            sentence=sentence,
            speaker_role=SpeakerRole.CFO,
            is_outlook_context=True,
            call_period="Q1 FY2026",
        )
        guidance = next((m for m in metrics if m["metric_name"] == "guidance_revenue"), None)
        self.assertIsNotNone(guidance)
        self.assertEqual(guidance["value"], 45.0)
        self.assertTrue(guidance["is_guidance"])


if __name__ == "__main__":
    unittest.main()
