"""
Tests for Transcript Ingestion & Parser
"""

import unittest
from pathlib import Path
from src.ingestion.transcript_parser import TranscriptParser
from src.db.models import SpeakerRole, SegmentType


class TestTranscriptParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = TranscriptParser()
        cls.sample_pdf = Path("data/raw/transcripts/2026/NVDA-Q1-2026-Earnings-Call-28-May-2025-5_00-PM-ET.pdf")

    def test_metadata_extraction(self):
        if not self.sample_pdf.exists():
            self.skipTest(f"Sample PDF not found at {self.sample_pdf}")
        raw_text = self.parser.extract_raw_text(self.sample_pdf)
        meta = self.parser.extract_metadata(self.sample_pdf, raw_text)

        self.assertEqual(meta["ticker"], "NVDA")
        self.assertEqual(meta["quarter"], "Q1")
        self.assertEqual(meta["fiscal_year"], 2026)
        self.assertIsNotNone(meta["call_date"])
        self.assertIn("NVIDIA", meta["company_name"])

    def test_section_splitting(self):
        if not self.sample_pdf.exists():
            self.skipTest(f"Sample PDF not found at {self.sample_pdf}")
        raw_text = self.parser.extract_raw_text(self.sample_pdf)
        mgmt_text, qa_text, mgmt_idx, qa_idx = self.parser.split_sections(raw_text)

        self.assertGreater(mgmt_idx, 0)
        self.assertGreater(qa_idx, mgmt_idx)
        self.assertGreater(len(mgmt_text), 500)
        self.assertGreater(len(qa_text), 500)

    def test_transcript_parsing(self):
        if not self.sample_pdf.exists():
            self.skipTest(f"Sample PDF not found at {self.sample_pdf}")
        parsed = self.parser.parse(self.sample_pdf)

        self.assertIn("metadata", parsed)
        self.assertIn("participants", parsed)
        self.assertIn("segments", parsed)

        segments = parsed["segments"]
        self.assertGreater(len(segments), 0)

        # Ensure speaker roles exist
        roles = {s["speaker_role"] for s in segments}
        self.assertTrue(SpeakerRole.CEO in roles or SpeakerRole.CFO in roles)

        # Ensure segment types exist
        types = {s["segment_type"] for s in segments}
        self.assertIn(SegmentType.PREPARED_REMARKS, types)
        self.assertTrue(SegmentType.QA_QUESTION in types or SegmentType.QA_ANSWER in types)


if __name__ == "__main__":
    unittest.main()
